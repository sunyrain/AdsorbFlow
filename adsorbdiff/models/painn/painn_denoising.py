"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

MIT License

Copyright (c) 2021 www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, segment_coo, segment_csr

from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import conditional_grad
from adsorbdiff.models.base import BaseModel
from adsorbdiff.models.gemnet_oc.layers.base_layers import ScaledSiLU
from adsorbdiff.models.gemnet_oc.layers.embedding_block import AtomEmbedding
from adsorbdiff.models.gemnet_oc.layers.radial_basis import RadialBasis
from adsorbdiff.modules.scaling import ScaleFactor
from adsorbdiff.modules.scaling.compat import load_scales_compat
from adsorbdiff.models.embeddings import ATOMIC_RADII


class PaiNN(BaseModel):
    r"""PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
    """

    def __init__(
        self,
        num_atoms: int,
        bond_feat_dim: int,
        num_targets: int,
        hidden_channels: int = 512,
        num_layers: int = 6,
        num_rbf: int = 128,
        cutoff: float = 12.0,
        max_neighbors: int = 50,
        rbf: Dict[str, str] = {"name": "gaussian"},
        envelope: Dict[str, Union[str, int]] = {
            "name": "polynomial",
            "exponent": 5,
        },
        regress_forces: bool = True,
        direct_forces: bool = True,
        use_pbc: bool = True,
        otf_graph: bool = True,
        num_elements: int = 110,
        scale_file: Optional[str] = None,
        so3_denoising: bool = True,
        energy_encoding="scalar",
        sampling: bool = False,
        flow_head_activation: str = "tanh",
        flow_head_scale: float = 5.0,
    ) -> None:
        super(PaiNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.so3_denoising = so3_denoising
        self.sampling = sampling
        self.flow_head_activation = str(flow_head_activation).lower()
        self.flow_head_scale = float(flow_head_scale)

        # Borrowed from GemNet.
        self.symmetric_edge_symmetrization = False

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        atom_radii = torch.zeros(101)
        for i in range(101):
            atom_radii[i] = ATOMIC_RADII[i]
        self.atom_radii = atom_radii / 100
        self.atom_radii = torch.nn.Parameter(atom_radii, requires_grad=False)
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        if energy_encoding == "scalar":
            # FiLM-style conditioning: learn per-channel scale & shift from the energy scalar.
            self.energy_embedding = nn.Sequential(
                nn.Linear(1, hidden_channels),
                ScaledSiLU(),
                nn.Linear(hidden_channels, hidden_channels * 2),
            )
            self.energy_null = nn.Parameter(torch.zeros(hidden_channels * 2))
        else:
            self.energy_embedding = None
            self.energy_null = None

        for i in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))
            setattr(self, "upd_out_scalar_scale_%d" % i, ScaleFactor())

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if self.regress_forces is True and self.direct_forces is True:
            self.out_forces = PaiNNOutput(
                hidden_channels,
                activation=self.flow_head_activation,
                activation_scale=20,
            )
            self.out_forces.set_active_ids_getter(lambda: self._format_active_ids())
        if self.so3_denoising:
            self.out_forces2 = PaiNNOutput(
                hidden_channels,
                activation=self.flow_head_activation,
                activation_scale=5,
            )
            self.out_forces2.set_active_ids_getter(lambda: self._format_active_ids())

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.p_cfg = 0.0
        self._activation_hooks = []

        self.reset_parameters()
        load_scales_compat(self, scale_file)
        self._register_activation_hooks()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)
        self.time_embedding.apply(self._reset_linear_module)
        if self.energy_embedding is not None:
            self.energy_embedding.apply(self._reset_linear_module)
        if self.energy_null is not None:
            self.energy_null.data.zero_()

    @staticmethod
    def _reset_linear_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
    
    def _register_activation_hooks(self) -> None:
        def make_hook(layer_type: str, layer_idx: int):
            def _hook(module, inputs, outputs):
                tensors = []
                if torch.is_tensor(outputs):
                    tensors = [outputs]
                elif isinstance(outputs, (tuple, list)):
                    tensors = [t for t in outputs if torch.is_tensor(t)]
                if not tensors:
                    return
                for t in tensors:
                    self._monitor_activation(f"{layer_type}[{layer_idx}]", t)
            return _hook

        for idx, layer in enumerate(self.message_layers):
            handle = layer.register_forward_hook(make_hook("PaiNNMessage", idx))
            self._activation_hooks.append(handle)
        for idx, layer in enumerate(self.update_layers):
            handle = layer.register_forward_hook(make_hook("PaiNNUpdate", idx))
            self._activation_hooks.append(handle)
        for idx, layer in enumerate(self.out_forces.output_network):
            handle = layer.register_forward_hook(make_hook("PaiNNOutput", idx))
            self._activation_hooks.append(handle)
        if self.so3_denoising and hasattr(self, "out_forces2"):
            for idx, layer in enumerate(self.out_forces2.output_network):
                handle = layer.register_forward_hook(make_hook("PaiNNOutput2", idx))
                self._activation_hooks.append(handle)

    def _monitor_activation(self, label: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        data = tensor.detach()
        finite = torch.isfinite(data)
        finite_ratio = float(finite.float().mean().item())
        if finite.any():
            min_val = float(data[finite].min().item())
            max_val = float(data[finite].max().item())
        else:
            min_val = float("nan")
            max_val = float("nan")
        logging.debug(
            "[activation] %s shape=%s finite_ratio=%.3f min=%.4e max=%.4e",
            label,
            tuple(data.shape),
            finite_ratio,
            min_val,
            max_val,
        )
        if not torch.all(finite):
            ids = self._format_active_ids()
            logging.error(
                "[activation] non-finite detected at %s ids=%s",
                label,
                ids,
            )
           # raise RuntimeError(f"Non-finite activation detected at {label} ids={ids}")

    def _format_active_ids(self) -> str:
        info = getattr(self, "_active_batch_debug", None)
        if not isinstance(info, dict):
            return "n/a"
        ids = info.get("ids")
        if isinstance(ids, dict):
            parts = []
            for key, val in ids.items():
                parts.append(f"{key}={val}")
            return ",".join(parts)
        return str(info)

    def tag_based_Z(self, data) -> torch.Tensor:
        an = data.atomic_numbers  
        cnho = (an == 1) | (an == 6) | (an == 7) | (an == 8)
        mask = (data.tags < 2) & cnho
        an_mod = an.clone()
        an_mod[mask] += 100
        return an_mod 

    # Borrowed from GemNet.
    def select_symmetric_edges(
        self, tensor, mask, reorder_idx, inverse_neg
    ) -> torch.Tensor:
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    # Borrowed from GemNet.
    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        batch_idx,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        If `symmetric_edge_symmetrization` is False,
        we only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        If `symmetric_edge_symmetrization` is True,
        we always use both directions.
        """
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat(
                [edge_index, edge_index.flip(0)],
                dim=1,
            )
            cell_offsets_bothdir = torch.cat(
                [cell_offsets, -cell_offsets],
                dim=0,
            )

            # Filter for unique edges
            edge_ids = get_edge_id(
                edge_index_bothdir, cell_offsets_bothdir, num_atoms
            )
            unique_ids, unique_inv = torch.unique(
                edge_ids, return_inverse=True
            )
            perm = torch.arange(
                unique_inv.size(0),
                dtype=unique_inv.dtype,
                device=unique_inv.device,
            )
            unique_idx = scatter(
                perm,
                unique_inv,
                dim=0,
                dim_size=unique_ids.shape[0],
                reduce="min",
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]

            # Order by target index
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]

            # Subindex remaining tensors
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]

            # Count edges per image
            # segment_coo assumes sorted edge_index_new[1] and batch_idx
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(
                ones, edge_index_new[1], dim_size=num_atoms
            )
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            # Generate mask
            mask_sep_atoms = edge_index[0] < edge_index[1]
            # Distinguish edges between the same (periodic) atom by ordering the cells
            cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | (
                    (cell_offsets[:, 0] == 0)
                    & (cell_offsets[:, 1] == 0)
                    & (cell_offsets[:, 2] < 0)
                )
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            # Mask out counter-edges
            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(
                2, -1
            )

            # Concatenate counter-edges after normal edges
            edge_index_cat = torch.cat(
                [edge_index_new, edge_index_new.flip(0)],
                dim=1,
            )

            # Count remaining edges per image
            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device),
                neighbors,
            )
            batch_edge = batch_edge[mask]
            # segment_coo assumes sorted batch_edge
            # Factor 2 since this is only one half of the edges
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(
                ones, batch_edge, dim_size=neighbors.size(0)
            )

            # Create indexing array
            edge_reorder_idx = repeat_blocks(
                torch.div(neighbors_per_image, 2, rounding_mode="floor"),
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            # Reorder everything so the edges of every image are consecutive
            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, False
                )
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, True
                )
                for tensor in reorder_tensors_invneg
            ]

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            edge_index_new.flip(0), -cell_offsets_new, num_atoms
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )

    def generate_graph_values(self, data):
        (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(edge_dist, torch.tensor(0.0), atol=1e-3)
        edge_dist[mask_zero] = 1.0e-3
        edge_vector = distance_vec / edge_dist[:, None]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [edge_dist],
            [edge_vector],
            id_swap,
        ) = self.symmetrize_edges(
            edge_index,
            cell_offsets,
            neighbors,
            data.batch,
            [edge_dist],
            [edge_vector],
        )

        return (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data, mode: Optional[str] = None):
        pos = data.pos
        batch = data.batch
        z = self.tag_based_Z(data).long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        if self.energy_embedding is not None:
            batch_idx = data.batch
            if hasattr(data, "natoms") and data.natoms is not None:
                batch_size = int(data.natoms.shape[0])
            else:
                batch_size = int(batch_idx.max().item()) + 1

            if hasattr(data, "energy") and data.energy is not None:
                node_energy = data.energy[batch_idx].unsqueeze(-1)
            else:
                node_energy = torch.zeros(batch_idx.size(0), 1, device=x.device)

            energy_cond = self.energy_embedding(node_energy.float()).to(x.dtype)
            cfg_conditioned = getattr(data, "cfg_conditioned", None)
            if cfg_conditioned is not None:
                if cfg_conditioned.numel() != batch_size:
                    raise ValueError("cfg_conditioned must have one flag per sample.")
                sample_mask = cfg_conditioned.to(device=x.device)
                node_mask = sample_mask[batch_idx].to(torch.bool)
            elif self.training and self.p_cfg > 0.0:
                keep_sample = torch.rand(batch_size, device=x.device) >= self.p_cfg
                node_mask = keep_sample[batch_idx]
            else:
                node_mask = torch.ones(batch_idx.size(0), dtype=torch.bool, device=x.device)

            null_vec = self.energy_null.view(1, -1).to(device=x.device, dtype=x.dtype).expand_as(energy_cond)
            emb = torch.where(node_mask.unsqueeze(-1), energy_cond, null_vec)
            gamma, beta = torch.chunk(emb, 2, dim=-1)
            x = x * (1 + gamma) + beta

        if hasattr(data, "t") and data.t is not None:
            t_val = data.t
            if t_val.dim() == 1:
                t_val = t_val.unsqueeze(-1)
            t_nodes = t_val[data.batch].to(x.dtype)
            x = x + self.time_embedding(t_nodes.float())

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_rbf, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        #### Output block #####################################################

        # per_atom_energy = self.out_energy(x).squeeze(1)
        # energy = scatter(per_atom_energy, batch, dim=0)

        # if self.regress_forces:
        #     if self.direct_forces:
        #         forces = self.out_forces(x, vec)
        #         return energy, forces
        #     else:
        #         forces = (
        #             -1
        #             * torch.autograd.grad(
        #                 x,
        #                 pos,
        #                 grad_outputs=torch.ones_like(x),
        #                 create_graph=True,
        #             )[0]
        #         )
        #         return energy, forces
        # else:
        #     return energy

        flow_context = self._build_flow_context(data)
        if hasattr(self, "out_forces") and self.out_forces is not None:
            self.out_forces.set_flow_context_fetcher(lambda ctx=flow_context: ctx)
        if self.so3_denoising and hasattr(self, "out_forces2") and self.out_forces2 is not None:
            self.out_forces2.set_flow_context_fetcher(lambda ctx=flow_context: ctx)

        forces = self.out_forces(x, vec)
        forces2 = self.out_forces2(x, vec) if self.so3_denoising else None

        if mode == "fm":
            return self._forward_flow(data, forces, forces2)

        if not self.so3_denoising:
            return forces
        return forces, forces2

    def _build_flow_context(self, batch) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        flow_debug = getattr(batch, "_flow_debug", None)
        if flow_debug:
            context["flow_debug"] = flow_debug
        v_tr_target = getattr(batch, "v_tr_target", None)
        if torch.is_tensor(v_tr_target):
            context["v_tr_target"] = v_tr_target.detach().cpu()
        v_rot_target = getattr(batch, "v_rot_target", None)
        if torch.is_tensor(v_rot_target):
            context["v_rot_target"] = v_rot_target.detach().cpu()
        t_val = getattr(batch, "t", None)
        if torch.is_tensor(t_val):
            context["t"] = t_val.detach().cpu()
        if hasattr(batch, "rot_symmetry"):
            context["rot_symmetry"] = list(getattr(batch, "rot_symmetry"))
        if hasattr(batch, "sid"):
            context["sid"] = batch.sid
        if hasattr(batch, "fid"):
            context["fid"] = batch.fid
        return context

    def _forward_flow(self, data, forces: torch.Tensor, forces2: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Project per-atom outputs to the rigid body (translation/rotation) flow heads."""
        if not hasattr(data, "batch") or not hasattr(data, "tags"):
            raise ValueError("Flow matching mode expects 'batch' and 'tags' attributes on the input data.")

        batch_idx = data.batch
        tags = data.tags
        ads_mask = tags == 2
        if not torch.any(ads_mask) and tags.dtype == torch.bool:
            ads_mask = tags

        if not torch.any(ads_mask):
            raise RuntimeError("No adsorbate atoms (tag==2) found; cannot form flow outputs.")

        if hasattr(data, "natoms"):
            batch_size = int(data.natoms.shape[0])
        else:
            batch_size = int(batch_idx.max().item()) + 1

        ads_forces = forces[ads_mask]
        force_stats: Dict[str, Union[float, bool, torch.Tensor]] = {}
        if ads_forces.numel() > 0:
            flat = ads_forces.detach()
            finite = torch.isfinite(flat)
            force_stats["has_nonfinite"] = not bool(torch.all(finite).item())
            if torch.any(finite):
                finite_vals = torch.abs(flat[finite])
                force_stats["global_max_abs"] = float(finite_vals.max().item())
            else:
                force_stats["global_max_abs"] = float("nan")
            per_atom_max = torch.abs(flat)
            per_atom_max = torch.nan_to_num(per_atom_max, nan=0.0, posinf=0.0, neginf=0.0)
            per_atom_max = per_atom_max.amax(dim=-1)
            per_sample_max = scatter(
                per_atom_max,
                batch_idx[ads_mask],
                dim=0,
                dim_size=batch_size,
                reduce="max",
            )
            force_stats["per_sample_max_abs"] = per_sample_max.detach().cpu()

        # === 1. Translation 部分 (保持不变: 平均线速度 = 质心速度) ===
        # Translation is a polar vector, mean pooling is correct for rigid translation
        translation = scatter(
            ads_forces,
            batch_idx[ads_mask],
            dim=0,
            dim_size=batch_size,
            reduce="mean",
        )
        translation = translation.clone()
        # Unlock Z-axis to allow vertical relaxation
        # if translation.size(-1) >= 3:
        #     translation[:, 2] = 0.0

        # === 2. Rotation 部分 (关键修改: 引入叉积聚合以匹配轴矢量宇称) ===
        if forces2 is not None:
            ads_forces2 = forces2[ads_mask]
            
            # (A) 计算吸附剂中心 (Center of Geometry/Mass)
            ads_pos = data.pos[ads_mask]
            ads_batch = batch_idx[ads_mask]
            centers = scatter(ads_pos, ads_batch, dim=0, dim_size=batch_size, reduce="mean")
            
            # (B) 计算相对坐标 r (relative position)
            # centers[ads_batch] 将中心坐标广播回每个原子
            rel_pos = ads_pos - centers[ads_batch]

            # (C) 计算"力矩"项: r x v
            # PaiNN 输出的是极矢量(类似切向速度)，通过叉积转换为轴矢量(旋转轴)
            # 这样可以解决"刚体旋转的平均线速度为0"的问题，同时匹配宇称(Polar x Polar = Axial)
            torque_like = torch.cross(rel_pos, ads_forces2, dim=-1)
            
            # (D) 聚合得到全局旋转量
            rotation = scatter(torque_like, ads_batch, dim=0, dim_size=batch_size, reduce="mean")
        else:
            rotation = torch.zeros(
                batch_size,
                3,
                device=translation.device,
                dtype=translation.dtype,
            )

        # === 3. 组装输出 ===
        outputs: Dict[str, torch.Tensor] = {
            "v_tr": translation,
            "v_rot": rotation,
        }
        if force_stats:
            outputs["_debug_force_pre_scatter"] = force_stats
        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"max_neighbors={self.max_neighbors}, "
            f"cutoff={self.cutoff})"
        )


class PaiNNMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super(PaiNNMessage, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.nan_clip = 1.0e6

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)
        rbfh = self._sanitize_tensor(
            rbfh,
            "PaiNNMessage.rbf_proj",
            extra_info=self._edge_info(edge_vector, edge_rbf),
        )

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs

    def _sanitize_tensor(self, tensor: torch.Tensor, label: str, extra_info: str = "") -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        logging.warning(
            "[PaiNNMessage] non-finite tensor detected in %s %s",
            label,
            extra_info,
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=self.nan_clip, neginf=-self.nan_clip)
        tensor = torch.clamp(tensor, min=-self.nan_clip, max=self.nan_clip)
        return tensor

    def _edge_info(self, edge_vector: torch.Tensor, edge_rbf: torch.Tensor) -> str:
        try:
            edge_norm = torch.linalg.norm(edge_vector.detach(), dim=-1)
            edge_str = f"edge_norm[min={edge_norm.min().item():.4e}, max={edge_norm.max().item():.4e}]"
        except Exception:
            edge_str = "edge_norm=n/a"
        try:
            rbf_norm = torch.linalg.norm(edge_rbf.detach(), dim=-1)
            rbf_str = f"rbf_norm[min={rbf_norm.min().item():.4e}, max={rbf_norm.max().item():.4e}]"
        except Exception:
            rbf_str = "rbf_norm=n/a"
        return f"{edge_str} {rbf_str}"


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.nan_clip = 1.0e6

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec1 = self._sanitize_tensor(vec1, "PaiNNUpdate.vec1")
        vec2 = self._sanitize_tensor(
            vec2,
            "PaiNNUpdate.vec2",
            extra_info=self._vec2_info(vec2),
        )
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        norm_arg = torch.sum(vec2**2, dim=-2)
        if torch.any(~torch.isfinite(norm_arg)):
            logging.warning("[PaiNNUpdate] non-finite norm_arg detected")
        norm_arg = torch.nan_to_num(norm_arg, nan=0.0, posinf=self.nan_clip**2, neginf=0.0)
        norm_arg = torch.clamp(norm_arg, min=0.0, max=self.nan_clip**2)

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        sqrt_term = torch.sqrt(norm_arg + 1e-8)
        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, sqrt_term], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1
        
        return dx, dvec

    def _sanitize_tensor(self, tensor: torch.Tensor, label: str, extra_info: str = "") -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        logging.warning("[PaiNNUpdate] non-finite tensor detected in %s %s", label, extra_info)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=self.nan_clip, neginf=-self.nan_clip)
        tensor = torch.clamp(tensor, min=-self.nan_clip, max=self.nan_clip)
        return tensor

    def _vec2_info(self, vec2: torch.Tensor) -> str:
        try:
            vec2_norm = torch.linalg.norm(vec2.detach(), dim=-2)
            return f"vec2_norm[min={vec2_norm.min().item():.4e}, max={vec2_norm.max().item():.4e}]"
        except Exception:
            return "vec2_norm=n/a"

        return dx, dvec


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels, activation: str = "identity", activation_scale: float = 1.0) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.activation = str(activation).lower()
        self.activation_scale = float(activation_scale)
        self._active_ids_getter: Optional[Callable[[], str]] = None
        self._flow_context_fetcher: Optional[Callable[[], Dict[str, Any]]] = None

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def set_active_ids_getter(self, getter: Optional[Callable[[], str]]) -> None:
        self._active_ids_getter = getter
        for layer in self.output_network:
            if hasattr(layer, "set_active_ids_getter"):
                layer.set_active_ids_getter(getter)

    def set_flow_context_fetcher(self, getter: Optional[Callable[[], Dict[str, Any]]]) -> None:
        self._flow_context_fetcher = getter
        for layer in self.output_network:
            if hasattr(layer, "set_flow_context_fetcher"):
                layer.set_flow_context_fetcher(getter)

    def forward(self, x, vec):
        orig_dtype = vec.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.to(torch.float32)
            vec_fp32 = vec.to(torch.float32)
            for layer in self.output_network:
                x_fp32, vec_fp32 = layer(x_fp32, vec_fp32)
            out = vec_fp32.squeeze()
            out = out.to(orig_dtype)

        if self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation not in {"", "identity", None}:
            out = torch.tanh(out)

        out = out * self.activation_scale
        return out


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    _instance_counter = 0
    _instances: List["GatedEquivariantBlock"] = []
    _debug_enabled = bool(int(os.getenv("PAINN_GEBLOCK_DEBUG", "0")))
    _norm_warn_limit = float(os.getenv("PAINN_GEBLOCK_NORM_LIMIT", "1.0e3"))
    _dump_dir = os.getenv("PAINN_GEBLOCK_DUMP_DIR", "/root/autodl-tmp/AdsorbDiff/pt")
    _topk = max(int(os.getenv("PAINN_GEBLOCK_TOPK", "3")), 0)
    _trace_full = bool(int(os.getenv("PAINN_GEBLOCK_TRACE_ALL", "1")))
    _peer_trace_limit = max(int(os.getenv("PAINN_GEBLOCK_PEER_LIMIT", "2")), 0)
    _dump_enabled = bool(int(os.getenv("PAINN_GEBLOCK_ENABLE_DUMP", "0")))

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ) -> None:
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels
        self._debug_id = GatedEquivariantBlock._instance_counter
        GatedEquivariantBlock._instance_counter += 1
        GatedEquivariantBlock._instances.append(self)
        self._last_inputs: Optional[Dict[str, torch.Tensor]] = None
        self._trace_cache: Dict[str, torch.Tensor] = {}
        self._trace_dumped = False
        self._active_ids_getter: Optional[Callable[[], str]] = None
        self._flow_context_fetcher: Optional[Callable[[], Dict[str, Any]]] = None
        self._internal_clip = float(os.getenv("PAINN_GEBLOCK_INTERNAL_CLIP", "0.0"))

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = ScaledSiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def _log_tensor_state(self, label: str, tensor: torch.Tensor) -> None:
        if not self._debug_enabled:
            return
        if tensor is None or not torch.is_tensor(tensor):
            logging.warning("[geblock][%d] %s unavailable ids=%s", self._debug_id, label, self._format_active_ids())
            return
        with torch.no_grad():
            data = tensor.detach().float()
            if data.device.type != "cpu":
                data = data.to("cpu")
            if data.numel() == 0:
                logging.warning(
                    "[geblock][%d] %s empty tensor ids=%s",
                    self._debug_id,
                    label,
                    self._format_active_ids(),
                )
                return
            self._cache_trace(label, data)
            finite_mask = torch.isfinite(data)
            abs_data = data.abs()
            max_abs = float(abs_data.max().item())

            sample_info = ""
            if data.dim() >= 1:
                sample_dim = data.shape[0]
                reshaped = abs_data.view(sample_dim, -1)
                sample_max, sample_argmax = reshaped.max(dim=1)
                topk = min(self._topk, sample_max.numel()) if sample_max.numel() > 0 else 0
                if topk > 0:
                    vals, idx = torch.topk(sample_max, k=topk)
                    entries = []
                    for rank in range(topk):
                        sid = int(idx[rank])
                        entries.append(
                            f"sample={sid} max={float(vals[rank]):.3e} flat_idx={int(sample_argmax[sid])}"
                        )
                    sample_info = " | ".join(entries)

            if not finite_mask.all():
                bad_vals = data[~finite_mask]
                preview = bad_vals.view(-1)[:5].tolist()
                logging.warning(
                    "[geblock][%d] %s non-finite detected preview=%s shape=%s %s ids=%s",
                    self._debug_id,
                    label,
                    preview,
                    tuple(data.shape),
                    sample_info,
                    self._format_active_ids(),
                )
                self._dump_tensor_snapshot(label, data, finite_mask)
                self._dump_trace_snapshot(label, "nonfinite")
                return

            if max_abs > self._norm_warn_limit:
                logging.warning(
                    "[geblock][%d] %s large magnitude max=%.4e limit=%.4e %s ids=%s",
                    self._debug_id,
                    label,
                    max_abs,
                    self._norm_warn_limit,
                    sample_info,
                    self._format_active_ids(),
                )
                self._dump_tensor_snapshot(label, data)
                self._dump_trace_snapshot(label, "overflow")

    def _dump_tensor_snapshot(
        self,
        label: str,
        data: torch.Tensor,
        finite_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not (self._debug_enabled and self._dump_enabled):
            return
        dump_dir = self._dump_dir or "/tmp"
        timestamp = int(time.time() * 1000)
        filename = f"geb_{self._debug_id}_{label}_{timestamp}.pt"
        path = os.path.join(dump_dir, filename)
        payload: Dict[str, Optional[torch.Tensor]] = {
            "label": label,
            "tensor": data.clone(),
            "finite_mask": finite_mask.clone() if torch.is_tensor(finite_mask) else None,
            "active_ids": self._format_active_ids(),
            "flow_context": self._format_flow_context(),
        }
        if isinstance(self._last_inputs, dict):
            x_in = self._last_inputs.get("x")
            v_in = self._last_inputs.get("v")
            if torch.is_tensor(x_in):
                payload["input_scalar"] = x_in.clone()
            if torch.is_tensor(v_in):
                payload["input_vec"] = v_in.clone()
        try:
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(payload, path)
            logging.warning(
                "[geblock][%d] dumped %s snapshot to %s ids=%s",
                self._debug_id,
                label,
                path,
                self._format_active_ids(),
            )
        except Exception as exc:
            logging.warning(
                "[geblock][%d] failed to dump %s snapshot: %s",
                self._debug_id,
                label,
                exc,
            )

    def _cache_trace(self, label: str, data: torch.Tensor) -> None:
        if not (self._debug_enabled and self._dump_enabled and self._trace_full and not self._trace_dumped):
            return
        try:
            self._trace_cache[label] = data.clone()
        except Exception:
            pass

    def _dump_trace_snapshot(self, trigger_label: str, reason: str) -> None:
        if not (self._debug_enabled and self._dump_enabled and self._trace_full) or self._trace_dumped:
            return
        dump_dir = self._dump_dir or "/tmp"
        timestamp = int(time.time() * 1000)
        filename = f"geb_{self._debug_id}_trace_{timestamp}.pt"
        path = os.path.join(dump_dir, filename)
        payload: Dict[str, Optional[torch.Tensor]] = {
            "trigger_label": trigger_label,
            "reason": reason,
            "trace": {k: v.clone() for k, v in self._trace_cache.items()},
            "input_scalar": None,
            "input_vec": None,
            "active_ids": self._format_active_ids(),
            "flow_context": self._format_flow_context(),
            "peer_traces": self._collect_peer_traces(),
        }
        if isinstance(self._last_inputs, dict):
            x_in = self._last_inputs.get("x")
            v_in = self._last_inputs.get("v")
            if torch.is_tensor(x_in):
                payload["input_scalar"] = x_in.clone()
            if torch.is_tensor(v_in):
                payload["input_vec"] = v_in.clone()
        try:
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(payload, path)
            logging.warning(
                "[geblock][%d] dumped full trace (trigger=%s) to %s ids=%s",
                self._debug_id,
                trigger_label,
                path,
                self._format_active_ids(),
            )
        except Exception as exc:
            logging.warning(
                "[geblock][%d] failed to dump full trace for %s: %s",
                self._debug_id,
                trigger_label,
                exc,
            )
        finally:
            self._trace_dumped = True

    def set_active_ids_getter(self, getter: Optional[Callable[[], str]]) -> None:
        self._active_ids_getter = getter

    def _format_active_ids(self) -> str:
        if callable(self._active_ids_getter):
            try:
                result = self._active_ids_getter()
                if isinstance(result, str):
                    return result
                return str(result)
            except Exception:
                return "unavailable"
        return "n/a"

    def set_flow_context_fetcher(self, getter: Optional[Callable[[], Dict[str, Any]]]) -> None:
        self._flow_context_fetcher = getter

    def _format_flow_context(self) -> Optional[Dict[str, Any]]:
        if not callable(self._flow_context_fetcher):
            return None
        try:
            ctx = self._flow_context_fetcher()
        except Exception:
            return None
        if not isinstance(ctx, dict):
            return None
        subset: Dict[str, Any] = {}
        for key in (
            "flow_debug",
            "v_tr_target",
            "v_rot_target",
            "t",
            "rot_symmetry",
            "sid",
            "fid",
        ):
            if key in ctx:
                subset[key] = ctx[key]
        return subset if subset else None

    def _clone_inputs(self) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(self._last_inputs, dict):
            return None
        cloned: Dict[str, torch.Tensor] = {}
        for key, tensor in self._last_inputs.items():
            if torch.is_tensor(tensor):
                try:
                    cloned[key] = tensor.clone()
                except Exception:
                    continue
        return cloned or None

    def _collect_peer_traces(self) -> Optional[Dict[str, Any]]:
        if self._peer_trace_limit <= 0:
            return None
        collected: Dict[str, Any] = {}
        peers = sorted(
            [blk for blk in GatedEquivariantBlock._instances if blk is not self and blk._debug_id < self._debug_id],
            key=lambda blk: blk._debug_id,
        )
        for blk in peers:
            if len(collected) >= self._peer_trace_limit:
                break
            peer_entry: Dict[str, Any] = {}
            if blk._trace_cache:
                peer_entry["trace"] = {k: v.clone() for k, v in blk._trace_cache.items()}
            inputs = blk._clone_inputs()
            if inputs:
                peer_entry["inputs"] = inputs
            peer_entry["active_ids"] = blk._format_active_ids()
            if peer_entry:
                collected[f"block_{blk._debug_id}"] = peer_entry
        return collected or None

    def forward(self, x, v):
        if self._debug_enabled:
            with torch.no_grad():
                try:
                    self._last_inputs = {
                        "x": x.detach().float().cpu(),
                        "v": v.detach().float().cpu(),
                    }
                except Exception:
                    self._last_inputs = None
            self._trace_cache = {}
            self._trace_dumped = False
        else:
            self._last_inputs = None
            self._trace_cache = {}
            self._trace_dumped = False

        self._log_tensor_state("input_scalar", x)
        self._log_tensor_state("input_vec", v)
        vec1_proj = self._apply_internal_clip("vec1_proj", self.vec1_proj(v))
        self._log_tensor_state("vec1_proj", vec1_proj)
        vec1 = torch.norm(vec1_proj, dim=-2)
        self._log_tensor_state("vec1_norm", vec1)
        vec2 = self._apply_internal_clip("vec2_proj", self.vec2_proj(v))
        self._log_tensor_state("vec2_proj", vec2)

        x_cat = torch.cat([x, vec1], dim=-1)
        self._log_tensor_state("update_input", x_cat)
        update_out = self._apply_internal_clip("update_output", self.update_net(x_cat))
        self._log_tensor_state("update_output", update_out)
        x, v_gate = torch.split(update_out, self.out_channels, dim=-1)
        x = self._apply_internal_clip("x_pre_gate", x)
        v_gate = self._apply_internal_clip("v_gate", v_gate)
        self._log_tensor_state("x_pre_gate", x)
        self._log_tensor_state("v_gate", v_gate)
        v = v_gate.unsqueeze(1) * vec2
        v = self._apply_internal_clip("vec_post_gate", v)
        self._log_tensor_state("vec_post_gate", v)

        x = self.act(x)
        x = self._apply_internal_clip("x_post_act", x)
        self._log_tensor_state("x_post_act", x)
        return x, v

    def _apply_internal_clip(self, label: str, tensor: torch.Tensor) -> torch.Tensor:
        limit = self._internal_clip
        if limit > 0.0 and torch.is_tensor(tensor):
            clipped = torch.clamp(tensor, min=-limit, max=limit)
            if self._debug_enabled and torch.any(clipped != tensor):
                logging.warning(
                    "[geblock][%d] %s clipped to +/-%.3e ids=%s",
                    self._debug_id,
                    label,
                    limit,
                    self._format_active_ids(),
                )
            return clipped
        return tensor


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing: bool = True,
    start_idx: int = 0,
    block_inc: int = 0,
    repeat_inc: int = 0,
) -> torch.Tensor:
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def get_edge_id(edge_idx, cell_offsets, num_atoms: int):
    cell_basis = cell_offsets.max() - cell_offsets.min() + 1
    cell_id = (
        (
            cell_offsets
            * cell_offsets.new_tensor([[1, cell_basis, cell_basis**2]])
        )
        .sum(-1)
        .long()
    )
    edge_id = edge_idx[0] + edge_idx[1] * num_atoms + cell_id * num_atoms**2
    return edge_id
