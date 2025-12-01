import math
import torch
from typing import Optional, Dict, Any, Union
from torch_scatter import scatter
from ase.data import atomic_masses

from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import conditional_grad

try:
    from e3nn import o3
except ImportError:
    pass

from adsorbdiff.models.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2
from adsorbdiff.models.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
)
from adsorbdiff.models.equiformer_v2.equiformer_v2_oc20 import (
    EquiformerV2_OC20,
)
from adsorbdiff.models.embeddings import ATOMIC_RADII

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = (
    23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100
)


class EquiformerV2S_OC20_DenoisingPos(EquiformerV2_OC20):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=110,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        norm_type="rms_norm_sh",
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="scaled_silu",
        # use_tp_reparam=False,
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        weight_init="normal",
        # norm_scale_nodes=_AVG_NUM_NODES,
        # norm_scale_degree=_AVG_DEGREE,
        enforce_max_neighbors_strictly=True,
        so3_denoising=False,
        FOR_denoising=False,
        energy_encoding=None,
        sampling=False,
    ):
        super().__init__(
            num_atoms,  # not used
            bond_feat_dim,  # not used
            num_targets,  # not used
            use_pbc,
            regress_forces,
            otf_graph,
            max_neighbors,
            max_radius,
            max_num_elements,
            num_layers,
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            norm_type,
            lmax_list,
            mmax_list,
            grid_resolution,
            num_sphere_samples,
            edge_channels,
            use_atom_edge_embedding,
            share_atom_edge_embedding,
            use_m_share_rad,
            distance_function,
            num_distance_basis,
            attn_activation,
            # use_tp_reparam,
            use_s2_act_attn,
            use_attn_renorm,
            ffn_activation,
            use_gate_act,
            use_grid_mlp,
            use_sep_s2_act,
            alpha_drop,
            drop_path_rate,
            proj_drop,
            weight_init,
            # norm_scale_nodes,
            # norm_scale_degree,
            enforce_max_neighbors_strictly,
        )

        # for denoising position, encode node-wise forces as node features
        self.irreps_sh = o3.Irreps.spherical_harmonics(
            lmax=max(self.lmax_list), p=1
        )
        # self.force_embedding = SO3_LinearV2(
        #     in_features=1, out_features=self.sphere_channels, lmax=max(self.lmax_list)
        # )
        if energy_encoding == "scalar":
            self.energy_embedding = torch.nn.Linear(
                in_features=1, out_features=self.sphere_channels
            )

        if FOR_denoising:
            self.force_block2 = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        self.FOR_denoising = FOR_denoising
        self.sampling = sampling

        # Initialize atom radii with support for shifted indices
        atom_radii = torch.zeros(max_num_elements + 1, device=self.device)
        for i in range(min(101, len(ATOMIC_RADII))):
            atom_radii[i] = ATOMIC_RADII[i]
        
        # Map shifted elements (1, 6, 7, 8 -> 101, 106, 107, 108)
        # We assume the shift is always +100 for these elements as per tag_based_Z
        for el in [1, 6, 7, 8]:
            if el + 100 <= max_num_elements:
                atom_radii[el + 100] = ATOMIC_RADII[el]

        self.atom_radii = atom_radii / 100
        self.atom_radii = torch.nn.Parameter(atom_radii, requires_grad=False)

    def tag_based_Z(self, data) -> torch.Tensor:
        # This will create new embeddings for adsorbate atom types in slabs
        an = data.atomic_numbers
        cnho_an = [1, 6, 7, 8]
        mask = (data.tags < 2) & (
            (an == cnho_an[0])
            | (an == cnho_an[1])
            | (an == cnho_an[2])
            | (an == cnho_an[3])
        )
        an_mod = an.clone()
        an_mod[mask] += 100
        return an_mod

    @conditional_grad(torch.enable_grad())
    def forward(self, data, mode: Optional[str] = None):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = self.tag_based_Z(data).long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        # Account for atomic radii while incorporating edge distance
        edge_distance = (
            edge_distance
            - self.atom_radii[atomic_numbers[edge_index[0]]]
            - self.atom_radii[atomic_numbers[edge_index[1]]]
        )

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Add energy encoding for condition denoising
        if hasattr(self, "energy_embedding"):
            if not self.sampling:
                node_wise_y = data.energy[data.batch].unsqueeze(-1)
            else:
                node_wise_y = torch.zeros_like(data.batch).unsqueeze(-1)
            energy_embedding = self.energy_embedding(node_wise_y.to(self.dtype))
            x.embedding[:, 0, :] = x.embedding[:, 0, :] + energy_embedding

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[
                edge_index[0]
            ]  # Source atom atomic number
            target_element = atomic_numbers[
                edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Force estimation
        ###############################################################

        forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3)

        forces2 = None
        if self.FOR_denoising:
            forces2 = self.force_block2(
                x, atomic_numbers, edge_distance, edge_index
            )
            forces2 = forces2.embedding.narrow(1, 1, 3)
            forces2 = forces2.view(-1, 3)

        if mode == "fm":
            return self._forward_flow(data, forces, forces2)

        if not self.FOR_denoising:
            return forces
        else:
            return forces, forces2

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
        
        # 准备质量数据 (用于质心计算)
        an = data.atomic_numbers[ads_mask].long()
        device = data.pos.device
        mass_table = torch.tensor(atomic_masses, device=device, dtype=ads_forces.dtype)
        masses = mass_table[an]
        ads_batch = batch_idx[ads_mask]
        
        # 计算总质量 (per graph)
        total_mass = scatter(masses, ads_batch, dim=0, dim_size=batch_size, reduce="sum")
        total_mass = torch.clamp(total_mass, min=1e-6)

        # === 1. Translation 部分 (修改为: 质心速度 = 动量 / 总质量) ===
        # Translation is a polar vector. Use mass-weighted mean for CoM velocity.
        weighted_forces = ads_forces * masses.unsqueeze(-1)
        sum_weighted_forces = scatter(weighted_forces, ads_batch, dim=0, dim_size=batch_size, reduce="sum")
        translation = sum_weighted_forces / total_mass.unsqueeze(-1)
        
        if translation.size(-1) >= 3:
            translation[:, 2] = 0.0

        # === 2. Rotation 部分 (关键修改: 引入叉积聚合以匹配轴矢量宇称) ===
        if forces2 is not None:
            ads_forces2 = forces2[ads_mask]
            
            # (A) 计算吸附剂中心 (Center of Mass)
            ads_pos = data.pos[ads_mask]
            
            # 计算质心: sum(m * r) / sum(m)
            weighted_pos = ads_pos * masses.unsqueeze(-1)
            sum_weighted_pos = scatter(weighted_pos, ads_batch, dim=0, dim_size=batch_size, reduce="sum")
            centers = sum_weighted_pos / total_mass.unsqueeze(-1)
            
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
        return outputs
