import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from torch_scatter import scatter
from ase.data import atomic_masses

from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import conditional_grad
from adsorbdiff.models.gemnet_oc.layers.base_layers import ScaledSiLU

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


class FourierFeatures(nn.Module):
    """NeRF-style sinusoidal positional encoding for scalar inputs."""

    def __init__(self, num_freqs: int = 16, log_max_freq: float = 4.0):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0.0, log_max_freq, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)  ->  (..., 2*num_freqs)
        x_proj = x * self.freqs  # broadcast
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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

        # === Energy conditioning (FiLM-style, matching PaiNN exactly) ===
        if energy_encoding == "fourier":
            num_freqs = 16
            self.energy_fourier = FourierFeatures(num_freqs=num_freqs, log_max_freq=4.0)
            self.energy_embedding = nn.Sequential(
                nn.Linear(2 * num_freqs, self.sphere_channels),
                ScaledSiLU(),
                nn.Linear(self.sphere_channels, self.sphere_channels * 2),
            )
            self.energy_null = nn.Parameter(torch.zeros(self.sphere_channels * 2))
        elif energy_encoding == "scalar":
            self.energy_fourier = None
            self.energy_embedding = nn.Sequential(
                nn.Linear(1, self.sphere_channels),
                ScaledSiLU(),
                nn.Linear(self.sphere_channels, self.sphere_channels * 2),
            )
            self.energy_null = nn.Parameter(torch.zeros(self.sphere_channels * 2))
        else:
            self.energy_fourier = None
            self.energy_embedding = None
            self.energy_null = None

        # === Time embedding (matching PaiNN exactly) ===
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.sphere_channels),
            ScaledSiLU(),
            nn.Linear(self.sphere_channels, self.sphere_channels),
        )

        # CFG dropout probability (set by trainer)
        self.p_cfg = 0.0
        # Flow regression target (set by trainer)
        self.fm_regression_target = "velocity"

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

        # === Initialize energy/time embeddings (matching PaiNN) ===
        self._reset_flow_embeddings()

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

    def _reset_flow_embeddings(self) -> None:
        """Initialize energy and time embeddings (matching PaiNN's reset_parameters)."""
        def _reset_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)

        if self.energy_embedding is not None:
            self.energy_embedding.apply(_reset_linear)
        if self.energy_null is not None:
            self.energy_null.data.zero_()
        self.time_embedding.apply(_reset_linear)

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

        # === Energy conditioning with CFG dropout (matching PaiNN exactly) ===
        if self.energy_embedding is not None:
            batch_idx = data.batch
            if hasattr(data, "natoms") and data.natoms is not None:
                batch_size = int(data.natoms.shape[0])
            else:
                batch_size = int(batch_idx.max().item()) + 1

            # Get energy values (matching PaiNN: no sampling check here)
            if hasattr(data, "energy") and data.energy is not None:
                node_energy = data.energy[batch_idx].unsqueeze(-1)
            else:
                node_energy = torch.zeros(batch_idx.size(0), 1, device=self.device)

            # Apply Fourier positional encoding if available
            if self.energy_fourier is not None:
                node_energy = self.energy_fourier(node_energy.float())

            # Compute energy embedding (matching PaiNN: use .float() then cast)
            energy_cond = self.energy_embedding(node_energy.float()).to(self.dtype)

            # === CFG dropout logic ===
            cfg_conditioned = getattr(data, "cfg_conditioned", None)
            if cfg_conditioned is not None:
                # Inference: use explicit cfg_conditioned flags
                if cfg_conditioned.numel() != batch_size:
                    raise ValueError("cfg_conditioned must have one flag per sample.")
                sample_mask = cfg_conditioned.to(device=self.device)
                node_mask = sample_mask[batch_idx].to(torch.bool)
            elif self.training and self.p_cfg > 0.0:
                # Training: random dropout
                keep_sample = torch.rand(batch_size, device=self.device) >= self.p_cfg
                node_mask = keep_sample[batch_idx]
            else:
                # No dropout
                node_mask = torch.ones(batch_idx.size(0), dtype=torch.bool, device=self.device)

            # Apply learnable null for dropped samples
            null_vec = self.energy_null.view(1, -1).to(device=self.device, dtype=self.dtype).expand_as(energy_cond)
            emb = torch.where(node_mask.unsqueeze(-1), energy_cond, null_vec)

            # FiLM conditioning: scale and shift (matching PaiNN exactly)
            # IMPORTANT: Use clone() to avoid inplace modification breaking autograd
            gamma, beta = torch.chunk(emb, 2, dim=-1)
            x_l0 = x.embedding[:, 0, :].clone()
            x_l0 = x_l0 * (1 + gamma) + beta
            x.embedding = torch.cat([x_l0.unsqueeze(1), x.embedding[:, 1:, :]], dim=1)

        # === Time embedding (matching PaiNN exactly) ===
        if hasattr(data, "t") and data.t is not None:
            t_val = data.t
            if t_val.dim() == 1:
                t_val = t_val.unsqueeze(-1)
            t_nodes = t_val[data.batch].to(self.dtype)
            time_emb = self.time_embedding(t_nodes.float())
            # IMPORTANT: Use clone() + concat to avoid inplace modification
            x_l0 = x.embedding[:, 0, :].clone() + time_emb
            x.embedding = torch.cat([x_l0.unsqueeze(1), x.embedding[:, 1:, :]], dim=1)

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
        # Force estimation (with output scaling to match PaiNN)
        ###############################################################

        forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3)
        # Apply tanh + scale for translation head (matching PaiNN: activation_scale=20)
        forces = torch.tanh(forces) * 40.0

        forces2 = None
        if self.FOR_denoising:
            forces2 = self.force_block2(
                x, atomic_numbers, edge_distance, edge_index
            )
            forces2 = forces2.embedding.narrow(1, 1, 3)
            forces2 = forces2.view(-1, 3)
            # Apply tanh + scale for rotation head
            # NOTE: Increase scale from 5 to 10 to compensate for direct mean pooling
            # (PaiNN uses torque->inertia conversion which amplifies the output)
            forces2 = torch.tanh(forces2) * 10.0

        if mode == "fm":
            return self._forward_flow(data, forces, forces2)

        if not self.FOR_denoising:
            return forces
        else:
            return forces, forces2

    def _forward_flow(self, data, forces: torch.Tensor, forces2: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Project per-atom outputs to rigid body (translation/rotation) flow heads.

        Key Insight for EquiformerV2:
        -----------------------------
        Unlike PaiNN which outputs polar vectors (forces), EquiformerV2's SO(3)-equivariant
        l=1 output with even parity (p=1) naturally represents **pseudovectors** (axial vectors).

        - Axis-angle rotation vectors are pseudovectors
        - Therefore, force_block2's l=1 output can DIRECTLY represent axis-angle velocities
        - No need for the torque-based conversion (L = r × F, ω = I⁻¹L) used in PaiNN

        This simplification:
        1. Avoids numerical issues from inertia tensor inversion
        2. Is more physically consistent with the equivariance properties
        3. Reduces computational overhead
        """
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
        ads_batch = batch_idx[ads_mask]
        device = data.pos.device

        # === 1. Translation: mean pooling of per-atom outputs ===
        # Translation velocity = mean of atomic "velocities" (rigid body translation)
        translation = scatter(ads_forces, ads_batch, dim=0, dim_size=batch_size, reduce="mean")

        # Handle allow_z flag
        allow_z = getattr(data, "allow_z", True)
        if isinstance(allow_z, torch.Tensor):
            allow_z = bool(allow_z.item()) if allow_z.numel() == 1 else True

        if not allow_z and translation.size(-1) >= 3:
            translation[:, 2] = 0.0

        # === 2. Rotation: DIRECT output (EquiformerV2 advantage) ===
        # EquiformerV2's l=1 output is a pseudovector, which can directly represent axis-angle.
        # Unlike PaiNN (which outputs polar vectors requiring torque conversion), EquiformerV2
        # learns to output "each atom's contribution to global ω" directly.
        # We simply aggregate per-atom contributions via mean pooling.
        if forces2 is not None:
            ads_forces2 = forces2[ads_mask]

            # Direct mean pooling - no torque conversion needed!
            # The network learns to output axis-angle velocity directly.
            rotation = scatter(ads_forces2, ads_batch, dim=0, dim_size=batch_size, reduce="mean")

            # Apply symmetry projection to constrain rotation to allowed DOFs
            rot_projector = getattr(data, "rot_projector", None)
            if torch.is_tensor(rot_projector):
                if rot_projector.device != rotation.device:
                    rot_projector = rot_projector.to(device=rotation.device, dtype=rotation.dtype)
                rotation = torch.matmul(rot_projector, rotation.unsqueeze(-1)).squeeze(-1)
        else:
            rotation = torch.zeros(
                batch_size,
                3,
                device=translation.device,
                dtype=translation.dtype,
            )

        # === 3. Assemble outputs ===
        outputs: Dict[str, torch.Tensor] = {
            "v_tr": translation,
            "v_rot": rotation,
        }
        return outputs
