"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import ase
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter

from adsorbdiff.relaxation.ase_utils import batch_to_atoms
from adsorbdiff.utils.utils import radius_graph_pbc


class LBFGS:
    def __init__(
        self,
        batch: Batch,
        model: "TorchCalc",
        maxstep: float = 0.01,
        memory: int = 100,
        damping: float = 0.25,
        alpha: float = 100.0,
        force_consistent=None,
        device: str = "cuda:0",
        save_full_traj: bool = True,
        traj_dir: Optional[Path] = None,
        traj_names=None,
        early_stop_batch: bool = False,
        max_adsorbate_surface_dist: Optional[float] = None,
        ads_clip_scale: float = 0.5,
        ads_clip_attempts: int = 3,
        ads_clip_margin: float = 0.5,
    ) -> None:
        self.batch = batch
        self.model = model
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.H0 = 1.0 / self.alpha
        self.force_consistent = force_consistent
        self.device = device
        self.save_full = save_full_traj
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.early_stop_batch = early_stop_batch
        self.otf_graph = model.model._unwrapped_model.otf_graph
        self.max_adsorbate_surface_dist = (
            float(max_adsorbate_surface_dist)
            if max_adsorbate_surface_dist is not None
            else None
        )
        self.ads_clip_scale = float(ads_clip_scale)
        self.ads_clip_attempts = int(ads_clip_attempts)
        self.ads_clip_margin = float(ads_clip_margin)
        if self.max_adsorbate_surface_dist is None or not hasattr(
            self.batch, "tags"
        ):
            # Disable clipping if the batch lacks tagging information or thresholds
            self.max_adsorbate_surface_dist = None
            self.ads_clip_attempts = 0
        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")

        if not self.otf_graph and "edge_index" not in batch:
            self.model.update_graph(self.batch)

    def get_energy_and_forces(self, apply_constraint: bool = True):
        energy, forces = self.model.get_energy_and_forces(
            self.batch, apply_constraint
        )
        return energy, forces

    def set_positions(self, update, update_mask) -> None:
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        self.batch.pos += update.to(dtype=torch.float32)

        if not self.otf_graph:
            self.model.update_graph(self.batch)

    def check_convergence(self, iteration, forces=None, energy=None):
        if forces is None or energy is None:
            energy, forces = self.get_energy_and_forces()
            forces = forces.to(dtype=torch.float64)

        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), self.batch.batch, reduce="max"
        )
        logging.info(
            f"{iteration} "
            + " ".join(f"{x:0.3f}" for x in max_forces_.tolist())
        )

        # (batch_size) -> (nAtoms)
        max_forces = max_forces_[self.batch.batch]

        return max_forces.ge(self.fmax), energy, forces

    def run(self, fmax, steps):
        self.fmax = fmax
        self.steps = steps

        self.s = deque(maxlen=self.memory)
        self.y = deque(maxlen=self.memory)
        self.rho = deque(maxlen=self.memory)
        self.r0 = self.f0 = None

        self.trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            self.trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in self.traj_names
            ]

        iteration = 0
        converged = False
        while iteration < steps and not converged:
            update_mask, energy, forces = self.check_convergence(iteration)
            converged = torch.all(torch.logical_not(update_mask))

            if self.trajectories is not None:
                if (
                    self.save_full
                    or converged
                    or iteration == steps - 1
                    or iteration == 0
                ):
                    self.write(energy, forces, update_mask)

            if not converged and iteration < steps - 1:
                self.step(iteration, forces, update_mask)

            iteration += 1

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}.traj_tmp", mode="w")
                traj_fl.rename(traj_fl.with_suffix(".traj"))

        self.batch.y, self.batch.force = self.get_energy_and_forces(
            apply_constraint=False
        )
        return self.batch

    def step(
        self,
        iteration: int,
        forces: Optional[torch.Tensor],
        update_mask: torch.Tensor,
    ) -> None:
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.batch.batch, reduce="max"
            )
            longest_steps = longest_steps[self.batch.batch]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        if forces is None:
            _, forces = self.get_energy_and_forces()

        r = self.batch.pos.clone().to(dtype=torch.float64)

        # Update s, y, rho
        if iteration > 0:
            s0 = (r - self.r0).flatten()
            self.s.append(s0)

            y0 = -(forces - self.f0).flatten()
            self.y.append(y0)

            self.rho.append(1.0 / torch.dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = forces.new_empty(loopmax)
        q = -forces.flatten()

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * torch.dot(self.s[i], q)  # b
            q -= alpha[i] * self.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * torch.dot(self.y[i], z)
            z += self.s[i] * (alpha[i] - beta)

        # descent direction
        p = -z.reshape((-1, 3))
        dr = determine_step(p)
        if (
            self.max_adsorbate_surface_dist is not None
            and self.ads_clip_attempts > 0
        ):
            dr = self._enforce_adsorbate_clip(dr)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        self.set_positions(dr, update_mask)

        self.r0 = r
        self.f0 = forces

    def write(self, energy, forces, update_mask) -> None:
        self.batch.y, self.batch.force = energy, forces
        atoms_objects = batch_to_atoms(self.batch)
        update_mask_ = torch.split(update_mask, self.batch.natoms.tolist())
        for atm, traj, mask in zip(
            atoms_objects, self.trajectories, update_mask_
        ):
            if mask[0] or not self.save_full:
                traj.write(atm)

    def _adsorbate_surface_distances(self, positions: torch.Tensor):
        """Return per-system min distance between adsorbate and surface atoms."""
        if self.max_adsorbate_surface_dist is None:
            return []
        natoms_list = self.batch.natoms.tolist()
        pos_split = torch.split(positions.detach(), natoms_list)
        tag_split = torch.split(self.batch.tags, natoms_list)
        distances = []
        for pos_i, tag_i in zip(pos_split, tag_split):
            ads_mask = tag_i == 2
            surf_mask = tag_i != 2
            if torch.count_nonzero(ads_mask) == 0 or torch.count_nonzero(
                surf_mask
            ) == 0:
                distances.append(None)
                continue
            ads_pos = pos_i[ads_mask].to(dtype=torch.float32)
            surf_pos = pos_i[surf_mask].to(dtype=torch.float32)
            dist = torch.cdist(ads_pos, surf_pos).min().item()
            distances.append(dist)
        return distances

    def _enforce_adsorbate_clip(self, dr: torch.Tensor) -> torch.Tensor:
        """Reduce per-system updates if the adsorbate moves too far from the surface."""
        if self.max_adsorbate_surface_dist is None:
            return dr
        natoms_list = self.batch.natoms.tolist()
        current_dist = self._adsorbate_surface_distances(self.batch.pos)
        if not any(d is not None for d in current_dist):
            return dr
        clipped = False
        threshold = self.max_adsorbate_surface_dist
        margin = max(self.ads_clip_margin, 0.0)
        scale = min(self.ads_clip_scale, 1.0)
        scale = scale if scale > 0.0 else 0.5
        for attempt in range(self.ads_clip_attempts):
            candidate_dist = self._adsorbate_surface_distances(
                self.batch.pos + dr
            )
            violation_found = False
            offset = 0
            for idx, (cur_d, new_d) in enumerate(
                zip(current_dist, candidate_dist)
            ):
                nat = natoms_list[idx]
                slc = slice(offset, offset + nat)
                offset += nat
                if cur_d is None or new_d is None:
                    continue
                if cur_d >= threshold - margin:
                    continue
                if new_d <= threshold:
                    continue
                violation_found = True
                clipped = True
                dr[slc] = dr[slc] * scale
            if not violation_found:
                if clipped:
                    logging.debug(
                        "[LBFGS] Reduced step size to prevent adsorbate desorption."
                    )
                return dr
        if clipped:
            logging.warning(
                "[LBFGS] Adsorbate-surface distance remained above threshold after clipping attempts."
            )
        return dr


class TorchCalc:
    def __init__(self, model, transform=None) -> None:
        self.model = model
        self.transform = transform

    def get_energy_and_forces(self, atoms, apply_constraint: bool = True):
        predictions = self.model.predict(
            atoms, per_image=False, disable_tqdm=True
        )
        energy = predictions["energy"]
        forces = predictions["forces"]
        if apply_constraint:
            fixed_idx = torch.where(atoms.fixed == 1)[0]
            forces[fixed_idx] = 0
        return energy, forces

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 50
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        if self.transform is not None:
            atoms = self.transform(atoms)
        return atoms
