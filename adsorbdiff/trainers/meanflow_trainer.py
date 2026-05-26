# =============================
# File: adsorbdiff/trainers/meanflow_trainer.py
# =============================
import copy
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch_geometric
from torch_scatter import scatter
from tqdm import tqdm
from ase.data import atomic_masses

from adsorbdiff.modules.evaluator import Evaluator
from adsorbdiff.modules.exponential_moving_average import ExponentialMovingAverage
from adsorbdiff.models.equiformer_v2.trainers.lr_scheduler import LRScheduler
from adsorbdiff.trainers import OCPTrainer
from adsorbdiff.utils import distutils, rot_utils
from adsorbdiff.utils.registry import registry
from adsorbdiff.modules.scaling.util import ensure_fitted
from adsorbdiff.relaxation.diffusers.flow_torch import FlowTorch


@torch.no_grad()
def _pbc_wrap_xy(vec_xyz: torch.Tensor, batch) -> torch.Tensor:
    """
    Wrap **COM/anchor displacement** to nearest PBC image on XY only.
    Only supports per-sample tensors of shape (B, 3), where B is batch size.

    NOTE:
    - Do NOT apply PBC wrapping per-atom for a rigid adsorbate.
      Per-atom wrapping can split a molecule crossing cell boundaries.
    """
    B = int(batch.natoms.size(0))
    if vec_xyz.shape[0] != B:
        raise ValueError(
            "_pbc_wrap_xy only supports per-sample (B,3) tensors. "
            "Apply PBC to the rigid COM/anchor displacement, not per-atom vectors."
        )
    out = torch.zeros_like(vec_xyz)
    cell = batch.cell.double()

    for b in range(B):
        frac = torch.linalg.solve(cell[b].t(), vec_xyz[b].double())
        frac[2] = 0.0
        frac[:2] = ((frac[:2] + 0.5) % 1.0) - 0.5
        cart = (frac.float() @ batch.cell[b].float())
        cart[2] = vec_xyz[b, 2]
        out[b] = cart
    return out


def _canonicalize_axis(axis: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Force a deterministic orientation to resolve the n vs -n ambiguity."""
    if axis is None or axis.numel() == 0:
        return axis
    abs_axis = torch.abs(axis)
    idx = int(torch.argmax(abs_axis))
    val = float(axis[idx].item())
    if val == 0.0:
        return axis
    sign = 1.0 if val > 0.0 else -1.0
    return axis * axis.new_tensor(sign)


def _is_linear_flip_symmetric(
    rel_points: torch.Tensor,
    atomic_numbers: Optional[torch.Tensor],
    axis: Optional[torch.Tensor],
    tol: float = 5.0e-2,
) -> bool:
    if axis is None or rel_points.numel() == 0 or atomic_numbers is None:
        return False
    axis_norm = torch.linalg.norm(axis)
    if axis_norm < 1.0e-8:
        return False
    axis_unit = axis / axis_norm
    coords = rel_points @ axis_unit
    order = torch.argsort(coords)
    coords = coords[order]
    atoms = atomic_numbers[order]
    flipped_coords = -coords.flip(0)
    coord_ok = torch.allclose(coords, flipped_coords, atol=tol)
    atoms_ok = torch.equal(atoms, atoms.flip(0))
    if coords.numel() % 2 == 1:
        mid = coords.shape[0] // 2
        coord_ok = coord_ok and abs(float(coords[mid].item())) <= tol
    return bool(coord_ok and atoms_ok)


def classify_adsorbate_symmetry(rel_points: torch.Tensor):
    """Standalone helper mirroring the trainer's PCA-based symmetry test."""
    device = rel_points.device
    dtype = rel_points.dtype if rel_points.numel() else torch.float32
    if rel_points.numel() == 0 or rel_points.shape[0] <= 1:
        return "spherical", None, torch.zeros(3, device=device, dtype=dtype)

    cov = rel_points.t().double() @ rel_points.double()
    cov = cov / max(rel_points.shape[0], 1)
    vals, vecs = torch.linalg.eigh(cov)
    vals = vals.float()
    vecs = vecs.float()

    max_val = vals.max()
    norm_vals = torch.zeros_like(vals)
    if not torch.isfinite(max_val) or max_val < 1.0e-8:
        return "spherical", None, norm_vals.to(device=device, dtype=dtype)

    norm_vals = (vals / max_val).to(device=device, dtype=dtype)
    spread = norm_vals[-1] - norm_vals[0]
    if spread < 1.0e-2:
        return "spherical", None, norm_vals

    linear_tol = 5.0e-2

    if norm_vals[1] < linear_tol and norm_vals[0] < linear_tol:
        axis = vecs[:, -1]
        if torch.linalg.norm(axis) > 0:
            return "linear", _canonicalize_axis(axis), norm_vals
        return "spherical", None, norm_vals

    return "general", None, norm_vals


@registry.register_trainer("meanflow")
def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "on"}:
            return True
        if norm in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


class _StaticSampler:
    def __init__(self, length: int) -> None:
        self.length = max(int(length), 1)

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:  # pragma: no cover - debug helper
        return


class _RepeatBatchLoader:
    def __init__(self, template_batch, length: int) -> None:
        self.template = self._prepare_batch(template_batch)
        self.length = max(int(length), 1)

    @staticmethod
    def _prepare_batch(batch):
        if hasattr(batch, "to"):
            batch = batch.to("cpu")
        return batch

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            yield self._clone_batch(self.template)

    @staticmethod
    def _clone_batch(batch):
        if hasattr(batch, "clone"):
            return batch.clone()
        return copy.deepcopy(batch)


class MeanFlowTrainer(OCPTrainer):
    """
    Trainer for Flow Matching (FM). Configure via config.optim.flow.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flow_cfg = self.config["optim"].get("flow", {})
        self.time_sampl = flow_cfg.get("time_sampl", "uniform")
        self.tr_sigma   = float(flow_cfg.get("tr_sigma", 2.0))
        self.tr_sigma_z_scale = float(flow_cfg.get("tr_sigma_z_scale", 0.3))
        self.allow_z = _parse_bool(flow_cfg.get("allow_z", True))
        self.rot_sigma  = float(flow_cfg.get("rot_sigma", 1.0))
        # lift_height removed as Z-axis is unlocked
        fm_regression = str(flow_cfg.get("fm_regression", "delta_clean")).lower()
        if fm_regression not in {"velocity", "vel"}:
            logging.warning(
                "[MeanFlowTrainer] fm_regression='%s' is no longer supported; falling back to 'velocity'.",
                fm_regression,
            )
        self.fm_regression_target = "velocity"
        tr_clip = flow_cfg.get("tr_clip", None)
        self.tr_clip = float(tr_clip) if tr_clip is not None else None
        rot_clip = flow_cfg.get("rot_clip", None)
        self.rot_clip = float(rot_clip) if rot_clip is not None else None
        self.grad_norm_guard = float(flow_cfg.get("grad_norm_guard", 0.0))

        # --- Curriculum Learning for Loss Weighting ---
        # Start (t=1 emphasis): 0.5 helps learn the global flow from noise.
        # End (t=0 emphasis): -0.5 helps refine the fine structure near data.
        self.fm_endpoint_weight_start = float(flow_cfg.get("endpoint_weight_exponent", 0))
        self.fm_endpoint_weight_target = float(flow_cfg.get("endpoint_weight_target",  0))
        self.fm_endpoint_weight_exponent = self.fm_endpoint_weight_start

        # Rotation loss weight: rebalance rad-scale rotation MSE vs A-scale
        # translation MSE. Setting this to (tr_sigma / rot_sigma)^2 makes the
        # two channels contribute commensurate gradient magnitudes; default 1.0
        # preserves backward compatibility.
        self.rot_loss_weight = float(flow_cfg.get("rot_loss_weight", 1.0))

        if distutils.is_master():
            logging.info(f"[MeanFlowTrainer] Loss Weight Schedule: {self.fm_endpoint_weight_start} -> {self.fm_endpoint_weight_target}")
            if self.rot_loss_weight != 1.0:
                logging.info(f"[MeanFlowTrainer] rot_loss_weight = {self.rot_loss_weight}")

        self.evaluator = Evaluator(task=self.name)
        self.ema = None
        p_cfg = float(self.config["optim"].get("p_cfg", 0.15))
        # 保证无论是否被 DDP/EMA 包装，都能把 dropout 概率写到真实模型上
        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = getattr(self.model, "model", self.model)
        base_model.p_cfg = p_cfg
        base_model.fm_regression_target = self.fm_regression_target
        ema_decay = self.config["optim"].get("ema_decay")
        if ema_decay:
            self.ema = ExponentialMovingAverage(self.model.parameters(), ema_decay)
        self.nan_retry_limit = int(flow_cfg.get("nan_retry_limit", 2))
        clip_cfg = flow_cfg.get("output_clip", 20)
        self.flow_output_clip_other = None
        self.flow_output_clip_tr = None
        self.flow_output_clip_rot = None
        if clip_cfg is None:
            pass
        elif isinstance(clip_cfg, dict):
            def _cast(val):
                return None if val is None else float(val)
            default_clip = _cast(clip_cfg.get("default"))
            self.flow_output_clip_other = default_clip
            self.flow_output_clip_tr = _cast(clip_cfg.get("translation", 20))
            self.flow_output_clip_rot = _cast(clip_cfg.get("rotation", 5))
        else:
            val = float(clip_cfg)
            self.flow_output_clip_other = val
            self.flow_output_clip_tr = val
            self.flow_output_clip_rot = val
        self.flow_debug_interval = int(flow_cfg.get("debug_interval", 100))
        self.flow_monitor_interval = int(flow_cfg.get("monitor_interval", 100))
        self.optimizer_log_interval = int(self.config["optim"].get("opt_log_interval", 100))
        self._set_active_batch_debug(None)
        self._cfg_flow_debug_counter = 0
        self._last_output_was_clipped = False
        self._clip_masks: Dict[str, torch.Tensor] = {}
        self._combined_clip_mask: Optional[torch.Tensor] = None
        self._flow_debug_tick = 0
        self._flow_monitor_tick = 0
        self._last_component_losses = None
        sanitize_debug_env = int(os.getenv("PAINN_SANITIZE_DEBUG", "0"))
        sanitize_debug_k_env = int(os.getenv("PAINN_SANITIZE_DEBUG_TOPK", "3"))
        sanitize_verbose_env = int(os.getenv("PAINN_SANITIZE_DEBUG_VERBOSE", "0"))
        self.flow_sanitize_debug = int(flow_cfg.get("sanitize_debug", sanitize_debug_env))
        self.flow_sanitize_debug_topk = int(flow_cfg.get("sanitize_debug_topk", sanitize_debug_k_env))
        self.flow_sanitize_debug_verbose = bool(flow_cfg.get("sanitize_debug_verbose", sanitize_verbose_env))
        nan_weight_env = int(os.getenv("PAINN_NAN_WEIGHT_DEBUG", "0"))
        self.flow_nan_weight_debug = bool(flow_cfg.get("nan_weight_debug", nan_weight_env))
        self._last_nan_weight_dump_step = -1
        self._last_force_pre_scatter_stats: Optional[Dict[str, Any]] = None
        self.single_batch_debug = _parse_bool(flow_cfg.get("single_batch_debug", False))
        self.single_batch_steps = int(flow_cfg.get("single_batch_steps", 1))
        seed_default = self.config["cmd"].get("seed")
        self.single_batch_seed = int(flow_cfg.get("single_batch_seed", seed_default if seed_default is not None else 0))
        self.single_batch_lock_sampling = _parse_bool(flow_cfg.get("single_batch_lock_sampling", True))
        self.single_batch_lock_interval = int(flow_cfg.get("single_batch_lock_interval", 0))
        if self.single_batch_lock_interval < 0:
            self.single_batch_lock_interval = 0
        self.single_batch_seed_increment = int(flow_cfg.get("single_batch_seed_increment", 1))
        if self.single_batch_seed_increment < 0:
            self.single_batch_seed_increment = 0
        self._single_batch_lock_counter = 0
        self._single_batch_current_seed = self.single_batch_seed
        self._single_batch_loader_active = False
        if self.single_batch_debug:
            self._enable_single_batch_debug_loader()

    def load_extras(self):
        super().load_extras()
        flow_cfg = self.config["optim"].get("flow", {})
        clip_val = flow_cfg.get("clip_grad_norm")
        if clip_val is None:
            clip_val = self.config["optim"].get("clip_grad_norm", 10.0)
        if clip_val:
            self.clip_grad_norm = float(clip_val)

    # ---------- A) build interpolant ----------
    @torch.no_grad()
    def _build_interpolant(self, batch):
        if self.single_batch_debug and self.single_batch_lock_sampling:
            self._reset_single_batch_rng(self._single_batch_current_seed)
            if self.single_batch_lock_interval > 0:
                self._single_batch_lock_counter += 1
                if self._single_batch_lock_counter >= self.single_batch_lock_interval:
                    self._single_batch_lock_counter = 0
                    self._single_batch_current_seed += self.single_batch_seed_increment
        device = batch.pos.device
        B = batch.natoms.size(0)
        x0 = batch.pos_relaxed if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None else batch.pos
        tags = batch.tags
        ads_mask = (tags == 2)

        # IMPORTANT: make adsorbate contiguous under PBC (XY) before computing COM/template.
        # If the dataset stores wrapped coordinates, a molecule crossing the unit cell can look
        # "split" (atoms on opposite sides). Taking a plain mean on those wrapped coords can
        # produce a polluted ads_center, which then corrupts both translation and rotation targets.
        x0_used = x0
        if ads_mask.any():
            x0_used = x0.clone()
            cell = batch.cell
            bidx = batch.batch
            B_local = int(batch.natoms.size(0))
            for bb in range(B_local):
                mask_bb = (bidx == bb) & ads_mask
                if not mask_bb.any():
                    continue
                idx_bb = torch.where(mask_bb)[0]
                ref = x0_used[idx_bb[0]]
                diffs = x0_used[idx_bb] - ref
                frac = torch.linalg.solve(cell[bb].double().t(), diffs.double().t()).t()
                frac[..., :2] = ((frac[..., :2] + 0.5) % 1.0) - 0.5
                diffs_wrapped = (frac.float() @ cell[bb].float())
                x0_used[idx_bb] = ref + diffs_wrapped

        ads_center = scatter(x0_used[ads_mask], batch.batch[ads_mask], dim=0, reduce="mean")
        # Use Center of Mass instead of Geometric Center
        # ads_atomic_numbers = batch.atomic_numbers[ads_mask]
        # masses = torch.tensor(atomic_masses[ads_atomic_numbers.long().cpu().numpy()], device=x0.device, dtype=x0.dtype)
        # weighted_pos = x0[ads_mask] * masses.unsqueeze(-1)
        # sum_weighted_pos = scatter(weighted_pos, batch.batch[ads_mask], dim=0, dim_size=B, reduce="sum")
        # sum_masses = scatter(masses, batch.batch[ads_mask], dim=0, dim_size=B, reduce="sum")
        # ads_center = sum_weighted_pos / sum_masses.unsqueeze(-1)

        if self.time_sampl == "uniform":
            t = torch.rand(B, device=device)
        else:
            u = torch.rand(B, device=device)
            t = torch.sin(0.5 * torch.pi * u) ** 2
        r = torch.zeros_like(t)
        # t = torch.clamp(t, min=0.02)#临时调整
        eps_tr = torch.zeros_like(ads_center).normal_() * self.tr_sigma
        # Unlock Z-axis in noise generation
        # eps_tr[:, -1] = 0.0
        if self.allow_z:
            eps_tr[:, 2] *= self.tr_sigma_z_scale
        else:
            eps_tr[:, 2] = 0.0
        eps_tr = _pbc_wrap_xy(eps_tr, batch)
        clip_flags = {}
        if self.tr_clip is not None:
            tr_norm = torch.linalg.norm(eps_tr[:, :2], dim=-1)
            over = tr_norm > self.tr_clip
            if over.any():
                scale = torch.ones_like(tr_norm)
                scale[over] = self.tr_clip / (tr_norm[over] + 1e-12)
                eps_tr[:, :2] = eps_tr[:, :2] * scale[:, None]
                clip_flags["translation"] = int(over.sum().item())

        ads_batch = batch.batch[ads_mask]
        rel_star = x0_used[ads_mask] - ads_center[ads_batch]
        base_atomic_numbers = getattr(batch, "atomic_numbers", None)
        ads_atomic_numbers = None
        if torch.is_tensor(base_atomic_numbers):
            ads_atomic_numbers = base_atomic_numbers[ads_mask]

        omega_list = []
        for _ in range(B):
            vec = rot_utils.sample_vec(eps=self.rot_sigma)
            omega_list.append(torch.tensor(vec, device=device, dtype=x0.dtype))
        omega = torch.stack(omega_list, dim=0)
        omega_sampled = omega.detach().clone()

        symmetry_labels = []
        rot_projectors = []
        rot_axes = []
        # Project angular velocities to remove redundant DOFs implied by molecular symmetry.
        sym_eigvals = []
        for b in range(B):
            idx = ads_batch == b
            if not idx.any():
                symmetry_labels.append("spherical")
                sym_eigvals.append(torch.zeros(3, dtype=rel_star.dtype, device=device))
                omega[b].zero_()
                rot_projectors.append(torch.zeros((3, 3), dtype=omega.dtype, device=device))
                rot_axes.append(torch.zeros(3, dtype=omega.dtype, device=device))
                continue
            rel_subset = rel_star[idx]
            sample_atomic_numbers = ads_atomic_numbers[idx] if ads_atomic_numbers is not None else None
            sym_type, axis, eigvals = self._classify_adsorbate_symmetry(rel_subset)
            sym_eigvals.append(eigvals.to(device=device, dtype=rel_star.dtype))
            identity = torch.eye(3, dtype=omega.dtype, device=device)

            if sym_type == "spherical":
                symmetry_labels.append("spherical")
                omega[b].zero_()
                rot_projectors.append(identity.new_zeros((3, 3)))
                rot_axes.append(identity.new_zeros(3))
                continue

            axis_unit = None
            if axis is not None:
                axis_t = axis.to(device=device, dtype=omega.dtype)
                axis_norm = torch.linalg.norm(axis_t)
                if axis_norm >= 1.0e-8:
                    axis_unit = axis_t / axis_norm
                    axis_unit = _canonicalize_axis(axis_unit)

            if axis_unit is None:
                symmetry_labels.append(sym_type)
                rot_projectors.append(identity)
                rot_axes.append(identity.new_zeros(3))
                continue

            projector = identity
            sym_label = sym_type
            if sym_type == "linear":
                if _is_linear_flip_symmetric(rel_subset, sample_atomic_numbers, axis_unit):
                    sym_label = "linear_flip"
                component = torch.dot(omega[b], axis_unit)
                omega[b] = omega[b] - component * axis_unit
                projector = identity - torch.outer(axis_unit, axis_unit)
            # Planar adsorbates are handled as generic 3D molecules, so no special-case projector.
            if not torch.isfinite(omega[b]).all():
                omega[b] = torch.zeros_like(omega[b])
            symmetry_labels.append(sym_label)
            rot_projectors.append(projector)
            rot_axes.append(axis_unit if axis_unit is not None else identity.new_zeros(3))

        if rot_projectors:
            batch.rot_projector = torch.stack(rot_projectors, dim=0)
        else:
            batch.rot_projector = torch.zeros((B, 3, 3), dtype=omega.dtype, device=device)
        batch.rot_symmetry = symmetry_labels
        if rot_axes:
            batch.rot_symmetry_axis = torch.stack(rot_axes, dim=0)
        else:
            batch.rot_symmetry_axis = torch.zeros((B, 3), dtype=omega.dtype, device=device)

        omega = torch.nan_to_num(omega)
        tiny_mask = torch.linalg.norm(omega, dim=-1) < 1.0e-8
        if tiny_mask.any():
            omega[tiny_mask] = 0.0

        if self.rot_clip is not None:
            rot_norm = torch.linalg.norm(omega, dim=-1)
            over_rot = rot_norm > self.rot_clip
            if over_rot.any():
                scale = torch.ones_like(rot_norm)
                scale[over_rot] = self.rot_clip / (rot_norm[over_rot] + 1e-12)
                omega = omega * scale[:, None]
                clip_flags["rotation"] = int(over_rot.sum().item())

        tr_sched = t
        ztr = ads_center + tr_sched[:, None] * eps_tr
        rot_sched = t
        zrot = rot_sched[:, None] * omega
        R_t = rot_utils.axis_angle_to_matrix(zrot)
        rel_t = torch.empty_like(rel_star)
        for b in range(B):
            idx = (ads_batch == b)
            if idx.any():
                rel_t[idx] = rel_star[idx] @ R_t[b].transpose(-1, -2)

        pos_t = x0.clone()
        # PBC handling for rigid adsorbate: wrap ONLY the COM/anchor displacement.
        # Never wrap per-atom displacements, which can split a molecule across the cell.
        ztr = ads_center + _pbc_wrap_xy(ztr - ads_center, batch)
        ads_pos_cur = rel_t + ztr[ads_batch]
        pos_t[ads_mask] = ads_pos_cur

        if not self.allow_z:
            # Move the adsorbate up by roughly 1A
            pos_t[ads_mask, 2] += 0.0


        batch.pos = pos_t
        tr_sched_deriv = torch.ones_like(t)
        rot_sched_deriv = torch.ones_like(t)
        # Directly use sampled noise as velocity targets so v_target stays constant over t
        v_tr_target = eps_tr.clone()
        v_rot_target = omega.clone()
        if not self.allow_z:
            v_tr_target[:, 2] = 0.0
        v_tr_target = torch.nan_to_num(v_tr_target)
        v_rot_target = torch.nan_to_num(v_rot_target)
        batch.t = t[:, None]
        batch.r = r[:, None]
        batch.ads_center = ads_center
        choice_tensor = self._compute_reflection_choice(v_rot_target, batch)
        if torch.is_tensor(choice_tensor):
            batch._rot_reflection_choice = choice_tensor.detach()
            aligned = self._align_discrete_symmetry_targets(v_rot_target, v_rot_target, batch)
            if torch.is_tensor(aligned):
                v_rot_target = aligned
        else:
            batch._rot_reflection_choice = None
        batch.v_tr_target = v_tr_target
        batch.v_rot_target = v_rot_target
        batch.allow_z = self.allow_z
        batch.tr_sched = tr_sched[:, None]
        batch.tr_sched_deriv = tr_sched_deriv[:, None]
        batch.rot_sched = rot_sched[:, None]
        batch.rot_sched_deriv = rot_sched_deriv[:, None]
        batch._flow_debug = self._make_flow_debug(
            batch,
            t,
            r,
            eps_tr,
            omega,
            ads_center,
            clip_flags,
            rot_sched,
            rot_sched_deriv,
            tr_sched,
            tr_sched_deriv,
            symmetry_labels,
        )
        if isinstance(batch._flow_debug, dict):
            try:
                batch._flow_debug["omega_sampled"] = omega_sampled.detach().cpu()
            except Exception:
                batch._flow_debug["omega_sampled"] = omega_sampled.detach()
            try:
                batch._flow_debug["omega_used"] = omega.detach().cpu()
            except Exception:
                batch._flow_debug["omega_used"] = omega.detach()
        if sym_eigvals:
            batch.rot_symmetry_eigvals = torch.stack(sym_eigvals, dim=0)
        else:
            batch.rot_symmetry_eigvals = torch.zeros((0, 3), dtype=rel_star.dtype, device=device)
        return batch

    def _clone_base_positions(self, batch):
        if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
            src = batch.pos_relaxed
        else:
            src = batch.pos
        return src.detach().cpu().clone()

    def _resample_interpolant(self, batch, base_pos):
        batch = batch.to("cpu")
        batch.pos = base_pos.clone()
        return self._build_interpolant(batch)

    def _reset_single_batch_rng(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.single_batch_seed
        self._single_batch_current_seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _enable_single_batch_debug_loader(self) -> None:
        if self.train_loader is None:
            logging.warning("[single-batch-debug] train_loader is None; skipping single batch override.")
            self.single_batch_debug = False
            return
        iterator = iter(self.train_loader)
        try:
            first_batch = next(iterator)
        except StopIteration:
            logging.warning("[single-batch-debug] train_loader was empty; skipping override.")
            self.single_batch_debug = False
            return
        steps = max(int(self.single_batch_steps), 1)
        self.train_loader = _RepeatBatchLoader(first_batch, steps)
        self.train_sampler = _StaticSampler(steps)
        self._single_batch_loader_active = True
        if getattr(self, "val_loader", None) is not None:
            logging.info("[single-batch-debug] disabling val_loader to avoid extra work during overfit test.")
            self.val_loader = None
        logging.info(
            "[single-batch-debug] Enabled deterministic loader: steps=%d seed=%d lock_sampling=%s lock_interval=%d seed_increment=%d",
            steps,
            self.single_batch_seed,
            str(self.single_batch_lock_sampling),
            self.single_batch_lock_interval,
            self.single_batch_seed_increment,
        )

    def _compute_rotation_head_grad_norm(self) -> Optional[float]:
        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = getattr(self.model, "module", None) or self.model
        rotation_head = getattr(base_model, "out_forces2", None)
        if rotation_head is None:
            return None
        sq_sum = 0.0
        has_grad = False
        for param in rotation_head.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            has_grad = True
            sq_sum += float(torch.sum(grad.detach().float() ** 2).item())
        if not has_grad:
            return 0.0
        return float(np.sqrt(max(sq_sum, 0.0)))

    def _classify_adsorbate_symmetry(self, rel_points: torch.Tensor):
        """Classify adsorbate shape to determine rotational symmetries."""
        return classify_adsorbate_symmetry(rel_points)

    def _tensor_stats(self, name, tensor):
        if tensor is None:
            logging.warning(f"[debug] {name}: None")
            return
        tensor = tensor.detach().cpu()
        finite = torch.isfinite(tensor)
        total = tensor.numel()
        finite_count = int(finite.sum())
        finite_ratio = finite_count / max(total, 1)
        if finite_count == 0:
            logging.warning(f"[debug] {name}: shape={tuple(tensor.shape)} all non-finite")
            return
        data = tensor[finite]
        min_val = float(data.min())
        max_val = float(data.max())
        mean_val = float(data.mean())
        logging.warning(
            f"[debug] {name}: shape={tuple(tensor.shape)} finite_ratio={finite_ratio:.3f} min={min_val:.4e} max={max_val:.4e} mean={mean_val:.4e}"
        )

    def _log_tensor_summary(self, name, tensor):
        if not torch.is_tensor(tensor):
            logging.warning(f"[sanitize-raw] {name}: not a tensor ({type(tensor)})")
            return
        with torch.no_grad():
            flat = tensor.detach().view(-1)
            finite = torch.isfinite(flat)
            finite_count = int(finite.sum().item())
            total = flat.numel()
            msg = [
                f"size={tuple(tensor.shape)}",
                f"finite={finite_count}/{total}",
            ]
            if finite_count > 0:
                data = flat[finite]
                msg.extend(
                    [
                        f"min={float(data.min()):.4e}",
                        f"max={float(data.max()):.4e}",
                        f"mean={float(data.mean()):.4e}",
                    ]
                )
            logging.warning(f"[sanitize-raw] {name} " + ", ".join(msg))

    def _format_sample_ids(self, sample_idx: int) -> str:
        info = getattr(self, "_active_batch_debug", None)
        if not isinstance(info, dict):
            return "n/a"
        ids = info.get("ids")
        if not isinstance(ids, dict):
            return "n/a"
        parts = []
        for key, values in ids.items():
            value = None
            if isinstance(values, (list, tuple)):
                if sample_idx < len(values):
                    value = values[sample_idx]
            elif torch.is_tensor(values):
                if values.dim() > 0 and sample_idx < values.shape[0]:
                    value = values[sample_idx].item()
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return ",".join(parts) if parts else "n/a"

    def _log_sanitize_details(self, key: str, tensor: torch.Tensor, sample_mask: torch.Tensor) -> None:
        if not self.flow_sanitize_debug or tensor.dim() == 0 or sample_mask is None:
            return
        if self.flow_sanitize_debug_topk <= 0:
            return
        with torch.no_grad():
            mask_cpu = sample_mask.detach().cpu()
            idxs = torch.nonzero(mask_cpu, as_tuple=False).view(-1)
            if idxs.numel() == 0:
                return
            topk = min(self.flow_sanitize_debug_topk, idxs.numel())
            tensor_cpu = tensor.detach().cpu()
            step_val = int(getattr(self, "step", -1))
            for local_idx in idxs[:topk]:
                sample_idx = int(local_idx.item())
                if sample_idx >= tensor_cpu.shape[0]:
                    continue
                sample = tensor_cpu[sample_idx]
                flat = sample.reshape(-1).float()
                finite = torch.isfinite(flat)
                finite_ratio = float(finite.sum().item()) / max(flat.numel(), 1)
                if finite.any():
                    min_val = float(flat[finite].min().item())
                    max_val = float(flat[finite].max().item())
                    mean_val = float(flat[finite].mean().item())
                    std_val = float(flat[finite].std(unbiased=False).item()) if flat[finite].numel() > 1 else 0.0
                else:
                    min_val = float("nan")
                    max_val = float("nan")
                    mean_val = float("nan")
                    std_val = float("nan")
                l2_val = float(torch.linalg.norm(flat.float()).item()) if flat.numel() > 0 else 0.0
                ids = self._format_sample_ids(sample_idx)
                logging.warning(
                    "[sanitize-debug] step=%d key=%s sample=%d ids=%s finite_ratio=%.3f min=%.4e max=%.4e mean=%.4e std=%.4e l2=%.4e",
                    step_val,
                    key,
                    sample_idx,
                    ids,
                    finite_ratio,
                    min_val,
                    max_val,
                    mean_val,
                    std_val,
                    l2_val,
                )
                if self.flow_sanitize_debug_verbose:
                    logging.warning(
                        "[sanitize-debug] step=%d key=%s sample=%d raw=%s",
                        step_val,
                        key,
                        sample_idx,
                        sample.tolist(),
                    )

    def _log_weight_snapshot(self, reason: str) -> None:
        if not self.flow_nan_weight_debug:
            return
        step_val = int(getattr(self, "step", -1))
        if self._last_nan_weight_dump_step == step_val:
            return
        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = self.model
        if base_model is None:
            return
        optimizer = getattr(self, "optimizer", None)
        logging.warning(
            "[nan-weights] step=%d reason=%s dumping parameter stats (once per step)",
            step_val,
            reason,
        )
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if not torch.is_tensor(param):
                    continue
                param_summary = self._tensor_summary(param)
                logging.warning("[nan-weights] param=%s | %s", name, param_summary)
                grad = param.grad
                if torch.is_tensor(grad):
                    grad_summary = self._tensor_summary(grad)
                    logging.warning("[nan-weights] grad=%s | %s", name, grad_summary)
                if optimizer is not None and hasattr(optimizer, "state") and param in optimizer.state:
                    state = optimizer.state[param]
                    if isinstance(state, dict):
                        state_step = state.get("step")
                        if state_step is not None:
                            logging.warning("[nan-weights] opt_state=%s.step=%s", name, state_step)
                        for state_name in ("exp_avg", "exp_avg_sq"):
                            state_tensor = state.get(state_name)
                            if torch.is_tensor(state_tensor):
                                state_summary = self._tensor_summary(state_tensor)
                                logging.warning(
                                    "[nan-weights] opt_state=%s.%s | %s",
                                    name,
                                    state_name,
                                    state_summary,
                                )
        self._last_nan_weight_dump_step = step_val

    def _log_force_pre_scatter_stats(self, key: str, sample_mask: Optional[torch.Tensor]) -> None:
        stats = getattr(self, "_last_force_pre_scatter_stats", None)
        if not isinstance(stats, dict):
            return
        global_max = stats.get("global_max_abs")
        has_nonfinite = stats.get("has_nonfinite")
        try:
            global_str = f"{float(global_max):.4e}"
        except (TypeError, ValueError):
            global_str = "n/a"
        logging.warning(
            "[sanitize] %s pre-scatter max|f_atom|=%s has_nonfinite=%s",
            key,
            global_str,
            bool(has_nonfinite),
        )
        per_sample = stats.get("per_sample_max_abs")
        if (
            not self.flow_sanitize_debug
            or self.flow_sanitize_debug_topk <= 0
            or sample_mask is None
            or not torch.is_tensor(per_sample)
        ):
            return
        sample_mask_cpu = sample_mask.detach().cpu()
        per_sample_cpu = per_sample.detach().cpu()
        if per_sample_cpu.shape[0] != sample_mask_cpu.shape[0]:
            return
        flagged_idx = torch.nonzero(sample_mask_cpu, as_tuple=False).view(-1)
        if flagged_idx.numel() == 0:
            return
        flagged_vals = per_sample_cpu[flagged_idx]
        order = torch.argsort(flagged_vals, descending=True)
        topk = min(int(self.flow_sanitize_debug_topk), order.numel())
        for rank in range(topk):
            sample_idx = int(flagged_idx[order[rank]])
            val = float(flagged_vals[order[rank]].item())
            sample_ids = self._format_sample_ids(sample_idx)
            logging.warning(
                "[sanitize] sample=%d ids=%s pre-scatter max|f_atom|=%.4e",
                sample_idx,
                sample_ids,
                val,
            )

    def _log_nonfinite_debug(self, loss, out, batch, attempt, context):
        loss_val = float(loss.detach().cpu()) if loss is not None and torch.isfinite(loss) else "nan"
        logging.warning(
            f"[{context}] non-finite loss debug: attempt={attempt + 1}/{self.nan_retry_limit + 1} step={self.step} loss={loss_val}"
        )
        if out is not None:
            for key in ["v_tr", "v_rot"]:
                if key in out:
                    self._tensor_stats(f"out[{key}]", out[key])
        if batch is not None:
            self._tensor_stats("batch.t", batch.t)
            self._tensor_stats("batch.r", batch.r)
            if hasattr(batch, "ztr"):
                self._tensor_stats("batch.ztr", batch.ztr)
            if hasattr(batch, "zrot"):
                self._tensor_stats("batch.zrot", batch.zrot)
            if hasattr(batch, "v_tr_target"):
                self._tensor_stats("batch.v_tr_target", batch.v_tr_target)
            if hasattr(batch, "v_rot_target"):
                self._tensor_stats("batch.v_rot_target", batch.v_rot_target)
        self._log_flow_debug_from_batch(batch, context)

    # ---------- B) forward ----------
    def _forward(self, batch):
        batch = batch.to(self.device)
        self._last_output_was_clipped = False
        self._clip_masks = {}
        self._combined_clip_mask = None
        self._sync_model_active_batch_debug()
        self._last_force_pre_scatter_stats = None
        out = self.model(batch, mode="fm")
        if isinstance(out, dict) and "_debug_force_pre_scatter" in out:
            self._last_force_pre_scatter_stats = out.pop("_debug_force_pre_scatter")
        self._apply_rot_projection(out, batch)
        self._log_cfg_flow_debug(batch)
        return self._sanitize_outputs(out)

    def _log_cfg_flow_debug(self, batch) -> None:
        debug_flag = os.getenv("PAINN_DEBUG_FLOW", "")
        monitor_due = False
        if self.flow_monitor_interval > 0 and self.model.training:
            self._flow_monitor_tick = (self._flow_monitor_tick + 1) % self.flow_monitor_interval
            monitor_due = self._flow_monitor_tick == 0

        if not debug_flag and not monitor_due:
            return
        debug_limit = int(os.getenv("PAINN_DEBUG_FLOW_MAX", "10")) if debug_flag else None
        if debug_flag and not monitor_due and self._cfg_flow_debug_counter >= debug_limit:
            return
        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            return
        had_sampling_attr = hasattr(base_model, "sampling")
        old_mode = base_model.training
        old_sampling = getattr(base_model, "sampling", None)
        with torch.no_grad():
            base_model.eval()
            if had_sampling_attr:
                base_model.sampling = True

            def _clone_batch(src):
                if hasattr(src, "clone"):
                    return src.clone()
                return copy.deepcopy(src)

            batch_un = _clone_batch(batch)
            batch_co = _clone_batch(batch)

            device = batch.pos.device if hasattr(batch, "pos") else self.device
            if hasattr(batch, "natoms") and batch.natoms is not None:
                batch_size = int(batch.natoms.shape[0])
            else:
                batch_size = int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 0

            if batch_size > 0:
                batch_un.cfg_conditioned = torch.zeros(batch_size, dtype=torch.bool, device=device)
                batch_co.cfg_conditioned = torch.ones(batch_size, dtype=torch.bool, device=device)

            if hasattr(batch_un, "to"):
                batch_un = batch_un.to(device)
            if hasattr(batch_co, "to"):
                batch_co = batch_co.to(device)

            out_un = base_model(batch_un, mode="fm")
            if had_sampling_attr:
                base_model.sampling = False
            out_co = base_model(batch_co, mode="fm")
            if had_sampling_attr:
                base_model.sampling = old_sampling
            if old_mode:
                base_model.train()
            else:
                base_model.eval()
        self._apply_rot_projection(out_un, batch_un)
        self._apply_rot_projection(out_co, batch_co)
        out_un = self._sanitize_outputs(out_un)
        out_co = self._sanitize_outputs(out_co)
        key_tr = "v_tr"
        key_rot = "v_rot"
        if key_tr not in out_un or key_tr not in out_co:
            return
        try:
            delta_tr = out_co[key_tr] - out_un[key_tr]
            rot_un = out_un.get(key_rot, None)
            rot_co = out_co.get(key_rot, None)
            delta_rot = (rot_co - rot_un) if (rot_co is not None and rot_un is not None) else None
        except Exception:
            return
        with torch.no_grad():
            tr_delta_norm = delta_tr.norm(dim=-1)
            rot_delta_norm = delta_rot.norm(dim=-1) if delta_rot is not None else None
            tr_un_norm = out_un[key_tr].norm(dim=-1)
            tr_co_norm = out_co[key_tr].norm(dim=-1)
            rot_un_norm = rot_un.norm(dim=-1) if rot_un is not None else None
            rot_co_norm = rot_co.norm(dim=-1) if rot_co is not None else None
            def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
                return (a * b).sum(dim=-1) / denom
            cos_tr = _cosine(out_un[key_tr], out_co[key_tr])
            cos_tr_delta = _cosine(out_un[key_tr], delta_tr)
            cos_rot = _cosine(rot_un, rot_co) if rot_un is not None and rot_co is not None else None
            cos_rot_delta = _cosine(rot_un, delta_rot) if rot_un is not None and delta_rot is not None else None
            label = "CFG-debug" if debug_flag else "CFG-monitor"
            logging.info(
                "[PaiNN][%s][step=%d] |Δtr|mean=%.4e max=%.4e |Δrot|mean=%s max=%s |tr_un|mean=%.4e |tr_co|mean=%.4e",
                label,
                int(getattr(self, "step", -1)),
                float(tr_delta_norm.mean().item()),
                float(tr_delta_norm.max().item()),
                f"{float(rot_delta_norm.mean().item()):.4e}" if rot_delta_norm is not None else "n/a",
                f"{float(rot_delta_norm.max().item()):.4e}" if rot_delta_norm is not None else "n/a",
                float(tr_un_norm.mean().item()),
                float(tr_co_norm.mean().item()),
            )
            if rot_un_norm is not None and rot_co_norm is not None:
                logging.info(
                    "[PaiNN][%s][step=%d] |rot_un|mean=%.4e |rot_co|mean=%.4e",
                    label,
                    int(getattr(self, "step", -1)),
                    float(rot_un_norm.mean().item()),
                    float(rot_co_norm.mean().item()),
                )
            logging.info(
                "[PaiNN][%s][step=%d] cos(tr_un,tr_co)=%.4f | cos(tr_un,Δtr)=%.4f",
                label,
                int(getattr(self, "step", -1)),
                float(cos_tr.mean().item()),
                float(cos_tr_delta.mean().item()),
            )
            if cos_rot is not None and cos_rot_delta is not None:
                logging.info(
                    "[PaiNN][%s][step=%d] cos(rot_un,rot_co)=%.4f | cos(rot_un,Δrot)=%.4f",
                    label,
                    int(getattr(self, "step", -1)),
                    float(cos_rot.mean().item()),
                    float(cos_rot_delta.mean().item()),
                )
        if debug_flag:
            self._cfg_flow_debug_counter += 1

    def _sanitize_outputs(self, out):
        if not isinstance(out, dict):
            return out
        safe_out = {}
        clipped_flow_head = False
        combined_mask = None
        for key, tensor in out.items():
            if not torch.is_tensor(tensor):
                safe_out[key] = tensor
                continue
            raw_tensor = tensor.detach()
            if not torch.isfinite(tensor).all():
                logging.warning(f"[sanitize] detected non-finite values in {key}, applying nan_to_num")
                self._log_tensor_summary(f"raw {key}", tensor)
            clip = self.flow_output_clip_other
            if key in {"v_tr"} and self.flow_output_clip_tr is not None:
                clip = self.flow_output_clip_tr
            elif key in {"v_rot"} and self.flow_output_clip_rot is not None:
                clip = self.flow_output_clip_rot
            clip_mask_sample = None
            sample_dim = tensor.shape[0] if tensor.dim() > 0 else None
            nan_mask_sample = None
            if sample_dim is not None:
                if raw_tensor.numel() == 0:
                    nan_mask_sample = torch.zeros(sample_dim, dtype=torch.bool, device=raw_tensor.device)
                else:
                    reshaped = raw_tensor.reshape(sample_dim, -1)
                    nan_mask_sample = ~torch.isfinite(reshaped).all(dim=-1)
            if (
                nan_mask_sample is not None
                and nan_mask_sample.numel() > 0
                and bool(nan_mask_sample.any().item())
            ):
                self._log_weight_snapshot(reason=f"{key}-nan")
                self._log_force_pre_scatter_stats(key, nan_mask_sample)
            if clip is not None and sample_dim is not None:
                over_limit = torch.abs(tensor) > clip
                if over_limit.dim() == 0:
                    over_limit = over_limit.unsqueeze(0)
                clip_mask_sample = over_limit.view(over_limit.shape[0], -1).any(dim=-1)
            anomaly_mask = None
            if nan_mask_sample is not None:
                anomaly_mask = nan_mask_sample if anomaly_mask is None else (anomaly_mask | nan_mask_sample)
            if clip_mask_sample is not None:
                anomaly_mask = clip_mask_sample if anomaly_mask is None else (anomaly_mask | clip_mask_sample)
            if anomaly_mask is not None and anomaly_mask.any():
                self._log_sanitize_details(key, raw_tensor, anomaly_mask)
            if clip is not None and clip_mask_sample is not None and clip_mask_sample.any():
                logging.warning(f"[sanitize] clipping {key} to ±{clip}")
                self._log_tensor_summary(f"raw {key}", tensor)
                if key in {"v_tr", "v_rot"}:
                    clipped_flow_head = True
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
                sanitized = torch.clamp(tensor, min=-clip, max=clip)
            else:
                sanitized = torch.nan_to_num(tensor, nan=0.0)
            if clip_mask_sample is None and sample_dim is not None:
                clip_mask_sample = torch.zeros(sample_dim, dtype=torch.bool, device=sanitized.device)
            if not torch.isfinite(sanitized).all():
                logging.warning(f"[sanitize] still non-finite after nan_to_num for {key}, zeroing problematic entries")
                sanitized = torch.where(torch.isfinite(sanitized), sanitized, torch.zeros_like(sanitized))
            safe_out[key] = sanitized
            if sanitized.dim() > 0 and clip_mask_sample is not None:
                self._clip_masks[key] = clip_mask_sample
                if combined_mask is None:
                    combined_mask = clip_mask_sample.clone()
                else:
                    combined_mask = combined_mask.to(clip_mask_sample.device)
                    combined_mask = combined_mask | clip_mask_sample
        if clipped_flow_head:
            if combined_mask is None:
                combined_mask = torch.zeros(1, dtype=torch.bool, device=self.device)
            self._combined_clip_mask = combined_mask
            if combined_mask.numel() > 0 and bool(torch.all(combined_mask).item()):
                self._last_output_was_clipped = True
            else:
                self._last_output_was_clipped = False
        else:
            self._combined_clip_mask = combined_mask
        return safe_out

    def _apply_rot_projection(self, out, batch) -> None:
        """Project rotational flows into the symmetry-allowed subspace before clipping/logging."""
        if not isinstance(out, dict) or "v_rot" not in out:
            return
        projector = getattr(batch, "rot_projector", None)
        if projector is None:
            return
        v_rot = out.get("v_rot")
        if not torch.is_tensor(v_rot):
            return
        if not torch.is_tensor(projector):
            projector = torch.as_tensor(projector)
        if projector.ndim != 3 or projector.shape[-2:] != (3, 3):
            logging.debug("[rot-projector] Unexpected projector shape: %s", projector.shape if hasattr(projector, "shape") else type(projector))
            return
        if projector.shape[0] != v_rot.shape[0]:
            logging.debug(
                "[rot-projector] Batch/projector size mismatch (proj=%d, batch=%d); skipping projection.",
                int(projector.shape[0]),
                int(v_rot.shape[0]),
            )
            return
        if projector.device != v_rot.device:
            projector = projector.to(v_rot.device)

        target_rot = getattr(batch, "v_rot_target", None)
        if torch.is_tensor(target_rot):
            if target_rot.shape[-1] > v_rot.shape[-1]:
                target_rot = target_rot[:, : v_rot.shape[-1]]
            target_rot = target_rot.to(v_rot)
        else:
            target_rot = None

        with torch.no_grad():
            pre_proj_pred = v_rot.detach().clone()
            pre_proj_target = target_rot.detach().clone() if torch.is_tensor(target_rot) else None

        projected_pred = torch.matmul(projector, v_rot.unsqueeze(-1)).squeeze(-1)
        projected_target = None
        if torch.is_tensor(target_rot):
            projected_target = torch.matmul(projector, target_rot.unsqueeze(-1)).squeeze(-1)
            projected_target = self._align_discrete_symmetry_targets(projected_pred, projected_target, batch)

        with torch.no_grad():
            out["_debug_rot_projection"] = {
                "pred_before": pre_proj_pred,
                "pred_after": projected_pred.detach().clone(),
                "target_before": pre_proj_target,
                "target_after": projected_target.detach().clone() if torch.is_tensor(projected_target) else None,
            }

        out["v_rot"] = projected_pred

    def _align_discrete_symmetry_targets(
        self,
        pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        batch,
    ) -> Optional[torch.Tensor]:
        if target is None or not torch.is_tensor(target) or pred is None or not torch.is_tensor(pred):
            return target
        sym_labels = getattr(batch, "rot_symmetry", None)
        sym_axes = getattr(batch, "rot_symmetry_axis", None)
        if not isinstance(sym_labels, list) or not sym_labels:
            return target
        if not torch.is_tensor(sym_axes):
            return target
        cached_choice = getattr(batch, "_rot_reflection_choice", None)
        choice_tensor = None
        if torch.is_tensor(cached_choice) and cached_choice.shape[0] == pred.shape[0]:
            choice_tensor = cached_choice.to(device=pred.device, dtype=torch.bool)
        if choice_tensor is None:
            choice_tensor = self._compute_reflection_choice(target, batch)
            if torch.is_tensor(choice_tensor):
                batch._rot_reflection_choice = choice_tensor.detach()
        axes = sym_axes.to(device=pred.device, dtype=pred.dtype)
        axis_norm = torch.linalg.norm(axes, dim=-1, keepdim=True)
        valid_axis = axis_norm > 1.0e-8
        mask_linear = torch.tensor(
            [lbl == "linear_flip" for lbl in sym_labels],
            device=pred.device,
            dtype=torch.bool,
        )
        mask_linear = mask_linear & valid_axis.squeeze(-1)
        if not mask_linear.any():
            return target
        target_adj = target.clone()
        new_choice = None
        if choice_tensor is None:
            new_choice = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)

        axes_linear = axes[mask_linear] / axis_norm[mask_linear]
        pred_linear = pred[mask_linear]
        tgt_linear = target_adj[mask_linear]
        reflected_linear = 2.0 * (tgt_linear * axes_linear).sum(dim=-1, keepdim=True) * axes_linear - tgt_linear
        if choice_tensor is None:
            diff_orig = (pred_linear - tgt_linear).pow(2).sum(dim=-1)
            diff_ref = (pred_linear - reflected_linear).pow(2).sum(dim=-1)
            use_ref = diff_ref < diff_orig
            if new_choice is not None:
                new_choice[mask_linear] = use_ref
        else:
            use_ref = choice_tensor[mask_linear]
        tgt_linear = torch.where(use_ref.unsqueeze(-1), reflected_linear, tgt_linear)
        target_adj[mask_linear] = tgt_linear

        if new_choice is not None:
            batch._rot_reflection_choice = new_choice.detach()

        return target_adj

    def _compute_reflection_choice(self, target: torch.Tensor, batch) -> Optional[torch.Tensor]:
        sym_labels = getattr(batch, "rot_symmetry", None)
        sym_axes = getattr(batch, "rot_symmetry_axis", None)
        if target is None or not torch.is_tensor(target) or not isinstance(sym_labels, list) or not sym_labels:
            return None
        if not torch.is_tensor(sym_axes):
            return None
        axes = sym_axes.to(device=target.device, dtype=target.dtype)
        axis_norm = torch.linalg.norm(axes, dim=-1, keepdim=True)
        valid_axis = axis_norm > 1.0e-8
        if not torch.any(valid_axis):
            return None
        mask_linear = torch.tensor(
            [lbl == "linear_flip" for lbl in sym_labels],
            device=target.device,
            dtype=torch.bool,
        )
        mask_linear = mask_linear & valid_axis.squeeze(-1)
        if not mask_linear.any():
            return None
        choice = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
        tol = 1.0e-10

        axes_linear = axes[mask_linear] / axis_norm[mask_linear]
        tgt_linear = target[mask_linear]
        ref_vecs = self._build_linear_reference_vectors(axes_linear)
        comp = (tgt_linear * ref_vecs).sum(dim=-1)
        use_ref = comp < -tol
        if use_ref.any():
            choice[mask_linear] = use_ref

        return choice

    def _build_linear_reference_vectors(self, axes: torch.Tensor) -> torch.Tensor:
        if axes.numel() == 0:
            return axes
        ref_vecs = torch.zeros_like(axes)
        base_options = [
            axes.new_tensor([1.0, 0.0, 0.0]),
            axes.new_tensor([0.0, 1.0, 0.0]),
            axes.new_tensor([0.0, 0.0, 1.0]),
        ]
        for idx in range(axes.shape[0]):
            axis = axes[idx]
            ref_vec = None
            for base in base_options:
                candidate = base - torch.dot(base, axis) * axis
                norm = torch.linalg.norm(candidate)
                if norm > 1.0e-6:
                    ref_vec = candidate / norm
                    break
            if ref_vec is None:
                ref_vec = axes.new_tensor([1.0, 0.0, 0.0])
            ref_vecs[idx] = ref_vec
        return ref_vecs

    def _make_flow_debug(
        self,
        batch,
        t,
        r,
        eps_tr,
        rot_vec,
        ads_center,
        clip_flags,
        rot_sched=None,
        rot_sched_deriv=None,
        tr_sched=None,
        tr_sched_deriv=None,
        symmetry_labels=None,
    ):
        def _to_list(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [v.detach().cpu().item() if torch.is_tensor(v) else v for v in value]
            return value

        ids = {}
        for key in ["sid", "fid", "eid", "data_id"]:
            if hasattr(batch, key):
                ids[key] = _to_list(getattr(batch, key))

        eps_tr_cpu = eps_tr.detach().cpu()
        rot_vec_cpu = rot_vec.detach().cpu()
        trans_norm = torch.linalg.norm(eps_tr_cpu[:, :2], dim=-1).tolist()
        rot_norm = torch.linalg.norm(rot_vec_cpu, dim=-1).tolist()

        debug = {
            "ids": ids,
            "time_samples": t.detach().cpu().tolist(),
            "radial_time": r.detach().cpu().tolist(),
            "translation": eps_tr_cpu.tolist(),
            "translation_norm": trans_norm,
            "rotation": rot_vec_cpu.tolist(),
            "rotation_norm": rot_norm,
            "ads_center": ads_center.detach().cpu().tolist(),
            "clip_flags": clip_flags,
        }
        if symmetry_labels is not None:
            debug["rotation_symmetry"] = list(symmetry_labels)
        if rot_sched is not None:
            debug["rotation_schedule"] = rot_sched.detach().cpu().tolist()
        if rot_sched_deriv is not None:
            debug["rotation_schedule_deriv"] = rot_sched_deriv.detach().cpu().tolist()
        if tr_sched is not None:
            debug["translation_schedule"] = tr_sched.detach().cpu().tolist()
        if tr_sched_deriv is not None:
            debug["translation_schedule_deriv"] = tr_sched_deriv.detach().cpu().tolist()
        return debug

    def _log_flow_debug(self, info, context):
        if not info:
            return
        logging.warning(
            f"[{context}] flow ids={info.get('ids')} clip={info.get('clip_flags')}"
        )
        logging.warning(
            f"[{context}] t={info.get('time_samples')} r={info.get('radial_time')}"
        )
        logging.warning(
            f"[{context}] translation_norm={info.get('translation_norm')} rotation_norm={info.get('rotation_norm')}"
        )
        logging.warning(f"[{context}] translation={info.get('translation')}")
        logging.warning(f"[{context}] rotation={info.get('rotation')}")

    def _log_flow_debug_from_batch(self, batch, context):
        if batch is None:
            return
        self._log_flow_debug(getattr(batch, "_flow_debug", None), context)

    def _log_active_flow_debug(self, context):
        self._log_flow_debug(self._active_batch_debug, context)

    def _set_active_batch_debug(self, info: Optional[Dict[str, Any]]) -> None:
        self._active_batch_debug = info
        self._sync_model_active_batch_debug()

    def _sync_model_active_batch_debug(self) -> None:
        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = self.model
        if base_model is not None:
            setattr(base_model, "_active_batch_debug", self._active_batch_debug)

    def _should_log_flow_samples(self) -> bool:
        if self.flow_debug_interval <= 0:
            return False
        self._flow_debug_tick += 1
        if self._flow_debug_tick % self.flow_debug_interval != 0:
            return False
        return True

    def _tensor_summary(self, tensor: torch.Tensor) -> str:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return "n/a"
        data = tensor.detach().float().reshape(-1)
        finite = torch.isfinite(data)
        if not finite.any():
            return "non-finite"
        data = data[finite]
        mean = float(data.mean())
        std = float(data.std(unbiased=False)) if data.numel() > 1 else 0.0
        dmin = float(data.min())
        dmax = float(data.max())
        return f"mean={mean:.4e} std={std:.4e} min={dmin:.4e} max={dmax:.4e}"

    def _log_flow_sample_stats(self, batch, out, context: str) -> None:
        if batch is None or out is None:
            return
        sample_ids: Dict[str, Any] = {}
        for key in ("sid", "fid", "eid", "data_id"):
            if hasattr(batch, key):
                value = getattr(batch, key)
                try:
                    sample_ids[key] = value.tolist() if hasattr(value, "tolist") else value
                except Exception:
                    sample_ids[key] = str(value)
        if sample_ids:
            logging.info("[flow-sample][%s] ids=%s", context, sample_ids)
        v_tr = out.get("v_tr")
        v_rot = out.get("v_rot")
        if v_tr is None or v_rot is None:
            logging.warning("[flow-sample][%s] Missing velocity outputs; skipping stats.", context)
            return
        tgt_tr = batch.v_tr_target[:, : v_tr.shape[-1]] if hasattr(batch, "v_tr_target") and v_tr is not None else None
        tgt_rot_raw = batch.v_rot_target if hasattr(batch, "v_rot_target") else None
        t_vals = batch.t if hasattr(batch, "t") else None

        rot_debug = out.get("_debug_rot_projection") if isinstance(out, dict) else None
        v_rot_raw = rot_debug.get("pred_before") if isinstance(rot_debug, dict) else None
        if not torch.is_tensor(v_rot_raw):
            v_rot_raw = v_rot
        tgt_rot_raw = rot_debug.get("target_before", tgt_rot_raw) if isinstance(rot_debug, dict) else tgt_rot_raw
        if torch.is_tensor(tgt_rot_raw) and tgt_rot_raw.shape[-1] > v_rot_raw.shape[-1]:
            tgt_rot_raw = tgt_rot_raw[:, : v_rot_raw.shape[-1]]
        tgt_rot_proj = rot_debug.get("target_after") if isinstance(rot_debug, dict) else None
        projector = getattr(batch, "rot_projector", None)
        if tgt_rot_proj is None and torch.is_tensor(projector) and torch.is_tensor(tgt_rot_raw):
            try:
                proj = projector.to(tgt_rot_raw.device)
                if proj.ndim == 3 and proj.shape[-2:] == (3, 3) and proj.shape[0] == tgt_rot_raw.shape[0]:
                    tgt_rot_proj = torch.matmul(proj, tgt_rot_raw.unsqueeze(-1)).squeeze(-1)
            except Exception:
                tgt_rot_proj = None
        rot_pred_proj = v_rot

        def _stat_compare(out_tensor, tgt_tensor):
            if not (torch.is_tensor(out_tensor) and torch.is_tensor(tgt_tensor)):
                return "n/a", "n/a"
            out_data = out_tensor.detach().float().reshape(-1)
            tgt_data = tgt_tensor.detach().float().reshape(-1)
            mask_out = torch.isfinite(out_data)
            mask_tgt = torch.isfinite(tgt_data)
            if not (mask_out.any() and mask_tgt.any()):
                return "n/a", "n/a"
            out_vals = out_data[mask_out]
            tgt_vals = tgt_data[mask_tgt]
            if out_vals.numel() == 0 or tgt_vals.numel() == 0:
                return "n/a", "n/a"
            out_std = float(out_vals.std(unbiased=False).item()) if out_vals.numel() > 1 else 0.0
            tgt_std = float(tgt_vals.std(unbiased=False).item()) if tgt_vals.numel() > 1 else 0.0
            if tgt_std > 0.0:
                std_ratio = f"{(out_std / tgt_std):.4e}"
            else:
                std_ratio = "n/a"
            mean_diff = f"{float(out_vals.mean().item() - tgt_vals.mean().item()):.4e}"
            return std_ratio, mean_diff

        tr_std_ratio, tr_mean_diff = _stat_compare(v_tr, tgt_tr)
        rot_std_ratio_raw, rot_mean_diff_raw = _stat_compare(v_rot_raw, tgt_rot_raw)
        rot_std_ratio_proj, rot_mean_diff_proj = _stat_compare(rot_pred_proj, tgt_rot_proj)

        logging.info(
            "[flow-sample][%s] t: %s | target_tr: %s | out_tr: %s",
            context,
            self._tensor_summary(t_vals),
            self._tensor_summary(tgt_tr),
            self._tensor_summary(v_tr),
        )
        logging.info(
            "[flow-sample][%s] rot_raw target=%s | out=%s",
            context,
            self._tensor_summary(tgt_rot_raw),
            self._tensor_summary(v_rot_raw),
        )
        logging.info(
            "[flow-sample][%s] rot_proj target=%s | out=%s",
            context,
            self._tensor_summary(tgt_rot_proj),
            self._tensor_summary(rot_pred_proj),
        )
        logging.info(
            "[flow-sample][%s] tr std_ratio=%s mean_diff=%s | rot_raw std_ratio=%s mean_diff=%s | rot_proj std_ratio=%s mean_diff=%s",
            context,
            tr_std_ratio,
            tr_mean_diff,
            rot_std_ratio_raw,
            rot_mean_diff_raw,
            rot_std_ratio_proj,
            rot_mean_diff_proj,
        )

        def _rot_alignment(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]):
            if not (torch.is_tensor(pred) and torch.is_tensor(target)):
                return "n/a", "n/a"
            v_flat = pred.detach().float()
            tgt_flat = target.detach().float()
            tgt_norm = torch.linalg.norm(tgt_flat, dim=-1)
            v_norm = torch.linalg.norm(v_flat, dim=-1)
            valid = tgt_norm > 1.0e-6
            if not valid.any():
                return "n/a", "n/a"
            dot = (v_flat[valid] * tgt_flat[valid]).sum(dim=-1)
            denom = v_norm[valid] * tgt_norm[valid] + 1.0e-8
            cos_mean = f"{float(torch.clamp(dot / denom, -1.0, 1.0).mean().item()):.4f}"
            norm_ratio = v_norm[valid] / (tgt_norm[valid] + 1.0e-8)
            return cos_mean, f"{float(norm_ratio.mean().item()):.4f}"

        rot_cosine_raw, rot_norm_ratio_raw = _rot_alignment(v_rot_raw, tgt_rot_raw)
        rot_cosine_proj, rot_norm_ratio_proj = _rot_alignment(rot_pred_proj, tgt_rot_proj)

        logging.info(
            "[flow-sample][%s] rot_raw cos_mean=%s norm_ratio=%s | rot_proj cos_mean=%s norm_ratio=%s",
            context,
            rot_cosine_raw,
            rot_norm_ratio_raw,
            rot_cosine_proj,
            rot_norm_ratio_proj,
        )

        recent_losses = getattr(self, "_last_component_losses", None)
        if isinstance(recent_losses, dict):
            logging.info(
                "[flow-sample][%s] loss_tr=%.4e | loss_rot=%.4e | valid=%d",
                context,
                recent_losses.get("translation", float("nan")),
                recent_losses.get("rotation", float("nan")),
                int(recent_losses.get("valid_samples", 0)),
            )

        def _preview(name: str, pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> None:
            if pred is None or target is None:
                logging.info("[flow-sample][%s] %s preview: n/a", context, name)
                return
            try:
                pred_cpu = pred.detach().float().cpu()
                tgt_cpu = target.detach().float().cpu()
            except Exception:
                logging.info("[flow-sample][%s] %s preview: detach failed", context, name)
                return
            if pred_cpu.ndim == 1:
                pred_cpu = pred_cpu.unsqueeze(-1)
            if tgt_cpu.ndim == 1:
                tgt_cpu = tgt_cpu.unsqueeze(-1)
            rows = min(4, pred_cpu.shape[0])
            pred_snippet = pred_cpu[:rows]
            tgt_snippet = tgt_cpu[:rows]
            pred_fmt = ["[" + ", ".join(f"{val:.3e}" for val in row.tolist()) + "]" for row in pred_snippet]
            tgt_fmt = ["[" + ", ".join(f"{val:.3e}" for val in row.tolist()) + "]" for row in tgt_snippet]
            logging.info(
                "[flow-sample][%s] %s pred: %s",
                context,
                name,
                " | ".join(pred_fmt),
            )
            logging.info(
                "[flow-sample][%s] %s label: %s",
                context,
                name,
                " | ".join(tgt_fmt),
            )

        _preview("translation", v_tr, tgt_tr)
        _preview("rotation-raw", v_rot_raw, tgt_rot_raw)
        _preview("rotation-proj", rot_pred_proj, tgt_rot_proj)

        sym_labels = getattr(batch, "rot_symmetry", None)
        if isinstance(sym_labels, list) and sym_labels:
            counts = Counter(sym_labels)
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            logging.info("[flow-sample][%s] rot_symmetry_counts: %s", context, summary)
            eigvals = getattr(batch, "rot_symmetry_eigvals", None)
            if eigvals is not None:
                if torch.is_tensor(eigvals):
                    eig_source = eigvals.detach().cpu()
                elif isinstance(eigvals, (list, tuple)) and eigvals and torch.is_tensor(eigvals[0]):
                    eig_source = torch.stack([e.detach().cpu() for e in eigvals], dim=0)
                else:
                    eig_source = torch.as_tensor(eigvals)
                    eig_source = eig_source.detach().cpu() if torch.is_tensor(eig_source) else None
                if eig_source is not None and eig_source.shape[0] == len(sym_labels):
                    grouped = defaultdict(list)
                    for label, vec in zip(sym_labels, eig_source):
                        grouped[label].append(vec)
                    formatted = []
                    for label in sorted(grouped):
                        vec_strs = [
                            "[" + ", ".join(f"{float(val):.4e}" for val in vec.tolist()) + "]"
                            for vec in grouped[label]
                        ]
                        formatted.append(f"{label}: {', '.join(vec_strs)}")
                    if formatted:
                        logging.info(
                            "[flow-sample][%s] rot_symmetry_eigvals(norm): %s",
                            context,
                            " | ".join(formatted),
                        )

        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = getattr(self.model, "model", self.model)
        energy_null = getattr(base_model, "energy_null", None)
        if torch.is_tensor(energy_null):
            with torch.no_grad():
                vals = energy_null.detach().float().reshape(-1)
                finite = torch.isfinite(vals)
                if finite.any():
                    eff = vals[finite]
                    norm = float(eff.norm().item())
                    mean = float(eff.mean().item())
                    std = float(eff.std(unbiased=False).item()) if eff.numel() > 1 else 0.0
                    emin = float(eff.min().item())
                    emax = float(eff.max().item())
                    logging.info(
                        "[flow-sample][%s] energy_null norm=%.4e mean=%.4e std=%.4e min=%.4e max=%.4e",
                        context,
                        norm,
                        mean,
                        std,
                        emin,
                        emax,
                    )
                else:
                    logging.info("[flow-sample][%s] energy_null has no finite entries", context)

    def _log_optimizer_stats(self, split: str) -> None:
        if self.logger is None or self.optimizer_log_interval <= 0:
            return
        if self.step % self.optimizer_log_interval != 0:
            return
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        stats: Dict[str, float] = {}
        for idx, group in enumerate(optimizer.param_groups):
            prefix = f"optimizer/group{idx}"
            lr = group.get("lr", None)
            if lr is not None:
                stats[f"{prefix}_lr"] = float(lr)
            wd = group.get("weight_decay", None)
            if wd is not None:
                stats[f"{prefix}_weight_decay"] = float(wd)
            eps = group.get("eps", None)
            if eps is not None:
                stats[f"{prefix}_eps"] = float(eps)
            momentum = group.get("momentum", None)
            if momentum is not None:
                stats[f"{prefix}_momentum"] = float(momentum)
            betas = group.get("betas", None)
            if betas is not None and len(betas) == 2:
                stats[f"{prefix}_beta1"] = float(betas[0])
                stats[f"{prefix}_beta2"] = float(betas[1])
        steps = []
        for state in optimizer.state.values():
            if isinstance(state, dict) and "step" in state and state["step"] is not None:
                try:
                    steps.append(int(state["step"]))
                except Exception:
                    continue
        if steps:
            stats["optimizer/state_step_max"] = float(max(steps))
            stats["optimizer/state_step_mean"] = float(sum(steps) / len(steps))
        if stats:
            self.logger.log(stats, step=self.step, split=split)

    # ---------- C) loss ----------
    def _compute_loss(self, out, batch):
        v_tr = out.get("v_tr")
        v_rot = out.get("v_rot")
        if v_tr is None or v_rot is None:
            raise KeyError("Flow matching output must include 'v_tr' and 'v_rot' when regressing velocity.")
        target_tr = batch.v_tr_target[:, : v_tr.shape[-1]].to(v_tr)
        target_rot = batch.v_rot_target.to(v_rot).clone()
        rot_proj = getattr(batch, "rot_projector", None)
        if rot_proj is not None:
            if rot_proj.device != v_rot.device:
                rot_proj = rot_proj.to(v_rot.device)
            target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
            v_rot = torch.matmul(rot_proj, v_rot.unsqueeze(-1)).squeeze(-1)
        target_rot = self._align_discrete_symmetry_targets(v_rot, target_rot, batch)
        diff_tr = v_tr - target_tr
        diff_rot = v_rot - target_rot

        combined_clip_mask = getattr(self, "_combined_clip_mask", None)
        if torch.is_tensor(combined_clip_mask):
            combined_mask = combined_clip_mask.to(v_tr.device)
            if combined_mask.shape[0] != v_tr.shape[0]:
                min_len = min(combined_mask.shape[0], v_tr.shape[0])

                combined_mask = combined_mask[:min_len]
                if min_len < v_tr.shape[0]:
                    pad = torch.zeros(v_tr.shape[0] - min_len, dtype=torch.bool, device=v_tr.device)
                    combined_mask = torch.cat([combined_mask, pad], dim=0)
            valid_mask = ~combined_mask.bool()
        else:
            valid_mask = torch.ones(v_tr.shape[0], dtype=torch.bool, device=v_tr.device)

        diff_tr_valid = diff_tr[valid_mask]
        diff_rot_valid = diff_rot[valid_mask]

        # --- Compute Unweighted Loss for Monitoring ---
        with torch.no_grad():
            if diff_tr_valid.numel() > 0:
                loss_tr_un = torch.mean(diff_tr_valid.pow(2))
            else:
                loss_tr_un = torch.tensor(0.0, device=v_tr.device)
            if diff_rot_valid.numel() > 0:
                loss_rot_un = torch.mean(diff_rot_valid.pow(2))
            else:
                loss_rot_un = torch.tensor(0.0, device=v_tr.device)
            loss_unweighted_val = float((loss_tr_un + loss_rot_un).detach().cpu())
        # ----------------------------------------------

        if self.fm_endpoint_weight_exponent != 0.0:
            t_vals = batch.t.squeeze(-1).to(v_tr)

            if self.fm_endpoint_weight_exponent > 0.0:
                # Emphasize t=1 (Noise) -> (1-t)^(-alpha)
                weight = torch.clamp(1.0 - t_vals, min=1.0e-3)
                weight = weight.pow(-self.fm_endpoint_weight_exponent)
            else:


                weight = torch.clamp(t_vals, min=1.0e-3)
                weight = weight.pow(self.fm_endpoint_weight_exponent) # exponent is negative here

            weight = weight[valid_mask]
            if weight.numel() == 0:
                loss_tr = diff_tr_valid.new_zeros(())
                loss_rot = diff_rot_valid.new_zeros(())
            else:
                loss_tr = torch.mean(diff_tr_valid.pow(2) * weight.unsqueeze(-1))
                loss_rot = torch.mean(diff_rot_valid.pow(2) * weight.unsqueeze(-1))
        else:
            if diff_tr_valid.numel() == 0:
                loss_tr = diff_tr_valid.new_zeros(())
            else:
                loss_tr = torch.mean(diff_tr_valid.pow(2))
            if diff_rot_valid.numel() == 0:
                loss_rot = diff_rot_valid.new_zeros(())
            else:
                loss_rot = torch.mean(diff_rot_valid.pow(2))
        try:
            loss_tr_val = float(loss_tr.detach().cpu())
        except Exception:
            loss_tr_val = float("nan")
        try:
            loss_rot_val = float(loss_rot.detach().cpu())
        except Exception:
            loss_rot_val = float("nan")
        valid_samples = int(valid_mask.sum().item()) if torch.is_tensor(valid_mask) else 0
        self._last_component_losses = {
            "translation": loss_tr_val,
            "rotation": loss_rot_val,
            "valid_samples": valid_samples,
            "loss_unweighted": loss_unweighted_val,
        }
        return loss_tr + self.rot_loss_weight * loss_rot

    # ---------- D) validate ----------
    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)
        if distutils.is_master():
            logging.info(f"[MeanFlowTrainer] Evaluating on {split}.")
        self.model.eval()
        if self.ema:
            self.ema.store(); self.ema.copy_to()
        metrics: Dict = {}
        evaluator = Evaluator(task=self.name)
        rank = distutils.get_rank()
        loader = self.val_loader if split == "val" else self.test_loader

        # Grid Search Setup
        grid_cfgs = [1.0]
        grid_steps = [10]
        grid_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})
        checkpoint_every_epoch = self.config["optim"].get("checkpoint_every_epoch", 1)

        do_grid_search = True # (int(self.epoch) > 0 and int(self.epoch) % checkpoint_every_epoch == 0)

        # Limit grid search to ~ 40 samples to save time
        grid_search_samples = 0
        grid_search_limit = 50

        for i, batch in tqdm(enumerate(loader), total=len(loader), position=rank, desc=f"device {rank}", disable=disable_tqdm):
            if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
                batch.pos = batch.pos_relaxed
            base_pos = self._clone_base_positions(batch)
            batch = self._build_interpolant(batch)
            self._set_active_batch_debug(getattr(batch, "_flow_debug", None))
            out = None
            loss = None
            for attempt in range(self.nan_retry_limit + 1):
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                if torch.isfinite(loss):
                    break
                self._log_nonfinite_debug(loss, out, batch, attempt, "val")
                logging.warning(
                    f"Non-finite loss detected during validation (attempt {attempt + 1}/{self.nan_retry_limit + 1}), resampling interpolant"
                )
                if attempt == self.nan_retry_limit:
                    self._log_nonfinite_debug(loss, out, batch, attempt, "val-final")
                    out = None
                    break
                batch = self._resample_interpolant(batch, base_pos)
                self._set_active_batch_debug(getattr(batch, "_flow_debug", None))
            if out is None or loss is None or not torch.isfinite(loss):
                self._log_nonfinite_debug(loss, out, batch, attempt, "val-skip")
                logging.warning("Skipping batch in validation due to non-finite loss")
                self._set_active_batch_debug(None)
                continue
            if self._should_log_flow_samples():
                self._log_flow_sample_stats(batch, out, "val")

            # --- Grid Search Inference (Subset) ---
            if do_grid_search and grid_search_samples < grid_search_limit:
                try:
                    # Prepare inference batch (clone once)
                    inf_batch_base = batch.clone()
                    if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
                        inf_batch_base.pos_relaxed = batch.pos_relaxed
                    else:
                        inf_batch_base.pos_relaxed = base_pos # Fallback to what we cloned earlier

                    ads_mask = (batch.tags == 2)
                    if ads_mask.any():
                        # Update sample count
                        current_bs = int(batch.natoms.size(0))
                        grid_search_samples += current_bs

                        for cfg in grid_cfgs:
                            for steps in grid_steps:
                                # Config for this run
                                val_flow_opt = {
                                    "tr_sigma": self.tr_sigma,
                                    "tr_sigma_z_scale": self.tr_sigma_z_scale,
                                    "rot_sigma": self.rot_sigma,
                                    "num_steps": steps,
                                    "cfg_scale": cfg,
                                    "time_sampl": self.time_sampl,
                                    "allow_z": self.allow_z,
                                }

                                # Run inference
                                # We need to be careful not to modify inf_batch_base in place permanently
                                # FlowTorch modifies batch.pos.
                                inf_batch = inf_batch_base.clone()

                                sampler = FlowTorch(
                                    batch=inf_batch,
                                    model=self.model,
                                    flow_opt=val_flow_opt,
                                    device=self.device,
                                    save_full_traj=False,
                                    traj_dir=None
                                )
                                final_batch = sampler.run()

                                # Calc Error
                                gen_pos = final_batch.pos
                                ref_pos = inf_batch.pos_relaxed
                                diff = gen_pos[ads_mask] - ref_pos[ads_mask]

                                # Wrap diff using PBC (XY only)
                                diff_wrapped = torch.zeros_like(diff)
                                ads_batch = inf_batch.batch[ads_mask]
                                cell = inf_batch.cell.double()
                                B_size = int(inf_batch.natoms.size(0))

                                for b in range(B_size):
                                    mask = (ads_batch == b)
                                    if not mask.any(): continue
                                    frac = torch.linalg.solve(cell[b].t(), diff[mask].double().t()).t()
                                    frac[..., :2] = ((frac[..., :2] + 0.5) % 1.0) - 0.5
                                    cart = (frac.float() @ cell[b].float())
                                    cart[..., 2] = diff[mask][..., 2]
                                    diff_wrapped[mask] = cart

                                dist = torch.norm(diff_wrapped, dim=-1)

                                key = f"cfg{cfg}_step{steps}"
                                grid_stats[key]["sum"] += dist.sum().item()
                                grid_stats[key]["count"] += dist.numel()
                except Exception as e:
                    logging.warning(f"[Val-Inference] Failed to run grid search inference: {e}")
            # ---------------------------------------------------------------

            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)
            if self._last_component_losses:
                metrics = evaluator.update("loss_unweighted", self._last_component_losses.get("loss_unweighted", 0.0), metrics)
            self._set_active_batch_debug(None)

            # if do_grid_search and grid_search_samples >= grid_search_limit:
            #     break

        # Aggregate Grid Search Results
        if do_grid_search:
            final_grid_results = {}
            for key, stats in grid_stats.items():
                total_sum = distutils.all_reduce(stats["sum"], average=False, device=self.device)
                total_count = distutils.all_reduce(stats["count"], average=False, device=self.device)
                if total_count > 0:
                    final_grid_results[key] = float(total_sum / total_count)

            if distutils.is_master():
                json_path = os.path.join(self.config["cmd"]["results_dir"], f"val_grid_epoch{self.epoch}.json")
                try:
                    with open(json_path, "w") as f:
                        json.dump(final_grid_results, f, indent=2)
                    logging.info(f"Saved grid search results to {json_path}")
                except Exception as e:
                    logging.warning(f"Failed to save grid search json: {e}")

        aggregated = {}
        for k in metrics:
            aggregated[k] = {
                "total": distutils.all_reduce(metrics[k]["total"], average=False, device=self.device),
                "numel": distutils.all_reduce(metrics[k]["numel"], average=False, device=self.device),
            }
            aggregated[k]["metric"] = aggregated[k]["total"] / aggregated[k]["numel"]
        metrics = aggregated

        # Inject representative pos_mae into metrics for filename generation
        if do_grid_search:
            rep_key = "cfg1.0_step10"
            if rep_key in final_grid_results:
                # Manually inject the metric structure expected by save_epoch_checkpoint
                metrics["pos_mae"] = {"metric": final_grid_results[rep_key]}

        log_dict = {k: metrics[k]["metric"] for k in metrics}; log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            logging.info(", ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()]))
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step, split=split)
        if self.ema: self.ema.restore()

        self._last_val_metrics = metrics
        self._last_val_step = self.step
        return metrics

    # ---------- E) train ----------
    def train(self, disable_eval_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)
        if distutils.is_master():
            logging.info(f"[MeanFlowTrainer] Training ")
        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get("primary_metric", self.evaluator.task_primary_metric[self.name])
        if (not hasattr(self, "primary_metric") or self.primary_metric != primary_metric):
            self.best_val_metric = 1e9
        else:
            primary_metric = self.primary_metric
        self.metrics = {}
        start_epoch = self.step // len(self.train_loader)
        nan_count = 0
        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            # Update Loss Weighting Schedule
            if self.fm_endpoint_weight_start != self.fm_endpoint_weight_target:
                progress = min(1.0, epoch_int / self.config["optim"]["max_epochs"])
                self.fm_endpoint_weight_exponent = self.fm_endpoint_weight_start + progress * (self.fm_endpoint_weight_target - self.fm_endpoint_weight_start)
                if distutils.is_master():
                    logging.info(f"[Epoch {epoch_int}] Updating endpoint weight exponent to {self.fm_endpoint_weight_exponent:.4f}")

            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_iter = iter(self.train_loader)
            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step  = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                batch = next(train_iter)
                if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
                    batch.pos = batch.pos_relaxed
                base_pos = self._clone_base_positions(batch)
                batch = self._build_interpolant(batch)
                self._set_active_batch_debug(getattr(batch, "_flow_debug", None))
                out = None
                loss = None
                for attempt in range(self.nan_retry_limit + 1):
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)
                    if torch.isfinite(loss):
                        break
                    self._log_nonfinite_debug(loss, out, batch, attempt, "train")
                    logging.warning(
                        f"Non-finite loss detected (attempt {attempt + 1}/{self.nan_retry_limit + 1}), resampling interpolant"
                    )
                    if attempt == self.nan_retry_limit:
                        self._log_nonfinite_debug(loss, out, batch, attempt, "train-final")
                        out = None
                        break
                    batch = self._resample_interpolant(batch, base_pos)
                    self._set_active_batch_debug(getattr(batch, "_flow_debug", None))
                if out is None or loss is None or not torch.isfinite(loss):
                    self._log_nonfinite_debug(loss, out, batch, attempt, "train-skip")
                    logging.warning("NaN loss detected, skipping step"); nan_count += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            st = self.optimizer.state.get(p, None)
                            if not st:
                                continue
                            if "exp_avg" in st:
                                st["exp_avg"].zero_()
                            if "exp_avg_sq" in st:
                                st["exp_avg_sq"].zero_()
                    self._set_active_batch_debug(None)
                    if nan_count > 10:
                        logging.warning("Too many NaN losses, stopping training"); break
                    continue
                else:
                    nan_count = 0
                max_loss = 1e6

                lv = float(loss.detach().cpu())
                if not np.isfinite(lv) or lv > max_loss:

                    logging.warning(f"[guard] skip step={self.step} due to loss={lv:.3e}")
                    self._log_flow_debug_from_batch(batch, "train-loss-guard")
                    self.optimizer.zero_grad(set_to_none=True)
                    # （可选但强烈推荐）重置异常参数的动量/二阶矩，避免“常驻高档”
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            st = self.optimizer.state.get(p, None)
                            if not st:
                                continue
                            if "exp_avg" in st:
                                st["exp_avg"].zero_()
                            if "exp_avg_sq" in st:
                                st["exp_avg_sq"].zero_()
                    self._set_active_batch_debug(None)
                    self._last_output_was_clipped = False
                    continue
                if self._last_output_was_clipped:
                    logging.warning(f"[guard] output clipping detected; skipping step={self.step}")
                    self._log_flow_debug_from_batch(batch, "train-output-clip")
                    self.optimizer.zero_grad(set_to_none=True)
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            st = self.optimizer.state.get(p, None)
                            if not st:
                                continue
                            if "exp_avg" in st:
                                st["exp_avg"].zero_()
                            if "exp_avg_sq" in st:
                                st["exp_avg_sq"].zero_()
                    self._set_active_batch_debug(None)
                    self._last_output_was_clipped = False
                    continue
                raw_loss = loss
                loss_for_backward = self.scaler.scale(raw_loss) if self.scaler else raw_loss
                self._set_active_batch_debug(getattr(batch, "_flow_debug", None))
                did_step = self._backward(loss_for_backward)
                self._set_active_batch_debug(None)
                if not did_step:
                    continue
                loss_value = float(raw_loss.detach().cpu())
                if self._should_log_flow_samples():
                    self._log_flow_sample_stats(batch, out, "train")
                self.metrics = self._compute_metrics(out, batch, self.evaluator, self.metrics)
                self.metrics = self.evaluator.update("loss", loss_value, self.metrics)
                if self._last_component_losses:
                    self.metrics = self.evaluator.update("loss_unweighted", self._last_component_losses.get("loss_unweighted", 0.0), self.metrics)
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update({"lr": self.scheduler.get_lr(), "epoch": self.epoch, "step": self.step})
                if (self.step % self.config["cmd"]["print_every"] == 0 or i == 0 or i == (len(self.train_loader) - 1)) and distutils.is_master():
                    logging.info(", ".join([f"{k}: {v:.2e}" for k, v in log_dict.items()]))
                if self.logger is not None:
                    self.logger.log(log_dict, step=self.step, split="train")
                self._log_optimizer_stats("train")
                if (checkpoint_every != -1) and (self.step % checkpoint_every == 0):
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)
                if self.step % eval_every == 0 or i == (len(self.train_loader) - 1):
                    if self.val_loader is not None:
                        val_metrics = self.validate(split="val", disable_tqdm=disable_eval_tqdm)
                        self.update_best(primary_metric, val_metrics, disable_eval_tqdm=disable_eval_tqdm)
                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        if self.step % eval_every == 0:
                            self.scheduler.step(metrics=val_metrics[primary_metric]["metric"])
                    else:
                        self.scheduler.step()
            torch.cuda.empty_cache()

            # Save epoch checkpoint
            checkpoint_every_epoch = self.config["optim"].get("checkpoint_every_epoch", 1)
            if (epoch_int + 1) % checkpoint_every_epoch == 0:
                self.save_epoch_checkpoint(
                    epoch_index=epoch_int,
                    disable_eval_tqdm=disable_eval_tqdm,
                )
            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.train_dataset.close_db()
        if self.config.get("val_dataset", False): self.val_dataset.close_db()
        if self.config.get("test_dataset", False): self.test_dataset.close_db()

    def save_epoch_checkpoint(
        self,
        epoch_index: int,
        disable_eval_tqdm: bool = True,
    ) -> None:
        """Persist one checkpoint per epoch with val loss and pos_mae encoded in filename."""
        metrics = self._last_val_metrics
        if (
            self.val_loader is not None
            and (metrics is None or self._last_val_step != self.step)
        ):
            metrics = self.validate(
                split="val", disable_tqdm=disable_eval_tqdm
            )

        # Prefer unweighted validation loss if available
        val_loss = None
        loss_label = "valloss"
        if metrics:
            if "loss_unweighted" in metrics:
                val_loss = metrics["loss_unweighted"].get("metric")
                loss_label = "unweightedvalloss"
            elif "loss" in metrics:
                val_loss = metrics["loss"].get("metric")
                loss_label = "valloss"

        if val_loss is None or not np.isfinite(val_loss):
            loss_tag = "noval" if val_loss is None else "invalid"
        else:
            loss_tag = f"{val_loss:.4f}"

        # Add pos_mae to filename if available
        pos_mae = None
        if metrics and "pos_mae" in metrics:
            pos_mae = metrics["pos_mae"].get("metric")

        if pos_mae is not None and np.isfinite(pos_mae):
            mae_tag = f"_posmae{pos_mae:.4f}"
        else:
            mae_tag = ""

        checkpoint_name = (
            f"epoch{epoch_index + 1:04d}_{loss_label}{loss_tag}{mae_tag}.pt"
        )
        self.save(
            metrics=metrics,
            checkpoint_file=checkpoint_name,
            training_state=False,
        )

    def _backward(self, loss) -> bool:
        self.optimizer.zero_grad()
        loss.backward()
        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                elif not hasattr(self, "warned_shared_param_no_grad"):
                    self.warned_shared_param_no_grad = True
                    logging.warning(
                        "Some shared parameters do not have a gradient. Please verify shared parameter configuration."
                    )

        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        rot_grad_norm = self._compute_rotation_head_grad_norm()
        grad_clip_val = self.clip_grad_norm if self.clip_grad_norm else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=grad_clip_val,
        )

        guard_triggered = False
        guard_reason = ""
        grad_norm_val = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        if not np.isfinite(grad_norm_val):
            guard_triggered = True
            guard_reason = f"non-finite grad_norm={grad_norm_val}"
        elif self.grad_norm_guard > 0.0 and grad_norm_val > self.grad_norm_guard:
            guard_triggered = True
            guard_reason = f"grad_norm={grad_norm_val:.2f} > guard={self.grad_norm_guard:.2f}"

        if guard_triggered:
            logging.warning(f"[guard] {guard_reason}; skipping optimizer step at step={self.step}")
            self._log_active_flow_debug("train-grad-guard")
            self.optimizer.zero_grad(set_to_none=True)
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    st = self.optimizer.state.get(p, None)
                    if not st:
                        continue
                    if "exp_avg" in st:
                        st["exp_avg"].zero_()
                    if "exp_avg_sq" in st:
                        st["exp_avg_sq"].zero_()
            return False

        if self.logger is not None:
            log_payload = {"grad_norm": grad_norm}
            if rot_grad_norm is not None:
                log_payload["grad_norm_rot_head"] = rot_grad_norm
            self.logger.log(log_payload, step=self.step, split="train")

        if (
            rot_grad_norm is not None
            and np.isfinite(rot_grad_norm)
            and self.flow_monitor_interval > 0
            and self.step % self.flow_monitor_interval == 0
        ):
            logging.info(
                "[grad-monitor] step=%d grad_norm=%.3e rot_grad=%.3e",
                self.step,
                float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                rot_grad_norm,
            )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()
        return True

    # ---------- F) metrics ----------
    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        if "v_tr" not in out:
            raise KeyError("Model output must include 'v_tr' for Flow Matching metrics.")
        target = {
            "natoms": torch.ones(out["v_tr"].shape[0], dtype=torch.long, device=self.device),
            "cell": batch.cell,
            "pbc": torch.tensor([True, True, True], device=self.device),
        }
        target["positions"] = batch.v_tr_target.to(self.device)
        out_pack = {
            "natoms": target["natoms"],
            "cell": target["cell"],
            "pbc": target["pbc"],
            "positions": out["v_tr"].to(self.device)
        }
        if out_pack["positions"].size(-1) == 2:
            out_pack["positions"] = torch.cat(
                [
                    out_pack["positions"],
                    torch.zeros(
                        out_pack["positions"].size(0),
                        1,
                        device=out_pack["positions"].device,
                        dtype=out_pack["positions"].dtype,
                    ),
                ],
                dim=-1,
            )
        metrics = evaluator.eval(out_pack, target, prev_metrics=metrics)
        return metrics

    # ---------- G) flow relaxations (sampling) ----------
    @torch.no_grad()
    def run_relaxations(self, split: str = "val") -> None:
        """Flow-based sampler: writes .traj frames; optional IS2RS/IS2RE if labels exist."""
        from adsorbdiff.utils.utils import check_traj_files
        from adsorbdiff.relaxation.ml_relaxation import ml_flow

        ensure_fitted(self._unwrapped_model)
        self.model.eval()
        if self.ema:
            self.ema.store(); self.ema.copy_to()

        # options
        flow_opt = self.config["task"].get("relax_opt", {})
        num_batches = int(self.config["task"].get("num_relaxation_batches", 10**12))
        traj_dir = flow_opt.get("traj_dir", None)
        assert traj_dir is not None, "task.relax_opt.traj_dir must be specified"
        os.makedirs(traj_dir, exist_ok=True)

        # detect whether we can score IS2RS/IS2RE
        has_posE = (hasattr(self.relax_dataset[0], "pos_relaxed") and self.relax_dataset[0].pos_relaxed is not None) \
                   and (hasattr(self.relax_dataset[0], "y_relaxed") and self.relax_dataset[0].y_relaxed is not None)
        split_eval = "val" if has_posE else "test"

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        for i, batch in tqdm(enumerate(self.relax_loader), total=len(self.relax_loader)):
            if i >= num_batches: break
            if check_traj_files(batch, traj_dir):
                try:
                    logging.info(f"[FM] Skip existing batch: {batch.sid.tolist()}")
                except Exception:
                    logging.info("[FM] Skip existing batch")
                continue
            relaxed_batch = ml_flow(
                batch=batch,
                model=self,  # 外层对象，内部有 .model 和 .model.sampling
                flow_opt=flow_opt,
                traj_dir=traj_dir,
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                logger=self.logger,
            )

            # write_pos npz aggregation (optional)
            if self.config["task"].get("write_pos", False):
                rank = distutils.get_rank()
                ids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]
                chunk_idx = natoms
                pos_filename = os.path.join(self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz")
                np.savez_compressed(pos_filename, ids=ids, pos=np.array(batch_relaxed_positions, dtype=object), chunk_idx=chunk_idx)

            # metrics if labels are present
            if split_eval == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for n in relaxed_batch.natoms:
                    natoms_free.append(torch.sum(mask[s_idx: s_idx + n]).item())
                    s_idx += n
                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }
                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }
                metrics_is2rs = evaluator_is2rs.eval(prediction, target, metrics_is2rs)
                metrics_is2re = evaluator_is2re.eval({"energy": prediction["energy"]}, {"energy": target["energy"]}, metrics_is2re)

        # merge npz shards on master
        if self.config["task"].get("write_pos", False):
            distutils.synchronize()
            if distutils.is_master():
                gather = defaultdict(list)
                full_path = os.path.join(self.config["cmd"]["results_dir"], "relaxed_positions.npz")
                for r in range(distutils.get_world_size()):
                    rp = os.path.join(self.config["cmd"]["results_dir"], f"relaxed_pos_{r}.npz")
                    rank_results = np.load(rp, allow_pickle=True)
                    gather["ids"].extend(rank_results["ids"])
                    gather["pos"].extend(rank_results["pos"])
                    gather["chunk_idx"].extend(rank_results["chunk_idx"])
                    os.remove(rp)
                _, idx = np.unique(gather["ids"], return_index=True)
                gather["ids"] = np.array(gather["ids"])[idx]
                gather["pos"] = np.concatenate(np.array(gather["pos"])[idx])
                gather["chunk_idx"] = np.cumsum(np.array(gather["chunk_idx"])[idx])[:-1]
                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather)

        # log metrics
        if split_eval == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated = {}
                for k in metrics:
                    aggregated[k] = {
                        "total": distutils.all_reduce(metrics[k]["total"], average=False, device=self.device),
                        "numel": distutils.all_reduce(metrics[k]["numel"], average=False, device=self.device),
                    }
                    aggregated[k]["metric"] = aggregated[k]["total"] / aggregated[k]["numel"]
                log_dict = {f"{task}_{k}": aggregated[k]["metric"] for k in aggregated}
                if self.logger is not None:
                    self.logger.log(log_dict, step=self.step, split=split)
                if distutils.is_master():
                    logging.info(log_dict)

        if self.ema: self.ema.restore()
