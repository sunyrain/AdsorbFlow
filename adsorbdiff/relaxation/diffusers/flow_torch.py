# adsorbdiff/relaxation/flow_torch.py
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import math
import ase
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_scatter import scatter

from adsorbdiff.relaxation.ase_utils import batch_to_atoms
from adsorbdiff.utils.rot_utils import axis_angle_to_matrix, sample_vec


@torch.no_grad()
def _pbc_wrap_xy(vec_xyz: torch.Tensor, batch) -> torch.Tensor:
    """
    Wrap **COM/anchor displacement** to nearest PBC image on XY only.
    Only supports per-sample tensors of shape (B, 3), where B is batch size.

    NOTE:
    - Do NOT apply PBC wrapping per-atom for a rigid adsorbate.
      Per-atom wrapping can split a molecule crossing cell boundaries.
    (等价于 MeanFlowTrainer 里的实现，这里内联一份方便独立使用)
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
        # IMPORTANT:
        # We only want to wrap the displacement in the a/b (XY) directions.
        # For non-orthogonal cells where c has x/y components, leaving frac[2]
        # would leak the z displacement into x/y when reconstructing cart.
        frac[2] = 0.0
        frac[:2] = ((frac[:2] + 0.5) % 1.0) - 0.5
        cart = (frac.float() @ batch.cell[b].float())
        cart[2] = vec_xyz[b, 2]
        out[b] = cart
    return out


class FlowTorch:
    """
    Flow-matching sampler with classifier-free guidance (CFG) for adsorbate-only rigid DOFs.
    - 平移：作用在吸附体质心（XY），Z 默认不动（可通过 allow_z_tr 控制）。
    - 旋转：轴角表示（对吸附体相对参考 COM 的坐标做刚体旋转）。
    """

    def __init__(
        self,
        batch: Batch,
        model,                                  # 需提供 .model(cond, mode="fm") 的接口（和 Trainer 一致）
        flow_opt: Dict,
        device: str = "cuda:0",
        save_full_traj: bool = True,
        traj_dir: Optional[Path] = None,
        traj_names=None,
        logger=None,
    ):
        self.batch = batch
        self.model = model
        self.device = device
        self.flow_opt = flow_opt or {}
        self.save_full = save_full_traj
        self.traj_dir = Path(traj_dir) if traj_dir else None
        self.traj_names = traj_names
        self.logger = logger
        self.collected_steps = None  # Optional list of per-step positions for debugging
        self.collected_step_ids = None  # Optional list[int] aligned with collected_steps

        if self.traj_dir:
            assert traj_names is not None and len(traj_names), \
                "Trajectory names should be specified to save trajectories"
            self.traj_dir.mkdir(exist_ok=True, parents=True)

    def run(self) -> Batch:
        """
        Integrate Flow Matching with CFG from t=1→0. Writes .traj if traj_dir provided.
        """
        torch.cuda.empty_cache()
        batch = self.batch.to(self.device)
        B = int(batch.natoms.size(0))
        store_steps = bool(self.flow_opt.get("store_steps", False))
        store_every = int(self.flow_opt.get("store_every", 1))

        # --- anchors / masks ---
        tags = batch.tags
        ads_mask = (tags == 2)
        bidx = batch.batch
        # 参考位置（若有 pos_relaxed 就用它）
        pos_ref = batch.pos_relaxed if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None else batch.pos
        cell = batch.cell

        # --- IMPORTANT: make adsorbate contiguous under PBC before computing COM/template ---
        # If the dataset stores wrapped coordinates, a molecule crossing the unit cell can look
        # "split". Using that split geometry as the rigid template makes step-0/step-1 look like
        # dissociation and yields huge MAE. We unwrap within each sample using minimum-image.
        def _make_contiguous_ads(pos_all: torch.Tensor) -> torch.Tensor:
            if pos_all.numel() == 0:
                return pos_all
            out = pos_all.clone()
            B_local = int(batch.natoms.size(0))
            for bb in range(B_local):
                mask_bb = (bidx == bb) & ads_mask
                if not mask_bb.any():
                    continue
                idx_bb = torch.where(mask_bb)[0]
                ref = out[idx_bb[0]]
                diffs = out[idx_bb] - ref
                frac = torch.linalg.solve(cell[bb].double().t(), diffs.double().t()).t()
                frac = ((frac + 0.5) % 1.0) - 0.5
                diffs_wrapped = (frac.float() @ cell[bb].float())
                out[idx_bb] = ref + diffs_wrapped
            return out

        pos_ref_contig = pos_ref
        if ads_mask.any():
            pos_ref_contig = _make_contiguous_ads(pos_ref)

        ads_center_ref = scatter(pos_ref_contig[ads_mask], bidx[ads_mask], dim=0, reduce="mean")  # (B,3)

        # Use Center of Mass instead of Geometric Center
        # ads_atomic_numbers = batch.atomic_numbers[ads_mask]
        # masses = torch.tensor(ase.data.atomic_masses[ads_atomic_numbers.long().cpu().numpy()], device=pos_ref.device, dtype=pos_ref.dtype)
        # weighted_pos = pos_ref[ads_mask] * masses.unsqueeze(-1)
        # sum_weighted_pos = scatter(weighted_pos, bidx[ads_mask], dim=0, dim_size=B, reduce="sum")
        # sum_masses = scatter(masses, bidx[ads_mask], dim=0, dim_size=B, reduce="sum")
        # ads_center_ref = sum_weighted_pos / sum_masses.unsqueeze(-1)

        # 以参考 COM 居中后的吸附体原子坐标（旋转作用在这上面，避免累积漂移）
        ads_pos_ref_centered = pos_ref_contig[ads_mask] - ads_center_ref[bidx[ads_mask]]

        # ---------------- NEW: 训练一致的旋转对称投影（linear/spherical） ----------------
        def _canonicalize_axis(axis: torch.Tensor) -> torch.Tensor:
            abs_axis = torch.abs(axis)
            idx = int(torch.argmax(abs_axis))
            val = float(axis[idx].item())
            if val == 0.0:
                return axis
            sign = 1.0 if val > 0.0 else -1.0
            return axis * axis.new_tensor(sign)

        def _build_rot_projector(rel_points: torch.Tensor) -> torch.Tensor:
            # rel_points: (N,3) centered coords of a single adsorbate
            device = rel_points.device
            dtype = rel_points.dtype
            I = torch.eye(3, device=device, dtype=dtype)
            if rel_points.numel() == 0 or rel_points.shape[0] <= 1:
                return torch.zeros((3, 3), device=device, dtype=dtype)
            cov = rel_points.t().double() @ rel_points.double()
            cov = cov / max(int(rel_points.shape[0]), 1)
            vals, vecs = torch.linalg.eigh(cov)
            vals = vals.float()
            vecs = vecs.float()
            max_val = vals.max()
            if (not torch.isfinite(max_val)) or float(max_val.item()) < 1.0e-8:
                return torch.zeros((3, 3), device=device, dtype=dtype)
            norm_vals = (vals / max_val).to(device=device, dtype=dtype)
            spread = norm_vals[-1] - norm_vals[0]
            if float(spread.item()) < 1.0e-2:
                return torch.zeros((3, 3), device=device, dtype=dtype)
            linear_tol = 5.0e-2
            if float(norm_vals[1].item()) < linear_tol and float(norm_vals[0].item()) < linear_tol:
                axis = vecs[:, -1]
                axis_norm = torch.linalg.norm(axis)
                if float(axis_norm.item()) < 1.0e-8:
                    return torch.zeros((3, 3), device=device, dtype=dtype)
                axis = _canonicalize_axis(axis / axis_norm)
                P = I - torch.outer(axis.to(dtype=dtype), axis.to(dtype=dtype))
                return P
            return I

        rot_projector = torch.zeros((B, 3, 3), device=pos_ref.device, dtype=pos_ref.dtype)
        for b in range(B):
            idx = (bidx[ads_mask] == b)
            if idx.any():
                rot_projector[b] = _build_rot_projector(ads_pos_ref_centered[idx])
            else:
                rot_projector[b] = torch.zeros((3, 3), device=pos_ref.device, dtype=pos_ref.dtype)

        # Optionally expose for downstream debugging
        batch.rot_projector = rot_projector


        # ---------------- NEW: 训练一致的“噪声锚点”初始化 ----------------
        # 噪声尺度可从 flow_opt 读，给出与训练一致的默认值
        tr_sigma  = float(self.flow_opt.get("tr_sigma", 3.0))
        rot_sigma = float(self.flow_opt.get("rot_sigma", 1.0))
        allow_z = bool(self.flow_opt.get("allow_z", True))

        # 先验上的平移噪声：只在 XY，有 PBC wrap
        eps_tr = torch.zeros_like(ads_center_ref).normal_() * tr_sigma
        # eps_tr[:, 2] = 0.0
        tr_sigma_z_scale = float(self.flow_opt.get("tr_sigma_z_scale", 0.3))

        if allow_z:
            eps_tr[:, 2] *= tr_sigma_z_scale
        else:
            eps_tr[:, 2] = 0.0

        # Match training: clip translation noise in XY
        tr_clip = self.flow_opt.get("tr_clip", None)
        if tr_clip is not None:
            tr_clip = float(tr_clip)
            tr_norm = torch.linalg.norm(eps_tr[:, :2], dim=-1)
            over = tr_norm > tr_clip
            if over.any():
                scale = torch.ones_like(tr_norm)
                scale[over] = tr_clip / (tr_norm[over] + 1e-12)
                eps_tr[:, :2] = eps_tr[:, :2] * scale[:, None]

        if allow_z:
            logging.info(f"[FlowTorch][Z-Debug] Initial Z noise (sigma_scale={tr_sigma_z_scale}): mean={eps_tr[:, 2].mean():.4f}, std={eps_tr[:, 2].std():.4f}, min={eps_tr[:, 2].min():.4f}, max={eps_tr[:, 2].max():.4f}")

        eps_tr = _pbc_wrap_xy(eps_tr, batch)  # wrap XY 到最近像
        # 先验上的旋转噪声：轴角
        eps_rot = torch.stack(
            [ads_center_ref.new_tensor(sample_vec(rot_sigma)) for _ in range(B)],
            dim=0,
        )

        # Match training: apply symmetry projector to the sampled rotation noise
        eps_rot = torch.matmul(rot_projector, eps_rot.unsqueeze(-1)).squeeze(-1)

        rot_clip = self.flow_opt.get("rot_clip", None)
        if rot_clip is not None:
            rot_clip = float(rot_clip)
            rot_norm = torch.linalg.norm(eps_rot, dim=-1)
            over_rot = rot_norm > rot_clip
            if over_rot.any():
                scale = torch.ones_like(rot_norm)
                scale[over_rot] = rot_clip / (rot_norm[over_rot] + 1e-12)
                eps_rot = eps_rot * scale[:, None]

        # 当前“锚点状态”：与训练 forward 中的 (ztr, zrot) 语义一致
        cur_ads_anchor = ads_center_ref + eps_tr           # (B,3)
        cur_rot = eps_rot.clone()                          # (B,3)

        # No lifting/overlap handling; rely on allow_z gating and noise scales only.

        # --- 构造第一帧坐标（避免 dtr_xy / zrot == 0 的静止） ---
        R0 = axis_angle_to_matrix(cur_rot).float()  # (B,3,3)
        ads_idx = torch.where(ads_mask)[0]
        sys_ids = bidx[ads_idx]
        x0 = torch.bmm(R0[sys_ids], ads_pos_ref_centered.unsqueeze(-1)).squeeze(-1)
        cur_pos = pos_ref.clone()
        cur_pos[ads_idx] = x0 + cur_ads_anchor[sys_ids]

        # Collect initial noisy state if requested
        step_positions = []
        step_ids = []
        if store_steps:
            step_positions.append(cur_pos.detach().clone().cpu())
            step_ids.append(0)

        # --- time grid ---
        num_steps = int(self.flow_opt.get("num_steps", 30))
        time_grid = str(self.flow_opt.get("time_grid", "cosine")).lower()
        if time_grid not in {"cosine", "uniform"}:
            raise ValueError(f"Unknown time_grid={time_grid}. Supported: cosine, uniform")
        if time_grid == "uniform":
            ts = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        else:
            kk = torch.arange(num_steps + 1, device=self.device)
            ts = 0.5 * (1.0 + torch.cos(np.pi * kk / num_steps))  # 1→0 的单调余弦
        cfg_scale = float(self.flow_opt.get("cfg_scale", 5.0))
        flow_type = str(self.flow_opt.pop("flow_type", "fm"))
        if flow_type != "fm":
            raise ValueError("FlowTorch supports only flow_type='fm'.")
        # allow_z moved to top
        write_every = int(self.flow_opt.get("write_every", 5))  # 0 表示只写末帧

        print(f"[FlowTorch] num_steps={num_steps}, cfg_scale={cfg_scale}, allow_z={allow_z}, write_every={write_every}, time_grid={time_grid}")

        integrator = str(self.flow_opt.get("integrator", "heun")).lower()
        if integrator not in {"euler", "heun"}:
            raise ValueError(f"Unknown integrator={integrator}. Supported: euler, heun")
        logging.info("[FlowTorch] integrator=%s", integrator)

        def _pose_to_positions(ads_anchor: torch.Tensor, rot_axis_angle: torch.Tensor) -> torch.Tensor:
            Rmat = axis_angle_to_matrix(rot_axis_angle).float()  # (B,3,3)
            x = torch.bmm(Rmat[sys_ids], ads_pos_ref_centered.unsqueeze(-1)).squeeze(-1)
            x = x + ads_anchor[sys_ids]
            new_pos_local = cur_pos.clone()  # slab + ads template
            new_pos_local[ads_idx] = x
            return new_pos_local

        def _wrap_anchor_update(anchor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            # Wrap only XY displacement into nearest PBC image
            delta_wrapped = _pbc_wrap_xy(delta, batch)
            anchor_new = anchor + delta_wrapped
            # 关键：把 anchor 相对 ads_center_ref 拉回训练分布
            anchor_new = ads_center_ref + _pbc_wrap_xy(anchor_new - ads_center_ref, batch)
            return anchor_new

        def _eval_velocity(
            pos_in: torch.Tensor,
            ads_anchor_in: torch.Tensor,
            rot_axis_angle_in: torch.Tensor,
            t_eval: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # 条件用“锚点状态”，不是实时 COM
            cond_base = batch.clone()
            cond_base.pos = pos_in
            cond_base.ztr = ads_anchor_in
            cond_base.zrot = rot_axis_angle_in
            cond_base.t = torch.full((B, 1), float(t_eval), device=self.device)
            cond_base.ads_center = ads_center_ref
            cond_base.energy = torch.zeros(B, device=self.device)
            cond_base.allow_z = allow_z

            cond_co = cond_base.clone()
            cond_co.cfg_conditioned = torch.ones(B, dtype=torch.bool, device=self.device)
            cond_co.allow_z = allow_z
            cond_un = cond_base.clone()
            cond_un.cfg_conditioned = torch.zeros(B, dtype=torch.bool, device=self.device)
            cond_un.allow_z = allow_z

            out_un = base_model(cond_un, mode="fm")
            out_co = base_model(cond_co, mode="fm")

            v_tr_un, v_rot_un = out_un.get("v_tr"), out_un.get("v_rot")
            v_tr_co, v_rot_co = out_co.get("v_tr"), out_co.get("v_rot")
            if any(x is None for x in [v_tr_un, v_rot_un, v_tr_co, v_rot_co]):
                raise RuntimeError("Model must return both translation & rotation fields for sampling.")

            v_tr  = v_tr_un  + cfg_scale * (v_tr_co  - v_tr_un)
            v_rot = v_rot_un + cfg_scale * (v_rot_co - v_rot_un)

            # Match training: project rotational velocity into symmetry-allowed subspace
            v_rot = torch.matmul(rot_projector, v_rot.unsqueeze(-1)).squeeze(-1)

            # 如果平移只有 xy，补 z=0
            if v_tr.shape[-1] == 2:
                v_tr = torch.cat([v_tr, torch.zeros(B, 1, device=self.device)], dim=-1)
            # 不允许 z 平移时强制置 0
            if not allow_z:
                v_tr[:, 2] = 0.0
            return v_tr, v_rot

        # --- prepare trajectories ---
        trajectories = None
        if self.traj_dir:
            trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in (self.traj_names or [f"{i}" for i in range(B)])
            ]

        base_model = getattr(self.model, "_unwrapped_model", None)
        if base_model is None:
            base_model = getattr(self.model, "model", self.model)
        original_training = base_model.training
        base_model.eval()
        fm_regression = getattr(base_model, "fm_regression_target", "velocity")
        if fm_regression != "velocity":
            logging.warning(
                "[FlowTorch] fm_regression_target=%s unsupported in velocity-only mode; continuing with velocities.",
                fm_regression,
            )

        logging.info("[FlowTorch] Start sampling with CFG (t=1→0)")
        for k in tqdm(range(num_steps)):
            t_cur, t_next = ts[k].item(), ts[k + 1].item()
            dt = t_cur - t_next

            if integrator == "euler":
                t_mid = 0.5 * (t_cur + t_next)
                v_tr, v_rot = _eval_velocity(cur_pos, cur_ads_anchor, cur_rot, t_mid)

                new_ads_anchor = _wrap_anchor_update(cur_ads_anchor, -v_tr * dt)
                new_rot = cur_rot - v_rot * dt
                new_pos = _pose_to_positions(new_ads_anchor, new_rot)
                cur_pos, cur_rot, cur_ads_anchor = new_pos, new_rot, new_ads_anchor

            else:  # heun / RK2
                # v1 @ (state, t_cur)
                v1_tr, v1_rot = _eval_velocity(cur_pos, cur_ads_anchor, cur_rot, t_cur)

                # predictor
                ads_anchor_pred = _wrap_anchor_update(cur_ads_anchor, -v1_tr * dt)
                rot_pred = cur_rot - v1_rot * dt
                pos_pred = _pose_to_positions(ads_anchor_pred, rot_pred)

                # v2 @ (predicted state, t_next)
                v2_tr, v2_rot = _eval_velocity(pos_pred, ads_anchor_pred, rot_pred, t_next)
                v_tr = 0.5 * (v1_tr + v2_tr)
                v_rot = 0.5 * (v1_rot + v2_rot)

                # corrector from original state
                new_ads_anchor = _wrap_anchor_update(cur_ads_anchor, -v_tr * dt)
                new_rot = cur_rot - v_rot * dt
                new_pos = _pose_to_positions(new_ads_anchor, new_rot)
                cur_pos, cur_rot, cur_ads_anchor = new_pos, new_rot, new_ads_anchor

            # 写轨迹（按需要每步写）
            if trajectories is not None and write_every > 0 and ((k + 1) % write_every == 0):
                tmp_batch = batch.clone()
                tmp_batch.pos = cur_pos
                self._write_traj_step(tmp_batch, trajectories)

            # 保存逐步坐标用于可视化/调试
            if store_steps and ((k + 1) % store_every == 0 or (k + 1) == num_steps):
                step_positions.append(cur_pos.detach().clone().cpu())
                step_ids.append(k + 1)

        # 末帧写盘
        if trajectories is not None:
            tmp_batch = batch.clone()
            tmp_batch.pos = cur_pos
            self._write_traj_step(tmp_batch, trajectories, final=True)

            for traj in trajectories:
                traj.close()
            # 把 .traj_tmp 改名为 .traj
            for name in (self.traj_names or [f"{i}" for i in range(B)]):
                tmp = self.traj_dir / f"{name}.traj_tmp"
                tmp.rename(tmp.with_suffix(".traj"))

        # 返回更新后的 batch
        out_batch = batch.clone()
        out_batch.pos = cur_pos

        if original_training:
            base_model.train()

        # 暴露给调用方的逐步坐标（list[Tensor])，仅在 store_steps=True 时非空
        if store_steps:
            self.collected_steps = step_positions
            self.collected_step_ids = step_ids
        else:
            self.collected_steps = None
            self.collected_step_ids = None

        return out_batch

    def _write_traj_step(self, batch: Batch, trajectories, final: bool = False):
        if not hasattr(batch, "y") or batch.y is None:
            batch.y = torch.zeros(batch.natoms.size(0), device=batch.pos.device)
        if not hasattr(batch, "force") or batch.force is None:
            batch.force = torch.zeros_like(batch.pos)
        atoms_list = batch_to_atoms(batch)
        for atm, traj in zip(atoms_list, trajectories):
            # 如果只保留末帧：final 时写；如果 save_full=False 则只写末帧
            if self.save_full or final:
                traj.write(atm)
