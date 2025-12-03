# adsorbdiff/relaxation/flow_torch.py
import logging
import os
from pathlib import Path
from typing import Optional, Dict
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
    Wrap displacement to nearest PBC image on XY only.
    Supports per-atom (sum(natoms),3) or per-sample (B,3).
    (等价于 MeanFlowTrainer 里的实现，这里内联一份方便独立使用)
    """
    B = int(batch.natoms.size(0))
    out = torch.zeros_like(vec_xyz)
    cell = batch.cell.double()

    if vec_xyz.shape[0] == batch.batch.shape[0]:
        # per-atom
        for b in range(B):
            mask = (batch.batch == b)
            if not mask.any():
                continue
            frac = torch.linalg.solve(cell[b].t(), vec_xyz[mask].double().t()).t()
            frac[..., :2] = ((frac[..., :2] + 0.5) % 1.0) - 0.5
            cart = (frac.float() @ batch.cell[b].float())
            cart[..., 2] = vec_xyz[mask][..., 2]
            out[mask] = cart
    elif vec_xyz.shape[0] == B:
        # per-sample
        for b in range(B):
            frac = torch.linalg.solve(cell[b].t(), vec_xyz[b].double())
            frac[:2] = ((frac[:2] + 0.5) % 1.0) - 0.5
            cart = (frac.float() @ batch.cell[b].float())
            cart[2] = vec_xyz[b, 2]
            out[b] = cart
    else:
        raise ValueError("vec_xyz first dim must be num_atoms or batch size B")
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

        # --- anchors / masks ---
        tags = batch.tags
        ads_mask = (tags == 2)
        bidx = batch.batch
        # 参考位置（若有 pos_relaxed 就用它）
        pos_ref = batch.pos_relaxed if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None else batch.pos
        # ads_center_ref = scatter(pos_ref[ads_mask], bidx[ads_mask], dim=0, reduce="mean")  # (B,3)
        
        # Use Center of Mass instead of Geometric Center
        ads_atomic_numbers = batch.atomic_numbers[ads_mask]
        masses = torch.tensor(ase.data.atomic_masses[ads_atomic_numbers.long().cpu().numpy()], device=pos_ref.device, dtype=pos_ref.dtype)
        weighted_pos = pos_ref[ads_mask] * masses.unsqueeze(-1)
        sum_weighted_pos = scatter(weighted_pos, bidx[ads_mask], dim=0, dim_size=B, reduce="sum")
        sum_masses = scatter(masses, bidx[ads_mask], dim=0, dim_size=B, reduce="sum")
        ads_center_ref = sum_weighted_pos / sum_masses.unsqueeze(-1)

        # 以参考 COM 居中后的吸附体原子坐标（旋转作用在这上面，避免累积漂移）
        ads_pos_ref_centered = pos_ref[ads_mask] - ads_center_ref[bidx[ads_mask]]


        # ---------------- NEW: 训练一致的“噪声锚点”初始化 ----------------
        # 噪声尺度可从 flow_opt 读，给出与训练一致的默认值
        tr_sigma  = float(self.flow_opt.get("tr_sigma", 2.0))
        rot_sigma = float(self.flow_opt.get("rot_sigma", 1.0))
        allow_z = bool(self.flow_opt.get("allow_z", True))  # 默认允许 z 方向平移

        # 先验上的平移噪声：只在 XY，有 PBC wrap
        eps_tr = torch.zeros_like(ads_center_ref).normal_() * tr_sigma
        # eps_tr[:, 2] = 0.0
        tr_sigma_z_scale = float(self.flow_opt.get("tr_sigma_z_scale", 0.3))
        
        if allow_z:
            eps_tr[:, 2] *= tr_sigma_z_scale
        else:
            eps_tr[:, 2] = 0.0

        eps_tr = _pbc_wrap_xy(eps_tr, batch)  # wrap XY 到最近像
        # 先验上的旋转噪声：轴角
        eps_rot = torch.stack(
            [ads_center_ref.new_tensor(sample_vec(rot_sigma)) for _ in range(B)],
            dim=0,
        )

        # 当前“锚点状态”：与训练 forward 中的 (ztr, zrot) 语义一致
        cur_ads_anchor = ads_center_ref + eps_tr           # (B,3)
        cur_rot = eps_rot.clone()                          # (B,3)
        
        # --- Overlap Check & Lift (Initial Placement) ---
        # Check if the initial random placement (t=1) causes overlap with the slab.
        # If so, lift the adsorbate.
        # If Z-axis is ignored (allow_z=False), lift by a fixed amount (1.0A) if overlap occurs.
        
        # 1. Reconstruct initial adsorbate positions (t=1)
        # Apply rotation R to relative positions
        R_init = axis_angle_to_matrix(cur_rot).float() # (B, 3, 3)
        ads_idx = torch.where(ads_mask)[0]
        sys_ids = bidx[ads_idx]
        
        # ads_pos_ref_centered is (N_ads, 3)
        # We need to apply R[sys_ids] to ads_pos_ref_centered
        rel_rotated = torch.bmm(R_init[sys_ids], ads_pos_ref_centered.unsqueeze(-1)).squeeze(-1) # (N_ads, 3)
        
        # Proposed Z positions of adsorbate atoms
        prop_ads_z = cur_ads_anchor[sys_ids, 2] + rel_rotated[:, 2]
        
        # 2. Find min Z of adsorbate per batch
        ads_min_z = scatter(prop_ads_z, sys_ids, dim=0, dim_size=B, reduce="min")
        
        # 3. Find max Z of slab per batch
        slab_mask = (tags != 2)
        if slab_mask.any():
            slab_z = pos_ref[slab_mask, 2]
            slab_batch = bidx[slab_mask]
            slab_max_z = scatter(slab_z, slab_batch, dim=0, dim_size=B, reduce="max")
            
            # 4. Check Overlap
            # Threshold: 0.5 A separation
            overlap_amount = (slab_max_z + 0.5) - ads_min_z
            lift_mask = overlap_amount > 0
            
            if not allow_z:
                # Unconditional lift of 1.0 A if Z is ignored
                cur_ads_anchor[:, 2] += 1.0
            elif lift_mask.any():
                # Lift to resolve overlap
                lift_val = torch.where(lift_mask, overlap_amount, torch.zeros_like(overlap_amount))
                
                # Apply lift to cur_ads_anchor (which effectively updates eps_tr)
                cur_ads_anchor[:, 2] += lift_val
                # logging.info(f"[FlowTorch] Lifted {lift_mask.sum()} samples due to overlap.")

        # ---------------------------------------------------------------

        # --- 构造第一帧坐标（避免 dtr_xy / zrot == 0 的静止） ---
        R0 = axis_angle_to_matrix(cur_rot).float()  # (B,3,3)
        ads_idx = torch.where(ads_mask)[0]
        sys_ids = bidx[ads_idx]
        x0 = torch.bmm(R0[sys_ids], ads_pos_ref_centered.unsqueeze(-1)).squeeze(-1)
        cur_pos = pos_ref.clone()
        cur_pos[ads_idx] = x0 + cur_ads_anchor[sys_ids]

        # --- time grid (CHANGED: 余弦网格，避免起步导数≈0) ---
        num_steps = int(self.flow_opt.get("num_steps", 30))
        k = torch.arange(num_steps + 1, device=self.device)
        ts = 0.5 * (1.0 + torch.cos(np.pi * k / num_steps))  # 1→0 的单调余弦
        cfg_scale = float(self.flow_opt.get("cfg_scale", 5.0))
        flow_type = str(self.flow_opt.pop("flow_type", "fm"))
        if flow_type != "fm":
            raise ValueError("FlowTorch supports only flow_type='fm'.")
        # allow_z moved to top
        write_every = int(self.flow_opt.get("write_every", 5))  # 0 表示只写末帧

        print(f"[FlowTorch] num_steps={num_steps}, cfg_scale={cfg_scale}, allow_z={allow_z}, write_every={write_every}")
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
            t_mid = 0.5 * (t_cur + t_next)
            dt = t_cur - t_next

            # -------------- CHANGED: 条件用“锚点状态”，不是实时 COM --------------
            cond_base = batch.clone()
            cond_base.pos = cur_pos
            cond_base.ztr = cur_ads_anchor                    # 与训练中的 ztr 语义一致
            cond_base.zrot = cur_rot                          # 与训练中的 zrot 语义一致
            cond_base.t = torch.full((B, 1), t_mid, device=self.device)
            cond_base.ads_center = ads_center_ref
            cond_base.energy = torch.zeros(B, device=self.device)
            cond_base.allow_z = allow_z

            cond_co = cond_base.clone()
            cond_co.cfg_conditioned = torch.ones(B, dtype=torch.bool, device=self.device)
            cond_co.allow_z = allow_z
            cond_un = cond_base.clone()
            cond_un.cfg_conditioned = torch.zeros(B, dtype=torch.bool, device=self.device)
            cond_un.allow_z = allow_z
            # ---------------------------------------------------------------------

            out_un = base_model(cond_un, mode="fm")
            out_co = base_model(cond_co, mode="fm")

            def _extract_fields(out_dict):
                return out_dict.get("v_tr"), out_dict.get("v_rot")

            v_tr_un, v_rot_un = _extract_fields(out_un)
            v_tr_co, v_rot_co = _extract_fields(out_co)
            if any(x is None for x in [v_tr_un, v_rot_un, v_tr_co, v_rot_co]):
                raise RuntimeError("Model must return both translation & rotation fields for sampling.")
            v_tr  = v_tr_un  + cfg_scale * (v_tr_co  - v_tr_un)
            v_rot = v_rot_un + cfg_scale * (v_rot_co - v_rot_un)

            if os.getenv("FLOW_DEBUG_CFG", ""):
                delta_tr = v_tr_co - v_tr_un
                delta_rot = v_rot_co - v_rot_un
                def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
                    return (a * b).sum(dim=-1) / denom
                cos_tr = _cosine(v_tr_un, v_tr_co)
                cos_rot = _cosine(v_rot_un, v_rot_co)
                cos_tr_delta = _cosine(v_tr_un, delta_tr)
                cos_rot_delta = _cosine(v_rot_un, delta_rot)
                logging.info(
                    "[FlowTorch][CFG-debug] Δv_tr max=%.4e mean=%.4e | Δv_rot max=%.4e mean=%.4e",
                    float(delta_tr.norm(dim=-1).max().item()),
                    float(delta_tr.norm(dim=1).mean().item()),
                    float(delta_rot.norm(dim=1).max().item()),
                    float(delta_rot.norm(dim=1).mean().item()),
                )
                logging.info(
                    "[FlowTorch][CFG-debug] ||v_tr_un|| mean=%.4e | ||v_tr_co|| mean=%.4e",
                    float(v_tr_un.norm(dim=-1).mean().item()),
                    float(v_tr_co.norm(dim=-1).mean().item()),
                )
                logging.info(
                    "[FlowTorch][CFG-debug] cos(v_tr_un,v_tr_co)=%.4f | cos(v_rot_un,v_rot_co)=%.4f",
                    float(cos_tr.mean().item()),
                    float(cos_rot.mean().item()),
                )
                logging.info(
                    "[FlowTorch][CFG-debug] cos(v_tr_un,Δv_tr)=%.4f | cos(v_rot_un,Δv_rot)=%.4f",
                    float(cos_tr_delta.mean().item()),
                    float(cos_rot_delta.mean().item()),
                )

            # 如果平移只有 xy，补 z=0
            if v_tr.shape[-1] == 2:
                v_tr = torch.cat([v_tr, torch.zeros(B, 1, device=self.device)], dim=-1)
            # 不允许 z 平移时强制置 0
            if not allow_z:
                v_tr[:, 2] = 0.0

            # --------- CHANGED: 用“锚点状态”进行 ODE 积分 & PBC wrap ----------
            new_ads_anchor = cur_ads_anchor - v_tr * dt     # 与 t 从 1→0 的方向配套
            new_rot        = cur_rot        - v_rot * dt

            # PBC：把 ads COM 的位移 wrap 到 XY 最近像，避免漂移到远处
            disp = new_ads_anchor - cur_ads_anchor
            disp_wrapped = _pbc_wrap_xy(disp, batch)
            new_ads_anchor = cur_ads_anchor + disp_wrapped
            # ---------------------------------------------------------------------

            # 刚体旋转 + 平移（只更新吸附体原子），基于“参考-居中”坐标避免数值漂移
            R = axis_angle_to_matrix(new_rot).float()  # (B,3,3)
            x = torch.bmm(R[sys_ids], ads_pos_ref_centered.unsqueeze(-1)).squeeze(-1)
            x = x + new_ads_anchor[sys_ids]
            new_pos = cur_pos.clone()
            new_pos[ads_idx] = x

            # 更新状态（注意：下一步条件仍使用锚点，而不是实时 COM）
            cur_pos, cur_rot, cur_ads_anchor = new_pos, new_rot, new_ads_anchor

            # 写轨迹（按需要每步写）
            if trajectories is not None and write_every > 0 and ((k + 1) % write_every == 0):
                tmp_batch = batch.clone()
                tmp_batch.pos = cur_pos
                self._write_traj_step(tmp_batch, trajectories)

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
