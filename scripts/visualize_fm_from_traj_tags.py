#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Flow Matching (FM) trajectory from a .traj file, using external tags in oc20dense_tags.pkl.

It will:
  1) Load initial (frame 0) and ground-truth final (last frame) from the .traj.
  2) Parse sid from the filename (default: join first 3 underscore-separated tokens).
  3) Read tags from oc20dense_tags.pkl via sid; treat atoms with tag==1 or tag==2 as adsorbate.
  4) Integrate the learned flow (mode='fm') from t=1 -> 0 to get a predicted final pose.
  5) Report COM-XY error and adsorbate RMSD; save a multi-frame XYZ for inspection.

Usage:
  python visualize_fm_from_traj_tags.py --traj /path/to/71_7114_7_heur18.traj \    --ckpt /path/to/model.pt --tags-pkl /path/to/oc20dense_tags.pkl \    --steps 80 --device cuda --xyz-out fm_traj.xyz

Requirements:
  - ase
  - torch, torch_geometric
  - Your project installed/importable so that load_model() can instantiate the trained model.
"""

import argparse
import os
import sys
import pickle
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from ase.io import read, write
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase import Atoms

def save_traj_traj(path, pos_list, numbers, cell, pbc, tags=None, info_per_frame=None):
    """
    pos_list: list[np.ndarray] or list[torch.Tensor] of shape (N,3)
    numbers : list[int] (atomic numbers)
    cell    : 3x3 array-like
    pbc     : tuple[bool,bool,bool] or bool
    tags    : optional, array-like of length N (will be stored in atoms.arrays['tags'])
    info_per_frame: optional, list[dict] same length as pos_list
    """
    traj = Trajectory(path, 'w')
    try:
        for i, pos in enumerate(pos_list):
            import numpy as np
            if not isinstance(pos, np.ndarray):
                pos = pos.detach().cpu().numpy()
            atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)
            if tags is not None:
                import numpy as np
                atoms.new_array('tags', np.asarray(tags, dtype=int))
            if info_per_frame is not None:
                atoms.info.update(info_per_frame[i])
            traj.write(atoms)
    finally:
        traj.close()
# -------------------------
# SO(3) utilities
# -------------------------
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """aa: (B,3) axis-angle -> (B,3,3) rotation matrix"""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / theta
    kx, ky, kz = k[...,0], k[...,1], k[...,2]
    O = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([ O, -kz,  ky], dim=-1),
        torch.stack([ kz,  O, -kx], dim=-1),
        torch.stack([-ky,  kx,  O], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=aa.device).expand(aa.size(0),3,3)
    s, c = torch.sin(theta)[...,None], torch.cos(theta)[...,None]
    return I + s*K + (1.0 - c) * (K @ K)

def log_SO3(R: torch.Tensor) -> torch.Tensor:
    """(B,3,3) -> (B,3) rotation vector (axis-angle)"""
    tr = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    wx = (R[...,2,1] - R[...,1,2]) / (2*torch.sin(theta))
    wy = (R[...,0,2] - R[...,2,0]) / (2*torch.sin(theta))
    wz = (R[...,1,0] - R[...,0,1]) / (2*torch.sin(theta))
    w = torch.stack([wx, wy, wz], dim=-1)
    return w * theta.unsqueeze(-1)

# -----------------------------------
# Geometry: reconstruct from a pose
# -----------------------------------
@torch.no_grad()
def reconstruct_pos_from_pose(x_star: torch.Tensor,
                              tags: torch.Tensor,
                              batch_vec: torch.Tensor,
                              ads_center_star: torch.Tensor,
                              ztr: torch.Tensor,
                              zrot: torch.Tensor) -> torch.Tensor:
    """Return current coordinates given pose (ztr, zrot)."""
    ads_mask  = (tags == 2)
    ads_batch = batch_vec[ads_mask]
    rel_star  = x_star[ads_mask] - ads_center_star[ads_batch]  # (N_ads,3)

    R = axis_angle_to_matrix(zrot)  # (B,3,3)
    rel_t = torch.empty_like(rel_star)
    B = ztr.size(0)
    for b in range(B):
        idx = (ads_batch == b)
        if idx.any():
            rel_t[idx] = rel_star[idx] @ R[b].transpose(-1, -2)

    pos_t = x_star.clone()
    pos_t[ads_mask] = rel_t + ztr[ads_batch]
    return pos_t

# ----------------------
# Model loading (EDIT!)
# ----------------------
def load_model(ckpt_path: str, device: str):
    from adsorbdiff.models.painn.painn_denoising import PaiNN
    import torch, yaml

    cfg = dict(hidden_channels=512, num_layers=6, cutoff=6.0,
               regress_forces=False, direct_forces=False)
    model = PaiNN(
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        cutoff=cfg.get("cutoff", 6.0),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# --------------------------
# FM integration (predict)
# --------------------------
@torch.no_grad()
def integrate_flow_fm(model: nn.Module,
                      data: Data,
                      x_star: torch.Tensor,
                      steps: int = 80,
                      tr_sigma: float = 1.0,
                      rot_sigma: float = 0.5,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (com_xy [T+1,2], rot_angle [T+1], pos_list [T+1 x (N,3)])."""
    torch.manual_seed(seed)
    device = x_star.device

    ads_mask = (data.tags == 2)
    B = 1
    ads_center_star = torch.zeros(B,3, device=device)
    ads_center_star[0] = x_star[ads_mask].mean(dim=0)

    # sample z1
    eps_tr  = torch.zeros(B,3, device=device).normal_(std=tr_sigma)
    eps_tr[:,2] = 0.0
    eps_rot = torch.zeros(B,3, device=device).normal_(std=rot_sigma)
    ztr  = ads_center_star + eps_tr
    zrot = eps_rot.clone()

    dt = 1.0 / steps
    com_xy, rot_ang, pos_list = [], [], []

    pos_t = reconstruct_pos_from_pose(x_star, data.tags, data.batch, ads_center_star, ztr, zrot)
    com_xy.append(ztr[0,:2].detach().cpu().numpy())
    rot_ang.append(zrot.norm(dim=-1)[0].detach().cpu().numpy().item())
    pos_list.append(pos_t.detach().cpu().numpy())

    for k in range(steps):
        t = torch.tensor([[max(1.0 - k*dt, 1e-3)]], device=device)  # (1,1)

        d = Data(pos=pos_t, batch=data.batch, tags=data.tags, atomic_numbers=data.atomic_numbers,
                 cell=data.cell, natoms=data.natoms)
        d.t = t; d.ztr = ztr; d.zrot = zrot; d.ads_center = ads_center_star

        out = model(d, mode='fm')
        v_tr, v_rot = out['v_tr'], out['v_rot']

        # Euler step
        ztr  = ztr.clone()
        ztr[:,:2] = ztr[:,:2] - dt * v_tr
        zrot = zrot - dt * v_rot

        pos_t = reconstruct_pos_from_pose(x_star, data.tags, data.batch, ads_center_star, ztr, zrot)
        com_xy.append(ztr[0,:2].detach().cpu().numpy())
        rot_ang.append(zrot.norm(dim=-1)[0].detach().cpu().numpy().item())
        pos_list.append(pos_t.detach().cpu().numpy())

    import numpy as np
    return np.stack(com_xy, axis=0), np.array(rot_ang), pos_list

# -----------------------
# Error computations
# -----------------------
def compute_errors(pos_pred0: np.ndarray, pos_gt: np.ndarray, tags: np.ndarray) -> dict:
    mask = (tags == 2)
    if mask.sum() == 0:
        raise ValueError('No adsorbate atoms (tag==1/2) found for this sid.')
    com_pred = pos_pred0[mask].mean(axis=0)
    com_gt   = pos_gt[mask].mean(axis=0)
    com_xy_err = float(np.linalg.norm((com_pred - com_gt)[:2]))
    diff = pos_pred0[mask] - pos_gt[mask]
    rmsd = float(np.sqrt((diff**2).sum(axis=1).mean()))
    return {'com_xy_err_A': com_xy_err, 'ads_rmsd_A': rmsd}

# -----------------------
# sid parser
# -----------------------
def parse_sid_from_fname(fname: str) -> str:
    """Default: join first 3 underscore-separated tokens (e.g., '71_7114_7_heur18.traj' -> '71_7114_7')."""
    base = os.path.basename(fname)
    tokens = base.split('_')
    if len(tokens) < 3:
        raise ValueError(f'Filename {base} too short to parse sid.')
    return '_'.join(tokens[:3])

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/root/autodl-tmp/AdsorbFlow/71_7114_7_heur18.traj")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/AdsorbFlow/checkpoints/2025-09-27-08-53-20/checkpoint.pt")
    ap.add_argument('--tags-pkl', default="/root/autodl-tmp/AdsorbFlow/oc20dense_tags.pkl")
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--xyz-out', default='fm_traj.xyz')
    args = ap.parse_args()

    device = args.device

    # 1) Load traj
    frames = read(args.traj, ':')
    if len(frames) < 2:
        print('traj must contain at least 2 frames (initial and final).'); sys.exit(1)
    ini, fin = frames[0], frames[-1]

    # 2) Load tags.pkl and parse sid
    with open(args.tags_pkl, 'rb') as f:
        tags_map = pickle.load(f)
    sid = parse_sid_from_fname(args.traj)
    if sid not in tags_map:
        print(f'ERROR: sid={sid} not found in {args.tags_pkl}'); sys.exit(1)
    tags_np = np.array(tags_map[sid], dtype=np.int64)
    if tags_np.shape[0] != len(ini):
        print(f'ERROR: tag length {tags_np.shape[0]} != natoms {len(ini)} for sid={sid}'); sys.exit(1)
    if ( (tags_np == 2)).sum() == 0:
        print(f'ERROR: sid={sid} has no adsorbate atoms (tag 1/2).'); sys.exit(1)

    # 3) Build single-sample Data
    Z = np.array(ini.get_atomic_numbers(), dtype=np.int64)
    pos_ini = torch.tensor(ini.get_positions(), dtype=torch.float32, device=device)
    pos_fin = torch.tensor(fin.get_positions(), dtype=torch.float32, device=device)
    cell = torch.tensor(fin.get_cell().array, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,3)
    N = pos_ini.size(0)
    data = Data(
        pos=pos_ini.clone(),
        batch=torch.zeros(N, dtype=torch.long, device=device),
        tags=torch.tensor(tags_np, dtype=torch.long, device=device),
        atomic_numbers=torch.tensor(Z, dtype=torch.long, device=device),
        cell=cell,
        natoms=torch.tensor([N], dtype=torch.long, device=device),
    )

    # 4) Load model
    model = load_model(args.ckpt, device)

    # 5) Integrate learned flow
    com_xy, rot_ang, pos_list = integrate_flow_fm(
        model, data, x_star=pos_fin, steps=args.steps, tr_sigma=1.0, rot_sigma=0.5, seed=args.seed
    )

    # 6) Compare predicted final to GT final
    pos_pred0 = pos_list[-1]
    errs = compute_errors(pos_pred0, pos_fin.cpu().numpy(), tags_np)
    print('==== Errors (pred final vs GT final) ====')
    for k,v in errs.items():
        print(f'{k}: {v:.6f}')

    # 7) Save multi-frame XYZ
    numbers = Z.tolist()
    frames_xyz = [Atoms(numbers=numbers, positions=pos, cell=fin.get_cell(), pbc=fin.get_pbc()) for pos in pos_list]
    write(args.xyz_out, frames_xyz)
    print(f'Saved trajectory to {args.xyz_out}')

    numbers = Z.tolist()
    pbc = fin.get_pbc()
    cell = fin.get_cell()
    # 可选：每帧记录 t/k 等信息
    info_per_frame = [{'k': k, 't': max(1.0 - k/args.steps, 0.0)} for k in range(len(pos_list))]
    
    save_traj_traj('fm_traj.traj', pos_list, numbers, cell, pbc,
                   tags=tags_np, info_per_frame=info_per_frame)
    print('Saved ASE trajectory to fm_traj.traj')

if __name__ == '__main__':
    main()
