#!/usr/bin/env python3
"""
公平 SR@k 计算：使用解析公式消除种子顺序偏差。

原理:
  每个 SID 有 N=10 个独立种子（site_idx 0-9），每次都做了 MLFF 弛豫。
  对每个种子独立判定是否成功（MLFF energy ≤ target + 0.1）。
  
  SR@k 的含义："随机从 N 个种子中选 k 个，至少一个成功的概率"
  
  解析公式:
    P(SID 成功 | k) = 1 - C(N-m, k) / C(N, k)
    其中 m = 该 SID 在 N 次中成功的次数
    
  SR@k = Σ P(SID 成功 | k) / total_SIDs

用法:
    python scripts/cluster_vasp/compute_fair_sr.py \
        --grid-search-dir <nsites_10/cfgX_stepsY 目录>
"""

import argparse
import os
import sys
import json
import pickle
import glob
import numpy as np
from math import comb
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from adsorbdiff.placement import DetectTrajAnomaly
import ase.io


ANOMALY_TYPES = ["dissociated", "desorbed", "surface_changed", "intercalated"]
SUCCESS_THRESHOLD = 0.1


def _make_ads_contiguous(atoms):
    tags = atoms.get_tags()
    ads_idx = np.where(tags == 2)[0]
    if ads_idx.size == 0:
        return atoms
    cell = atoms.get_cell()
    try:
        inv_cell = np.linalg.inv(cell.T)
    except Exception:
        return atoms
    pos = atoms.get_positions()
    ref = pos[ads_idx[0]]
    diffs = pos[ads_idx] - ref
    frac = diffs @ inv_cell
    frac = (frac + 0.5) % 1.0 - 0.5
    pos[ads_idx] = ref + frac @ cell
    new_atoms = atoms.copy()
    new_atoms.set_positions(pos)
    return new_atoms


def check_anomaly(traj_path, sid):
    """检查单个 traj 是否存在异常"""
    try:
        traj = ase.io.read(traj_path, ":")
        initial = _make_ads_contiguous(traj[0])
        final = _make_ads_contiguous(traj[-1])
        tags = initial.get_tags()
        detector = DetectTrajAnomaly(initial, final, tags)
        anom = np.array([
            detector.is_adsorbate_dissociated(),
            detector.is_adsorbate_desorbed(),
            detector.has_surface_changed(),
            detector.is_adsorbate_intercalated(),
        ])
        return anom.any()
    except Exception:
        return True


def compute_per_seed_success(cfg_dir, nsites=10):
    """
    对每个 SID、每个 seed 独立判定 MLFF 预测是否成功。
    
    返回:
        sid_seed_success: {sid: [bool, bool, ..., bool]}  长度=nsites
        total_sids: int
    """
    # 加载 target 数据
    target_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_targets.pkl")
    with open(target_path, "rb") as f:
        targets = pickle.load(f)
    # 如果 targets 是嵌套 dict，转为 sid -> min_energy
    sample_value = next(iter(targets.values()))
    if isinstance(sample_value, dict):
        converted = {}
        for sid, candidates in targets.items():
            energies = [v for v in candidates.values() if isinstance(v, (int, float))]
            if energies:
                converted[sid] = min(energies)
        targets = converted

    # 收集每个 seed 的每个 SID 的结果
    sid_seed_results = defaultdict(lambda: {})  # {sid: {seed: energy_or_None}}
    all_sids = set()

    for seed in range(nsites):
        relax_dir = os.path.join(cfg_dir, str(seed), "relaxations")
        if not os.path.isdir(relax_dir):
            continue
        for traj_path in glob.glob(os.path.join(relax_dir, "*.traj")):
            filename = os.path.basename(traj_path).replace(".traj", "")
            if filename.count("_") == 3:
                sid = "_".join(filename.split("_")[:-1])
            else:
                sid = filename
            all_sids.add(sid)

            try:
                # 检查异常
                is_anom = check_anomaly(traj_path, sid)
                if is_anom:
                    sid_seed_results[sid][seed] = None  # 异常 → 视为失败
                    continue

                atoms = ase.io.read(traj_path)
                energy_raw = atoms.get_potential_energy()
                energy = float(np.atleast_1d(energy_raw).reshape(-1)[0])

                if sid not in targets:
                    sid_seed_results[sid][seed] = None
                    continue

                target_energy = targets[sid]
                diff = energy - target_energy
                sid_seed_results[sid][seed] = diff <= SUCCESS_THRESHOLD
            except Exception:
                sid_seed_results[sid][seed] = None

    # 整理为 {sid: [success_0, success_1, ..., success_9]}
    sid_seed_success = {}
    for sid in sorted(all_sids):
        results = []
        for seed in range(nsites):
            val = sid_seed_results[sid].get(seed, None)
            results.append(bool(val) if val is not None else False)
        sid_seed_success[sid] = results

    return sid_seed_success, len(all_sids)


def analytical_sr_at_k(sid_seed_success, total_sids, k, N=10):
    """
    使用解析公式计算公平的 SR@k
    
    P(SID 成功 | k) = 1 - C(N-m, k) / C(N, k)
    其中 m = 该 SID 的成功种子数
    
    当 N-m < k 时，C(N-m, k) = 0，P = 1 (必然成功)
    """
    total_prob = 0.0
    for sid, successes in sid_seed_success.items():
        m = sum(successes)
        if m == 0:
            p = 0.0
        elif N - m < k:
            p = 1.0  # 不可能选 k 个全部失败
        else:
            p = 1.0 - comb(N - m, k) / comb(N, k)
        total_prob += p

    return total_prob / total_sids * 100


def sequential_sr_at_k(sid_seed_success, total_sids, k):
    """
    传统顺序 SR@k (与 grid_search 脚本一致): 
    SR@k = |{sid: ∃ seed < k, success}| / total_sids
    """
    success_count = 0
    for sid, successes in sid_seed_success.items():
        if any(successes[:k]):
            success_count += 1
    return success_count / total_sids * 100


def main():
    parser = argparse.ArgumentParser(description="公平 SR@k 计算")
    parser.add_argument("--grid-search-dir", required=True,
                        help="Grid search 结果目录 (包含 0/ 1/ ... 9/ 子目录)")
    parser.add_argument("--nsites", type=int, default=10)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()

    print(f"计算目录: {args.grid_search_dir}")
    print(f"种子数: {args.nsites}")
    print()

    sid_seed_success, total_sids = compute_per_seed_success(args.grid_search_dir, args.nsites)
    print(f"总 SIDs: {total_sids}")

    # 统计每个 SID 的成功种子数分布
    m_dist = defaultdict(int)
    for sid, successes in sid_seed_success.items():
        m = sum(successes)
        m_dist[m] += 1

    print(f"\n每个 SID 成功的种子数分布:")
    for m in range(args.nsites + 1):
        if m_dist[m] > 0:
            print(f"  m={m:>2}: {m_dist[m]} 个 SID")

    # 计算 SR@k
    print(f"\n{'k':>3}  {'顺序SR@k':>10}  {'公平SR@k':>10}  {'差异':>8}")
    print(f"{'':->3}  {'':->10}  {'':->10}  {'':->8}")

    for k in args.levels:
        if k > args.nsites:
            break
        seq_sr = sequential_sr_at_k(sid_seed_success, total_sids, k)
        fair_sr = analytical_sr_at_k(sid_seed_success, total_sids, k, args.nsites)
        diff = fair_sr - seq_sr
        print(f"{k:>3}  {seq_sr:>9.1f}%  {fair_sr:>9.1f}%  {diff:>+7.1f}%")

    # 详细每个 SID 结果
    print(f"\n详细结果 (每个 SID 在各种子的成功/失败):")
    print(f"{'SID':>15}  {'成功数':>5}  {'P@1':>6}  {'P@5':>6}  {'seeds':>12}")
    print(f"{'':->15}  {'':->5}  {'':->6}  {'':->6}  {'':->12}")
    for sid in sorted(sid_seed_success.keys()):
        successes = sid_seed_success[sid]
        m = sum(successes)
        p1 = 1.0 - comb(args.nsites - m, 1) / comb(args.nsites, 1) if m > 0 else 0
        p5 = 1.0 - (comb(args.nsites - m, 5) / comb(args.nsites, 5) if args.nsites - m >= 5 else 0) if m > 0 else 0
        seeds_str = "".join(["✓" if s else "✗" for s in successes])
        print(f"{sid:>15}  {m:>5}  {p1:>5.1%}  {p5:>5.1%}  {seeds_str}")


if __name__ == "__main__":
    main()
