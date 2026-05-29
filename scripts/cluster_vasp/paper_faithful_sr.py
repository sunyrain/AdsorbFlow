#!/usr/bin/env python3
"""
论文一致的 DFT SR@k 计算 — 对比三种方法

方法 A (论文方法 / Paper-exact):
  对每个 Nsites=k:
    1) 从 seeds 0..k-1 中，找 MLFF 能量最低的非异常结构
    2) 检查该结构的 DFT 结果是否 ≤ target + 0.1 eV
    3) SR@k = 成功 SID 数 / 44
  → This matches the original AdsorbDiff DFT selection protocol.
  → 每个 SID 只依赖 1 个 DFT 结果（MLFF-best 的那个）

方法 B (DFT 并集 / DFT-union):
  对每个 k:
    1) 从 seeds 0..k-1 中，检查是否有 *任意一个* seed 的 DFT 结果通过
    2) SR@k = 成功 SID 数 / 44
  → 这是 AdsorbML 的严格定义: rank by ML, select top k, DFT all k, any pass → success
  → 但我们用 seed 顺序代替 MLFF 排序 (和论文的 Nsites 实验一致)

方法 C (公平分析 / Fair-analytical):
  1) 所有 10 个 seed 都有 DFT 结果
  2) 计算每个 SID 的成功 seed 数 m
  3) Fair SR@k = mean(1 - C(N-m,k)/C(N,k)) 消除 seed 顺序偏差

三者关系:
  - 在 k=10=N 时: B 和 C 完全一致（都是全部 seed 的并集）
  - A ≤ B: A 只依赖 MLFF-best 的 DFT 结果，B 检查所有 k 个 DFT 结果
  - A 是论文实际用的方法
"""

import os
import sys
import json
import pickle
import glob
import numpy as np
from math import comb
from collections import defaultdict
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from adsorbdiff.placement.flag_anomaly import DetectTrajAnomaly

# ==================== CONFIG ====================
MODELS = {
    "painn_2d": {
        "traj_base": os.path.join(
            _PROJECT_ROOT,
            "grid_search_runs/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint"
            "/val_nonrelaxed_update/nsites_10/cfg5_steps5",
        ),
        "label": "PaiNN-2D",
    },
    "eqv2_2d": {
        "traj_base": os.path.join(
            _PROJECT_ROOT,
            "grid_search_runs/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
            "_epoch0180_unweightedvalloss1.0316_posmae0.9085"
            "/val_nonrelaxed_update/nsites_10/cfg7_steps5",
        ),
        "label": "EqV2-2D",
    },
}

VASP_BASE = os.path.join(_PROJECT_ROOT, "vasp_fair_all2d/vasp_fair_work/vasp_fair")
NSITES = 10
TOTAL_SIDS = 44
SUCCESS_THRESHOLD = 0.1

# ==================== HELPERS ====================

def _make_ads_contiguous(atoms):
    tags = atoms.get_tags()
    ads_idx = np.where(tags == 2)[0]
    if ads_idx.size < 2:
        return atoms
    cell = atoms.get_cell()
    try:
        inv_cell = np.linalg.inv(cell)
    except Exception:
        return atoms
    pos = atoms.get_positions().copy()
    ref = pos[ads_idx[0]]
    diffs = pos[ads_idx] - ref
    frac = diffs @ inv_cell
    frac = (frac + 0.5) % 1.0 - 0.5
    pos[ads_idx] = ref + frac @ cell
    new_atoms = atoms.copy()
    new_atoms.set_positions(pos)
    return new_atoms

import ase.io

def is_anomalous(traj):
    try:
        initial = _make_ads_contiguous(traj[0])
        final = _make_ads_contiguous(traj[-1])
        tags = initial.get_tags()
        detector = DetectTrajAnomaly(initial, final, tags)
        return any([
            detector.is_adsorbate_dissociated(),
            detector.is_adsorbate_desorbed(),
            detector.has_surface_changed(),
            detector.is_adsorbate_intercalated(),
        ])
    except Exception:
        return True


def extract_sid(filename):
    name = filename.replace(".traj", "")
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return name


def parse_vasp_energy(outcar_path):
    energy = None
    if not os.path.exists(outcar_path):
        return None
    with open(outcar_path) as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                energy = float(line.strip().split()[-2])
    return energy


def load_all_data():
    """
    加载所有数据:
    返回 per-model dict:
      data[model_key] = {
        sid: {
          seed: {
            "mlff_energy": float or None,   # MLFF 弛豫后预测的吸附能
            "dft_energy": float or None,     # VASP DFT 吸附能
            "anomalous": bool,               # 是否异常
          }
        }
      }
    """
    ref_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_ref_energies.pkl")
    target_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_targets.pkl")
    with open(ref_path, "rb") as f:
        ref_energies = pickle.load(f)
    with open(target_path, "rb") as f:
        targets = pickle.load(f)
    if isinstance(next(iter(targets.values())), dict):
        targets = {sid: min(v.values()) for sid, v in targets.items()}

    data = {}
    for model_key, cfg in MODELS.items():
        traj_base = cfg["traj_base"]
        label = cfg["label"]
        print(f"\n加载 {label} ...")
        model_data = defaultdict(dict)

        # Step 1: 读取 MLFF 轨迹，获取能量和异常状态
        for seed in range(NSITES):
            relax_dir = os.path.join(traj_base, str(seed), "relaxations")
            if not os.path.isdir(relax_dir):
                print(f"  警告: 缺少 seed {seed} 的 relaxations 目录")
                continue

            traj_files = sorted(glob.glob(os.path.join(relax_dir, "*.traj")))
            for tf in tqdm(traj_files, desc=f"  {label} seed {seed}", leave=False):
                sid = extract_sid(os.path.basename(tf))
                try:
                    traj = ase.io.read(tf, index=":")
                    anom = is_anomalous(traj)

                    # MLFF energy
                    try:
                        mlff_e = traj[-1].get_potential_energy()
                    except Exception:
                        mlff_e = traj[-1].info.get("energy")
                    if mlff_e is not None:
                        mlff_e = float(np.atleast_1d(mlff_e).reshape(-1)[0])

                    model_data[sid][seed] = {
                        "mlff_energy": mlff_e,
                        "anomalous": anom,
                    }
                except Exception as e:
                    model_data[sid][seed] = {
                        "mlff_energy": None,
                        "anomalous": True,
                    }

        # Step 2: 读取 VASP DFT 结果
        vasp_count = 0
        for seed in range(NSITES):
            task_name = None  # 动态从目录扫
            for sid in model_data:
                vasp_dir = os.path.join(VASP_BASE, f"{model_key}__{sid}__site{seed}")
                outcar = os.path.join(vasp_dir, "OUTCAR")
                dft_total_e = parse_vasp_energy(outcar)
                if dft_total_e is not None and sid in ref_energies:
                    dft_ads_e = dft_total_e - ref_energies[sid]
                    if sid in model_data and seed in model_data[sid]:
                        model_data[sid][seed]["dft_energy"] = dft_ads_e
                        vasp_count += 1
                    else:
                        model_data[sid][seed] = {
                            "mlff_energy": None,
                            "anomalous": False,
                            "dft_energy": dft_ads_e,
                        }
                else:
                    if sid in model_data and seed in model_data[sid]:
                        model_data[sid][seed].setdefault("dft_energy", None)

        print(f"  {label}: {len(model_data)} SIDs, {vasp_count} VASP 结果")
        data[model_key] = dict(model_data)

    return data, targets


def compute_paper_exact_sr(model_data, targets, k, nsites=10, total_sids=44):
    """
    方法 A (论文方法): 从 seeds 0..k-1, 选 MLFF-best 非异常结构, 检查其 DFT.
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]

        # 从 seeds 0..k-1 找 MLFF-best 非异常结构
        best_seed = None
        best_mlff_e = float("inf")
        for s in range(k):
            info = model_data[sid].get(s)
            if info is None:
                continue
            if info.get("anomalous", True):
                continue
            me = info.get("mlff_energy")
            if me is not None and me < best_mlff_e:
                best_mlff_e = me
                best_seed = s

        if best_seed is None:
            continue  # 所有结构都异常

        # 检查 MLFF-best 结构的 DFT 结果
        dft_e = model_data[sid][best_seed].get("dft_energy")
        if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
            success += 1

    return success / total_sids * 100


def compute_dft_union_sr(model_data, targets, k, nsites=10, total_sids=44):
    """
    方法 B (DFT 并集): seeds 0..k-1 中任意一个 DFT 通过即成功.
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]

        any_pass = False
        for s in range(k):
            info = model_data[sid].get(s)
            if info is None:
                continue
            if info.get("anomalous", True):
                continue
            dft_e = info.get("dft_energy")
            if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
                any_pass = True
                break

        if any_pass:
            success += 1

    return success / total_sids * 100


def compute_fair_analytical_sr(model_data, targets, k, nsites=10, total_sids=44):
    """
    方法 C (公平分析): P(success | k random draws) = 1 - C(N-m, k) / C(N, k)
    """
    total_prob = 0.0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]

        # 统计成功 seed 数 m (跨全部 nsites)
        m = 0
        n_valid = 0
        for s in range(nsites):
            info = model_data[sid].get(s)
            if info is None:
                continue
            if info.get("anomalous", True):
                continue
            n_valid += 1
            dft_e = info.get("dft_energy")
            if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
                m += 1

        # 使用全部 nsites 作为 N (包括异常的，视为失败)
        N = nsites
        if m == 0:
            p = 0.0
        elif N - m < k:
            p = 1.0
        else:
            p = 1.0 - comb(N - m, k) / comb(N, k)
        total_prob += p

    return total_prob / total_sids * 100


def compute_mlff_ranked_dft_sr(model_data, targets, k, nsites=10, total_sids=44):
    """
    附加方法 D (MLFF 排序 + DFT 并集):
    从全部 10 个 seed 中按 MLFF 能量排序(升序), 取 top-k,
    然后检查这 top-k 的 DFT 结果中是否有任意通过.
    与方法 B 不同: B 用 seed 顺序(0..k-1), D 用 MLFF 能量排序.
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]

        # 收集所有 (seed, mlff_e, dft_e) 非异常的
        candidates = []
        for s in range(nsites):
            info = model_data[sid].get(s)
            if info is None:
                continue
            if info.get("anomalous", True):
                continue
            me = info.get("mlff_energy")
            de = info.get("dft_energy")
            if me is not None:
                candidates.append((me, de, s))

        # 按 MLFF 能量升序
        candidates.sort(key=lambda x: x[0])

        # 取 top-k
        top_k = candidates[:k]

        # 检查任意一个 DFT 通过
        any_pass = False
        for me, de, s in top_k:
            if de is not None and (de - target_e) <= SUCCESS_THRESHOLD:
                any_pass = True
                break

        if any_pass:
            success += 1

    return success / total_sids * 100


def compute_mlff_sr(model_data, targets, k, nsites=10, total_sids=44):
    """
    MLFF 级别 SR@k (sequential union, 与 grid search 一致):
    从 seeds 0..k-1, 每个 seed 找该 SID 的 min MLFF 能量非异常结构,
    若 mlff_energy - target ≤ 0.1 则该 seed 成功, 取并集.
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]

        any_pass = False
        for s in range(k):
            info = model_data[sid].get(s)
            if info is None:
                continue
            if info.get("anomalous", True):
                continue
            me = info.get("mlff_energy")
            if me is not None and (me - target_e) <= SUCCESS_THRESHOLD:
                any_pass = True
                break

        if any_pass:
            success += 1

    return success / total_sids * 100


def main():
    data, targets = load_all_data()

    k_values = [1, 2, 3, 5, 10]

    for model_key in sorted(data.keys()):
        label = MODELS[model_key]["label"]
        model_data = data[model_key]

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        # 统计
        n_sids = len(model_data)
        n_seeds_with_dft = sum(
            1 for sid in model_data
            for s in model_data[sid]
            if model_data[sid][s].get("dft_energy") is not None
        )
        n_anomalous = sum(
            1 for sid in model_data
            for s in model_data[sid]
            if model_data[sid][s].get("anomalous", True)
        )
        print(f"  SIDs: {n_sids}, DFT 结果数: {n_seeds_with_dft}, 异常数: {n_anomalous}")

        header = f"  {'k':>3}  |  {'MLFF SR':>10}  |  {'A.论文方法':>10}  |  {'B.DFT并集':>10}  |  {'C.Fair分析':>10}  |  {'D.MLFF排序':>10}"
        print(header)
        print(f"  {'─'*3}  |  {'─'*10}  |  {'─'*10}  |  {'─'*10}  |  {'─'*10}  |  {'─'*10}")

        for k in k_values:
            mlff_sr = compute_mlff_sr(model_data, targets, k)
            paper_sr = compute_paper_exact_sr(model_data, targets, k)
            union_sr = compute_dft_union_sr(model_data, targets, k)
            fair_sr = compute_fair_analytical_sr(model_data, targets, k)
            ranked_sr = compute_mlff_ranked_dft_sr(model_data, targets, k)

            print(f"  {k:>3}  |  {mlff_sr:>9.2f}%  |  {paper_sr:>9.2f}%  |  {union_sr:>9.2f}%  |  {fair_sr:>9.2f}%  |  {ranked_sr:>9.2f}%")

        # Per-SID 详细对比 (k=10)
        print(f"\n  Per-SID 详情 (k=10, 只显示差异):")
        print(f"  {'SID':>15}  {'MLFF-best':>10}  {'DFT-best':>10}  {'target':>10}  {'A.论文':>6}  {'B.并集':>6}  {'差异':>4}")
        print(f"  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*4}")

        diff_count = 0
        for sid in sorted(model_data.keys()):
            if sid not in targets:
                continue
            target_e = targets[sid]

            # MLFF-best seed
            best_seed = None
            best_mlff_e = float("inf")
            for s in range(NSITES):
                info = model_data[sid].get(s)
                if info is None or info.get("anomalous", True):
                    continue
                me = info.get("mlff_energy")
                if me is not None and me < best_mlff_e:
                    best_mlff_e = me
                    best_seed = s

            # Paper-exact result
            paper_pass = False
            if best_seed is not None:
                dft_e = model_data[sid][best_seed].get("dft_energy")
                if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
                    paper_pass = True

            # DFT-union result
            union_pass = False
            best_dft_e = float("inf")
            for s in range(NSITES):
                info = model_data[sid].get(s)
                if info is None or info.get("anomalous", True):
                    continue
                dft_e = info.get("dft_energy")
                if dft_e is not None:
                    if dft_e < best_dft_e:
                        best_dft_e = dft_e
                    if (dft_e - target_e) <= SUCCESS_THRESHOLD:
                        union_pass = True

            if paper_pass != union_pass:
                diff_count += 1
                mlff_str = f"{best_mlff_e:.4f}" if best_mlff_e < float("inf") else "N/A"
                dft_str = f"{best_dft_e:.4f}" if best_dft_e < float("inf") else "N/A"
                p_mark = "✓" if paper_pass else "✗"
                u_mark = "✓" if union_pass else "✗"
                print(f"  {sid:>15}  {mlff_str:>10}  {dft_str:>10}  {target_e:>10.4f}  {p_mark:>6}  {u_mark:>6}  {'←!!':>4}")

        if diff_count == 0:
            print(f"  (所有 SID 在方法 A 和 B 间一致)")
        else:
            print(f"\n  共 {diff_count} 个 SID 在方法 A 和 B 之间有差异")
            print(f"  (B'并集'比 A'论文' 多检出的 = B 额外覆盖的 SID)")

    # 总结表格
    print(f"\n\n{'='*80}")
    print(f"  总结对比表 (所有方法)")
    print(f"{'='*80}")
    print(f"  论文定义: '... DFT SR = 从 N 个候选中选 MLFF-best, DFT SP 验证'")
    print(f"  → 对应方法 A")
    print()

    for model_key in sorted(data.keys()):
        label = MODELS[model_key]["label"]
        model_data = data[model_key]
        print(f"  {label}:")
        print(f"    {'':>15}  {'SR@1':>8}  {'SR@5':>8}  {'SR@10':>8}")
        print(f"    {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}")

        for method_name, compute_fn in [
            ("MLFF级别", compute_mlff_sr),
            ("A.论文方法", compute_paper_exact_sr),
            ("B.DFT并集", compute_dft_union_sr),
            ("C.Fair分析", compute_fair_analytical_sr),
            ("D.MLFF排序", compute_mlff_ranked_dft_sr),
        ]:
            sr1 = compute_fn(model_data, targets, 1)
            sr5 = compute_fn(model_data, targets, 5)
            sr10 = compute_fn(model_data, targets, 10)
            print(f"    {method_name:>15}  {sr1:>7.2f}%  {sr5:>7.2f}%  {sr10:>7.2f}%")
        print()

    print(f"  方法说明:")
    print(f"    MLFF级别:  GemNet-OC 预测能量 vs DFT target (无 VASP 验证)")
    print(f"    A.论文方法: seeds 0..k-1 中 MLFF-best → DFT SP → 检查")
    print(f"    B.DFT并集:  seeds 0..k-1 中 *任意* DFT 通过 → 成功")
    print(f"    C.Fair分析: 全部 10 seed DFT → 解析公式 (random k draws)")
    print(f"    D.MLFF排序: 全部 10 seed 按 MLFF 排序 → 取 top-k → 任意 DFT 通过")
    print()
    print(f"  关键关系:")
    print(f"    - 论文 Figure 3 对应方法 A")
    print(f"    - k=10 时: B = C (全部 seed 并集)")
    print(f"    - A ≤ B ≤ MLFF (每层都有可能损失)")
    print(f"    - D 是 MLFF 排序版的 B, 应比 B(seed顺序) 效果好或相当")


if __name__ == "__main__":
    main()
