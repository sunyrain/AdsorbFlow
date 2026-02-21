#!/usr/bin/env python3
"""
统计 Nsite=1 和 Nsite=2 时，论文方法 A（MLFF-best → DFT SP）在不同 seed 选择下的分布。

Nsite=1: 10 种选择（单独用 seed 0/1/.../9），统计 SR 的均值、最大、最小
Nsite=2: C(10,2)=45 种选择（任取 2 个 seed），统计 SR 的均值、最大、最小
"""

import os, sys, glob, pickle
import numpy as np
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from adsorbdiff.placement.flag_anomaly import DetectTrajAnomaly
import ase.io

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


def load_data():
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

        for seed in range(NSITES):
            relax_dir = os.path.join(traj_base, str(seed), "relaxations")
            if not os.path.isdir(relax_dir):
                continue
            traj_files = sorted(glob.glob(os.path.join(relax_dir, "*.traj")))
            for tf in tqdm(traj_files, desc=f"  {label} seed {seed}", leave=False):
                sid = extract_sid(os.path.basename(tf))
                try:
                    traj = ase.io.read(tf, index=":")
                    anom = is_anomalous(traj)
                    try:
                        mlff_e = traj[-1].get_potential_energy()
                    except Exception:
                        mlff_e = traj[-1].info.get("energy")
                    if mlff_e is not None:
                        mlff_e = float(np.atleast_1d(mlff_e).reshape(-1)[0])
                    model_data[sid][seed] = {"mlff_energy": mlff_e, "anomalous": anom}
                except Exception:
                    model_data[sid][seed] = {"mlff_energy": None, "anomalous": True}

        # VASP DFT
        for sid in model_data:
            for seed in range(NSITES):
                vasp_dir = os.path.join(VASP_BASE, f"{model_key}__{sid}__site{seed}")
                outcar = os.path.join(vasp_dir, "OUTCAR")
                dft_total_e = parse_vasp_energy(outcar)
                if dft_total_e is not None and sid in ref_energies:
                    dft_ads_e = dft_total_e - ref_energies[sid]
                    if seed in model_data[sid]:
                        model_data[sid][seed]["dft_energy"] = dft_ads_e
                    else:
                        model_data[sid][seed] = {"mlff_energy": None, "anomalous": False, "dft_energy": dft_ads_e}
                else:
                    if seed in model_data[sid]:
                        model_data[sid][seed].setdefault("dft_energy", None)

        print(f"  {label}: {len(model_data)} SIDs loaded")
        data[model_key] = dict(model_data)
    return data, targets


def paper_sr_for_seed_subset(model_data, targets, seed_subset):
    """
    论文方法 A: 从给定 seed 子集中选 MLFF-best 非异常结构，检查其 DFT。
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]
        best_seed = None
        best_mlff_e = float("inf")
        for s in seed_subset:
            info = model_data[sid].get(s)
            if info is None or info.get("anomalous", True):
                continue
            me = info.get("mlff_energy")
            if me is not None and me < best_mlff_e:
                best_mlff_e = me
                best_seed = s
        if best_seed is None:
            continue
        dft_e = model_data[sid][best_seed].get("dft_energy")
        if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
            success += 1
    return success


def dft_union_for_seed_subset(model_data, targets, seed_subset):
    """
    方法 B: seed 子集中任意一个 DFT 通过即成功。
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]
        for s in seed_subset:
            info = model_data[sid].get(s)
            if info is None or info.get("anomalous", True):
                continue
            dft_e = info.get("dft_energy")
            if dft_e is not None and (dft_e - target_e) <= SUCCESS_THRESHOLD:
                success += 1
                break
    return success


def mlff_sr_for_seed_subset(model_data, targets, seed_subset):
    """
    MLFF 级别: seed 子集中任意一个 MLFF 通过即成功。
    """
    success = 0
    for sid in model_data:
        if sid not in targets:
            continue
        target_e = targets[sid]
        for s in seed_subset:
            info = model_data[sid].get(s)
            if info is None or info.get("anomalous", True):
                continue
            me = info.get("mlff_energy")
            if me is not None and (me - target_e) <= SUCCESS_THRESHOLD:
                success += 1
                break
    return success


def main():
    data, targets = load_data()

    for model_key in sorted(data.keys()):
        label = MODELS[model_key]["label"]
        model_data = data[model_key]

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        # ==================== Nsite=1 ====================
        print(f"\n  ── Nsite=1 (每个 seed 单独作为 Nsite=1 实验) ──")
        print(f"  {'seed':>6}  {'MLFF SR':>10}  {'A.论文DFT':>10}  {'B.DFT并集':>10}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")

        mlff_1s, paper_1s, union_1s = [], [], []
        for seed in range(NSITES):
            subset = (seed,)
            mlff = mlff_sr_for_seed_subset(model_data, targets, subset) / TOTAL_SIDS * 100
            paper = paper_sr_for_seed_subset(model_data, targets, subset) / TOTAL_SIDS * 100
            union = dft_union_for_seed_subset(model_data, targets, subset) / TOTAL_SIDS * 100
            mlff_1s.append(mlff)
            paper_1s.append(paper)
            union_1s.append(union)
            print(f"  {seed:>6}  {mlff:>9.2f}%  {paper:>9.2f}%  {union:>9.2f}%")

        # Note: 对于 Nsite=1, A 和 B 完全一致（只有 1 个结构）
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"  {'均值':>6}  {np.mean(mlff_1s):>9.2f}%  {np.mean(paper_1s):>9.2f}%  {np.mean(union_1s):>9.2f}%")
        print(f"  {'最大':>6}  {np.max(mlff_1s):>9.2f}%  {np.max(paper_1s):>9.2f}%  {np.max(union_1s):>9.2f}%")
        print(f"  {'最小':>6}  {np.min(mlff_1s):>9.2f}%  {np.min(paper_1s):>9.2f}%  {np.min(union_1s):>9.2f}%")
        print(f"  {'标准差':>6}  {np.std(mlff_1s):>9.2f}%  {np.std(paper_1s):>9.2f}%  {np.std(union_1s):>9.2f}%")

        # 10 个 seed 中每个 SID 的逐个情况
        print(f"\n  ── Nsite=1 逐 SID 分析 (哪些 SID 在不同 seed 下稳定/不稳定) ──")
        sid_pass_counts_mlff = {}
        sid_pass_counts_dft = {}
        for sid in sorted(model_data.keys()):
            if sid not in targets:
                continue
            target_e = targets[sid]
            mlff_passes = 0
            dft_passes = 0
            for s in range(NSITES):
                info = model_data[sid].get(s)
                if info is None or info.get("anomalous", True):
                    continue
                me = info.get("mlff_energy")
                if me is not None and (me - target_e) <= SUCCESS_THRESHOLD:
                    mlff_passes += 1
                de = info.get("dft_energy")
                if de is not None and (de - target_e) <= SUCCESS_THRESHOLD:
                    dft_passes += 1
            sid_pass_counts_mlff[sid] = mlff_passes
            sid_pass_counts_dft[sid] = dft_passes

        # 不稳定 SID (0 < passes < 10)
        unstable = [(sid, sid_pass_counts_mlff[sid], sid_pass_counts_dft[sid])
                     for sid in sid_pass_counts_mlff
                     if 0 < sid_pass_counts_dft[sid] < NSITES]
        unstable.sort(key=lambda x: x[2])

        if unstable:
            print(f"  {'SID':>15}  {'MLFF通过':>8}/{NSITES}  {'DFT通过':>8}/{NSITES}")
            print(f"  {'─'*15}  {'─'*12}  {'─'*12}")
            for sid, mp, dp in unstable:
                bar_m = "█" * mp + "░" * (NSITES - mp)
                bar_d = "█" * dp + "░" * (NSITES - dp)
                print(f"  {sid:>15}  {mp:>4}/{NSITES} {bar_m}  {dp:>4}/{NSITES} {bar_d}")

        # ==================== Nsite=2 ====================
        print(f"\n  ── Nsite=2 (C(10,2)=45 种 seed 组合) ──")

        all_pairs = list(combinations(range(NSITES), 2))
        mlff_2s, paper_2s, union_2s = [], [], []
        for pair in all_pairs:
            mlff = mlff_sr_for_seed_subset(model_data, targets, pair) / TOTAL_SIDS * 100
            paper = paper_sr_for_seed_subset(model_data, targets, pair) / TOTAL_SIDS * 100
            union = dft_union_for_seed_subset(model_data, targets, pair) / TOTAL_SIDS * 100
            mlff_2s.append(mlff)
            paper_2s.append(paper)
            union_2s.append(union)

        print(f"  {'指标':>8}  {'MLFF SR':>10}  {'A.论文DFT':>10}  {'B.DFT并集':>10}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"  {'均值':>8}  {np.mean(mlff_2s):>9.2f}%  {np.mean(paper_2s):>9.2f}%  {np.mean(union_2s):>9.2f}%")
        print(f"  {'最大':>8}  {np.max(mlff_2s):>9.2f}%  {np.max(paper_2s):>9.2f}%  {np.max(union_2s):>9.2f}%")
        print(f"  {'最小':>8}  {np.min(mlff_2s):>9.2f}%  {np.min(paper_2s):>9.2f}%  {np.min(union_2s):>9.2f}%")
        print(f"  {'标准差':>8}  {np.std(mlff_2s):>9.2f}%  {np.std(paper_2s):>9.2f}%  {np.std(union_2s):>9.2f}%")

        # 最好和最差的组合
        best_idx_paper = int(np.argmax(paper_2s))
        worst_idx_paper = int(np.argmin(paper_2s))
        best_pair = all_pairs[best_idx_paper]
        worst_pair = all_pairs[worst_idx_paper]
        print(f"\n  论文方法最佳组合: seeds {best_pair} → SR={paper_2s[best_idx_paper]:.2f}%")
        print(f"  论文方法最差组合: seeds {worst_pair} → SR={paper_2s[worst_idx_paper]:.2f}%")

        best_idx_union = int(np.argmax(union_2s))
        worst_idx_union = int(np.argmin(union_2s))
        print(f"  DFT并集最佳组合: seeds {all_pairs[best_idx_union]} → SR={union_2s[best_idx_union]:.2f}%")
        print(f"  DFT并集最差组合: seeds {all_pairs[worst_idx_union]} → SR={union_2s[worst_idx_union]:.2f}%")

        # 分布直方图 (文字)
        print(f"\n  Nsite=2 论文方法 A 分布 (45 种组合):")
        vals = sorted(set(paper_2s))
        for v in vals:
            cnt = paper_2s.count(v)
            bar = "█" * cnt
            print(f"    {v:>6.2f}%: {cnt:>3}次  {bar}")

        # ==================== Nsite=3,5 均值 ====================
        print(f"\n  ── Nsite=3,5 统计 (抽样) ──")
        for nsite_k in [3, 5]:
            all_combos = list(combinations(range(NSITES), nsite_k))
            paper_ks, union_ks, mlff_ks = [], [], []
            for combo in all_combos:
                mlff = mlff_sr_for_seed_subset(model_data, targets, combo) / TOTAL_SIDS * 100
                paper = paper_sr_for_seed_subset(model_data, targets, combo) / TOTAL_SIDS * 100
                union = dft_union_for_seed_subset(model_data, targets, combo) / TOTAL_SIDS * 100
                mlff_ks.append(mlff)
                paper_ks.append(paper)
                union_ks.append(union)

            print(f"\n  Nsite={nsite_k} ({len(all_combos)} 种组合):")
            print(f"  {'指标':>8}  {'MLFF SR':>10}  {'A.论文DFT':>10}  {'B.DFT并集':>10}")
            print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
            print(f"  {'均值':>8}  {np.mean(mlff_ks):>9.2f}%  {np.mean(paper_ks):>9.2f}%  {np.mean(union_ks):>9.2f}%")
            print(f"  {'最大':>8}  {np.max(mlff_ks):>9.2f}%  {np.max(paper_ks):>9.2f}%  {np.max(union_ks):>9.2f}%")
            print(f"  {'最小':>8}  {np.min(mlff_ks):>9.2f}%  {np.min(paper_ks):>9.2f}%  {np.min(union_ks):>9.2f}%")
            print(f"  {'标准差':>8}  {np.std(mlff_ks):>9.2f}%  {np.std(paper_ks):>9.2f}%  {np.std(union_ks):>9.2f}%")


if __name__ == "__main__":
    main()
