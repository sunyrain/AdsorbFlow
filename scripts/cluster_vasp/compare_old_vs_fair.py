#!/usr/bin/env python3
"""
对比旧方法 (best-per-level) 和新方法 (Fair SR@k) 的 VASP 结果。

分析 EqV2-2D 的结果差异：
- 旧方法: SR@10 = 61.4% (27/44), 101 VASP jobs
- 新方法: SR@10 = 68.18% (30/44), 345 VASP jobs
"""

import os
import sys
import pickle
import glob
import re
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ==================== PATHS ====================
# Old EqV2-2D VASP results (best-per-level)
OLD_BASE = os.path.join(
    _PROJECT_ROOT,
    "grid_search_runs/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
    "_epoch0180_unweightedvalloss1.0316_posmae0.9085"
    "/val_nonrelaxed_update/nsites_10/cfg7_steps5"
)

# New Fair VASP results (extracted from cluster)
FAIR_BASE = os.path.join(_PROJECT_ROOT, "vasp_fair_all2d", "vasp_fair_work", "vasp_fair")

# Reference data
REF_PATH = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_ref_energies.pkl")
TARGET_PATH = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_targets.pkl")

THRESHOLD = 0.1  # AdsorbML success threshold

# 44 evaluation SIDs
EVAL_SIDS_BASE = os.path.join(
    _PROJECT_ROOT,
    "grid_search_runs/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
    "_epoch0180_unweightedvalloss1.0316_posmae0.9085"
    "/val_nonrelaxed_update/nsites_10/cfg7_steps5/0/relaxations"
)

def get_eval_sids():
    """Get the 44 evaluation SIDs."""
    sids = set()
    for f in os.listdir(EVAL_SIDS_BASE):
        if f.endswith('.traj'):
            name = f.replace('.traj', '')
            parts = name.split('_')
            sid = '_'.join(parts[:3])
            sids.add(sid)
    return sids

# ==================== HELPERS ====================
def parse_toten(outcar_path):
    """Extract last TOTEN from OUTCAR."""
    try:
        with open(outcar_path) as f:
            content = f.read()
        totens = re.findall(r"TOTEN\s*=\s*([-\d.]+)", content)
        if totens:
            return float(totens[-1])
    except Exception:
        pass
    return None


def load_ref_and_targets():
    with open(REF_PATH, "rb") as f:
        ref = pickle.load(f)
    with open(TARGET_PATH, "rb") as f:
        targets = pickle.load(f)
    return ref, targets


def compute_ads_energy(total_e, ref_e):
    return total_e - ref_e


def is_success(ads_e, target):
    return (ads_e - target) <= THRESHOLD


# ==================== ANALYZE OLD RESULTS ====================
def analyze_old_results(ref, targets):
    """Analyze old best-per-level VASP results for EqV2-2D."""
    print("=" * 60)
    print("  旧方法 (best-per-level) EqV2-2D VASP 结果")
    print("=" * 60)

    levels = [1, 2, 5, 10]
    # For each SID, track which (SID, site) were tested and results
    sid_results = defaultdict(dict)  # sid -> {level -> (site, total_e, ads_e, success)}

    # Collect ALL VASP results across levels
    all_vasp_by_sid = defaultdict(list)  # sid -> [(level, site_idx, total_e, ads_e, success)]

    for level in levels:
        level_dir = os.path.join(OLD_BASE, f"vasp_level_{level}")
        if not os.path.isdir(level_dir):
            continue

        for sid_dir_name in sorted(os.listdir(level_dir)):
            outcar = os.path.join(level_dir, sid_dir_name, "OUTCAR")
            total_e = parse_toten(outcar)
            if total_e is None:
                continue

            # Parse SID from dir name: "{a}_{b}_{c}_{fid}" -> SID = "{a}_{b}_{c}", site = fid
            parts = sid_dir_name.split("_")
            sid = "_".join(parts[:3])
            site_idx = int(parts[3]) if len(parts) > 3 else 0

            if sid not in ref or sid not in targets:
                continue

            ads_e = compute_ads_energy(total_e, ref[sid])
            target = targets[sid]
            success = is_success(ads_e, target)

            all_vasp_by_sid[sid].append({
                "level": level,
                "site_idx": site_idx,
                "total_e": total_e,
                "ads_e": ads_e,
                "target": target,
                "diff": ads_e - target,
                "success": success,
            })

    eval_sids = get_eval_sids()

    # Compute old SR@k: SID succeeds at level k if any VASP at level ≤ k succeeded
    old_sr = {}
    for k in levels:
        successes = 0
        for sid in sorted(eval_sids):
            vasp_runs = all_vasp_by_sid.get(sid, [])
            any_success = any(r["success"] for r in vasp_runs if r["level"] <= k)
            if any_success:
                successes += 1
        n_sids = len(eval_sids)
        old_sr[k] = (successes, n_sids)

    print(f"\n评估 SID 数: {len(eval_sids)}")
    print(f"有 VASP 结果的 SID 数: {len([s for s in eval_sids if s in all_vasp_by_sid])}")
    print(f"总 VASP 运行数: {sum(len(v) for v in all_vasp_by_sid.values())}")

    print(f"\n旧方法 SR@k:")
    for k in levels:
        s, n = old_sr[k]
        print(f"  SR@{k:2d} = {s}/{n} = {s/n*100:.1f}%")

    # Detailed per-SID results
    print(f"\n各 SID 在旧方法中的最终结果 (level ≤ 10):")
    old_success_sids = set()
    old_fail_sids = set()
    for sid in sorted(eval_sids):
        vasp_runs = all_vasp_by_sid.get(sid, [])
        if not vasp_runs:
            status = "无VASP (全异常被跳过)"
            old_fail_sids.add(sid)
        elif any(r["success"] for r in vasp_runs):
            best = min(vasp_runs, key=lambda r: r["diff"])
            status = f"✅ 成功 (best diff={best['diff']:.4f})"
            old_success_sids.add(sid)
        else:
            best = min(vasp_runs, key=lambda r: r["diff"])
            status = f"❌ 失败 (best diff={best['diff']:.4f})"
            old_fail_sids.add(sid)
        print(f"  {sid}: {status} ({len(vasp_runs)} VASP runs)")

    return old_success_sids, old_fail_sids, all_vasp_by_sid


# ==================== ANALYZE FAIR RESULTS ====================
def analyze_fair_results(ref, targets, model_key="eqv2_2d"):
    """Analyze fair VASP results."""
    print("\n" + "=" * 60)
    print(f"  Fair 方法 {model_key.upper()} VASP 结果")
    print("=" * 60)

    # Parse all fair VASP results for this model
    fair_by_sid = defaultdict(list)  # sid -> [(site_idx, total_e, ads_e, success)]

    prefix = f"{model_key}__"
    for task_dir in sorted(os.listdir(FAIR_BASE)):
        if not task_dir.startswith(prefix):
            continue

        outcar = os.path.join(FAIR_BASE, task_dir, "OUTCAR")
        total_e = parse_toten(outcar)
        if total_e is None:
            continue

        # Parse: "{model_key}__{sid}__site{idx}"
        parts = task_dir.replace(prefix, "").split("__site")
        if len(parts) != 2:
            continue
        sid = parts[0]
        site_idx = int(parts[1])

        if sid not in ref or sid not in targets:
            continue

        ads_e = compute_ads_energy(total_e, ref[sid])
        target = targets[sid]
        success = is_success(ads_e, target)

        fair_by_sid[sid].append({
            "site_idx": site_idx,
            "total_e": total_e,
            "ads_e": ads_e,
            "target": target,
            "diff": ads_e - target,
            "success": success,
        })

    print(f"\n有 VASP 结果的 SID 数: {len(fair_by_sid)}")
    print(f"总 VASP 运行数: {sum(len(v) for v in fair_by_sid.values())}")

    eval_sids = get_eval_sids()

    # Count per-SID success
    fair_success_sids = set()
    fair_fail_sids = set()
    sid_m = {}  # sid -> m (number of successful sites)

    for sid in sorted(eval_sids):
        results = fair_by_sid.get(sid, [])
        m = sum(1 for r in results if r["success"])
        n = len(results)
        sid_m[sid] = (m, n)

        if m > 0:
            fair_success_sids.add(sid)
        else:
            fair_fail_sids.add(sid)

    # Fair SR@10: if m >= 1, P(at least 1 success in 10 draws) = 1 (since N=10, k=10)
    # More precisely: Fair SR@k = 1 - C(N-m, k) / C(N, k)
    from math import comb
    N = 10  # total sites per SID
    fair_sr = {}
    for k in [1, 2, 5, 10]:
        total_p = 0
        for sid in sorted(eval_sids):
            m, n = sid_m.get(sid, (0, 0))
            if N < k:
                p = 0
            elif m == 0:
                p = 0
            else:
                p = 1 - comb(N - m, k) / comb(N, k)
            total_p += p
        fair_sr[k] = total_p / len(eval_sids) * 100
        print(f"  Fair SR@{k:2d} = {fair_sr[k]:.2f}%")

    # Print SIDs with at least 1 success
    print(f"\n成功 SID 数 (至少1个site成功): {len(fair_success_sids)}/{len(eval_sids)}")
    print(f"  = {len(fair_success_sids)/len(eval_sids)*100:.1f}%")

    return fair_success_sids, fair_fail_sids, fair_by_sid, sid_m


# ==================== MAIN ====================
def main():
    ref, targets = load_ref_and_targets()

    # Analyze old results
    old_success, old_fail, old_by_sid = analyze_old_results(ref, targets)

    # Analyze fair results for EqV2-2D
    fair_success, fair_fail, fair_by_sid, sid_m = analyze_fair_results(ref, targets, "eqv2_2d")

    # Compare
    print("\n" + "=" * 60)
    print("  旧方法 vs Fair 方法对比 (EqV2-2D)")
    print("=" * 60)

    both_success = old_success & fair_success
    old_only = old_success - fair_success
    fair_only = fair_success - old_success
    both_fail = old_fail & fair_fail

    print(f"\n  两方法都成功: {len(both_success)} SIDs")
    print(f"  旧方法成功 + Fair失败: {len(old_only)} SIDs")
    print(f"  旧方法失败 + Fair成功: {len(fair_only)} SIDs ⬅️ 关键!")
    print(f"  两方法都失败: {len(both_fail)} SIDs")

    if fair_only:
        print(f"\n  🔍 Fair 新增成功的 SID (旧方法失败但 Fair 成功):")
        for sid in sorted(fair_only):
            fair_results = fair_by_sid.get(sid, [])
            m = sum(1 for r in fair_results if r["success"])
            n = len(fair_results)
            successes = [r for r in fair_results if r["success"]]
            old_results = old_by_sid.get(sid, [])

            print(f"\n    {sid}:")
            print(f"      Fair: {m}/{n} sites 成功")
            for s in successes:
                print(f"        site{s['site_idx']}: diff={s['diff']:.4f} eV ✅")
            print(f"      旧方法: 测了 {len(old_results)} 个 site")
            for o in old_results:
                print(f"        site{o['site_idx']} (level {o['level']}): diff={o['diff']:.4f} eV {'✅' if o['success'] else '❌'}")

    if old_only:
        print(f"\n  ⚠️ 旧方法成功但 Fair 失败的 SID:")
        for sid in sorted(old_only):
            print(f"    {sid}")

    # Also analyze PaiNN-2D fair results
    print("\n\n")
    fair_success_p, fair_fail_p, fair_by_sid_p, sid_m_p = analyze_fair_results(ref, targets, "painn_2d")

    # Check if PaiNN-2D has any old VASP results
    painn_old_base = os.path.join(
        _PROJECT_ROOT,
        "grid_search_runs/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint"
        "/val_nonrelaxed_update/nsites_10/cfg5_steps5"
    )
    has_old_painn = os.path.exists(painn_old_base)
    vasp_level_dirs = glob.glob(os.path.join(painn_old_base, "vasp_level_*")) if has_old_painn else []

    print("\n" + "=" * 60)
    print("  PaiNN-2D 旧 VASP 结果检查")
    print("=" * 60)
    print(f"  PaiNN-2D 基础路径存在: {has_old_painn}")
    print(f"  vasp_level_* 目录数: {len(vasp_level_dirs)}")
    if vasp_level_dirs:
        for d in vasp_level_dirs:
            count = len(os.listdir(d))
            print(f"    {os.path.basename(d)}: {count} dirs")
    else:
        print("  ❌ PaiNN-2D 没有任何旧的 VASP 结果!")
        print("  用户可能混淆了 PaiNN-3D (SR@10=54.5%) 和 PaiNN-2D")

    # Summary
    print("\n" + "=" * 60)
    print("  📊 最终总结")
    print("=" * 60)
    print("""
  1. EqV2-2D:
     旧方法 (best-per-level): SR@10 = 61.4% (27/44)
       → 每个 SID 只测了 MLFF 排名第1的 site
       → 如果该 site VASP 失败，即使其他 site 能成功，SID 也算失败
       → 总共只做了 101 次 VASP

     Fair 方法: SR@10 = 68.18% (30/44)
       → 每个 SID 测了所有 site (8-10 个)
       → 只要有任何 site VASP 成功，SID 就算成功
       → 总共做了 345 次 VASP

     → Fair SR@10 高于旧方法是完全正常的！
       原因: 旧方法只测最佳排名 site，会漏掉 MLFF 排名不高但 VASP 实际成功的 site

  2. PaiNN-2D:
     没有旧的 VASP 结果！
     之前只有 MLFF SR@10 = 63.6%
     现在 VASP Fair SR@10 = 50.0%
     下降原因: 9 个 SID 的 GemNet-OC 能量预测有偏差（MLFF 假阳性）

     ⚠️ 用户可能混淆了:
       - PaiNN-3D-0.15 旧 VASP: SR@10 = 54.5% (这是 3D 模型，不是 2D)
       - PaiNN-2D MLFF: SR@10 = 63.6% (这是 MLFF，不是 VASP)
""")


if __name__ == "__main__":
    main()
