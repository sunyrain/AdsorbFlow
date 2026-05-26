"""
多层次 site VASP 结果分析脚本

分析不同 site 层级的计算结果，比较：
- 1 site vs 2 site vs 5 site vs 10 site 的能量差异
- 在更多 site 选择下是否找到了更稳定的结构
- 计算吸附能并与 DFT target 对比，评估 Success Rate

重要：统计的是每个 level 下 44 个 SID 各自选出的最优位点的 success rate，
而不是 vasp_level_* 目录下的文件数。
"""

import os
import sys
import json
import numpy as np
import pickle
from collections import defaultdict

# Ensure project root is on PYTHONPATH
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import ase.io
from ase.io.vasp import read_vasp_out

# ==================== 配置区域 ====================

BASE_PATH = "grid_search_runs/2025-12-17-19-22-40-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_epoch0180_unweightedvalloss1.4265_posmae0.6214/val_nonrelaxed_update"
NSITES_DIR = "nsites_10"
CFG_DIR = "cfg3_steps10"

PATH = os.path.join(BASE_PATH, NSITES_DIR, CFG_DIR)

SITE_LEVELS = [1, 2, 5, 10]

# 参考能量和目标文件路径
REF_ENERGIES_PATH = "oc20_dense_mappings/oc20dense_ref_energies.pkl"
TARGETS_PATH = "oc20_dense_mappings/oc20dense_targets.pkl"

# Success 判定阈值 (eV)
SUCCESS_THRESHOLD = 0.1

# ==================== 加载参考数据 ====================

def load_reference_data():
    """加载参考能量和目标值"""
    print("Loading reference energies...")
    with open(REF_ENERGIES_PATH, "rb") as f:
        ref_energies = pickle.load(f)

    print("Loading targets...")
    with open(TARGETS_PATH, "rb") as f:
        targets = pickle.load(f)
        # 如果是 dict of dict，转换为最小能量
        if isinstance(next(iter(targets.values())), dict):
            converted = {}
            for sid, candidates in targets.items():
                energies = [val for val in candidates.values() if isinstance(val, (int, float))]
                if energies:
                    converted[sid] = min(energies)
            targets = converted

    return ref_energies, targets


def load_multisite_selection(base_path):
    """
    加载 multisite_vasp_results.json，获取每个 level 每个 SID 选择的 site

    Returns:
        dict: {level: {sid: {"site_idx": ..., "status": ...}}}
    """
    results_file = os.path.join(base_path, "multisite_vasp_results.json")
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return {}

    with open(results_file, "r") as f:
        data = json.load(f)

    # 转换格式: {level_str: [list of items]} -> {level_int: {sid: item}}
    selection = {}
    for level_str, items in data.items():
        level = int(level_str)
        selection[level] = {}
        for item in items:
            sid = item["sid"]
            selection[level][sid] = item

    return selection

# ==================== 分析函数 ====================

def parse_vasp_energy(outcar_path):
    """从 OUTCAR 解析能量"""
    try:
        atoms = read_vasp_out(outcar_path)
        return atoms.get_potential_energy()
    except Exception as e:
        print(f"Error reading {outcar_path}: {e}")
        return None


def collect_all_vasp_results(base_path, levels, ref_energies, targets):
    """
    收集所有层级目录下的 VASP 结果

    由于目录命名是 {sid}_{fid}（不包含 site_idx），无法从目录名解析 site_idx。
    所以我们只收集每个 sid 在每个 level 的 VASP 结果。

    Returns:
        dict: {(sid, level): {"energy": ..., "ads_energy": ..., ...}}
    """
    all_results = {}

    for level in levels:
        level_dir = os.path.join(base_path, f"vasp_level_{level}")
        if not os.path.exists(level_dir):
            continue

        for entry in os.listdir(level_dir):
            entry_path = os.path.join(level_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            outcar = os.path.join(entry_path, "OUTCAR")
            if not os.path.exists(outcar):
                continue

            # 解析 sid: entry 格式为 {sid}_{fid} 如 "65_2771_212_0"
            # sid 可能包含多个下划线，fid 是最后一个部分
            parts = entry.rsplit("_", 1)
            if len(parts) == 2:
                sid = parts[0]
            else:
                sid = entry

            energy = parse_vasp_energy(outcar)
            if energy is not None:
                result_entry = {
                    "energy": energy,
                    "path": entry_path,
                    "entry": entry,
                    "level": level,
                }

                # 计算吸附能
                if sid in ref_energies:
                    ref_e = ref_energies[sid]
                    ads_energy = energy - ref_e
                    result_entry["ref_energy"] = ref_e
                    result_entry["ads_energy"] = ads_energy

                    # 与 target 对比
                    if sid in targets:
                        target_e = targets[sid]
                        diff = ads_energy - target_e
                        result_entry["target"] = target_e
                        result_entry["diff"] = diff
                        result_entry["success"] = diff <= SUCCESS_THRESHOLD
                    else:
                        result_entry["target"] = None
                        result_entry["diff"] = None
                        result_entry["success"] = None
                else:
                    result_entry["ref_energy"] = None
                    result_entry["ads_energy"] = None

                # 使用 (sid, level) 作为 key
                all_results[(sid, level)] = result_entry

    print(f"Collected {len(all_results)} VASP results across all levels")
    return all_results


def build_level_results(selection, all_vasp_results, levels):
    """
    根据每个 level 的选择，构建每个 level 的 44 个 SID 累积结果

    关键逻辑：
    - Level 1: 只看 vasp_level_1 的结果
    - Level 2: 累积 vasp_level_1 和 vasp_level_2 的结果（只要有一个成功就算成功）
    - Level 5: 累积 vasp_level_1, 2, 5 的结果
    - Level 10: 累积 vasp_level_1, 2, 5, 10 的结果

    注意：由于去重机制，如果 Level 2 选的 site 与 Level 1 相同，则不会生成新的 VASP 目录，
    此时 Level 2 应该复用 Level 1 的结果。

    Args:
        selection: 从 multisite_vasp_results.json 加载的选择信息
        all_vasp_results: 所有 VASP 结果 {(sid, level): {...}}
        levels: [1, 2, 5, 10]

    Returns:
        dict: {level: {sid: {"energy": ..., "ads_energy": ..., "success": ..., ...}}}
    """
    results = {}

    # 收集所有 SID
    all_sids = set()
    for level in levels:
        if level in selection:
            all_sids.update(selection[level].keys())

    for level in levels:
        level_results = {}

        # 累积当前 level 及之前所有 level
        relevant_levels = [l for l in levels if l <= level]

        for sid in all_sids:
            # 收集该 SID 在所有相关 level 的 VASP 结果
            vasp_results_for_sid = []
            for l in relevant_levels:
                key = (sid, l)
                if key in all_vasp_results:
                    vasp_results_for_sid.append({
                        "level": l,
                        "result": all_vasp_results[key]
                    })

            if not vasp_results_for_sid:
                # 没有任何 VASP 结果
                level_results[sid] = {
                    "contributing_levels": [],
                    "energy": None,
                    "ads_energy": None,
                    "success": None,
                    "status": "no_vasp_result"
                }
                continue

            # 找到能量最低的结果（作为该 level 的代表）
            best_result = min(vasp_results_for_sid,
                              key=lambda x: x["result"].get("ads_energy", float('inf'))
                              if x["result"].get("ads_energy") is not None else float('inf'))

            # 检查是否有任一 VASP 结果成功
            any_success = any(
                r["result"].get("success") == True
                for r in vasp_results_for_sid
            )

            # 收集所有 diff
            diffs = [r["result"]["diff"] for r in vasp_results_for_sid
                     if r["result"].get("diff") is not None]
            min_diff = min(diffs) if diffs else None

            level_results[sid] = {
                "contributing_levels": [r["level"] for r in vasp_results_for_sid],
                "num_vasp_results": len(vasp_results_for_sid),
                "energy": best_result["result"].get("energy"),
                "ads_energy": best_result["result"].get("ads_energy"),
                "best_from_level": best_result["level"],
                "target": best_result["result"].get("target"),
                "diff": min_diff,  # 使用最小 diff
                "success": any_success,  # 只要有一个成功就算成功
            }

        results[level] = level_results

        # 统计
        total = len(level_results)
        with_results = sum(1 for v in level_results.values() if v.get("energy") is not None)
        success_count = sum(1 for v in level_results.values() if v.get("success") == True)
        avg_vasp_count = np.mean([v.get("num_vasp_results", 0) for v in level_results.values()])
        print(f"Level {level}: {total} SIDs, {with_results} with VASP results, "
              f"{success_count} cumulative success, avg {avg_vasp_count:.1f} VASP results per SID")

    return results


def compare_levels(results):
    """
    比较不同层级的结果

    分析在增加 site 数量后是否找到了更低能量的结构
    """
    levels = sorted(results.keys())

    # 收集所有 sid
    all_sids = set()
    for level in levels:
        all_sids.update(results[level].keys())

    comparison = []

    for sid in sorted(all_sids):
        sid_data = {"sid": sid}

        for level in levels:
            if sid in results[level]:
                sid_data[f"energy_level_{level}"] = results[level][sid]["energy"]
                sid_data[f"ads_energy_level_{level}"] = results[level][sid].get("ads_energy")
                sid_data[f"success_level_{level}"] = results[level][sid].get("success")
                sid_data[f"diff_level_{level}"] = results[level][sid].get("diff")
            else:
                sid_data[f"energy_level_{level}"] = None
                sid_data[f"ads_energy_level_{level}"] = None
                sid_data[f"success_level_{level}"] = None
                sid_data[f"diff_level_{level}"] = None

        # 获取 target
        for level in levels:
            if sid in results[level] and results[level][sid].get("target") is not None:
                sid_data["target"] = results[level][sid]["target"]
                break

        # 计算能量改善
        energies = []
        for level in levels:
            e = sid_data.get(f"energy_level_{level}")
            if e is not None:
                energies.append((level, e))

        if len(energies) >= 2:
            # 找到最低能量及对应的层级
            min_level, min_energy = min(energies, key=lambda x: x[1])
            first_level, first_energy = energies[0]

            sid_data["best_level"] = min_level
            sid_data["energy_improvement"] = first_energy - min_energy  # 正值表示改善
            sid_data["improved"] = min_level > first_level  # 是否在更高层级找到更好结果

        comparison.append(sid_data)

    return comparison


def generate_report(comparison, levels, results):
    """生成分析报告"""

    print("\n" + "="*80)
    print("MULTI-SITE VASP ANALYSIS REPORT")
    print("="*80)

    # 基本统计
    total = len(comparison)
    improved_count = sum(1 for c in comparison if c.get("improved", False))

    print(f"\nTotal systems analyzed: {total}")
    print(f"Systems with energy improvement at higher site levels: {improved_count} ({100*improved_count/total:.1f}%)")

    # ==================== Success Rate 分析 ====================
    print("\n" + "="*80)
    print("SUCCESS RATE ANALYSIS (Threshold: %.2f eV)" % SUCCESS_THRESHOLD)
    print("="*80)

    print("\n{:<10} | {:>10} | {:>10} | {:>12} | {:>10}".format(
        "Level", "Total", "Success", "Success Rate", "Avg Diff"))
    print("-" * 60)

    success_by_level = {}
    for level in levels:
        level_data = results[level]
        total_with_target = sum(1 for d in level_data.values() if d.get("success") is not None)
        success_count = sum(1 for d in level_data.values() if d.get("success") == True)
        diffs = [d["diff"] for d in level_data.values() if d.get("diff") is not None]
        avg_diff = np.mean(diffs) if diffs else float('nan')

        success_rate = 100 * success_count / total_with_target if total_with_target > 0 else 0
        success_by_level[level] = {
            "total": total_with_target,
            "success": success_count,
            "rate": success_rate,
            "avg_diff": avg_diff
        }

        print("{:<10} | {:>10} | {:>10} | {:>11.1f}% | {:>10.4f}".format(
            f"Level {level}", total_with_target, success_count, success_rate, avg_diff))

    # ==================== 详细对比表格 ====================
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80)

    # 表头
    header = "{:<20} | {:>10}".format("SID", "Target")
    for level in levels:
        header += " | {:>12}".format(f"L{level} AdsE")
    header += " | {:>10}".format("Best")
    print(header)
    print("-" * (25 + 15 * len(levels) + 15))

    # 按 target 排序输出
    sorted_comparison = sorted(comparison, key=lambda x: x.get("target", float('inf')) if x.get("target") is not None else float('inf'))

    for c in sorted_comparison[:30]:  # 只显示前30个
        target = c.get("target")
        target_str = f"{target:.4f}" if target is not None else "N/A"

        row = "{:<20} | {:>10}".format(c["sid"], target_str)

        best_level = c.get("best_level", 1)
        for level in levels:
            ads_e = c.get(f"ads_energy_level_{level}")
            success = c.get(f"success_level_{level}")
            if ads_e is not None:
                mark = "✓" if success else "✗"
                row += " | {:>10.4f} {}".format(ads_e, mark)
            else:
                row += " | {:>12}".format("-")

        row += " | {:>10}".format(f"L{best_level}")
        print(row)

    if len(comparison) > 30:
        print(f"... ({len(comparison) - 30} more systems)")

    # 层级覆盖统计
    print("\n" + "-"*60)
    print("Level coverage:")
    for level in levels:
        count = sum(1 for c in comparison if c.get(f"energy_level_{level}") is not None)
        print(f"  Level {level}: {count} systems ({100*count/total:.1f}%)")

    # 最佳层级分布
    best_level_dist = defaultdict(int)
    for c in comparison:
        if "best_level" in c:
            best_level_dist[c["best_level"]] += 1

    print("\nBest level distribution (where lowest energy was found):")
    for level in levels:
        count = best_level_dist.get(level, 0)
        print(f"  Level {level}: {count} systems ({100*count/total:.1f}%)")

    # 能量改善统计
    improvements = [c["energy_improvement"] for c in comparison if "energy_improvement" in c]
    if improvements:
        print("\nEnergy improvement statistics (first level -> best level):")
        print(f"  Mean: {np.mean(improvements):.4f} eV")
        print(f"  Max: {np.max(improvements):.4f} eV")
        print(f"  Median: {np.median(improvements):.4f} eV")

    # 显示改善最大的案例
    improved_cases = [c for c in comparison if c.get("improved", False)]
    if improved_cases:
        print(f"\nTop 10 systems with largest energy improvement:")
        improved_cases.sort(key=lambda x: x.get("energy_improvement", 0), reverse=True)
        for c in improved_cases[:10]:
            print(f"  {c['sid']}: {c['energy_improvement']:.4f} eV (best at level {c['best_level']})")

    return {
        "total": total,
        "improved_count": improved_count,
        "best_level_distribution": dict(best_level_dist),
        "mean_improvement": float(np.mean(improvements)) if improvements else None,
        "max_improvement": float(np.max(improvements)) if improvements else None,
        "success_by_level": success_by_level,
    }


def main():
    print(f"Analyzing results from: {PATH}")
    print(f"Site levels: {SITE_LEVELS}")

    # 加载参考数据
    ref_energies, targets = load_reference_data()
    print(f"Loaded {len(ref_energies)} reference energies, {len(targets)} targets")

    # 加载每个 level 的选择信息
    selection = load_multisite_selection(PATH)
    if not selection:
        print("ERROR: Could not load multisite selection data.")
        print("Please run write_vasp_inputs_multisite.py first.")
        return

    # 收集所有 VASP 结果
    all_vasp_results = collect_all_vasp_results(PATH, SITE_LEVELS, ref_energies, targets)

    # 根据选择构建每个 level 的结果（每个 level 44 个 SID）
    results = build_level_results(selection, all_vasp_results, SITE_LEVELS)

    # 比较层级
    comparison = compare_levels(results)

    # 生成报告
    summary = generate_report(comparison, SITE_LEVELS, results)

    # 保存结果
    output_file = os.path.join(PATH, "multisite_analysis.json")
    output_data = {
        "summary": summary,
        "comparison": comparison,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nFull analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
