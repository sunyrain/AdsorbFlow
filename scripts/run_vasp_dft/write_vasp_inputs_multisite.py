"""
多层次 site 选择的 VASP 输入生成脚本

对于 nsites_N (如 nsites_10)，分别对 1, 2, 5, 10 个 site 取最小能量结构：
- site_1: 只考虑 site 0，选出每个 sid 的最优结构
- site_2: 考虑 site 0-1，选出每个 sid 的最优结构
- site_5: 考虑 site 0-4，选出每个 sid 的最优结构
- site_10: 考虑 site 0-9，选出每个 sid 的最优结构

去重逻辑：如果某个 (sid, traj_file) 已在更少 site 数下被选中，则不重复生成 VASP 输入。
"""

import os
import sys

# Ensure project root (AdsorbFlow/) is on PYTHONPATH when running via a script path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["VASP_PP_PATH"] = "potpaw_PBE_54"

import numpy as np
import ase.io
from tqdm import tqdm
import glob
import pickle
import json
from collections import defaultdict

sys.path.append("Open-Catalyst-Dataset")
from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement import DetectTrajAnomaly

# ==================== 配置区域 ====================

# 基础路径 - 指向 nsites_N 的上级目录
BASE_PATH = "grid_search_runs/2025-12-17-19-22-40-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_epoch0180_unweightedvalloss1.4265_posmae0.6214/val_nonrelaxed_update"

# 要处理的 nsites 目录
NSITES_DIR = "nsites_10"
CFG_DIR = "cfg3_steps10"

# 输入路径
TRAJ_INPUT_PATH = os.path.join(BASE_PATH, NSITES_DIR, CFG_DIR)

# 输出路径
EXPORT_PATH = "vasp_cluster_inputs"

# 要分析的 site 数量层级
SITE_LEVELS = [1, 2, 5, 10]

# tags 文件路径
TAG_PATH = "oc20_dense_mappings/oc20dense_tags.pkl"

VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 0,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": True,
    "lwave": False,
    "ncore": 4,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
    "setups": "minimal",
}

# ==================== 加载数据 ====================

print("Loading tags map...")
with open(TAG_PATH, "rb") as h:
    tags_map = pickle.load(h)


# 异常类型名称
ANOMALY_TYPES = ["dissociated", "desorbed", "surface_changed", "intercalated"]


def _make_ads_contiguous(atoms):
    """
    Unwrap adsorbate atoms with minimum image so a molecule split across PBC
    doesn't look dissociated to the connectivity checker.
    (与 eval.py 保持一致)
    """
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


def anomalous_structure(traj, sid=None):
    """
    检测结构是否异常
    与 eval.py 保持一致的实现
    """
    initial_atoms = _make_ads_contiguous(traj[0])
    final_atoms = _make_ads_contiguous(traj[-1])
    # 从 traj 获取 tags（与 eval.py 一致）
    atom_tags = initial_atoms.get_tags()
    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anom = np.array([
        detector.is_adsorbate_dissociated(),
        detector.is_adsorbate_desorbed(),
        detector.has_surface_changed(),
        detector.is_adsorbate_intercalated(),
    ])
    return anom


def get_anomaly_details(traj, sid):
    """
    获取详细的异常信息

    Returns:
        dict: {"is_anomalous": bool, "types": [异常类型列表], "details": {类型: bool}}
    """
    anom = anomalous_structure(traj, sid)
    details = {t: bool(a) for t, a in zip(ANOMALY_TYPES, anom)}
    return {
        "is_anomalous": anom.any(),
        "types": [t for t, a in zip(ANOMALY_TYPES, anom) if a],
        "details": details,
    }


def get_traj_files_for_sites(base_path, site_indices):
    """
    获取指定 site 范围内的所有 traj 文件

    Args:
        base_path: cfg3_steps10 的路径
        site_indices: site 索引列表，如 [0], [0,1], [0,1,2,3,4] 等

    Returns:
        dict: {sid: [(traj_path, site_idx), ...]}
    """
    sid_to_trajs = defaultdict(list)

    for site_idx in site_indices:
        site_dir = os.path.join(base_path, str(site_idx))
        if not os.path.exists(site_dir):
            print(f"Warning: site directory {site_dir} does not exist")
            continue

        # 优先从 relaxations 子目录读取（与 grid_search 的 eval 一致）
        relaxations_dir = os.path.join(site_dir, "relaxations")
        if os.path.exists(relaxations_dir):
            traj_files = glob.glob(os.path.join(relaxations_dir, "*.traj"))
        else:
            # fallback 到根目录
            traj_files = glob.glob(os.path.join(site_dir, "*.traj"))

        for traj_path in traj_files:
            # 解析 sid：文件名格式为 {sid}.traj 或 {sid}_{fid}.traj
            filename = os.path.basename(traj_path).replace(".traj", "")
            if filename.count("_") == 3:
                sid = "_".join(filename.split("_")[:-1])
            else:
                sid = filename

            sid_to_trajs[sid].append((traj_path, site_idx))

    return sid_to_trajs


def find_best_structure_with_anomaly_stats(sid, traj_list):
    """
    在给定的 traj 列表中找到最优结构，并返回其 anomaly 状态

    选择逻辑：
    1. 优先选择非 anomaly 的最低能量结构
    2. 如果所有结构都是 anomaly，则选择最低能量的 anomaly 结构

    Args:
        sid: system id
        traj_list: [(traj_path, site_idx), ...]

    Returns:
        tuple: (best_result, selected_anomaly_info)
        - best_result: (best_traj_path, site_idx, energy) or None
        - selected_anomaly_info: {
            "is_anomalous": bool,  # 被选中的最优结构是否是 anomaly
            "anomaly_types": [...],  # 如果是 anomaly，具体类型
            "all_candidates_anomalous": bool,  # 是否所有候选都是 anomaly
            "total_candidates": int,
            "normal_candidates": int,
            "anomalous_candidates": int,
          }
    """
    candidates = []

    for traj_path, site_idx in traj_list:
        try:
            atoms = ase.io.read(traj_path)
            energy = atoms.get_potential_energy()
            candidates.append((traj_path, site_idx, energy))
        except Exception as e:
            print(f"Warning: Failed to read {traj_path}: {e}")
            continue

    if not candidates:
        return None, {
            "is_anomalous": True,
            "anomaly_types": [],
            "all_candidates_anomalous": True,
            "total_candidates": 0,
            "normal_candidates": 0,
            "anomalous_candidates": 0,
        }

    # 按能量排序
    candidates.sort(key=lambda x: x[2])

    # 检查每个候选的 anomaly 状态
    normal_candidates = []
    anomalous_candidates = []

    for traj_path, site_idx, energy in candidates:
        try:
            traj = ase.io.read(traj_path, ":")
            anom_info = get_anomaly_details(traj, sid)

            if anom_info["is_anomalous"]:
                anomalous_candidates.append((traj_path, site_idx, energy, anom_info["types"]))
            else:
                normal_candidates.append((traj_path, site_idx, energy))

        except Exception as e:
            print(f"Warning: Failed to check anomaly for {traj_path}: {e}")
            continue

    # 选择最优结构：优先非 anomaly，否则选 anomaly 中能量最低的
    if normal_candidates:
        best_traj_path, best_site_idx, best_energy = normal_candidates[0]
        selected_anomaly_info = {
            "is_anomalous": False,
            "anomaly_types": [],
            "all_candidates_anomalous": False,
            "total_candidates": len(candidates),
            "normal_candidates": len(normal_candidates),
            "anomalous_candidates": len(anomalous_candidates),
        }
        return (best_traj_path, best_site_idx, best_energy), selected_anomaly_info
    elif anomalous_candidates:
        best_traj_path, best_site_idx, best_energy, anom_types = anomalous_candidates[0]
        selected_anomaly_info = {
            "is_anomalous": True,
            "anomaly_types": anom_types,
            "all_candidates_anomalous": True,
            "total_candidates": len(candidates),
            "normal_candidates": 0,
            "anomalous_candidates": len(anomalous_candidates),
        }
        return (best_traj_path, best_site_idx, best_energy), selected_anomaly_info
    else:
        return None, {
            "is_anomalous": True,
            "anomaly_types": [],
            "all_candidates_anomalous": True,
            "total_candidates": len(candidates),
            "normal_candidates": 0,
            "anomalous_candidates": 0,
        }


def main():
    print(f"Processing: {TRAJ_INPUT_PATH}")
    print(f"Site levels: {SITE_LEVELS}")

    # 检查可用的 site 目录
    available_sites = []
    for i in range(max(SITE_LEVELS)):
        site_dir = os.path.join(TRAJ_INPUT_PATH, str(i))
        if os.path.exists(site_dir):
            available_sites.append(i)

    print(f"Available sites: {available_sites}")

    # 调整 SITE_LEVELS，确保不超过可用数量
    valid_site_levels = [n for n in SITE_LEVELS if n <= len(available_sites)]
    print(f"Valid site levels: {valid_site_levels}")

    # 存储每个层级的结果
    results_by_level = {}

    # 存储已生成的 (sid, traj_path) 组合，用于去重
    generated_structures = set()

    # 统计信息
    stats = {
        "total_sids": set(),
        "by_level": {},
    }

    # 异常统计信息
    anomaly_stats_by_level = {}

    for num_sites in valid_site_levels:
        print(f"\n{'='*60}")
        print(f"Processing site level: {num_sites} (sites 0-{num_sites-1})")
        print(f"{'='*60}")

        site_indices = list(range(num_sites))
        sid_to_trajs = get_traj_files_for_sites(TRAJ_INPUT_PATH, site_indices)

        level_results = []
        new_structures_count = 0
        skipped_count = 0

        # 被选中结构的 anomaly 统计（每个 sid 选一个最优）
        level_selected_anomaly = {
            "total_sids": 0,
            "selected_anomalous": 0,  # 被选中的最优结构是 anomaly 的数量
            "selected_normal": 0,  # 被选中的最优结构是 normal 的数量
            "by_type": {t: 0 for t in ANOMALY_TYPES},  # 被选中 anomaly 结构的类型分布
            "anomalous_sids": [],  # 最优结构是 anomaly 的 sid 列表
        }

        for sid in tqdm(sid_to_trajs.keys(), desc=f"Finding best for {num_sites} sites"):
            traj_list = sid_to_trajs[sid]
            best, selected_anomaly_info = find_best_structure_with_anomaly_stats(sid, traj_list)

            level_selected_anomaly["total_sids"] += 1

            # 统计被选中结构的 anomaly 状态
            if selected_anomaly_info["is_anomalous"]:
                level_selected_anomaly["selected_anomalous"] += 1
                level_selected_anomaly["anomalous_sids"].append(sid)
                for anom_type in selected_anomaly_info["anomaly_types"]:
                    level_selected_anomaly["by_type"][anom_type] += 1
            else:
                level_selected_anomaly["selected_normal"] += 1

            if best is None:
                print(f"Warning: No structure found for {sid} at level {num_sites}")
                continue

            traj_path, site_idx, energy = best
            stats["total_sids"].add(sid)

            # 检查是否已生成（去重）- 基于 (sid, site_idx) 因为同一个 sid 的不同 site 有相同的 traj 文件名
            structure_key = (sid, site_idx)

            if structure_key in generated_structures:
                skipped_count += 1
                level_results.append({
                    "sid": sid,
                    "traj_path": traj_path,
                    "site_idx": site_idx,
                    "energy": energy,
                    "status": "skipped_duplicate",
                })
                continue

            # 标记为已生成
            generated_structures.add(structure_key)
            new_structures_count += 1

            # 读取结构并生成 VASP 输入
            try:
                traj = ase.io.read(traj_path, ":")
                relaxed_struct = traj[-1]

                # 设置约束
                tags = tags_map[sid]
                fixed_atoms = np.where(tags == 2)[0]
                relaxed_struct.set_constraint(ase.constraints.FixAtoms(fixed_atoms))

                # 解析 fid
                filename = os.path.basename(traj_path).replace(".traj", "")
                if filename.count("_") == 3:
                    fid = filename.split("_")[-1]
                else:
                    fid = 0

                # 创建输出目录名：包含 level 信息以便追踪
                output_name = f"{sid}_{fid}"

                # 1. 导出到原始位置 (按 level 分组)
                level_vasp_dir = os.path.join(TRAJ_INPUT_PATH, f"vasp_level_{num_sites}")
                os.makedirs(level_vasp_dir, exist_ok=True)
                write_vasp_input_files(
                    relaxed_struct,
                    outdir=os.path.join(level_vasp_dir, output_name),
                    vasp_flags=VASP_FLAGS,
                )

                # 2. 导出到集群目录
                export_dir = os.path.join(EXPORT_PATH, output_name)
                os.makedirs(export_dir, exist_ok=True)
                write_vasp_input_files(
                    relaxed_struct,
                    outdir=export_dir,
                    vasp_flags=VASP_FLAGS,
                )

                level_results.append({
                    "sid": sid,
                    "traj_path": traj_path,
                    "site_idx": site_idx,
                    "energy": energy,
                    "status": "generated",
                    "output_dir": export_dir,
                })

            except Exception as e:
                print(f"Error processing {sid}: {e}")
                level_results.append({
                    "sid": sid,
                    "traj_path": traj_path,
                    "site_idx": site_idx,
                    "energy": energy,
                    "status": "error",
                    "error": str(e),
                })

        results_by_level[num_sites] = level_results

        # 计算被选中结构的 anomaly 率
        total_sids = level_selected_anomaly["total_sids"]
        selected_anomaly_rate = level_selected_anomaly["selected_anomalous"] / total_sids if total_sids > 0 else 0
        anomaly_stats_by_level[num_sites] = level_selected_anomaly

        stats["by_level"][num_sites] = {
            "total_sids": len(sid_to_trajs),
            "new_structures": new_structures_count,
            "skipped_duplicates": skipped_count,
            "errors": len([r for r in level_results if r.get("status") == "error"]),
            "selected_anomaly_stats": {
                "total_sids": total_sids,
                "selected_anomalous": level_selected_anomaly["selected_anomalous"],
                "selected_normal": level_selected_anomaly["selected_normal"],
                "anomaly_rate": selected_anomaly_rate,
                "by_type": level_selected_anomaly["by_type"],
            },
        }

        print(f"\nLevel {num_sites} summary:")
        print(f"  Total SIDs: {len(sid_to_trajs)}")
        print(f"  New structures generated: {new_structures_count}")
        print(f"  Skipped (duplicates): {skipped_count}")
        print(f"  Selected structure anomaly statistics (每个 sid 选出的最优结构):")
        print(f"    Total SIDs: {total_sids}")
        print(f"    Selected Anomalous: {level_selected_anomaly['selected_anomalous']} ({100*selected_anomaly_rate:.1f}%)")
        print(f"    Selected Normal: {level_selected_anomaly['selected_normal']} ({100*(1-selected_anomaly_rate):.1f}%)")
        if level_selected_anomaly["selected_anomalous"] > 0:
            print(f"    Anomaly types in selected structures:")
            for anom_type in ANOMALY_TYPES:
                count = level_selected_anomaly['by_type'][anom_type]
                if count > 0:
                    print(f"      {anom_type}: {count}")

    # 保存统计结果
    stats_file = os.path.join(TRAJ_INPUT_PATH, "multisite_vasp_stats.json")
    stats["total_unique_sids"] = len(stats["total_sids"])
    stats["total_sids"] = list(stats["total_sids"])  # 转换为 list 以便 JSON 序列化
    stats["total_unique_structures"] = len(generated_structures)

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    # 保存详细结果
    results_file = os.path.join(TRAJ_INPUT_PATH, "multisite_vasp_results.json")

    # 转换 results_by_level 的 key 为字符串
    results_json = {str(k): v for k, v in results_by_level.items()}
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Detailed results saved to: {results_file}")

    # 打印总结
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total unique SIDs processed: {len(stats['total_sids'])}")
    print(f"Total unique structures to calculate: {len(generated_structures)}")
    print("\nBreakdown by level:")
    for level in valid_site_levels:
        level_stats = stats["by_level"][level]
        print(f"  Level {level:2d}: {level_stats['new_structures']:3d} new, {level_stats['skipped_duplicates']:3d} skipped")

    # 打印被选中结构的 anomaly 统计汇总表
    print("\n" + "="*70)
    print("SELECTED STRUCTURE ANOMALY SUMMARY (每个 level 选出的最优结构)")
    print("="*70)
    print(f"{'Level':>6} | {'Total SIDs':>10} | {'Sel. Normal':>11} | {'Sel. Anomaly':>12} | {'Anomaly Rate':>12}")
    print("-" * 70)
    for level in valid_site_levels:
        sel_stats = stats["by_level"][level]["selected_anomaly_stats"]
        print(f"{level:>6} | {sel_stats['total_sids']:>10} | {sel_stats['selected_normal']:>11} | {sel_stats['selected_anomalous']:>12} | {100*sel_stats['anomaly_rate']:>11.1f}%")

    # 按类型统计
    print("\nSelected anomaly by type across levels:")
    print(f"{'Level':>6} | {'dissociated':>12} | {'desorbed':>10} | {'surf_changed':>12} | {'intercalated':>12}")
    print("-" * 70)
    for level in valid_site_levels:
        by_type = stats["by_level"][level]["selected_anomaly_stats"]["by_type"]
        print(f"{level:>6} | {by_type['dissociated']:>12} | {by_type['desorbed']:>10} | {by_type['surface_changed']:>12} | {by_type['intercalated']:>12}")

    print(f"\nVASP inputs exported to:")
    print(f"  - Original location: {TRAJ_INPUT_PATH}/vasp_level_*")
    print(f"  - Cluster export: {EXPORT_PATH}")

    # 保存详细异常统计
    anomaly_report_file = os.path.join(TRAJ_INPUT_PATH, "multisite_anomaly_report.json")
    anomaly_report = {}
    for level in valid_site_levels:
        level_anom = anomaly_stats_by_level[level].copy()
        anomaly_report[str(level)] = level_anom

    with open(anomaly_report_file, "w") as f:
        json.dump(anomaly_report, f, indent=2)
    print(f"\nAnomaly report saved to: {anomaly_report_file}")


if __name__ == "__main__":
    main()
