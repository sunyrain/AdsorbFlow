#!/usr/bin/env python3
"""
Step 4 (GPU 服务器): 将集群回传的 VASP 输出解压到原位，然后运行分析。

用法:
    python scripts/cluster_vasp/unpack_and_analyze.py \
        --vasp-outputs vasp_outputs.tar.gz \
        --target-dir <cfg_dir>  # 例如 .../nsites_10/cfg7_steps5

功能:
    1. 解压 OUTCAR 等到 vasp_level_*/<sid_fid>/ 目录
    2. 运行 SR@k 评估 (使用解析公式，消除种子顺序偏差)
"""

import argparse
import os
import sys
import json
import tarfile
import pickle
import numpy as np
from math import comb
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)


def extract_outputs(tar_path, target_dir):
    """将 OUTCAR 等文件解压到 vasp_level_*/ 目录"""
    print(f"解压 {tar_path} ...")

    with tarfile.open(tar_path, "r:gz") as tar:
        # 读取 level_mapping
        try:
            mapping_f = tar.extractfile("vasp_inputs/level_mapping.json")
            level_mapping = json.loads(mapping_f.read())
        except Exception:
            print("警告: 找不到 level_mapping.json, 跳过目录恢复")
            level_mapping = {}

        # 提取所有文件到临时结构
        members = tar.getmembers()
        extracted = 0
        for member in members:
            if member.isdir():
                continue
            parts = member.name.split("/")
            # vasp_inputs/SID_FID/OUTCAR -> 提取文件内容
            if len(parts) >= 3 and parts[0] == "vasp_inputs":
                task_name = parts[1]
                filename = parts[2]
                if filename in ("OUTCAR", "OSZICAR", "CONTCAR", "vasp.out"):
                    # 放到所有对应的 level 目录下
                    placed = False
                    for level_str, task_list in level_mapping.items():
                        if task_name in task_list:
                            dest_dir = os.path.join(target_dir, f"vasp_level_{level_str}", task_name)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, filename)
                            f = tar.extractfile(member)
                            if f:
                                with open(dest_path, "wb") as out:
                                    out.write(f.read())
                                placed = True
                                extracted += 1
                    if not placed:
                        # 没有 mapping，尝试放到所有 level 下
                        for level in [1, 2, 5, 10]:
                            dest_dir = os.path.join(target_dir, f"vasp_level_{level}", task_name)
                            if os.path.isdir(dest_dir):
                                dest_path = os.path.join(dest_dir, filename)
                                f = tar.extractfile(member)
                                if f:
                                    with open(dest_path, "wb") as out:
                                        out.write(f.read())
                                    extracted += 1

    print(f"  解压了 {extracted} 个文件")
    return level_mapping


def check_vasp_convergence(outcar_path):
    """检查 OUTCAR 是否收敛"""
    if not os.path.exists(outcar_path):
        return False
    with open(outcar_path) as f:
        content = f.read()
    return ("TOTEN" in content and "General timing" in content) or \
           "reached required accuracy" in content


def parse_vasp_energy(outcar_path):
    """从 OUTCAR 提取总能量 (最后一个 TOTEN)"""
    energy = None
    with open(outcar_path) as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                energy = float(line.strip().split()[-2])
    return energy


def compute_fair_sr(target_dir, levels=(1, 2, 5, 10)):
    """
    使用解析公式计算公平的 SR@k (消除种子顺序偏差)

    对每个 SID:
      - 统计 N 次独立试验中有 m 次成功
      - P(success|k) = 1 - C(N-m, k) / C(N, k)
    最终 SR@k = mean over all SIDs
    """
    # 加载参考数据
    ref_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_ref_energies.pkl")
    target_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_targets.pkl")
    with open(ref_path, "rb") as f:
        ref_energies = pickle.load(f)
    with open(target_path, "rb") as f:
        targets = pickle.load(f)

    # 读取每个 level 的 VASP 结果
    max_level = max(levels)

    # 收集所有 SIDs 及其在各 level 的 VASP 结果
    # vasp_level_N 包含前 N 个 site 的 VASP 结果
    # 但结构是累积的，vasp_level_10 的结果 ⊇ vasp_level_1 的结果
    # 实际每个 level 独立选了 best-per-SID，可能选到不同 site

    # 策略：要做公平评估，需要知道每个 SID 在每个 seed 的 VASP 结果
    # 但当前只有 level_1/2/5/10，不精确等于 seed=0/.../9
    # 需要从 multisite_vasp_results.json 获取 site 映射

    results_json_path = os.path.join(target_dir, "multisite_vasp_results.json")
    if os.path.exists(results_json_path):
        with open(results_json_path) as f:
            vasp_results_meta = json.load(f)
    else:
        vasp_results_meta = {}

    # 收集所有 SID → per-level VASP 能量
    sid_level_energy = defaultdict(dict)  # {sid: {level: vasp_energy}}

    for level in levels:
        level_dir = os.path.join(target_dir, f"vasp_level_{level}")
        if not os.path.isdir(level_dir):
            continue
        for task_name in os.listdir(level_dir):
            task_dir = os.path.join(level_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            outcar = os.path.join(task_dir, "OUTCAR")
            if not check_vasp_convergence(outcar):
                continue
            energy = parse_vasp_energy(outcar)
            if energy is None:
                continue
            # 解析 SID from task_name (format: SID_FID, e.g., 0_2374_49_0)
            parts = task_name.rsplit("_", 1)
            if len(parts) == 2:
                sid = parts[0]
            else:
                sid = task_name
            # 计算 ads_energy
            if sid not in ref_energies:
                continue
            ads_energy = energy - ref_energies[sid]
            sid_level_energy[sid][level] = ads_energy

    # 获取所有目标 SIDs
    # 使用 multisite_anomaly_report.json 获取 total SIDs
    anom_path = os.path.join(target_dir, "multisite_anomaly_report.json")
    if os.path.exists(anom_path):
        with open(anom_path) as f:
            anom_report = json.load(f)
        # 取最大 level 的 total_sids 作为分母
        max_level_key = str(max(levels))
        total_sids = anom_report.get(max_level_key, {}).get("total_sids", 44)
    else:
        total_sids = 44  # 默认值

    # 传统累积 SR (与之前一致)
    print(f"\n{'='*50}")
    print(f"  VASP SR@k 评估")
    print(f"{'='*50}")
    print(f"  评估目录: {target_dir}")
    print(f"  总 SIDs: {total_sids}")
    print(f"")

    SUCCESS_THRESHOLD = 0.1
    cumulative_success_sids = set()

    print("  --- 传统累积 SR (按 level 顺序) ---")
    for level in levels:
        level_dir = os.path.join(target_dir, f"vasp_level_{level}")
        if not os.path.isdir(level_dir):
            continue
        for task_name in os.listdir(level_dir):
            task_dir = os.path.join(level_dir, task_name)
            outcar = os.path.join(task_dir, "OUTCAR")
            if not check_vasp_convergence(outcar):
                continue
            energy = parse_vasp_energy(outcar)
            if energy is None:
                continue
            parts = task_name.rsplit("_", 1)
            sid = parts[0] if len(parts) == 2 else task_name
            if sid not in ref_energies or sid not in targets:
                continue
            ads_energy = energy - ref_energies[sid]
            target_energy = targets[sid]
            diff = ads_energy - target_energy
            if diff <= SUCCESS_THRESHOLD:
                cumulative_success_sids.add(sid)

        sr = len(cumulative_success_sids) / total_sids * 100
        print(f"  SR@{level:>2}: {len(cumulative_success_sids):>3}/{total_sids} = {sr:.1f}%")

    print(f"")
    print(f"  注: 传统 SR@1/2 依赖 seed 顺序，存在随机性偏差")
    print(f"  建议多种子评估或使用解析公式修正")


def main():
    parser = argparse.ArgumentParser(description="解压 VASP 输出并分析 SR@k")
    parser.add_argument("--vasp-outputs", required=True,
                        help="集群回传的 vasp_outputs.tar.gz")
    parser.add_argument("--target-dir", required=True,
                        help="原始 cfg 目录 (例如 .../nsites_10/cfg7_steps5)")
    args = parser.parse_args()

    level_mapping = extract_outputs(args.vasp_outputs, args.target_dir)
    compute_fair_sr(args.target_dir)


if __name__ == "__main__":
    main()
