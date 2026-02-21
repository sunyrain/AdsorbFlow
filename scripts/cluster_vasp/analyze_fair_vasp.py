#!/usr/bin/env python3
"""
分析 Fair VASP SP 结果，计算 Fair SR@k。

用法:
    python scripts/cluster_vasp/analyze_fair_vasp.py \
        --vasp-dir vasp_fair_all2d/vasp_fair_work/vasp_fair \
        --stats vasp_fair_all2d/generation_stats.json
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
from math import comb
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

SUCCESS_THRESHOLD = 0.1


def parse_vasp_energy(outcar_path):
    """从 OUTCAR 提取总能量 (最后一个 TOTEN)"""
    energy = None
    with open(outcar_path) as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                energy = float(line.strip().split()[-2])
    return energy


def check_vasp_convergence(outcar_path):
    """检查 OUTCAR 是否收敛"""
    if not os.path.exists(outcar_path):
        return False
    with open(outcar_path) as f:
        content = f.read()
    return ("TOTEN" in content and "General timing" in content) or \
           "reached required accuracy" in content


def analytical_sr_at_k(sid_seed_success, total_sids, k, N=10):
    """
    P(SID成功|k) = 1 - C(N-m, k) / C(N, k)
    SR@k = mean over all SIDs
    """
    total_prob = 0.0
    for sid, successes in sid_seed_success.items():
        m = sum(successes)
        if m == 0:
            p = 0.0
        elif N - m < k:
            p = 1.0
        else:
            p = 1.0 - comb(N - m, k) / comb(N, k)
        total_prob += p
    # SIDs without any VASP result count as failures (0 prob)
    return total_prob / total_sids * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vasp-dir", required=True,
                        help="VASP结果目录 (包含 {model}__{sid}__site{i} 子目录)")
    parser.add_argument("--stats", required=True,
                        help="generation_stats.json 路径")
    parser.add_argument("--nsites", type=int, default=10)
    args = parser.parse_args()

    # 加载参考数据
    ref_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_ref_energies.pkl")
    target_path = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_targets.pkl")
    with open(ref_path, "rb") as f:
        ref_energies = pickle.load(f)
    with open(target_path, "rb") as f:
        targets = pickle.load(f)
    # targets 已是 flat: {sid: energy}
    if isinstance(next(iter(targets.values())), dict):
        targets = {sid: min(v.values()) for sid, v in targets.items()}

    with open(args.stats) as f:
        gen_stats = json.load(f)

    nsites = args.nsites
    total_sids = 44  # OC20 dense val set

    # 扫描所有子目录，解析能量
    # 目录名格式: {model_key}__{sid}__site{idx}
    all_tasks = [d for d in os.listdir(args.vasp_dir)
                 if os.path.isdir(os.path.join(args.vasp_dir, d)) and "__site" in d]

    # 按 model 分组
    model_data = defaultdict(lambda: defaultdict(dict))
    # model_data[model_key][sid][site_idx] = ads_energy or None

    converged = 0
    failed = 0
    missing_ref = 0

    for task_name in sorted(all_tasks):
        # 解析: model_key__sid__site{idx}
        parts = task_name.split("__")
        if len(parts) != 3:
            continue
        model_key = parts[0]
        sid = parts[1]
        site_str = parts[2]  # "site0", "site1", ...
        site_idx = int(site_str.replace("site", ""))

        outcar = os.path.join(args.vasp_dir, task_name, "OUTCAR")
        if not os.path.exists(outcar):
            model_data[model_key][sid][site_idx] = None
            continue

        if not check_vasp_convergence(outcar):
            failed += 1
            model_data[model_key][sid][site_idx] = None
            continue

        energy = parse_vasp_energy(outcar)
        if energy is None:
            failed += 1
            model_data[model_key][sid][site_idx] = None
            continue

        if sid not in ref_energies:
            missing_ref += 1
            model_data[model_key][sid][site_idx] = None
            continue

        ads_energy = energy - ref_energies[sid]
        model_data[model_key][sid][site_idx] = ads_energy
        converged += 1

    print(f"{'='*60}")
    print(f"  Fair VASP SP 评估结果")
    print(f"{'='*60}")
    print(f"  VASP目录: {args.vasp_dir}")
    print(f"  收敛: {converged}, 失败/未完成: {failed}, 缺少ref: {missing_ref}")
    print(f"  总SIDs: {total_sids}, 每SID种子数: {nsites}")
    print()

    # 对每个模型分别计算
    for model_key in sorted(model_data.keys()):
        stats = gen_stats.get(model_key, {})
        label = stats.get("label", model_key)
        n_normal = stats.get("normal", 0)
        n_anomalous = stats.get("anomalous", 0)

        sid_results = model_data[model_key]

        # 构建 sid_seed_success: 每个 SID 的每个 seed 是否成功
        sid_seed_success = {}
        all_sids_for_model = set()

        # 收集所有出现的 SIDs
        for sid in sid_results:
            all_sids_for_model.add(sid)

        # 也要包含 anomalous 的 SIDs (它们没有 VASP 结果，全部视为失败)
        # 但我们不知道具体哪些是 anomalous 的 SID
        # 这些 SID 不会出现在 sid_results 中任何 site，所以不影响

        for sid in sorted(all_sids_for_model):
            successes = []
            for site_idx in range(nsites):
                ads_e = sid_results[sid].get(site_idx, None)
                if ads_e is not None:
                    target_e = targets.get(sid)
                    if target_e is not None:
                        diff = ads_e - target_e
                        successes.append(diff <= SUCCESS_THRESHOLD)
                    else:
                        successes.append(False)
                else:
                    # 没有 VASP 结果 → 异常/失败 → False
                    successes.append(False)
            sid_seed_success[sid] = successes

        # 统计
        n_sids_with_results = len(all_sids_for_model)

        print(f"  {'─'*56}")
        print(f"  模型: {label} ({model_key})")
        print(f"  {'─'*56}")
        print(f"  正常轨迹: {n_normal}, 异常: {n_anomalous}, 总: {n_normal + n_anomalous}")
        print(f"  有VASP结果的SIDs: {n_sids_with_results}/{total_sids}")

        # 每个SID的成功种子数分布
        m_dist = defaultdict(int)
        for sid, successes in sid_seed_success.items():
            m = sum(successes)
            m_dist[m] += 1

        print(f"\n  成功种子数分布 (m = 该SID有多少个seed成功):")
        for m in range(nsites + 1):
            if m_dist[m] > 0:
                bar = "█" * m_dist[m]
                print(f"    m={m:>2}: {m_dist[m]:>3} SIDs  {bar}")

        # SR@k
        print(f"\n  {'k':>5}  {'Fair SR@k':>12}  {'备注':>20}")
        print(f"  {'─'*5}  {'─'*12}  {'─'*20}")

        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if k > nsites:
                break
            fair_sr = analytical_sr_at_k(sid_seed_success, total_sids, k, nsites)
            note = ""
            if k == 1:
                note = "← 单次采样"
            elif k == 5:
                note = "← 5次采样"
            elif k == 10:
                note = "← 全部采样"
            print(f"  {k:>5}  {fair_sr:>11.2f}%  {note}")

        # 详细每个SID结果
        print(f"\n  每个SID详情:")
        print(f"  {'SID':>15}  {'m':>3}  {'最低ads_E':>10}  {'target':>10}  {'diff':>8}  {'seeds'}")
        print(f"  {'─'*15}  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")

        for sid in sorted(sid_seed_success.keys()):
            successes = sid_seed_success[sid]
            m = sum(successes)
            seeds_str = "".join(["✓" if s else "✗" for s in successes])

            # 最低能量
            energies = []
            for si in range(nsites):
                e = sid_results[sid].get(si)
                if e is not None:
                    energies.append(e)

            if energies:
                min_e = min(energies)
                target_e = targets.get(sid, float('nan'))
                diff = min_e - target_e
                print(f"  {sid:>15}  {m:>3}  {min_e:>10.4f}  {target_e:>10.4f}  {diff:>+8.4f}  {seeds_str}")
            else:
                print(f"  {sid:>15}  {m:>3}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}  {seeds_str}")

        print()

    # 汇总对比表
    print(f"\n{'='*60}")
    print(f"  模型对比汇总")
    print(f"{'='*60}")
    header = f"  {'模型':>20}"
    for k in [1, 2, 3, 5, 10]:
        header += f"  {'SR@'+str(k):>8}"
    print(header)
    print(f"  {'─'*20}" + f"  {'─'*8}" * 5)

    for model_key in sorted(model_data.keys()):
        label = gen_stats.get(model_key, {}).get("label", model_key)
        sid_results = model_data[model_key]
        all_sids_for_model = set(sid_results.keys())
        sid_seed_success = {}
        for sid in all_sids_for_model:
            successes = []
            for site_idx in range(nsites):
                ads_e = sid_results[sid].get(site_idx, None)
                if ads_e is not None:
                    target_e = targets.get(sid)
                    if target_e is not None:
                        successes.append((ads_e - target_e) <= SUCCESS_THRESHOLD)
                    else:
                        successes.append(False)
                else:
                    successes.append(False)
            sid_seed_success[sid] = successes

        row = f"  {label:>20}"
        for k in [1, 2, 3, 5, 10]:
            sr = analytical_sr_at_k(sid_seed_success, total_sids, k, nsites)
            row += f"  {sr:>7.2f}%"
        print(row)

    print()


if __name__ == "__main__":
    main()
