#!/usr/bin/env python3
"""
Step 1 (GPU 服务器): 将 VASP 输入文件打包为 tar.gz，用于传输到 CPU 集群。

用法:
    python scripts/cluster_vasp/pack_vasp_inputs.py \
        --vasp-dir <cfg_dir containing vasp_level_*> \
        --output vasp_inputs.tar.gz

打包内容 (按 level 独立保留，不同 level 选的结构可能不同):
    vasp_inputs/
        task_list.txt          # 全局任务列表（level/sid_fid 格式）
        level_mapping.json     # level → task 列表映射
        level_<K>/
            <sid_fid>/
                POSCAR, INCAR, POTCAR, KPOINTS

不包含任何 Python 依赖、traj 文件、模型权重。
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description="打包 VASP 输入文件到 tar.gz")
    parser.add_argument("--vasp-dir", required=True,
                        help="cfg 目录，包含 vasp_level_* 子目录")
    parser.add_argument("--output", default="vasp_inputs.tar.gz",
                        help="输出 tar.gz 路径 (默认: vasp_inputs.tar.gz)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 5, 10],
                        help="要打包的 level (默认: 1 2 5 10)")
    args = parser.parse_args()

    vasp_dir = os.path.abspath(args.vasp_dir)
    required_files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]

    # ---- 收集每个 level 的任务（不跨 level 去重）----
    level_tasks = {}          # level -> {task_name: task_path}
    all_poscar_hashes = {}    # (level, task_name) -> md5
    for level in sorted(args.levels):
        level_dir = os.path.join(vasp_dir, f"vasp_level_{level}")
        if not os.path.isdir(level_dir):
            print(f"警告: {level_dir} 不存在，跳过")
            continue
        tasks = {}
        for task_name in sorted(os.listdir(level_dir)):
            task_path = os.path.join(level_dir, task_name)
            if not os.path.isdir(task_path):
                continue
            if all(os.path.exists(os.path.join(task_path, f)) for f in required_files):
                tasks[task_name] = task_path
                with open(os.path.join(task_path, "POSCAR"), "rb") as pf:
                    all_poscar_hashes[(level, task_name)] = hashlib.md5(pf.read()).hexdigest()
            else:
                missing = [f for f in required_files if not os.path.exists(os.path.join(task_path, f))]
                print(f"警告: level {level}/{task_name} 缺少 {missing}，跳过")
        level_tasks[level] = tasks

    total_pairs = sum(len(t) for t in level_tasks.values())
    if total_pairs == 0:
        print("错误: 没有找到任何有效的 VASP 任务目录")
        sys.exit(1)

    # ---- 对重复 POSCAR 做智能去重（仅对完全相同的 POSCAR 共享） ----
    # 构建 (task_name, hash) -> 首次出现的 (level, task_path)
    unique_poscars = {}   # (task_name, hash) -> (level, task_path)
    # 记录每个 (level, task_name) 指向哪个唯一计算
    calc_id_map = {}      # (level, task_name) -> calc_id string
    for level in sorted(level_tasks):
        for task_name, task_path in sorted(level_tasks[level].items()):
            h = all_poscar_hashes[(level, task_name)]
            key = (task_name, h)
            if key not in unique_poscars:
                unique_poscars[key] = (level, task_path)
            # 每个 (level, task) 的 calc_id 就是首次出现的 level
            first_level = unique_poscars[key][0]
            calc_id_map[(level, task_name)] = f"level_{first_level}/{task_name}"

    n_unique = len(unique_poscars)
    print(f"共 {total_pairs} 个 (level, task) 对, 去重后 {n_unique} 个不同 VASP 计算")

    # ---- level mapping: 记录每个 level 的任务及其指向的 calc_id ----
    level_mapping = {}
    for level in sorted(level_tasks):
        entries = {}
        for task_name in sorted(level_tasks[level]):
            entries[task_name] = calc_id_map[(level, task_name)]
        level_mapping[level] = entries

    # ---- 打包 ----
    print(f"正在打包到 {args.output} ...")
    with tarfile.open(args.output, "w:gz") as tar:
        # 全局 task_list: 列出所有需要跑的唯一计算路径
        unique_calc_ids = sorted(set(calc_id_map.values()))
        task_list_str = "\n".join(unique_calc_ids) + "\n"
        info = tarfile.TarInfo(name="vasp_inputs/task_list.txt")
        info.size = len(task_list_str.encode())
        tar.addfile(info, io.BytesIO(task_list_str.encode()))

        # level 映射 JSON
        mapping_json = json.dumps(level_mapping, indent=2)
        info = tarfile.TarInfo(name="vasp_inputs/level_mapping.json")
        info.size = len(mapping_json.encode())
        tar.addfile(info, io.BytesIO(mapping_json.encode()))

        # 打包每个唯一计算的 4 个文件
        for (task_name, h), (first_level, task_path) in sorted(unique_poscars.items()):
            calc_dir = f"vasp_inputs/level_{first_level}/{task_name}"
            for fname in required_files:
                fpath = os.path.join(task_path, fname)
                tar.add(fpath, arcname=f"{calc_dir}/{fname}")

    output_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"完成: {args.output} ({output_size:.1f} MB, {n_unique} 个唯一计算)")
    print(f"\nLevel 统计:")
    for level in sorted(level_tasks):
        n_tasks = len(level_tasks[level])
        # 看有多少是复用其他 level 的
        reused = sum(1 for t in level_tasks[level]
                     if calc_id_map[(level, t)] != f"level_{level}/{t}")
        print(f"  Level {level:>2}: {n_tasks} 个任务 ({reused} 个复用已有计算)")
    print(f"\n下一步: scp {args.output} user@cluster:/path/to/work/")


if __name__ == "__main__":
    main()
