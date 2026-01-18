"""
多层次 site 的 VASP 并行运行脚本

支持运行 write_vasp_inputs_multisite.py 生成的多层级 VASP 输入
"""

import sys
import subprocess
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import glob
from datetime import datetime

# ==================== 配置区域 ====================

# 基础路径 - 和 write_vasp_inputs_multisite.py 保持一致
BASE_PATH = "/root/autodl-tmp/AdsorbFlow/grid_search_runs/2025-12-17-19-22-40-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_epoch0180_unweightedvalloss1.4265_posmae0.6214/val_nonrelaxed_update"
NSITES_DIR = "nsites_10"
CFG_DIR = "cfg3_steps10"

PATH = os.path.join(BASE_PATH, NSITES_DIR, CFG_DIR)

# 也可以从独立导出目录运行
EXPORT_PATH = "/root/autodl-tmp/vasp_cluster_inputs"

# 选择运行模式: "levels" (按层级运行) 或 "export" (从导出目录运行)
RUN_MODE = "levels"  # or "export"

# 要运行的层级（仅在 RUN_MODE="levels" 时有效）
SITE_LEVELS = [1, 2, 5, 10]

MAX_CALCS = 500
CORES_PER_JOB = 8

# 计算完成后要删除的临时文件（节省磁盘空间）
# 单点计算不需要 WAVECAR, CHG, CHGCAR 等
FILES_TO_CLEANUP = [
    "WAVECAR",      # 波函数文件，通常很大（几百MB到几GB）
    "CHG",          # 电荷密度
    "CHGCAR",       # 电荷密度（用于续算）
    "DOSCAR",       # 态密度（单点通常不需要）
    "EIGENVAL",     # 本征值
    "IBZKPT",       # k点
    "PCDAT",        # 对关联函数
    "XDATCAR",      # 离子步轨迹（单点只有1步）
    "PROCAR",       # 投影波函数
    "LOCPOT",       # 局域势
    "run_vasp.sh",  # 临时脚本
]

# ==================== 函数定义 ====================

def cleanup_vasp_files(script_dir):
    """
    清理 VASP 计算产生的临时文件，节省磁盘空间
    """
    cleaned_size = 0
    for filename in FILES_TO_CLEANUP:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                os.remove(filepath)
                cleaned_size += size
            except Exception:
                pass
    return cleaned_size


def check_vasp_success(script_dir):
    """
    检查 VASP 计算是否成功完成
    通过检查 OUTCAR 中的 "reached required accuracy" 或能量收敛标志
    """
    outcar = os.path.join(script_dir, "OUTCAR")
    if not os.path.exists(outcar):
        return False, "OUTCAR not found"
    
    try:
        with open(outcar, 'r') as f:
            content = f.read()
            # 单点计算检查能量是否计算完成
            if "TOTEN" in content and "General timing" in content:
                return True, "completed"
            elif "reached required accuracy" in content:
                return True, "converged"
            else:
                return False, "not converged"
    except Exception as e:
        return False, str(e)


def run_vasp_task(script_dir):
    """
    运行单个 VASP 任务的函数
    """
    try:
        # 检查是否已完成（断点续算）
        success, msg = check_vasp_success(script_dir)
        if success:
            # 已完成，直接清理并返回
            cleaned = cleanup_vasp_files(script_dir)
            return {
                "dir": script_dir, 
                "success": True, 
                "status": "skipped_completed",
                "cleaned_bytes": cleaned
            }
        
        run_script_path = os.path.join(script_dir, "run_vasp.sh")
        
        cmd = (
            f"cd {script_dir} && "
            f"ulimit -s unlimited && "
            f"mpirun -np {CORES_PER_JOB} /root/autodl-tmp/vasp-autodl/vasp.6.3.0/bin/vasp_std > vasp.out 2>&1"
        )
        
        with open(run_script_path, "w") as f:
            f.write(cmd)
            
        subprocess.run(["bash", "run_vasp.sh"], cwd=script_dir, check=True)
        
        # 检查是否成功
        success, msg = check_vasp_success(script_dir)
        
        # 清理临时文件
        cleaned = cleanup_vasp_files(script_dir)
        
        return {
            "dir": script_dir, 
            "success": success, 
            "status": msg,
            "cleaned_bytes": cleaned
        }
    except subprocess.CalledProcessError as e:
        # 即使失败也尝试清理
        cleaned = cleanup_vasp_files(script_dir)
        return {
            "dir": script_dir, 
            "success": False, 
            "status": "subprocess_error",
            "error": str(e),
            "cleaned_bytes": cleaned
        }
    except Exception as e:
        return {
            "dir": script_dir, 
            "success": False, 
            "status": "exception",
            "error": str(e),
            "cleaned_bytes": 0
        }


def collect_jobs_from_levels(base_path, levels):
    """
    从多个 vasp_level_* 目录收集任务
    """
    script_list = []
    level_counts = {}
    
    for level in levels:
        level_dir = os.path.join(base_path, f"vasp_level_{level}")
        if not os.path.exists(level_dir):
            print(f"Warning: {level_dir} does not exist")
            continue
        
        count = 0
        for subdir, dirs, files in os.walk(level_dir):
            if "INCAR" in files and "OUTCAR" not in files:
                script_list.append(subdir)
                count += 1
        
        level_counts[level] = count
        print(f"Level {level}: {count} jobs")
    
    return script_list, level_counts


def collect_jobs_from_export(export_path):
    """
    从独立导出目录收集任务
    """
    script_list = []
    
    for subdir, dirs, files in os.walk(export_path):
        if "INCAR" in files and "OUTCAR" not in files:
            script_list.append(subdir)
    
    return script_list


def main():
    print(f"Run mode: {RUN_MODE}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if RUN_MODE == "levels":
        print(f"Base path: {PATH}")
        print(f"Levels: {SITE_LEVELS}")
        script_list, level_counts = collect_jobs_from_levels(PATH, SITE_LEVELS)
    elif RUN_MODE == "export":
        print(f"Export path: {EXPORT_PATH}")
        script_list = collect_jobs_from_export(EXPORT_PATH)
    else:
        print(f"Unknown run mode: {RUN_MODE}")
        sys.exit(1)
    
    # 限制最大任务数
    original_count = len(script_list)
    script_list = script_list[:MAX_CALCS]
    print(f"\nFound {original_count} directories to run, limited to {len(script_list)}")
    
    if not script_list:
        print("No jobs to run.")
        sys.exit(0)
    
    # 计算并行度 - 检测容器实际 CPU 配额
    # 注意：multiprocessing.cpu_count() 返回宿主机核心数，不是容器配额
    total_cores = None
    try:
        # cgroup v2 格式
        with open('/sys/fs/cgroup/cpu.max', 'r') as f:
            content = f.read().strip().split()
            if content[0] != 'max':
                quota = int(content[0])
                period = int(content[1])
                total_cores = quota // period
    except Exception:
        pass
    
    if total_cores is None:
        try:
            # cgroup v1 格式
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                period = int(f.read().strip())
            if quota > 0:
                total_cores = quota // period
        except Exception:
            pass
    
    if total_cores is None:
        total_cores = multiprocessing.cpu_count()
    
    max_workers = max(1, total_cores // CORES_PER_JOB)
    
    print(f"\nTotal CPUs: {total_cores}")
    print(f"Cores per job: {CORES_PER_JOB}")
    print(f"Max parallel jobs: {max_workers}")
    
    print(f"\nStarting parallel execution with {max_workers} workers...")
    print("="*60)
    
    # 使用进程池并行运行，带进度显示
    results = []
    total_cleaned = 0
    completed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_dir = {executor.submit(run_vasp_task, d): d for d in script_list}
        
        # 按完成顺序处理结果
        for future in as_completed(future_to_dir):
            result = future.result()
            results.append(result)
            completed += 1
            total_cleaned += result.get("cleaned_bytes", 0)
            
            if result.get("status") == "skipped_completed":
                skipped += 1
                status_str = "SKIP"
            elif result["success"]:
                status_str = "OK"
            else:
                status_str = "FAIL"
            
            # 实时进度显示
            dir_name = os.path.basename(result["dir"])
            print(f"[{completed}/{len(script_list)}] {status_str}: {dir_name}")
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total jobs: {len(results)}")
    print(f"Successful: {success_count} (including {skipped} skipped/already done)")
    print(f"Failed: {fail_count}")
    print(f"Total cleaned: {total_cleaned / 1024 / 1024:.1f} MB")
    
    # 保存运行结果
    results_file = os.path.join(PATH, "vasp_run_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "total": len(results),
            "success": success_count,
            "failed": fail_count,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    if fail_count > 0:
        print("\nFailed jobs:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['dir']}: {r.get('error', 'Unknown error')}")
    
    print("\nAll done.")


if __name__ == "__main__":
    main()
