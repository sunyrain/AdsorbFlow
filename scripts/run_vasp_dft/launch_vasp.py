import sys
import subprocess
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Link to the path with ML relaxations
PATH = "grid_search_runs/2025-12-17-19-22-40-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_epoch0180_unweightedvalloss1.4265_posmae0.6214/val_nonrelaxed_update/nsites_3/cfg3_steps10"

MAX_CALCS = 200
CORES_PER_JOB = 8  # 每个任务使用的核心数。对于小体系，16-32核通常是最优的，太多会导致崩溃。

def run_vasp_task(script_dir):
    """
    运行单个 VASP 任务的函数
    """
    try:
        print(f"Starting VASP in: {script_dir}")

        # 在目标目录下创建独立的运行脚本，避免并行冲突
        run_script_path = os.path.join(script_dir, "run_vasp.sh")

        # 运行命令：
        # 1. 进入目录
        # 2. 解除栈限制 (ulimit -s unlimited)
        # 3. 使用指定核心数运行 VASP (mpirun -np CORES_PER_JOB)
        # 4. 重定向输出到 vasp.out
        cmd = (
            f"cd {script_dir} && "
            f"ulimit -s unlimited && "
            f"mpirun -np {CORES_PER_JOB} vasp_std > vasp.out 2>&1"
        )

        with open(run_script_path, "w") as f:
            f.write(cmd)

        # 执行脚本
        subprocess.run(["bash", "run_vasp.sh"], cwd=script_dir, check=True)

        print(f"Finished VASP in: {script_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running VASP in {script_dir}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in {script_dir}: {e}")
        return False

if __name__ == "__main__":
    print(f"Checking PATH: {PATH}")
    if not os.path.exists(PATH):
        print(f"Error: PATH does not exist: {PATH}")
        sys.exit(1)

    script_list = []
    print("Walking through directories...")
    for subdir, dirs, files in os.walk(f"{PATH}/vasp2"):
        # 只有包含 INCAR 且没有 OUTCAR (未完成) 的目录才加入任务列表
        if "INCAR" in files:
            if "OUTCAR" not in files:
                script_list.append(subdir)

    # 限制最大任务数
    script_list = script_list[:MAX_CALCS]
    print(f"Found {len(script_list)} directories to run.")

    # 计算并行度
    total_cores = multiprocessing.cpu_count()
    # 确保至少有1个worker
    max_workers = max(1, total_cores // CORES_PER_JOB )-1

    print(f"Total CPUs: {total_cores}")
    print(f"Cores per job: {CORES_PER_JOB}")
    print(f"Max parallel jobs: {max_workers}")

    if not script_list:
        print("No jobs to run.")
        sys.exit(0)

    print(f"Starting parallel execution with {max_workers} workers...")

    # 使用进程池并行运行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_vasp_task, script_list))

    print("All done.")

