"""
一键脚本：为 EqV2-2D ep180 cfg7_steps5 生成 VASP 输入 + 运行 VASP SP

1. 调用 write_vasp_inputs_multisite 的逻辑生成 VASP 输入
2. 并行运行 VASP SP 计算
"""

import os
import sys
import subprocess
import json
import pickle
import glob
import signal
import numpy as np
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# ==================== PATH CONFIG ====================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
_OCP_PATH = os.environ.get("ADSORBFLOW_OCP_PATH")
if _OCP_PATH:
    sys.path.append(os.path.abspath(_OCP_PATH))
if not os.environ.get("VASP_PP_PATH"):
    raise RuntimeError(
        "Set VASP_PP_PATH to your local VASP pseudopotential directory before "
        "generating VASP inputs."
    )
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement import DetectTrajAnomaly
import ase.io
import ase.constraints

# ==================== CONFIG ====================
BASE_PATH = "grid_search_runs/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2_epoch0180_unweightedvalloss1.0316_posmae0.9085/val_nonrelaxed_update"
NSITES_DIR = "nsites_10"
CFG_DIR = "cfg7_steps5"

TRAJ_INPUT_PATH = os.path.join(BASE_PATH, NSITES_DIR, CFG_DIR)
TAG_PATH = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_tags.pkl")

SITE_LEVELS = [1, 2, 5, 10]
CORES_PER_JOB = 8
VASP_CMD = "vasp_std"
VASP_TIMEOUT = 1800  # 30 min per job

VASP_FLAGS = {
    "ibrion": 2, "nsw": 0, "isif": 0, "isym": 0,
    "lreal": "Auto", "ediffg": -0.03, "symprec": 1e-10,
    "encut": 350.0, "laechg": True, "lwave": False,
    "ncore": 4, "gga": "RP", "pp": "PBE", "xc": "PBE",
    "setups": "minimal",
}

FILES_TO_CLEANUP = [
    "WAVECAR", "CHG", "CHGCAR", "DOSCAR", "EIGENVAL",
    "IBZKPT", "PCDAT", "XDATCAR", "PROCAR", "LOCPOT", "run_vasp.sh",
]

ANOMALY_TYPES = ["dissociated", "desorbed", "surface_changed", "intercalated"]

# ==================== LOAD DATA ====================
print("Loading tags map...")
with open(TAG_PATH, "rb") as h:
    tags_map = pickle.load(h)


def _make_ads_contiguous(atoms):
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


def get_anomaly_details(traj, sid):
    initial_atoms = _make_ads_contiguous(traj[0])
    final_atoms = _make_ads_contiguous(traj[-1])
    atom_tags = initial_atoms.get_tags()
    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anom = np.array([
        detector.is_adsorbate_dissociated(),
        detector.is_adsorbate_desorbed(),
        detector.has_surface_changed(),
        detector.is_adsorbate_intercalated(),
    ])
    return {
        "is_anomalous": anom.any(),
        "types": [t for t, a in zip(ANOMALY_TYPES, anom) if a],
    }


def get_traj_files_for_sites(base_path, site_indices):
    sid_to_trajs = defaultdict(list)
    for site_idx in site_indices:
        relaxations_dir = os.path.join(base_path, str(site_idx), "relaxations")
        if not os.path.exists(relaxations_dir):
            continue
        for traj_path in glob.glob(os.path.join(relaxations_dir, "*.traj")):
            filename = os.path.basename(traj_path).replace(".traj", "")
            if filename.count("_") == 3:
                sid = "_".join(filename.split("_")[:-1])
            else:
                sid = filename
            sid_to_trajs[sid].append((traj_path, site_idx))
    return sid_to_trajs


def find_best_structure(sid, traj_list):
    """Find best non-anomalous structure; fallback to best anomalous."""
    candidates = []
    for traj_path, site_idx in traj_list:
        try:
            atoms = ase.io.read(traj_path)
            energy = atoms.get_potential_energy()
            candidates.append((traj_path, site_idx, energy))
        except Exception:
            continue
    if not candidates:
        return None, {"is_anomalous": True, "types": []}

    candidates.sort(key=lambda x: x[2])
    normal, anomalous = [], []
    for traj_path, site_idx, energy in candidates:
        try:
            traj = ase.io.read(traj_path, ":")
            anom_info = get_anomaly_details(traj, sid)
            if anom_info["is_anomalous"]:
                anomalous.append((traj_path, site_idx, energy, anom_info["types"]))
            else:
                normal.append((traj_path, site_idx, energy))
        except Exception:
            continue

    if normal:
        return (normal[0][0], normal[0][1], normal[0][2]), {"is_anomalous": False, "types": []}
    elif anomalous:
        return (anomalous[0][0], anomalous[0][1], anomalous[0][2]), {"is_anomalous": True, "types": anomalous[0][3]}
    return None, {"is_anomalous": True, "types": []}


# ==================== VASP runner ====================
def cleanup_vasp_files(d):
    for fn in FILES_TO_CLEANUP:
        fp = os.path.join(d, fn)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass


def check_vasp_success(d):
    outcar = os.path.join(d, "OUTCAR")
    if not os.path.exists(outcar):
        return False
    with open(outcar) as f:
        content = f.read()
        return ("TOTEN" in content and "General timing" in content) or "reached required accuracy" in content


def run_vasp_task(script_dir):
    if check_vasp_success(script_dir):
        cleanup_vasp_files(script_dir)
        return {"dir": script_dir, "success": True, "status": "skipped_completed"}
    cmd = f"cd {script_dir} && ulimit -s unlimited && mpirun -np {CORES_PER_JOB} {VASP_CMD} > vasp.out 2>&1"
    try:
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setpgrp,
        )
        proc.wait()  # 无超时限制
        success = check_vasp_success(script_dir)
        cleanup_vasp_files(script_dir)
        return {"dir": script_dir, "success": success, "status": "completed" if success else "failed"}
    except Exception as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except Exception:
            pass
        cleanup_vasp_files(script_dir)
        return {"dir": script_dir, "success": False, "status": str(e)}


# ==================== MAIN ====================
def main():
    print(f"\n{'='*60}")
    print(f"  VASP SP for EqV2-2D ep180 cfg7_steps5")
    print(f"  Path: {TRAJ_INPUT_PATH}")
    print(f"{'='*60}")

    # Check available sites
    available_sites = [i for i in range(10) if os.path.exists(os.path.join(TRAJ_INPUT_PATH, str(i)))]
    print(f"Available sites: {available_sites}")

    # ===== STEP 1: Generate VASP inputs =====
    print("\n--- Step 1: Generating VASP inputs ---")
    results_by_level = {}
    generated_structures = set()
    anomaly_stats_by_level = {}
    all_vasp_dirs = []

    for num_sites in SITE_LEVELS:
        print(f"\nProcessing level {num_sites}...")
        site_indices = list(range(num_sites))
        sid_to_trajs = get_traj_files_for_sites(TRAJ_INPUT_PATH, site_indices)

        level_results = []
        new_count, skip_count = 0, 0
        level_anom = {"total_sids": 0, "selected_anomalous": 0, "selected_normal": 0,
                      "by_type": {t: 0 for t in ANOMALY_TYPES}, "anomalous_sids": []}

        for sid in tqdm(sid_to_trajs.keys(), desc=f"Level {num_sites}"):
            traj_list = sid_to_trajs[sid]
            best, anom_info = find_best_structure(sid, traj_list)

            level_anom["total_sids"] += 1
            if anom_info["is_anomalous"]:
                level_anom["selected_anomalous"] += 1
                level_anom["anomalous_sids"].append(sid)
                for t in anom_info["types"]:
                    level_anom["by_type"][t] += 1
            else:
                level_anom["selected_normal"] += 1

            if best is None:
                continue

            # 跳过 MLFF 异常结构，不做 VASP
            if anom_info["is_anomalous"]:
                continue

            traj_path, site_idx, energy = best
            structure_key = (sid, site_idx)

            if structure_key in generated_structures:
                skip_count += 1
                level_results.append({
                    "sid": sid, "traj_path": traj_path, "site_idx": site_idx,
                    "energy": energy, "status": "skipped_duplicate",
                })
                continue

            generated_structures.add(structure_key)
            new_count += 1

            try:
                traj = ase.io.read(traj_path, ":")
                relaxed_struct = traj[-1]
                tags = tags_map[sid]
                fixed_atoms = np.where(tags == 2)[0]
                relaxed_struct.set_constraint(ase.constraints.FixAtoms(fixed_atoms))

                filename = os.path.basename(traj_path).replace(".traj", "")
                fid = filename.split("_")[-1] if filename.count("_") == 3 else 0
                output_name = f"{sid}_{fid}"

                level_vasp_dir = os.path.join(TRAJ_INPUT_PATH, f"vasp_level_{num_sites}", output_name)
                os.makedirs(os.path.dirname(level_vasp_dir), exist_ok=True)
                write_vasp_input_files(relaxed_struct, outdir=level_vasp_dir, vasp_flags=VASP_FLAGS)
                all_vasp_dirs.append(level_vasp_dir)

                level_results.append({
                    "sid": sid, "traj_path": traj_path, "site_idx": site_idx,
                    "energy": energy, "status": "generated",
                    "output_dir": level_vasp_dir,
                })
            except Exception as e:
                print(f"Error: {sid}: {e}")
                level_results.append({
                    "sid": sid, "traj_path": traj_path, "site_idx": site_idx,
                    "energy": energy, "status": "error", "error": str(e),
                })

        results_by_level[num_sites] = level_results
        anomaly_stats_by_level[num_sites] = level_anom
        print(f"  Level {num_sites}: new={new_count}, skipped={skip_count}, "
              f"anom={level_anom['selected_anomalous']}/{level_anom['total_sids']}")

    # Save results JSON
    json_out = {str(k): v for k, v in results_by_level.items()}
    with open(os.path.join(TRAJ_INPUT_PATH, "multisite_vasp_results.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # Save anomaly report
    anom_out = {str(k): v for k, v in anomaly_stats_by_level.items()}
    with open(os.path.join(TRAJ_INPUT_PATH, "multisite_anomaly_report.json"), "w") as f:
        json.dump(anom_out, f, indent=2)

    print(f"\nTotal VASP dirs to run: {len(all_vasp_dirs)}")

    # ===== STEP 2: Run VASP =====
    print(f"\n--- Step 2: Running VASP SP ({len(all_vasp_dirs)} jobs) ---")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine parallelism
    total_cores = 64
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            parts = f.read().strip().split()
            if parts[0] != 'max':
                total_cores = int(parts[0]) // int(parts[1])
    except Exception:
        pass

    # Use all cores since no GPU task running
    max_workers = max(1, total_cores // CORES_PER_JOB)
    print(f"Cores={total_cores}, workers={max_workers}, cores/job={CORES_PER_JOB}")

    success_count = 0
    fail_count = 0
    timeout_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_vasp_task, d): d for d in all_vasp_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="VASP SP"):
            result = future.result()
            if result["success"]:
                success_count += 1
            else:
                fail_count += 1
                if "timeout" in result["status"]:
                    timeout_count += 1
                print(f"  FAILED: {os.path.basename(result['dir'])} - {result['status']}")

    print(f"\n{'='*60}")
    print(f"  VASP SP SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:   {len(all_vasp_dirs)}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count} (timeout: {timeout_count})")
    print(f"  End:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save VASP run results
    vasp_results_file = os.path.join(TRAJ_INPUT_PATH, "vasp_run_results.json")
    with open(vasp_results_file, "w") as f:
        json.dump({
            "total": len(all_vasp_dirs),
            "success": success_count,
            "failed": fail_count,
            "timeout": timeout_count,
        }, f, indent=2)
    print(f"\nResults saved to: {vasp_results_file}")


if __name__ == "__main__":
    main()
