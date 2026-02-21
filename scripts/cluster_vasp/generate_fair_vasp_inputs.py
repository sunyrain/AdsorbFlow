#!/usr/bin/env python3
"""
公平评估 VASP 输入生成器: 为每个 (model, SID, site_idx) 组合独立生成 VASP SP 输入。

用法:
    python scripts/cluster_vasp/generate_fair_vasp_inputs.py \
        --output-dir /root/autodl-tmp/AdsorbFlow/vasp_fair_all2d \
        --output-tar vasp_fair_all2d.tar.gz
"""

import argparse
import io
import json
import os
import pickle
import sys
import tarfile
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "Open-Catalyst-Dataset"))

os.environ["VASP_PP_PATH"] = "/root/autodl-tmp/potpaw_PBE_54"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement.flag_anomaly import DetectTrajAnomaly
import ase.io
import ase.constraints

# ==================== CONFIG ====================
MODELS = {
    "painn_2d": {
        "base_path": os.path.join(
            _PROJECT_ROOT,
            "grid_search_runs/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint"
            "/val_nonrelaxed_update/nsites_10/cfg5_steps5",
        ),
        "label": "PaiNN-2D cfg5_steps5",
    },
    "eqv2_2d": {
        "base_path": os.path.join(
            _PROJECT_ROOT,
            "grid_search_runs/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
            "_epoch0180_unweightedvalloss1.0316_posmae0.9085"
            "/val_nonrelaxed_update/nsites_10/cfg7_steps5",
        ),
        "label": "EqV2-2D cfg7_steps5",
    },
}

TAG_PATH = os.path.join(_PROJECT_ROOT, "oc20_dense_mappings/oc20dense_tags.pkl")
NSITES = 10  # site indices 0..9

VASP_FLAGS = {
    "ibrion": 2, "nsw": 0, "isif": 0, "isym": 0,
    "lreal": "Auto", "ediffg": -0.03, "symprec": 1e-10,
    "encut": 350.0, "laechg": True, "lwave": False,
    "ncore": 4, "gga": "RP", "pp": "PBE", "xc": "PBE",
    "setups": "minimal",
}


# ==================== HELPERS ====================
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
    """Check if a relaxation trajectory is anomalous."""
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
        return True  # 出错视为异常


def extract_sid(filename):
    """从 traj 文件名提取 SID。"""
    name = filename.replace(".traj", "")
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])  # e.g. "0_2374_49"
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "vasp_fair_all2d"),
                        help="本地 VASP 输入输出目录")
    parser.add_argument("--output-tar", default=os.path.join(_PROJECT_ROOT, "vasp_fair_all2d.tar.gz"),
                        help="打包 tar.gz 路径")
    parser.add_argument("--skip-tar", action="store_true", help="跳过打包步骤")
    args = parser.parse_args()

    print("Loading tags map...")
    with open(TAG_PATH, "rb") as h:
        tags_map = pickle.load(h)

    os.makedirs(args.output_dir, exist_ok=True)

    all_tasks = []  # (task_name, task_dir)
    stats = {}

    for model_key, model_cfg in MODELS.items():
        base_path = model_cfg["base_path"]
        label = model_cfg["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Path: {base_path}")
        print(f"{'='*60}")

        model_normal = 0
        model_anomalous = 0
        model_error = 0
        model_tasks = []

        for site_idx in range(NSITES):
            relax_dir = os.path.join(base_path, str(site_idx), "relaxations")
            if not os.path.isdir(relax_dir):
                print(f"  WARNING: site_{site_idx} relaxations not found, skipping")
                continue

            traj_files = sorted(glob.glob(os.path.join(relax_dir, "*.traj")))
            site_normal = 0
            site_anom = 0

            for tf in tqdm(traj_files, desc=f"  {model_key} site_{site_idx}", leave=False):
                sid = extract_sid(os.path.basename(tf))
                try:
                    traj = ase.io.read(tf, index=":")
                    if is_anomalous(traj):
                        site_anom += 1
                        model_anomalous += 1
                        continue

                    # Normal → generate VASP input
                    relaxed = traj[-1]
                    if sid in tags_map:
                        tags = tags_map[sid]
                        fixed = np.where(tags == 2)[0]
                        relaxed.set_constraint(ase.constraints.FixAtoms(fixed))

                    task_name = f"{model_key}__{sid}__site{site_idx}"
                    task_dir = os.path.join(args.output_dir, task_name)

                    # 如果已生成且有 POSCAR，跳过
                    if os.path.exists(os.path.join(task_dir, "POSCAR")):
                        model_tasks.append((task_name, task_dir))
                        site_normal += 1
                        model_normal += 1
                        continue

                    os.makedirs(task_dir, exist_ok=True)
                    write_vasp_input_files(relaxed, outdir=task_dir, vasp_flags=VASP_FLAGS)
                    model_tasks.append((task_name, task_dir))
                    site_normal += 1
                    model_normal += 1

                except Exception as e:
                    model_error += 1
                    print(f"    ERROR {sid} site_{site_idx}: {e}")

            print(f"  site_{site_idx}: normal={site_normal}, anomalous={site_anom}")

        all_tasks.extend(model_tasks)
        stats[model_key] = {
            "label": label,
            "normal": model_normal,
            "anomalous": model_anomalous,
            "error": model_error,
            "total": model_normal + model_anomalous + model_error,
        }
        print(f"\n  {label} Summary:")
        print(f"    Normal (VASP needed): {model_normal}")
        print(f"    Anomalous (skip):     {model_anomalous}")
        print(f"    Errors:               {model_error}")

    # Write task list
    task_list_path = os.path.join(args.output_dir, "task_list.txt")
    with open(task_list_path, "w") as f:
        for task_name, _ in all_tasks:
            f.write(task_name + "\n")

    # Write stats
    stats_path = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*60}")
    for mk, ms in stats.items():
        print(f"  {ms['label']}: {ms['normal']} tasks ({ms['anomalous']} anomalous skipped)")
    print(f"  Total VASP tasks: {len(all_tasks)}")
    print(f"  Task list: {task_list_path}")

    # Pack tar.gz
    if not args.skip_tar:
        print(f"\n  Packing {args.output_tar} ...")
        required_files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        with tarfile.open(args.output_tar, "w:gz") as tar:
            # task_list.txt
            info = tarfile.TarInfo(name="vasp_fair/task_list.txt")
            with open(task_list_path, "rb") as f:
                data = f.read()
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            # stats
            stats_data = json.dumps(stats, indent=2).encode()
            info = tarfile.TarInfo(name="vasp_fair/generation_stats.json")
            info.size = len(stats_data)
            tar.addfile(info, io.BytesIO(stats_data))

            # VASP files
            for task_name, task_dir in tqdm(all_tasks, desc="Packing"):
                for fname in required_files:
                    fpath = os.path.join(task_dir, fname)
                    if os.path.exists(fpath):
                        tar.add(fpath, arcname=f"vasp_fair/{task_name}/{fname}")

        tar_size_mb = os.path.getsize(args.output_tar) / (1024 * 1024)
        print(f"  Done: {args.output_tar} ({tar_size_mb:.1f} MB)")

    print(f"\n  Next: scp {args.output_tar} qiujiangjie@166.111.35.183:/public/home/qiujiangjie/adsorbFlow/")


if __name__ == "__main__":
    main()
