#!/usr/bin/env python3
"""
为单个 cfg 目录生成多 level 的 VASP 输入：
- level=1: 可选 fixed_seed（单 seed baseline）
- level>1: 论文前缀策略（从前 k 个 seed 中选）

输出目录结构（在 cfg_dir 下）：
  vasp_level_1/
  vasp_level_2/
  vasp_level_5/
  vasp_level_10/

每个 level 内为 <sid>_<fid>/POSCAR|INCAR|POTCAR|KPOINTS。
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import ase.constraints
import ase.io
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("VASP_PP_PATH", os.path.join(PROJECT_ROOT, "potpaw_PBE_54"))

import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Open-Catalyst-Dataset"))

from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement.flag_anomaly import DetectTrajAnomaly

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-dir", required=True, help=".../nsites_10/cfg*_steps*")
    parser.add_argument("--tag-path", default=os.path.join(PROJECT_ROOT, "oc20_dense_mappings/oc20dense_tags.pkl"))
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--max-sites", type=int, default=10)
    parser.add_argument("--level1-mode", choices=["prefix", "fixed_seed", "best_seed"], default="best_seed")
    parser.add_argument("--fixed-seed-index", type=int, default=0, help="Seed index used when level1-mode=fixed_seed.")
    parser.add_argument("--clean-existing", action="store_true", help="Remove existing vasp_level_* folders before writing new inputs.")
    return parser.parse_args()


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


def parse_sid_fid(path):
    stem = os.path.basename(path).replace(".traj", "")
    parts = stem.split("_")
    if len(parts) >= 4:
        sid = "_".join(parts[:-1])
        fid = parts[-1]
    elif len(parts) >= 3:
        sid = stem
        fid = parts[-1]
    elif len(parts) == 2:
        sid = stem
        fid = parts[-1]
    else:
        sid = stem
        fid = "0"
    return sid, fid


def is_anomalous(traj):
    initial = _make_ads_contiguous(traj[0])
    final = _make_ads_contiguous(traj[-1])
    tags = initial.get_tags()
    detector = DetectTrajAnomaly(initial, final, tags)
    flags = [
        detector.is_adsorbate_dissociated(),
        detector.is_adsorbate_desorbed(),
        detector.has_surface_changed(),
        detector.is_adsorbate_intercalated(),
    ]
    return bool(any(flags))


def collect_candidates(cfg_dir, site_indices):
    sid_to_candidates = defaultdict(list)
    for site_idx in site_indices:
        relax_dir = os.path.join(cfg_dir, str(site_idx), "relaxations")
        if not os.path.isdir(relax_dir):
            continue
        for tf in glob.glob(os.path.join(relax_dir, "*.traj")):
            sid, fid = parse_sid_fid(tf)
            sid_to_candidates[sid].append((tf, fid, site_idx))
    return sid_to_candidates


def select_best_normal(sid, candidates):
    best = None
    for tf, fid, site_idx in candidates:
        try:
            traj = ase.io.read(tf, ":")
            if is_anomalous(traj):
                continue
            energy = traj[-1].get_potential_energy()
            energy_val = float(np.asarray(energy).flat[0])
            if best is None or energy_val < best["energy"]:
                best = {
                    "sid": sid,
                    "fid": str(fid),
                    "site_idx": int(site_idx),
                    "traj_path": tf,
                    "energy": energy_val,
                }
        except Exception:
            continue
    return best


def write_vasp(best, out_dir, tags_map):
    traj = ase.io.read(best["traj_path"], ":")
    relaxed = traj[-1]
    sid = best["sid"]
    if sid in tags_map:
        tags = tags_map[sid]
        fixed_atoms = np.where(tags == 2)[0]
        relaxed.set_constraint(ase.constraints.FixAtoms(fixed_atoms))
    output_name = f"{sid}_{best['fid']}"
    write_vasp_input_files(relaxed, outdir=os.path.join(out_dir, output_name), vasp_flags=VASP_FLAGS)
    return output_name


def main():
    args = parse_args()
    cfg_dir = os.path.abspath(args.cfg_dir)

    with open(args.tag_path, "rb") as f:
        tags_map = pickle.load(f)

    levels = sorted(set(args.levels))
    all_sites = list(range(args.max_sites))

    # auto-detect best seed from JSONL when best_seed mode
    best_seed_idx = args.fixed_seed_index
    if args.level1_mode == "best_seed":
        # look for JSONL one or two levels up
        for rel in [
            os.path.join(cfg_dir, "logs", "grid_search_results_nsites10.jsonl"),
            os.path.join(os.path.dirname(cfg_dir), "grid_search_results_nsites10.jsonl"),
        ]:
            if os.path.exists(rel):
                with open(rel) as jf:
                    for jline in jf:
                        d = json.loads(jline)
                        ssp = d.get("site_success_percent", [])
                        if ssp:
                            best_seed_idx = int(np.argmax(ssp))
                            print(f"[best_seed] auto-detected best seed = {best_seed_idx} (SR={ssp[best_seed_idx]:.1f}%) from {rel}")
                        break
                break
        else:
            print(f"[best_seed] WARNING: no JSONL found, falling back to seed {best_seed_idx}")

    summary = {
        "cfg_dir": cfg_dir,
        "level1_mode": args.level1_mode,
        "best_seed_idx": best_seed_idx if args.level1_mode == "best_seed" else None,
        "levels": levels,
        "by_level": {},
    }

    for k in levels:
        if k == 1 and args.level1_mode in ("fixed_seed", "best_seed"):
            site_indices = [best_seed_idx]
        else:
            site_indices = list(range(min(k, args.max_sites)))

        level_dir = os.path.join(cfg_dir, f"vasp_level_{k}")
        if args.clean_existing and os.path.isdir(level_dir):
            import shutil
            shutil.rmtree(level_dir)
        os.makedirs(level_dir, exist_ok=True)

        sid_to_candidates = collect_candidates(cfg_dir, site_indices)

        selected = 0
        filtered_all_anom = 0
        errors = 0
        task_names = []

        for sid, cand in tqdm(sorted(sid_to_candidates.items()), desc=f"level {k}"):
            best = select_best_normal(sid, cand)
            if best is None:
                filtered_all_anom += 1
                continue
            try:
                task_name = write_vasp(best, level_dir, tags_map)
                task_names.append(task_name)
                selected += 1
            except Exception:
                errors += 1

        with open(os.path.join(level_dir, "task_list.txt"), "w") as f:
            for name in sorted(task_names):
                f.write(name + "\n")

        summary["by_level"][str(k)] = {
            "site_indices": site_indices,
            "selected": selected,
            "filtered_all_anomalous": filtered_all_anom,
            "errors": errors,
            "tasks": len(task_names),
        }

    out_stats = os.path.join(cfg_dir, "vasp_multilevel_stats.json")
    with open(out_stats, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_stats}")


if __name__ == "__main__":
    main()
