#!/usr/bin/env python3
"""Analyze rigid motion between initial LMDB structures and Flow outputs.

For each trajectory in a Flow output directory, this script compares the
adsorbate-only degrees of freedom against the corresponding initial structure
from an LMDB dataset, reporting translation and rotation magnitudes.

Example:
    python scripts/analyze_flow_displacements.py --lmdb val_nonrelaxed_update --flow-dir grid_search_runs/cfg10_steps100/0 --topk 20
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, Tuple

import ase.io
import numpy as np

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Flow outputs vs. initial LMDB structures")
    parser.add_argument(
        "--lmdb",
        required=True,
        help="Path to the LMDB directory that provided the initial structures (e.g. val_nonrelaxed_update)",
    )
    parser.add_argument(
        "--flow-dir",
        required=True,
        help="Directory containing Flow-generated .traj files (single-frame trajectories)",
    )
    parser.add_argument(
        "--pattern",
        default="*.traj",
        help="Glob pattern relative to flow-dir for selecting trajectory files (default: *.traj)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="If >0, only display the top-K trajectories with the largest COM translation",
    )
    return parser.parse_args()


def build_initial_lookup(lmdb_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    dataset = LmdbDataset(
        {
            "src": str(lmdb_path),
            "key_mapping": {"y": "energy", "force": "forces"},
        }
    )
    lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for data in dataset:
        sid = str(data.sid)
        lookup[sid] = (data.pos.cpu().numpy(), data.tags.cpu().numpy())
    return lookup


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return optimal rotation matrix (3x3) taking P -> Q."""
    # Both inputs expected shape (N, 3) and zero-centred.
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(np.dot(V, Wt))
    D = np.diag([1.0, 1.0, np.sign(d)])
    R = V @ D @ Wt
    return R


def rotation_angle_degrees(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def main() -> None:
    args = parse_args()
    lmdb_path = Path(args.lmdb).expanduser().resolve()
    flow_dir = Path(args.flow_dir).expanduser().resolve()

    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")
    if not flow_dir.exists():
        raise FileNotFoundError(f"Flow directory not found: {flow_dir}")

    initial_lookup = build_initial_lookup(lmdb_path)

    rows = []
    for traj_path in sorted(flow_dir.glob(args.pattern)):
        traj = ase.io.read(traj_path.as_posix(), ":")
        final_atoms = traj[-1]
        tags = np.array(final_atoms.get_tags())
        ads_mask = tags == 2
        if not ads_mask.any():
            continue  # no adsorbate atoms

        sid = traj_path.stem
        if sid not in initial_lookup:
            # Try relaxed naming convention with fid suffix removed
            parts = sid.split("_")
            if len(parts) >= 4:
                sid_candidate = "_".join(parts[:-1])
                if sid_candidate in initial_lookup:
                    sid = sid_candidate
            if sid not in initial_lookup:
                continue

        init_pos, init_tags = initial_lookup[sid]
        init_ads = init_pos[init_tags == 2]
        final_ads = final_atoms.get_positions()[ads_mask]

        if init_ads.shape[0] != final_ads.shape[0]:
            continue  # mismatch in adsorbate size

        com_init = init_ads.mean(axis=0)
        com_final = final_ads.mean(axis=0)
        translation = com_final - com_init

        centred_init = init_ads - com_init
        centred_final = final_ads - com_final

        rotation_deg = float("nan")
        rmsd = float("nan")
        if centred_init.shape[0] >= 3:
            R = kabsch_rotation(centred_init, centred_final)
            rotation_deg = rotation_angle_degrees(R)
            rotated = centred_init @ R.T
            rmsd = float(np.sqrt(np.mean(np.sum((rotated - centred_final) ** 2, axis=1))))
        else:
            # With 1-2 atoms we can only report translation.
            rotation_deg = float("nan")
            rmsd = float(np.sqrt(np.mean(np.sum((centred_final - centred_init) ** 2, axis=1))))

        rows.append(
            {
                "sid": traj_path.stem,
                "translation": translation,
                "translation_norm": float(np.linalg.norm(translation)),
                "translation_xy": float(np.linalg.norm(translation[:2])),
                "translation_z": float(translation[2]),
                "rotation_deg": rotation_deg,
                "rmsd": rmsd,
            }
        )

    if not rows:
        print("No matching trajectories found.")
        return

    rows.sort(key=lambda r: r["translation_norm"], reverse=True)
    if args.topk > 0:
        rows = rows[: args.topk]

    header = f"{'sid':<20}{'| dCOM (Å)':>12}{'| dXY (Å)':>12}{'| dz (Å)':>10}{'| rot (deg)':>12}{'| RMSD (Å)':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['sid']:<20}"
            f"{row['translation_norm']:>12.4f}"
            f"{row['translation_xy']:>12.4f}"
            f"{row['translation_z']:>10.4f}"
            f"{row['rotation_deg']:>12.3f}"
            f"{row['rmsd']:>12.4f}"
        )


if __name__ == "__main__":
    main()
