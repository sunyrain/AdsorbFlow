#!/usr/bin/env python3
"""Analyze translation/rotation magnitudes across training trajectories.

Given a directory (or glob pattern) of ASE .traj files that contain full
relaxation trajectories, this script compares the first and last frames for
each system and reports statistics about adsorbate motion. The results help
choose sensible translation/rotation strengths for Flow Matching models.

Example:
    python scripts/analyze_training_traj_motion.py 0/ --lmdb train_allE --output-csv motion_stats.csv
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ase.io
import numpy as np

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure adsorbate motion over relaxation trajectories")
    parser.add_argument(
        "traj_root",
        nargs="*",
        help="Optional directory or glob pattern pointing to .traj files (e.g. 'train_trajs/*.traj')",
    )
    parser.add_argument(
        "--lmdb",
        nargs="+",
        default=None,
        help="Optional LMDB directory (or directories) containing training relaxations (e.g. train_allE)",
    )
    parser.add_argument(
        "--adsorbate-tag",
        type=int,
        default=2,
        help="Atom tag identifying adsorbate atoms (default: 2)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top-N trajectories with largest COM translation (default: 10)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV file to write per-trajectory metrics",
    )
    return parser.parse_args()


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return the optimal rotation matrix that maps P to Q."""
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R


def rotation_angle_degrees(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def compute_motion(
    sid: str,
    init_pos: np.ndarray,
    init_tags: np.ndarray,
    final_pos: np.ndarray,
    final_tags: np.ndarray,
    ads_tag: int,
) -> Dict[str, float]:
    ads_mask_init = init_tags == ads_tag
    ads_mask_final = final_tags == ads_tag

    if not np.any(ads_mask_init) or not np.any(ads_mask_final):
        raise ValueError("no adsorbate atoms detected")

    init_ads = init_pos[ads_mask_init]
    final_ads = final_pos[ads_mask_final]

    if init_ads.shape[0] != final_ads.shape[0]:
        raise ValueError("adsorbate atom count changed between frames")

    com_init = init_ads.mean(axis=0)
    com_final = final_ads.mean(axis=0)
    translation = com_final - com_init

    centered_init = init_ads - com_init
    centered_final = final_ads - com_final

    rotation_deg = float("nan")
    rmsd = float("nan")
    if centered_init.shape[0] >= 3:
        R = kabsch_rotation(centered_init, centered_final)
        rotation_deg = rotation_angle_degrees(R)
        rotated = centered_init @ R.T
        rmsd = float(np.sqrt(np.mean(np.sum((rotated - centered_final) ** 2, axis=1))))
    elif centered_init.shape[0] >= 1:
        rmsd = float(np.sqrt(np.mean(np.sum((centered_final - centered_init) ** 2, axis=1))))

    return {
        "sid": sid,
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "translation_z": float(translation[2]),
        "translation_xy": float(np.linalg.norm(translation[:2])),
        "translation_norm": float(np.linalg.norm(translation)),
        "rotation_deg": rotation_deg,
        "rmsd": rmsd,
    }


def analyze_traj(path: Path, ads_tag: int) -> Dict[str, float]:
    images = ase.io.read(path.as_posix(), ":")
    if len(images) < 2:
        raise ValueError("trajectory must contain at least two frames")

    initial = images[0]
    final = images[-1]

    return compute_motion(
        sid=path.stem,
        init_pos=initial.get_positions(),
        init_tags=np.array(initial.get_tags()),
        final_pos=final.get_positions(),
        final_tags=np.array(final.get_tags()),
        ads_tag=ads_tag,
    )


def analyze_lmdb(path: Path, ads_tag: int) -> Iterable[Dict[str, float]]:
    dataset = LmdbDataset({"src": str(path)})
    buffers: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}

    for data in dataset:
        sid = str(data.sid)
        fid = float(getattr(data, "fid", -1))
        pos = data.pos.cpu().numpy()
        tags = data.tags.cpu().numpy()

        entry = buffers.setdefault(sid, {})
        if fid == 0 and "initial" not in entry:
            entry["initial"] = (pos, tags, fid)
        # Treat fid == -1 as final. Otherwise keep the highest frame index seen.
        final_key = entry.get("final")
        if fid == -1 or final_key is None or (
            final_key[2] != -1 and fid > final_key[2]
        ):
            entry["final"] = (pos, tags, fid)

    for sid, payload in buffers.items():
        if "initial" not in payload or "final" not in payload:
            continue
        init_pos, init_tags, _ = payload["initial"]
        final_pos, final_tags, _ = payload["final"]
        try:
            result = compute_motion(sid, init_pos, init_tags, final_pos, final_tags, ads_tag)
        except Exception as exc:
            print(f"[warn] skipping {sid} in {path.name}: {exc}")
            continue
        result["sid"] = f"{sid}@{path.name}"
        yield result


def summarize(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No trajectories analyzed.")
        return

    translations = np.array([row["translation_norm"] for row in rows], dtype=float)
    translations_xy = np.array([row["translation_xy"] for row in rows], dtype=float)
    translations_z = np.array([row["translation_z"] for row in rows], dtype=float)
    rotations = np.array([row["rotation_deg"] for row in rows if not math.isnan(row["rotation_deg"])], dtype=float)

    def pct(data: np.ndarray, values=(50, 75, 90, 95, 99)) -> Dict[int, float]:
        return {p: float(np.percentile(data, p)) for p in values}

    print(f"Analyzed {len(rows)} trajectories")
    print("Translation |norm| statistics (Å):")
    print("  min: {0:.4f}  mean: {1:.4f}  max: {2:.4f}".format(translations.min(), translations.mean(), translations.max()))
    print("  percentiles:", pct(translations))

    print("Translation XY component (Å):")
    print("  mean: {0:.4f}  max: {1:.4f}".format(translations_xy.mean(), translations_xy.max()))

    print("Translation Z component (Å):")
    print("  mean: {0:.4f}  max: {1:.4f}".format(translations_z.mean(), translations_z.max()))

    if rotations.size:
        print("Rotation angle statistics (deg):")
        print("  mean: {0:.4f}  max: {1:.4f}".format(rotations.mean(), rotations.max()))
        print("  percentiles:", pct(rotations))
    else:
        print("Rotation angle statistics: insufficient adsorbate atoms (<3) to estimate rotations.")


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote per-trajectory metrics to {path}")


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, float]] = []

    if args.lmdb:
        for lmdb_path in args.lmdb:
            resolved = Path(lmdb_path).expanduser().resolve()
            if not resolved.exists():
                print(f"[warn] LMDB path not found: {resolved}")
                continue
            rows.extend(analyze_lmdb(resolved, ads_tag=args.adsorbate_tag))

    files: List[Path] = []
    for pattern in args.traj_root:
        base_path = Path(pattern)
        if base_path.is_dir():
            matches = glob.glob(str(base_path / "*.traj"))
        else:
            matches = glob.glob(pattern)
        if not matches:
            print(f"[warn] pattern returned no files: {pattern}")
        files.extend(Path(m) for m in matches)

    files = sorted({Path(f).resolve() for f in files if str(f).endswith(".traj")})
    rows_from_traj: List[Dict[str, float]] = []
    for traj_file in files:
        try:
            rows_from_traj.append(analyze_traj(traj_file, ads_tag=args.adsorbate_tag))
        except Exception as exc:
            print(f"[warn] skipping {traj_file.name}: {exc}")

    rows.extend(rows_from_traj)

    if not rows:
        print("No trajectories analyzed.")
        return

    summarize(rows)

    rows.sort(key=lambda r: r["translation_norm"], reverse=True)
    top_n = args.top
    if top_n > 0 and rows:
        print(f"\nTop {min(top_n, len(rows))} trajectories by COM translation:")
        for row in rows[:top_n]:
            print(
                f"  {row['sid']:<20} |dCOM|={row['translation_norm']:.4f} Å  "
                f"dXY={row['translation_xy']:.4f} Å  dz={row['translation_z']:.4f} Å  rot={row['rotation_deg']:.3f} deg"
            )

    if args.output_csv and rows:
        write_csv(rows, Path(args.output_csv))


if __name__ == "__main__":
    main()
