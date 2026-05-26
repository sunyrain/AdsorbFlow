#!/usr/bin/env python3
"""Export Flow and relaxation trajectories to OVITO-friendly formats.

Reads *.traj files produced by Flow sampling and subsequent relaxation, then
writes them out as extended XYZ files (one per trajectory) so they can be
inspected locally with OVITO or ASE.

Example:
    python scripts/export_trajs_for_ovito.py --flow-dir grid_search_runs/cfg10_steps100/0 --relax-dir relaxations --output exports/ovito_xyz --lmdb val_nonrelaxed_update --single-flow-frame
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import ase.io
import numpy as np
from ase import Atoms

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Flow/relaxation .traj files to .xyz for OVITO")
    parser.add_argument(
        "--run_dir",
        help="Directory containing Flow output .traj files",
    )
    parser.add_argument(
        "--sids",
        nargs="*",
        default=None,
        help="Optional subset of sid filenames to export (without extension). If omitted, export all",
    )
    parser.add_argument(
        "--single-flow-frame",
        action="store_true",
        help="If set, only write the final frame from the Flow trajectory",
    )
    parser.add_argument(
        "--lmdb",
        default="val_nonrelaxed_update",
        help="Optional LMDB directory containing the initial structures for comparison",
    )
    parser.add_argument(
        "--initial-dir",
        default=None,
        help="Optional directory with pre-exported initial *.traj files (takes precedence over LMDB)",
    )
    parser.add_argument(
        "--format",
        default="extxyz",
        help="ASE-supported output format (default: extxyz)",
    )
    parser.add_argument(
        "--ckpt-label",
        default="",
        help="Optional label for the checkpoint used (e.g. 'epoch50')",
    )
    parser.add_argument(
        "--cfg",
        default="",
        help="Optional CFG scale to include in filename (e.g. '5.0')",
    )
    parser.add_argument(
        "--steps",
        default="10",
        help="Optional step count to include in filename (e.g. '50')",
    )
    return parser.parse_args()


def iter_traj_files(root: Path, sids: Iterable[str] | None) -> Iterable[Path]:
    if sids is None:
        yield from sorted(root.glob("*.traj"))
    else:
        for sid in sids:
            candidate = root / f"{sid}.traj"
            if not candidate.exists():
                raise FileNotFoundError(f"Trajectory not found: {candidate}")
            yield candidate


def read_anomalous_sids(relax_dir: Path) -> Optional[list[str]]:
    """Read anomalous_structures_new.txt if present and return list of SIDs.

    The file is expected to have a header, a separator line, and then rows like:
    "SID | FID | Anomalies". We take the first column as sid.
    """
    txt_path = relax_dir / "anomalous_structures_new.txt"
    if not txt_path.exists():
        return None
    sids: list[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SID ") or line.startswith("-"):
                continue
            parts = line.split()
            if not parts:
                continue
            sids.append(parts[0])
    return sids if sids else None


def _to_atoms(record) -> Atoms:
    atoms = Atoms(numbers=record.atomic_numbers.numpy())
    atoms.set_positions(record.pos.numpy())
    atoms.set_tags(record.tags.numpy())
    cell = record.cell
    if hasattr(cell, "numpy"):
        cell = cell.numpy()
    cell = np.asarray(cell)
    if cell.shape == (3, 3):
        pass
    elif cell.shape == (3,):
        cell = np.diag(cell)
    elif cell.size == 9:
        cell = cell.reshape(3, 3)
    else:
        raise ValueError(f"Unexpected cell shape for sid {record.sid}: {cell.shape}")
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])
    return atoms


def build_initial_cache(lmdb_path: Path, target_sids: Optional[Iterable[str]]) -> Dict[str, Atoms]:
    dataset = LmdbDataset({"src": str(lmdb_path), "key_mapping": {"y": "energy", "force": "forces"}})
    wanted = set(target_sids) if target_sids is not None else None
    cache: Dict[str, Atoms] = {}
    for record in dataset:
        sid = str(record.sid)
        if wanted is not None and sid not in wanted:
            continue
        cache[sid] = _to_atoms(record)
        if wanted is not None and len(cache) == len(wanted):
            break
    return cache


def match_sid(sid: str, cache: Dict[str, Atoms]) -> Optional[str]:
    if sid in cache:
        return sid
    parts = sid.split("_")
    while len(parts) > 2:
        parts = parts[:-1]
        candidate = "_".join(parts)
        if candidate in cache:
            return candidate
    return None


def export_single(traj_path: Path, out_path: Path, fmt: str, single_flow: bool = False) -> None:
    images = ase.io.read(traj_path.as_posix(), ":")
    if not images:
        raise RuntimeError(f"No frames in {traj_path}")
    if single_flow:
        images = [images[-1]]
    ase.io.write(out_path.as_posix(), images, format=fmt)


def main() -> None:
    args = parse_args()
    flow_dir = Path(args.run_dir).resolve()
    relax_dir = flow_dir / "relaxations"

    try:
        rel_path = flow_dir.relative_to(Path.cwd())
        out_name = str(rel_path).replace("/", "_")
    except ValueError:
        out_name = flow_dir.name

    out_dir = Path("exports") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not flow_dir.exists():
        raise FileNotFoundError(f"Flow directory not found: {flow_dir}")
    if not relax_dir.exists():
        raise FileNotFoundError(f"Relax directory not found: {relax_dir}")

    # If anomaly file exists, override sids with the anomalous list
    sids = args.sids
    anomalous_sids = read_anomalous_sids(relax_dir)
    if anomalous_sids is not None:
        sids = anomalous_sids
        print(f"[info] Found anomalous_structures_new.txt; exporting {len(sids)} anomalous trajectories only.")
    flow_single = args.single_flow_frame
    fmt = args.format
    ext = "xyz" if fmt == "extxyz" else fmt

    # Infer metadata from path if not provided
    cfg_val = args.cfg
    steps_val = args.steps

    if not cfg_val or not steps_val:
        path_str = str(flow_dir)
        if not cfg_val:
            m = re.search(r"cfg([0-9\.]+)", path_str)
            if m: cfg_val = m.group(1)
        if not steps_val:
            m = re.search(r"steps([0-9]+)", path_str)
            if m: steps_val = m.group(1)

    # Construct filename suffix from metadata
    meta_parts = []
    if args.ckpt_label:
        meta_parts.append(args.ckpt_label)
    if cfg_val:
        meta_parts.append(f"cfg{cfg_val}")
    if steps_val:
        meta_parts.append(f"steps{steps_val}")

    meta_suffix = "_" + "_".join(meta_parts) if meta_parts else ""

    initial_dir = Path(args.initial_dir).resolve() if args.initial_dir else None
    initial_cache: Dict[str, Atoms] = {}

    if args.lmdb:
        lmdb_path = Path(args.lmdb).resolve()
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB directory not found: {lmdb_path}")
        lmdb_targets = sids if sids is not None else None
        initial_cache = build_initial_cache(lmdb_path, lmdb_targets)
        if not initial_cache and lmdb_targets:
            print("[warn] no LMDB entries matched requested sids; initial structures will be skipped")

    for flow_traj in iter_traj_files(flow_dir, sids):
        sid = flow_traj.stem
        relax_traj = relax_dir / f"{sid}.traj"
        if not relax_traj.exists():
            print(f"[warn] relaxation traj missing, skipping {sid}")
            continue

        flow_out = out_dir / f"{sid}{meta_suffix}_flow.{ext}"
        relax_out = out_dir / f"{sid}{meta_suffix}_relax.{ext}"
        initial_out = out_dir / f"{sid}_initial.{ext}"

        print(f"Exporting {sid}: flow -> {flow_out.name}, relax -> {relax_out.name}")
        export_single(flow_traj, flow_out, fmt, single_flow=flow_single)
        export_single(relax_traj, relax_out, fmt, single_flow=False)

        initial_atoms: Optional[Atoms] = None
        if initial_dir and (initial_dir / f"{sid}.traj").exists():
            initial_atoms = ase.io.read((initial_dir / f"{sid}.traj").as_posix())
        elif initial_cache:
            matched = match_sid(sid, initial_cache)
            if matched:
                initial_atoms = initial_cache[matched]
        if initial_atoms is not None:
            ase.io.write(initial_out.as_posix(), initial_atoms, format=fmt)
        else:
            print(f"  [warn] initial frame unavailable for {sid}; skipped initial export")


if __name__ == "__main__":
    main()
