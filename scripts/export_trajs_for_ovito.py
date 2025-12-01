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
from pathlib import Path
from typing import Dict, Iterable, Optional

import ase.io
import numpy as np
from ase import Atoms

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Flow/relaxation .traj files to .xyz for OVITO")
    parser.add_argument(
        "--flow-dir",
        required=True,
        help="Directory containing Flow output .traj files",
    )
    parser.add_argument(
        "--relax-dir",
        required=True,
        help="Directory containing relaxation .traj files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for exported XYZ files",
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
        default=None,
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
    flow_dir = Path(args.flow_dir).resolve()
    relax_dir = Path(args.relax_dir).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not flow_dir.exists():
        raise FileNotFoundError(f"Flow directory not found: {flow_dir}")
    if not relax_dir.exists():
        raise FileNotFoundError(f"Relax directory not found: {relax_dir}")

    sids = args.sids
    flow_single = args.single_flow_frame
    fmt = args.format
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

        flow_out = out_dir / f"{sid}_flow.{fmt if fmt != 'extxyz' else 'xyz'}"
        relax_out = out_dir / f"{sid}_relax.{fmt if fmt != 'extxyz' else 'xyz'}"
        initial_out = out_dir / f"{sid}_initial.{fmt if fmt != 'extxyz' else 'xyz'}"

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
