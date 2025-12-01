#!/usr/bin/env python3
"""Check LMDB energy offsets per system and report distribution statistics."""

# Usage: python scripts/check_lmdb_energy_offsets.py train_allE --progress

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LMDB energy offsets per system.")
    parser.add_argument("lmdb_root", type=Path, help="Path to an LMDB file or directory of shards.")
    parser.add_argument("--energy-key", default="y", help="Tensor attribute holding the energy label.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of items to scan.")
    parser.add_argument("--zero-tol", type=float, default=1e-6, help="Tolerance when checking whether minima are zero.")
    parser.add_argument("--progress", action="store_true", help="Print progress every 10k samples.")
    return parser.parse_args()


def _norm_tensor_as_float(value) -> float:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"Energy tensor must be scalar, got shape {tuple(tensor.shape)}")
    return float(tensor.item())


def _normalize_sid(raw_sid) -> str:
    if raw_sid is None:
        return ""
    if isinstance(raw_sid, bytes):
        return raw_sid.decode("utf-8")
    if isinstance(raw_sid, str):
        return raw_sid
    if isinstance(raw_sid, torch.Tensor):
        if raw_sid.numel() == 1:
            return str(raw_sid.item())
        return "_".join(str(int(x)) for x in raw_sid.view(-1).tolist())
    return str(raw_sid)


def _iter_indices(dataset: LmdbDataset, max_samples: Optional[int]) -> Iterable[int]:
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(total):
        yield idx


def _summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()

    dataset = LmdbDataset({"src": str(args.lmdb_root)})

    per_system: Dict[str, List[float]] = defaultdict(list)
    global_energies: List[float] = []
    missing_sid = 0

    for idx in _iter_indices(dataset, args.max_samples):
        data = dataset[idx]
        if not hasattr(data, args.energy_key):
            raise AttributeError(f"Sample {idx} missing attribute '{args.energy_key}'")
        energy = _norm_tensor_as_float(getattr(data, args.energy_key))
        global_energies.append(energy)

        sid = _normalize_sid(getattr(data, "sid", None))
        if not sid:
            missing_sid += 1
        else:
            per_system[sid].append(energy)

        if args.progress and (idx + 1) % 10_000 == 0:
            print(f"Scanned {idx + 1} samples ...")

    dataset.close_db()

    violations: List[Tuple[str, float]] = []
    negative_offsets: List[Tuple[str, float]] = []

    for sid, energies in per_system.items():
        min_energy = min(energies)
        if not math.isclose(min_energy, 0.0, abs_tol=args.zero_tol):
            violations.append((sid, min_energy))
        for rel_energy in energies:
            if rel_energy < -args.zero_tol:
                negative_offsets.append((sid, rel_energy))

    summary = _summarize(global_energies)

    print("=== Energy Offset Check ===")
    print(f"LMDB root: {args.lmdb_root}")
    print(f"Samples scanned: {len(global_energies)}")
    print(f"Systems covered: {len(per_system)}")
    print(f"Samples without sid: {missing_sid}")
    print(f"Systems with non-zero minima (> tol): {len(violations)}")
    if violations:
        preview = ", ".join(f"{sid}:{min_val:.3e}" for sid, min_val in violations[:10])
        print(f"  Example violations: {preview}")
    print(f"Samples with negative relative energy (< -tol): {len(negative_offsets)}")
    if negative_offsets:
        preview = ", ".join(f"{sid}:{val:.3e}" for sid, val in negative_offsets[:10])
        print(f"  Example negatives: {preview}")

    if summary:
        print("=== Global Energy Distribution ===")
        for key in ["count", "mean", "std", "min", "q25", "median", "q75", "q90", "q99", "max"]:
            print(f"{key:>7}: {summary[key]:.6f}")
    else:
        print("No energies collected; nothing to report.")


if __name__ == "__main__":
    main()
