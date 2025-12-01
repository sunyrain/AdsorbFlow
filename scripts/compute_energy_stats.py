#!/usr/bin/env python3
"""Compute mean/variance (and min/max) of energy labels in LMDB datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def finalize(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No samples were processed.")
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        std = float(np.sqrt(variance))
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": variance,
            "std": std,
            "min": self.minimum,
            "max": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute energy statistics for LMDB datasets")
    parser.add_argument("lmdb", nargs="+", help="One or more LMDB directories or files")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only consider this many samples per LMDB",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump the statistics as JSON",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a tqdm progress bar while iterating",
    )
    return parser.parse_args()


def extract_energy(sample) -> float:
    for attr in ("energy", "y", "target", "y_relaxed"):
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() == 0:
                    continue
                return float(value.detach().cpu().view(-1)[0].item())
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
            if isinstance(value, (int, float, np.floating)):
                return float(value)
    raise AttributeError("Sample does not carry an energy-like attribute")


def iter_indices(length: int, limit: Optional[int]) -> Iterable[int]:
    if limit is None or limit >= length:
        return range(length)
    return range(limit)


def process_lmdb(
    path: Path,
    max_samples: Optional[int],
    show_progress: bool,
    aggregate: Optional[RunningStats] = None,
) -> RunningStats:
    dataset = LmdbDataset({"src": str(path)})
    stats = RunningStats()
    indices = iter_indices(len(dataset), max_samples)
    iterable = tqdm(indices, desc=path.name, unit="sample") if show_progress else indices
    try:
        for idx in iterable:
            sample = dataset[idx]
            try:
                energy = extract_energy(sample)
            except AttributeError:
                continue
            stats.update(energy)
            if aggregate is not None:
                aggregate.update(energy)
    finally:
        dataset.close_db()
    return stats


def main() -> None:
    args = parse_args()
    aggregate = RunningStats()
    results = {}

    for entry in args.lmdb:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"LMDB path not found: {path}")
        stats = process_lmdb(path, args.max_samples, args.show_progress, aggregate)
        if stats.count == 0:
            print(f"\n=== {path} ===")
            print("No energy entries found in this dataset.")
            continue
        summary = stats.finalize()
        results[str(path)] = summary
        print(f"\n=== {path} ===")
        for key, value in summary.items():
            print(f"{key:>8}: {value:.6e}" if isinstance(value, float) else f"{key:>8}: {value}")

    if aggregate.count > 0:
        combined = aggregate.finalize()
        results["combined"] = combined
        print("\n=== Combined ===")
        for key, value in combined.items():
            print(f"{key:>8}: {value:.6e}" if isinstance(value, float) else f"{key:>8}: {value}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved statistics to {output_path}")

if __name__ == "__main__":
    main()
