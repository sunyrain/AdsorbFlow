#!/usr/bin/env python3
"""Compute descriptive statistics for energy labels stored in LMDB datasets.

This helper walks over one or more LMDB directories and gathers the per-structure
energy values (``energy`` or ``y``) to report a summary of their distribution.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize energy distribution across LMDB datasets")
    parser.add_argument(
        "--lmdb",
        nargs="+",
        required=True,
        help="One or more LMDB directories (or single .lmdb files) to inspect",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of items to read per LMDB (default: all)",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Sample random entries instead of taking the first N (requires --max-samples)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed when --random-sample is enabled (default: 0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="If >0, also print a histogram with this many bins",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="*",
        default=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100),
        help="Percentiles to report (default: common summary percentiles)",
    )
    return parser.parse_args()


def extract_energy(sample) -> float:
    for attr in ("energy", "y", "target", "y_relaxed"):
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu()
                if value.numel() == 1:
                    return float(value.item())
                return float(value.view(-1)[0].item())
            if isinstance(value, (list, tuple)):
                arr = np.asarray(value, dtype=np.float64)
                if arr.size:
                    return float(arr.reshape(-1)[0])
            if isinstance(value, (int, float, np.floating)):
                return float(value)
    raise AttributeError("Sample does not contain an energy-like attribute")


def iter_indices(length: int, max_samples: int | None, random_sample: bool, rng: random.Random) -> Iterable[int]:
    if max_samples is None or max_samples >= length:
        return range(length)
    if random_sample:
        return rng.sample(range(length), k=max_samples)
    return range(max_samples)


def summarize(values: Sequence[float], percentiles: Sequence[float], bins: int) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        print("No energy values collected.")
        return

    mean = float(arr.mean())
    std = float(arr.std())
    min_val = float(arr.min())
    max_val = float(arr.max())

    print(f"count: {arr.size}")
    print(f"mean:  {mean:.6e}")
    print(f"std:   {std:.6e}")
    print(f"min:   {min_val:.6e}")
    print(f"max:   {max_val:.6e}")

    percentile_values = np.percentile(arr, percentiles)
    header = "percentile | value"
    print("\n" + header)
    print("-" * len(header))
    for p, v in zip(percentiles, percentile_values):
        print(f"{p:9.2f} | {float(v): .6e}")

    if bins > 0:
        hist, edges = np.histogram(arr, bins=bins)
        print("\nHistogram (bin edges in eV):")
        for left, right, count in zip(edges[:-1], edges[1:], hist):
            print(f"[{left: .6e}, {right: .6e}) : {int(count)}")


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    all_values: List[float] = []

    for lmdb_path in args.lmdb:
        path = Path(lmdb_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"LMDB path not found: {path}")
        print(f"\n=== {path} ===")
        dataset = LmdbDataset({"src": str(path)})
        indices = iter_indices(len(dataset), args.max_samples, args.random_sample, rng)
        values: List[float] = []
        pbar = tqdm(indices, desc=path.name, unit="sample")
        try:
            for idx in pbar:
                sample = dataset[idx]
                try:
                    energy = extract_energy(sample)
                except AttributeError:
                    continue
                values.append(energy)
        finally:
            dataset.close_db()
        if not values:
            print("No energy values extracted from this dataset.")
        else:
            summarize(values, args.percentiles, args.bins)
            all_values.extend(values)

    if len(args.lmdb) > 1 and all_values:
        print("\n=== Combined ===")
        summarize(all_values, args.percentiles, args.bins)


if __name__ == "__main__":
    main()
