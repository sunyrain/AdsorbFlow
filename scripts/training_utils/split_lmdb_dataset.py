#!/usr/bin/env python3
"""Utility to split an LMDB dataset into train and validation subsets."""

# Usage example:
#   python scripts/split_lmdb_dataset.py train_allE train_split val_split --train-ratio 0.9 --seed 42

from __future__ import annotations

import argparse
import bisect
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import lmdb
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split LMDB dataset into train and validation sets.")
    parser.add_argument("input", type=Path, help="Path to the source LMDB directory or file.")
    parser.add_argument("train_output", type=Path, help="Output directory for the train split.")
    parser.add_argument("val_output", type=Path, help="Output directory for the validation split.")
    parser.add_argument("--train-ratio", type=float, default=0.9, dest="train_ratio",
                        help="Fraction of samples to place in the train split (default: 0.9).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for shuffling.")
    parser.add_argument("--commit-interval", type=int, default=2048,
                        help="Number of samples written per LMDB transaction commit (default: 2048).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output LMDB files if present.")
    return parser.parse_args()


def connect_read_env(db_path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(db_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )


def connect_write_env(db_path: Path, map_size: int) -> lmdb.Environment:
    return lmdb.open(
        str(db_path),
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
        map_size=map_size,
    )


def get_num_entries(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        length_bytes = txn.get(b"length")
    if length_bytes is not None:
        return pickle.loads(length_bytes)
    stats = env.stat()
    return int(stats["entries"])


def cumulative_lengths(lengths: Sequence[int]) -> List[int]:
    total = 0
    cumsum = []
    for length in lengths:
        total += length
        cumsum.append(total)
    return cumsum


def map_global_to_local(idx: int, cum_lengths: Sequence[int]) -> Tuple[int, int]:
    db_idx = bisect.bisect_right(cum_lengths, idx)
    prev_total = 0 if db_idx == 0 else cum_lengths[db_idx - 1]
    return db_idx, idx - prev_total


def fetch_raw(env: lmdb.Environment, key: int) -> bytes:
    with env.begin(write=False) as txn:
        raw = txn.get(str(key).encode("ascii"))
    if raw is None:
        raise KeyError(f"Key {key} not found in LMDB")
    return raw


def prepare_output_dir(directory: Path, overwrite: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("data.*.lmdb")) + list(directory.glob("data.*.lmdb-lock"))
    if existing and not overwrite:
        raise FileExistsError(f"{directory} already contains LMDB files. Use --overwrite to replace.")
    for path in existing:
        if path.is_file() or path.is_symlink():
            path.unlink()


def estimate_map_size(source_files: Iterable[Path], ratio: float, floor_bytes: int = 1 << 28) -> int:
    total_bytes = sum(f.stat().st_size for f in source_files)
    estimate = int(total_bytes * max(ratio, 1e-6) * 1.1)
    return max(estimate, floor_bytes)


def write_split(
    split_indices: np.ndarray,
    output_env: lmdb.Environment,
    source_envs: Sequence[lmdb.Environment],
    cum_lengths: Sequence[int],
    commit_interval: int,
) -> int:
    count = 0
    txn = output_env.begin(write=True)
    for i, global_idx in enumerate(split_indices):
        env_idx, local_idx = map_global_to_local(int(global_idx), cum_lengths)
        raw = fetch_raw(source_envs[env_idx], local_idx)
        txn.put(str(count).encode("ascii"), raw)
        count += 1
        if commit_interval > 0 and (count % commit_interval == 0):
            txn.commit()
            txn = output_env.begin(write=True)
    txn.commit()

    with output_env.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(count, protocol=-1))
    output_env.sync()
    return count


def copy_metadata_if_present(source: Path, destination: Path) -> None:
    metadata_src = source / "metadata.npz"
    if metadata_src.exists():
        shutil.copy2(metadata_src, destination / "metadata.npz")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input path {args.input} does not exist")

    if args.input.is_file():
        input_lmdbs = [args.input]
        input_root = args.input.parent
    else:
        input_lmdbs = sorted(args.input.glob("*.lmdb"))
        input_root = args.input

    if not input_lmdbs:
        raise FileNotFoundError(f"No LMDB files found inside {args.input}")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive)")

    rng = np.random.default_rng(args.seed)

    source_envs = [connect_read_env(path) for path in input_lmdbs]
    try:
        lengths = [get_num_entries(env) for env in source_envs]
        total_samples = sum(lengths)
        if total_samples == 0:
            raise RuntimeError("Input dataset is empty")

        cum_lengths = cumulative_lengths(lengths)

        global_indices = np.arange(total_samples)
        rng.shuffle(global_indices)

        train_cutoff = int(total_samples * args.train_ratio)
        train_indices = global_indices[:train_cutoff]
        val_indices = global_indices[train_cutoff:]

        prepare_output_dir(args.train_output, args.overwrite)
        prepare_output_dir(args.val_output, args.overwrite)

        train_lmdb_path = args.train_output / "data.0000.lmdb"
        val_lmdb_path = args.val_output / "data.0000.lmdb"

        train_map_size = estimate_map_size(input_lmdbs, args.train_ratio)
        val_map_size = estimate_map_size(input_lmdbs, 1.0 - args.train_ratio)

        train_env = connect_write_env(train_lmdb_path, train_map_size)
        val_env = connect_write_env(val_lmdb_path, val_map_size)
        try:
            train_count = write_split(train_indices, train_env, source_envs, cum_lengths, args.commit_interval)
            val_count = write_split(val_indices, val_env, source_envs, cum_lengths, args.commit_interval)
        finally:
            train_env.close()
            val_env.close()

        copy_metadata_if_present(input_root, args.train_output)
        copy_metadata_if_present(input_root, args.val_output)

        print(f"Wrote {train_count} samples to {train_lmdb_path}")
        print(f"Wrote {val_count} samples to {val_lmdb_path}")
    finally:
        for env in source_envs:
            env.close()


if __name__ == "__main__":
    main()
