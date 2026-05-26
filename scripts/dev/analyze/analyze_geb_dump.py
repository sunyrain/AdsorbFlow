#!/usr/bin/env python
"""Utility to inspect PaiNN GatedEquivariantBlock debug dumps."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Any

import torch


def _tensor_stats(tensor: torch.Tensor) -> dict:
    if tensor.numel() == 0:
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": 0,
            "finite_ratio": 1.0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "abs_max": math.nan,
            "non_finite_preview": [],
        }

    flat = tensor.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_values = flat[finite_mask]
    finite_ratio = float(finite_values.numel()) / float(flat.numel())
    if finite_values.numel() == 0:
        stats = {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    else:
        stats = {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
        }

    abs_vals = torch.nan_to_num(flat.abs(), nan=0.0, posinf=0.0)
    abs_max = float(abs_vals.max().item()) if abs_vals.numel() > 0 else math.nan

    non_finite_preview = []
    if not finite_mask.all():
        bad_values = flat[~finite_mask]
        non_finite_preview = [float(v) for v in bad_values[:5].tolist()]

    stats.update(
        {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "finite_ratio": finite_ratio,
            "abs_max": abs_max,
            "non_finite_preview": non_finite_preview,
        }
    )
    return stats


def _topk_sample_max(tensor: torch.Tensor, k: int) -> list[dict]:
    if tensor.ndim < 2 or tensor.shape[0] == 0 or k <= 0:
        return []
    sample_dim = tensor.shape[0]
    reshaped = tensor.reshape(sample_dim, -1)
    sample_max, sample_argmax = torch.nan_to_num(reshaped.abs(), nan=0.0, posinf=0.0).max(dim=1)
    k = min(k, sample_dim)
    if k <= 0:
        return []
    vals, idx = torch.topk(sample_max, k=k)
    return [
        {
            "sample_index": int(sample_idx),
            "abs_max": float(val.item()),
            "flat_idx": int(sample_argmax[sample_idx]),
        }
        for sample_idx, val in zip(idx.tolist(), vals)
    ]


def _indent(level: int) -> str:
    return "  " * level


def describe_tensor(name: str, tensor: Optional[torch.Tensor], *, topk: int, indent: int = 0) -> None:
    pad = _indent(indent)
    if tensor is None:
        print(f"{pad}- {name}: <missing>")
        return
    tensor = tensor.detach().cpu().float()
    stats = _tensor_stats(tensor)
    print(f"{pad}- {name}:")
    print(
        f"{pad}  shape={stats['shape']} dtype={stats['dtype']} numel={stats['numel']} "
        f"finite_ratio={stats['finite_ratio']:.6f} abs_max={stats['abs_max']:.4e}"
    )
    print(
        f"{pad}  min={stats['min']:.4e} max={stats['max']:.4e} "
        f"mean={stats['mean']:.4e} std={stats['std']:.4e}"
    )
    if stats["non_finite_preview"]:
        print(f"{pad}  non-finite preview: {stats['non_finite_preview']}")
    sample_infos = _topk_sample_max(tensor, topk)
    if sample_infos:
        print(f"{pad}  top-|sample| abs maxima:")
        for entry in sample_infos:
            print(
                f"{pad}    sample={entry['sample_index']} "
                f"abs_max={entry['abs_max']:.4e} flat_idx={entry['flat_idx']}"
            )


def _preview_sequence(seq: Sequence[Any], *, indent: int, topk: int, prefix: str) -> None:
    pad = _indent(indent)
    max_preview = 10
    length = len(seq)
    print(f"{pad}- {prefix}: <{type(seq).__name__}> len={length}")
    if length == 0:
        return
    for idx, item in enumerate(seq[:max_preview]):
        describe_value(f"[{idx}]", item, indent=indent + 1, topk=topk)
    if length > max_preview:
        print(f"{pad}  ... ({length - max_preview} more entries)")


def describe_value(name: str, value: Any, *, indent: int = 0, topk: int) -> None:
    pad = _indent(indent)
    if torch.is_tensor(value):
        describe_tensor(name, value, topk=topk, indent=indent)
        return
    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{pad}- {name}: <dict> keys={keys}")
        for key in keys:
            describe_value(str(key), value[key], indent=indent + 1, topk=topk)
        return
    if isinstance(value, (list, tuple)):
        _preview_sequence(value, indent=indent, topk=topk, prefix=name)
        return
    if isinstance(value, (int, float, str, bool)) or value is None:
        print(f"{pad}- {name}: {value!r}")
        return
    print(f"{pad}- {name}: <{type(value).__name__}> {value}")


def analyze_dump(path: Path, keys: Iterable[str], topk: int) -> None:
    data = torch.load(path, map_location="cpu")
    label = data.get("label", data.get("reason", "<unknown>"))
    print(f"Loaded dump '{path}' label='{label}' available_keys={list(data.keys())}")

    keys_to_show = list(keys) if keys else list(data.keys())
    if not keys_to_show:
        print("No keys to display.")
        return

    for key in keys_to_show:
        if key not in data:
            print(f"- {key}: <missing key>")
            continue
        describe_value(key, data[key], indent=0, topk=topk)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_path", type=Path, help="Path to geb_*.pt dump file")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Subset of tensor keys to report. Defaults to all tensor-valued entries.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1000,
        help="How many sample-level maxima to show per tensor (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_dump(args.dump_path, args.keys, args.topk)


if __name__ == "__main__":
    main()
