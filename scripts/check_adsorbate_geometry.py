#!/usr/bin/env python
"""Inspect adsorbate geometries for linear or planar degeneracies.

This utility walks over one or more LMDB datasets and classifies the adsorbate
geometry in each structure as linear, planar, full-rank (3D), or degenerate.
The classification logic mirrors the PCA code used during training, so the
results are directly comparable with the runtime behaviour of
``compute_adsorbate_frames``.

Example
-------
$ python scripts/check_adsorbate_geometry.py \
    /path/to/train_split \
    --limit 10000 --progress --use-relaxed

"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

from adsorbdiff.datasets import LmdbDataset
from adsorbdiff.utils.pose_utils import compute_adsorbate_frames


_CLASS_NAMES = {
    "no_adsorbate": "no_adsorbate",
    "single_atom": "single_atom",
    "invalid": "invalid",
    "linear": "linear",
    "planar": "planar",
    "full_rank": "full_rank",
}


def _compute_eigenvalues(points: torch.Tensor) -> torch.Tensor:
    """Return eigenvalues (descending) of the covariance of ``points``.

    Parameters
    ----------
    points: torch.Tensor, shape (N, 3)
        Adsorbate coordinates.
    """
    if points.dim() != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.size(0) < 2:
        return torch.zeros(3, dtype=points.dtype, device=points.device)
    centered = points - points.mean(dim=0, keepdim=True)
    if torch.allclose(centered, torch.zeros_like(centered), atol=1.0e-12):
        return torch.zeros(3, dtype=points.dtype, device=points.device)
    cov = centered.t().matmul(centered)
    evals = torch.linalg.eigvalsh(cov.double())
    evals = torch.flip(evals, dims=[0])  # descending order
    return evals.to(points.dtype)


def _projected_ratios(evals: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """Return λ₂/λ₁ and λ₃/λ₂ when denominators are positive."""
    if evals.numel() != 3:
        return None, None
    lam1 = float(evals[0].item())
    lam2 = float(evals[1].item())
    lam3 = float(evals[2].item())
    ratio21 = lam2 / lam1 if lam1 > 0.0 else None
    ratio32 = lam3 / lam2 if lam2 > 0.0 else None
    return ratio21, ratio32


def _extract_identifier(data, fallback: str) -> str:
    """Build a human-readable identifier for a dataset sample."""
    parts: List[str] = []
    for key in ["sid", "fid", "eid", "tid", "traj_id", "data_id", "id"]:
        if hasattr(data, key):
            value = getattr(data, key)
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.tolist()
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else fallback


def _classify_adsorbate(
    data,
    idx: int,
    use_relaxed: bool,
) -> Optional[Dict]:
    if not hasattr(data, "tags"):
        return None

    tags = data.tags
    if not torch.is_tensor(tags):
        tags = torch.as_tensor(tags)
    ads_mask = tags == 2
    ads_count = int(ads_mask.sum().item())

    positions = None
    if use_relaxed and hasattr(data, "pos_relaxed") and data.pos_relaxed is not None:
        positions = data.pos_relaxed
    if positions is None:
        positions = data.pos

    if positions is None:
        return None

    positions = positions.clone().detach()
    if positions.dim() != 2 or positions.size(-1) != 3:
        raise ValueError("Positions tensor must have shape (N, 3)")

    identifier = _extract_identifier(data, fallback=str(idx))

    if ads_count == 0:
        return {
            "classification": _CLASS_NAMES["no_adsorbate"],
            "ads_count": ads_count,
            "pose_valid": False,
            "degeneracy": -1,
            "evals": [0.0, 0.0, 0.0],
            "ratio21": None,
            "ratio32": None,
            "id": identifier,
        }
    if ads_count == 1:
        return {
            "classification": _CLASS_NAMES["single_atom"],
            "ads_count": ads_count,
            "pose_valid": False,
            "degeneracy": -1,
            "evals": [0.0, 0.0, 0.0],
            "ratio21": None,
            "ratio32": None,
            "id": identifier,
        }

    batch_idx = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)
    centroids, axes_wc, _, pose_valid, degeneracy = compute_adsorbate_frames(
        positions,
        batch_idx,
        ads_mask,
    )
    _ = centroids  # unused, silence linter
    _ = axes_wc

    pose_valid_flag = bool(pose_valid[0].item()) if pose_valid.numel() else False
    deg_code = int(degeneracy[0].item()) if degeneracy.numel() else -1

    if not pose_valid_flag and deg_code <= 0:
        classification = _CLASS_NAMES["invalid"]
    elif deg_code == 1:
        classification = _CLASS_NAMES["linear"]
    elif deg_code == 2:
        classification = _CLASS_NAMES["planar"]
    else:
        classification = _CLASS_NAMES["full_rank"]

    ads_points = positions[ads_mask].to(torch.float64)
    evals = _compute_eigenvalues(ads_points).cpu()
    ratio21, ratio32 = _projected_ratios(evals)

    return {
        "classification": classification,
        "ads_count": ads_count,
        "pose_valid": pose_valid_flag,
        "degeneracy": deg_code,
        "evals": [float(x) for x in evals.tolist()],
        "ratio21": ratio21,
        "ratio32": ratio32,
        "id": identifier,
    }


def _init_ratio_stats() -> Dict[str, float]:
    return {"count": 0, "sum": 0.0, "min": math.inf, "max": 0.0}


def _update_ratio_stats(stats: Dict[str, float], value: float) -> None:
    if value is None:
        return
    stats["count"] += 1
    stats["sum"] += value
    stats["min"] = min(stats["min"], value)
    stats["max"] = max(stats["max"], value)


def _finalise_stats(stats: Dict[str, float]) -> Dict[str, float]:
    if stats["count"] == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(stats["count"]),
        "mean": stats["sum"] / stats["count"],
        "min": stats["min"],
        "max": stats["max"],
    }


def analyse_dataset(
    src: Path,
    *,
    stride: int,
    limit: Optional[int],
    use_relaxed: bool,
    progress: bool,
    max_examples: int,
) -> Dict:
    dataset = LmdbDataset({"src": str(src)})
    try:
        indices: Iterable[int] = range(0, len(dataset), stride)
        max_len = len(range(0, len(dataset), stride))
        total_steps = max_len if limit is None else min(max_len, limit)
        iterator = indices
        if progress:
            iterator = tqdm(indices, total=total_steps, desc=f"scan {src}", leave=False)

        class_counts: Counter = Counter()
        examples: Dict[str, List[Dict]] = defaultdict(list)
        ratio21_stats = _init_ratio_stats()
        ratio32_stats = _init_ratio_stats()

        processed = 0
        for local_step, idx in enumerate(iterator):
            if limit is not None and processed >= limit:
                break
            data = dataset[idx]
            result = _classify_adsorbate(data, idx=idx, use_relaxed=use_relaxed)
            if result is None:
                continue
            cls = result["classification"]
            class_counts[cls] += 1
            if result["ratio21"] is not None:
                _update_ratio_stats(ratio21_stats, result["ratio21"])
            if result["ratio32"] is not None:
                _update_ratio_stats(ratio32_stats, result["ratio32"])
            if len(examples[cls]) < max_examples:
                examples[cls].append(
                    {
                        "id": result["id"],
                        "ads_count": result["ads_count"],
                        "degeneracy": result["degeneracy"],
                        "evals": result["evals"],
                        "ratio21": result["ratio21"],
                        "ratio32": result["ratio32"],
                    }
                )
            processed += 1

        analyzable = processed - class_counts[_CLASS_NAMES["no_adsorbate"]] - class_counts[_CLASS_NAMES["single_atom"]]

        summary = {
            "source": str(src),
            "structures_processed": processed,
            "stride": stride,
            "limit": limit,
            "use_relaxed": use_relaxed,
            "counts": {cls: int(class_counts[cls]) for cls in sorted(_CLASS_NAMES.values()) if class_counts[cls] > 0},
            "examples": {cls: examples[cls] for cls in sorted(examples.keys())},
            "analyzable_structures": int(analyzable),
            "ratio21": _finalise_stats(ratio21_stats),
            "ratio32": _finalise_stats(ratio32_stats),
        }
    finally:
        dataset.close_db()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check adsorbate PCA degeneracies.")
    parser.add_argument("src", nargs="+", help="LMDB file or directory")
    parser.add_argument("--stride", type=int, default=1, help="Stride when traversing dataset (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of structures to inspect")
    parser.add_argument("--use-relaxed", action="store_true", help="Use pos_relaxed when available")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bar")
    parser.add_argument("--max-examples", type=int, default=5, help="Number of example IDs to retain per class")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to dump results")
    args = parser.parse_args()

    results = []
    for src in args.src:
        summary = analyse_dataset(
            Path(src),
            stride=args.stride,
            limit=args.limit,
            use_relaxed=args.use_relaxed,
            progress=args.progress,
            max_examples=args.max_examples,
        )
        results.append(summary)

        print("=" * 72)
        print(f"Dataset: {summary['source']}")
        print(f"Processed structures: {summary['structures_processed']} (stride={summary['stride']}, limit={summary['limit']})")
        analyzable = summary["analyzable_structures"]
        print(f"Analyzable (>=2 ads atoms): {analyzable}")
        counts = summary["counts"]
        total = max(summary["structures_processed"], 1)
        for cls in sorted(counts.keys()):
            count = counts[cls]
            pct = 100.0 * count / total
            print(f"  {cls:12s}: {count:8d} ({pct:5.2f} %)")
        print("  ratio λ₂/λ₁:", summary["ratio21"])
        print("  ratio λ₃/λ₂:", summary["ratio32"])
        for cls, samples in summary["examples"].items():
            print(f"  examples[{cls}] ({len(samples)}):")
            for sample in samples:
                print(
                    f"    - {sample['id']} | ads={sample['ads_count']} | deg={sample['degeneracy']} | "
                    f"evals={sample['evals']} | r21={sample['ratio21']} | r32={sample['ratio32']}"
                )

    if args.output is not None:
        serialisable = []
        for summary in results:
            serialisable.append(summary)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
