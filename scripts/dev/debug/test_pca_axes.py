#!/usr/bin/env python3
"""Utility to inspect PCA-derived molecular axes for a single LMDB sample.

Example
-------
python scripts/test_pca_axes.py --config configs/flow/painn_so3_flow.yml --split train --index 0 --use-relaxed --random-samples 3 --translation-sigma 0.2
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import yaml

from adsorbdiff.utils.registry import registry


def _orthonormalize(axes: torch.Tensor) -> torch.Tensor:
    # Re-orthogonalize in case small numerical drift occurs after permutations/sign flips.
    q, _ = torch.linalg.qr(axes, mode="reduced")
    # Ensure columns retain original orientation as much as possible by matching signs.
    sign = torch.sign(torch.sum(q * axes, dim=0, keepdim=True))
    sign[sign == 0] = 1
    return q * sign


def _resolve_dataset_cfg(args: argparse.Namespace) -> Tuple[str, dict]:
    if args.lmdb is not None:
        cfg = {"src": args.lmdb}
        return args.dataset_name, cfg

    if args.config is None:
        raise ValueError("Provide either --lmdb or --config")

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)

    dataset_entries = raw_cfg.get("dataset", [])
    if isinstance(dataset_entries, dict):
        dataset_entries = [dataset_entries]
    if not dataset_entries:
        raise RuntimeError(f"No dataset entries found in {config_path}")

    split_map = {
        "train": 0,
        "val": 1,
        "test": 2,
    }
    split_idx = split_map.get(args.split, 0)
    if split_idx >= len(dataset_entries):
        raise IndexError(
            f"Requested split '{args.split}' but config only defines {len(dataset_entries)} dataset entries"
        )

    dataset_cfg = dataset_entries[split_idx]
    dataset_cfg = dict(dataset_cfg)

    dataset_name = raw_cfg.get("task", {}).get("dataset", args.dataset_name)
    return dataset_name, dataset_cfg


def _compute_pca_axes(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("Expecting positions with shape (N, 3)")
    if points.size(0) < 2:
        raise ValueError("Need at least two atoms to define principal directions")

    orig_dtype = points.dtype
    pts = points.to(dtype=torch.float64)
    centroid = pts.mean(dim=0, keepdim=True)
    centered = pts - centroid

    # Singular values correspond to sqrt of covariance eigenvalues.
    _, singular_vals, v_t = torch.linalg.svd(centered, full_matrices=True)
    axes = v_t.transpose(-2, -1)

    if torch.det(axes) < 0:
        axes[:, -1] = -axes[:, -1]

    n = centered.size(0)
    variances = (singular_vals**2) / max(n - 1, 1)
    if variances.numel() < 3:
        pad = torch.zeros(3 - variances.numel(), dtype=variances.dtype, device=variances.device)
        variances = torch.cat([variances, pad], dim=0)

    centroid = centroid.squeeze(0).to(dtype=orig_dtype)
    axes = axes.to(dtype=orig_dtype)
    variances = variances.to(dtype=orig_dtype)
    return centroid, axes, variances


def _align_axes_to_world(axes: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]:
    import itertools

    axes64 = axes.to(dtype=torch.float64)
    best_perm = None
    best_score = -float("inf")
    best_axes = None

    for perm in itertools.permutations(range(3)):
        perm_axes = axes64[:, perm]
        score = torch.sum(torch.abs(torch.diagonal(perm_axes))).item()
        if score > best_score:
            best_score = score
            best_perm = perm
            best_axes = perm_axes

    aligned = best_axes.clone()
    for idx in range(3):
        if aligned[idx, idx] < 0:
            aligned[:, idx] *= -1.0

    if torch.det(aligned) < 0:
        diag_abs = torch.abs(torch.diagonal(aligned))
        flip_idx = int(torch.argmin(diag_abs).item())
        aligned[:, flip_idx] *= -1.0

    aligned = _orthonormalize(aligned)
    return aligned.to(dtype=axes.dtype), best_perm, axes64[:, best_perm].to(dtype=axes.dtype)


def _pretty_tensor(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu()
    with torch.no_grad():
        flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return "[]"

    values = ", ".join(f"{float(val): .5f}" for val in flat)
    return values


def _safe_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return int(value.sum().item())
    return int(value)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.norm(axis)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )


def _matrix_to_axis_angle(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1.0e-9
    trace = torch.trace(matrix)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < eps:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=matrix.dtype, device=matrix.device)
        return axis, torch.tensor(0.0, dtype=matrix.dtype, device=matrix.device)

    if math.pi - float(theta) < 1.0e-5:
        diag = torch.diagonal(matrix) + 1.0
        axis = torch.sqrt(torch.clamp(diag / 2.0, min=eps))
        axis = axis / torch.linalg.norm(axis)
        return axis, theta

    axis = torch.tensor(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=matrix.dtype,
        device=matrix.device,
    ) / (2.0 * torch.sin(theta))
    axis = axis / torch.linalg.norm(axis)
    return axis, theta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Training config with dataset entries")
    parser.add_argument("--lmdb", type=str, default=None, help="Direct path to LMDB file or directory")
    parser.add_argument("--dataset-name", type=str, default="lmdb", help="Registered dataset name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Which dataset entry to use from the config")
    parser.add_argument("--index", type=int, default=0, help="Sample index to load")
    parser.add_argument("--adsorbate-tag", type=int, default=2, help="Tag value that marks adsorbate atoms")
    parser.add_argument("--use-relaxed", action="store_true", help="Use pos_relaxed when available")
    parser.add_argument("--max-print", type=int, default=8, help="How many aligned coordinates to print")
    parser.add_argument("--random-samples", type=int, default=0, help="Number of random pose samples to generate around the relaxed structure")
    parser.add_argument("--translation-sigma", type=float, default=0.5, help="Std-dev for sampled translations (Å)")
    parser.add_argument("--interp-steps", type=int, default=4, help="Number of interpolation intervals for SO3 geodesic (>=1)")
    args = parser.parse_args()

    try:
        dataset_name, dataset_cfg = _resolve_dataset_cfg(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    dataset_class = registry.get_dataset_class(dataset_name)
    dataset = dataset_class(dataset_cfg)

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Sample index {args.index} out of bounds for dataset of length {len(dataset)}")

    sample = dataset[args.index]

    pos_attr = "pos_relaxed" if args.use_relaxed and hasattr(sample, "pos_relaxed") else "pos"
    positions = getattr(sample, pos_attr, None)
    if positions is None:
        raise AttributeError(f"Sample does not contain attribute '{pos_attr}'")

    tags = getattr(sample, "tags", None)
    if tags is None:
        raise AttributeError("Sample does not contain 'tags'")

    tags = tags.to(positions.device)
    mask = tags == args.adsorbate_tag
    if mask.sum() == 0:
        raise RuntimeError(f"No atoms found with tag {args.adsorbate_tag}")

    ads_positions = positions[mask]
    centroid, axes, variances = _compute_pca_axes(ads_positions)
    axes_aligned, perm, axes_before_sign = _align_axes_to_world(axes)
    rotation_relaxed = axes_aligned  # standard pose -> relaxed pose
    rotation_to_canonical = axes_aligned.transpose(0, 1)

    aligned = (ads_positions.to(axes_aligned.dtype) - centroid) @ axes_aligned

    print("Loaded sample")
    for key in ["id", "sid", "fid", "eid", "data_id"]:
        if hasattr(sample, key):
            print(f"  {key}: {getattr(sample, key)}")
    natoms_val = _safe_int(getattr(sample, "natoms", None), ads_positions.size(0))
    print(f"  atom_count: {natoms_val}")
    print(f"  adsorbate_atoms: {ads_positions.size(0)}")
    print(f"  position_source: {pos_attr}")

    print("\nPrincipal component frame")
    print(f"  centroid: [{_pretty_tensor(centroid)}]")
    print("  axes (columns form a rotation matrix):")
    for idx in range(3):
        print(f"    axis_{idx}: [{_pretty_tensor(axes[:, idx])}]")
    det_axes = torch.det(axes.to(dtype=torch.float64))
    print(f"  det(axes): {float(det_axes):.6f}")
    print(f"  variances: [{_pretty_tensor(variances)}]")

    print("\nAlignment to world frame")
    print(f"  best_permutation: {perm}")
    print("  axes_before_sign (permuted):")
    for idx in range(3):
        print(f"    axis_perm_{idx}: [{_pretty_tensor(axes_before_sign[:, idx])}]")
    print("  axes_aligned (world <- standard):")
    for idx in range(3):
        print(f"    axis_aligned_{idx}: [{_pretty_tensor(axes_aligned[:, idx])}]")
    print(f"  det(aligned): {float(torch.det(axes_aligned.to(dtype=torch.float64))):.6f}")

    print("\nRotation matrices")
    print("  R_relaxed (standard -> relaxed):")
    for row in rotation_relaxed:
        print(f"    [{_pretty_tensor(row)}]")
    print("  R_to_canonical (relaxed -> standard):")
    for row in rotation_to_canonical:
        print(f"    [{_pretty_tensor(row)}]")

    if args.random_samples > 0:
        torch.manual_seed(12345)
        print("\nRandom pose samples")
        interp_steps = max(args.interp_steps, 1)
        t_values = torch.linspace(0.0, 1.0, interp_steps + 1, dtype=rotation_relaxed.dtype)
        for idx in range(args.random_samples):
            axis = torch.randn(3, dtype=rotation_relaxed.dtype)
            if torch.linalg.norm(axis) < 1e-6:
                axis = torch.tensor([1.0, 0.0, 0.0], dtype=rotation_relaxed.dtype)
            angle = torch.rand((), dtype=rotation_relaxed.dtype) * (2.0 * math.pi)
            rot_noise = _axis_angle_to_matrix(axis, angle)
            translation = torch.randn(3, dtype=rotation_relaxed.dtype) * args.translation_sigma

            centered = ads_positions.to(rotation_relaxed.dtype) - centroid
            random_positions = centroid + centered @ rot_noise.T + translation
            rotation_random = rot_noise @ rotation_relaxed
            delta_rot = rotation_random.transpose(0, 1) @ rotation_relaxed
            delta_axis, delta_angle = _matrix_to_axis_angle(delta_rot)
            omega_vec = delta_axis * delta_angle
            v_tr = -translation

            print(f"  Sample {idx}:")
            print(f"    axis: [{_pretty_tensor(axis)}]")
            print(f"    angle(rad): {float(angle):.5f}")
            print(f"    translation: [{_pretty_tensor(translation)}]")
            print("    rotation_random (standard -> random):")
            for row in rotation_random:
                print(f"      [{_pretty_tensor(row)}]")
            delta = random_positions[: min(args.max_print, random_positions.size(0))] - centroid
            print("    sample_coords (first atoms relative to centroid):")
            for atom_idx, vec in enumerate(delta):
                print(f"      atom {atom_idx:02d}: [{_pretty_tensor(vec)}]")
            if random_positions.size(0) > args.max_print:
                print(f"      ... ({random_positions.size(0) - args.max_print} more atoms)")

            print("    SO3 geodesic interpolation (random -> relaxed):")
            for t in t_values:
                rot_t = rotation_random @ _axis_angle_to_matrix(delta_axis, delta_angle * t)
                trans_t = translation * (1.0 - t)
                print(f"      t={float(t):.3f}")
                print("        rotation_t:")
                for row in rot_t:
                    print(f"          [{_pretty_tensor(row)}]")
                print(f"        translation_t: [{_pretty_tensor(trans_t)}]")
                print(f"        v_rot: [{_pretty_tensor(omega_vec)}]")
                print(f"        v_tr: [{_pretty_tensor(v_tr)}]")

    max_rows = min(args.max_print, aligned.size(0))
    print("\nAligned coordinates (adsorbate frame)")
    for idx in range(max_rows):
        row = aligned[idx]
        print(f"  atom {idx:02d}: [{_pretty_tensor(row)}]")
    if aligned.size(0) > max_rows:
        print(f"  ... ({aligned.size(0) - max_rows} more atoms)")

    if hasattr(dataset, "close_db"):
        dataset.close_db()


if __name__ == "__main__":
    torch.set_printoptions(precision=5, sci_mode=False)
    main()
