"""Utilities for determining canonical rigid-body frames using PCA."""

from __future__ import annotations

import itertools
import logging
import os
from typing import Optional, Tuple

import torch

_POSE_DEBUG_EVALS = bool(int(os.getenv("POSE_DEBUG_EVALS", "0")))


def _frame_from_axis(axis: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1.0e-12)
    device = axis.device
    dtype = axis.dtype
    ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    if torch.abs(torch.dot(axis, ref)) > 0.9:
        ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    x_axis = torch.cross(ref, axis)
    norm_x = torch.linalg.norm(x_axis)
    if norm_x < 1.0e-12:
        ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        x_axis = torch.cross(ref, axis)
        norm_x = torch.linalg.norm(x_axis)
    x_axis = x_axis / torch.clamp(norm_x, min=1.0e-12)
    y_axis = torch.cross(axis, x_axis)
    y_axis = y_axis / torch.clamp(torch.linalg.norm(y_axis), min=1.0e-12)
    return torch.stack([x_axis, y_axis, axis], dim=1)


def _compute_pca_axes(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, int]:
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("points must have shape (N, 3)")
    centroid = points.mean(dim=0)
    centered = points - centroid
    if centered.size(0) < 2:
        axes = torch.eye(3, device=points.device, dtype=points.dtype)
        return centroid, axes, False, -1
    centered_d = centered.double()
    cov = centered_d.transpose(0, 1).matmul(centered_d)
    if torch.allclose(cov, torch.zeros_like(cov), atol=1.0e-10):
        axes = torch.eye(3, device=points.device, dtype=points.dtype)
        return centroid, axes, False, -1
    try:
        evals, evecs = torch.linalg.eigh(cov)
    except RuntimeError:
        axes = torch.eye(3, device=points.device, dtype=points.dtype)
        if _POSE_DEBUG_EVALS:
            logging.warning("[pose_utils] eigh failed; falling back to identity axes")
        return centroid, axes, False, -1
    order = torch.argsort(evals, descending=True)
    evals_sorted = evals[order]
    axes = evecs[:, order].to(points.dtype)
    if evals_sorted.numel() < 3:
        axes = torch.eye(3, device=points.device, dtype=points.dtype)
        return centroid, axes, False, -1

    max_eval = float(evals_sorted[0].item())
    second_eval = float(evals_sorted[1].item())
    third_eval = float(evals_sorted[2].item())
    scale = max(max_eval, 1.0e-8)
    rel_tol = 5.0e-3
    linear_case = second_eval <= rel_tol * scale
    planar_case = (not linear_case) and (third_eval <= rel_tol * scale) and (abs(max_eval - second_eval) <= rel_tol * scale)

    degeneracy = 0

    if linear_case:
        principal = axes[:, 0]
        axes = _frame_from_axis(principal)
        valid = True
        degeneracy = 1
        if _POSE_DEBUG_EVALS:
            logging.warning(
                "[pose_utils] linear degeneracy evals=%s (λ2≈0, λ3≈0); using principal axis only",
                [float(v) for v in evals_sorted.tolist()],
            )
    elif planar_case:
        normal = axes[:, -1]
        axes = _frame_from_axis(normal)
        valid = True
        degeneracy = 2
        if _POSE_DEBUG_EVALS:
            logging.warning(
                "[pose_utils] planar degeneracy evals=%s (λ0≈λ1≫λ2); locking normal axis",
                [float(v) for v in evals_sorted.tolist()],
            )
    else:
        tol = 1.0e-5
        valid = bool(second_eval > tol and third_eval > tol)
        degeneracy = 0 if valid else -1
        if _POSE_DEBUG_EVALS and not valid:
            logging.warning(
                "[pose_utils] unstable PCA evals=%s (fallback to default frame)",
                [float(v) for v in evals_sorted.tolist()],
            )
    if torch.det(axes) < 0:
        axes = axes.clone()
        axes[:, -1] *= -1.0
    return centroid.to(points.dtype), axes, bool(valid), degeneracy


def _align_axes_to_world(axes: torch.Tensor) -> torch.Tensor:
    best_axes = axes
    best_score = -float("inf")
    for perm in itertools.permutations(range(3)):
        candidate = axes[:, perm]
        score = torch.abs(torch.diagonal(candidate)).sum().item()
        if score > best_score:
            best_score = score
            best_axes = candidate
    if torch.det(best_axes) < 0:
        best_axes = best_axes.clone()
        best_axes[:, -1] *= -1.0
    return best_axes


def _align_axes_to_reference(axes: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    aligned = axes.clone()
    flipped = []
    # Flip axes whose direction disagrees with reference to maximize alignment.
    for idx in range(3):
        if torch.dot(aligned[:, idx], reference[:, idx]) < 0:
            aligned[:, idx] *= -1.0
            flipped.append(idx)
    # Ensure we keep a right-handed frame even if an odd number of flips occurred.
    if torch.det(aligned) < 0:
        dots = torch.stack([torch.dot(aligned[:, i], reference[:, i]) for i in range(3)])
        # Prefer to flip the axis least aligned with the reference to minimize disruption.
        flip_idx = int(torch.argmin(torch.abs(dots)))
        aligned[:, flip_idx] *= -1.0
    return aligned


def compute_adsorbate_frames(
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    ads_mask: torch.Tensor,
    reference_axes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if positions.ndim != 2 or positions.size(-1) != 3:
        raise ValueError("positions must have shape (total_atoms, 3)")
    B = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    device = positions.device
    dtype = positions.dtype
    centroids = torch.zeros(B, 3, device=device, dtype=dtype)
    axes_wc = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
    degeneracy = torch.full((B,), -1, dtype=torch.int64, device=device)
    for b in range(B):
        atom_mask = (batch_idx == b) & ads_mask
        if not atom_mask.any():
            continue
        pts = positions[atom_mask]
        centroid, axes, valid, deg = _compute_pca_axes(pts)
        axes = _align_axes_to_world(axes)
        if reference_axes is not None:
            axes = _align_axes_to_reference(axes, reference_axes[b])
        centroids[b] = centroid
        axes_wc[b] = axes
        valid_mask[b] = valid
        degeneracy[b] = deg
    axes_cw = axes_wc.transpose(-1, -2)
    return centroids, axes_wc, axes_cw, valid_mask, degeneracy
