#!/usr/bin/env python3
"""Verify that sampled omegas match the rotation implied by pos_t geometries."""
from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from adsorbdiff.utils import rot_utils
from adsorbdiff.utils.utils import build_config, new_trainer_context, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare geometric rotation with omega targets.")
    parser.add_argument("--config-yml", required=True, type=str, help="Training config to reuse.")
    parser.add_argument("--identifier", default="omega_debug", type=str, help="Identifier used for trainer instantiation.")
    parser.add_argument("--override", nargs="*", default=[], help="Optional config overrides (key=value) applied after YAML.")
    parser.add_argument("--batches", type=int, default=3, help="Number of train batches to inspect (default: 3).")
    parser.add_argument("--resamples", type=int, default=5, help="Number of resamples per batch (default: 5).")
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Deprecated; retained for CLI compatibility.",
    )
    parser.add_argument("--omega-tol", type=float, default=1e-3, help="Tolerance on |omega_geo - omega_target| (default: 1e-3).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for Trainer instantiation (default: 0).")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory used for logging (default: logs).")
    parser.add_argument("--max-report", type=int, default=10, help="Maximum number of mismatched samples to print (default: 10).")
    return parser.parse_args()


def make_runner_args(cli: argparse.Namespace) -> Namespace:
    return Namespace(
        mode="train",
        config_yml=Path(cli.config_yml),
        identifier=cli.identifier,
        debug=True,
        run_dir="./",
        print_every=10,
        seed=cli.seed,
        amp=False,
        checkpoint=None,
        timestamp_id=None,
        submit=False,
        summit=False,
        logdir=Path(cli.logdir),
        slurm_partition="debug",
        slurm_mem=8,
        slurm_timeout=1,
        num_gpus=1,
        distributed=False,
        cpu=True,
        num_nodes=1,
        distributed_port=13356,
        distributed_backend="nccl",
        local_rank=0,
        no_ddp=True,
        gp_gpus=None,
    )


def iter_sample_slices(natoms: Iterable[int]) -> Iterable[Tuple[int, slice]]:
    start = 0
    for idx, count in enumerate(natoms):
        end = start + int(count)
        yield idx, slice(start, end)
        start = end


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return [float(x) for x in tensor.view(-1).tolist()]


def compute_geometric_rotation(
    base_slice: torch.Tensor,
    pos_slice: torch.Tensor,
    tags_slice: torch.Tensor,
    min_atoms: int = 3,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    ads_mask = (tags_slice == 2)
    num_ads = int(ads_mask.sum().item())
    if num_ads < min_atoms:
        return None, f"insufficient_adsorbate_atoms({num_ads})"
    base_ads = base_slice[ads_mask].clone()
    pos_ads = pos_slice[ads_mask].clone()
    if base_ads.numel() == 0 or pos_ads.numel() == 0:
        return None, "empty_adsorbate_slice"
    # base_ads[:, 2] += lift_height # Removed
    base_center = base_ads.mean(dim=0, keepdim=True)
    pos_center = pos_ads.mean(dim=0, keepdim=True)
    base_rel = (base_ads - base_center).t()
    pos_rel = (pos_ads - pos_center).t()
    if not torch.isfinite(base_rel).all() or not torch.isfinite(pos_rel).all():
        return None, "non_finite_coords"
    if torch.linalg.norm(base_rel) < 1.0e-8 or torch.linalg.norm(pos_rel) < 1.0e-8:
        return None, "degenerate_geometry"
    try:
        R, _ = rot_utils.rigid_transform_Kabsch_3D_torch(base_rel, pos_rel)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"kabsch_failure:{exc}"
    rotvec = rot_utils.matrix_to_axis_angle(R)
    return rotvec, None


def collect(
    records: List[Dict[str, torch.Tensor]],
    failures: List[Dict[str, object]],
    batch,
    base_pos: torch.Tensor,
    batch_idx: int,
    resample_idx: int,
) -> None:
    flow_debug = getattr(batch, "_flow_debug", None)
    if not isinstance(flow_debug, dict):
        return
    omega_tensor = flow_debug.get("omega_used")
    if omega_tensor is None:
        return
    omega = omega_tensor.detach().cpu()
    t_values = batch.t.detach().cpu().view(-1)
    pos = batch.pos.detach().cpu()
    tags = batch.tags.detach().cpu()
    natoms = batch.natoms.detach().cpu().tolist()
    for sample_idx, slc in iter_sample_slices(natoms):
        base_slice = base_pos[slc]
        pos_slice = pos[slc]
        tags_slice = tags[slc]
        rot_geo, err = compute_geometric_rotation(base_slice, pos_slice, tags_slice)
        if err is not None:
            failures.append(
                {
                    "batch_idx": batch_idx,
                    "resample_idx": resample_idx,
                    "sample_idx": sample_idx,
                    "reason": err,
                }
            )
            continue
        records.append(
            {
                "batch_idx": batch_idx,
                "resample_idx": resample_idx,
                "sample_idx": sample_idx,
                "t": float(t_values[sample_idx].item()),
                "rot_geo": rot_geo.detach().cpu(),
                "omega_target": omega[sample_idx].detach().cpu(),
            }
        )


def summarize(
    records: List[Dict[str, torch.Tensor]],
    failures: List[Dict[str, object]],
    omega_tol: float,
    max_report: int,
) -> None:
    if failures:
        counts = Counter(entry["reason"] for entry in failures)
        print(f"Skipped {len(failures)} samples due to geometric issues.")
        for reason, count in counts.most_common():
            print(f"  {reason}: {count}")
    if not records:
        print("No valid samples were analyzed; nothing to report.")
        return

    results = []
    for entry in records:
        rot_geo = entry["rot_geo"].clone()
        omega_target = entry["omega_target"].clone()
        t_val = entry["t"]
        safe_t = max(abs(t_val), 1.0e-6)
        rot_target = omega_target * t_val
        omega_geo = rot_geo / safe_t
        rot_error = torch.linalg.norm(rot_geo - rot_target).item()
        omega_error = torch.linalg.norm(omega_geo - omega_target).item()
        angle_geo = torch.linalg.norm(rot_geo).item()
        angle_target = torch.linalg.norm(rot_target).item()
        results.append(
            {
                **entry,
                "rot_target": rot_target,
                "omega_geo": omega_geo,
                "rot_error": rot_error,
                "omega_error": omega_error,
                "angle_geo": angle_geo,
                "angle_target": angle_target,
            }
        )

    omega_errors = torch.tensor([item["omega_error"] for item in results])
    rot_errors = torch.tensor([item["rot_error"] for item in results])
    print(f"Processed {len(results)} samples with geometric fits.")
    print(
        f"|omega_geo - omega_target| mean={omega_errors.mean().item():.4e} "
        f"std={omega_errors.std(unbiased=False).item():.4e} max={omega_errors.max().item():.4e}"
    )
    print(
        f"|rot_geo - rot_target| mean={rot_errors.mean().item():.4e} "
        f"std={rot_errors.std(unbiased=False).item():.4e} max={rot_errors.max().item():.4e}"
    )

    mismatches = [item for item in results if item["omega_error"] > omega_tol]
    print(f"Samples exceeding tolerance {omega_tol:.2e}: {len(mismatches)}")
    if not mismatches:
        return
    mismatches.sort(key=lambda item: item["omega_error"], reverse=True)
    for idx, item in enumerate(mismatches[: max_report], start=1):
        angle_geo_deg = math.degrees(item["angle_geo"])
        angle_target_deg = math.degrees(item["angle_target"])
        print(
            f"[Mismatch {idx}] batch={item['batch_idx']} sample={item['sample_idx']} "
            f"resample={item['resample_idx']} t={item['t']:.4f}"
        )
        print(f"  |omega_geo - omega_target| = {item['omega_error']:.4e}")
        print(f"  |rot_geo - rot_target|   = {item['rot_error']:.4e}")
        print(f"  angle_geo={angle_geo_deg:.2f} deg, angle_target={angle_target_deg:.2f} deg")
        print(f"  omega_geo   = {tensor_to_list(item['omega_geo'])}")
        print(f"  omega_tgt   = {tensor_to_list(item['omega_target'])}")
        print(f"  rot_geo     = {tensor_to_list(item['rot_geo'])}")
        print(f"  rot_target  = {tensor_to_list(item['rot_target'])}")


def main() -> None:
    cli = parse_args()
    runner_args = make_runner_args(cli)
    setup_logging()
    config = build_config(runner_args, cli.override)

    with new_trainer_context(config=config, args=runner_args) as ctx:
        task = ctx.task
        trainer = ctx.trainer
        task.setup(trainer)
        loader = trainer.train_loader
        iterator = iter(loader)
        records: List[Dict[str, torch.Tensor]] = []
        failures: List[Dict[str, object]] = []
        for batch_idx in range(min(cli.batches, len(loader))):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            base_pos = trainer._clone_base_positions(batch)
            batch = trainer._build_interpolant(batch)
            collect(records, failures, batch, base_pos, batch_idx, resample_idx=0)
            total_draws = max(cli.resamples, 1)
            for resample_idx in range(1, total_draws):
                batch = trainer._resample_interpolant(batch, base_pos)
                collect(
                    records,
                    failures,
                    batch,
                    base_pos,
                    batch_idx,
                    resample_idx=resample_idx,
                )

    summarize(records, failures, cli.omega_tol, cli.max_report)


if __name__ == "__main__":
    main()
