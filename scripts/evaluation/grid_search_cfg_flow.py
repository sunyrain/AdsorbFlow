#!/usr/bin/env python3
"""Grid search over cfg_scale and num_steps for Flow Matching relaxations.

This script runs the sampling + relaxation pipeline for each parameter pair,
then evaluates success rate with the logic provided in scripts.eval.
"""

# Example (single checkpoint):
# python scripts/grid_search_cfg_flow.py --cfg-scales 1.0 1.5 --num-steps 40 60 \
#     --flow-checkpoint checkpoints/flow/model.ckpt --relax-checkpoint checkpoints/relax/model.ckpt
# Example (directory of checkpoints):
# python scripts/grid_search_cfg_flow.py --cfg-scales 1.0 --num-steps 40 \
#     --flow-checkpoint checkpoints/flow_dir --relax-checkpoint checkpoints/relax/model.ckpt

import argparse
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import eval as eval_module  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Flow Matching CFG parameters")
    parser.add_argument("--cfg-scales", type=float, nargs="+", required=True, help="List of cfg_scale values")
    parser.add_argument("--num-steps", type=int, nargs="+", required=True, help="List of Flow Matching num_steps values")
    parser.add_argument("--output-root", type=str, default="grid_search_runs", help="Root directory for grid search outputs")
    parser.add_argument("--model-type", type=str, default="painn", choices=["painn", "eqv2"], help="Model architecture type (painn or eqv2)")
    parser.add_argument("--flow-config", type=str, default=None, help="Config file for flow sampling stage (auto-detected from --model-type if not specified)")
    parser.add_argument(
        "--flow-checkpoint",
        type=str,
        required=True,
        help="Checkpoint for the flow model or directory containing checkpoints",
    )
    parser.add_argument("--relax-config", type=str, default="configs/relaxation/gemnet_oc/gemnet_relax.yml", help="Config file for relaxation stage")
    parser.add_argument("--relax-checkpoint", type=str, required=True, help="Checkpoint for the relaxation model")
    parser.add_argument("--relax-dataset", type=str, default="val_nonrelaxed_update", help="LMDB dataset used for flow sampling stage")
    parser.add_argument("--nsites", type=int, default=1, help="Number of site seeds to process (mirrors run.py behaviour)")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs per command")
    parser.add_argument("--master-port", type=int, default=1235, help="Master port for torch.distributed.launch")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker count for LMDB conversion stage")
    parser.add_argument("--dft-targets", type=str, default="oc20_dense_mappings/oc20dense_targets.pkl", help="Pickle file with DFT target energies")
    parser.add_argument("--results-file", type=str, default=None, help="Optional path for the aggregated results file")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs already present in the results file")
    parser.add_argument("--cuda-devices", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES mask")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def _safe_model_folder(flow_checkpoint: Path) -> str:
    """Derive a filesystem-friendly folder name from the flow checkpoint path."""
    parent_name = flow_checkpoint.parent.name or "model"
    identifier = f"{parent_name}_{flow_checkpoint.stem}" if parent_name else flow_checkpoint.stem
    sanitized = identifier.replace(os.sep, "_").replace(":", "_").replace(" ", "_")
    return sanitized or "model"


def _collect_flow_checkpoints(path: Path) -> List[Path]:
    resolved = path.resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Flow checkpoint path not found: {resolved}")
    patterns = ("*.ckpt", "*.pt", "*.pth")
    candidates = []
    seen = set()
    for pattern in patterns:
        for candidate in resolved.rglob(pattern):
            if not candidate.is_file():
                continue
            candidate_resolved = candidate.resolve()
            if candidate_resolved in seen:
                continue
            seen.add(candidate_resolved)
            candidates.append(candidate_resolved)
    if not candidates:
        raise ValueError(f"No checkpoints with extensions {patterns} found in {resolved}")
    candidates.sort()
    return candidates


def load_dft_targets(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"DFT target file not found: {path}")
    with path.open("rb") as handle:
        targets = pickle.load(handle)
    sample_value = next(iter(targets.values()))
    if isinstance(sample_value, dict):  # convert to sid -> energy map
        converted = {}
        for sid, candidates in targets.items():
            energies = [val for val in candidates.values() if isinstance(val, (int, float))]
            if energies:
                converted[sid] = min(energies)
        targets = converted
    elif isinstance(sample_value, list):  # oc20_dense format: list of (config, energy) tuples
        converted = {}
        for sid, candidates in targets.items():
            energies = []
            for item in candidates:
                if isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[1], (int, float)):
                    energies.append(item[1])
                elif isinstance(item, (int, float)):
                    energies.append(item)
            if energies:
                converted[sid] = min(energies)
        targets = converted
    return targets


def load_existing_results(path: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = f"{data['cfg_scale']}:{data['num_steps']}"
            records[key] = data
    return records


def save_results(records: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record) + "\n")


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_subprocess(cmd: List[str], log_path: Path, env: Dict[str, str], dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable_cmd = " ".join(cmd)
    print(f"[grid-search] Running: {printable_cmd}")
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed (exit {completed.returncode}): {printable_cmd}. See {log_path} for details.")


def evaluate_success(traj_dir: Path, dft_targets: Dict[str, float]) -> Tuple[float, int, int, Optional[float], Optional[float], set, int]:
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    success_percent, valid_count, success_count, diff_mean, diff_var, successful_sids, total_targets = eval_module.get_success_from_trajs_rewrite(
        str(traj_dir), dft_targets
    )
    return success_percent, valid_count, success_count, diff_mean, diff_var, successful_sids, total_targets


def build_flow_command(
    args: argparse.Namespace,
    step_dir: Path,
    cfg_scale: float,
    num_steps: int,
    seed: int,
    flow_checkpoint: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        "main.py",
        "--mode",
        "run-relaxations",
        "--config-yml",
        str((REPO_ROOT / args.flow_config).resolve()),
        f"--task.relax_dataset.src={(REPO_ROOT / args.relax_dataset).resolve()}",
        f"--task.relax_opt.traj_dir={step_dir}",
        f"--task.relax_opt.cfg_scale={cfg_scale}",
        f"--task.relax_opt.num_steps={num_steps}",
        "--checkpoint",
    str(flow_checkpoint.resolve()),
        "--distributed",
        # "--amp",
        "--model.sampling=True",
        f"--seed={seed}",
        "--debug",
    ]


def build_lmdb_command(step_dir: Path, num_workers: int) -> List[str]:
    return [
        sys.executable,
        "scripts/create_lmdbs/pred_traj_to_lmdb.py",
        "--data-path",
        str(step_dir),
        "--out-path",
        str(step_dir / "final_struct_lmdb"),
        "--num-workers",
        str(num_workers),
    ]


def build_relax_command(args: argparse.Namespace, step_dir: Path) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.launch",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        "main.py",
        "--mode",
        "run-relaxations",
        "--config-yml",
        str((REPO_ROOT / args.relax_config).resolve()),
        "--checkpoint",
        str((REPO_ROOT / args.relax_checkpoint).resolve()),
        f"--task.relax_dataset.src={step_dir / 'final_struct_lmdb'}",
        f"--task.relax_opt.traj_dir={step_dir / 'relaxations'}",
        "--distributed",
        # "--amp",
        "--debug",
    ]


def main() -> None:
    args = parse_args()

    # Auto-detect model type from checkpoint path if using default
    if args.model_type == "painn":
        # Check if checkpoint path contains eqv2/equiformer keywords
        ckpt_path_lower = args.flow_checkpoint.lower()
        if "eqv2" in ckpt_path_lower or "equiformer" in ckpt_path_lower:
            args.model_type = "eqv2"
            print(f"[grid-search] Auto-detected model type from checkpoint path: eqv2")

    # Auto-detect flow config based on model type if not explicitly specified
    if args.flow_config is None:
        model_type_to_config = {
            "painn": "configs/flow/painn_conditional_flow.yml",
            "eqv2": "configs/flow/eqv2_conditional_flow.yml",
        }
        args.flow_config = model_type_to_config[args.model_type]
        print(f"[grid-search] Using flow config for {args.model_type}: {args.flow_config}")

    flow_checkpoint_path = Path(args.flow_checkpoint)
    if not flow_checkpoint_path.is_absolute():
        flow_checkpoint_path = (REPO_ROOT / flow_checkpoint_path).resolve()
    else:
        flow_checkpoint_path = flow_checkpoint_path.resolve()
    checkpoint_paths = _collect_flow_checkpoints(flow_checkpoint_path)
    if len(checkpoint_paths) > 1:
        print(f"[grid-search] Found {len(checkpoint_paths)} checkpoints under {flow_checkpoint_path}")

    base_output_root = (REPO_ROOT / args.output_root).resolve()
    dft_targets = load_dft_targets((REPO_ROOT / args.dft_targets).resolve())
    base_env = os.environ.copy()
    if args.cuda_devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.results_file:
        results_base_path = Path(args.results_file)
        if not results_base_path.is_absolute():
            results_base_path = (REPO_ROOT / results_base_path).resolve()
        else:
            results_base_path = results_base_path.resolve()
    else:
        results_base_path = None

    multiple_checkpoints = len(checkpoint_paths) > 1

    for checkpoint in checkpoint_paths:
        print(f"[grid-search] Processing flow checkpoint: {checkpoint}")
        model_folder = _safe_model_folder(checkpoint)

        dataset_path = Path(args.relax_dataset)
        dataset_name = dataset_path.name if dataset_path.name else dataset_path.parent.name

        output_root = base_output_root / model_folder / dataset_name / f"nsites_{args.nsites}"
        output_root.mkdir(parents=True, exist_ok=True)

        if results_base_path is not None:
            suffix = results_base_path.suffix
            stem = results_base_path.stem
            if multiple_checkpoints:
                named = f"{stem}_{model_folder}_{dataset_name}_nsites{args.nsites}{suffix}"
            else:
                named = f"{stem}_{dataset_name}_nsites{args.nsites}{suffix}"
            results_path = results_base_path.with_name(named)
        else:
            results_path = output_root / f"grid_search_results_nsites{args.nsites}.jsonl"

        results = load_existing_results(results_path)

        for cfg_scale, num_steps in itertools.product(args.cfg_scales, args.num_steps):
            key = f"{cfg_scale}:{num_steps}"
            if args.skip_existing and key in results:
                print(
                    f"[grid-search] Skipping cfg_scale={cfg_scale}, num_steps={num_steps} (already recorded)"
                )
                continue

            combo_root = output_root / f"cfg{cfg_scale:g}_steps{num_steps}"
            print(f"[grid-search] === cfg_scale={cfg_scale}, num_steps={num_steps} ===")
            site_success_rates: List[float] = []
            site_valid_counts: List[int] = []
            site_success_counts: List[int] = []
            site_diff_means: List[Optional[float]] = []
            site_diff_variances: List[Optional[float]] = []
            site_successful_sids: List[set] = []
            last_total_targets = 0

            for site_idx in range(args.nsites):
                step_dir = combo_root / str(site_idx)
                if not args.dry_run:
                    clear_directory(step_dir)
                    (step_dir / "relaxations").mkdir(parents=True, exist_ok=True)
                flow_cmd = build_flow_command(
                    args,
                    step_dir,
                    cfg_scale,
                    num_steps,
                    seed=site_idx,
                    flow_checkpoint=checkpoint,
                )
                lmdb_cmd = build_lmdb_command(step_dir, args.num_workers)
                relax_cmd = build_relax_command(args, step_dir)

                run_subprocess(
                    flow_cmd,
                    combo_root / "logs" / f"site{site_idx}_flow.log",
                    base_env,
                    dry_run=args.dry_run,
                )
                run_subprocess(
                    lmdb_cmd,
                    combo_root / "logs" / f"site{site_idx}_lmdb.log",
                    base_env,
                    dry_run=args.dry_run,
                )
                run_subprocess(
                    relax_cmd,
                    combo_root / "logs" / f"site{site_idx}_relax.log",
                    base_env,
                    dry_run=args.dry_run,
                )

                if args.dry_run:
                    continue
                relax_dir = step_dir / "relaxations"
                (
                    success_percent,
                    valid_count,
                    success_count,
                    diff_mean,
                    diff_var,
                    successful_sids,
                    total_targets,
                ) = evaluate_success(relax_dir, dft_targets)
                site_success_rates.append(success_percent)
                site_valid_counts.append(valid_count)
                site_success_counts.append(success_count)
                site_diff_means.append(diff_mean)
                site_diff_variances.append(diff_var)
                site_successful_sids.append(successful_sids)
                last_total_targets = total_targets

            if args.dry_run:
                continue
            aggregated = mean(site_success_rates) if site_success_rates else float("nan")

            # Calculate cumulative union success rates (Success Rate @ k)
            cumulative_union_success_rates = []
            current_union_sids = set()
            for sids in site_successful_sids:
                current_union_sids.update(sids)
                rate = (len(current_union_sids) / last_total_targets * 100.0) if last_total_targets > 0 else 0.0
                cumulative_union_success_rates.append(rate)

            union_success_rate = cumulative_union_success_rates[-1] if cumulative_union_success_rates else 0.0
            max_single_success_rate = max(site_success_rates) if site_success_rates else 0.0

            total_valid_count = sum(site_valid_counts)
            total_success_count = sum(site_success_counts)
            weighted_count = sum(
                count
                for count, mean_val in zip(site_valid_counts, site_diff_means)
                if count > 0 and mean_val is not None
            )
            if weighted_count > 0:
                combined_mean = sum(
                    count * mean_val
                    for count, mean_val in zip(site_valid_counts, site_diff_means)
                    if count > 0 and mean_val is not None
                ) / weighted_count
                combined_variance = 0.0
                for count, mean_val, var_val in zip(
                    site_valid_counts, site_diff_means, site_diff_variances
                ):
                    if count <= 0 or mean_val is None:
                        continue
                    var_component = var_val if var_val is not None else 0.0
                    combined_variance += count * (var_component + (mean_val - combined_mean) ** 2)
                combined_variance /= weighted_count
            else:
                combined_mean = None
                combined_variance = None
            record = {
                "cfg_scale": cfg_scale,
                "num_steps": num_steps,
                "mean_success_percent": aggregated,
                "union_success_percent": union_success_rate,
                "max_single_success_percent": max_single_success_rate,
                "success_rate_at_k": cumulative_union_success_rates,
                "site_success_percent": site_success_rates,
                "site_valid_counts": site_valid_counts,
                "site_success_counts": site_success_counts,
                "site_energy_diff_mean": site_diff_means,
                "site_energy_diff_variance": site_diff_variances,
                "energy_diff_mean": combined_mean,
                "energy_diff_variance": combined_variance,
                "total_valid_count": total_valid_count,
                "total_success_count": total_success_count,
                "output_dir": str(combo_root),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "flow_checkpoint": str(checkpoint),
            }
            results[key] = record
            save_results(results, results_path)
            log_msg = f"[grid-search] Recorded result: mean_success={aggregated:.2f}%, union_success={union_success_rate:.2f}%, max_single={max_single_success_rate:.2f}%"
            if combined_mean is not None and combined_variance is not None:
                log_msg += f", diff_mean={combined_mean:.4f}, diff_var={combined_variance:.4f}"
            print(log_msg)


if __name__ == "__main__":
    main()
