#!/usr/bin/env python3
"""Parallel grid search over (checkpoint, cfg_scale, num_steps, seed).

Each job = one (ckpt, cfg, steps, seed) tuple, uses a single GPU, runs:
  flow sampling  ->  pred_traj_to_lmdb  ->  gemnet relaxation  ->  eval
Across N GPUs, up to N jobs run in parallel. After all seeds for a combo
finish, aggregates into the same jsonl schema as grid_search_cfg_flow.py.

Example:
  python scripts/evaluation/parallel_grid_search.py \
    --flow-checkpoints checkpoints/eval_candidates/ep183.pt \
    --relax-checkpoint checkpoints/gemnet_oc_base_s2ef_2M.pt \
    --model-type eqv2 \
    --relax-dataset val_nonrelaxed_update \
    --cfg-scales 3 5 7 10 \
    --num-steps 5 10 \
    --seeds 0 1 2 \
    --output-root grid_search_runs/coarse_ep183 \
    --gpus 0 1 2 3
"""

import argparse
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation import eval as eval_module  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel grid search for Flow Matching")
    p.add_argument("--flow-checkpoints", type=str, nargs="+", required=True,
                   help="List of flow checkpoint files (or directories to expand)")
    p.add_argument("--relax-checkpoint", type=str, required=True,
                   help="GemNet MLFF checkpoint (e.g. gemnet_oc_base_s2ef_2M.pt)")
    p.add_argument("--relax-config", type=str,
                   default="configs/relaxation/gemnet_oc/gemnet_relax.yml")
    p.add_argument("--flow-config", type=str, default=None,
                   help="Flow config (auto-detected from --model-type if omitted)")
    p.add_argument("--model-type", type=str, default="eqv2", choices=["painn", "eqv2"])
    p.add_argument("--relax-dataset", type=str, default="val_nonrelaxed_update")
    p.add_argument("--cfg-scales", type=float, nargs="+", required=True)
    p.add_argument("--num-steps", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="List of seed indices to evaluate (e.g. 0 1 2 ... 9)")
    p.add_argument("--gpus", type=int, nargs="+", required=True,
                   help="List of GPU device indices to use as workers (e.g. 0 1 2 3)")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--dft-targets", type=str,
                   default="oc20_dense_mappings/oc20dense_targets.pkl")
    p.add_argument("--num-workers", type=int, default=2,
                   help="LMDB conversion workers")
    p.add_argument("--master-port-base", type=int, default=12340,
                   help="Each worker uses master_port_base + gpu_idx")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip jobs whose eval output already exists")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ----------------------------- utilities ------------------------------------

def _collect_flow_checkpoints(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    seen: Set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p.is_file():
            if p not in seen:
                seen.add(p); out.append(p)
        elif p.is_dir():
            for pat in ("*.ckpt", "*.pt", "*.pth"):
                for c in sorted(p.rglob(pat)):
                    c = c.resolve()
                    if c.is_file() and c not in seen:
                        seen.add(c); out.append(c)
        else:
            raise FileNotFoundError(f"Flow checkpoint not found: {p}")
    if not out:
        raise ValueError("No flow checkpoints found")
    return out


def _safe_model_folder(ckpt: Path) -> str:
    parent = ckpt.parent.name or "model"
    ident = f"{parent}_{ckpt.stem}"
    return ident.replace(os.sep, "_").replace(":", "_").replace(" ", "_")


def _load_dft_targets(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"DFT target file not found: {path}")
    with path.open("rb") as h:
        t = pickle.load(h)
    sample = next(iter(t.values()))
    if isinstance(sample, dict):
        out = {}
        for sid, cands in t.items():
            es = [v for v in cands.values() if isinstance(v, (int, float))]
            if es: out[sid] = min(es)
        return out
    if isinstance(sample, list):
        out = {}
        for sid, cands in t.items():
            es = []
            for item in cands:
                if isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[1], (int, float)):
                    es.append(item[1])
                elif isinstance(item, (int, float)):
                    es.append(item)
            if es: out[sid] = min(es)
        return out
    return t


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], log_path: Path, env: Dict[str, str], dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry] {' '.join(cmd)}")
        return
    with log_path.open("w", encoding="utf-8") as lf:
        completed = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cmd failed (exit {completed.returncode}): {' '.join(cmd)}\nSee {log_path}")


def _flow_cmd(flow_config: Path, flow_ckpt: Path, relax_dataset: Path,
              step_dir: Path, cfg_scale: float, num_steps: int, seed: int,
              master_port: int) -> List[str]:
    return [
        sys.executable, "-u", "-m", "torch.distributed.launch",
        "--nproc_per_node=1", f"--master_port={master_port}",
        "main.py", "--mode", "run-relaxations",
        "--config-yml", str(flow_config),
        f"--task.relax_dataset.src={relax_dataset}",
        f"--task.relax_opt.traj_dir={step_dir}",
        f"--task.relax_opt.cfg_scale={cfg_scale}",
        f"--task.relax_opt.num_steps={num_steps}",
        "--checkpoint", str(flow_ckpt),
        "--distributed", "--model.sampling=True",
        f"--seed={seed}", "--debug",
    ]


def _lmdb_cmd(step_dir: Path, num_workers: int) -> List[str]:
    return [
        sys.executable, "scripts/create_lmdbs/pred_traj_to_lmdb.py",
        "--data-path", str(step_dir),
        "--out-path", str(step_dir / "final_struct_lmdb"),
        "--num-workers", str(num_workers),
    ]


def _relax_cmd(relax_config: Path, relax_ckpt: Path, step_dir: Path,
               master_port: int) -> List[str]:
    return [
        sys.executable, "-u", "-m", "torch.distributed.launch",
        "--nproc_per_node=1", f"--master_port={master_port}",
        "main.py", "--mode", "run-relaxations",
        "--config-yml", str(relax_config),
        "--checkpoint", str(relax_ckpt),
        f"--task.relax_dataset.src={step_dir / 'final_struct_lmdb'}",
        f"--task.relax_opt.traj_dir={step_dir / 'relaxations'}",
        "--distributed", "--debug",
    ]


# --------------------------- job / worker -----------------------------------

class Job:
    __slots__ = ("ckpt", "cfg", "steps", "seed", "step_dir", "eval_file")
    def __init__(self, ckpt: Path, cfg: float, steps: int, seed: int,
                 step_dir: Path, eval_file: Path):
        self.ckpt = ckpt; self.cfg = cfg; self.steps = steps; self.seed = seed
        self.step_dir = step_dir; self.eval_file = eval_file

    def key(self) -> str:
        return f"{self.ckpt.name}|cfg{self.cfg}|steps{self.steps}|seed{self.seed}"


def worker_loop(worker_id: int, gpu_idx: int, job_q: "queue.Queue[Job]",
                args: argparse.Namespace, flow_config: Path, relax_config: Path,
                relax_ckpt: Path, relax_dataset: Path, dft_targets: Dict[str, float],
                results_lock: threading.Lock, results_store: Dict[str, Dict],
                progress: Dict[str, int]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    master_port = args.master_port_base + worker_id

    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        try:
            step_dir = job.step_dir
            log_dir = step_dir.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_tag = f"seed{job.seed}"

            t_start = time.time()
            if not args.skip_existing or not job.eval_file.exists():
                _clear_directory(step_dir)
                (step_dir / "relaxations").mkdir(parents=True, exist_ok=True)

                # 1) flow sampling
                _run(_flow_cmd(flow_config, job.ckpt, relax_dataset, step_dir,
                               job.cfg, job.steps, job.seed, master_port),
                     log_dir / f"{log_tag}_flow.log", env, args.dry_run)

                # 2) LMDB conversion
                _run(_lmdb_cmd(step_dir, args.num_workers),
                     log_dir / f"{log_tag}_lmdb.log", env, args.dry_run)

                # 3) MLFF relaxation
                _run(_relax_cmd(relax_config, relax_ckpt, step_dir, master_port),
                     log_dir / f"{log_tag}_relax.log", env, args.dry_run)

                # 4) eval
                if args.dry_run:
                    job_q.task_done(); continue
                (succ_pct, valid, hits, diff_m, diff_v, succ_sids, total) = \
                    eval_module.get_success_from_trajs_rewrite(
                        str(step_dir / "relaxations"), dft_targets)
                eval_rec = {
                    "ckpt": str(job.ckpt), "cfg_scale": job.cfg,
                    "num_steps": job.steps, "seed": job.seed,
                    "success_percent": succ_pct, "valid_count": valid,
                    "success_count": hits, "energy_diff_mean": diff_m,
                    "energy_diff_variance": diff_v,
                    "successful_sids": sorted(list(succ_sids)),
                    "total_targets": total, "gpu": gpu_idx,
                    "elapsed_sec": time.time() - t_start,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                job.eval_file.parent.mkdir(parents=True, exist_ok=True)
                with job.eval_file.open("w", encoding="utf-8") as f:
                    json.dump(eval_rec, f, indent=2)
            else:
                with job.eval_file.open("r", encoding="utf-8") as f:
                    eval_rec = json.load(f)

            with results_lock:
                results_store[job.key()] = eval_rec
                progress["done"] += 1
                done = progress["done"]; total_j = progress["total"]
            print(f"[worker {worker_id} / gpu{gpu_idx}] done {job.key()}  "
                  f"SR={eval_rec.get('success_percent', 'NA')}%  "
                  f"[{done}/{total_j}]")
        except Exception as e:
            print(f"[worker {worker_id} / gpu{gpu_idx}] FAILED {job.key()}: {e}",
                  file=sys.stderr)
        finally:
            job_q.task_done()


def aggregate_combos(results_store: Dict[str, Dict], ckpts: List[Path],
                     cfgs: List[float], steps_list: List[int], seeds: List[int],
                     out_root: Path, dataset_name: str, nsites: int) -> None:
    """Aggregate per-seed eval records into the same jsonl schema as
    grid_search_cfg_flow.py (one line per (cfg, steps), collecting all seeds)."""
    for ckpt in ckpts:
        model_folder = _safe_model_folder(ckpt)
        combo_out_dir = out_root / model_folder / dataset_name / f"nsites_{nsites}"
        combo_out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = combo_out_dir / f"grid_search_results_nsites{nsites}.jsonl"
        records: List[Dict] = []
        for cfg in cfgs:
            for steps in steps_list:
                site_rates: List[float] = []
                site_valid: List[int] = []
                site_hits: List[int] = []
                site_diff_means: List[Optional[float]] = []
                site_diff_vars: List[Optional[float]] = []
                site_sids: List[set] = []
                total_targets = 0
                for seed in seeds:
                    key = f"{ckpt.name}|cfg{cfg}|steps{steps}|seed{seed}"
                    r = results_store.get(key)
                    if r is None:
                        continue
                    site_rates.append(r["success_percent"])
                    site_valid.append(r["valid_count"])
                    site_hits.append(r["success_count"])
                    site_diff_means.append(r.get("energy_diff_mean"))
                    site_diff_vars.append(r.get("energy_diff_variance"))
                    site_sids.append(set(r.get("successful_sids", [])))
                    total_targets = max(total_targets, r.get("total_targets", 0))
                if not site_rates:
                    continue

                # cumulative union success_rate@k
                cum: List[float] = []
                union = set()
                for sids in site_sids:
                    union.update(sids)
                    cum.append(len(union) / total_targets * 100.0 if total_targets else 0.0)
                union_pct = cum[-1] if cum else 0.0
                mean_pct = mean(site_rates)
                max_pct = max(site_rates)

                # weighted combined energy_diff stats
                total_valid = sum(site_valid)
                total_hits = sum(site_hits)
                w_count = sum(c for c, m in zip(site_valid, site_diff_means)
                              if c > 0 and m is not None)
                if w_count > 0:
                    comb_mean = sum(c * m for c, m in zip(site_valid, site_diff_means)
                                    if c > 0 and m is not None) / w_count
                    comb_var = 0.0
                    for c, m, v in zip(site_valid, site_diff_means, site_diff_vars):
                        if c <= 0 or m is None: continue
                        vc = v if v is not None else 0.0
                        comb_var += c * (vc + (m - comb_mean) ** 2)
                    comb_var /= w_count
                else:
                    comb_mean = None; comb_var = None

                records.append({
                    "cfg_scale": cfg, "num_steps": steps,
                    "mean_success_percent": mean_pct,
                    "union_success_percent": union_pct,
                    "max_single_success_percent": max_pct,
                    "success_rate_at_k": cum,
                    "site_success_percent": site_rates,
                    "site_valid_counts": site_valid,
                    "site_success_counts": site_hits,
                    "site_energy_diff_mean": site_diff_means,
                    "site_energy_diff_variance": site_diff_vars,
                    "energy_diff_mean": comb_mean,
                    "energy_diff_variance": comb_var,
                    "total_valid_count": total_valid,
                    "total_success_count": total_hits,
                    "output_dir": str(combo_out_dir / f"cfg{cfg:g}_steps{steps}"),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "flow_checkpoint": str(ckpt),
                    "n_seeds": len(site_rates),
                })
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        # Summary table
        print(f"\n=== Summary: {model_folder} ===")
        print(f"{'cfg':>6} {'steps':>6} {'SR@1':>8} {'SR@k':>8} {'mean':>8} "
              f"{'E_diff':>8} {'k':>4}")
        for r in sorted(records, key=lambda x: -x["union_success_percent"]):
            sr1 = r["success_rate_at_k"][0] if r["success_rate_at_k"] else 0.0
            print(f"{r['cfg_scale']:>6.1f} {r['num_steps']:>6d} {sr1:>8.2f} "
                  f"{r['union_success_percent']:>8.2f} {r['mean_success_percent']:>8.2f} "
                  f"{(r['energy_diff_mean'] or 0):>8.3f} {r['n_seeds']:>4d}")


# --------------------------------- main -------------------------------------

def main() -> None:
    args = parse_args()

    # flow config auto-detect
    if args.flow_config is None:
        args.flow_config = ("configs/flow/eqv2_conditional_flow.yml"
                            if args.model_type == "eqv2"
                            else "configs/flow/painn_conditional_flow.yml")
    flow_config = (REPO_ROOT / args.flow_config).resolve()
    relax_config = (REPO_ROOT / args.relax_config).resolve()
    relax_ckpt = (REPO_ROOT / args.relax_checkpoint).resolve()
    relax_dataset = (REPO_ROOT / args.relax_dataset).resolve()
    out_root = (REPO_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpts = _collect_flow_checkpoints(args.flow_checkpoints)
    dft_targets = _load_dft_targets((REPO_ROOT / args.dft_targets).resolve())
    dataset_name = relax_dataset.name or relax_dataset.parent.name
    nsites = len(args.seeds)

    print(f"[grid] Flow config: {flow_config}")
    print(f"[grid] Relax ckpt : {relax_ckpt}")
    print(f"[grid] Checkpoints: {[c.name for c in ckpts]}")
    print(f"[grid] cfgs       : {args.cfg_scales}")
    print(f"[grid] steps      : {args.num_steps}")
    print(f"[grid] seeds      : {args.seeds}")
    print(f"[grid] GPUs       : {args.gpus}")

    # build job list
    jobs: List[Job] = []
    for ckpt in ckpts:
        model_folder = _safe_model_folder(ckpt)
        base = out_root / model_folder / dataset_name / f"nsites_{nsites}"
        for cfg in args.cfg_scales:
            for steps in args.num_steps:
                combo_root = base / f"cfg{cfg:g}_steps{steps}"
                for seed in args.seeds:
                    step_dir = combo_root / str(seed)
                    eval_file = combo_root / "eval_records" / f"seed{seed}.json"
                    jobs.append(Job(ckpt, cfg, steps, seed, step_dir, eval_file))

    print(f"[grid] Total jobs : {len(jobs)}")
    if args.dry_run:
        for j in jobs[:10]:
            print(f"  {j.key()} -> {j.step_dir}")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs)-10} more")
        return

    # --- worker pool ---
    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results_store: Dict[str, Dict] = {}
    results_lock = threading.Lock()
    progress = {"done": 0, "total": len(jobs)}

    # Preload existing records if skip-existing
    if args.skip_existing:
        for j in jobs:
            if j.eval_file.exists():
                try:
                    with j.eval_file.open("r") as f:
                        results_store[j.key()] = json.load(f)
                except Exception:
                    pass

    threads: List[threading.Thread] = []
    for wid, gpu in enumerate(args.gpus):
        t = threading.Thread(
            target=worker_loop,
            args=(wid, gpu, job_q, args, flow_config, relax_config,
                  relax_ckpt, relax_dataset, dft_targets,
                  results_lock, results_store, progress),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # aggregate
    aggregate_combos(results_store, ckpts, args.cfg_scales, args.num_steps,
                     args.seeds, out_root, dataset_name, nsites)
    print(f"\n[grid] Done. Results written under {out_root}")


if __name__ == "__main__":
    main()
