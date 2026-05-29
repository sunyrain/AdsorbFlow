#!/usr/bin/env python3
"""
Overnight Comprehensive Experiment Suite for AdsorbFlow
========================================================
Runs three phases automatically:

Phase A — Epoch Sweep (coarse):
  22 epochs × {cfg=5,7} × steps=5 × 3 seeds = 132 jobs
  Goal: Find the optimal epoch (training convergence vs generalization)

Phase B — Fine Grid on Top-3 Epochs:
  Top-3 epochs × {cfg=3,5,7,10,15} × steps=5 × 5 seeds = 75 jobs
  Goal: Find optimal cfg for each promising epoch

Phase C — Full 10-seed Evaluation:
  Top-3 (epoch, cfg) combos × 10 seeds = 30 jobs
  Goal: Final SR@10 numbers for paper

Total: ~237 jobs ÷ 4 GPU × ~8 min ≈ 8 hours

Usage:
  python scripts/evaluation/overnight_experiment.py [--phases A B C] [--gpus 0 1 2 3]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "checkpoints" / "2026-04-16-12-43-44-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2_fulldata_valsplit"
RELAX_CKPT = REPO_ROOT / "checkpoints" / "gemnet_oc_base_s2ef_2M.pt"
LOG_DIR = REPO_ROOT / "logs" / "eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Overnight experiment suite")
    p.add_argument("--phases", nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "C"], help="Which phases to run")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--phase-a-seeds", type=int, default=3,
                   help="Seeds per combo in Phase A")
    p.add_argument("--phase-b-seeds", type=int, default=5,
                   help="Seeds per combo in Phase B")
    p.add_argument("--phase-c-seeds", type=int, default=10,
                   help="Seeds per combo in Phase C")
    p.add_argument("--phase-b-top", type=int, default=3,
                   help="Top N epochs to expand in Phase B")
    p.add_argument("--phase-c-top", type=int, default=3,
                   help="Top N (epoch,cfg) combos in Phase C")
    p.add_argument("--output-root", type=str, default="grid_search_runs/overnight")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip jobs that already have eval records")
    p.add_argument("--cleanup-intermediates", action="store_true",
                   help="Remove traj/lmdb intermediates after eval (saves ~100MB/job)")
    return p.parse_args()


def discover_checkpoints() -> Dict[int, Tuple[str, float, float]]:
    """Return {epoch: (filename, val_loss, pos_mae)}"""
    entries = {}
    for f in os.listdir(CKPT_DIR):
        m = re.match(r'epoch(\d+)_unweightedvalloss([\d.]+)_posmae([\d.]+)\.pt', f)
        if m:
            ep = int(m.group(1))
            loss = float(m.group(2))
            mae = float(m.group(3))
            entries[ep] = (f, loss, mae)
    return entries


def select_phase_a_epochs(entries: Dict[int, Tuple]) -> List[int]:
    """Select epochs for Phase A sweep."""
    available = sorted(entries.keys())
    selected = set()

    # Every 10 from 85 to 192
    for ep in range(85, 193, 10):
        if ep in available:
            selected.add(ep)

    # Best by val_loss (top 3 among ep>=100)
    by_loss = sorted([(ep, v[1]) for ep, v in entries.items() if ep >= 100],
                     key=lambda x: x[1])
    for ep, _ in by_loss[:3]:
        selected.add(ep)

    # Best by pos_mae (top 3 among ep>=100)
    by_mae = sorted([(ep, v[2]) for ep, v in entries.items() if ep >= 100],
                    key=lambda x: x[1])
    for ep, _ in by_mae[:3]:
        selected.add(ep)

    # Always include key epochs
    for ep in [140, 169, 183, 192]:
        if ep in available:
            selected.add(ep)

    return sorted(selected & set(available))


def run_parallel_grid(flow_ckpts: List[str], cfg_scales: List[float],
                      num_steps: List[int], seeds: List[int],
                      gpus: List[int], output_root: str,
                      skip_existing: bool = False,
                      cleanup: bool = False) -> Path:
    """Run parallel_grid_search.py and return the output root."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"overnight_{ts}.log"

    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "evaluation" / "parallel_grid_search.py"),
        "--flow-checkpoints", *flow_ckpts,
        "--relax-checkpoint", str(RELAX_CKPT),
        "--model-type", "eqv2",
        "--cfg-scales", *[str(c) for c in cfg_scales],
        "--num-steps", *[str(s) for s in num_steps],
        "--seeds", *[str(s) for s in seeds],
        "--gpus", *[str(g) for g in gpus],
        "--output-root", output_root,
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"\n{'='*80}")
    print(f"[overnight] Launching: {len(flow_ckpts)} ckpts × {len(cfg_scales)} cfgs × "
          f"{len(num_steps)} steps × {len(seeds)} seeds = "
          f"{len(flow_ckpts)*len(cfg_scales)*len(num_steps)*len(seeds)} jobs")
    print(f"[overnight] GPUs: {gpus}")
    print(f"[overnight] Log: {log_path}")
    print(f"{'='*80}")

    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=REPO_ROOT, check=False)

    if proc.returncode != 0:
        print(f"[overnight] WARNING: exited with code {proc.returncode}")
        # Don't fail — partial results are still useful

    return Path(output_root)


def load_all_eval_records(output_root: Path) -> List[Dict]:
    """Load all seed-level eval records from an experiment output."""
    records = []
    for path in output_root.rglob("seed*.json"):
        try:
            with open(path) as f:
                rec = json.load(f)
                rec["_path"] = str(path)
                records.append(rec)
        except Exception:
            pass
    return records


def aggregate_by_combo(records: List[Dict]) -> Dict[str, Dict]:
    """Group records by (ckpt, cfg, steps) and compute aggregate stats."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        ckpt_name = Path(r.get("ckpt", "")).name
        key = f"{ckpt_name}|cfg{r['cfg_scale']}|steps{r['num_steps']}"
        groups[key].append(r)

    agg = {}
    for key, recs in groups.items():
        recs = sorted(recs, key=lambda x: x["seed"])
        srs = [r["success_percent"] for r in recs]

        # Union SR@k
        all_sids = []
        total_t = max((r.get("total_targets", 44) for r in recs), default=44)
        for r in recs:
            all_sids.append(set(r.get("successful_sids", [])))
        union = set()
        sr_at_k = []
        for sids in all_sids:
            union.update(sids)
            sr_at_k.append(len(union) / total_t * 100.0 if total_t else 0)

        from statistics import mean, stdev
        agg[key] = {
            "ckpt": recs[0].get("ckpt", ""),
            "cfg_scale": recs[0]["cfg_scale"],
            "num_steps": recs[0]["num_steps"],
            "n_seeds": len(recs),
            "mean_sr": mean(srs),
            "std_sr": stdev(srs) if len(srs) > 1 else 0,
            "max_sr": max(srs),
            "sr_at_k": sr_at_k,
            "union_sr": sr_at_k[-1] if sr_at_k else 0,
            "e_diff_mean": mean([r.get("energy_diff_mean", 0) or 0 for r in recs]),
        }
    return agg


def extract_epoch(ckpt_path: str) -> int:
    """Extract epoch number from checkpoint path."""
    m = re.search(r'epoch(\d+)', ckpt_path)
    return int(m.group(1)) if m else -1


def print_phase_summary(agg: Dict[str, Dict], title: str):
    """Print a summary table sorted by union_sr."""
    items = sorted(agg.values(), key=lambda x: -x["union_sr"])
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'epoch':>5} {'cfg':>5} {'steps':>5} | {'SR@1':>6} {'SR@3':>6} "
          f"{'SR@k':>6} | {'mean':>6}±{'std':>4} | {'E_diff':>6} | {'k':>3}")
    print(f"  {'-'*80}")
    for item in items:
        ep = extract_epoch(item["ckpt"])
        srak = item["sr_at_k"]
        sr1 = srak[0] if len(srak) >= 1 else 0
        sr3 = srak[2] if len(srak) >= 3 else srak[-1] if srak else 0
        print(f"  {ep:>5d} {item['cfg_scale']:>5.0f} {item['num_steps']:>5d} | "
              f"{sr1:>6.1f} {sr3:>6.1f} {item['union_sr']:>6.1f} | "
              f"{item['mean_sr']:>6.1f}±{item['std_sr']:>4.1f} | "
              f"{item['e_diff_mean']:>6.3f} | {item['n_seeds']:>3d}")


def cleanup_intermediates(output_root: Path):
    """Remove large intermediate files (trajs, lmdbs) to save disk."""
    import shutil
    count = 0
    for p in output_root.rglob("final_struct_lmdb"):
        if p.is_dir():
            shutil.rmtree(p); count += 1
    # Keep relaxations dir (needed for anomaly analysis) but remove traj files > 1MB
    for p in output_root.rglob("*.traj"):
        if p.stat().st_size > 1_000_000:
            p.unlink(); count += 1
    if count:
        print(f"[overnight] Cleaned {count} intermediate files/dirs")


def main():
    args = parse_args()
    entries = discover_checkpoints()
    print(f"[overnight] Found {len(entries)} epoch checkpoints")
    print(f"[overnight] Phases: {args.phases}")
    print(f"[overnight] GPUs: {args.gpus}")

    t_start = time.time()
    base_output = REPO_ROOT / args.output_root

    # ===================================================================
    # PHASE A: Epoch Sweep
    # ===================================================================
    phase_a_agg = {}
    if "A" in args.phases:
        print(f"\n{'#'*80}")
        print(f"#  PHASE A: Epoch Sweep — find optimal training epoch")
        print(f"{'#'*80}")

        epochs = select_phase_a_epochs(entries)
        print(f"[Phase A] Selected {len(epochs)} epochs: {epochs}")

        # Print training metrics for selected epochs
        print(f"\n  {'epoch':>5} {'val_loss':>9} {'pos_mae':>9}")
        for ep in epochs:
            _, loss, mae = entries[ep]
            print(f"  {ep:>5d} {loss:>9.4f} {mae:>9.4f}")

        ckpt_paths = [str(CKPT_DIR / entries[ep][0]) for ep in epochs]
        output_a = str(base_output / "phase_a_epoch_sweep")

        run_parallel_grid(
            flow_ckpts=ckpt_paths,
            cfg_scales=[5.0, 7.0],
            num_steps=[5],
            seeds=list(range(args.phase_a_seeds)),
            gpus=args.gpus,
            output_root=output_a,
            skip_existing=args.skip_existing,
        )

        records_a = load_all_eval_records(Path(output_a))
        phase_a_agg = aggregate_by_combo(records_a)
        print_phase_summary(phase_a_agg, "PHASE A RESULTS: Epoch Sweep (cfg={5,7}, steps=5)")

        if args.cleanup_intermediates:
            cleanup_intermediates(Path(output_a))

        # Save phase A summary
        summary_path = Path(output_a) / "phase_a_summary.json"
        with open(summary_path, "w") as f:
            # Make sr_at_k JSON serializable
            serializable = {}
            for k, v in phase_a_agg.items():
                serializable[k] = {kk: vv for kk, vv in v.items()}
            json.dump(serializable, f, indent=2, default=str)
        print(f"[Phase A] Summary saved to {summary_path}")

    # ===================================================================
    # PHASE B: Fine CFG Search on Top Epochs
    # ===================================================================
    phase_b_agg = {}
    if "B" in args.phases:
        print(f"\n{'#'*80}")
        print(f"#  PHASE B: Fine CFG Grid on Top-{args.phase_b_top} Epochs")
        print(f"{'#'*80}")

        # If Phase A wasn't run this session, try loading saved results
        if not phase_a_agg:
            saved = base_output / "phase_a_epoch_sweep" / "phase_a_summary.json"
            if saved.exists():
                with open(saved) as f:
                    phase_a_agg = json.load(f)
                print(f"[Phase B] Loaded Phase A results from {saved}")
            else:
                print("[Phase B] ERROR: No Phase A results found. Run Phase A first.")
                return

        # Find top epochs by union_sr (best cfg per epoch)
        epoch_best = {}
        for key, item in phase_a_agg.items():
            ep = extract_epoch(item["ckpt"] if isinstance(item["ckpt"], str) else str(item["ckpt"]))
            if ep not in epoch_best or item["union_sr"] > epoch_best[ep]["union_sr"]:
                epoch_best[ep] = item

        top_epochs = sorted(epoch_best.items(), key=lambda x: -x[1]["union_sr"])[:args.phase_b_top]

        print(f"[Phase B] Top-{args.phase_b_top} epochs from Phase A:")
        for ep, item in top_epochs:
            print(f"  ep{ep}: union_SR={item['union_sr']:.1f}% (cfg={item['cfg_scale']})")

        top_epoch_nums = [ep for ep, _ in top_epochs]
        ckpt_paths = [str(CKPT_DIR / entries[ep][0]) for ep in top_epoch_nums if ep in entries]
        output_b = str(base_output / "phase_b_cfg_grid")

        run_parallel_grid(
            flow_ckpts=ckpt_paths,
            cfg_scales=[3.0, 5.0, 7.0, 10.0, 15.0],
            num_steps=[5],
            seeds=list(range(args.phase_b_seeds)),
            gpus=args.gpus,
            output_root=output_b,
            skip_existing=args.skip_existing,
        )

        records_b = load_all_eval_records(Path(output_b))
        phase_b_agg = aggregate_by_combo(records_b)
        print_phase_summary(phase_b_agg, "PHASE B RESULTS: Fine CFG Grid on Top Epochs")

        if args.cleanup_intermediates:
            cleanup_intermediates(Path(output_b))

        summary_path = Path(output_b) / "phase_b_summary.json"
        with open(summary_path, "w") as f:
            json.dump({k: v for k, v in phase_b_agg.items()}, f, indent=2, default=str)
        print(f"[Phase B] Summary saved to {summary_path}")

    # ===================================================================
    # PHASE C: Full 10-seed Evaluation of Top Combos
    # ===================================================================
    if "C" in args.phases:
        print(f"\n{'#'*80}")
        print(f"#  PHASE C: Full {args.phase_c_seeds}-seed Eval on Top-{args.phase_c_top} Combos")
        print(f"{'#'*80}")

        # Load Phase B if needed
        if not phase_b_agg:
            saved = base_output / "phase_b_cfg_grid" / "phase_b_summary.json"
            if saved.exists():
                with open(saved) as f:
                    phase_b_agg = json.load(f)
                print(f"[Phase C] Loaded Phase B results from {saved}")
            else:
                print("[Phase C] ERROR: No Phase B results found. Run Phase B first.")
                return

        # Top combos by union_sr
        top_combos = sorted(phase_b_agg.values(),
                           key=lambda x: -x["union_sr"])[:args.phase_c_top]

        print(f"[Phase C] Top-{args.phase_c_top} combos from Phase B:")
        for item in top_combos:
            ep = extract_epoch(item["ckpt"] if isinstance(item["ckpt"], str) else "")
            print(f"  ep{ep} cfg={item['cfg_scale']} → union_SR={item['union_sr']:.1f}%")

        # Run each combo separately to get clean results
        for i, item in enumerate(top_combos):
            ep = extract_epoch(item["ckpt"] if isinstance(item["ckpt"], str) else "")
            cfg = item["cfg_scale"]
            ckpt_path = str(CKPT_DIR / entries[ep][0]) if ep in entries else item["ckpt"]
            output_c = str(base_output / f"phase_c_final_ep{ep}_cfg{cfg:g}")

            print(f"\n[Phase C] Running combo {i+1}/{len(top_combos)}: "
                  f"ep{ep}, cfg={cfg}")

            run_parallel_grid(
                flow_ckpts=[ckpt_path],
                cfg_scales=[cfg],
                num_steps=[5],
                seeds=list(range(args.phase_c_seeds)),
                gpus=args.gpus,
                output_root=output_c,
                skip_existing=args.skip_existing,
            )

        # Final summary across all Phase C
        print(f"\n{'='*90}")
        print(f"  PHASE C: FINAL PAPER-READY RESULTS ({args.phase_c_seeds} seeds)")
        print(f"{'='*90}")

        for item in top_combos:
            ep = extract_epoch(item["ckpt"] if isinstance(item["ckpt"], str) else "")
            cfg = item["cfg_scale"]
            output_c = base_output / f"phase_c_final_ep{ep}_cfg{cfg:g}"
            records = load_all_eval_records(output_c)
            if not records:
                print(f"  ep{ep} cfg={cfg}: NO RESULTS")
                continue
            agg = aggregate_by_combo(records)
            for key, a in agg.items():
                srak = a["sr_at_k"]
                sr1 = srak[0] if len(srak) >= 1 else 0
                sr5 = srak[4] if len(srak) >= 5 else srak[-1] if srak else 0
                sr10 = srak[9] if len(srak) >= 10 else srak[-1] if srak else 0
                print(f"  ep{ep:>3d} cfg={cfg:>4.0f} steps=5 | "
                      f"SR@1={sr1:>6.2f}% SR@5={sr5:>6.2f}% SR@10={sr10:>6.2f}% | "
                      f"mean={a['mean_sr']:.2f}±{a['std_sr']:.2f}% | "
                      f"E_diff={a['e_diff_mean']:.3f}")

        print(f"\n  Baselines:")
        print(f"  {'Old ep180 (historical)':>30s}: SR@10 = 61.4%")
        print(f"  {'New ep180 (cfg=7, this run)':>30s}: SR@10 = 63.6%")

    elapsed = time.time() - t_start
    print(f"\n[overnight] Total elapsed: {elapsed/3600:.1f} hours")
    print(f"[overnight] All results under: {base_output}")


if __name__ == "__main__":
    main()
