#!/usr/bin/env python
"""
NRR grid search: test multiple cfg_scale × num_steps combinations.
Runs run_nrr_flow.py for each combo and collects results.

Usage:
    python scripts/evaluation/nrr_grid_search.py \
        --flow-ckpt checkpoints/.../best_checkpoint.pt \
        --flow-config configs/flow/xxx.yml \
        --relax-ckpt checkpoints/gemnet_oc_base_s2ef_2M.pt \
        --cfg-scales 0 3 5 7 --num-steps-list 5 10 \
        --output-base examples/NRR/grid_runB \
        --device cuda:0
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def run_nrr(flow_ckpt, flow_config, relax_ckpt, cfg_scale, num_steps, output_dir, device):
    """Run a single NRR evaluation."""
    cmd = [
        sys.executable, "-u", "scripts/case_studies/run_nrr_flow.py",
        "--flow-ckpt", flow_ckpt,
        "--flow-config", flow_config,
        "--relax-ckpt", relax_ckpt,
        "--cfg-scale", str(cfg_scale),
        "--num-steps", str(num_steps),
        "--output-dir", output_dir,
        "--device", device,
    ]
    print(f"\n{'='*60}")
    print(f"[NRR-grid] cfg={cfg_scale}, K={num_steps} → {output_dir}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"[NRR-grid] Done in {elapsed:.0f}s, exit code={result.returncode}")
    return result.returncode


def analyze_nrr(output_dir, lit_path):
    """Analyze NRR results vs literature."""
    results_csv = os.path.join(output_dir, "results.csv")
    if not os.path.exists(results_csv):
        return None

    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))
    ml = pd.read_csv(results_csv)

    ml_H = ml[ml["adsorbate"] == "H"][["bulk_id", "min_E_ml"]].rename(columns={"min_E_ml": "E_ml_H"})
    ml_NNH = ml[ml["adsorbate"] == "NNH"][["bulk_id", "min_E_ml"]].rename(columns={"min_E_ml": "E_ml_NNH"})
    merged = lit.merge(ml_H, on="bulk_id", how="left").merge(ml_NNH, on="bulk_id", how="left")

    result = {}
    result["n_H"] = int(merged["E_ml_H"].notna().sum())
    result["n_NNH"] = int(merged["E_ml_NNH"].notna().sum())

    # H parity
    valid_H = merged.dropna(subset=["E_ml_H"])
    if len(valid_H) >= 3:
        _, _, r_H, _, _ = stats.linregress(valid_H["E_lit_H"], valid_H["E_ml_H"])
        result["R2_H"] = round(r_H ** 2, 3)
        result["MAE_H"] = round(np.abs(valid_H["E_lit_H"] - valid_H["E_ml_H"]).mean(), 3)

    # NNH parity
    valid_NNH = merged.dropna(subset=["E_ml_NNH"])
    if len(valid_NNH) >= 3:
        _, _, r_NNH, _, _ = stats.linregress(valid_NNH["E_lit_NNH"], valid_NNH["E_ml_NNH"])
        result["R2_NNH"] = round(r_NNH ** 2, 3)
        result["MAE_NNH"] = round(np.abs(valid_NNH["E_lit_NNH"] - valid_NNH["E_ml_NNH"]).mean(), 3)

    # Classification
    both = merged.dropna(subset=["E_ml_H", "E_ml_NNH"])
    if len(both) > 0:
        correct = 0
        for _, row in both.iterrows():
            pred = "NRR" if row["E_ml_NNH"] < -0.5 else "HER"
            if pred == row["reaction"]:
                correct += 1
        result["classify_acc"] = round(correct / len(both), 3)
        result["classify_n"] = len(both)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--flow-config", required=True)
    parser.add_argument("--relax-ckpt", required=True)
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[0, 3, 5, 7])
    parser.add_argument("--num-steps-list", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parent.parent)
    lit_path = os.path.join(main_path, "examples/NRR/literature_data.pkl")

    all_results = []
    for num_steps in args.num_steps_list:
        for cfg_scale in args.cfg_scales:
            output_dir = os.path.join(args.output_base, f"cfg{cfg_scale:.0f}_K{num_steps}")

            # Skip if results.csv already exists
            if os.path.exists(os.path.join(output_dir, "results.csv")):
                print(f"\n[NRR-grid] Skipping cfg={cfg_scale}, K={num_steps} (already done)")
            else:
                run_nrr(args.flow_ckpt, args.flow_config, args.relax_ckpt,
                        cfg_scale, num_steps, output_dir, args.device)

            analysis = analyze_nrr(output_dir, lit_path)
            if analysis:
                analysis["cfg_scale"] = cfg_scale
                analysis["num_steps"] = num_steps
                all_results.append(analysis)

    # Summary table
    print(f"\n{'='*80}")
    print("NRR Grid Search Summary")
    print(f"{'='*80}")
    print(f"{'cfg':>5s} {'K':>3s} | {'H':>4s} {'NNH':>4s} | {'R²_H':>6s} {'MAE_H':>6s} | {'R²_NNH':>7s} {'MAE_NNH':>8s} | {'Classify':>10s}")
    print(f"{'-'*5} {'-'*3} | {'-'*4} {'-'*4} | {'-'*6} {'-'*6} | {'-'*7} {'-'*8} | {'-'*10}")
    for r in all_results:
        cls = f"{r.get('classify_acc',0)*100:.0f}% ({r.get('classify_n',0)})" if 'classify_acc' in r else "N/A"
        print(f"{r['cfg_scale']:5.0f} {r['num_steps']:3d} | {r['n_H']:4d} {r['n_NNH']:4d} | "
              f"{r.get('R2_H', 0):6.3f} {r.get('MAE_H', 0):6.3f} | "
              f"{r.get('R2_NNH', 0):7.3f} {r.get('MAE_NNH', 0):8.3f} | {cls}")

    # Save to jsonl
    out_jsonl = os.path.join(args.output_base, "nrr_grid_results.jsonl")
    with open(out_jsonl, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_jsonl}")


if __name__ == "__main__":
    main()
