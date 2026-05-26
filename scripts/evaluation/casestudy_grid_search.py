#!/usr/bin/env python
"""
Grid search for OER and CO₂RR case studies.
Runs multiple cfg_scale × num_steps combos and collects results.

Usage:
    # OER volcano
    python scripts/casestudy_grid_search.py \
        --study oer \
        --flow-ckpt checkpoints/.../best_checkpoint.pt \
        --flow-config configs/flow/eqv2_fourier_cosine.yml \
        --relax-ckpt configs/relaxation/gemnet_oc/gemnet_oc_base_s2ef_2M.pt \
        --cfg-scales 3 5 7 --num-steps-list 5 10 \
        --output-base examples/OER/grid \
        --device cuda:0

    # CO₂RR screening
    python scripts/casestudy_grid_search.py \
        --study co2rr \
        --flow-ckpt checkpoints/.../best_checkpoint.pt \
        --flow-config configs/flow/eqv2_fourier_cosine.yml \
        --relax-ckpt configs/relaxation/gemnet_oc/gemnet_oc_base_s2ef_2M.pt \
        --cfg-scales 3 5 7 --num-steps-list 5 10 \
        --output-base examples/CO2RR/grid \
        --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def run_study(study, flow_ckpt, flow_config, relax_ckpt, cfg_scale, num_steps, output_dir, device):
    """Run a single case study evaluation."""
    if study == "oer":
        script = "scripts/run_oer_flow.py"
    elif study == "co2rr":
        script = "scripts/run_co2rr_flow.py"
    else:
        raise ValueError(f"Unknown study: {study}")

    cmd = [
        sys.executable, "-u", script,
        "--flow-ckpt", flow_ckpt,
        "--flow-config", flow_config,
        "--relax-ckpt", relax_ckpt,
        "--cfg-scale", str(cfg_scale),
        "--num-steps", str(num_steps),
        "--output-dir", output_dir,
        "--device", device,
    ]
    print(f"\n{'='*60}")
    print(f"[{study.upper()}-grid] cfg={cfg_scale}, K={num_steps}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"[{study.upper()}-grid] Done in {elapsed:.0f}s, exit={result.returncode}")
    return result.returncode


def analyze_oer(output_dir, lit_path):
    """Analyze OER results vs literature."""
    import pickle
    from scipy import stats

    summary_csv = os.path.join(output_dir, "oer_summary.csv")
    if not os.path.exists(summary_csv):
        return None

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))

    merged = ml.merge(lit, on="bulk_id", how="inner", suffixes=("_ml", "_lit"))
    if len(merged) < 3:
        return {"n_metals": len(ml), "n_matched": len(merged)}

    result = {"n_metals": len(ml), "n_matched": len(merged)}

    # Correlation: ΔG_OH (ML) vs ΔG_OH (lit)
    if "dG_OH" in merged.columns:
        valid = merged.dropna(subset=["dG_OH", "dG_OH_lit"])
        if len(valid) >= 3:
            _, _, r, _, _ = stats.linregress(valid["dG_OH_lit"], valid["dG_OH"])
            result["R2_dG_OH"] = round(r**2, 3)
            result["MAE_dG_OH"] = round(np.abs(valid["dG_OH_lit"] - valid["dG_OH"]).mean(), 3)

    # Overpotential correlation
    if "eta_oer" in merged.columns and "eta_oer_lit" in merged.columns:
        valid = merged.dropna(subset=["eta_oer", "eta_oer_lit"])
        if len(valid) >= 3:
            _, _, r, _, _ = stats.linregress(valid["eta_oer_lit"], valid["eta_oer"])
            result["R2_eta"] = round(r**2, 3)
            result["MAE_eta"] = round(np.abs(valid["eta_oer_lit"] - valid["eta_oer"]).mean(), 3)

    return result


def analyze_co2rr(output_dir, lit_path):
    """Analyze CO₂RR results vs literature."""
    import pickle
    from scipy import stats

    summary_csv = os.path.join(output_dir, "co2rr_summary.csv")
    if not os.path.exists(summary_csv):
        return None

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))

    merged = ml.merge(lit, on="bulk_id", how="inner", suffixes=("_ml", "_lit"))
    if len(merged) < 3:
        return {"n_catalysts": len(ml), "n_matched": len(merged)}

    result = {"n_catalysts": len(ml), "n_matched": len(merged)}

    # ΔE_CO correlation
    if "dE_CO_ml" in merged.columns and "dE_CO_lit" in merged.columns:
        valid = merged.dropna(subset=["dE_CO_ml", "dE_CO_lit"])
        if len(valid) >= 3:
            _, _, r, _, _ = stats.linregress(valid["dE_CO_lit"], valid["dE_CO_ml"])
            result["R2_dE_CO"] = round(r**2, 3)
            result["MAE_dE_CO"] = round(np.abs(valid["dE_CO_lit"] - valid["dE_CO_ml"]).mean(), 3)

    # Selectivity classification accuracy
    if "predicted_selectivity" in merged.columns and "main_product" in merged.columns:
        # Simplified: count correct product category predictions
        # Map literature products to our categories
        product_map = {
            "CH4/C2H4": "CH4/C2", "CO": "CO/formate", "formate": "CO/formate",
            "H2": "H2 (poisoned)", "CO/H2": "H2 (poisoned)",
        }
        merged["lit_category"] = merged["main_product"].map(
            lambda x: product_map.get(x, x))
        correct = (merged["predicted_selectivity"] == merged["lit_category"]).sum()
        result["selectivity_accuracy"] = round(correct / len(merged) * 100, 1)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, choices=["oer", "co2rr"])
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--flow-config", required=True)
    parser.add_argument("--relax-ckpt", required=True)
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[3, 5, 7])
    parser.add_argument("--num-steps-list", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parent.parent)
    if args.study == "oer":
        lit_path = os.path.join(main_path, "examples/OER/literature_data.pkl")
    else:
        lit_path = os.path.join(main_path, "examples/CO2RR/literature_data.pkl")

    results_file = os.path.join(args.output_base, f"{args.study}_grid_results.jsonl")
    os.makedirs(args.output_base, exist_ok=True)

    combos = [(cfg, K) for cfg in args.cfg_scales for K in args.num_steps_list]
    print(f"\n{args.study.upper()} Grid Search: {len(combos)} configs")
    print(f"cfg_scales={args.cfg_scales}, num_steps={args.num_steps_list}")

    for cfg, K in combos:
        tag = f"cfg{cfg:.0f}_K{K}"
        output_dir = os.path.join(args.output_base, tag)

        # Skip if already done
        if args.skip_existing:
            summary_name = "oer_summary.csv" if args.study == "oer" else "co2rr_summary.csv"
            if os.path.exists(os.path.join(output_dir, summary_name)):
                print(f"\n[SKIP] {tag} — already exists")
                continue

        run_study(args.study, args.flow_ckpt, args.flow_config, args.relax_ckpt,
                  cfg, K, output_dir, args.device)

        # Analyze
        if args.study == "oer":
            metrics = analyze_oer(output_dir, lit_path)
        else:
            metrics = analyze_co2rr(output_dir, lit_path)

        if metrics:
            record = {"cfg_scale": cfg, "num_steps": K, **metrics}
            with open(results_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"\n[RESULT] {tag}: {metrics}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{args.study.upper()} Grid Search Complete")
    print(f"Results → {results_file}")
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
