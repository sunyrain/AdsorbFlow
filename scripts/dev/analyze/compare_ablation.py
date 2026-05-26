#!/usr/bin/env python
"""
Ablation analysis: compare Flow+GemNet-OC vs Heuristic+GemNet-OC.

Reads both sets of results and computes comparative metrics:
  - R², Spearman ρ, MAE for each descriptor
  - Win/loss counts
  - Summary table for paper

Outputs to figures/ablation/

Usage:
    python scripts/compare_ablation.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "sans-serif",
})

MAIN_PATH = str(Path(__file__).resolve().parent.parent)
OUT_DIR = os.path.join(MAIN_PATH, "figures", "ablation")

FLOW_OER = os.path.join(MAIN_PATH, "examples/OER/data_flow_runB/oer_summary.csv")
HEUR_OER = os.path.join(MAIN_PATH, "examples/OER/data_heuristic/oer_summary.csv")
LIT_OER  = os.path.join(MAIN_PATH, "examples/OER/literature_data.pkl")

FLOW_CO2RR = os.path.join(MAIN_PATH, "examples/CO2RR/data_flow_runB/co2rr_summary.csv")
HEUR_CO2RR = os.path.join(MAIN_PATH, "examples/CO2RR/data_heuristic/co2rr_summary.csv")
LIT_CO2RR  = os.path.join(MAIN_PATH, "examples/CO2RR/literature_data.pkl")


def calibrate(x_ml, y_lit):
    slope, intercept, r, p, se = stats.linregress(x_ml, y_lit)
    y_cal = slope * x_ml + intercept
    r2 = r ** 2
    mae = np.abs(y_lit - y_cal).mean()
    rho, _ = stats.spearmanr(x_ml, y_lit)
    return {"slope": slope, "intercept": intercept, "r2": r2, "mae": mae,
            "rho": rho, "y_cal": y_cal, "n": len(x_ml)}


def compute_oer_overpotential(dG_OH):
    dG_O = 2 * dG_OH
    dG_OOH = dG_OH + 3.2
    dG1 = dG_OH
    dG2 = dG_O - dG_OH
    dG3 = dG_OOH - dG_O
    dG4 = 4.92 - dG_OOH
    return max(dG1, dG2, dG3, dG4) - 1.23


def compare_oer():
    """Compare OER: Flow vs Heuristic vs DFT."""
    print("=" * 60)
    print("OER ABLATION")
    print("=" * 60)

    flow = pd.read_csv(FLOW_OER)
    heur = pd.read_csv(HEUR_OER)
    lit = pd.DataFrame(pickle.load(open(LIT_OER, "rb")))

    results = {}
    descs = [
        ("dE_OH", "dG_OH", r"$\Delta G_{\mathrm{OH}}$"),
        ("dE_O",  "dG_O",  r"$\Delta G_{\mathrm{O}}$"),
        ("dE_OOH", "dG_OOH", r"$\Delta G_{\mathrm{OOH}}$"),
    ]

    print(f"\n{'Descriptor':<15} {'Method':<12} {'R²':>6} {'ρ':>6} {'MAE':>6} {'n':>3}")
    print("-" * 52)

    for ml_col, lit_col, label in descs:
        for method, df, tag in [("Flow", flow, "flow"), ("Heuristic", heur, "heur")]:
            merged = df.merge(lit, on="bulk_id", how="inner")
            valid = merged.dropna(subset=[ml_col, lit_col])
            if len(valid) >= 3:
                cal = calibrate(valid[ml_col].values, valid[lit_col].values)
                print(f"{lit_col:<15} {method:<12} {cal['r2']:6.3f} {cal['rho']:6.3f} {cal['mae']:6.3f} {cal['n']:>3}")
                results[f"{lit_col}_{tag}"] = cal

    # η comparison
    print(f"\n{'η_OER':<15} {'Method':<12} {'R²':>6} {'ρ':>6} {'MAE':>6} {'n':>3}")
    print("-" * 52)
    for method, df, tag in [("Flow", flow, "flow"), ("Heuristic", heur, "heur")]:
        merged = df.merge(lit, on="bulk_id", how="inner")
        valid = merged.dropna(subset=["dE_OH", "dG_OH"]).copy()
        if "dG_OH" in results.get(f"dG_OH_{tag}", {}):
            pass  # use pre-computed
        cal_oh = calibrate(valid["dE_OH"].values, valid["dG_OH"].values)
        valid["dG_OH_cal"] = cal_oh["slope"] * valid["dE_OH"] + cal_oh["intercept"]
        valid["eta_cal"] = [compute_oer_overpotential(g) for g in valid["dG_OH_cal"]]
        valid["eta_dft"] = valid["eta_oer"]
        eta_valid = valid.dropna(subset=["eta_dft"])
        if len(eta_valid) >= 3:
            r2 = np.corrcoef(eta_valid["eta_dft"], eta_valid["eta_cal"])[0, 1] ** 2
            rho, _ = stats.spearmanr(eta_valid["eta_dft"], eta_valid["eta_cal"])
            mae = np.abs(eta_valid["eta_dft"] - eta_valid["eta_cal"]).mean()
            print(f"{'η_OER':<15} {method:<12} {r2:6.3f} {rho:6.3f} {mae:6.3f} {len(eta_valid):>3}")
            results[f"eta_{tag}"] = {"r2": r2, "rho": rho, "mae": mae, "n": len(eta_valid)}

    return results


def compare_co2rr():
    """Compare CO2RR: Flow vs Heuristic vs DFT."""
    print("\n" + "=" * 60)
    print("CO₂RR ABLATION")
    print("=" * 60)

    flow = pd.read_csv(FLOW_CO2RR)
    heur = pd.read_csv(HEUR_CO2RR)
    lit = pd.DataFrame(pickle.load(open(LIT_CO2RR, "rb")))

    results = {}
    descs = [
        ("dE_CO", "dE_CO_lit", r"$\Delta E_{\mathrm{CO}}$"),
        ("dE_H",  "dE_H_lit",  r"$\Delta E_{\mathrm{H}}$"),
        ("dE_CHO","dE_CHO_lit",r"$\Delta E_{\mathrm{CHO}}$"),
        ("dE_COOH","dE_COOH_lit",r"$\Delta E_{\mathrm{COOH}}$"),
    ]

    # Check actual lit column names
    print(f"  Lit columns: {list(lit.columns)}")

    print(f"\n{'Descriptor':<15} {'Method':<12} {'R²':>6} {'ρ':>6} {'MAE':>6} {'n':>3}")
    print("-" * 52)

    for ml_col, lit_col, label in descs:
        # Try possible lit column names
        actual_lit_col = None
        for candidate in [lit_col, ml_col, lit_col.replace("_lit", "")]:
            if candidate in lit.columns:
                actual_lit_col = candidate
                break
        if actual_lit_col is None:
            print(f"  SKIP {ml_col}: no matching lit column")
            continue

        for method, df, tag in [("Flow", flow, "flow"), ("Heuristic", heur, "heur")]:
            merged = df.merge(lit, on="bulk_id", how="inner", suffixes=("_ml", "_dft"))
            # Figure out the merged column names
            if f"{ml_col}_ml" in merged.columns:
                mc = f"{ml_col}_ml"
            elif ml_col in merged.columns:
                mc = ml_col
            else:
                continue
            lc = actual_lit_col if actual_lit_col in merged.columns else f"{actual_lit_col}_dft"
            if lc not in merged.columns:
                continue

            valid = merged.dropna(subset=[mc, lc])
            if len(valid) >= 3:
                cal = calibrate(valid[mc].values, valid[lc].values)
                print(f"{ml_col:<15} {method:<12} {cal['r2']:6.3f} {cal['rho']:6.3f} {cal['mae']:6.3f} {cal['n']:>3}")
                results[f"{ml_col}_{tag}"] = cal

    return results


def make_comparison_figure(oer_results, co2rr_results):
    """Create a bar chart comparing Flow vs Heuristic R² and ρ."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect metrics
    metrics = []
    for key_base, label in [
        ("dG_OH", "ΔG_OH"), ("dG_O", "ΔG_O"), ("dG_OOH", "ΔG_OOH"),
        ("eta", "η_OER"),
    ]:
        flow_key = f"{key_base}_flow"
        heur_key = f"{key_base}_heur"
        if flow_key in oer_results and heur_key in oer_results:
            metrics.append({
                "desc": label,
                "task": "OER",
                "r2_flow": oer_results[flow_key]["r2"],
                "r2_heur": oer_results[heur_key]["r2"],
                "rho_flow": oer_results[flow_key]["rho"],
                "rho_heur": oer_results[heur_key]["rho"],
            })

    for key_base, label in [
        ("dE_CO", "ΔE_CO"), ("dE_H", "ΔE_H"),
        ("dE_CHO", "ΔE_CHO"), ("dE_COOH", "ΔE_COOH"),
    ]:
        flow_key = f"{key_base}_flow"
        heur_key = f"{key_base}_heur"
        if flow_key in co2rr_results and heur_key in co2rr_results:
            metrics.append({
                "desc": label,
                "task": "CO₂RR",
                "r2_flow": co2rr_results[flow_key]["r2"],
                "r2_heur": co2rr_results[heur_key]["r2"],
                "rho_flow": co2rr_results[flow_key]["rho"],
                "rho_heur": co2rr_results[heur_key]["rho"],
            })

    if not metrics:
        print("No comparable metrics found")
        return

    df = pd.DataFrame(metrics)

    # ── Figure: R² comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    descs = df["desc"].values
    x = np.arange(len(descs))
    width = 0.35

    for ax_idx, (metric, ylabel) in enumerate([("r2", "R²"), ("rho", "Spearman ρ")]):
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, df[f"{metric}_flow"], width, label="Flow + GemNet-OC",
                       color="#2196F3", alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, df[f"{metric}_heur"], width, label="Heuristic + GemNet-OC",
                       color="#FF9800", alpha=0.85, edgecolor="white")

        # Value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#2196F3")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#FF9800")

        ax.set_xticks(x)
        ax.set_xticklabels(descs, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.set_title(f"Ablation: {ylabel} comparison")

        # Divider between OER and CO2RR
        oer_count = sum(1 for m in metrics if m["task"] == "OER")
        if 0 < oer_count < len(metrics):
            ax.axvline(oer_count - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.text(oer_count/2 - 0.5, 1.08, "OER", ha="center", fontsize=9, color="gray")
            ax.text((oer_count + len(metrics))/2 - 0.5, 1.08, "CO₂RR", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(OUT_DIR, f"ablation_comparison{ext}"), bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {OUT_DIR}/ablation_comparison.pdf")

    # ── Δ improvement table ──
    print("\n" + "=" * 60)
    print("IMPROVEMENT: Flow over Heuristic")
    print("=" * 60)
    print(f"{'Descriptor':<12} {'ΔR²':>8} {'Δρ':>8} {'Winner':>10}")
    print("-" * 42)
    for _, row in df.iterrows():
        dr2 = row["r2_flow"] - row["r2_heur"]
        drho = row["rho_flow"] - row["rho_heur"]
        winner = "Flow" if (dr2 + drho) > 0 else "Heuristic"
        print(f"{row['desc']:<12} {dr2:+8.3f} {drho:+8.3f} {winner:>10}")

    # Save all metrics as JSON
    all_metrics = {"oer": {}, "co2rr": {}}
    for k, v in oer_results.items():
        all_metrics["oer"][k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                  for kk, vv in v.items() if kk != "y_cal"}
    for k, v in co2rr_results.items():
        all_metrics["co2rr"][k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                    for kk, vv in v.items() if kk != "y_cal"}
    json_path = os.path.join(OUT_DIR, "ablation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {json_path}")


def main():
    # Check files exist
    for f, label in [(FLOW_OER, "Flow OER"), (HEUR_OER, "Heuristic OER"),
                     (LIT_OER, "Lit OER"), (FLOW_CO2RR, "Flow CO2RR"),
                     (HEUR_CO2RR, "Heuristic CO2RR"), (LIT_CO2RR, "Lit CO2RR")]:
        if not os.path.exists(f):
            print(f"WARNING: {label} not found at {f}")

    oer_results = compare_oer()
    co2rr_results = compare_co2rr()
    make_comparison_figure(oer_results, co2rr_results)


if __name__ == "__main__":
    main()
