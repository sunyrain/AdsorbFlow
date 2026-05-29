#!/usr/bin/env python
"""
Generate publication-quality figures for OER volcano and CO₂RR case studies.

Approach: ML descriptor ΔE_ml = E(adslab) - E(slab) is on GemNet's internal
reference scale.  We calibrate via linear regression against DFT literature
values and report R², MAE, slope.

Produces:
  1. OER: (a) volcano (calibrated ΔG_OH → η_OER), (b) ΔG_OH parity
  2. CO₂RR: (a) selectivity map (calibrated ΔE_CO vs ΔE_H), (b) ΔE_CO parity

Usage:
    python scripts/viz/plot_casestudies_v1.py \
        --oer-dir examples/OER/data_flow \
        --co2rr-dir examples/CO2RR/data_flow \
        --output-dir figures/casestudies
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── Publication style ──
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})

COLORS = {
    "ml": "#2196F3",
    "lit": "#F44336",
    "volcano": "#4CAF50",
    "highlight": "#FF9800",
    "gray": "#9E9E9E",
}


def compute_oer_overpotential(dG_OH):
    """Theoretical OER η using Man et al. 2011 scaling relations."""
    dG_O = 2 * dG_OH
    dG_OOH = dG_OH + 3.2
    dG1 = dG_OH
    dG2 = dG_O - dG_OH
    dG3 = dG_OOH - dG_O
    dG4 = 4.92 - dG_OOH
    return max(dG1, dG2, dG3, dG4) - 1.23


def calibrate(x_ml, y_lit):
    """Linear regression: y_lit = slope * x_ml + intercept.
    Returns slope, intercept, r2, mae, calibrated y values."""
    slope, intercept, r, p, se = stats.linregress(x_ml, y_lit)
    y_cal = slope * x_ml + intercept
    r2 = r ** 2
    mae = np.abs(y_lit - y_cal).mean()
    return slope, intercept, r2, mae, y_cal


def plot_oer(oer_dir, lit_path, out_dir):
    """OER: volcano + ΔG_OH parity using linear calibration."""
    summary_csv = os.path.join(oer_dir, "oer_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"  SKIP OER: {summary_csv} not found")
        return {}

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))

    # Merge ML with literature
    merged = ml.merge(lit, on="bulk_id", how="inner")
    valid = merged.dropna(subset=["dE_OH", "dG_OH"])
    if len(valid) < 3:
        print(f"  SKIP OER: only {len(valid)} matched metals (need >= 3)")
        return {}

    print(f"  Matched {len(valid)} metals for OER parity")

    # Linear calibration: dG_OH_lit = a * dE_OH_ml + b
    slope, intercept, r2, mae, dG_OH_cal = calibrate(
        valid["dE_OH"].values, valid["dG_OH"].values
    )
    valid = valid.copy()
    valid["dG_OH_cal"] = dG_OH_cal
    valid["eta_cal"] = [compute_oer_overpotential(g) for g in dG_OH_cal]

    print(f"  OER dG_OH calibration: slope={slope:.3f}, intercept={intercept:.3f}")
    print(f"  R2={r2:.3f}, MAE={mae:.3f} eV")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Volcano
    ax = axes[0]
    dG_range = np.linspace(-0.5, 2.5, 200)
    eta_theory = [compute_oer_overpotential(dg) for dg in dG_range]
    ax.plot(dG_range, eta_theory, "-", color=COLORS["gray"], lw=2, alpha=0.5,
            label="Scaling relation", zorder=1)

    # Literature
    for _, row in lit.iterrows():
        if pd.notna(row.get("dG_OH")) and pd.notna(row.get("eta_oer")):
            ax.scatter(row["dG_OH"], row["eta_oer"], marker="D", s=60,
                       c=COLORS["lit"], edgecolors="black", linewidth=0.5, zorder=3, alpha=0.8)
            ax.annotate(row["metal"], (row["dG_OH"], row["eta_oer"]),
                        fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", color=COLORS["lit"])

    # ML (calibrated)
    for _, row in valid.iterrows():
        ax.scatter(row["dG_OH_cal"], row["eta_cal"], marker="o", s=80,
                   c=COLORS["ml"], edgecolors="black", linewidth=0.5, zorder=4)
        label = row.get("metal", row["bulk_id"][:6])
        ax.annotate(label, (row["dG_OH_cal"], row["eta_cal"]),
                    fontsize=7, ha="center", va="top", xytext=(0, -6),
                    textcoords="offset points", color=COLORS["ml"])

    ax.set_xlabel(r"$\Delta G_{\mathrm{OH}}$ (eV)")
    ax.set_ylabel(r"$\eta_{\mathrm{OER}}$ (V)")
    ax.set_title("(a) OER Volcano Plot")
    ax.legend([
        plt.Line2D([0], [0], marker="D", color=COLORS["lit"], markersize=6, ls=""),
        plt.Line2D([0], [0], marker="o", color=COLORS["ml"], markersize=7, ls=""),
    ], ["DFT (literature)", "AdsorbFlow (calibrated)"], loc="upper right")
    ax.set_xlim(-0.6, 2.5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    # Panel B: Parity
    ax = axes[1]
    ax.scatter(valid["dG_OH"], valid["dG_OH_cal"], s=70,
               c=COLORS["ml"], edgecolors="black", linewidth=0.5, zorder=3)
    for _, row in valid.iterrows():
        label = row.get("metal", row["bulk_id"][:6])
        ax.annotate(label, (row["dG_OH"], row["dG_OH_cal"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    lims = [min(valid["dG_OH"].min(), valid["dG_OH_cal"].min()) - 0.3,
            max(valid["dG_OH"].max(), valid["dG_OH_cal"].max()) + 0.3]
    ax.plot(lims, lims, "--", color="black", lw=1, alpha=0.5)

    ax.text(0.05, 0.95, f"R$^2$ = {r2:.3f}\nMAE = {mae:.3f} eV\nn = {len(valid)}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$\Delta G_{\mathrm{OH}}^{\mathrm{DFT}}$ (eV)")
    ax.set_ylabel(r"$\Delta G_{\mathrm{OH}}^{\mathrm{cal}}$ (eV)")
    ax.set_title(r"(b) $\Delta G_{\mathrm{OH}}$ Parity")
    ax.set_aspect("equal")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"oer_volcano{ext}"))
    print(f"  Saved OER figures -> {out_dir}/oer_volcano.pdf,.png")
    plt.close()

    return {
        "n_metals": int(len(valid)),
        "r2_dG_OH": round(r2, 3),
        "mae_dG_OH": round(mae, 3),
        "slope": round(slope, 3),
        "intercept": round(intercept, 3),
    }


def plot_co2rr(co2rr_dir, lit_path, out_dir):
    """CO2RR: selectivity map + dE_CO parity using linear calibration."""
    summary_csv = os.path.join(co2rr_dir, "co2rr_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"  SKIP CO2RR: {summary_csv} not found")
        return {}

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))

    merged = ml.merge(lit, on="bulk_id", how="inner", suffixes=("_ml", "_lit"))
    valid_co = merged.dropna(subset=["dE_CO_ml", "dE_CO_lit"])
    if len(valid_co) < 3:
        print(f"  SKIP CO2RR: only {len(valid_co)} matched catalysts (need >= 3)")
        return {}

    print(f"  Matched {len(valid_co)} catalysts for CO2RR parity")

    # Calibrate dE_CO
    slope_co, intercept_co, r2_co, mae_co, dE_CO_cal = calibrate(
        valid_co["dE_CO_ml"].values, valid_co["dE_CO_lit"].values
    )
    valid_co = valid_co.copy()
    valid_co["dE_CO_cal"] = dE_CO_cal

    print(f"  CO2RR dE_CO calibration: slope={slope_co:.3f}, intercept={intercept_co:.3f}")
    print(f"  R2={r2_co:.3f}, MAE={mae_co:.3f} eV")

    # Calibrate dE_H if available
    valid_h = merged.dropna(subset=["dE_H_ml", "dE_H_lit"])
    r2_h, mae_h = None, None
    slope_h, intercept_h = None, None
    if len(valid_h) >= 3:
        slope_h, intercept_h, r2_h, mae_h, dE_H_cal = calibrate(
            valid_h["dE_H_ml"].values, valid_h["dE_H_lit"].values
        )
        valid_h = valid_h.copy()
        valid_h["dE_H_cal"] = dE_H_cal
        print(f"  CO2RR dE_H calibration: R2={r2_h:.3f}, MAE={mae_h:.3f} eV")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Selectivity map (calibrated dE_CO vs calibrated dE_H)
    ax = axes[0]

    regions = [
        ((-1.5, -0.8), (-0.8, 0.6), "#FFCDD2", "H$_2$\n(poisoned)"),
        ((-0.8, -0.3), (-0.8, 0.6), "#C8E6C9", "CH$_4$/C$_2$"),
        ((-0.3,  0.1), (-0.8, 0.6), "#BBDEFB", "CO/C$_2$"),
        (( 0.1,  1.0), (-0.8, 0.6), "#FFF9C4", "CO/formate"),
    ]
    for (x0, x1), (y0, y1), color, label in regions:
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor=color, alpha=0.4, zorder=0)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, y1 - 0.05, label, ha="center", va="top",
                fontsize=8, fontstyle="italic", alpha=0.7)

    # Literature points
    lit_valid = lit.dropna(subset=["dE_CO", "dE_H"])
    ax.scatter(lit_valid["dE_CO"], lit_valid["dE_H"], marker="D", s=60,
               c=COLORS["lit"], edgecolors="black", linewidth=0.5, zorder=3, alpha=0.8)
    for _, row in lit_valid.iterrows():
        ax.annotate(row["name"], (row["dE_CO"], row["dE_H"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", color=COLORS["lit"])

    # ML calibrated points (only those with both CO and H)
    if slope_h is not None:
        both = merged.dropna(subset=["dE_CO_ml", "dE_H_ml"]).copy()
        if len(both) > 0:
            both["dE_CO_cal"] = slope_co * both["dE_CO_ml"] + intercept_co
            both["dE_H_cal"] = slope_h * both["dE_H_ml"] + intercept_h
            ax.scatter(both["dE_CO_cal"], both["dE_H_cal"], marker="o", s=80,
                       c=COLORS["ml"], edgecolors="black", linewidth=0.5, zorder=4)
            for _, row in both.iterrows():
                label = row.get("name", row["bulk_id"][:8])
                ax.annotate(label, (row["dE_CO_cal"], row["dE_H_cal"]),
                            fontsize=7, ha="center", va="top", xytext=(0, -6),
                            textcoords="offset points", color=COLORS["ml"])

    ax.set_xlabel(r"$\Delta E_{\mathrm{CO}}$ (eV)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{H}}$ (eV)")
    ax.set_title("(a) CO$_2$RR Selectivity Map")
    ax.legend([
        plt.Line2D([0], [0], marker="D", color=COLORS["lit"], markersize=6, ls=""),
        plt.Line2D([0], [0], marker="o", color=COLORS["ml"], markersize=7, ls=""),
    ], ["DFT (literature)", "AdsorbFlow (calibrated)"], loc="upper left")
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-0.8, 0.6)

    # Panel B: dE_CO parity
    ax = axes[1]
    ax.scatter(valid_co["dE_CO_lit"], valid_co["dE_CO_cal"], s=70,
               c=COLORS["ml"], edgecolors="black", linewidth=0.5, zorder=3)
    for _, row in valid_co.iterrows():
        label = row.get("name", row["bulk_id"][:6])
        ax.annotate(label, (row["dE_CO_lit"], row["dE_CO_cal"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    lims = [min(valid_co["dE_CO_lit"].min(), valid_co["dE_CO_cal"].min()) - 0.3,
            max(valid_co["dE_CO_lit"].max(), valid_co["dE_CO_cal"].max()) + 0.3]
    ax.plot(lims, lims, "--", color="black", lw=1, alpha=0.5)

    ax.text(0.05, 0.95, f"R$^2$ = {r2_co:.3f}\nMAE = {mae_co:.3f} eV\nn = {len(valid_co)}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$\Delta E_{\mathrm{CO}}^{\mathrm{DFT}}$ (eV)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{CO}}^{\mathrm{cal}}$ (eV)")
    ax.set_title(r"(b) $\Delta E_{\mathrm{CO}}$ Parity")
    ax.set_aspect("equal")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"co2rr_selectivity{ext}"))
    print(f"  Saved CO2RR figures -> {out_dir}/co2rr_selectivity.pdf,.png")
    plt.close()

    metrics = {
        "n_catalysts": int(len(valid_co)),
        "r2_dE_CO": round(r2_co, 3),
        "mae_dE_CO": round(mae_co, 3),
        "slope_CO": round(slope_co, 3),
        "intercept_CO": round(intercept_co, 3),
    }
    if r2_h is not None:
        metrics["r2_dE_H"] = round(r2_h, 3)
        metrics["mae_dE_H"] = round(mae_h, 3)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oer-dir", default=None, help="OER results directory")
    parser.add_argument("--co2rr-dir", default=None, help="CO2RR results directory")
    parser.add_argument("--output-dir", default="figures/casestudies")
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parents[2])
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}

    if args.oer_dir:
        lit_oer = os.path.join(main_path, "examples/OER/literature_data.pkl")
        print("Plotting OER volcano...")
        m = plot_oer(args.oer_dir, lit_oer, args.output_dir)
        all_metrics["OER"] = m

    if args.co2rr_dir:
        lit_co2rr = os.path.join(main_path, "examples/CO2RR/literature_data.pkl")
        print("Plotting CO2RR selectivity...")
        m = plot_co2rr(args.co2rr_dir, lit_co2rr, args.output_dir)
        all_metrics["CO2RR"] = m

    if all_metrics:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics -> {metrics_path}")
        print(json.dumps(all_metrics, indent=2))

    if not args.oer_dir and not args.co2rr_dir:
        print("No data directories specified. Use --oer-dir and/or --co2rr-dir.")


if __name__ == "__main__":
    main()
