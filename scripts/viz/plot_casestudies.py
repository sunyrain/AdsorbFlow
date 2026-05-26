#!/usr/bin/env python
"""
Comprehensive evaluation and publication figures for OER volcano and CO₂RR case studies.

Approach: ML descriptor ΔE_ml = E(adslab) - E(slab) is on GemNet's internal
reference scale.  We calibrate via linear regression against DFT literature
values and report R², MAE, Spearman ρ.

Produces:
  OER (3 figures):
    oer_volcano.pdf        — (a) volcano with DFT + ML-calibrated, (b) η parity
    oer_parity_all.pdf     — parity for ΔG_OH, ΔG_O, ΔG_OOH (3 panels)
    oer_ranking.pdf        — ML vs DFT rank comparison bar chart

  CO₂RR (3 figures):
    co2rr_selectivity.pdf  — (a) selectivity map, (b) ΔE_CO parity
    co2rr_parity_all.pdf   — parity for ΔE_CO, ΔE_H (2 panels)
    co2rr_classification.pdf — pairwise ordering + classification table

Usage:
    python scripts/plot_casestudies.py \
        --oer-dir examples/OER/data_flow_runB \
        --co2rr-dir examples/CO2RR/data_flow_runB \
        --output-dir figures/casestudies
"""

import argparse
import json
import os
import pickle
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
    """Linear regression: y_lit = slope * x_ml + intercept."""
    slope, intercept, r, p, se = stats.linregress(x_ml, y_lit)
    y_cal = slope * x_ml + intercept
    r2 = r ** 2
    mae = np.abs(y_lit - y_cal).mean()
    rho, p_rho = stats.spearmanr(x_ml, y_lit)
    return {
        "slope": slope, "intercept": intercept,
        "r2": r2, "mae": mae, "rho": rho, "p_rho": p_rho,
        "y_cal": y_cal,
    }


def _save(fig, out_dir, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"{name}{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_dir}/{name}.pdf,.png")


def _parity_panel(ax, x_dft, y_cal, labels, title, xlabel, ylabel, cal):
    """Draw a parity panel with annotation and stats box."""
    ax.scatter(x_dft, y_cal, s=70, c=COLORS["ml"],
               edgecolors="black", linewidth=0.5, zorder=3)
    for xv, yv, lab in zip(x_dft, y_cal, labels):
        ax.annotate(lab, (xv, yv), fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    all_vals = np.concatenate([x_dft, y_cal])
    pad = (all_vals.max() - all_vals.min()) * 0.15
    lo, hi = all_vals.min() - pad, all_vals.max() + pad
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.4)

    box_text = (f"R² = {cal['r2']:.3f}\n"
                f"MAE = {cal['mae']:.3f} eV\n"
                f"ρ = {cal['rho']:.3f}\n"
                f"n = {len(x_dft)}")
    ax.text(0.05, 0.95, box_text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


# ═══════════════════════════════════════════
#                   OER
# ═══════════════════════════════════════════

def plot_oer_all(oer_dir, lit_path, out_dir):
    """Generate all OER figures and metrics."""
    summary_csv = os.path.join(oer_dir, "oer_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"  SKIP OER: {summary_csv} not found")
        return {}

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))
    merged = ml.merge(lit, on="bulk_id", how="inner")

    print(f"  Merged {len(merged)} metals for OER")

    # ── Calibrate all 3 descriptors ──
    descs = [
        ("dE_OH", "dG_OH", r"$\Delta G_{\mathrm{OH}}$"),
        ("dE_O",  "dG_O",  r"$\Delta G_{\mathrm{O}}$"),
        ("dE_OOH", "dG_OOH", r"$\Delta G_{\mathrm{OOH}}$"),
    ]
    cals = {}
    for ml_col, lit_col, _ in descs:
        valid = merged.dropna(subset=[ml_col, lit_col])
        if len(valid) >= 3:
            cal = calibrate(valid[ml_col].values, valid[lit_col].values)
            cal["n"] = len(valid)
            cals[lit_col] = cal
            print(f"    {lit_col}: R²={cal['r2']:.3f}, MAE={cal['mae']:.3f} eV, "
                  f"ρ={cal['rho']:.3f}, n={cal['n']}")

    if "dG_OH" not in cals:
        print("  SKIP OER: not enough dG_OH matches")
        return {}

    # ── Figure 1: Volcano + η parity ──
    cal_oh = cals["dG_OH"]
    valid_oh = merged.dropna(subset=["dE_OH", "dG_OH"]).copy()
    valid_oh["dG_OH_cal"] = cal_oh["slope"] * valid_oh["dE_OH"] + cal_oh["intercept"]
    valid_oh["eta_cal"] = [compute_oer_overpotential(g) for g in valid_oh["dG_OH_cal"]]

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
                       c=COLORS["lit"], edgecolors="black", linewidth=0.5,
                       zorder=3, alpha=0.8)
            ax.annotate(row["metal"], (row["dG_OH"], row["eta_oer"]),
                        fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points", color=COLORS["lit"])

    # ML (calibrated)
    for _, row in valid_oh.iterrows():
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

    # Panel B: η_OER parity (DFT η vs ML-calibrated η)
    ax = axes[1]
    eta_dft = valid_oh["eta_oer"].values
    eta_ml = valid_oh["eta_cal"].values

    ax.scatter(eta_dft, eta_ml, s=70, c=COLORS["ml"],
               edgecolors="black", linewidth=0.5, zorder=3)
    for _, row in valid_oh.iterrows():
        label = row.get("metal", row["bulk_id"][:6])
        ax.annotate(label, (row["eta_oer"], row["eta_cal"]),
                    fontsize=6.5, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    all_eta = np.concatenate([eta_dft, eta_ml])
    pad = (all_eta.max() - all_eta.min()) * 0.15
    lo, hi = all_eta.min() - pad, all_eta.max() + pad
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.4)

    rho_eta, _ = stats.spearmanr(eta_dft, eta_ml)
    r2_eta = np.corrcoef(eta_dft, eta_ml)[0, 1] ** 2
    mae_eta = np.abs(eta_dft - eta_ml).mean()
    box_text = (f"R² = {r2_eta:.3f}\nMAE = {mae_eta:.3f} V\n"
                f"ρ = {rho_eta:.3f}\nn = {len(eta_dft)}")
    ax.text(0.05, 0.95, box_text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$\eta_{\mathrm{OER}}^{\mathrm{DFT}}$ (V)")
    ax.set_ylabel(r"$\eta_{\mathrm{OER}}^{\mathrm{cal}}$ (V)")
    ax.set_title(r"(b) $\eta_{\mathrm{OER}}$ Parity")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout()
    _save(fig, out_dir, "oer_volcano")

    # ── Figure 2: All 3 descriptor parity plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panel_labels = ["(a)", "(b)", "(c)"]
    for i, (ml_col, lit_col, desc_label) in enumerate(descs):
        ax = axes[i]
        valid = merged.dropna(subset=[ml_col, lit_col]).copy()
        if lit_col not in cals:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        cal = cals[lit_col]
        valid["y_cal"] = cal["slope"] * valid[ml_col] + cal["intercept"]
        labels = [row.get("metal", row["bulk_id"][:6]) for _, row in valid.iterrows()]

        _parity_panel(
            ax,
            valid[lit_col].values,
            valid["y_cal"].values,
            labels,
            f"{panel_labels[i]} {desc_label} Parity",
            f"{desc_label}" + r"$^{\mathrm{DFT}}$ (eV)",
            f"{desc_label}" + r"$^{\mathrm{cal}}$ (eV)",
            cal,
        )

    plt.tight_layout()
    _save(fig, out_dir, "oer_parity_all")

    # ── Figure 3: Ranking comparison ──
    valid_rank = merged.dropna(subset=["dE_OH", "dG_OH"]).copy()
    valid_rank["dG_OH_cal"] = cal_oh["slope"] * valid_rank["dE_OH"] + cal_oh["intercept"]
    valid_rank["rank_dft"] = valid_rank["dG_OH"].rank()
    valid_rank["rank_ml"] = valid_rank["dG_OH_cal"].rank()

    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = valid_rank.sort_values("rank_dft")
    names = [r.get("metal", r["bulk_id"][:6]) for _, r in sorted_df.iterrows()]
    dft_ranks = sorted_df["rank_dft"].values
    ml_ranks = sorted_df["rank_ml"].values

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, dft_ranks, w, label="DFT rank", color=COLORS["lit"], alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, ml_ranks, w, label="ML rank", color=COLORS["ml"], alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"Rank (ascending $\Delta G_{\mathrm{OH}}$)")
    ax.set_title("OER Catalyst Ranking: DFT vs AdsorbFlow")
    ax.legend()

    tau, p_tau = stats.kendalltau(dft_ranks, ml_ranks)
    ax.text(0.02, 0.95, f"Kendall τ = {tau:.3f}\nSpearman ρ = {cal_oh['rho']:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    _save(fig, out_dir, "oer_ranking")

    # ── Metrics ──
    metrics = {"n_metals": int(len(merged))}
    for lit_col, cal in cals.items():
        key = lit_col.replace("dG_", "")
        metrics[f"r2_{key}"] = round(cal["r2"], 3)
        metrics[f"mae_{key}"] = round(cal["mae"], 3)
        metrics[f"rho_{key}"] = round(cal["rho"], 3)
        metrics[f"slope_{key}"] = round(cal["slope"], 3)
        metrics[f"intercept_{key}"] = round(cal["intercept"], 3)
    metrics["r2_eta"] = round(r2_eta, 3)
    metrics["mae_eta_V"] = round(mae_eta, 3)
    metrics["rho_eta"] = round(rho_eta, 3)
    metrics["kendall_tau"] = round(tau, 3)

    return metrics


# ═══════════════════════════════════════════
#                  CO₂RR
# ═══════════════════════════════════════════

def plot_co2rr_all(co2rr_dir, lit_path, out_dir):
    """Generate all CO₂RR figures and metrics."""
    summary_csv = os.path.join(co2rr_dir, "co2rr_summary.csv")
    if not os.path.exists(summary_csv):
        print(f"  SKIP CO2RR: {summary_csv} not found")
        return {}

    ml = pd.read_csv(summary_csv)
    lit = pd.DataFrame(pickle.load(open(lit_path, "rb")))
    merged = ml.merge(lit, on="bulk_id", how="inner", suffixes=("_ml", "_lit"))

    print(f"  Merged {len(merged)} catalysts for CO2RR")

    # ── Calibrate dE_CO and dE_H ──
    cals = {}
    desc_pairs = [
        ("dE_CO_ml", "dE_CO_lit", r"$\Delta E_{\mathrm{CO}}$", "CO"),
        ("dE_H_ml",  "dE_H_lit",  r"$\Delta E_{\mathrm{H}}$",  "H"),
    ]
    for ml_col, lit_col, _, key in desc_pairs:
        valid = merged.dropna(subset=[ml_col, lit_col])
        if len(valid) >= 3:
            cal = calibrate(valid[ml_col].values, valid[lit_col].values)
            cal["n"] = len(valid)
            cals[key] = cal
            print(f"    {key}: R²={cal['r2']:.3f}, MAE={cal['mae']:.3f} eV, "
                  f"ρ={cal['rho']:.3f}, n={cal['n']}")

    if "CO" not in cals:
        print("  SKIP CO2RR: not enough dE_CO matches")
        return {}

    cal_co = cals["CO"]
    cal_h = cals.get("H")

    # Auto-detect outliers via iterative Cook's distance (threshold 4/n)
    valid_co_all = merged.dropna(subset=["dE_CO_ml", "dE_CO_lit"]).copy()
    outlier_ids = set()
    outlier_names = []
    clean_df = valid_co_all.copy()
    for iteration in range(3):  # max 3 rounds
        n = len(clean_df)
        if n < 5:
            break
        x = clean_df["dE_CO_ml"].values
        y = clean_df["dE_CO_lit"].values
        sl, ic, r, p, se = stats.linregress(x, y)
        pred = sl * x + ic
        res = pred - y
        x_mean = x.mean()
        h = 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)
        mse = np.sum(res**2) / (n - 2)
        cooks = res**2 / (2 * mse) * h / (1 - h)**2
        threshold = 4 / n
        bad_mask = cooks > threshold
        if not bad_mask.any():
            break
        bad_rows = clean_df[bad_mask]
        for _, r_row in bad_rows.iterrows():
            name = r_row.get("name", r_row["bulk_id"])
            bid = r_row["bulk_id"]
            cd = cooks[clean_df.index.get_loc(r_row.name)]
            outlier_ids.add(bid)
            outlier_names.append(name)
            print(f"    Outlier (Cook={cd:.3f} > {threshold:.3f}): {name} "
                  f"(ML={r_row['dE_CO_ml']:.3f}, DFT={r_row['dE_CO_lit']:.3f})")
        clean_df = clean_df[~bad_mask]

    if outlier_names:
        print(f"    Total outliers: {outlier_names}")
        if len(clean_df) >= 3:
            cal_co_clean = calibrate(clean_df["dE_CO_ml"].values, clean_df["dE_CO_lit"].values)
            cal_co_clean["n"] = len(clean_df)
            cals["CO_clean"] = cal_co_clean
            print(f"    CO (clean): R²={cal_co_clean['r2']:.3f}, MAE={cal_co_clean['mae']:.3f} eV, "
                  f"ρ={cal_co_clean['rho']:.3f}, n={cal_co_clean['n']}")

    # ── Figure 1: Selectivity map + dE_CO parity ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Selectivity map
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

    # Literature
    lit_valid = lit.dropna(subset=["dE_CO", "dE_H"])
    ax.scatter(lit_valid["dE_CO"], lit_valid["dE_H"], marker="D", s=60,
               c=COLORS["lit"], edgecolors="black", linewidth=0.5, zorder=3, alpha=0.8)
    for _, row in lit_valid.iterrows():
        ax.annotate(row["name"], (row["dE_CO"], row["dE_H"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", color=COLORS["lit"])

    # Use clean calibration if available
    cal_co_use = cals.get("CO_clean", cal_co)

    # ML calibrated
    if cal_h is not None:
        both = merged.dropna(subset=["dE_CO_ml", "dE_H_ml"]).copy()
        if len(both) > 0:
            both["dE_CO_cal"] = cal_co_use["slope"] * both["dE_CO_ml"] + cal_co_use["intercept"]
            both["dE_H_cal"] = cal_h["slope"] * both["dE_H_ml"] + cal_h["intercept"]
            is_outlier = both["bulk_id"].isin(outlier_ids)
            normal = both[~is_outlier]
            outliers = both[is_outlier]
            ax.scatter(normal["dE_CO_cal"], normal["dE_H_cal"], marker="o", s=80,
                       c=COLORS["ml"], edgecolors="black", linewidth=0.5, zorder=4)
            if len(outliers) > 0:
                ax.scatter(outliers["dE_CO_cal"], outliers["dE_H_cal"], marker="X", s=90,
                           c=COLORS["highlight"], edgecolors="black", linewidth=0.5, zorder=4)
            for _, row in both.iterrows():
                label = row.get("name", row["bulk_id"][:8])
                color = COLORS["highlight"] if row["bulk_id"] in outlier_ids else COLORS["ml"]
                ax.annotate(label, (row["dE_CO_cal"], row["dE_H_cal"]),
                            fontsize=7, ha="center", va="top", xytext=(0, -6),
                            textcoords="offset points", color=color)

    ax.set_xlabel(r"$\Delta E_{\mathrm{CO}}$ (eV)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{H}}$ (eV)")
    ax.set_title("(a) CO$_2$RR Selectivity Map")
    legend_handles = [
        plt.Line2D([0], [0], marker="D", color=COLORS["lit"], markersize=6, ls=""),
        plt.Line2D([0], [0], marker="o", color=COLORS["ml"], markersize=7, ls=""),
    ]
    legend_labels = ["DFT (literature)", "AdsorbFlow (calibrated)"]
    if outlier_names:
        legend_handles.append(plt.Line2D([0], [0], marker="X", color=COLORS["highlight"],
                                         markersize=7, ls=""))
        legend_labels.append("Outlier")
    ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=9)
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-0.8, 0.6)

    # Panel B: dE_CO parity
    ax = axes[1]
    valid_co = merged.dropna(subset=["dE_CO_ml", "dE_CO_lit"]).copy()
    valid_co["dE_CO_cal"] = cal_co_use["slope"] * valid_co["dE_CO_ml"] + cal_co_use["intercept"]
    is_out = valid_co["bulk_id"].isin(outlier_ids)
    normal = valid_co[~is_out]
    outliers_df = valid_co[is_out]

    ax.scatter(normal["dE_CO_lit"], normal["dE_CO_cal"], s=70, c=COLORS["ml"],
               edgecolors="black", linewidth=0.5, zorder=3)
    if len(outliers_df) > 0:
        ax.scatter(outliers_df["dE_CO_lit"], outliers_df["dE_CO_cal"], s=80,
                   c=COLORS["highlight"], marker="X", edgecolors="black", linewidth=0.5, zorder=3)
    for _, row in valid_co.iterrows():
        label = row.get("name", row["bulk_id"][:6])
        ax.annotate(label, (row["dE_CO_lit"], row["dE_CO_cal"]),
                    fontsize=6.5, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    all_vals = np.concatenate([valid_co["dE_CO_lit"].values, valid_co["dE_CO_cal"].values])
    pad = (all_vals.max() - all_vals.min()) * 0.15
    lo, hi = all_vals.min() - pad, all_vals.max() + pad
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.4)

    if "CO_clean" in cals:
        cc = cals["CO_clean"]
        box_text = (f"R² = {cc['r2']:.3f} (excl. {len(outlier_names)} outlier{'s' if len(outlier_names)>1 else ''})\n"
                    f"MAE = {cc['mae']:.3f} eV\n"
                    f"ρ = {cc['rho']:.3f}\n"
                    f"n = {cc['n']} / {cal_co['n']}")
    else:
        box_text = (f"R² = {cal_co['r2']:.3f}\nMAE = {cal_co['mae']:.3f} eV\n"
                    f"ρ = {cal_co['rho']:.3f}\nn = {cal_co['n']}")
    ax.text(0.05, 0.95, box_text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$\Delta E_{\mathrm{CO}}^{\mathrm{DFT}}$ (eV)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{CO}}^{\mathrm{cal}}$ (eV)")
    ax.set_title(r"(b) $\Delta E_{\mathrm{CO}}$ Parity")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout()
    _save(fig, out_dir, "co2rr_selectivity")

    # ── Figure 2: All descriptor parity (CO and H) ──
    n_panels = sum(1 for k in ["CO", "H"] if k in cals)
    if n_panels > 0:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0
        panel_labels = ["(a)", "(b)"]
        for ml_col, lit_col, desc_label, key in desc_pairs:
            if key not in cals:
                continue
            ax = axes[panel_idx]
            cal = cals[key]
            valid = merged.dropna(subset=[ml_col, lit_col]).copy()
            valid["y_cal"] = cal["slope"] * valid[ml_col] + cal["intercept"]
            labels = [r.get("name", r["bulk_id"][:6]) for _, r in valid.iterrows()]

            _parity_panel(
                ax, valid[lit_col].values, valid["y_cal"].values, labels,
                f"{panel_labels[panel_idx]} {desc_label} Parity",
                f"{desc_label}" + r"$^{\mathrm{DFT}}$ (eV)",
                f"{desc_label}" + r"$^{\mathrm{cal}}$ (eV)",
                cal,
            )
            panel_idx += 1

        plt.tight_layout()
        _save(fig, out_dir, "co2rr_parity_all")

    # ── Figure 3: Classification + pairwise ordering ──
    # Pairwise ordering (compute for all and clean separately)
    valid_co_pair = merged.dropna(subset=["dE_CO_ml", "dE_CO_lit"])
    n_pair = len(valid_co_pair)
    pairwise_acc = None
    pairwise_acc_clean = None
    class_acc = None

    if n_pair >= 3:
        dft_co = valid_co_pair["dE_CO_lit"].values
        ml_co = valid_co_pair["dE_CO_ml"].values
        names_co = [r.get("name", r["bulk_id"][:6]) for _, r in valid_co_pair.iterrows()]

        correct = 0
        total = 0
        for i in range(n_pair):
            for j in range(i + 1, n_pair):
                total += 1
                if (dft_co[i] - dft_co[j]) * (ml_co[i] - ml_co[j]) > 0:
                    correct += 1
        pairwise_acc = correct / total if total > 0 else 0

        # Clean pairwise (excluding outliers)
        clean_pair = valid_co_pair[~valid_co_pair["bulk_id"].isin(outlier_ids)]
        if len(clean_pair) >= 3:
            dft_clean = clean_pair["dE_CO_lit"].values
            ml_clean = clean_pair["dE_CO_ml"].values
            correct_c = 0
            total_c = 0
            for i in range(len(dft_clean)):
                for j in range(i + 1, len(dft_clean)):
                    total_c += 1
                    if (dft_clean[i] - dft_clean[j]) * (ml_clean[i] - ml_clean[j]) > 0:
                        correct_c += 1
            pairwise_acc_clean = correct_c / total_c if total_c > 0 else 0

        # Selectivity classification
        cat_mapping = {
            "H2": "H$_2$", "CO": "CO", "CO/H2": "CO",
            "CH4/C2H4": "CH$_x$", "formate": "Formate",
            "CO/C2": "CO", "C2": "CH$_x$", "formate/CO": "Formate",
            "CO/formate": "CO", "CH4": "CH$_x$",
        }

        valid_class = merged.dropna(subset=["dE_CO_ml", "main_product"]).copy()
        valid_class["dE_CO_cal"] = cal_co_use["slope"] * valid_class["dE_CO_ml"] + cal_co_use["intercept"]

        def classify_co(de_co):
            if de_co < -0.8:
                return "H$_2$"
            elif de_co < -0.3:
                return "CH$_x$"
            elif de_co < 0.1:
                return "CO"
            else:
                return "Formate"

        valid_class["cat_dft"] = valid_class["main_product"].map(cat_mapping)
        valid_class["cat_ml"] = valid_class["dE_CO_cal"].apply(classify_co)
        valid_class = valid_class.dropna(subset=["cat_dft"])

        class_acc = (valid_class["cat_dft"] == valid_class["cat_ml"]).mean()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel A: Pairwise ordering (show both all and clean)
        ax = axes[0]
        bar_labels = ["All\ncorrect", "All\nincorrect"]
        bar_vals = [correct, total - correct]
        bar_colors = [COLORS["ml"], COLORS["lit"]]
        if pairwise_acc_clean is not None:
            bar_labels += ["Clean\ncorrect", "Clean\nincorrect"]
            bar_vals += [correct_c, total_c - correct_c]
            bar_colors += [COLORS["volcano"], COLORS["highlight"]]
        ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Number of pairs")
        title_parts = [f"All: {pairwise_acc:.1%} ({total} pairs)"]
        if pairwise_acc_clean is not None:
            title_parts.append(f"Clean: {pairwise_acc_clean:.1%} ({total_c} pairs)")
        ax.set_title(f"(a) Pairwise ΔE$_{{\\mathrm{{CO}}}}$ Ordering\n" + ", ".join(title_parts))

        # Panel B: Classification table
        ax = axes[1]
        ax.axis("off")
        table_data = []
        for _, row in valid_class.iterrows():
            name = row.get("name", row["bulk_id"][:6])
            de_co_lit = row.get("dE_CO_lit", np.nan)
            de_co_lit_str = f'{de_co_lit:.2f}' if pd.notna(de_co_lit) else "—"
            match = "✓" if row["cat_dft"] == row["cat_ml"] else "✗"
            table_data.append([name, de_co_lit_str, f'{row["dE_CO_cal"]:.2f}',
                               row["cat_dft"], row["cat_ml"], match])
        table = ax.table(
            cellText=table_data,
            colLabels=["Catalyst", "ΔE_CO\n(DFT)", "ΔE_CO\n(cal)", "DFT class", "ML class", "Match"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        for i, row_data in enumerate(table_data):
            color = "#C8E6C9" if row_data[-1] == "✓" else "#FFCDD2"
            table[(i + 1, 5)].set_facecolor(color)
        ax.set_title(f"(b) Selectivity Classification (Accuracy: {class_acc:.1%})")

        plt.tight_layout()
        _save(fig, out_dir, "co2rr_classification")

    # ── Metrics ──
    metrics = {"n_catalysts": int(len(merged))}
    for key, cal in cals.items():
        metrics[f"r2_{key}"] = round(cal["r2"], 3)
        metrics[f"mae_{key}"] = round(cal["mae"], 3)
        metrics[f"rho_{key}"] = round(cal["rho"], 3)
        metrics[f"n_{key}"] = cal["n"]
    if pairwise_acc is not None:
        metrics["pairwise_ordering_acc"] = round(pairwise_acc, 3)
    if pairwise_acc_clean is not None:
        metrics["pairwise_ordering_acc_clean"] = round(pairwise_acc_clean, 3)
    if class_acc is not None:
        metrics["selectivity_class_acc"] = round(class_acc, 3)
    if outlier_names:
        metrics["outliers"] = outlier_names

    return metrics


# ═══════════════════════════════════════════
#                  MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oer-dir", default=None)
    parser.add_argument("--co2rr-dir", default=None)
    parser.add_argument("--output-dir", default="figures/casestudies")
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parent.parent)
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}

    if args.oer_dir:
        lit_oer = os.path.join(main_path, "examples/OER/literature_data.pkl")
        print("=" * 60)
        print("OER Volcano Case Study")
        print("=" * 60)
        m = plot_oer_all(args.oer_dir, lit_oer, args.output_dir)
        all_metrics["OER"] = m
        print()

    if args.co2rr_dir:
        lit_co2rr = os.path.join(main_path, "examples/CO2RR/literature_data.pkl")
        print("=" * 60)
        print("CO₂RR Screening Case Study")
        print("=" * 60)
        m = plot_co2rr_all(args.co2rr_dir, lit_co2rr, args.output_dir)
        all_metrics["CO2RR"] = m
        print()

    if all_metrics:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print("=" * 60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        print(json.dumps(all_metrics, indent=2))
        print(f"\nMetrics saved -> {metrics_path}")

    if not args.oer_dir and not args.co2rr_dir:
        print("No data directories specified. Use --oer-dir and/or --co2rr-dir.")


if __name__ == "__main__":
    main()
