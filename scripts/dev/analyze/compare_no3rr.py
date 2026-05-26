#!/usr/bin/env python
"""
NO3RR comparison analysis: Flow vs Heuristic on alloy surfaces.

Compares:
  1. Adsorption energies (ΔE) for each intermediate
  2. Anomaly rates
  3. Correlation with DFT literature (where available)
  4. SO(3) rotation impact (multi-atom vs single-atom adsorbates)

Outputs figures to figures/no3rr/

Usage:
    python scripts/compare_no3rr.py
"""

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
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "sans-serif",
})

MAIN_PATH = str(Path(__file__).resolve().parent.parent)
OUT_DIR = os.path.join(MAIN_PATH, "figures", "no3rr")

FLOW_DIR  = os.path.join(MAIN_PATH, "examples/NO3RR/data_flow_runB")
HEUR_DIR  = os.path.join(MAIN_PATH, "examples/NO3RR/data_heuristic")
LIT_PATH  = os.path.join(MAIN_PATH, "examples/NO3RR/literature_data.pkl")

ADS_NAMES = ["NO3", "NO2", "NO", "N", "NH", "NH3"]
ADS_ATOMS = {"NO3": 4, "NO2": 3, "NO": 2, "N": 1, "NH": 2, "NH3": 4}
ADS_ROTATION = {"NO3": "full_3D", "NO2": "planar", "NO": "linear",
                "N": "none", "NH": "linear", "NH3": "full_3D"}


def load_results(results_dir):
    """Load results.csv and summary from an experiment directory."""
    csv_path = os.path.join(results_dir, "results.csv")
    sum_path = os.path.join(results_dir, "no3rr_summary.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found")
        return None, None
    df = pd.read_csv(csv_path)
    df_sum = pd.read_csv(sum_path) if os.path.exists(sum_path) else None
    return df, df_sum


def calibrate(x, y):
    """Linear calibration: return R², ρ, MAE, slope, intercept."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.array(x)[mask], np.array(y)[mask]
    if len(x) < 3:
        return {"r2": np.nan, "rho": np.nan, "mae": np.nan, "n": len(x)}
    slope, intercept, r, p, se = stats.linregress(x, y)
    y_cal = slope * x + intercept
    rho, _ = stats.spearmanr(x, y)
    return {"r2": r**2, "rho": rho, "mae": np.abs(y - y_cal).mean(),
            "slope": slope, "intercept": intercept, "n": len(x),
            "x": x, "y": y, "y_cal": y_cal}


def compare_energies():
    """Compare ΔE between Flow and Heuristic for each adsorbate."""
    flow_raw, flow_sum = load_results(FLOW_DIR)
    heur_raw, heur_sum = load_results(HEUR_DIR)

    if flow_sum is None or heur_sum is None:
        print("ERROR: Missing results. Run experiments first.")
        return None

    # Merge on bulk_id
    merged = flow_sum.merge(heur_sum, on="bulk_id", suffixes=("_flow", "_heur"))
    print(f"\n=== Energy Comparison ({len(merged)} catalysts) ===")

    results = {}
    for ads in ADS_NAMES:
        col_f = f"dE_{ads}_flow"
        col_h = f"dE_{ads}_heur"
        if col_f not in merged.columns or col_h not in merged.columns:
            col_f = f"dE_{ads}"  # same column name if no suffix
            col_h = f"dE_{ads}"
            continue

        valid = merged[[col_f, col_h]].dropna()
        if len(valid) == 0:
            continue

        de_flow = valid[col_f].values
        de_heur = valid[col_h].values
        diff = de_flow - de_heur

        # Flow gives lower energy = better placement
        n_flow_wins = np.sum(diff < -0.05)
        n_heur_wins = np.sum(diff > 0.05)
        n_tie = len(diff) - n_flow_wins - n_heur_wins

        results[ads] = {
            "n": len(valid),
            "mean_dE_flow": np.mean(de_flow),
            "mean_dE_heur": np.mean(de_heur),
            "mean_diff": np.mean(diff),
            "flow_wins": n_flow_wins,
            "heur_wins": n_heur_wins,
            "ties": n_tie,
            "n_atoms": ADS_ATOMS[ads],
            "rotation": ADS_ROTATION[ads],
        }

        print(f"\n  {ads} ({ADS_ATOMS[ads]} atoms, {ADS_ROTATION[ads]} rotation):")
        print(f"    ΔE_flow={np.mean(de_flow):.3f}±{np.std(de_flow):.3f}  "
              f"ΔE_heur={np.mean(de_heur):.3f}±{np.std(de_heur):.3f}")
        print(f"    Flow lower: {n_flow_wins}/{len(valid)}  "
              f"Heur lower: {n_heur_wins}/{len(valid)}  "
              f"Tie: {n_tie}/{len(valid)}")

    return results, merged


def compare_anomalies():
    """Compare anomaly rates."""
    flow_raw, _ = load_results(FLOW_DIR)
    heur_raw, _ = load_results(HEUR_DIR)

    if flow_raw is None or heur_raw is None:
        return None

    print(f"\n=== Anomaly Comparison ===")
    results = {}
    for ads in ADS_NAMES:
        f_sub = flow_raw[flow_raw["adsorbate"] == ads]
        h_sub = heur_raw[heur_raw["adsorbate"] == ads]
        f_anom = f_sub["anomaly"].sum() if len(f_sub) > 0 else 0
        h_anom = h_sub["anomaly"].sum() if len(h_sub) > 0 else 0
        f_total = len(f_sub)
        h_total = len(h_sub)
        results[ads] = {
            "flow_anom": f_anom, "flow_total": f_total,
            "heur_anom": h_anom, "heur_total": h_total,
        }
        print(f"  {ads}: Flow {f_anom}/{f_total} ({100*f_anom/max(f_total,1):.0f}%)  "
              f"Heur {h_anom}/{h_total} ({100*h_anom/max(h_total,1):.0f}%)")

    return results


def compare_with_literature():
    """Compare ML predictions with DFT literature values."""
    if not os.path.exists(LIT_PATH):
        print("\nNo literature data found.")
        return None, None

    with open(LIT_PATH, "rb") as f:
        lit = pickle.load(f)
    lit_df = pd.DataFrame(lit)

    _, flow_sum = load_results(FLOW_DIR)
    _, heur_sum = load_results(HEUR_DIR)

    if flow_sum is None or heur_sum is None:
        return None, None

    flow_results = {}
    heur_results = {}

    print(f"\n=== Literature Comparison ({len(lit_df)} catalysts with DFT) ===")

    for ads in ADS_NAMES:
        lit_col = f"dE_{ads}"
        if lit_col not in lit_df.columns:
            continue

        # Flow vs literature
        lit_sub = lit_df[["src_id", lit_col]].rename(columns={lit_col: f"{lit_col}_lit"})
        merged_f = flow_sum.merge(lit_sub, left_on="bulk_id",
                                   right_on="src_id", how="inner")
        if len(merged_f) >= 3:
            cal = calibrate(merged_f[f"dE_{ads}"].values, merged_f[f"{lit_col}_lit"].values)
            flow_results[ads] = cal
            print(f"\n  {ads} Flow vs DFT: R²={cal['r2']:.3f}, ρ={cal['rho']:.3f}, "
                  f"MAE={cal['mae']:.3f} eV (n={cal['n']})")

        # Heuristic vs literature
        merged_h = heur_sum.merge(lit_sub, left_on="bulk_id",
                                   right_on="src_id", how="inner")
        if len(merged_h) >= 3:
            cal = calibrate(merged_h[f"dE_{ads}"].values, merged_h[f"{lit_col}_lit"].values)
            heur_results[ads] = cal
            print(f"  {ads} Heur vs DFT: R²={cal['r2']:.3f}, ρ={cal['rho']:.3f}, "
                  f"MAE={cal['mae']:.3f} eV (n={cal['n']})")

    return flow_results, heur_results


def plot_energy_comparison(merged):
    """Plot ΔE comparison: Flow vs Heuristic for each adsorbate."""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ads in enumerate(ADS_NAMES):
        ax = axes[idx]
        col_f = f"dE_{ads}_flow"
        col_h = f"dE_{ads}_heur"

        if col_f not in merged.columns or col_h not in merged.columns:
            ax.set_visible(False)
            continue

        valid = merged[[col_f, col_h, "formula_flow"]].dropna()
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = valid[col_f].values
        y = valid[col_h].values
        labels = valid["formula_flow"].values

        # Color by alloy vs pure metal
        colors = ["#e74c3c" if "3" in str(f) or "Cu" in str(f) and len(set(str(f))) > 3
                   else "#3498db" for f in labels]

        # Determine if pure metal or alloy
        is_alloy = []
        for f in labels:
            syms = set()
            import re
            for m in re.finditer(r'[A-Z][a-z]?', str(f)):
                syms.add(m.group())
            is_alloy.append(len(syms) > 1)

        alloy_mask = np.array(is_alloy)

        # Plot
        if np.any(~alloy_mask):
            ax.scatter(x[~alloy_mask], y[~alloy_mask], c="#3498db", s=60,
                       edgecolors="k", linewidth=0.5, label="Pure metals", zorder=3)
        if np.any(alloy_mask):
            ax.scatter(x[alloy_mask], y[alloy_mask], c="#e74c3c", s=60,
                       edgecolors="k", linewidth=0.5, label="Alloys", zorder=3)

        # y=x line
        lims = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

        # Label outliers (diff > 0.5 eV)
        diff = np.abs(x - y)
        for j in np.where(diff > 0.5)[0]:
            ax.annotate(labels[j], (x[j], y[j]), fontsize=7, alpha=0.7,
                        xytext=(5, 5), textcoords="offset points")

        rot_label = ADS_ROTATION[ads]
        n_atoms = ADS_ATOMS[ads]
        ax.set_title(f"*{ads} ({n_atoms} atoms, {rot_label})")
        ax.set_xlabel("ΔE Flow (eV)")
        ax.set_ylabel("ΔE Heuristic (eV)")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

        # Stats
        r, _ = stats.pearsonr(x, y)
        ax.text(0.95, 0.05, f"r={r:.2f}\nn={len(x)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "flow_vs_heuristic_dE.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_anomaly_bars(anomaly_results):
    """Bar chart of anomaly rates."""
    os.makedirs(OUT_DIR, exist_ok=True)

    ads_list = [a for a in ADS_NAMES if a in anomaly_results]
    flow_rates = [100 * anomaly_results[a]["flow_anom"] / max(anomaly_results[a]["flow_total"], 1) for a in ads_list]
    heur_rates = [100 * anomaly_results[a]["heur_anom"] / max(anomaly_results[a]["heur_total"], 1) for a in ads_list]

    x = np.arange(len(ads_list))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, flow_rates, w, label="Flow + GemNet-OC", color="#2ecc71", edgecolor="k", linewidth=0.5)
    ax.bar(x + w/2, heur_rates, w, label="Heuristic + GemNet-OC", color="#e67e22", edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"*{a}\n({ADS_ATOMS[a]} atoms)" for a in ads_list])
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("NO3RR on Alloy (100) Surfaces: Relaxation Anomaly Rate")
    ax.legend()
    ax.set_ylim(0, max(max(flow_rates, default=0), max(heur_rates, default=0)) * 1.3 + 5)

    path = os.path.join(OUT_DIR, "anomaly_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_rotation_advantage(energy_results):
    """Show Flow advantage as function of molecular complexity (rotation type)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    if not energy_results:
        return

    rot_order = {"none": 0, "linear": 1, "planar": 2, "full_3D": 3}
    ads_sorted = sorted(energy_results.keys(), key=lambda a: rot_order.get(ADS_ROTATION[a], 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(ads_sorted))
    flow_win_pct = []
    for ads in ads_sorted:
        r = energy_results[ads]
        total = r["flow_wins"] + r["heur_wins"] + r["ties"]
        pct = 100 * r["flow_wins"] / max(total, 1)
        flow_win_pct.append(pct)

    colors = []
    for ads in ads_sorted:
        rot = ADS_ROTATION[ads]
        if rot == "full_3D":
            colors.append("#e74c3c")
        elif rot == "planar":
            colors.append("#e67e22")
        elif rot == "linear":
            colors.append("#3498db")
        else:
            colors.append("#95a5a6")

    bars = ax.bar(x_pos, flow_win_pct, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(50, color="k", linestyle="--", alpha=0.3, label="Equal")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"*{a}\n({ADS_ROTATION[a]})" for a in ads_sorted], fontsize=9)
    ax.set_ylabel("Flow wins (%)")
    ax.set_title("Flow Advantage by Adsorbate Rotation Complexity")
    ax.set_ylim(0, 105)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="k", label="Full 3D rotation"),
        Patch(facecolor="#e67e22", edgecolor="k", label="Planar rotation"),
        Patch(facecolor="#3498db", edgecolor="k", label="Linear rotation"),
        Patch(facecolor="#95a5a6", edgecolor="k", label="No rotation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    path = os.path.join(OUT_DIR, "rotation_advantage.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_reaction_diagram(merged):
    """Plot NO3RR reaction free energy diagram for selected catalysts."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Select a few representative catalysts
    targets = ["mp-30", "mp-23", "no3rr-Cu3Ni", "no3rr-Cu3Pd"]
    target_labels = {"mp-30": "Cu", "mp-23": "Ni",
                     "no3rr-Cu3Ni": "Cu₃Ni", "no3rr-Cu3Pd": "Cu₃Pd"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label, suffix in [(ax1, "Flow", "_flow"), (ax2, "Heuristic", "_heur")]:
        for bulk_id in targets:
            row = merged[merged["bulk_id"] == bulk_id]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            energies = []
            valid = True
            for ads in ADS_NAMES:
                col = f"dE_{ads}{suffix}"
                if col not in merged.columns or pd.isna(row.get(col)):
                    valid = False
                    break
                energies.append(row[col])

            if not valid or len(energies) != 6:
                continue

            x_steps = np.arange(len(ADS_NAMES))
            ax.plot(x_steps, energies, "o-", label=target_labels.get(bulk_id, bulk_id),
                    linewidth=2, markersize=6)

        ax.set_xticks(range(len(ADS_NAMES)))
        ax.set_xticklabels([f"*{a}" for a in ADS_NAMES])
        ax.set_ylabel("ΔE (eV)")
        ax.set_title(f"NO3RR Pathway — {label}")
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", linestyle="--", alpha=0.2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "reaction_diagram.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_alloy_vs_pure(energy_results):
    """Show that Flow advantage is bigger on alloys than pure metals."""
    os.makedirs(OUT_DIR, exist_ok=True)

    flow_raw, flow_sum = load_results(FLOW_DIR)
    heur_raw, heur_sum = load_results(HEUR_DIR)

    if flow_sum is None or heur_sum is None:
        return

    merged = flow_sum.merge(heur_sum, on="bulk_id", suffixes=("_flow", "_heur"))

    # Classify as pure or alloy
    import re
    def is_alloy(formula):
        syms = set(re.findall(r'[A-Z][a-z]?', str(formula)))
        return len(syms) > 1

    merged["is_alloy"] = merged["formula_flow"].apply(is_alloy)

    fig, ax = plt.subplots(figsize=(8, 5))

    for ads_idx, ads in enumerate(["NO3", "NH3", "NO2", "N"]):
        col_f = f"dE_{ads}_flow"
        col_h = f"dE_{ads}_heur"
        if col_f not in merged.columns:
            continue

        valid = merged[["is_alloy", col_f, col_h]].dropna()
        diff = valid[col_f] - valid[col_h]

        alloy_diff = diff[valid["is_alloy"]].mean()
        pure_diff = diff[~valid["is_alloy"]].mean()

        x = ads_idx * 3
        ax.bar(x, pure_diff, color="#3498db", edgecolor="k", linewidth=0.5, width=0.8,
               label="Pure metals" if ads_idx == 0 else "")
        ax.bar(x + 1, alloy_diff, color="#e74c3c", edgecolor="k", linewidth=0.5, width=0.8,
               label="Alloys" if ads_idx == 0 else "")

    tick_positions = [i * 3 + 0.5 for i in range(4)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["*NO3\n(4 atoms)", "*NH3\n(4 atoms)", "*NO2\n(3 atoms)", "*N\n(1 atom)"])
    ax.set_ylabel("ΔE(Flow) − ΔE(Heur) (eV)")
    ax.set_title("Flow vs Heuristic Energy Difference\n(negative = Flow better)")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.legend()

    path = os.path.join(OUT_DIR, "alloy_vs_pure_advantage.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(energy_results, anomaly_results, flow_lit, heur_lit):
    """Print a comprehensive summary table."""
    print("\n" + "=" * 90)
    print("NO3RR Case Study: Flow vs Heuristic Summary")
    print("=" * 90)
    print(f"{'Adsorbate':>10} {'Atoms':>5} {'Rotation':>10} "
          f"{'Flow wins':>10} {'Heur wins':>10} "
          f"{'Flow anom':>10} {'Heur anom':>10} "
          f"{'ΔE diff':>8}")
    print("-" * 90)

    for ads in ADS_NAMES:
        n_atoms = ADS_ATOMS[ads]
        rot = ADS_ROTATION[ads]

        e_res = energy_results.get(ads, {})
        a_res = anomaly_results.get(ads, {})

        fw = e_res.get("flow_wins", 0)
        hw = e_res.get("heur_wins", 0)
        diff = e_res.get("mean_diff", 0)
        fa = a_res.get("flow_anom", 0)
        ft = a_res.get("flow_total", 0)
        ha = a_res.get("heur_anom", 0)
        ht = a_res.get("heur_total", 0)

        print(f"{'*'+ads:>10} {n_atoms:>5} {rot:>10} "
              f"{fw:>10} {hw:>10} "
              f"{fa}/{ft:>8} {ha}/{ht:>8} "
              f"{diff:>+8.3f}")

    print("=" * 90)

    if flow_lit and heur_lit:
        print("\n=== Literature Comparison (R²) ===")
        for ads in ADS_NAMES:
            fl = flow_lit.get(ads, {})
            hl = heur_lit.get(ads, {})
            if fl or hl:
                print(f"  *{ads}: Flow R²={fl.get('r2', np.nan):.3f}  "
                      f"Heur R²={hl.get('r2', np.nan):.3f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check data availability
    for d, label in [(FLOW_DIR, "Flow"), (HEUR_DIR, "Heuristic")]:
        if not os.path.exists(os.path.join(d, "results.csv")):
            print(f"ERROR: {label} results not found at {d}")
            print("Run the experiment first.")
            sys.exit(1)

    # Compare energies
    result = compare_energies()
    if result is None:
        sys.exit(1)
    energy_results, merged = result

    # Compare anomalies
    anomaly_results = compare_anomalies()

    # Compare with literature
    flow_lit, heur_lit = compare_with_literature()

    # Generate figures
    plot_energy_comparison(merged)
    plot_anomaly_bars(anomaly_results)
    plot_rotation_advantage(energy_results)
    plot_reaction_diagram(merged)
    plot_alloy_vs_pure(energy_results)

    # Summary table
    print_summary_table(energy_results, anomaly_results, flow_lit, heur_lit)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
