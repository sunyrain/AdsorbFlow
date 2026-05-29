#!/usr/bin/env python
"""
OER heuristic baseline (ablation study): skip Flow model entirely.

Pipeline: bulk → slab(111) → heuristic placement → GemNet-OC relaxation → ΔE

This isolates the contribution of the Flow model by comparing:
  - Flow + GemNet-OC (full pipeline)   vs
  - Heuristic + GemNet-OC (this script, no Flow)

Usage:
    python scripts/run_oer_heuristic.py \
        --relax-ckpt checkpoints/gemnet_oc_base_s2ef_2M.pt \
        --output-dir examples/OER/data_heuristic \
        --device cuda:0
"""

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import ase.io
from ase.optimize import BFGS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adsorbdiff.placement import (
    Adsorbate,
    AdsorbateSlabConfig,
    Bulk,
    Slab,
    DetectTrajAnomaly,
)
from adsorbdiff.relaxation.calculator import AdsorbDiffCalculator

# ── OER adsorbates (same as flow script) ──
OER_ADSORBATES = {
    "OH":  {"smiles": "*OH",  "relax_steps": 30, "interstitial_gap": 0.1},
    "O":   {"smiles": "*O",   "relax_steps": 30, "interstitial_gap": 0.1},
    "OOH": {"smiles": "*OOH", "relax_steps": 50, "interstitial_gap": 0.2},
}


def relax_bare_slab(slab_atoms, calc, out_dir, fmax=0.03, max_steps=50):
    """Relax a bare slab and return (energy, relaxed Atoms)."""
    slab_copy = slab_atoms.copy()
    slab_copy.calc = calc
    traj_path = os.path.join(out_dir, "slab_opt.traj")
    opt = BFGS(slab_copy, trajectory=traj_path,
               logfile=os.path.join(out_dir, "slab_log.log"))
    opt.run(fmax=fmax, steps=max_steps)
    return slab_copy.get_potential_energy(), slab_copy


def main():
    parser = argparse.ArgumentParser(description="OER heuristic baseline (ablation)")
    parser.add_argument("--relax-ckpt", required=True)
    parser.add_argument("--output-dir", default="examples/OER/data_heuristic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-bulks", type=int, default=None)
    parser.add_argument("--miller", type=int, nargs=3, default=[1, 1, 1])
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parent.parent.parent)
    db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/adsorbates.pkl")
    bulks_path = os.path.join(main_path, "examples/OER/OER_bulks.pkl")
    miller = tuple(args.miller)

    if not os.path.exists(bulks_path):
        print("ERROR: Run 'python scripts/prepare_oer_data.py' first.")
        sys.exit(1)

    with open(bulks_path, "rb") as f:
        bulks = pickle.load(f)
    bulk_ids = [row["src_id"] for row in bulks]
    if args.max_bulks:
        bulk_ids = bulk_ids[:args.max_bulks]

    adsorbates = {}
    for ads_name, ads_info in OER_ADSORBATES.items():
        adsorbates[ads_name] = Adsorbate(
            adsorbate_smiles_from_db=ads_info["smiles"],
            adsorbate_db_path=db_path,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"OER Heuristic Baseline: {len(bulk_ids)} metals × {len(OER_ADSORBATES)} adsorbates")
    print(f"miller={miller}, NO Flow model")
    t_start = time.time()

    # ══════ Load GemNet-OC calculator ══════
    calc = AdsorbDiffCalculator(checkpoint_path=args.relax_ckpt, cpu=False)

    all_energies = []

    for i, bulk_id in enumerate(bulk_ids):
        print(f"\n[{i+1}/{len(bulk_ids)}] {bulk_id}...")
        try:
            bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulks_path)
            slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
            if not slab:
                print(f"  SKIP: no slab for miller={miller}")
                continue
            slab_atoms = slab[0]

            # ── Relax bare slab ──
            slab_dir = os.path.join(args.output_dir, f"{bulk_id}_slab")
            os.makedirs(slab_dir, exist_ok=True)
            slab_ase = slab_atoms.atoms.copy() if hasattr(slab_atoms, "atoms") else slab_atoms.copy()
            E_slab, slab_relaxed = relax_bare_slab(slab_ase, calc, slab_dir)
            print(f"  slab E = {E_slab:.3f} eV")

            # ── Heuristic place & relax each adsorbate ──
            for ads_name, ads_info in OER_ADSORBATES.items():
                ads_obj = adsorbates[ads_name]
                out_dir = os.path.join(args.output_dir, f"{bulk_id}_{ads_name}")
                os.makedirs(out_dir, exist_ok=True)

                try:
                    # Heuristic placement (same call as flow script's initial guess)
                    adslab = AdsorbateSlabConfig(
                        slab_atoms, ads_obj, mode="heuristic", num_sites=1,
                    ).atoms_list[0]

                    # Save initial structure
                    ase.io.write(os.path.join(out_dir, "init.xyz"), adslab)

                    # GemNet-OC relaxation
                    adslab.calc = calc
                    opt = BFGS(adslab, trajectory=os.path.join(out_dir, "opt.traj"),
                               logfile=os.path.join(out_dir, "log.log"))
                    opt.run(fmax=0.05, steps=ads_info["relax_steps"])

                    E_adslab = adslab.get_potential_energy()
                    print(f"  {ads_name}: E = {E_adslab:.3f} eV")

                    # Anomaly detection
                    traj = ase.io.read(os.path.join(out_dir, "opt.traj"), ":")
                    detector = DetectTrajAnomaly(
                        traj[0], traj[-1], traj[0].get_tags(),
                        final_slab_atoms=slab_relaxed,
                    )
                    is_anomaly = (detector.is_adsorbate_dissociated() or
                                  detector.is_adsorbate_desorbed() or
                                  detector.has_surface_changed() or
                                  detector.is_adsorbate_intercalated())

                    all_energies.append({
                        "bulk_id": bulk_id,
                        "adsorbate": ads_name,
                        "E_adslab": E_adslab,
                        "E_slab": E_slab,
                        "anomaly": is_anomaly,
                    })
                except Exception as e:
                    print(f"  ERROR {ads_name}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # ══════ Save results ══════
    if all_energies:
        df = pd.DataFrame(all_energies)
        csv_path = os.path.join(args.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nRaw results → {csv_path}")

        summary = []
        for bulk_id in df["bulk_id"].unique():
            sub = df[df["bulk_id"] == bulk_id]
            row = {"bulk_id": bulk_id}
            valid = True
            for ads_name in ["OH", "O", "OOH"]:
                match = sub[(sub["adsorbate"] == ads_name) & (~sub["anomaly"])]
                if len(match) == 0:
                    match = sub[sub["adsorbate"] == ads_name]
                if len(match) == 0:
                    valid = False
                    break
                E_adslab = match.iloc[0]["E_adslab"]
                E_slab = match.iloc[0]["E_slab"]
                if E_slab is None or pd.isna(E_slab):
                    valid = False
                    break
                dE = E_adslab - E_slab
                row[f"dE_{ads_name}"] = round(dE, 3)

            if valid:
                summary.append(row)

        if summary:
            df_sum = pd.DataFrame(summary)
            sum_path = os.path.join(args.output_dir, "oer_summary.csv")
            df_sum.to_csv(sum_path, index=False)
            print(f"\nOER summary → {sum_path}")
            print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
