#!/usr/bin/env python
"""
NO3RR heuristic baseline (ablation): skip Flow model entirely.

Pipeline: alloy bulk → slab(100) → heuristic placement → GemNet-OC relaxation → ΔE

Usage:
    python scripts/run_no3rr_heuristic.py \
        --relax-ckpt configs/relaxation/gemnet_oc/gemnet_oc_base_s2ef_2M.pt \
        --output-dir examples/NO3RR/data_heuristic \
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

NO3RR_ADSORBATES = {
    "NO3": {"smiles": "*NO3", "relax_steps": 50, "interstitial_gap": 0.2},
    "NO2": {"smiles": "*NO2", "relax_steps": 40, "interstitial_gap": 0.2},
    "NO":  {"smiles": "*NO",  "relax_steps": 30, "interstitial_gap": 0.1},
    "N":   {"smiles": "*N",   "relax_steps": 30, "interstitial_gap": 0.1},
    "NH":  {"smiles": "*NH",  "relax_steps": 30, "interstitial_gap": 0.1},
    "NH3": {"smiles": "*NH3", "relax_steps": 40, "interstitial_gap": 0.2},
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
    parser = argparse.ArgumentParser(description="NO3RR heuristic baseline")
    parser.add_argument("--relax-ckpt", required=True)
    parser.add_argument("--output-dir", default="examples/NO3RR/data_heuristic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-bulks", type=int, default=None)
    parser.add_argument("--miller", type=int, nargs=3, default=[1, 0, 0])
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parent.parent.parent)
    db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/adsorbates.pkl")
    bulks_path = os.path.join(main_path, "examples/NO3RR/NO3RR_bulks.pkl")
    miller = tuple(args.miller)

    if not os.path.exists(bulks_path):
        print("ERROR: Run 'python scripts/prepare_no3rr_data.py' first.")
        sys.exit(1)

    with open(bulks_path, "rb") as f:
        bulks = pickle.load(f)
    bulk_entries = bulks[:args.max_bulks] if args.max_bulks else bulks

    adsorbates = {}
    for ads_name, ads_info in NO3RR_ADSORBATES.items():
        adsorbates[ads_name] = Adsorbate(
            adsorbate_smiles_from_db=ads_info["smiles"],
            adsorbate_db_path=db_path,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"NO3RR Heuristic: {len(bulk_entries)} catalysts × {len(NO3RR_ADSORBATES)} adsorbates")
    print(f"miller={miller}, NO Flow model")
    t_start = time.time()

    calc = AdsorbDiffCalculator(checkpoint_path=args.relax_ckpt, cpu=False)

    all_energies = []

    for i, entry in enumerate(bulk_entries):
        bulk_id = entry["src_id"]
        formula = entry["atoms"].get_chemical_formula()
        print(f"\n[{i+1}/{len(bulk_entries)}] {bulk_id} ({formula})...")
        try:
            bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulks_path)
            slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
            if not slabs:
                print(f"  SKIP: no slab for miller={miller}")
                continue
            slab_atoms = slabs[0]

            # Bare slab
            slab_dir = os.path.join(args.output_dir, f"{bulk_id}_slab")
            os.makedirs(slab_dir, exist_ok=True)
            slab_ase = slab_atoms.atoms.copy() if hasattr(slab_atoms, "atoms") else slab_atoms.copy()
            E_slab, slab_relaxed = relax_bare_slab(slab_ase, calc, slab_dir)
            print(f"  slab E = {E_slab:.3f} eV")

            # Each adsorbate
            for ads_name, ads_info in NO3RR_ADSORBATES.items():
                ads_obj = adsorbates[ads_name]
                out_dir = os.path.join(args.output_dir, f"{bulk_id}_{ads_name}")
                os.makedirs(out_dir, exist_ok=True)

                try:
                    adslab = AdsorbateSlabConfig(
                        slab_atoms, ads_obj, mode="heuristic", num_sites=1,
                    ).atoms_list[0]

                    ase.io.write(os.path.join(out_dir, "init.xyz"), adslab)

                    adslab.calc = calc
                    opt = BFGS(adslab, trajectory=os.path.join(out_dir, "opt.traj"),
                               logfile=os.path.join(out_dir, "log.log"))
                    opt.run(fmax=0.05, steps=ads_info["relax_steps"])

                    E_adslab = adslab.get_potential_energy()
                    print(f"  {ads_name}: E = {E_adslab:.3f} eV")

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
                        "formula": formula,
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

    if all_energies:
        df = pd.DataFrame(all_energies)
        csv_path = os.path.join(args.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nRaw results → {csv_path}")

        summary = []
        for bulk_id in df["bulk_id"].unique():
            sub = df[df["bulk_id"] == bulk_id]
            row = {"bulk_id": bulk_id, "formula": sub.iloc[0]["formula"]}
            for ads_name in NO3RR_ADSORBATES:
                match = sub[(sub["adsorbate"] == ads_name) & (~sub["anomaly"])]
                if len(match) == 0:
                    match = sub[sub["adsorbate"] == ads_name]
                if len(match) == 0:
                    row[f"dE_{ads_name}"] = None
                    continue
                E_adslab = match.iloc[0]["E_adslab"]
                E_slab = match.iloc[0]["E_slab"]
                if E_slab is None or pd.isna(E_slab):
                    row[f"dE_{ads_name}"] = None
                    continue
                row[f"dE_{ads_name}"] = round(E_adslab - E_slab, 3)
                row[f"anom_{ads_name}"] = bool(match.iloc[0]["anomaly"])

            summary.append(row)

        df_sum = pd.DataFrame(summary)
        sum_path = os.path.join(args.output_dir, "no3rr_summary.csv")
        df_sum.to_csv(sum_path, index=False)
        print(f"\nNO3RR summary → {sum_path}")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
