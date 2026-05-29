#!/usr/bin/env python
"""
Prepare OER volcano study data: bulk pickle + literature reference energies.

Reference data from:
  - Man et al., ChemCatChem 3, 1159 (2011)  — Universal OER scaling relations
  - Nørskov et al., J. Phys. Chem. B 108, 17886 (2004) — OER/ORR volcano
  - Rossmeisl et al., J. Electroanal. Chem. 607, 83 (2007)

The OER volcano uses ΔG_OH as descriptor; overpotential η is computed from
  ΔG1 = ΔG_OH
  ΔG2 = ΔG_O  − ΔG_OH
  ΔG3 = ΔG_OOH − ΔG_O
  ΔG4 = 4.92 − ΔG_OOH
  η = max(ΔG1, ΔG2, ΔG3, ΔG4)/e − 1.23 V

Adsorbate free-energy corrections (standard, from Nørskov group):
  ΔG_OH  = ΔE_OH  + 0.35 eV
  ΔG_O   = ΔE_O   + 0.05 eV
  ΔG_OOH = ΔE_OOH + 0.40 eV

Usage:
    python scripts/case_studies/prepare_oer_data.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ─── Literature reference data (DFT values from Man et al. 2011 + Rossmeisl 2007) ───
# Format: {metal: {ΔG_OH, ΔG_O, ΔG_OOH}} in eV (free energies vs H2O/H2 reference)
# These are widely-cited benchmark values for (111) surfaces.
OER_LITERATURE = {
    "Pt": {"dG_OH": 0.80, "dG_O": 1.60, "dG_OOH": 3.56, "eta_exp": 0.52},
    "Pd": {"dG_OH": 0.72, "dG_O": 1.47, "dG_OOH": 3.49, "eta_exp": 0.56},
    "Ir": {"dG_OH": 0.57, "dG_O": 1.21, "dG_OOH": 3.35, "eta_exp": 0.37},
    "Ru": {"dG_OH": 0.43, "dG_O": 1.14, "dG_OOH": 3.21, "eta_exp": 0.37},
    "Rh": {"dG_OH": 0.55, "dG_O": 1.36, "dG_OOH": 3.34, "eta_exp": 0.43},
    "Au": {"dG_OH": 1.63, "dG_O": 3.28, "dG_OOH": 4.42, "eta_exp": 0.78},
    "Ag": {"dG_OH": 1.59, "dG_O": 3.20, "dG_OOH": 4.38, "eta_exp": 0.74},
    "Cu": {"dG_OH": 0.53, "dG_O": 1.29, "dG_OOH": 3.32, "eta_exp": 0.55},
    "Ni": {"dG_OH": 0.25, "dG_O": 0.82, "dG_OOH": 3.03, "eta_exp": 0.57},
    "Co": {"dG_OH": 0.16, "dG_O": 0.97, "dG_OOH": 2.94, "eta_exp": 0.48},
    "Fe": {"dG_OH": 0.05, "dG_O": 0.83, "dG_OOH": 2.83, "eta_exp": 0.58},
    "Mn": {"dG_OH": 0.00, "dG_O": 0.73, "dG_OOH": 2.78, "eta_exp": 0.62},
    "Ti": {"dG_OH": -0.31, "dG_O": 0.38, "dG_OOH": 2.47, "eta_exp": 0.80},
    "Cr": {"dG_OH": -0.05, "dG_O": 0.68, "dG_OOH": 2.73, "eta_exp": 0.67},
    "Os": {"dG_OH": 0.30, "dG_O": 1.22, "dG_OOH": 3.08, "eta_exp": None},
    "Re": {"dG_OH": -0.04, "dG_O": 0.75, "dG_OOH": 2.74, "eta_exp": None},
    "W":  {"dG_OH": -0.20, "dG_O": 0.48, "dG_OOH": 2.58, "eta_exp": None},
    "V":  {"dG_OH": -0.25, "dG_O": 0.45, "dG_OOH": 2.53, "eta_exp": None},
    "Mo": {"dG_OH": -0.11, "dG_O": 0.60, "dG_OOH": 2.67, "eta_exp": None},
    "Nb": {"dG_OH": -0.35, "dG_O": 0.30, "dG_OOH": 2.43, "eta_exp": None},
    "Ta": {"dG_OH": -0.40, "dG_O": 0.25, "dG_OOH": 2.38, "eta_exp": None},
}

# mp-id mapping for pure metals (from OC20 bulk database)
METAL_MPID = {
    "Pt": "mp-126", "Pd": "mp-2",    "Ir": "mp-101",  "Ru": "mp-33",
    "Rh": "mp-74",  "Au": "mp-81",   "Ag": "mp-124",  "Cu": "mp-30",
    "Ni": "mp-23",  "Co": "mp-54",   "Fe": "mp-13",   "Mn": "mp-35",
    "Ti": "mp-72",  "Cr": "mp-90",   "Os": "mp-49",   "Re": "mp-8",
    "W":  "mp-91",  "V":  "mp-146",  "Mo": "mp-129",  "Nb": "mp-75",
    "Ta": "mp-50",
}

# Free-energy corrections: ΔG = ΔE_ads + correction
# Standard values from Nørskov group
FE_CORRECTIONS = {
    "OH":  0.35,   # ZPE + TS, eV
    "O":   0.05,   # ZPE + TS, eV
    "OOH": 0.40,   # ZPE + TS, eV
}


def compute_overpotential(dG_OH, dG_O, dG_OOH):
    """Compute OER overpotential from Gibbs free energies."""
    dG1 = dG_OH
    dG2 = dG_O - dG_OH
    dG3 = dG_OOH - dG_O
    dG4 = 4.92 - dG_OOH
    return max(dG1, dG2, dG3, dG4) - 1.23


def main():
    main_path = str(Path(__file__).resolve().parents[2])
    bulk_db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/bulks.pkl")
    out_dir = os.path.join(main_path, "examples/OER")
    os.makedirs(out_dir, exist_ok=True)

    # Load full bulk database
    with open(bulk_db_path, "rb") as f:
        all_bulks = pickle.load(f)
    src_id_to_entry = {entry["src_id"]: entry for entry in all_bulks}

    # Build OER bulks pickle
    oer_bulks = []
    for metal, mpid in METAL_MPID.items():
        if mpid in src_id_to_entry:
            oer_bulks.append(src_id_to_entry[mpid])
            print(f"  ✓ {metal:3s} ({mpid}): {src_id_to_entry[mpid]['atoms'].get_chemical_formula()}")
        else:
            print(f"  ✗ {metal:3s} ({mpid}): NOT FOUND in bulk DB")

    bulks_out = os.path.join(out_dir, "OER_bulks.pkl")
    with open(bulks_out, "wb") as f:
        pickle.dump(oer_bulks, f)
    print(f"\nSaved {len(oer_bulks)} bulks → {bulks_out}")

    # Build literature data pickle
    lit_data = []
    for metal, data in OER_LITERATURE.items():
        mpid = METAL_MPID.get(metal)
        if mpid and mpid in src_id_to_entry:
            eta = compute_overpotential(data["dG_OH"], data["dG_O"], data["dG_OOH"])
            lit_data.append({
                "bulk_id": mpid,
                "metal": metal,
                "dG_OH":  data["dG_OH"],
                "dG_O":   data["dG_O"],
                "dG_OOH": data["dG_OOH"],
                "eta_oer": round(eta, 3),
                "eta_exp": data.get("eta_exp"),
            })

    lit_out = os.path.join(out_dir, "literature_data.pkl")
    with open(lit_out, "wb") as f:
        pickle.dump(lit_data, f)
    print(f"Saved {len(lit_data)} literature entries → {lit_out}")

    # Print summary table
    print("\n=== OER Literature Reference (sorted by ΔG_OH) ===")
    print(f"{'Metal':>5} {'ΔG_OH':>7} {'ΔG_O':>7} {'ΔG_OOH':>7} {'η_OER':>7}")
    for row in sorted(lit_data, key=lambda x: x["dG_OH"]):
        print(f"{row['metal']:>5} {row['dG_OH']:>7.2f} {row['dG_O']:>7.2f} "
              f"{row['dG_OOH']:>7.2f} {row['eta_oer']:>7.3f}")


if __name__ == "__main__":
    main()
