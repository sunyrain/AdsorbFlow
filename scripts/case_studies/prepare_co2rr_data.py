#!/usr/bin/env python
"""
Prepare CO₂RR screening study data: bulk pickle + literature reference energies.

Reference data from:
  - Peterson & Nørskov, J. Phys. Chem. Lett. 3, 251 (2012)  — CO₂RR on metals
  - Kuhl et al., Energy Environ. Sci. 5, 7050 (2012) — experimental product distribution
  - Bagger et al., ChemPhysChem 18, 3266 (2017) — selectivity descriptors

Key descriptor: ΔE_CO (CO binding energy) determines selectivity:
  - Strong CO binding (< -0.5 eV): CO stays, further reduction → CH₄, C₂H₄ (Cu-like)
  - Weak CO binding (> -0.3 eV): CO desorbs → CO is main product (Au, Ag, Zn)
  - Too strong binding: catalyst poisoning (Pt, Pd, Ni, Fe)
  - H binding: determines HER vs CO₂RR selectivity

Adsorbates needed:
  - *CO  (key selectivity descriptor)
  - *H   (HER competing reaction)
  - *CHO (rate-determining for CH₄ pathway)
  - *COOH (first intermediate from CO₂)
  - *OCHO (formate pathway)

Usage:
    python scripts/prepare_co2rr_data.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ─── Literature reference data ───
# Adsorption energies ΔE in eV (DFT, from Peterson & Nørskov 2012, Bagger 2017)
# Experimental selectivity from Kuhl et al. 2012
CO2RR_LITERATURE = {
    # Pure metals — key baselines
    "Cu":  {"dE_CO": -0.50, "dE_H": -0.20, "dE_CHO": 0.74,  "dE_COOH": 0.43,
            "main_product": "CH4/C2H4", "FE_CO2RR": 72.3},
    "Au":  {"dE_CO":  0.25, "dE_H":  0.13, "dE_CHO": None,   "dE_COOH": 0.52,
            "main_product": "CO",       "FE_CO2RR": 87.1},
    "Ag":  {"dE_CO":  0.21, "dE_H":  0.10, "dE_CHO": None,   "dE_COOH": 0.65,
            "main_product": "CO",       "FE_CO2RR": 81.5},
    "Zn":  {"dE_CO":  0.39, "dE_H": -0.04, "dE_CHO": None,   "dE_COOH": 0.88,
            "main_product": "CO",       "FE_CO2RR": 79.4},
    "Sn":  {"dE_CO":  0.35, "dE_H":  0.25, "dE_CHO": None,   "dE_COOH": 0.23,
            "main_product": "formate",  "FE_CO2RR": 88.4},
    "In":  {"dE_CO":  0.42, "dE_H":  0.32, "dE_CHO": None,   "dE_COOH": 0.19,
            "main_product": "formate",  "FE_CO2RR": 85.0},
    "Bi":  {"dE_CO":  0.48, "dE_H":  0.38, "dE_CHO": None,   "dE_COOH": 0.15,
            "main_product": "formate",  "FE_CO2RR": 91.2},
    "Pb":  {"dE_CO":  0.50, "dE_H":  0.30, "dE_CHO": None,   "dE_COOH": 0.20,
            "main_product": "formate",  "FE_CO2RR": 72.0},
    "Pt":  {"dE_CO": -1.13, "dE_H": -0.36, "dE_CHO": -0.21,  "dE_COOH": 0.11,
            "main_product": "H2",       "FE_CO2RR": 0.1},
    "Pd":  {"dE_CO": -0.80, "dE_H": -0.37, "dE_CHO": 0.05,   "dE_COOH": 0.20,
            "main_product": "CO/H2",    "FE_CO2RR": 28.3},
    "Ni":  {"dE_CO": -1.10, "dE_H": -0.25, "dE_CHO": -0.15,  "dE_COOH": 0.19,
            "main_product": "H2",       "FE_CO2RR": 1.2},
    "Fe":  {"dE_CO": -0.91, "dE_H": -0.26, "dE_CHO": -0.08,  "dE_COOH": 0.25,
            "main_product": "H2",       "FE_CO2RR": 0.0},
    "Co":  {"dE_CO": -1.09, "dE_H": -0.28, "dE_CHO": -0.11,  "dE_COOH": 0.21,
            "main_product": "H2",       "FE_CO2RR": 0.0},
}

# Cu-X bimetallic alloys for screening (key CO2RR catalyst design space)
# dE_CO from DFT calculations in literature (various sources)
CO2RR_BIMETALLICS = {
    "CuAu":  {"src_id": "mp-1184018", "dE_CO_est": -0.15, "category": "CO"},
    "CuAg":  {"src_id": "mp-1184011", "dE_CO_est": -0.30, "category": "CO/C2"},
    "CuZn":  {"src_id": "mp-1215401", "dE_CO_est": -0.35, "category": "C2"},
    "CuSn":  {"src_id": "mp-10598",   "dE_CO_est": -0.05, "category": "formate/CO"},
    "CuPd":  {"src_id": "mp-1018029", "dE_CO_est": -0.65, "category": "CH4"},
    "CuPt":  {"src_id": "mp-12086",   "dE_CO_est": -0.75, "category": "H2"},
    "CuIn":  {"src_id": "mp-21985",   "dE_CO_est":  0.05, "category": "CO/formate"},
    "CuGa":  {"src_id": "mp-1183995", "dE_CO_est": -0.25, "category": "C2"},
    "CuAl":  {"src_id": "mp-1008555", "dE_CO_est": -0.20, "category": "C2"},
    "CuNi":  None,  # Not in bulks.pkl as CuNi, skip
}

# mp-id mapping for pure metals
METAL_MPID = {
    "Cu": "mp-30",    "Au": "mp-81",   "Ag": "mp-124",  "Zn": "mp-79",
    "Sn": "mp-117",   "In": "mp-85",   "Bi": "mp-23152", "Pb": "mp-20483",
    "Pt": "mp-126",   "Pd": "mp-2",    "Ni": "mp-23",   "Fe": "mp-13",
    "Co": "mp-54",
}


def main():
    main_path = str(Path(__file__).resolve().parent.parent)
    bulk_db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/bulks.pkl")
    out_dir = os.path.join(main_path, "examples/CO2RR")
    os.makedirs(out_dir, exist_ok=True)

    # Load bulk database
    with open(bulk_db_path, "rb") as f:
        all_bulks = pickle.load(f)
    src_id_to_entry = {entry["src_id"]: entry for entry in all_bulks}

    # ──── Pure metals ────
    co2rr_bulks = []
    print("=== Pure metals ===")
    for metal, mpid in METAL_MPID.items():
        if mpid in src_id_to_entry:
            co2rr_bulks.append(src_id_to_entry[mpid])
            print(f"  ✓ {metal:3s} ({mpid}): {src_id_to_entry[mpid]['atoms'].get_chemical_formula()}")
        else:
            print(f"  ✗ {metal:3s} ({mpid}): NOT FOUND")

    # ──── Cu-X bimetallics ────
    print("\n=== Cu-X bimetallics ===")
    for name, info in CO2RR_BIMETALLICS.items():
        if info is None:
            continue
        src_id = info["src_id"]
        if src_id in src_id_to_entry:
            co2rr_bulks.append(src_id_to_entry[src_id])
            print(f"  ✓ {name:8s} ({src_id}): {src_id_to_entry[src_id]['atoms'].get_chemical_formula()}")
        else:
            print(f"  ✗ {name:8s} ({src_id}): NOT FOUND")

    bulks_out = os.path.join(out_dir, "CO2RR_bulks.pkl")
    with open(bulks_out, "wb") as f:
        pickle.dump(co2rr_bulks, f)
    print(f"\nSaved {len(co2rr_bulks)} bulks → {bulks_out}")

    # ──── Literature data ────
    lit_data = []

    # Pure metals
    for metal, data in CO2RR_LITERATURE.items():
        mpid = METAL_MPID.get(metal)
        if mpid and mpid in src_id_to_entry:
            lit_data.append({
                "bulk_id": mpid,
                "name": metal,
                "type": "pure",
                "dE_CO":  data["dE_CO"],
                "dE_H":   data["dE_H"],
                "dE_CHO": data.get("dE_CHO"),
                "dE_COOH": data.get("dE_COOH"),
                "main_product": data["main_product"],
                "FE_CO2RR": data.get("FE_CO2RR"),
            })

    # Bimetallics
    for name, info in CO2RR_BIMETALLICS.items():
        if info is None:
            continue
        src_id = info["src_id"]
        if src_id in src_id_to_entry:
            lit_data.append({
                "bulk_id": src_id,
                "name": name,
                "type": "bimetallic",
                "dE_CO":  info["dE_CO_est"],
                "dE_H":   None,
                "dE_CHO": None,
                "dE_COOH": None,
                "main_product": info["category"],
                "FE_CO2RR": None,
            })

    lit_out = os.path.join(out_dir, "literature_data.pkl")
    with open(lit_out, "wb") as f:
        pickle.dump(lit_data, f)
    print(f"Saved {len(lit_data)} literature entries → {lit_out}")

    # Print summary
    print("\n=== CO₂RR Literature Reference ===")
    print(f"{'Name':>8} {'Type':>10} {'ΔE_CO':>7} {'ΔE_H':>7} {'Product':>12} {'FE%':>6}")
    for row in lit_data:
        dE_CO = f"{row['dE_CO']:.2f}" if row['dE_CO'] is not None else "  —"
        dE_H  = f"{row['dE_H']:.2f}"  if row['dE_H']  is not None else "  —"
        fe    = f"{row['FE_CO2RR']:.1f}" if row['FE_CO2RR'] is not None else "  —"
        print(f"{row['name']:>8} {row['type']:>10} {dE_CO:>7} {dE_H:>7} {row['main_product']:>12} {fe:>6}")


if __name__ == "__main__":
    main()
