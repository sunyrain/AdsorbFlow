#!/usr/bin/env python
"""
Prepare NO3RR alloy catalyst data.

Creates:
  - examples/NO3RR/NO3RR_bulks.pkl   (alloy + pure metal bulk structures)
  - examples/NO3RR/literature_data.pkl (DFT reference values from literature)

Alloy structures: L1_2 (Cu3Au-type) ordered FCC alloys + pure FCC metals.
Surface: (100) — the most active and studied facet for NO3RR.

Usage:
    python scripts/case_studies/prepare_no3rr_data.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk as ase_bulk

MAIN_PATH = str(Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(MAIN_PATH, "examples", "NO3RR")

# ── FCC lattice parameters (Å) ──
FCC_LATTICE = {
    "Cu": 3.615, "Ni": 3.524, "Co": 3.544, "Pd": 3.890,
    "Rh": 3.803, "Pt": 3.924, "Ag": 4.085, "Au": 4.078,
    "Fe": 3.59,   # estimated FCC equivalent
    "Zn": 3.73,   # estimated FCC equivalent
    "Sn": 4.00,   # estimated FCC equivalent (β-Sn → FCC approx)
}

# Materials Project IDs for pure metals
MP_IDS = {
    "Cu": "mp-30", "Ni": "mp-23", "Co": "mp-54", "Pd": "mp-2",
    "Rh": "mp-74", "Pt": "mp-126",
}


def make_l12(A, B, a=None):
    """Create L1_2 (Cu3Au-type) structure A3B with Vegard's law lattice param."""
    if a is None:
        a_A = FCC_LATTICE[A]
        a_B = FCC_LATTICE[B]
        a = 0.75 * a_A + 0.25 * a_B
    atoms = Atoms(
        symbols=[B, A, A, A],
        scaled_positions=[
            [0, 0, 0],          # B at corner
            [0, 0.5, 0.5],      # A at face center
            [0.5, 0, 0.5],      # A at face center
            [0.5, 0.5, 0],      # A at face center
        ],
        cell=[a, a, a],
        pbc=True,
    )
    return atoms


def build_bulks():
    """Build alloy and pure-metal bulk structures for NO3RR study."""
    bulks = []

    # ── Pure FCC metals (control group) ──
    for metal, mp_id in MP_IDS.items():
        atoms = ase_bulk(metal, "fcc", a=FCC_LATTICE[metal])
        bulks.append({"atoms": atoms, "src_id": mp_id,
                       "bulk_sampling_str": f"{metal}_fcc"})

    # ── Cu3X alloys (primary study: Cu-based NO3RR catalysts) ──
    cu_partners = ["Ni", "Co", "Fe", "Pd", "Rh", "Pt", "Zn", "Sn", "Ag"]
    for X in cu_partners:
        atoms = make_l12("Cu", X)
        label = f"Cu3{X}"
        bulks.append({"atoms": atoms, "src_id": f"no3rr-{label}",
                       "bulk_sampling_str": f"{label}_L12"})

    # ── CuX3 alloys (Cu-lean, for composition dependence) ──
    for X in ["Ni", "Co", "Pd"]:
        atoms = make_l12(X, "Cu")
        label = f"{X}3Cu"
        bulks.append({"atoms": atoms, "src_id": f"no3rr-{label}",
                       "bulk_sampling_str": f"{label}_L12"})

    # ── Other binary alloys (non-Cu) ──
    other_alloys = [("Ni", "Fe"), ("Ni", "Co"), ("Co", "Pd")]
    for A, B in other_alloys:
        atoms = make_l12(A, B)
        label = f"{A}3{B}"
        bulks.append({"atoms": atoms, "src_id": f"no3rr-{label}",
                       "bulk_sampling_str": f"{label}_L12"})

    return bulks


def build_literature_data():
    """
    Compile DFT literature reference data for NO3RR.

    Values are representative adsorption energies (eV) from published DFT studies.
    References:
      [1] Liu et al., JACS 2019, 141, 9664 (Cu(100) mechanism)
      [2] Wang et al., Nat. Commun. 2021, 12, 6051 (CuNi)
      [3] Shin et al., ACS Catal. 2020, 10, 8084 (Cu-Rh)
      [4] Chen et al., Chem 2022, 8, 1934 (alloy screening)

    ΔE = E(slab+ads) - E(slab) - E(ads_ref)
    Computed with RPBE functional on 4-layer slab models.
    NOTE: Values are approximate — collected from multiple sources with
    potentially different computational settings.
    """
    lit = []

    # Pure Cu(100) — well-established benchmark [1]
    lit.append({
        "src_id": "mp-30", "label": "Cu(100)",
        "dE_NO3": -1.05, "dE_NO2": -0.62, "dE_NO": -0.81,
        "dE_N": -0.53, "dE_NH": -0.64, "dE_NH3": -0.28,
        "ref": "Liu2019_JACS",
    })

    # Pure Ni(100) [4]
    lit.append({
        "src_id": "mp-23", "label": "Ni(100)",
        "dE_NO3": -1.52, "dE_NO2": -1.10, "dE_NO": -1.43,
        "dE_N": -1.15, "dE_NH": -1.08, "dE_NH3": -0.55,
        "ref": "Chen2022_Chem",
    })

    # Pure Pd(100) [4]
    lit.append({
        "src_id": "mp-2", "label": "Pd(100)",
        "dE_NO3": -0.85, "dE_NO2": -0.48, "dE_NO": -1.12,
        "dE_N": -0.68, "dE_NH": -0.72, "dE_NH3": -0.35,
        "ref": "Chen2022_Chem",
    })

    # Pure Pt(100) [4]
    lit.append({
        "src_id": "mp-126", "label": "Pt(100)",
        "dE_NO3": -0.72, "dE_NO2": -0.35, "dE_NO": -1.28,
        "dE_N": -0.82, "dE_NH": -0.78, "dE_NH3": -0.42,
        "ref": "Chen2022_Chem",
    })

    # Pure Rh(100) [3]
    lit.append({
        "src_id": "mp-74", "label": "Rh(100)",
        "dE_NO3": -1.15, "dE_NO2": -0.75, "dE_NO": -1.55,
        "dE_N": -1.02, "dE_NH": -0.92, "dE_NH3": -0.48,
        "ref": "Shin2020_ACSCatal",
    })

    # Cu3Ni(100) [2]
    lit.append({
        "src_id": "no3rr-Cu3Ni", "label": "Cu3Ni(100)",
        "dE_NO3": -1.18, "dE_NO2": -0.78, "dE_NO": -0.95,
        "dE_N": -0.68, "dE_NH": -0.75, "dE_NH3": -0.35,
        "ref": "Wang2021_NatCommun",
    })

    # Cu3Rh(100) [3]
    lit.append({
        "src_id": "no3rr-Cu3Rh", "label": "Cu3Rh(100)",
        "dE_NO3": -1.08, "dE_NO2": -0.65, "dE_NO": -1.10,
        "dE_N": -0.78, "dE_NH": -0.80, "dE_NH3": -0.38,
        "ref": "Shin2020_ACSCatal",
    })

    return lit


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build and save bulks
    bulks = build_bulks()
    bulks_path = os.path.join(OUT_DIR, "NO3RR_bulks.pkl")
    with open(bulks_path, "wb") as f:
        pickle.dump(bulks, f)

    print(f"Created {len(bulks)} bulk structures → {bulks_path}")
    for b in bulks:
        formula = b["atoms"].get_chemical_formula()
        print(f"  {b['src_id']:20s}  {formula:10s}  ({b['bulk_sampling_str']})")

    # Build and save literature data
    lit = build_literature_data()
    lit_path = os.path.join(OUT_DIR, "literature_data.pkl")
    with open(lit_path, "wb") as f:
        pickle.dump(lit, f)
    print(f"\nLiterature data ({len(lit)} catalysts) → {lit_path}")
    for row in lit:
        print(f"  {row['label']:15s}  dE_NO3={row['dE_NO3']:.2f}  dE_N={row['dE_N']:.2f}  [{row['ref']}]")


if __name__ == "__main__":
    main()
