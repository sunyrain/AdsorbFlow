#!/usr/bin/env python
"""
CO₂RR screening case study using AdsorbFlow.

For each catalyst surface, place and relax key CO₂RR intermediates (*CO, *H, *CHO, *COOH)
to compute selectivity descriptors and compare with experimental product distributions.

Key insight: ΔE_CO determines product selectivity (CO vs C₂ vs formate vs H₂).

Pipeline: bulk → slab(111) → flow placement → GemNet-OC relaxation → ΔE

Usage:
    python scripts/case_studies/run_co2rr_flow.py \
        --flow-ckpt checkpoints/{adsorbflow_checkpoint}.pt \
        --flow-config configs/flow/eqv2_fourier_cosine.yml \
        --relax-ckpt checkpoints/gemnet_oc_base_s2ef_2M.pt \
        --cfg-scale 7 --num-steps 5 \
        --output-dir examples/CO2RR/data_flow \
        --device cuda:0
"""

import argparse
import gc
import os
import pickle
import sys
import time
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ase.io
from ase.optimize import BFGS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adsorbdiff.placement import (
    Adsorbate,
    AdsorbateSlabConfig,
    Bulk,
    Slab,
    DetectTrajAnomaly,
)
from adsorbdiff.relaxation.calculator import AdsorbDiffCalculator
from adsorbdiff.relaxation.ml_relaxation import ml_flow
from adsorbdiff.datasets import data_list_collater
from adsorbdiff.utils.atoms_to_graphs import AtomsToGraphs
from adsorbdiff.utils.utils import load_config, setup_imports, setup_logging, update_config
from adsorbdiff.utils.registry import registry
from adsorbdiff.modules.scaling.util import ensure_fitted

# ── CO₂RR adsorbates ──
CO2RR_ADSORBATES = {
    "CO":   {"smiles": "*CO",   "relax_steps": 30, "interstitial_gap": 0.1},
    "H":    {"smiles": "*H",    "relax_steps": 20, "interstitial_gap": 0.1},
    "CHO":  {"smiles": "*CHO",  "relax_steps": 40, "interstitial_gap": 0.1},
    "COOH": {"smiles": "*COOH", "relax_steps": 50, "interstitial_gap": 0.2},
}

# Reference gas-phase energies — not used; GemNet has its own per-atom references
# E_CO2 = -22.96, E_H2O = -14.22, E_H2 = -6.77, E_CO_gas = -14.78


def load_flow_model(config_yml, checkpoint_path, device="cuda:0"):
    """Load flow matching model."""
    setup_imports()
    setup_logging()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    config["trainer"] = config.get("trainer", "ocp")
    if "model_attributes" in config:
        config["model_attributes"]["name"] = config.pop("model")
        config["model"] = config["model_attributes"]
    if "relax_dataset" in config.get("task", {}):
        del config["task"]["relax_dataset"]
    config["model"]["otf_graph"] = True
    config = update_config(config)
    config["checkpoint"] = checkpoint_path
    if "src" in config.get("dataset", {}):
        del config["dataset"]["src"]
    trainer = registry.get_trainer_class(config["trainer"])(
        task=config["task"], model=config["model"], dataset=[config["dataset"]],
        outputs=config["outputs"], loss_fns=config["loss_fns"],
        eval_metrics=config["eval_metrics"], optimizer=config["optim"],
        identifier="", slurm=config.get("slurm", {}),
        local_rank=config.get("local_rank", 0), is_debug=True, cpu=False,
        amp=config.get("amp", False),
    )
    trainer.load_checkpoint(checkpoint_path, checkpoint)
    trainer.set_seed(42)
    return trainer, config


def atoms_to_batch(atoms, a2g, device="cuda:0"):
    """Convert ASE Atoms → PyG Batch for FlowTorch."""
    data = a2g.convert(atoms)
    data.sid = ["co2rr_0"]
    data.tags = torch.tensor(atoms.get_tags(), dtype=torch.long)
    data.pos_relaxed = data.pos.clone()
    data.y = torch.tensor([0.0])
    data.y_relaxed = torch.tensor([0.0])
    data.force = torch.zeros_like(data.pos)
    data.fixed = torch.tensor([int(t < 2) for t in atoms.get_tags()], dtype=torch.long)
    data.energy = torch.tensor([0.0])
    batch = data_list_collater([data], otf_graph=True)
    return batch.to(device)


@torch.no_grad()
def run_flow_placement(trainer, config, atoms, a2g,
                       cfg_scale=7.0, num_steps=5, integrator="euler",
                       device="cuda:0",
                       traj_dir=None, traj_name="flow"):
    """Run flow matching placement."""
    ensure_fitted(trainer._unwrapped_model)
    trainer.model.eval()
    if trainer.ema:
        trainer.ema.store()
        trainer.ema.copy_to()
    batch = atoms_to_batch(atoms, a2g, device)
    flow_cfg = config.get("optim", {}).get("flow", {})
    flow_opt = {
        "tr_sigma": flow_cfg.get("tr_sigma", 3.0),
        "tr_sigma_z_scale": flow_cfg.get("tr_sigma_z_scale", 0.0),
        "rot_sigma": flow_cfg.get("rot_sigma", 1.0),
        "allow_z": flow_cfg.get("allow_z", False),
        "num_steps": num_steps, "cfg_scale": cfg_scale,
        "integrator": integrator, "time_grid": "cosine",
        "tr_clip": flow_cfg.get("tr_clip", 15),
        "rot_clip": flow_cfg.get("rot_clip", 5),
    }
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
        batch.sid = [traj_name]
    relaxed_batch = ml_flow(
        batch=batch, model=trainer, flow_opt=flow_opt,
        traj_dir=traj_dir, save_full_traj=True, device=device,
    )
    if trainer.ema:
        trainer.ema.restore()
    from adsorbdiff.relaxation.ase_utils import batch_to_atoms
    result_atoms = batch_to_atoms(relaxed_batch)
    return result_atoms[0] if len(result_atoms) == 1 else result_atoms


def relax_bare_slab(slab_atoms, calc, out_dir, fmax=0.03, max_steps=50):
    """Relax bare slab and return (energy, relaxed Atoms).

    The relaxed slab is used as ``final_slab_atoms`` for DetectTrajAnomaly so that
    legitimate slab relaxation is not mis-flagged as ``has_surface_changed``.
    """
    slab_copy = slab_atoms.copy()
    slab_copy.calc = calc
    traj_path = os.path.join(out_dir, "slab_opt.traj")
    opt = BFGS(slab_copy, trajectory=traj_path,
               logfile=os.path.join(out_dir, "slab_log.log"))
    opt.run(fmax=fmax, steps=max_steps)
    return slab_copy.get_potential_energy(), slab_copy


def main():
    parser = argparse.ArgumentParser(description="CO₂RR screening with AdsorbFlow")
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--flow-config", default=None)
    parser.add_argument("--relax-ckpt", required=True)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--integrator", choices=["euler", "heun"], default="euler",
                        help="ODE integrator for the flow sampler. heun is 2nd-order.")
    parser.add_argument("--output-dir", default="examples/CO2RR/data_flow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-bulks", type=int, default=None)
    parser.add_argument("--miller", type=int, nargs=3, default=[1, 1, 1])
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parents[2])
    db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/adsorbates.pkl")
    bulks_path = os.path.join(main_path, "examples/CO2RR/CO2RR_bulks.pkl")
    miller = tuple(args.miller)

    if not os.path.exists(bulks_path):
        print("ERROR: Run 'python scripts/case_studies/prepare_co2rr_data.py' first.")
        sys.exit(1)

    a2g = AtomsToGraphs(
        max_neigh=20, radius=12, r_energy=False, r_forces=False,
        r_distances=False, r_edges=False, r_pbc=True,
    )

    with open(bulks_path, "rb") as f:
        bulks = pickle.load(f)
    bulk_ids = [row["src_id"] for row in bulks]
    if args.max_bulks:
        bulk_ids = bulk_ids[:args.max_bulks]

    adsorbates = {}
    for ads_name, ads_info in CO2RR_ADSORBATES.items():
        adsorbates[ads_name] = Adsorbate(
            adsorbate_smiles_from_db=ads_info["smiles"],
            adsorbate_db_path=db_path,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"CO₂RR Screening: {len(bulk_ids)} catalysts × {len(CO2RR_ADSORBATES)} adsorbates")
    print(f"cfg_scale={args.cfg_scale}, K={args.num_steps}, miller={miller}")
    t_start = time.time()

    # ══════ Phase 1: Flow placement ══════
    print("\n=== Phase 1: Flow matching placement ===")
    trainer, config = load_flow_model(args.flow_config, args.flow_ckpt, args.device)

    placement_results = {}

    for i, bulk_id in enumerate(bulk_ids):
        print(f"\n[{i+1}/{len(bulk_ids)}] {bulk_id}...")
        try:
            bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulks_path)
            slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
            if not slab:
                print(f"  SKIP: no slab for miller={miller}")
                continue
            slab_atoms = slab[0]

            results_this = {"slab": slab_atoms}
            for ads_name, ads_info in CO2RR_ADSORBATES.items():
                ads_obj = adsorbates[ads_name]
                init_adslab = AdsorbateSlabConfig(
                    slab_atoms, ads_obj, mode="heuristic", num_sites=1,
                ).atoms_list[0]

                out_dir = os.path.join(args.output_dir, f"{bulk_id}_{ads_name}")
                os.makedirs(out_dir, exist_ok=True)

                with torch.cuda.amp.autocast():
                    diffused = run_flow_placement(
                        trainer, config, init_adslab, a2g,
                        cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                        integrator=args.integrator,
                        device=args.device, traj_dir=out_dir, traj_name=f"flow_{ads_name}",
                    )
                torch.cuda.empty_cache()

                site = diffused.get_positions()[diffused.get_tags() == 2]
                gap = ads_info["interstitial_gap"]
                adslab = AdsorbateSlabConfig(
                    slab_atoms, ads_obj, sites=site, interstitial_gap=gap,
                ).atoms_list[0]
                results_this[ads_name] = adslab
                print(f"  {ads_name}: placed ({len(adslab)} atoms)")

            placement_results[bulk_id] = results_this
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nPlaced {len(placement_results)}/{len(bulk_ids)} catalysts.")

    # ══════ Phase 2: MLFF relaxation ══════
    print("\n=== Phase 2: GemNet-OC relaxation ===")
    calc = AdsorbDiffCalculator(checkpoint_path=args.relax_ckpt, cpu=False)

    all_energies = []

    for i, bulk_id in enumerate(bulk_ids):
        if bulk_id not in placement_results:
            continue
        res = placement_results[bulk_id]
        print(f"\n[{i+1}/{len(bulk_ids)}] Relaxing {bulk_id}...")

        # Bare slab
        slab_dir = os.path.join(args.output_dir, f"{bulk_id}_slab")
        os.makedirs(slab_dir, exist_ok=True)
        try:
            slab_obj = res["slab"]
            slab_ase = slab_obj.atoms.copy() if hasattr(slab_obj, "atoms") else slab_obj.copy()
            E_slab, slab_relaxed = relax_bare_slab(slab_ase, calc, slab_dir)
            print(f"  slab E = {E_slab:.3f} eV")
        except Exception as e:
            print(f"  ERROR slab: {e}")
            E_slab = None
            slab_relaxed = None

        # Adsorbate-slab systems
        for ads_name, ads_info in CO2RR_ADSORBATES.items():
            if ads_name not in res:
                continue
            out_dir = os.path.join(args.output_dir, f"{bulk_id}_{ads_name}")
            try:
                adslab = res[ads_name]
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
                    "adsorbate": ads_name,
                    "E_adslab": E_adslab,
                    "E_slab": E_slab,
                    "anomaly": is_anomaly,
                })
            except Exception as e:
                print(f"  ERROR {ads_name}: {e}")
                traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # ══════ Compute descriptors & save ══════
    if all_energies:
        df = pd.DataFrame(all_energies)
        csv_path = os.path.join(args.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nRaw results → {csv_path}")

        # ML adsorption energy descriptors: ΔE_ml = E_ml(adslab) - E_ml(bare_slab)
        # GemNet-OC's per-atom references mean this gives adsorption energy
        # on an ML-specific scale. Trends correlate with DFT (like NRR study).

        summary = []
        for bulk_id in df["bulk_id"].unique():
            sub = df[df["bulk_id"] == bulk_id]
            row = {"bulk_id": bulk_id}
            for ads_name in CO2RR_ADSORBATES:
                match = sub[(sub["adsorbate"] == ads_name) & (~sub["anomaly"])]
                if len(match) > 0 and match.iloc[0]["E_slab"] is not None and not pd.isna(match.iloc[0]["E_slab"]):
                    dE = match.iloc[0]["E_adslab"] - match.iloc[0]["E_slab"]
                    row[f"dE_{ads_name}"] = round(dE, 3)
                else:
                    row[f"dE_{ads_name}"] = None

            summary.append(row)

        df_sum = pd.DataFrame(summary)
        sum_path = os.path.join(args.output_dir, "co2rr_summary.csv")
        df_sum.to_csv(sum_path, index=False)
        print(f"\nCO₂RR summary → {sum_path}")
        print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
