#!/usr/bin/env python
"""
NRR case study using AdsorbFlow (Flow Matching + GemNet-OC MLFF relaxation).
Adapted from examples/NRR/NRR_example-gemnet.ipynb.

Usage:
    python scripts/case_studies/run_nrr_flow.py \
        --flow-ckpt checkpoints/{adsorbflow_checkpoint}.pt \
        --flow-config configs/flow/eqv2_conditional_flow.yml \
        --relax-ckpt checkpoints/gemnet_oc_base_s2ef_2M.pt \
        --cfg-scale 7 --num-steps 5 \
        --output-dir examples/NRR/data_flow
"""

import argparse
import os
import sys
import pickle
import time
import logging
from glob import glob
from pathlib import Path

import numpy as np
import torch
import ase.io
from ase.optimize import BFGS

# AdsorbDiff imports
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
from adsorbdiff.utils.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from adsorbdiff.utils.registry import registry
from adsorbdiff.modules.scaling.util import ensure_fitted


def load_flow_model(config_yml, checkpoint_path, device="cuda:0"):
    """Load flow matching model as a trainer object (needed by FlowTorch)."""
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
        task=config["task"],
        model=config["model"],
        dataset=[config["dataset"]],
        outputs=config["outputs"],
        loss_fns=config["loss_fns"],
        eval_metrics=config["eval_metrics"],
        optimizer=config["optim"],
        identifier="",
        slurm=config.get("slurm", {}),
        local_rank=config.get("local_rank", 0),
        is_debug=True,
        cpu=False,
        amp=config.get("amp", False),
    )

    trainer.load_checkpoint(checkpoint_path, checkpoint)
    trainer.set_seed(42)

    return trainer, config


def atoms_to_batch(atoms, a2g, device="cuda:0"):
    """Convert ASE Atoms (with tags) to a PyG Batch for FlowTorch."""
    data = a2g.convert(atoms)
    data.sid = ["nrr_0"]
    data.tags = torch.tensor(atoms.get_tags(), dtype=torch.long)
    # Flow matching needs pos_relaxed as reference geometry
    data.pos_relaxed = data.pos.clone()
    data.y = torch.tensor([0.0])
    data.y_relaxed = torch.tensor([0.0])
    data.force = torch.zeros_like(data.pos)
    data.fixed = torch.tensor([int(t < 2) for t in atoms.get_tags()], dtype=torch.long)
    # energy conditioning: 0.0 = request lowest energy
    data.energy = torch.tensor([0.0])
    batch = data_list_collater([data], otf_graph=True)
    return batch.to(device)


@torch.no_grad()
def run_flow_placement(trainer, config, atoms, a2g,
                       cfg_scale=7.0, num_steps=5, device="cuda:0",
                       traj_dir=None, traj_name="flow"):
    """Run flow matching to predict optimal adsorbate placement."""
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
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "integrator": "euler",
        "time_grid": "cosine",
        "tr_clip": flow_cfg.get("tr_clip", 15),
        "rot_clip": flow_cfg.get("rot_clip", 5),
    }

    _traj_dir = traj_dir
    if _traj_dir:
        os.makedirs(_traj_dir, exist_ok=True)
        batch.sid = [traj_name]

    relaxed_batch = ml_flow(
        batch=batch,
        model=trainer,
        flow_opt=flow_opt,
        traj_dir=_traj_dir,
        save_full_traj=True,
        device=device,
    )

    if trainer.ema:
        trainer.ema.restore()

    # Convert back to ASE Atoms
    from adsorbdiff.relaxation.ase_utils import batch_to_atoms
    result_atoms = batch_to_atoms(relaxed_batch)
    return result_atoms[0] if len(result_atoms) == 1 else result_atoms


def main():
    parser = argparse.ArgumentParser(description="NRR case study with AdsorbFlow")
    parser.add_argument("--flow-ckpt", required=True, help="Path to flow matching checkpoint")
    parser.add_argument("--flow-config", default=None, help="Path to flow config yml (optional, loaded from ckpt)")
    parser.add_argument("--relax-ckpt", required=True, help="Path to GemNet-OC checkpoint for MLFF relaxation")
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--num-steps", type=int, default=5, help="Flow sampling steps (K)")
    parser.add_argument("--output-dir", default="examples/NRR/data_flow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-bulks", type=int, default=None, help="Limit number of bulks to process")
    args = parser.parse_args()

    main_path = str(Path(__file__).resolve().parents[2])
    db_path = os.path.join(main_path, "adsorbdiff/placement/pkls/adsorbates.pkl")
    bulks_path = os.path.join(main_path, "examples/NRR/NRR_example_bulks.pkl")

    # AtomsToGraphs converter
    a2g = AtomsToGraphs(
        max_neigh=20,
        radius=12,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=False,
        r_pbc=True,
    )

    # Load bulks
    with open(bulks_path, "rb") as f:
        bulks = pickle.load(f)
    bulk_ids = [row['src_id'] for row in bulks]
    if args.max_bulks:
        bulk_ids = bulk_ids[:args.max_bulks]

    adsorbate_H = Adsorbate(adsorbate_smiles_from_db="*H", adsorbate_db_path=db_path)
    adsorbate_NNH = Adsorbate(adsorbate_smiles_from_db="*N*NH", adsorbate_db_path=db_path)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(bulk_ids)} bulks with cfg_scale={args.cfg_scale}, K={args.num_steps}")
    tinit = time.time()

    # ======= Phase 1: Flow matching placement =======
    print("\n=== Phase 1: Flow matching placement ===")
    print("Loading flow matching model...")
    trainer, config = load_flow_model(args.flow_config, args.flow_ckpt, args.device)

    placement_results = {}  # bulk_src_id -> {H: atoms, NNH: atoms}

    for i, bulk_src_id in enumerate(bulk_ids):
        print(f"\n[{i+1}/{len(bulk_ids)}] Flow placement for {bulk_src_id}...")
        try:
            bulk = Bulk(bulk_src_id_from_db=bulk_src_id, bulk_db_path=bulks_path)
            slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=(1, 1, 1))

            random_adslabs_H = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="heuristic", num_sites=1).atoms_list[0]
            random_adslabs_NNH = AdsorbateSlabConfig(slab[0], adsorbate_NNH, mode="heuristic", num_sites=1).atoms_list[0]

            out_dir_H = os.path.join(args.output_dir, f"{bulk_src_id}_H")
            out_dir_NNH = os.path.join(args.output_dir, f"{bulk_src_id}_NNH")
            os.makedirs(out_dir_H, exist_ok=True)
            os.makedirs(out_dir_NNH, exist_ok=True)

            # Flow placement H
            with torch.cuda.amp.autocast():
                diffused_H = run_flow_placement(
                    trainer, config, random_adslabs_H, a2g,
                    cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                    device=args.device,
                    traj_dir=out_dir_H, traj_name="flow_H",
                )
            torch.cuda.empty_cache()

            # Flow placement NNH
            with torch.cuda.amp.autocast():
                diffused_NNH = run_flow_placement(
                    trainer, config, random_adslabs_NNH, a2g,
                    cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                    device=args.device,
                    traj_dir=out_dir_NNH, traj_name="flow_NNH",
                )
            torch.cuda.empty_cache()

            # Extract predicted sites
            site_H = diffused_H.get_positions()[diffused_H.get_tags() == 2]
            site_NNH = diffused_NNH.get_positions()[diffused_NNH.get_tags() == 2]

            # Recreate adslab with predicted sites
            adslab_H = AdsorbateSlabConfig(slab[0], adsorbate_H, sites=site_H).atoms_list[0]
            adslab_NNH = AdsorbateSlabConfig(slab[0], adsorbate_NNH, sites=site_NNH, interstitial_gap=0.2).atoms_list[0]

            placement_results[bulk_src_id] = {"H": adslab_H, "NNH": adslab_NNH, "slab": slab[0]}
            print(f"  H placement done ({len(adslab_H)} atoms), NNH done ({len(adslab_NNH)} atoms)")

        except Exception as e:
            print(f"  ERROR in flow placement: {e}")
            import traceback
            traceback.print_exc()

    # Free flow model memory
    del trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"\nFlow model unloaded. Placed {len(placement_results)}/{len(bulk_ids)} systems.")

    # ======= Phase 2: MLFF relaxation =======
    print("\n=== Phase 2: MLFF relaxation (GemNet-OC) ===")
    print("Loading GemNet-OC...")
    calc_opt = AdsorbDiffCalculator(checkpoint_path=args.relax_ckpt, cpu=False)

    for i, bulk_src_id in enumerate(bulk_ids):
        if bulk_src_id not in placement_results:
            continue
        print(f"\n[{i+1}/{len(bulk_ids)}] Relaxing {bulk_src_id}...")
        try:
            res = placement_results[bulk_src_id]
            out_dir_H = os.path.join(args.output_dir, f"{bulk_src_id}_H")
            out_dir_NNH = os.path.join(args.output_dir, f"{bulk_src_id}_NNH")

            # H relaxation
            adslab_H = res["H"]
            adslab_H.calc = calc_opt
            opt = BFGS(adslab_H, trajectory=os.path.join(out_dir_H, "opt.traj"),
                       logfile=os.path.join(out_dir_H, "log.log"))
            opt.run(fmax=0.05, steps=20)
            print(f"  H: E = {adslab_H.get_potential_energy():.3f} eV")

            # NNH relaxation
            adslab_NNH = res["NNH"]
            adslab_NNH.calc = calc_opt
            opt = BFGS(adslab_NNH, trajectory=os.path.join(out_dir_NNH, "opt.traj"),
                       logfile=os.path.join(out_dir_NNH, "log.log"))
            opt.run(fmax=0.05, steps=50)
            print(f"  NNH: E = {adslab_NNH.get_potential_energy():.3f} eV")

        except Exception as e:
            print(f"  ERROR in relaxation: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - tinit
    print(f"\nTotal elapsed: {elapsed:.1f} seconds")

    # --- Parse results ---
    print("\n=== Parsing results ===")
    min_E = []
    total_anomalies = 0
    for file_outer in sorted(glob(os.path.join(args.output_dir, "*"))):
        if not os.path.isdir(file_outer):
            continue
        basename = os.path.basename(file_outer)
        parts = basename.rsplit("_", 1)
        if len(parts) != 2:
            continue
        bulk_id, ads = parts[0], parts[1]

        opt_traj = os.path.join(file_outer, "opt.traj")
        if not os.path.exists(opt_traj):
            continue

        traj = ase.io.read(opt_traj, ":")
        detector = DetectTrajAnomaly(traj[0], traj[-1], traj[0].get_tags())
        anom = (
            detector.is_adsorbate_dissociated()
            or detector.is_adsorbate_desorbed()
            or detector.has_surface_changed()
            or detector.is_adsorbate_intercalated()
        )
        if anom:
            print(f"  Anomaly: {basename}")
            total_anomalies += 1
        else:
            rx_energy = traj[-1].get_potential_energy()
            min_E.append({"adsorbate": ads, "bulk_id": bulk_id, "min_E_ml": rx_energy})

    print(f"\nTotal anomalies: {total_anomalies}")
    print(f"Valid results: {len(min_E)}")

    if min_E:
        import pandas as pd
        df = pd.DataFrame(min_E)
        df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
        print(f"\nResults saved to {args.output_dir}/results.csv")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
