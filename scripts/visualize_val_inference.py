#!/usr/bin/env python3
"""
Script to run inference on the validation set and visualize Flow Matching results 
vs Ground Truth in 3D plots.

Usage:
    python scripts/visualize_val_inference.py --config-yml configs/denoising/your_config.yml --checkpoint /path/to/checkpoint.pt --output-dir viz_results --num-samples 10
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

from adsorbdiff.utils.flags import flags
from adsorbdiff.utils.utils import build_config, new_trainer_context
from adsorbdiff.relaxation.diffusers.flow_torch import FlowTorch
from torch_scatter import scatter
from ase import Atoms
from ase.io import write
from ase.data import covalent_radii

def save_ovito_xyz(sid, pos_slab, pos_ads_gt, pos_ads_pred, z_slab, z_ads, cell, output_path):
    """
    Saves a combined XYZ file for OVITO visualization.
    StructureType: 0=Slab, 1=GT Adsorbate, 2=Pred Adsorbate
    - Makes GT contiguous (minimum-image) so the molecule isn’t split across PBC.
    - Wraps prediction to the nearest image of GT in XY so it doesn’t drift outside the box.
    """
    # Fix GT continuity and wrap prediction near GT
    pos_ads_gt_contig = make_contiguous(pos_ads_gt, cell)
    pos_ads_pred_wrapped, _ = wrap_pred_to_gt(pos_ads_pred, pos_ads_gt_contig, cell)
    # Shift both GT and pred by the same lattice vectors so COM is inside primary XY cell
    pos_ads_gt_shifted, shift_int = center_ads_to_primary_cell(pos_ads_gt_contig, cell)
    if shift_int is not None:
        shift_vec = shift_int[0] * cell[0] + shift_int[1] * cell[1]
        pos_ads_pred_shifted = pos_ads_pred_wrapped - shift_vec
    else:
        pos_ads_pred_shifted = pos_ads_pred_wrapped

    # Create Atoms objects
    atoms_slab = Atoms(numbers=z_slab, positions=pos_slab, cell=cell, pbc=True)
    atoms_gt = Atoms(numbers=z_ads, positions=pos_ads_gt_shifted, cell=cell, pbc=True)
    
    # Use a dummy element (e.g., Lr - Lawrencium, Z=103) for predicted atoms
    # so they show up with a different color in OVITO by default.
    # We still preserve the original atomic numbers in the StructureType or could add a property.
    # But changing the element is the most direct way for visual distinction.
    z_pred_dummy = [103] * len(z_ads) 
    atoms_pred = Atoms(numbers=z_pred_dummy, positions=pos_ads_pred_shifted, cell=cell, pbc=True)
    
    # Combine
    combined = atoms_slab + atoms_gt + atoms_pred
    # Wrap everything into the primary cell for visualization
    try:
        combined.wrap(eps=0)
    except Exception:
        pass
    
    # Create StructureType array
    n_slab = len(atoms_slab)
    n_ads = len(atoms_gt)
    # Initialize with 0 (Slab)
    structure_types = np.zeros(len(combined), dtype=int)
    # Set GT (1)
    structure_types[n_slab:n_slab+n_ads] = 1
    # Set Pred (2)
    structure_types[n_slab+n_ads:] = 2
    
    combined.new_array("StructureType", structure_types)
    
    # Also save the true atomic numbers of prediction as a property
    true_z = np.concatenate([z_slab, z_ads, z_ads])
    combined.new_array("TrueAtomicNumber", true_z)

    # Add Radius property so OVITO shows correct sizes even when coloring by StructureType
    # covalent_radii is indexed by atomic number
    # Ensure z is integer for indexing
    radii = np.array([covalent_radii[int(z)] for z in true_z])
    combined.new_array("Radius", radii)
    
    write(output_path, combined, format="extxyz")

def make_contiguous(positions, cell):
    """
    Unwrap positions to be contiguous using Minimum Image Convention relative to the first atom.
    Fixes the issue where molecules appear broken across PBC boundaries.
    """
    if len(positions) < 2:
        return positions
    
    try:
        inv_cell = np.linalg.inv(cell)
    except:
        return positions

    ref = positions[0]
    new_pos = [ref]
    
    for i in range(1, len(positions)):
        p = positions[i]
        diff = p - ref
        # Fractional diff
        f = diff @ inv_cell
        # Wrap to [-0.5, 0.5]
        f = (f + 0.5) % 1.0 - 0.5
        # Cartesian diff
        d = f @ cell
        new_pos.append(ref + d)
        
    return np.array(new_pos)


def wrap_pred_to_gt(pos_ads_pred, pos_ads_gt, cell):
    """Wrap prediction to the nearest image of GT (XY only) and return wrapped pos + MAE."""
    if len(pos_ads_gt) == 0:
        return pos_ads_pred, None

    diff = pos_ads_pred - pos_ads_gt
    try:
        diff_frac = np.linalg.solve(cell.T, diff.T).T
        diff_frac[:, :2] = (diff_frac[:, :2] + 0.5) % 1.0 - 0.5
        diff_wrapped_cart = diff_frac @ cell
        diff_wrapped_cart[:, 2] = diff[:, 2]
        mae = float(np.mean(np.linalg.norm(diff_wrapped_cart, axis=1)))
        return pos_ads_gt + diff_wrapped_cart, mae
    except np.linalg.LinAlgError:
        return pos_ads_pred, None


def center_ads_to_primary_cell(pos_ads, cell):
    """Translate the whole adsorbate by lattice vectors so its COM falls into [0,1) XY fractional box.
    Returns shifted positions and the integer shift (nx, ny).
    """
    if len(pos_ads) == 0:
        return pos_ads, np.array([0, 0])
    try:
        inv_cell_T = np.linalg.inv(cell.T)
    except np.linalg.LinAlgError:
        return pos_ads, np.array([0, 0])
    com = pos_ads.mean(axis=0)
    frac = com @ inv_cell_T  # fractional coords
    shift_int = np.floor(frac[:2]).astype(int)  # move into [0,1)
    if np.all(shift_int == 0):
        return pos_ads, shift_int
    shift_vec = shift_int[0] * cell[0] + shift_int[1] * cell[1]
    pos_shifted = pos_ads - shift_vec
    return pos_shifted, shift_int


def save_step_xyz_sequence(sid, pos_slab, pos_ads_gt, step_preds, z_slab, z_ads, cell, output_path):
    """
    Save a multi-frame extxyz where GT stays fixed and only the predicted adsorbate moves per step.
    - GT is made contiguous under PBC.
    - Each predicted frame is wrapped to the nearest image of GT in XY to avoid far-away images.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    pos_ads_gt_contig = make_contiguous(pos_ads_gt, cell)
    pos_ads_gt_shifted, shift_int = center_ads_to_primary_cell(pos_ads_gt_contig, cell)

    for step_idx, pos_ads_pred in enumerate(step_preds):
        pos_ads_pred_wrapped, _ = wrap_pred_to_gt(pos_ads_pred, pos_ads_gt_contig, cell)
        if shift_int is not None:
            shift_vec = shift_int[0] * cell[0] + shift_int[1] * cell[1]
            pos_ads_pred_shifted = pos_ads_pred_wrapped - shift_vec
        else:
            pos_ads_pred_shifted = pos_ads_pred_wrapped
        atoms_slab = Atoms(numbers=z_slab, positions=pos_slab, cell=cell, pbc=True)
        atoms_gt = Atoms(numbers=z_ads, positions=pos_ads_gt_shifted, cell=cell, pbc=True)

        # Dummy element for predicted atoms so OVITO shows a distinct color
        z_pred_dummy = [103] * len(z_ads)
        atoms_pred = Atoms(numbers=z_pred_dummy, positions=pos_ads_pred_shifted, cell=cell, pbc=True)

        combined = atoms_slab + atoms_gt + atoms_pred

        try:
            combined.wrap(eps=0)
        except Exception:
            pass

        n_slab = len(atoms_slab)
        n_ads = len(atoms_gt)
        structure_types = np.zeros(len(combined), dtype=int)
        structure_types[n_slab:n_slab + n_ads] = 1
        structure_types[n_slab + n_ads:] = 2
        combined.new_array("StructureType", structure_types)

        true_z = np.concatenate([z_slab, z_ads, z_ads])
        combined.new_array("TrueAtomicNumber", true_z)
        radii = np.array([covalent_radii[int(z)] for z in true_z])
        combined.new_array("Radius", radii)
        combined.new_array("Step", np.full(len(combined), step_idx, dtype=int))

        write(output_path, combined, format="extxyz", append=True)

def plot_structure(ax, pos, tags, atomic_numbers, title, color_scheme=None, alpha=1.0, s=50):
    """
    Helper to scatter plot atoms.
    color_scheme: dict mapping 'slab', 'ads' to colors.
    """
    if color_scheme is None:
        color_scheme = {'slab': 'grey', 'ads': 'red'}
    
    # Slab
    slab_mask = (tags != 2)
    if slab_mask.any():
        ax.scatter(
            pos[slab_mask, 0], pos[slab_mask, 1], pos[slab_mask, 2],
            c=color_scheme['slab'], marker='o', s=s/2, alpha=0.3, label='Slab' if 'Slab' not in ax.get_legend_handles_labels()[1] else ""
        )
    
    # Adsorbate
    ads_mask = (tags == 2)
    if ads_mask.any():
        ax.scatter(
            pos[ads_mask, 0], pos[ads_mask, 1], pos[ads_mask, 2],
            c=color_scheme['ads'], marker='o', s=s, alpha=alpha, label=title
        )

def visualize_sample(sid, pos_slab, pos_ads_gt, pos_ads_pred, atomic_numbers, output_path, mae=None):
    """
    Draws slab (grey), GT adsorbate (green), Pred adsorbate (red).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Combine slab and ads for plotting
    # Slab
    ax.scatter(
        pos_slab[:, 0], pos_slab[:, 1], pos_slab[:, 2],
        c='lightgrey', s=20, alpha=0.2, label='Slab'
    )
    
    # GT Adsorbate
    ax.scatter(
        pos_ads_gt[:, 0], pos_ads_gt[:, 1], pos_ads_gt[:, 2],
        c='green', s=100, alpha=0.8, label='Ground Truth'
    )
    
    # Pred Adsorbate
    ax.scatter(
        pos_ads_pred[:, 0], pos_ads_pred[:, 1], pos_ads_pred[:, 2],
        c='red', s=100, alpha=0.8, label='Flow Prediction'
    )
    
    # Draw lines between GT and Pred for corresponding atoms (assuming order is preserved)
    if len(pos_ads_gt) == len(pos_ads_pred):
        for i in range(len(pos_ads_gt)):
            ax.plot(
                [pos_ads_gt[i, 0], pos_ads_pred[i, 0]],
                [pos_ads_gt[i, 1], pos_ads_pred[i, 1]],
                [pos_ads_gt[i, 2], pos_ads_pred[i, 2]],
                c='black', linestyle='--', alpha=0.3
            )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    title_text = f"Sample {sid}: GT (Green) vs Pred (Red)"
    if mae is not None:
        title_text += f"\nPos MAE: {mae:.4f} Å"
    ax.set_title(title_text)
    ax.legend()
    
    # Adjust view to see the adsorbate clearly
    # Center view on the adsorbate
    if len(pos_ads_gt) > 0:
        center = pos_ads_gt.mean(axis=0)
        ax.set_xlim(center[0] - 5, center[0] + 5)
        ax.set_ylim(center[1] - 5, center[1] + 5)
        ax.set_zlim(center[2] - 5, center[2] + 5)
    
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    # Parse args
    parser = flags.get_parser()
    parser.add_argument("--output-dir", default="viz_results", help="Directory to save images")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--steps", type=int, default=50, help="Number of flow steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--save-step-xyz", action="store_true", help="Dump per-step predicted adsorbate trajectory (GT fixed)")
    parser.add_argument("--store-every", type=int, default=1, help="Store every k steps when saving step xyz")
    parser.add_argument("--plot-step-mae", action="store_true", help="Compute per-step adsorbate pos MAE over the same samples and plot a curve")
    parser.add_argument("--integrator", default="heun", choices=["euler", "heun"], help="ODE integrator (heun=RK2, 2x model eval)")
    parser.add_argument("--time-grid", default="cosine", choices=["cosine", "uniform"], help="Inference time grid for t in [1,0]")
    
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    
    # Reduce batch size for visualization to avoid OOM
    # We only need a few samples, so processing huge batches is wasteful and memory intensive
    viz_batch_size = 8
    config["optim"]["batch_size"] = viz_batch_size
    config["optim"]["eval_batch_size"] = viz_batch_size

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup trainer context
    # Force mode to 'validate' to ensure validation loader is set up if available
    if not hasattr(args, "mode") or args.mode is None:
        args.mode = "validate"
        
    with new_trainer_context(args=args, config=config) as ctx:
        trainer = ctx.trainer

        # Explicitly load the provided checkpoint; new_trainer_context does not auto-load it.
        if args.checkpoint:
            if not os.path.isfile(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        else:
            raise ValueError("--checkpoint is required for inference/visualization")
        
        # Ensure val_loader exists
        if getattr(trainer, "val_loader", None) is None:
            print("Validation loader not found in trainer. Attempting to create it...")
            if "val_dataset" in config:
                trainer.val_loader = trainer.get_dataloader(
                    config["val_dataset"],
                    config["optim"]["batch_size"],
                    shuffle=False,
                    collater=trainer.collater
                )
            else:
                print("Warning: No 'val_dataset' in config. Trying 'test_dataset'...")
                if "test_dataset" in config:
                     trainer.val_loader = trainer.get_dataloader(
                        config["test_dataset"],
                        config["optim"]["batch_size"],
                        shuffle=False,
                        collater=trainer.collater
                    )
                else:
                    raise ValueError("No validation or test dataset found in config.")

        val_loader = trainer.val_loader
        device = trainer.device
        
        print(f"Loaded model from {args.checkpoint}")
        trainer.model.eval()
        if getattr(trainer, "ema", None):
            trainer.ema.store(); trainer.ema.copy_to()
        print(f"Running inference on validation set for {args.num_samples} samples...")

        # Optional: collect per-step MAE curves across samples
        step_ids_ref = None
        step_mae_rows = []  # list[np.ndarray]
        step_mae_sids = []  # list[str]
        
        mae_list = []
        count = 0
        # Use no_grad to prevent gradient graph construction during inference (saves massive memory)
        with torch.no_grad():
            for batch in tqdm(val_loader):
                if count >= args.num_samples:
                    break
                
                # Prepare batch
                if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
                    batch.pos = batch.pos_relaxed
                    print("Using pos_relaxed as Ground Truth reference.")
                else:
                    print("pos_relaxed not found; using batch.pos as reference.")
                
                # Clone for inference
                inf_batch = batch.clone()
            
                # Flow options
                flow_opt = {
                    "tr_sigma": trainer.tr_sigma,
                    "tr_sigma_z_scale": trainer.tr_sigma_z_scale,
                    "rot_sigma": trainer.rot_sigma,
                    "tr_clip": getattr(trainer, "tr_clip", None),
                    "rot_clip": getattr(trainer, "rot_clip", None),
                    "num_steps": args.steps,
                    "cfg_scale": args.cfg,
                    "time_sampl": getattr(trainer, "time_sampl", "uniform"),
                    "allow_z": getattr(trainer, "allow_z", False),
                    "store_steps": (args.save_step_xyz or args.plot_step_mae),
                    "store_every": args.store_every,
                    "integrator": args.integrator,
                    "time_grid": args.time_grid,
                }

                sampler = FlowTorch(
                    batch=inf_batch,
                    model=trainer.model,
                    flow_opt=flow_opt,
                    device=device,
                    save_full_traj=False
                )
                final_batch = sampler.run()
                step_positions = sampler.collected_steps
                step_ids = getattr(sampler, "collected_step_ids", None)
                
                # Extract data for visualization
                # We need to split the batch into individual samples
                num_graphs = batch.num_graphs
                batch_idx = batch.batch.cpu().numpy()
                
                if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
                    pos_gt_all = batch.pos_relaxed.cpu().numpy()
                else:
                    pos_gt_all = batch.pos.cpu().numpy()
                pos_pred_all = final_batch.pos.cpu().numpy()
                tags_all = batch.tags.cpu().numpy()
                atomic_numbers_all = batch.atomic_numbers.cpu().numpy()
                
                # Get cell info
                cell_all = batch.cell.cpu().numpy()
                
                for i in range(num_graphs):
                    if count >= args.num_samples:
                        break
                    
                    mask = (batch_idx == i)
                    if hasattr(batch, "sid"):
                        sid_val = batch.sid[i]
                        if hasattr(sid_val, "item"):
                            sid = str(sid_val.item())
                        else:
                            sid = str(sid_val)
                    else:
                        sid = f"sample_{count}"
                    
                    pos_gt = pos_gt_all[mask]
                    pos_pred = pos_pred_all[mask]
                    tags = tags_all[mask]
                    z = atomic_numbers_all[mask]
                    cell = cell_all[i]
                    
                    # Separate slab and adsorbate
                    slab_mask = (tags != 2)
                    ads_mask = (tags == 2)
                    
                    pos_slab = pos_gt[slab_mask]
                    pos_ads_gt = pos_gt[ads_mask]
                    z_slab = z[slab_mask]
                    z_ads = z[ads_mask]
                    
                    # Fix PBC breakage for GT adsorbate
                    # This ensures the molecule looks connected in visualization
                    pos_ads_gt = make_contiguous(pos_ads_gt, cell)
                    
                    pos_ads_pred = pos_pred[ads_mask]

                    # PBC wrap prediction to GT and compute MAE
                    pos_ads_pred, mae = wrap_pred_to_gt(pos_ads_pred, pos_ads_gt, cell)
                    if mae is not None:
                        mae_list.append(mae)
                        print(f"Sample {sid} MAE: {mae:.4f} (Running Avg: {np.mean(mae_list):.4f})")

                    # Fix PBC breakage for Pred adsorbate as well
                    pos_ads_pred = make_contiguous(pos_ads_pred, cell)

                    # Optional: dump per-step trajectory (GT fixed)
                    if args.save_step_xyz and step_positions:
                        step_preds = []
                        for step_pos in step_positions:
                            step_np = step_pos.numpy()[mask]
                            step_ads_pred = step_np[ads_mask]
                            step_ads_pred, _ = wrap_pred_to_gt(step_ads_pred, pos_ads_gt, cell)
                            step_ads_pred = make_contiguous(step_ads_pred, cell)
                            step_preds.append(step_ads_pred)
                        step_xyz_path = os.path.join(args.output_dir, f"{sid}_steps.xyz")
                        save_step_xyz_sequence(sid, pos_slab, pos_ads_gt, step_preds, z_slab=z_slab, z_ads=z_ads, cell=cell, output_path=step_xyz_path)

                    # Optional: compute per-step MAE curve (same samples)
                    if args.plot_step_mae and step_positions and step_ids:
                        if step_ids_ref is None:
                            step_ids_ref = list(step_ids)
                        if list(step_ids) != list(step_ids_ref):
                            print(f"[warn] step_ids mismatch for sample {sid}; skipping step-MAE for this sample")
                        else:
                            mae_steps = []
                            for step_pos in step_positions:
                                step_np = step_pos.numpy()[mask]
                                step_ads_pred = step_np[ads_mask]
                                _, step_mae = wrap_pred_to_gt(step_ads_pred, pos_ads_gt, cell)
                                mae_steps.append(np.nan if step_mae is None else float(step_mae))
                            step_mae_rows.append(np.array(mae_steps, dtype=float))
                            step_mae_sids.append(str(sid))

                    # Save XYZ for OVITO
                    xyz_path = os.path.join(args.output_dir, f"{sid}_viz.xyz")
                    save_ovito_xyz(sid, pos_slab, pos_ads_gt, pos_ads_pred, z_slab, z_ads, cell, xyz_path)

                    output_path = os.path.join(args.output_dir, f"{sid}_viz.png")
                    visualize_sample(sid, pos_slab, pos_ads_gt, pos_ads_pred, z, output_path, mae=mae)
                    
                    count += 1
                
        print(f"Visualization complete. Results saved to {args.output_dir}")

        if args.plot_step_mae and step_mae_rows and step_ids_ref is not None:
            step_mae_mat = np.stack(step_mae_rows, axis=0)  # (N, T)
            mean_curve = np.nanmean(step_mae_mat, axis=0)
            std_curve = np.nanstd(step_mae_mat, axis=0)
            x = np.array(step_ids_ref, dtype=int)

            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(111)
            ax.plot(x, mean_curve, color="tab:blue", lw=2, label="Mean step MAE")
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color="tab:blue", alpha=0.2, label="±1 std")
            ax.set_xlabel("Integrator step")
            ax.set_ylabel("Adsorbate pos MAE (Å)")
            ax.set_title("Per-step adsorbate MAE (same samples)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            out_png = os.path.join(args.output_dir, "step_mae_curve.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

            out_npz = os.path.join(args.output_dir, "step_mae_curve.npz")
            np.savez(
                out_npz,
                step_ids=x,
                mae_per_sample=step_mae_mat,
                mean=mean_curve,
                std=std_curve,
                sids=np.array(step_mae_sids, dtype=object),
            )
            print(f"Saved step-MAE curve: {out_png}")
            print(f"Saved step-MAE raw data: {out_npz}")

if __name__ == "__main__":
    main()
