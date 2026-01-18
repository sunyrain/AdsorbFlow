
import pickle
import ase.io
import os
import numpy as np
import glob

# Paths
base_path = "/root/autodl-tmp/AdsorbFlow"
ref_path = os.path.join(base_path, "oc20_dense_mappings/oc20dense_ref_energies.pkl")
target_path = os.path.join(base_path, "oc20_dense_mappings/oc20dense_targets.pkl")
vasp_base_dir = "/root/autodl-tmp/AdsorbFlow/grid_search_runs/2025-12-17-19-22-40-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_epoch0180_unweightedvalloss1.4265_posmae0.6214/val_nonrelaxed_update/nsites_3/cfg3_steps10/vasp2"

# Load data
print("Loading reference energies...")
with open(ref_path, "rb") as f:
    ref_energies = pickle.load(f)

print("Loading targets...")
with open(target_path, "rb") as f:
    targets = pickle.load(f)
    if isinstance(next(iter(targets.values())), dict):
        converted = {}
        for sid, candidates in targets.items():
            energies = [val for val in candidates.values() if isinstance(val, (int, float))]
            if energies:
                converted[sid] = min(energies)
        targets = converted

# Find all OUTCARs
outcar_files = glob.glob(os.path.join(vasp_base_dir, "*", "OUTCAR"))
print(f"Found {len(outcar_files)} OUTCAR files.")

print("\n" + "="*80)
print(f"{'System ID':<20} | {'VASP E (eV)':<15} | {'Ref E (eV)':<15} | {'Pred Ads E':<15} | {'Target E':<15} | {'Diff (eV)':<15} | {'Result'}")
print("-" * 80)

success_count = 0
total_count = 0

for outcar_path in outcar_files:
    folder_name = os.path.basename(os.path.dirname(outcar_path))
    # Folder name format: sid_fid_siteidx (e.g., 45_7673_39_0)
    # But wait, write_vasp_inputs_nsite.py writes to {sid}_{fid}
    # Let's check the folder name structure.
    # The script used: outdir=f"{TRAJ_INPUT_PATH}/vasp/{sid}_{fid}"
    # So folder name is sid_fid.
    # But wait, sid itself can contain underscores (e.g. 45_7673).
    # Let's parse carefully.
    
    # Assuming the last part is fid, and the rest is sid.
    # Example: 45_7673_39_0 -> sid=45_7673_39, fid=0? No.
    # The script logic:
    # if traj_path.split("/")[-1].count("_") == 3: sid, fid = ...
    # Let's just try to match the sid with keys in ref_energies.
    
    # Heuristic: try to find the longest prefix that exists in ref_energies
    parts = folder_name.split("_")
    sid = None
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in ref_energies:
            sid = candidate
            break
    
    if sid is None:
        print(f"{folder_name:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | Unknown SID")
        continue

    try:
        # Read VASP energy
        atoms = ase.io.read(outcar_path)
        vasp_energy = atoms.get_potential_energy()
        
        ref_e = ref_energies[sid]
        pred_ads_energy = vasp_energy - ref_e
        
        if sid not in targets:
            target_e = np.nan
            diff = np.nan
            result = "No Target"
        else:
            target_e = targets[sid]
            diff = pred_ads_energy - target_e
            if diff <= 0.1:
                result = "SUCCESS"
                success_count += 1
            else:
                result = "FAILURE"
        
        total_count += 1
        print(f"{folder_name:<20} | {vasp_energy:<15.6f} | {ref_e:<15.6f} | {pred_ads_energy:<15.6f} | {target_e:<15.6f} | {diff:<15.6f} | {result}")
        
    except Exception as e:
        print(f"{folder_name:<20} | Error: {e}")

print("-" * 80)
print(f"Total: {total_count}, Success: {success_count}, Rate: {success_count/total_count*100:.2f}%" if total_count > 0 else "No valid results.")
