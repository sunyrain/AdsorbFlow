
import ase.io
import os
import pickle
import glob
import numpy as np

# Paths
base_path = "/root/autodl-tmp/AdsorbDiff"
target_path = os.path.join(base_path, "oc20_dense_mappings/oc20dense_targets.pkl")
vasp_base_dir = "/root/autodl-tmp/AdsorbDiff/grid_search_runs/pt_z1_epoch0021_valloss3.4507/val_nonrelaxed_update/nsites_10/cfg3_steps30/vasp"
ref_base_dir = "/root/autodl-tmp/AdsorbDiff/grid_search_runs/pt_z1_epoch0021_valloss3.4507/val_nonrelaxed_update/nsites_10/cfg3_steps30/vasp_refs"

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

print("\n" + "="*120)
print(f"{'System ID':<20} | {'E_sys':<12} | {'E_slab':<12} | {'E_gas':<12} | {'Pred E_ads':<12} | {'Target':<12} | {'Diff':<12} | {'Result'}")
print("-" * 120)

# Only check systems that have complete references
ref_systems = []
if os.path.exists(ref_base_dir):
    for sid in os.listdir(ref_base_dir):
        slab_out = os.path.join(ref_base_dir, sid, "slab", "OUTCAR")
        gas_out = os.path.join(ref_base_dir, sid, "gas", "OUTCAR")
        if os.path.exists(slab_out) and os.path.exists(gas_out):
            # Check if they are complete (simple check for now)
            if os.path.getsize(slab_out) > 1000 and os.path.getsize(gas_out) > 1000:
                ref_systems.append(sid)

print(f"Found {len(ref_systems)} systems with complete references.")

for sid in ref_systems:
    # Find corresponding system OUTCAR
    # System folder is sid_fid (e.g. 48_1815_16_0)
    # We need to find a folder that starts with sid
    sys_outcar = None
    sys_folder_name = None
    
    candidates = glob.glob(os.path.join(vasp_base_dir, f"{sid}*"))
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "OUTCAR")):
            sys_outcar = os.path.join(cand, "OUTCAR")
            sys_folder_name = os.path.basename(cand)
            break
    
    if not sys_outcar:
        print(f"{sid:<20} | No System OUTCAR found")
        continue

    try:
        # Read E_sys
        sys_atoms = ase.io.read(sys_outcar)
        e_sys = sys_atoms.get_potential_energy()
        
        # Read E_slab
        slab_outcar = os.path.join(ref_base_dir, sid, "slab", "OUTCAR")
        slab_atoms = ase.io.read(slab_outcar)
        e_slab = slab_atoms.get_potential_energy()
        
        # Read E_gas
        gas_outcar = os.path.join(ref_base_dir, sid, "gas", "OUTCAR")
        gas_atoms = ase.io.read(gas_outcar)
        e_gas = gas_atoms.get_potential_energy()
        
        # Calculate Ads E
        pred_ads_e = e_sys - e_slab - e_gas
        
        # Get Target
        if sid in targets:
            target_e = targets[sid]
            diff = pred_ads_e - target_e
            if abs(diff) <= 0.1:
                result = "SUCCESS"
                
            else:
                result = "FAILURE"
        else:
            target_e = np.nan
            diff = np.nan
            result = "No Target"
            
        print(f"{sys_folder_name:<20} | {e_sys:<12.4f} | {e_slab:<12.4f} | {e_gas:<12.4f} | {pred_ads_e:<12.4f} | {target_e:<12.4f} | {diff:<12.4f} | {result}")

    except Exception as e:
        print(f"{sid:<20} | Error: {e}")

print("-" * 120)
