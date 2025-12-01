import os
os.environ["VASP_PP_PATH"] = "/root/autodl-tmp/potpaw_PBE_54"

import numpy as np
import ase.io
from tqdm import tqdm
import lmdb, time, copy, shutil, glob, random, sys, datetime, pickle
sys.path.append("/root/autodl-tmp/AdsorbDiff/Open-Catalyst-Dataset")
from ocdata.utils.vasp import write_vasp_input_files
from adsorbdiff.placement import DetectTrajAnomaly
import os

# Link to the directory with all simulations for an adslab system
# [Auto-filled] 指向你 grid search 中效果较好的一个配置 (例如 cfg1_steps50)
TRAJ_INPUT_PATH = "/root/autodl-tmp/AdsorbDiff/grid_search_runs/pt_z1_epoch0021_valloss3.4507/val_nonrelaxed_update/nsites_10/cfg1_steps30"
EXPORT_PATH = "/root/autodl-tmp/vasp_cluster_inputs"

# Add link to the tags.pkl file
tag_path = "/root/autodl-tmp/AdsorbDiff/oc20_dense_mappings/oc20dense_tags.pkl"

VASP_FLAGS = {
    "ibrion": 2, # Static calculation (no relaxation)
    "nsw": 0,     # 0 ionic steps
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": True,
    "lwave": False,
    "ncore": 4,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
    "setups": "minimal",
}

with open(os.path.join(tag_path), "rb") as h:
    tags_map = pickle.load(h)

# Modified glob pattern to match the directory structure: .../site_id/relaxations/*.traj
traj_paths = glob.glob(
    f"{TRAJ_INPUT_PATH}/*/relaxations/*.traj"
)


def anomalous_structure(traj, sid):
    initial_atoms = traj[0]
    final_atoms = traj[-1]
    atom_tags = tags_map[sid]
    detector = DetectTrajAnomaly(initial_atoms, final_atoms, atom_tags)
    anom = np.array(
        [
            detector.is_adsorbate_dissociated(),
            detector.is_adsorbate_desorbed(),
            detector.has_surface_changed(),
            detector.is_adsorbate_intercalated(),
        ]
    )
    return anom

uniques_sids = {}
for traj_path in tqdm(traj_paths):
    #traj = ase.io.read(traj_path, ":")
    
    if traj_path.split("/")[-1].count("_") == 3:
        sid, fid = traj_path.split("/")[-1].split(".")[0].rsplit("_", 1)
    elif traj_path.split("/")[-1].count("_") == 2:

        sid = traj_path.split("/")[-1].split(".")[0]
        fid = 0
    
    if sid in uniques_sids:
        continue
    else:
        uniques_sids[sid] = 1

    files_per_sid = glob.glob(f"{TRAJ_INPUT_PATH}/*/relaxations/{sid}*.traj")

    # get the minimum energy structure
    energies = np.array(
        list(map(lambda x: ase.io.read(x).get_potential_energy(), files_per_sid))
    ).flatten()
    sorted_energy_idx = np.argsort(energies)
    count = 0
    while count < len(sorted_energy_idx):
        traj = ase.io.read(files_per_sid[int(sorted_energy_idx[0])], ":")
        if anomalous_structure(traj, sid).any():
            sorted_energy_idx = sorted_energy_idx[1:]
        else:
            break
     
    if count == len(sorted_energy_idx):
        print("All structures are anomalous for ", sid)
        continue

    relaxed_struct = traj[-1]

    # set constraints based on tags
    tags = tags_map[sid]
    fixed_atoms = np.where(tags == 2)[0]
    relaxed_struct.set_constraint(ase.constraints.FixAtoms(fixed_atoms))
    
    # 1. Export to original location
    os.makedirs(f"{TRAJ_INPUT_PATH}/vasp", exist_ok=True)
    write_vasp_input_files(
        relaxed_struct,
        outdir=f"{TRAJ_INPUT_PATH}/vasp/{sid}_{fid}",
        vasp_flags=VASP_FLAGS,
    )

    # 2. Export to independent folder for cluster
    export_dir = os.path.join(EXPORT_PATH, f"{sid}_{fid}")
    os.makedirs(export_dir, exist_ok=True)
    
    write_vasp_input_files(
        relaxed_struct,
        outdir=export_dir,
        vasp_flags=VASP_FLAGS,
    )
    print(f"Generated inputs for {sid} in {export_dir} and original path")
