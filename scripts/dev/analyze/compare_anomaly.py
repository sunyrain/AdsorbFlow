"""
对比两种 _make_ads_contiguous (inv(cell) vs inv(cell.T)) 在真实 traj 上的异常检测差异
"""
import os, sys, glob, numpy as np, ase.io
sys.path.insert(0, ".")
from adsorbdiff.placement.flag_anomaly import DetectTrajAnomaly

BASE = "grid_search_runs"
BASE_OOD = "grid_search_runs_ood"

configs = {
    "EqV2-valID": os.path.join(BASE,
        "2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
        "_epoch0180_unweightedvalloss1.0316_posmae0.9085",
        "val_nonrelaxed_update/nsites_10/cfg7_steps5"),
    "PaiNN-valID": os.path.join(BASE,
        "2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint",
        "val_nonrelaxed_update/nsites_10/cfg5_steps5"),
    "EqV2-OOD": os.path.join(BASE_OOD,
        "2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2"
        "_epoch0180_unweightedvalloss1.0316_posmae0.9085",
        "valood50_R1I0.1/nsites_10/cfg7_steps5"),
    "PaiNN-OOD": os.path.join(BASE_OOD,
        "2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint",
        "valood50_R1I0.1/nsites_10/cfg5_steps5"),
}

def make_contiguous_correct(atoms):
    tags = atoms.get_tags()
    ads_idx = np.where(tags == 2)[0]
    if ads_idx.size < 2:
        return atoms
    cell = atoms.get_cell()
    try:
        inv_cell = np.linalg.inv(cell)
    except:
        return atoms
    pos = atoms.get_positions().copy()
    ref = pos[ads_idx[0]]
    diffs = pos[ads_idx] - ref
    frac = diffs @ inv_cell
    frac = (frac + 0.5) % 1.0 - 0.5
    pos[ads_idx] = ref + frac @ np.array(cell)
    new = atoms.copy()
    new.set_positions(pos)
    return new

def make_contiguous_eval(atoms):
    tags = atoms.get_tags()
    ads_idx = np.where(tags == 2)[0]
    if ads_idx.size == 0:
        return atoms
    cell = atoms.get_cell()
    try:
        inv_cell = np.linalg.inv(np.array(cell).T)
    except:
        return atoms
    pos = atoms.get_positions()
    ref = pos[ads_idx[0]]
    diffs = pos[ads_idx] - ref
    frac = diffs @ inv_cell
    frac = (frac + 0.5) % 1.0 - 0.5
    pos[ads_idx] = ref + frac @ np.array(cell)
    new = atoms.copy()
    new.set_positions(pos)
    return new

def is_anomalous_with(traj, make_fn):
    try:
        initial = make_fn(traj[0])
        final = make_fn(traj[-1])
        tags = initial.get_tags()
        detector = DetectTrajAnomaly(initial, final, tags)
        return any([
            detector.is_adsorbate_dissociated(),
            detector.is_adsorbate_desorbed(),
            detector.has_surface_changed(),
            detector.is_adsorbate_intercalated(),
        ])
    except:
        return True

for label, base_dir in configs.items():
    disagree = 0
    total = 0
    anom_correct = 0
    anom_eval = 0
    for seed in range(10):
        relax_dir = os.path.join(base_dir, str(seed), "relaxations")
        trajs = sorted(glob.glob(os.path.join(relax_dir, "*.traj")))
        for f in trajs:
            try:
                traj = ase.io.read(f, ":")
            except:
                continue
            total += 1
            a_correct = is_anomalous_with(traj, make_contiguous_correct)
            a_eval = is_anomalous_with(traj, make_contiguous_eval)
            if a_correct:
                anom_correct += 1
            if a_eval:
                anom_eval += 1
            if a_correct != a_eval:
                disagree += 1
    print(f"RESULT {label}: total={total} anom_inv_cell={anom_correct} anom_inv_cellT={anom_eval} disagree={disagree}")
