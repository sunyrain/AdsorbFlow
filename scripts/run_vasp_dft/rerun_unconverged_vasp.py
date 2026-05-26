"""
重跑 EqV2 ep187 cfg7_steps5 中未收敛的 VASP SP 计算。

原因：原脚本 1800s 超时导致 VASP 被 kill，SCF 未收敛。
修改：
  - 超时 7200s（2h）
  - INCAR 增加 NELM=300（默认60不够）
  - 只针对 15 个未收敛的 case
"""

import os
import subprocess
import signal
import re
from datetime import datetime

BASE = "grid_search_runs/2026-01-13-23-21-36-z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr2.0-4_para_eqv2_epoch0187_unweightedvalloss1.2734_posmae0.8954/val_nonrelaxed_update/nsites_10/cfg7_steps5"

VASP_CMD = "vasp_std"
CORES_PER_JOB = 8
TIMEOUT = 7200  # 2 hours

# 13 unconverged cases (excluded 65_2771_212: 219 atoms Cr144, too large to converge)
TARGETS = [
    # (1, "65_2771_212_0"),  # SKIP: 219 atoms, SCF oscillates, 2h timeout not enough
    # (5, "65_2771_212_0"),  # SKIP: same structure
    (1, "10_133_5_0"),
    (5, "10_133_5_0"),
    (10, "10_133_5_0"),
    (1, "12_1990_4_0"),
    (2, "12_1990_4_0"),
    (1, "12_8736_30_0"),
    (5, "12_8736_30_0"),
    (1, "57_7_1_0"),
    (1, "18_3771_63_0"),
    (2, "18_3771_63_0"),
    (1, "40_5239_7_0"),
    (2, "40_5239_7_0"),
    (5, "40_5239_7_0"),
]

FILES_TO_CLEANUP = [
    "WAVECAR", "CHG", "CHGCAR", "DOSCAR", "EIGENVAL",
    "IBZKPT", "PCDAT", "XDATCAR", "PROCAR", "LOCPOT",
]


def patch_incar(incar_path, nelm=300):
    """Add/update NELM in INCAR."""
    with open(incar_path) as f:
        lines = f.readlines()

    has_nelm = any("NELM" in l for l in lines)
    if has_nelm:
        lines = [re.sub(r'NELM\s*=\s*\d+', f'NELM = {nelm}', l) for l in lines]
    else:
        lines.append(f" NELM = {nelm}\n")

    with open(incar_path, "w") as f:
        f.writelines(lines)
    print(f"    INCAR patched: NELM={nelm}")


def check_converged(outcar_path):
    if not os.path.isfile(outcar_path):
        return False
    with open(outcar_path) as f:
        txt = f.read()
    return ("reached required accuracy" in txt) or ("General timing" in txt and "TOTEN" in txt)


def run_one(script_dir, idx, total):
    sid = os.path.basename(script_dir)
    level = os.path.basename(os.path.dirname(script_dir))
    label = f"[{idx+1}/{total}] {level}/{sid}"

    if check_converged(os.path.join(script_dir, "OUTCAR")):
        print(f"{label}: already converged, skipping")
        return True

    # Clean old outputs
    for fn in ["OUTCAR", "vasp.out", "OSZICAR"] + FILES_TO_CLEANUP:
        fp = os.path.join(script_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)

    # Patch INCAR
    patch_incar(os.path.join(script_dir, "INCAR"))

    # Clean HDF5 lock files
    for fn in os.listdir(script_dir):
        if fn.endswith(".h5") or fn.endswith(".lock"):
            os.remove(os.path.join(script_dir, fn))

    # Run VASP (disable HDF5 file locking to avoid VASP 6.3 issue)
    cmd = f"cd {script_dir} && export HDF5_USE_FILE_LOCKING=FALSE && ulimit -s unlimited && mpirun -np {CORES_PER_JOB} {VASP_CMD} > vasp.out 2>&1"
    print(f"{label}: running VASP (timeout={TIMEOUT}s)...")
    start = datetime.now()

    try:
        proc = subprocess.Popen(["bash", "-c", cmd], preexec_fn=os.setpgrp)
        proc.wait(timeout=TIMEOUT)
        elapsed = (datetime.now() - start).total_seconds()
        success = check_converged(os.path.join(script_dir, "OUTCAR"))

        # Cleanup large files
        for fn in FILES_TO_CLEANUP:
            fp = os.path.join(script_dir, fn)
            if os.path.isfile(fp):
                os.remove(fp)

        if success:
            # Extract final energy
            with open(os.path.join(script_dir, "OUTCAR")) as f:
                energies = re.findall(r"TOTEN\s*=\s*([-\d.]+)", f.read())
            final_e = energies[-1] if energies else "?"
            print(f"{label}: CONVERGED in {elapsed:.0f}s, E={final_e} eV")
        else:
            print(f"{label}: FAILED (not converged after {elapsed:.0f}s)")
        return success

    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
        print(f"{label}: TIMEOUT ({TIMEOUT}s)")
        return False
    except Exception as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except Exception:
            pass
        print(f"{label}: ERROR: {e}")
        return False


def main():
    print(f"{'='*60}")
    print(f"  Rerun unconverged VASP SP calculations")
    print(f"  {len(TARGETS)} jobs, timeout={TIMEOUT}s, cores={CORES_PER_JOB}")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    dirs = []
    for level, sid_folder in TARGETS:
        d = os.path.join(BASE, f"vasp_level_{level}", sid_folder)
        if not os.path.isdir(d):
            print(f"WARNING: {d} not found, skipping")
            continue
        dirs.append(d)

    success = 0
    fail = 0
    for i, d in enumerate(dirs):
        if run_one(d, i, len(dirs)):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"  Done! Success={success}, Failed={fail}")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
