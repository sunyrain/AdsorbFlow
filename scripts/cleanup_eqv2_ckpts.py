#!/usr/bin/env python3
"""Clean up EqV2 checkpoints: keep every 5th epoch + epoch 187/208 + best + checkpoint."""
import os, re, glob, shutil

ckpt_base = "checkpoints"
eqv2_dirs = sorted([d for d in os.listdir(ckpt_base) if "eqv2" in d.lower()])

# Important epochs to ALWAYS keep (for reproducibility)
IMPORTANT_EPOCHS = {187, 208}

deleted_files = 0
deleted_bytes = 0
deleted_dirs = 0

for dirname in eqv2_dirs:
    dirpath = os.path.join(ckpt_base, dirname)
    pt_files = sorted(glob.glob(os.path.join(dirpath, "*.pt")))

    # Delete empty dirs
    if not pt_files:
        all_files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        if len(all_files) == 0:
            shutil.rmtree(dirpath)
            deleted_dirs += 1
            print(f"DEL dir (empty): {dirname}")
            continue

    epoch_files = []
    special_files = []
    for f in pt_files:
        basename = os.path.basename(f)
        match = re.match(r"epoch(\d+)_", basename)
        if match:
            epoch_files.append((int(match.group(1)), f, os.path.getsize(f)))
        else:
            special_files.append((basename, f, os.path.getsize(f)))

    # Early lr1.5-4 runs: all superseded, delete entirely
    if "lr1.5-4" in dirname:
        total_size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)))
        shutil.rmtree(dirpath)
        deleted_dirs += 1
        deleted_bytes += total_size
        print(f"DEL dir (superseded lr1.5): {dirname} ({total_size/1e9:.2f} GB)")
        continue

    # Short lr2.0-4 early run (ep1-11), superseded by 23-21-36 run
    if "21-28-32" in dirname:
        total_size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)))
        shutil.rmtree(dirpath)
        deleted_dirs += 1
        deleted_bytes += total_size
        print(f"DEL dir (superseded short): {dirname} ({total_size/1e9:.2f} GB)")
        continue

    # Only checkpoint.pt, no epochs -> keep as is
    if "20-56-32" in dirname:
        print(f"SKIP (only checkpoint.pt): {dirname}")
        continue

    # Main runs: keep every 5th + epoch1 + max + important epochs + best + checkpoint
    if len(epoch_files) <= 5:
        print(f"SKIP (few epochs): {dirname}")
        continue

    max_epoch = max(e for e, _, _ in epoch_files)
    del_count = 0
    for epoch, fpath, size in epoch_files:
        if epoch % 5 == 0 or epoch == 1 or epoch == max_epoch or epoch in IMPORTANT_EPOCHS:
            continue  # keep
        os.remove(fpath)
        deleted_files += 1
        deleted_bytes += size
        del_count += 1

    kept = len(epoch_files) - del_count
    print(f"CLEAN {dirname}: kept {kept}/{len(epoch_files)} epochs (incl 187,208), deleted {del_count}")

print(f"\n{'='*60}")
print(f"Freed: {deleted_files} files + {deleted_dirs} dirs = {deleted_bytes/1e9:.2f} GB")

# Verify important files still exist
main_dir = os.path.join(ckpt_base, "2026-01-13-23-21-36-z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr2.0-4_para_eqv2")
if os.path.exists(main_dir):
    for ep in [187, 208]:
        matches = glob.glob(os.path.join(main_dir, f"epoch{ep:04d}_*.pt"))
        status = "PRESERVED" if matches else "MISSING!"
        names = [os.path.basename(m) for m in matches]
        print(f"  epoch {ep}: {status} - {names}")
