
import os
import glob

def clean_directory(root_dir):
    print(f"Cleaning VASP files in: {root_dir}")

    # Files to delete
    patterns = [
        "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1", "AECCAR2",
        "REPORT", "vaspout.h5"
    ]

    deleted_size = 0
    deleted_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in patterns:
                filepath = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_size += size
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

    print("-" * 40)
    print(f"Deleted {deleted_count} files.")
    print(f"Freed {deleted_size / (1024*1024):.2f} MB.")

if __name__ == "__main__":
    # Target directories
    base_dir = "grid_search_runs/pt_z1_epoch0021_valloss3.4507/val_nonrelaxed_update/nsites_10/cfg3_steps30"

    # Clean main vasp folder
    clean_directory(os.path.join(base_dir, "vasp"))

    # Clean refs folder
    clean_directory(os.path.join(base_dir, "vasp_refs"))
