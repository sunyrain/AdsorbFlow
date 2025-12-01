
import os
import sys

def get_size_str(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def analyze_directory(path):
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    print(f"Analyzing storage usage for: {path}")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Size':<15} | {'Category'}")
    print("-" * 60)

    total_size = 0
    essential_size = 0
    disposable_size = 0

    # Define categories
    essential_files = [
        "OUTCAR", "vasprun.xml", "OSZICAR", "CONTCAR", 
        "INCAR", "POSCAR", "POTCAR", "KPOINTS", "vasp.log"
    ]
    # WAVECAR and CHGCAR are usually the largest and needed for restart/properties
    # but not for just getting energy/structure.
    disposable_files = [
        "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1", "AECCAR2",
        "PROCAR", "DOSCAR", "EIGENVAL", "IBZKPT", "PCDAT", "XDATCAR", "REPORT"
    ]

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()

    for f in files:
        fp = os.path.join(path, f)
        size = os.path.getsize(fp)
        total_size += size
        
        category = "Other"
        if f in essential_files:
            category = "Essential (Result)"
            essential_size += size
        elif any(f.startswith(prefix) for prefix in disposable_files):
            category = "Disposable (Large)"
            disposable_size += size
        
        print(f"{f:<30} | {get_size_str(size):<15} | {category}")

    print("-" * 60)
    print(f"Total Size:       {get_size_str(total_size)}")
    print(f"Essential Size:   {get_size_str(essential_size)} (Keep)")
    print(f"Disposable Size:  {get_size_str(disposable_size)} (Safe to delete)")
    print(f"Potential Saving: {disposable_size / total_size * 100:.1f}%")

if __name__ == "__main__":
    # Use a known existing directory
    sample_dir = "/root/autodl-tmp/AdsorbDiff/grid_search_runs/pt_z1_epoch0021_valloss3.4507/val_nonrelaxed_update/nsites_10/cfg3_steps30/vasp/12_1990_4_0"
    analyze_directory(sample_dir)
