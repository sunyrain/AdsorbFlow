import os
import glob
import numpy as np
from ase.io import read

def check_z_distances(directory):
    traj_files = glob.glob(os.path.join(directory, "*.traj"))
    if not traj_files:
        print(f"No .traj files found in {directory}")
        return

    print(f"Found {len(traj_files)} files. Checking z-distances...")
    print(f"{'Filename':<30} {'Min Ads Z':<12} {'Max Surf Z':<12} {'Z-Dist':<12} {'Min Pair Dist':<12}")
    print("-" * 85)

    distances = []
    
    for filepath in sorted(traj_files):
        try:
            # Read the last image in the trajectory (the relaxed structure)
            atoms = read(filepath, index=-1)
            
            tags = atoms.get_tags()
            # Assuming tag 2 is adsorbate, others are surface/slab
            # If tags are all 0, this heuristic fails.
            
            ads_indices = np.where(tags == 2)[0]
            surf_indices = np.where(tags != 2)[0]
            
            if len(ads_indices) == 0:
                print(f"{os.path.basename(filepath):<30} No adsorbate atoms found (tag=2)")
                continue
                
            if len(surf_indices) == 0:
                print(f"{os.path.basename(filepath):<30} No surface atoms found")
                continue

            positions = atoms.get_positions()
            z_coords = positions[:, 2]
            
            z_ads = z_coords[ads_indices]
            z_surf = z_coords[surf_indices]
            
            min_ads_z = np.min(z_ads)
            max_surf_z = np.max(z_surf)
            
            z_distance = min_ads_z - max_surf_z
            
            # Calculate minimum pairwise distance between adsorbate and surface atoms
            # to distinguish between hollow sites and actual collisions
            ads_pos = positions[ads_indices]
            surf_pos = positions[surf_indices]
            
            # Simple brute force distance check (efficient enough for small systems)
            # shape: (n_ads, n_surf)
            dists = np.linalg.norm(ads_pos[:, None, :] - surf_pos[None, :, :], axis=-1)
            min_pairwise_dist = np.min(dists)
            
            distances.append(z_distance)
            
            print(f"{os.path.basename(filepath):<30} {min_ads_z:<12.4f} {max_surf_z:<12.4f} {z_distance:<12.4f} {min_pairwise_dist:<12.4f}")
            
        except Exception as e:
            print(f"Error reading {os.path.basename(filepath)}: {e}")

    if distances:
        distances = np.array(distances)
        print("-" * 85)
        print(f"Statistics for {len(distances)} structures:")
        print(f"Minimum Z-Distance: {np.min(distances):.4f}")
        print(f"Maximum Z-Distance: {np.max(distances):.4f}")
        
        # Check for potential penetration (negative distance or very small positive)
        penetrating = distances[distances < 0]
        if len(penetrating) > 0:
            print(f"\nAnalysis of negative Z-distances ({len(penetrating)} structures):")
            print("Note: Negative Z-distance with reasonable pairwise distance (>1.5A) usually means")
            print("      the adsorbate is sitting in a hollow site (valid).")
            print("      Very small pairwise distance (<1.0A) indicates collision (invalid).")

if __name__ == "__main__":
    # Check the '0' directory as identified in the file list
    check_z_distances("/root/autodl-tmp/AdsorbFlow/grid_search_runs/2025-11-19-09-44-32-debug-head_epoch0116_valloss2.2769/cfg2_steps30/0/relaxations")
