import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from adsorbdiff.datasets.lmdb_dataset import LmdbDataset

def check_z_distribution():
    # Config for the dataset
    config = {
        "src": "val_nonrelaxed_update",
        "train_on_oc20_total_energies": False
    }

    print(f"Loading dataset from {config['src']}...")
    dataset = LmdbDataset(config)
    print(f"Dataset size: {len(dataset)}")

    min_dists = []
    com_dists = []

    # Check a random subset if dataset is too large, or all if manageable
    # 1000 samples should be enough for a distribution check
    num_samples = min(len(dataset), 2000)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"Analyzing {num_samples} samples (using 'pos' for initial distribution)...")

    for idx in tqdm(indices):
        data = dataset[idx]
        # For checking initial distribution of test set, we want 'pos' (unrelaxed)
        pos = data.pos

        tags = data.tags

        # 0: bulk, 1: surface, 2: adsorbate
        slab_mask = tags != 2
        ads_mask = tags == 2

        if not ads_mask.any():
            continue
        if not slab_mask.any():
            continue

        slab_pos = pos[slab_mask]
        ads_pos = pos[ads_mask]

        slab_z_max = slab_pos[:, 2].max().item()
        ads_z_min = ads_pos[:, 2].min().item()
        ads_z_com = ads_pos[:, 2].mean().item()

        # Simple PBC check: if distance is very large negative, assume wrapping
        # We only care about the "on top" case for the distribution check
        dist = ads_z_min - slab_z_max
        if dist > -5.0: # Filter out likely PBC wrapped cases
            min_dists.append(dist)
            com_dists.append(ads_z_com - slab_z_max)

    min_dists = np.array(min_dists)
    com_dists = np.array(com_dists)

    print("\n--- Z-Distance Statistics (Adsorbate - Slab Max Z) [Filtered PBC < -5Å] ---")
    print(f"Sample count: {len(min_dists)}")

    print("\nMinimum Z Distance (Closest Atom):")
    print(f"  Mean:   {min_dists.mean():.4f} Å")
    print(f"  Median: {np.median(min_dists):.4f} Å")
    print(f"  Min:    {min_dists.min():.4f} Å")
    print(f"  Max:    {min_dists.max():.4f} Å")
    print(f"  Std:    {min_dists.std():.4f} Å")

    print("\nCenter of Mass Z Distance:")
    print(f"  Mean:   {com_dists.mean():.4f} Å")
    print(f"  Median: {np.median(com_dists):.4f} Å")
    print(f"  Min:    {com_dists.min():.4f} Å")
    print(f"  Max:    {com_dists.max():.4f} Å")
    print(f"  Std:    {com_dists.std():.4f} Å")

    # Simple ASCII Histogram for Min Distance
    print("\nHistogram of Minimum Z Distances:")
    hist, bin_edges = np.histogram(min_dists, bins=20)
    max_count = hist.max()
    for count, edge in zip(hist, bin_edges):
        bar = '#' * int(50 * count / max_count)
        print(f"{edge:6.2f} Å | {bar} ({count})")

if __name__ == "__main__":
    check_z_distribution()
