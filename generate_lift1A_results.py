import sys
import os
import pickle
import json
from pathlib import Path
from statistics import mean
import numpy as np

# Add root to path
REPO_ROOT = Path("/root/autodl-tmp/AdsorbFlow")
sys.path.insert(0, str(REPO_ROOT))

from scripts import eval as eval_module

def load_dft_targets(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    dft_targets_path = REPO_ROOT / "oc20_dense_mappings/oc20dense_targets.pkl"
    print(f"Loading DFT targets from {dft_targets_path}")
    dft_targets = load_dft_targets(dft_targets_path)
    
    # Path to Lift 1A experiment
    exp_dir = REPO_ROOT / "grid_search_runs/test_z_0_geo_lift1A_cfg_0.15_epoch0075_valloss0.6023_posmae1.0425/val_nonrelaxed_update/nsites_3"
    output_file = exp_dir / "grid_search_results_nsites3.jsonl"
    
    # We only have cfg1_steps10
    configs = [("1", "10")]
    
    results = []
    
    for cfg, steps in configs:
        combo_dir = exp_dir / f"cfg{cfg}_steps{steps}"
        if not combo_dir.exists():
            print(f"Skipping {combo_dir} (not found)")
            continue
            
        print(f"Processing {combo_dir}")
        
        site_success_rates = []
        site_valid_counts = []
        site_success_counts = []
        site_diff_means = []
        site_diff_variances = []
        site_successful_sids = []
        
        nsites = 3 # We know it is nsites_3
        
        max_total_targets = 0
        
        for site_idx in range(nsites):
            relax_dir = combo_dir / str(site_idx) / "relaxations"
            if not relax_dir.exists():
                print(f"  Site {site_idx} relaxations not found")
                # Treat as 0 success
                site_success_rates.append(0.0)
                site_valid_counts.append(0)
                site_success_counts.append(0)
                site_diff_means.append(None)
                site_diff_variances.append(None)
                site_successful_sids.append(set())
                continue
                
            try:
                (
                    success_percent,
                    valid_count,
                    success_count,
                    diff_mean,
                    diff_var,
                    successful_sids,
                    t_targets,
                ) = eval_module.get_success_from_trajs_rewrite(str(relax_dir), dft_targets)
                
                site_success_rates.append(success_percent)
                site_valid_counts.append(valid_count)
                site_success_counts.append(success_count)
                site_diff_means.append(diff_mean)
                site_diff_variances.append(diff_var)
                site_successful_sids.append(successful_sids)
                if t_targets > max_total_targets:
                    max_total_targets = t_targets
                
            except Exception as e:
                print(f"  Error processing site {site_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Treat as 0 success
                site_success_rates.append(0.0)
                site_valid_counts.append(0)
                site_success_counts.append(0)
                site_diff_means.append(None)
                site_diff_variances.append(None)
                site_successful_sids.append(set())

        # Aggregate results
        if not site_success_rates:
            continue

        # Union success rate
        all_sids = set()
        for sids in site_successful_sids:
            all_sids.update(sids)
        
        union_success_count = len(all_sids)
        union_success_percent = (union_success_count / max_total_targets) * 100.0 if max_total_targets > 0 else 0.0
        
        mean_success_percent = mean(site_success_rates)
        max_single_success_percent = max(site_success_rates)
        
        # Calculate energy diff stats
        valid_diff_means = [m for m in site_diff_means if m is not None]
        valid_diff_vars = [v for v in site_diff_variances if v is not None]
        
        energy_diff_mean = mean(valid_diff_means) if valid_diff_means else None
        energy_diff_variance = mean(valid_diff_vars) if valid_diff_vars else None

        result = {
            "cfg_scale": float(cfg),
            "num_steps": int(steps),
            "mean_success_percent": mean_success_percent,
            "union_success_percent": union_success_percent,
            "max_single_success_percent": max_single_success_percent,
            "site_success_percent": site_success_rates,
            "site_valid_counts": site_valid_counts,
            "site_success_counts": site_success_counts,
            "total_valid_count": sum(site_valid_counts),
            "total_success_count": sum(site_success_counts),
            "output_dir": str(combo_dir),
            "nsites": nsites
        }
        
        results.append(result)
        print(f"  Mean Success: {mean_success_percent:.2f}%")
        print(f"  Union Success: {union_success_percent:.2f}%")

    # Write to file
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"Written results to {output_file}")

if __name__ == "__main__":
    main()
