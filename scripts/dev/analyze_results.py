import json
import os
import pandas as pd

base_dir = "grid_search_runs"

experiments = {
    "lift1A": [
        "test_z_0_geo_lift1A_cfg_0.15_epoch0015_valloss4.4015_posmae1.4320",
        "test_z_0_geo_lift1A_cfg_0.15_epoch0025_valloss3.0019_posmae1.3189",
        "test_z_0_geo_lift1A_cfg_0.15_epoch0040_valloss3.5483_posmae1.0714",
        "test_z_0_geo_lift1A_cfg_0.15_epoch0060_valloss0.9849_posmae1.4180",
        "test_z_0_geo_lift1A_cfg_0.15_epoch0070_valloss1.2386_posmae1.0050",
        "test_z_0_geo_lift1A_cfg_0.15_epoch0075_valloss0.6023_posmae1.0425"
    ],
    "nolift": [
        "test_z_0_geo_nolift_cfg_0.15_epoch0015_valloss5.5176_posmae2.1918",
        "test_z_0_geo_nolift_cfg_0.15_epoch0025_valloss2.2208_posmae1.1429",
        "test_z_0_geo_nolift_cfg_0.15_epoch0040_valloss3.0623_posmae0.9447",
        "test_z_0_geo_nolift_cfg_0.15_epoch0060_valloss0.6098_posmae1.3194",
        "test_z_0_geo_nolift_cfg_0.15_epoch0075_valloss0.7579_posmae0.6770",
        "test_z_0_geo_nolift_cfg_0.15_epoch0090_valloss1.0582_posmae1.1466"
    ]
}

results = []

for exp_type, folders in experiments.items():
    for folder in folders:
        # Extract epoch from folder name
        try:
            epoch_str = folder.split("epoch")[1].split("_")[0]
            epoch = int(epoch_str)
        except:
            epoch = -1

        file_path = os.path.join(base_dir, folder, "val_nonrelaxed_update/nsites_1/grid_search_results_nsites1.jsonl")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract relevant metrics
                    # Assuming the structure based on typical grid search results
                    # We need to check the keys. I'll print keys of the first item found to be sure.

                    res = {
                        "type": exp_type,
                        "epoch": epoch,
                        "folder": folder,
                        "cfg": data.get("cfg_scale"), # or similar key
                        "steps": data.get("num_steps"), # or similar key
                        "success_rate": data.get("success_rate"),
                        "ads_rate": data.get("ads_rate"),
                        "pos_mae": data.get("pos_mae")
                    }

                    # Handle potential key variations if needed, but let's try these first
                    if res["cfg"] is None: res["cfg"] = data.get("scale")

                    results.append(res)
                except json.JSONDecodeError:
                    continue

df = pd.DataFrame(results)

# Group by type, epoch, cfg, steps and print summary
if not df.empty:
    print(df.groupby(['type', 'epoch', 'cfg', 'steps'])[['success_rate', 'ads_rate', 'pos_mae']].mean().to_string())
else:
    print("No data found.")

# Also print the raw keys of one entry to verify
if results:
    print("\nKeys in first record:", results[0].keys())
    # Re-read first file to get actual keys from file content
    first_file = os.path.join(base_dir, experiments["lift1A"][0], "val_nonrelaxed_update/nsites_1/grid_search_results_nsites1.jsonl")
    if os.path.exists(first_file):
        with open(first_file, 'r') as f:
            print("Actual keys in JSON:", json.loads(f.readline()).keys())
