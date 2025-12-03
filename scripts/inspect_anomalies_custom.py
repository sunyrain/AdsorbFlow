import pickle
import sys
import numpy as np

def inspect_anomalies(pkl_path):
    print(f"Inspecting: {pkl_path}")
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    if not data:
        print("No anomalies found (empty dictionary).")
        return

    anomaly_labels = [
        "Dissociated",
        "Desorbed",
        "Surface changed",
        "Intercalated"
    ]

    print(f"{'SID':<30} | {'FID':<5} | Anomalies")
    print("-" * 80)

    for sid, fid_dict in data.items():
        for fid, anom_vec in fid_dict.items():
            anom_indices = np.where(anom_vec)[0]
            anom_types = [anomaly_labels[i] for i in anom_indices]
            anom_str = ", ".join(anom_types)
            print(f"{sid:<30} | {fid:<5} | {anom_str}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = "anomalous_structures_new.pkl"
    
    inspect_anomalies(pkl_path)
