import pickle
import sys
from pathlib import Path

def inspect_pkl(path):
    print(f"Inspecting {path}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            print(f"Sample keys: {list(data.keys())[:5]}")
            sample_val = next(iter(data.values()))
            print(f"Sample value: {sample_val}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect_pkl("oc20_dense_mappings/oc20dense_targets.pkl")
inspect_pkl("oc20_dense_mappings/oc20dense_ref_energies.pkl")
