import lmdb
import pickle
import torch
from torch_geometric.data import Data

def inspect_lmdb(path):
    print(f"Inspecting {path}...")
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        count = 0
        for key, value in cursor:
            if count >= 1: break
            try:
                # OCP LMDBs usually store pickled PyG Data objects
                data = pickle.loads(value)
                print(f"Key: {key}")
                print(f"Type: {type(data)}")
                if hasattr(data, 'y'):
                    print(f"Has 'y': {data.y}")
                else:
                    print("Has 'y': False")

                if hasattr(data, 'energy'):
                    print(f"Has 'energy': {data.energy}")

                print(f"Keys: {data.keys}")
            except Exception as e:
                print(f"Error decoding value: {e}")
            count += 1
    env.close()

print("--- val_nonrelaxed_update ---")
inspect_lmdb("val_nonrelaxed_update/data.0000.lmdb")

print("\n--- valood50_R1I0.1 ---")
inspect_lmdb("valood50_R1I0.1/data.0000.lmdb")
