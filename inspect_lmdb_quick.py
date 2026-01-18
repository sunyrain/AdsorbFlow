import lmdb
import pickle
import sys
import os
from torch_geometric.data import Data

def inspect(path):
    # Find .lmdb file
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith('.lmdb')]
        if not files:
            print(f"No .lmdb files found in {path}")
            return
        path = os.path.join(path, files[0])
    
    env = lmdb.open(path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data = pickle.loads(value)
                print(f"Key: {key}")
                print(f"Data keys: {data.keys}")
                if hasattr(data, 'pos_relaxed'):
                    print("pos_relaxed found!")
                else:
                    print("pos_relaxed NOT found.")
                break
            except Exception as e:
                print(f"Error loading key {key}: {e}")

if __name__ == "__main__":
    inspect(sys.argv[1])
