#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inspect scale factors .pt file")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file (e.g., painn_nb6_scaling_factors.pt)")
    args = parser.parse_args()

    path = Path(args.pt_file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    data = torch.load(path, map_location="cpu")

    print(f"Loaded file: {path}")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print("\nScale factors:")
        for k, v in data.items():
            if torch.is_tensor(v):
                v = v.item() if v.numel() == 1 else v.tolist()
            print(f"  {k}: {v}")
    else:
        print("File content is not a dict, here is the raw object:")
        print(data)

if __name__ == "__main__":
    main()
