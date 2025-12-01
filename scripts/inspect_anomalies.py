#!/usr/bin/env python3
"""Quick diagnostics for anomalous_structures_new.pkl files.

Given the pickle written by scripts/eval.py::get_success_from_trajs_rewrite,
this script reports how many SIDs/FIDs were flagged for each anomaly type and
shows a few concrete examples so we can trace back which trajectories are
failing.
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

# Mapping matches DetectTrajAnomaly order in adsorbdiff/placement/flag_anomaly.py
ANOMALY_LABELS = {
    0: "dissociated",   # is_adsorbate_dissociated
    1: "desorbed",      # is_adsorbate_desorbed
    2: "surface_changed",  # has_surface_changed
    3: "intercalated",  # is_adsorbate_intercalated
}


def load_anomalies(pkl_path: Path):
    with pkl_path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at top level, got {type(data)}")
    return data


def summarize(anomalies):
    per_type = Counter()
    per_sid = Counter()
    examples = defaultdict(list)

    for sid, fid_dict in anomalies.items():
        for fid, flags in fid_dict.items():
            per_sid[sid] += 1
            if flags is None:
                continue
            try:
                iter_flags = list(flags)
            except TypeError:
                raise TypeError(
                    f"Expected iterable flags for {sid}/{fid}, got {type(flags)}"
                )
            for idx, flag in enumerate(iter_flags):
                if flag:
                    per_type[idx] += 1
                    # only keep a couple of examples per type
                    if len(examples[idx]) < 5:
                        examples[idx].append((sid, fid))
    return per_type, per_sid, examples


def main():
    parser = argparse.ArgumentParser(description="Inspect anomaly pickle")
    parser.add_argument(
        "pickle_path",
        nargs="?",
        default="anomalous_structures_new.pkl",
        help="Path to anomalous_structures_new.pkl (default: current dir)",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pickle_path)
    if pkl_path.is_dir():
        pkl_path = pkl_path / "anomalous_structures_new.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Cannot find {pkl_path}")

    anomalies = load_anomalies(pkl_path)
    per_type, per_sid, examples = summarize(anomalies)

    print(f"Loaded anomalies for {len(anomalies)} SIDs from {pkl_path}")
    print(f"Total flagged sid/fid pairs: {sum(per_sid.values())}")
    if not per_type:
        print("No anomaly flags found.")
        return

    print("\nBreakdown by anomaly type:")
    for idx, count in sorted(per_type.items()):
        label = ANOMALY_LABELS.get(idx, f"unknown_{idx}")
        print(f"  [{idx}] {label:16s}: {count}")
        if examples[idx]:
            example_str = ", ".join(f"{sid}/{fid}" for sid, fid in examples[idx])
            print(f"      e.g. {example_str}")

    if per_sid:
        worst = per_sid.most_common(10)
        print("\nTop offending SIDs (sid -> #flagged fids):")
        for sid, count in worst:
            print(f"  {sid}: {count}")


if __name__ == "__main__":
    main()
