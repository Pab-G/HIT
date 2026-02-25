"""
Extract lightweight SMPL params + body_verts from the large v1 repackaged PKL files.
Creates a single lookup file: v1_smpl_lookup.pkl

This avoids loading 28GB+ PKL files during training.

Usage:
    python extract_v1_smpl_lookup.py
"""

import os
import pickle
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REPACKAGED_DIR = os.path.join(BASE, "hit_dataset_v1.0", "repackaged")
OUTPUT_PATH = os.path.join(REPACKAGED_DIR, "v1_smpl_lookup.pkl")

# All v1 PKL files (gender_split.pkl)
PKL_FILES = [
    "female_train.pkl",
    "female_val.pkl",
    # "female_test.pkl",  # does not exist locally
    "male_train.pkl",
    "male_val.pkl",
    "male_test.pkl",
]

KEYS_TO_EXTRACT = [
    "betas",        # (N, 10)
    "body_pose",    # (N, 69)
    "global_orient",  # (N, 3)
    "transl",       # (N, 3)
    "body_verts",   # (N, 6890, 3)
    "seq_names",    # list of str
]


def main():
    lookup = {}  # key: (gender, v1_split, seq_name_id) -> dict of SMPL params

    for pkl_name in PKL_FILES:
        pkl_path = os.path.join(REPACKAGED_DIR, pkl_name)
        if not os.path.exists(pkl_path):
            print(f"Skipping {pkl_name} (not found)")
            continue

        # Parse gender and split from filename
        parts = pkl_name.replace(".pkl", "").split("_")
        gender = parts[0]  # "female" or "male"
        v1_split = parts[1]  # "train", "val", "test"

        print(f"Loading {pkl_name} ...", flush=True)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        n_subjects = len(data["seq_names"])
        print(f"  Found {n_subjects} subjects in {gender}_{v1_split}")

        for i in range(n_subjects):
            seq_name = data["seq_names"][i]
            entry = {}
            for key in KEYS_TO_EXTRACT:
                if key == "seq_names":
                    continue
                val = data[key]
                if hasattr(val, '__getitem__'):
                    entry[key] = val[i]
                else:
                    entry[key] = val

            lookup_key = f"{gender}_{v1_split}_{seq_name}"
            lookup[lookup_key] = entry

        # Free memory
        del data
        print(f"  Extracted {n_subjects} entries")

    print(f"\nTotal entries in lookup: {len(lookup)}")
    print(f"Saving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Check file size
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved! File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
