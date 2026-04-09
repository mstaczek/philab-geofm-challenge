import os
import shutil
import random
import argparse
from pathlib import Path
import re

from tqdm import tqdm


"""
Sample usage:
python split_data.py --train-ratio 0.8 --source-embeddings-dir data/public/embed2heights/data/train/terramind_s1_emb --source-targets-dir data/public/embed2heights/data/train/labels --train-embeddings-output-dir data/public/terramind_s1_train_test_split/train/embeddings --train-targets-output-dir data/public/terramind_s1_train_test_split/train/labels --test-embeddings-output-dir data/public/terramind_s1_train_test_split/test/embeddings --test-targets-output-dir data/public/terramind_s1_train_test_split/test/labels
"""

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)

    parser.add_argument("--source-embeddings-dir", required=True)
    parser.add_argument("--source-targets-dir", required=True)

    parser.add_argument("--train-embeddings-output-dir", required=True)
    parser.add_argument("--train-targets-output-dir", required=True)

    parser.add_argument("--test-embeddings-output-dir", required=True)
    parser.add_argument("--test-targets-output-dir", required=True)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()

    # Ensure output directories exist
    ensure_dirs(
        args.train_embeddings_output_dir,
        args.train_targets_output_dir,
        args.test_embeddings_output_dir,
        args.test_targets_output_dir,
    )


    def extract_key(filename):
        """
        Extract pattern like 0000_XX from filename.
        """
        match = re.search(r"\d{4}_[A-Z]{2}", filename)
        return match.group(0) if match else None


    # Build mapping: key -> file path
    def build_map(directory):
        mapping = {}
        for p in Path(directory).iterdir():
            if not p.is_file():
                continue
            key = extract_key(p.name)
            if key:
                mapping[key] = p
        return mapping


    source_embeddings_id_path_mapping = build_map(args.source_embeddings_dir)
    source_targets_id_path_mapping = build_map(args.source_targets_dir)

    # Find matching keys
    common_keys = sorted(set(source_embeddings_id_path_mapping) & set(source_targets_id_path_mapping))

    if not common_keys:
        raise ValueError("No matching files found between embeddings and targets")
    else:
        print(f"Found {len(common_keys)} matching pairs of files")

    # Optional: warn if mismatch
    if len(source_embeddings_id_path_mapping) != len(source_targets_id_path_mapping):
        print("Warning: source dirs have different number of files")
        print("In first, and not in second:")
        print(set(source_embeddings_id_path_mapping) - set(source_targets_id_path_mapping))
        print("In second, and not in first:")
        print(set(source_targets_id_path_mapping) - set(source_embeddings_id_path_mapping))
        # exit()

    # Shuffle
    random.seed(args.seed)
    random.shuffle(common_keys)

    # Split
    split_idx = int(len(common_keys) * args.train_ratio)
    train_keys = common_keys[:split_idx]
    test_keys = common_keys[split_idx:]

    print(f"Train: {len(train_keys)}")
    print(f"Test: {len(test_keys)}")


    def copy_pairs(keys, output_embeddings_dir, output_labels_dir):
        for k in tqdm(keys):
            shutil.copy2(source_embeddings_id_path_mapping[k], Path(output_embeddings_dir) / source_embeddings_id_path_mapping[k].name)
            shutil.copy2(source_targets_id_path_mapping[k], Path(output_labels_dir) / source_targets_id_path_mapping[k].name)


    # Copy
    copy_pairs(train_keys,
            args.train_embeddings_output_dir,
            args.train_targets_output_dir)

    copy_pairs(test_keys,
            args.test_embeddings_output_dir,
            args.test_targets_output_dir)


if __name__ == "__main__":
    main()