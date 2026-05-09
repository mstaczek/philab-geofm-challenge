from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from multi_folder_dataset import MultiFolderDataset


# =========================================================
# SETTINGS
# =========================================================

SOURCE_ROOT = "data/embed2heights/data"

OUTPUT_ROOT = Path("data/embed2heights_npy")

TRAIN_INPUT_FOLDERS = [
    "alphaearth_emb",
    "terramind_s1_emb",
    "terramind_s2_emb",
    "tessera_emb",
    "thor_s1_emb",
    "thor_s2_emb",
]

TEST_INPUT_FOLDERS = [
    "alphaearth_test_emb",
    "terramind_test_s1_emb",
    "terramind_test_s2_emb",
    "tessera_test_emb",
    "thor_test_s1_emb",
    "thor_test_s2_emb",
]

LABEL_FOLDER = "labels"

SAVE_DTYPE = np.float16


# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = MultiFolderDataset(
    root=SOURCE_ROOT,
    split="train",
    input_folders=TRAIN_INPUT_FOLDERS,
)

test_dataset = MultiFolderDataset(
    root=SOURCE_ROOT,
    split="test",
    input_folders=TEST_INPUT_FOLDERS,
)


# =========================================================
# HELPERS
# =========================================================

def fix_spatial_size(data):
    """
    Fix slightly malformed spatial dimensions.

    Keeps:
        16x16 unchanged

    Converts:
        256x255 -> 256x256
        255x256 -> 256x256
        255x255 -> 256x256
    """

    _, h, w = data.shape

    # latent embeddings
    if (h, w) == (16, 16):
        return data

    # already correct
    if (h, w) == (256, 256):
        return data

    # allow only near-256 cases
    if h not in (255, 256) or w not in (255, 256):
        raise ValueError(
            f"Unexpected spatial size {(h, w)}. "
            f"Expected 16x16 or near-256 shapes."
        )

    pad_h = 256 - h
    pad_w = 256 - w

    data = np.pad(
        data,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    return data


def save_tif_as_npy(src_path, dst_path):
    """
    Load TIFF and save canonical float16 NPY.
    """

    with rasterio.open(src_path) as src:
        arr = src.read()

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    arr = np.nan_to_num(arr)

    # -----------------------------------------------------
    # Fix malformed spatial dimensions
    # -----------------------------------------------------

    arr = fix_spatial_size(arr)

    # -----------------------------------------------------
    # Convert dtype
    # -----------------------------------------------------

    arr = arr.astype(SAVE_DTYPE)

    # -----------------------------------------------------
    # Make contiguous
    # -----------------------------------------------------

    arr = np.ascontiguousarray(arr)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(dst_path, arr)


def export_split(
    dataset,
    split_name,
    input_folders,
):
    """
    Export one split to NPY.
    """

    print(f"\nExporting split: {split_name}")

    split_root = OUTPUT_ROOT / split_name

    # -----------------------------------------------------
    # Create folder structure
    # -----------------------------------------------------

    for folder in input_folders:
        (split_root / folder).mkdir(parents=True, exist_ok=True)

    if dataset.has_labels:
        (split_root / LABEL_FOLDER).mkdir(
            parents=True,
            exist_ok=True
        )

    # -----------------------------------------------------
    # Export inputs
    # -----------------------------------------------------

    for sample_id in tqdm(dataset.sample_ids):

        # -------------------------------------------------
        # Inputs
        # -------------------------------------------------

        for folder in input_folders:

            src_path = dataset.input_maps[folder][sample_id]

            dst_name = Path(src_path).stem + ".npy"

            dst_path = split_root / folder / dst_name

            save_tif_as_npy(src_path, dst_path)

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        if dataset.has_labels:

            src_path = dataset.label_map[sample_id]

            dst_name = Path(src_path).stem + ".npy"

            dst_path = split_root / LABEL_FOLDER / dst_name

            save_tif_as_npy(src_path, dst_path)

    print(f"Finished split: {split_name}")


# =========================================================
# RUN EXPORT
# =========================================================

export_split(
    dataset=train_dataset,
    split_name="train",
    input_folders=TRAIN_INPUT_FOLDERS,
)

export_split(
    dataset=test_dataset,
    split_name="test",
    input_folders=TEST_INPUT_FOLDERS,
)

print("\nAll conversions complete.")