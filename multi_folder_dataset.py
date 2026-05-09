import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

HEIGHT_NORM_CONSTANT = 30.0


def _normalize_core_id(filename, strip_year_suffix=True):
    """
    Extract matching sample ID from filename.
    """

    base = os.path.splitext(os.path.basename(filename))[0]

    if base.startswith("label_"):
        base = base[len("label_"):]

    for prefix in (
        "emb_",
        "gee_emb_",
        "tessera_emb_",
        "s2_",
        "s1_",
    ):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    for suffix in (
        "_embeddings",
        "_embedding",
        "_quantized",
        "_quantized",
        "_merged",
    ):
        if base.endswith(suffix):
            base = base[:-len(suffix)]

    if strip_year_suffix:
        base = re.sub(r'_\d{4}$', '', base)

    return base

class MultiFolderDataset(Dataset):
    """
    Multi-modal dataset loader.

    Train sample:
        {
            "alphaearth_emb": tensor,
            ...
            "label": tensor
        }

    Test sample:
        {
            "alphaearth_emb": tensor,
            ...
        }
    """

    def __init__(
        self,
        root,
        split="train",
        input_folders=None,
        label_folder="labels",
        normalize_height=True,
        dtype=torch.float32,
    ):
        if split not in ("train", "test"):
            raise ValueError("Only train and test splits are supported.")

        self.split = split
        self.root = Path(root) / split
        self.input_folders = input_folders
        self.label_folder = label_folder
        self.normalize_height = normalize_height
        self.dtype = dtype

        self.has_labels = split == "train"

        if input_folders is None or len(input_folders) == 0:
            raise ValueError("input_folders must be provided")

        # ---------------------------------------------------------
        # Build input lookup maps
        # ---------------------------------------------------------

        self.input_maps = {}

        for folder in input_folders:
            folder_path = self.root / folder

            files = glob.glob(
                str(folder_path / "**" / "*.tif"),
                recursive=True
            )

            mapping = {}

            for f in files:
                sample_id = _normalize_core_id(f)
                mapping[sample_id] = f

            self.input_maps[folder] = mapping

        # ---------------------------------------------------------
        # TRAIN: labels required
        # TEST: labels skipped
        # ---------------------------------------------------------

        self.label_map = {}

        if self.has_labels:
            label_path = self.root / label_folder

            label_files = glob.glob(
                str(label_path / "**" / "*.tif"),
                recursive=True
            )

            for f in label_files:
                sample_id = _normalize_core_id(f)
                self.label_map[sample_id] = f

        # ---------------------------------------------------------
        # Keep only common IDs
        # ---------------------------------------------------------

        common_ids = set(self.input_maps[input_folders[0]].keys())

        for folder in input_folders[1:]:
            common_ids &= set(self.input_maps[folder].keys())

        if self.has_labels:
            common_ids &= set(self.label_map.keys())

        self.sample_ids = sorted(list(common_ids))

        if len(self.sample_ids) == 0:
            raise ValueError("No matching samples found.")

    def __len__(self):
        return len(self.sample_ids)

    def _fix_spatial_size(self, data):
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

        # latent embeddings -> leave untouched
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

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)

        data = np.nan_to_num(data)

        # normalize malformed spatial dimensions
        data = self._fix_spatial_size(data)

        return torch.tensor(data, dtype=self.dtype)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        sample = {}

        # ---------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------

        for folder in self.input_folders:
            path = self.input_maps[folder][sample_id]

            image = self._load_tif(path)

            sample[folder] = image

        # ---------------------------------------------------------
        # Labels (train only)
        # ---------------------------------------------------------

        if self.has_labels:
            label_path = self.label_map[sample_id]

            label = self._load_tif(label_path)

            if self.normalize_height:
                label[3] = torch.clamp(
                    label[3] / HEIGHT_NORM_CONSTANT,
                    min=0.0,
                    max=1.5
                )

            sample["label"] = label

        return sample