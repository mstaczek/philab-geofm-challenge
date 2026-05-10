import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src_ours.constants import HEIGHT_NORM_CONSTANT


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

def normalize_core_id(filename, strip_year_suffix=True):
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

def find_files_map(folder_path, files_format="tif"):
    files = glob.glob(
        str(folder_path / "**" / f"*.{files_format}"),
        recursive=True,
    )

    return {
        normalize_core_id(f): f
        for f in files
    }

def find_common_ids(input_maps, input_folders, label_map=None):
    common_ids = set(input_maps[input_folders[0]].keys())
    for folder in input_folders[1:]:
        common_ids &= set(input_maps[folder].keys())
    if label_map is not None:
        common_ids &= set(label_map.keys())
    return sorted(list(common_ids))

def build_input_maps(root, input_folders, files_format):
    return {
        folder: find_files_map(
            root / folder,
            files_format,
        )
        for folder in input_folders
    }

def build_label_map(root, label_folder, files_format):
    return find_files_map(
        root / label_folder,
        files_format,
    )
    
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

    This Dataset class can load samples from specified subfolders
    and preprocesses them accordingly.
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

        if input_folders is None or len(input_folders) == 0:
            raise ValueError("input_folders must be provided")
        
        FILES_FORMAT = "tif"

        self.split = split
        self.root = Path(root) / split
        self.input_folders = input_folders
        self.label_folder = label_folder
        self.normalize_height = normalize_height
        self.dtype = dtype
        self.has_labels = split == "train"

        self.input_maps = build_input_maps(
            self.root,
            input_folders,
            FILES_FORMAT,
        )

        self.label_map = {}
        if self.has_labels:
            self.label_map = build_label_map(
                self.root,
                label_folder,
                FILES_FORMAT,
            )

        self.sample_ids = find_common_ids(
            self.input_maps,
            input_folders,
            self.label_map if self.has_labels else None,
        )

        if len(self.sample_ids) == 0:
            raise ValueError("No matching samples found.")


    def __len__(self):
        return len(self.sample_ids)

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)

        data = np.nan_to_num(data)
        # normalize malformed spatial dimensions
        data = fix_spatial_size(data)

        return torch.tensor(data, dtype=self.dtype)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        sample = {}
        # Inputs
        for folder in self.input_folders:
            path = self.input_maps[folder][sample_id]
            image = self._load_tif(path)
            sample[folder] = image
        # Labels (train only)
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
    

class MultiFolderNpyDataset(Dataset):
    """
    This Dataset supports loading data from different subfolders.

    Assumes preprocessing already done:
        - no nans are present
        - spatial sizes all match from a given folder

    Applies height channel normalization of the labels
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

        if input_folders is None or len(input_folders) == 0:
            raise ValueError("input_folders must be provided")
        
        FILES_FORMAT = "npy"

        self.split = split
        self.root = Path(root) / split
        self.input_folders = input_folders
        self.label_folder = label_folder
        self.normalize_height = normalize_height
        self.dtype = dtype
        self.has_labels = split == "train"

        self.input_maps = build_input_maps(
            self.root,
            input_folders,
            FILES_FORMAT,
        )

        self.label_map = {}
        if self.has_labels:
            self.label_map = build_label_map(
                self.root,
                label_folder,
                FILES_FORMAT,
            )

        self.sample_ids = find_common_ids(
            self.input_maps,
            input_folders,
            self.label_map if self.has_labels else None,
        )

        if len(self.sample_ids) == 0:
            raise ValueError("No matching samples found.")

    def __len__(self):
        return len(self.sample_ids)

    def _load_npy(self, path):
        tensor = torch.from_numpy(np.load(path))

        if tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)

        return tensor

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        sample = {
            folder: self._load_npy(
                self.input_maps[folder][sample_id]
            )
            for folder in self.input_folders
        }

        if self.has_labels:
            label = self._load_npy(
                self.label_map[sample_id]
            )

            if self.normalize_height:
                label[3] = torch.clamp(
                    label[3] / HEIGHT_NORM_CONSTANT,
                    min=0.0,
                    max=1.5
                )

            sample["label"] = label

        return sample
