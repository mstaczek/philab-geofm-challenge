## Links

- Competition: https://platform.ai4eo.eu/geoai/data
- Dataset: https://www.eotdl.com/datasets/embed2heights
- Sample repo code: https://github.com/VMarsocci/emb2heights-baselines/tree/main
- Drive: https://drive.google.com/drive/folders/1yxXF3bJ1C6zCsZ1XLQWnNfkf98VI-21f


## Create local symlink to data in NAS

Usable in VS Code, from local folder to sample shared network drive.
```bash
mklink /D C:\Users\matem\T\python_projects\philab\data_from_drive\public \\192.168.0.77\public\philab-dataset
```

Optionally, to re-download data:
```bash
pip install eotdl 
eotdl datasets get embed2heights --path . --version 1
eotdl datasets get embed2heights --path . --version 1 --assets --verbose --force 
```

## Installation (ours)

Create new venv called .venv:
```bash
python -m venv .venv
```

Activate venv:
```bash
# Windows:
.venv\Scripts\activate
# Linux 
source .venv/bin/activate
```

then install requirements:
```bash
pip install -r requirements.txt
```

## Copy and split data from labelled dataset to train/test

Target is a separate folder in NAS, e.g.: terramind_s1_train_test_split:

```bash
python split_data.py --train-ratio 0.8 --source-embeddings-dir data/public/embed2heights/data/train/terramind_s1_emb --source-targets-dir data/public/embed2heights/data/train/labels --train-embeddings-output-dir data/public/terramind_s1_train_test_split/train/embeddings --train-targets-output-dir data/public/terramind_s1_train_test_split/train/labels --test-embeddings-output-dir data/public/terramind_s1_train_test_split/test/embeddings --test-targets-output-dir data/public/terramind_s1_train_test_split/test/labels
```

## Run training

Internally, it handles train/val splitting of the input data BUT by splitting data earlier and passing a single split, we can limit the size of the dataset for training.

```bash
python train.py --model-type decoder_residual --output-dir runs --train-embeddings-dir data/terramind_s1_train_test_split/test/embeddings --train-targets-dir data/terramind_s1_train_test_split/test/labels --experiment-name test_terramind_s1_decoder_residual_v2 --epochs 10 --batch-size 16 --patch-size 256 --device cuda
```

## Run predict

Save a few train predictions
```bash
python predict.py --experiment-name test_terramind_s1_decoder_residual_v2_cuda --base-dir runs --model-type decoder_residual --model-path runs/test_terramind_s1_decoder_residual_v2_cuda/model_best.pth --test-embeddings-dir data/terramind_s1_train_test_split/test/embeddings --predictions-dir runs/test_terramind_s1_decoder_residual_v2_cuda/predictions_train --patch-size 256 --max-samples 5 --device cuda
```

Test 5 samples predictions
```bash
python predict.py --experiment-name test_terramind_s1_decoder_residual --base-dir runs --model-type decoder_residual --model-path runs/test_terramind_s1_decoder_residual/model_best.pth --test-embeddings-dir data/public/embed2heights/data/test/terramind_test_s1_emb --predictions-dir runs/test_terramind_s1_decoder_residual/predictions --patch-size 256 --max-samples 5 --device cuda
```

Compute all predictions
```bash
python predict.py --experiment-name test_terramind_s1_decoder_residual --base-dir runs --model-type decoder_residual --model-path runs/test_terramind_s1_decoder_residual/model_best.pth --test-embeddings-dir data/public/embed2heights/data/test/terramind_test_s1_emb --predictions-dir runs/test_terramind_s1_decoder_residual/predictions --patch-size 256 --device cuda
```

## Send submission

Using last cell in `starter_pack-embed2heights.ipynb`, set correct experiment name, run, find the new zip in the `submissions/` folder, and upload the `submission.zip` file to https://platform.ai4eo.eu/geoai/submissions (can be uploaded every 12h).

---
# Copied from https://www.eotdl.com/datasets/embed2heights?ref=philabchallenges-cms.earthpulse.es
---

# embed2heights Challenge - Reaching New Heights with GeoFM Embeddings

## Overview

The objective of the **embed2heights** challenge is to develop a multi-task method that uses Geospatial Foundation Model embeddings to map land cover and estimate heights at scale. Participants are asked to combine multiple embedding sources to segment buildings, vegetation, and water, and to predict building and vegetation heights.

Participants receive an AI-ready package with **pre-computed embeddings** from four GFMs: **AlphaEarth, TESSERA, TerraMind, and THOR**, plus reference labels derived from IGN airborne LiDAR products. This enables direct experimentation on feature fusion and multi-task modeling without running large-scale embedding generation.

## Dataset

The training dataset includes **2,024 patches** of size **256x256 at 10 m resolution**, sampled over major French cities and selected rural areas in France. Labels are derived from IGN products based on airborne LiDAR.

Labels are **not discrete categories**. For each pixel, they represent the percentage contribution of each class within a **10x10 m cell**. Label data is provided as 4-band TIFF files:

- **Band 1**: percentage of building
- **Band 2**: percentage of vegetation
- **Band 3**: percentage of water
- **Band 4**: relative height above ground (nDSM)

The source data is generated at **1 m spatial resolution** and includes four classes: **Background**, **Buildings**, **Trees/HighVegetation**, and **Unclassified**. The *Unclassified* class captures mixed/overlap cases (for example, a tree attached to a house).

The test set (around **1,000 patches**) is generated with similar data, but from different regions and years.

## Training Data Folder Structure

Inside the `data/train/` directory, the following subdirectories are provided:

- `alphaearth_emb`: AlphaEarth pixel-level embeddings, shape `(256, 256, 64)`, total size **33.93 GB**.
- `labels`: reference label tensors (not a model output), pixel-level, shape `(256, 256, 4)`, total size **2.12 GB**.
- `terramind_s1_emb`: TerraMind (S1) patch-level embeddings, shape `(16, 16, 768)`, total size **1.60 GB**.
- `terramind_s2_emb`: TerraMind (S2) patch-level embeddings, shape `(16, 16, 768)`, total size **1.60 GB**.
- `tessera_emb`: Tessera pixel-level embeddings, shape `(256, 256, 128)`, total size **67.82 GB**.
- `thor_s1_emb`: THOR (S1) patch-level embeddings, shape `(16, 16, 768)`, total size **1.96 GB**.
- `thor_s2_emb`: THOR (S2) patch-level embeddings, shape `(16, 16, 768)`, total size **1.95 GB**.

All subdirectories currently contain the same number of files: **2,024**.

In addition, the `data/` directory also includes a `catalog.parquet` manifest file (~**2.1 MB**), with **14,169 rows** and **10 columns** (`type`, `stac_version`, `stac_extensions`, `datetime`, `id`, `bbox`, `geometry`, `assets`, `links`, `repository`). The catalog stores one entry per object, with **2,024** entries for each data subdirectory. The `assets` field provides per-file metadata such as checksum, source `href`, file size, and timestamp.

## Baseline

A reference baseline implementation is available on GitHub:

- <https://github.com/VMarsocci/emb2heights-baselines>

## Evaluation

The team score is computed with a **weighted multi-metric evaluation** combining segmentation and height accuracy: `mIoU_buildings` (25%), `mIoU_trees` (15%), `mIoU_water` (15%), `RMSE_buildings` (25%), and `RMSE_vegetation` (20%). The final leaderboard score is the weighted mean of these five metrics.

## Submission Requirements

Each submission **must include predictions for all 946 test patches**.

Each evaluated submission receives:

- a **public score**, computed on a subset of the test set;
- a **private score**, computed on the full test set and used for final ranking.

The exact patch-level composition of the public evaluation subset is not disclosed to participants.

At the end of the challenge, private scores are revealed and the final leaderboard is computed using the private score.

---
# end of copied section
---


---
# BELOW IS COPIED FROM https://github.com/VMarsocci/emb2heights-baselines/tree/main
---

# Emb2Heights: Urban Structure and Land Cover Prediction

This repository is a baseline for the **Emb2Heights challenge**. It trains and runs inference for a model that predicts sub-pixel land cover percentages (Building, Vegetation, Water) and continuous structure heights (nDSM) directly from Earth Observation embeddings. Predictions are saved as `.npy` files with **4 output channels**: `[% Building, % Vegetation, % Water, Height (m)]`.

## Project Overview

Predicting urban morphology from satellite imagery is challenging: building footprints are sparse, and height values operate on a different scale than land-cover probabilities. This project addresses these challenges through a composite loss with **4 terms**:

- **MAE** (with background/foreground split): direct pixel-level regression.
- **SSIM + Gradient Loss**: enforces sharp structural boundaries on land-cover channels.
- **Tversky Loss**: penalizes false negatives heavily, forcing the model to capture sparse building footprints (α=0.3, β=0.7).
- **Structure-Boosted Height Loss**: height errors on building pixels are penalized 2x more than background pixels.

Training is further stabilized with AdamW (weight decay) and gradient clipping to prevent collapse on complex urban patches.

---

## Repository Structure

```text
emb2heights_baselines/
├── core/
│   ├── __init__.py
│   ├── model.py        # LightUNet + Decoder model factory
│   ├── dataset.py      # Dataset classes + embedding/label pairing utilities
│   └── losses.py       # ImprovedCompositeLoss (MAE, SSIM, Gradient, Tversky)
├── train.py            # Training entrypoint (fully CLI-configurable)
├── predict.py          # Inference entrypoint (loads checkpoint, saves .npy predictions)
├── environment.yml     # Conda environment definition
├── readme.md
└── runs/               # Auto-generated experiment outputs
    └── <experiment_name>/
        ├── model_best.pth
        ├── model_last.pth
        ├── loss_curve.png
        ├── training_params.txt
        ├── visualizations/
        └── predictions/
```

---

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate emb2heights
```

---

## Model Architecture

Architecture is selected via `--model-type`:

| Value | Description |
|---|---|
| `lightunet` | Lightweight encoder-decoder with skip connections |
| `decoder` | Transposed-convolution decoder |
| `decoder_residual` | Deeper decoder with residual blocks + global embedding skip fusion (recommended for high-channel embeddings) |
| `auto` | Selects `decoder` when input channels = 768, otherwise `lightunet` |

**Output**: a 4-channel tensor — `[0: % Building, 1: % Vegetation, 2: % Water, 3: Height (m)]`.

**Loss function**: `ImprovedCompositeLoss` with 4 terms — see [Project Overview](#project-overview).

---

## Training

Run training from the CLI — no file edits needed.

```bash
python train.py \
    --model-type decoder_residual \
    --train-embeddings-dir /path/to/train/embeddings \
    --train-targets-dir /path/to/train/labels \
    --test-embeddings-dir /path/to/test/embeddings \
    --test-targets-dir /path/to/test/labels \
    --experiment-name my_run \
    --epochs 30 \
    --batch-size 8 \
    --patch-size 256
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `auto` | Architecture: `auto`, `lightunet`, `decoder`, `decoder_residual` |
| `--train-embeddings-dir` | — | Path to training embedding `.tif` files |
| `--train-targets-dir` | — | Path to training label `.tif` files |
| `--test-embeddings-dir` | — | Path to test embeddings (used for post-training visualization) |
| `--test-targets-dir` | — | Path to test labels (used for post-training visualization) |
| `--experiment-name` | `terramid_run02` | Subfolder name under `./runs/` |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--patch-size` | `256` | Spatial crop size for dataset loader |

Outputs are written to `./runs/<experiment_name>/`: hyperparameter log, `model_best.pth`, `model_last.pth`, loss curve, and sample visualizations.

---

## Inference

Load a trained checkpoint and save predictions as `.npy` files (shape `[4, H, W]`, channels: building %, vegetation %, water %, height in meters).

```bash
python predict.py \
    --experiment-name my_run \
    --model-type decoder_residual \
    --test-embeddings-dir /path/to/test/embeddings \
    --test-targets-dir /path/to/test/labels
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--experiment-name` | `terramind_decoder_run01` | Experiment folder under `--base-dir` |
| `--base-dir` | `./runs` | Root directory of experiment folders |
| `--model-type` | `decoder_residual` | Architecture (must match training) |
| `--model-path` | `<base-dir>/<experiment-name>/model_best.pth` | Path to `.pth` checkpoint |
| `--test-embeddings-dir` | required | Directory with embedding `.tif` files |
| `--test-targets-dir` | required | Directory with label `.tif` files (used only for file pairing) |
| `--predictions-dir` | `<base-dir>/<experiment-name>/predictions` | Output directory for `.npy` files |
| `--patch-size` | `256` | Spatial crop size |
| `--max-samples` | `0` (all) | Limit inference to N samples |

Each output file is named `pred_<core_id>.npy` and contains a `float32` array of shape `[4, H, W]`:
- Channel 0: Building coverage (0–1)
- Channel 1: Vegetation coverage (0–1)
- Channel 2: Water coverage (0–1)
- Channel 3: Normalized surface height in meters