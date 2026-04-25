# Solution for embed2heights Challenge - Reaching New Heights with GeoFM Embeddings

Team: 2theMoon 

Team members:
- Agata Kaczmarek
- Mateusz Stączek


## Challenge TLDR (simplified)

Goal: predict 4 channel outputs given aerial photos embeddings:
- 3x image segmentation (buildings, vegetation, water, scale 0-1),
- 1x height regression (relative, scale 0-1, already normalized labels),

Input: 
- embeddings of images of earth computed with different models. 
    - 4 folders with embeddings of shapes 16x16x768,
    - 2 folders with embeddings of shapes 256x256x64 (or x128),
- correct labels - 256x256x4, each output channel represents a different goal.

Output:
- 256x256x4 matrices, with channels representing predictions for different tasks (segmentation/regression).


## Links

- Competition: https://platform.ai4eo.eu/geoai/data
- Dataset: https://www.eotdl.com/datasets/embed2heights
- Sample repo code: https://github.com/VMarsocci/emb2heights-baselines/tree/main
- Drive: https://drive.google.com/drive/folders/1yxXF3bJ1C6zCsZ1XLQWnNfkf98VI-21f

### Also see [ALL DESCRIPTIONS IN ONE PLACE](readme_copies.md)


# Experiments

| Name                                             | DATA         | MODEL            | final_score | iou_build | iou_veg | iou_water | rmse_h_build | rmse_h_veg |
|--------------------------------------------------|--------------|------------------|-------------|-----------|---------|-----------|--------------|------------|
| 1) Baseline terramind_s1 50 epochs               | terramind_s1 | decoder_residual | 0.1380      | 0.1175    | 0.5988  | 0.1253    | 3.2991       | 6.8749     |
| 2) Baseline alphaearth 50 epochs                 | alphaearth   | lightunet        | 0.3558      | 0.3270    | 0.7761  | 0.3957    | 2.3230       | 3.9531     |
| 3) Baseline alphaearth 50 epochs pixelwise model | alphaearth   | pixelwise        | 0.1898      | 0.1655    | 0.5696  | 0.2146    | 2.7266       | 4.7987     |

### Models

- decoder_residual: 
    - basic decoder model: (counts of channels) 768 -> 256 -> 128 -> 64 -> 32 -> 16 -> 4,
    - expects input 16x16, then scales it up so has poor final output resolution,
    - provided by challenge organizers in a sample solution,
- lightunet:
    - simple unet model: (counts of channels):
        - down: 64/128 -> 32 -> 64 -> 128 -> 256,
        - up: 256 -> 128 + concat -> 64 + concat -> 32 + concat -> 4,
    - provided by challenge organizers in a sample solution,
- pixelwise:
    - 2 simple convolution layers with kernel size 1 and ReLU between,
        - equivalent to a simple MLP applied to each pixel independently,
        - channels counts: 128/64 -> 16 -> 4,
    - no spatial awareness / no neighbor pixels taken into account,


## 1) Baseline terramind_s1 50 epochs

| Component Losses                                        | Loss Curve                                        |
| ------------------------------------------------------- | ------------------------------------------------- |
| ![](runs/terramind_s1_v3_50epochs/component_losses.png) | ![](runs/terramind_s1_v3_50epochs/loss_curve.png) |


![](runs/terramind_s1_v3_50epochs/visualizations/viz_4.png)

## 2) Baseline alphaearth 50 epochs

| Component Losses                                          | Loss Curve                                          |
| --------------------------------------------------------- | --------------------------------------------------- |
| ![](runs/alphaearth_emb_v3_50epochs/component_losses.png) | ![](runs/alphaearth_emb_v3_50epochs/loss_curve.png) |

![](runs/alphaearth_emb_v3_50epochs/visualizations/viz_4.png)

## 3) Baseline alphaearth 50 epochs pixelwise model

| Component Losses                                                 | Loss Curve                                                 |
| ---------------------------------------------------------------- | ---------------------------------------------------------- |
| ![](runs/alphaearth_emb_pixelwise_50epochs/component_losses.png) | ![](runs/alphaearth_emb_pixelwise_50epochs/loss_curve.png) |

![](runs/alphaearth_emb_pixelwise_50epochs/visualizations/viz_4.png)


---

# Local setup

(Tested on Windows with Python 3.14.2)

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

And install requirements:
```bash
pip install -r requirements.txt
```

## Download data

Then, go to data folder and run to download dataset metadata:
```bash
cd data
eotdl datasets get embed2heights --path . --version 1
```

Check, if the folder `data/embed2heights` appeared. 

After ensuring the data would be downloaded into the correct directory, download 100+ GB with:
```bash
eotdl datasets get embed2heights --path . --version 1 --assets --verbose --force 
```

## Repo structure

`readmes_copied.md`:
- readme from sample solution repo,
- readme about the dataset.

Folders:
- core/ - source code for torch datasets, models and custom loss functions,
- data/ - place for the dataset,
- runs/ - experiments will save their artefacts here (predictions, plots, models, predictions for submission),
- submission/ - ZIPs with model predictions on test set will appear here. 

Scripts:
- train.py - run training, save results locally, and prepare a ZIP with predictions on the test dataset,
- predict.py - load a model, compute predictions and save them.

Notebooks:
- starter_pack-embed2heights.ipynb - copied from baseline solution, contains sample visualizations and creating a ZIP,
- visualize_predictions.ipynb - contains sample prediction visualizations based on code copied from sample solution notebook.

## Run training

```bash
python train.py --help
```

Examples:

```bash
python train.py \
    --model-type lightunet \
    --output-dir runs \
    --train-embeddings-dir data/embed2heights/data/train/alphaearth_emb \
    --train-targets-dir data/embed2heights/data/train/labels \
    --experiment-name alphaearth_emb_v3_1epochs \
    --epochs 1 \
    --batch-size 4 \
    --patch-size 256 \
    --device cuda \
    --test-submission-embeddings-dir data/embed2heights/data/test/alphaearth_test_emb \
    --predictions-subfolder predictions_submission_1 \
    --zip-output submission_1_epochs.zip
```

or

```bash
python train.py \
    --model-type lightunet \
    --output-dir runs \
    --train-embeddings-dir data/embed2heights/data/train/tessera_emb \
    --train-targets-dir data/embed2heights/data/train/labels \
    --experiment-name tessera_emb_50epochs \
    --epochs 50 \
    --batch-size 2 \
    --patch-size 256 \
    --device cuda \
    --test-submission-embeddings-dir data/embed2heights/data/test/tessera_test_emb \
    --predictions-subfolder predictions \
    --zip-output submission_test_50_epochs.zip
```

## Run predict

Save a few train predictions
```bash
python predict.py \
    --experiment-name alphaearth_emb_pixelwise_50epochs \
    --base-dir runs \
    --model-type pixelwise \
    --model-path runs/alphaearth_emb_pixelwise_50epochs/model_best.pth \
    --test-embeddings-dir data/embed2heights/data/test/alphaearth_test_emb \
    --predictions-dir runs/alphaearth_emb_pixelwise_50epochs/predictions_fixed \
    --patch-size 256 \
    --device cuda \
    --zip-output submission_50_epochs_alphaearth_pixelwise_fixed.zip
```

Add `--max-samples 5` to save just 5 samples.

Creating of ZIP is optional - `--zip-output` can be ommitted.

## Send submission

Filenames pattern must match `3123_AB_2022.npy` where `3123_AB_2022` comes from the source embedding filename, and there is no `pred_` or `label_` prefix, nor any extra suffix such as `_merged` or `_embedding`.

Create a zip with a `submissions/` folder with all 946 npy predictions by adding a --zip-output arg to `predict.py` script. (Alternative: use last cell in `starter_pack-embed2heights.ipynb.`)

Upload the `submission.zip` file to https://platform.ai4eo.eu/geoai/submissions (can be uploaded every 12h).
