import os
from pathlib import Path
import random
import zipfile

from matplotlib import pyplot as plt
import numpy as np
import torch

from core.dataset import HEIGHT_NORM_CONSTANT


def save_experiment_config(*, params_dict, config_log_path):
    """Logs all hyperparameters to a text file in the experiment folder."""
    with open(config_log_path, "w") as f:
        for key, value in params_dict.items():
            f.write(f"{key}: {value}\n")



def visualize_predictions(
        *,
        model, 
        dataset, 
        device,
        viz_output_dir,
        num_samples=3):
    """Generates sample visualizations from the dataset."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(device)
            target_batch = target_tensor.unsqueeze(0).to(device)

            output_batch = model(input_batch)

            pred = output_batch.squeeze().cpu().numpy()
            true = target_batch.squeeze().cpu().numpy()

            # UN-NORMALIZE HEIGHT FOR VISUALIZATION
            pred[3] = pred[3] * HEIGHT_NORM_CONSTANT
            true[3] = true[3] * HEIGHT_NORM_CONSTANT

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, HEIGHT_NORM_CONSTANT)
                axes[0, c].imshow(true[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, c].set_title(f"True {target_names[c]}")
                axes[0, c].axis('off')

                axes[1, c].imshow(pred[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, c].set_title(f"Pred {target_names[c]}")
                axes[1, c].axis('off')

            plt.suptitle(f"{model.__class__.__name__} Prediction (Sample {i})")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_output_dir, f"viz_{i}.png"))
            plt.close()


def build_zip(predictions_dir, zip_output_name):
    predictions_dir = Path(predictions_dir)
    submission_dir = Path("submission")
    submission_dir.mkdir(exist_ok=True)

    submission_zip_path = submission_dir / zip_output_name

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Prediction folder not found: {predictions_dir}")

    prediction_files = sorted(predictions_dir.glob("*.npy"))
    if not prediction_files:
        raise FileNotFoundError(f"No .npy files found in: {predictions_dir}")

    print(f"Found {len(prediction_files)} files to zip.")

    # Sanity check: first file must be [4, H, W].
    sample = np.load(prediction_files[0])
    if sample.ndim != 3 or sample.shape[0] != 4:
        raise ValueError(
            f"Invalid prediction shape {sample.shape} in {prediction_files[0].name}. "
            "Expected [4, H, W]."
        )

    # Build submission zip with required internal folder structure.
    with zipfile.ZipFile(submission_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for npy_file in prediction_files:
            zf.write(npy_file, arcname=f"predictions/{npy_file.name}")

    print("Submission ready:")
    print(f"  source folder: {predictions_dir}")
    print(f"  files zipped: {len(prediction_files)}")
    print(f"  sample file: {prediction_files[0].name}")
    print(f"  sample shape: {sample.shape}")
    print(f"  sample dtype: {sample.dtype}")
    print(f"  zip file: {zip_output_name}")
