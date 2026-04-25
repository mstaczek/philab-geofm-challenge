import os
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
import zipfile


from core.model import build_model, load_model
from core.dataset import PixelEmbeddingDataset, LatentTokenDataset, find_file_pairs, _normalize_core_id, \
    HEIGHT_NORM_CONSTANT
from core.utils import build_zip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run inference, saving predictions as .npy files."
    )
    parser.add_argument("--experiment-name", type=str, default="experiment_name")
    parser.add_argument("--base-dir", type=str, default="./runs",
                        help="Root directory containing experiment subfolders.")
    parser.add_argument("--model-type", type=str,
                        help="Model architecture used during training.")
    parser.add_argument("--dataset-type", type=str,
                        help="Dataset type: 'pixel' for PixelEmbeddingDataset or 'latent' for LatentTokenDataset")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the .pth checkpoint. Defaults to <base-dir>/<experiment-name>/model_best.pth.")
    parser.add_argument("--test-embeddings-dir", type=str, required=True,
                        help="Directory containing embedding .tif files.")
    parser.add_argument("--test-targets-dir", type=str, default=None,
                        help="Optional labels directory. If omitted, inference only uses embeddings.")
    parser.add_argument("--predictions-dir", type=str, default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit inference to N samples (0 = all).")
    parser.add_argument("--device", type=str, default="cpu", help="What torch device to use.")
    parser.add_argument("--zip-output", type=str, default=None, 
                        help="Zip name in submissions folder with all files from the predictions folder will be created.")
    return parser.parse_args()

def get_prediction_dataset(test_embeddings_dir, patch_size, dataset_type, max_samples=0, test_targets_dir=None):
    print(f"Loading file pairs from embeddings: {test_embeddings_dir}")
    pairs = find_file_pairs(test_embeddings_dir, test_targets_dir)
    if not pairs:
        raise RuntimeError("No matching file pairs found. Check --test-embeddings-dir and --test-targets-dir.")
    if max_samples > 0:
        pairs = pairs[:max_samples]

    # --- Dataset ---
    if dataset_type == "pixel":
        test_ds = PixelEmbeddingDataset(pairs, patch_size=patch_size, is_train=False)
    elif dataset_type == "latent":
        test_ds = LatentTokenDataset(pairs, patch_size=patch_size, scale_factor=16, is_train=False)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'pixel' or 'latent'.")
    
    return test_ds

def run_inference(model, dataset, device, predictions_dir):
    """
    Run inference on dataset and save predictions.
    """
    os.makedirs(predictions_dir, exist_ok=True)
    model.eval()

    print(f"Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            img_tensor, _ = dataset[i]
            img_batch = img_tensor.unsqueeze(0).to(device)

            output_batch = model(img_batch)
            pred_np = output_batch.squeeze().cpu().numpy().astype(np.float32)

            # Denormalize height channel
            pred_np[3] = pred_np[3] * HEIGHT_NORM_CONSTANT

            # Clip outputs
            pred_np[[0, 1, 2], :, :] = np.clip(pred_np[[0, 1, 2], :, :], 0, 1)
            pred_np[[3], :, :] = np.clip(pred_np[[3], :, :], 0, 1000)

            emb_path, _ = dataset.file_pairs[i]
            core_id = _normalize_core_id(emb_path, strip_year_suffix=False)

            save_path = os.path.join(predictions_dir, f"{core_id}.npy")
            np.save(save_path, pred_np)

    print(f"Predictions saved to: {predictions_dir}")
    print(f"Output shape per file: {pred_np.shape} [building%, veg%, water%, height_m]")


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")

    print("DEVICE: ", device)
    os.makedirs(predictions_dir, exist_ok=True)

    # --- Dataset ---
    test_ds = get_prediction_dataset(
        test_embeddings_dir=args.test_embeddings_dir,
        patch_size=args.patch_size,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        test_targets_dir=args.test_targets_dir
    )

    # --- Load model ---
    model = load_model(
        dataset=test_ds,
        model_type=args.model_type,
        model_path=model_path,
        device=device
    )

    # --- Inference ---
    run_inference(model, test_ds, device, predictions_dir)

    # --- Zip (optional) ---
    zip_output_path = args.zip_output
    if zip_output_path:
        build_zip(predictions_dir, zip_output_path)



if __name__ == "__main__":
    main()