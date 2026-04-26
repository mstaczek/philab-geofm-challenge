import os
import argparse
from core.training_utils import run_prediction
from core.utils import get_torch_device

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


def main():
    args = parse_args()
    
    exp_dir = os.path.join(args.base_dir, args.experiment_name)

    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")

    run_prediction(
        device=get_torch_device(args.device),
        model_path=model_path,
        predictions_dir=predictions_dir,
        test_embeddings_dir=args.test_embeddings_dir,
        patch_size=args.patch_size,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        model_type=args.model_type,
        zip_output_path=args.zip_output
    )


if __name__ == "__main__":
    main()