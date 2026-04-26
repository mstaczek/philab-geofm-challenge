import os
import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from core.model import build_model, load_model
from core.dataset import build_dataloader, find_file_pairs
from core.losses import ImprovedCompositeLoss
from core.training_utils import generate_training_metrics_plots, run_training_loop
from core.utils import get_torch_device, save_experiment_config, set_seeds, visualize_predictions
from predict import run_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Train emb2heights baseline models")
    parser.add_argument("--model-type", type=str, default="lightunet")
    parser.add_argument("--dataset-type", type=str, default="pixel",
                        help="Dataset type: 'pixel' for PixelEmbeddingDataset or 'latent' for LatentTokenDataset")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--train-embeddings-dir", type=str)
    parser.add_argument("--train-targets-dir", type=str)
    parser.add_argument("--experiment-name", type=str, default="experiment_1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    

    parser.add_argument("--test-submission-embeddings-dir", type=str, default='',
                        help="Directory containing embedding .tif files.")
    parser.add_argument("--predictions-subfolder", type=str, default="predictions",
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--zip-output", type=str, default=None, 
                        help="Zip name in submissions folder with all files from the predictions folder will be created.")

    return parser.parse_args()



def run_training(
        model_type, 
        dataset_type, 
        base_runs_dir, 
        train_embeddings_dir, 
        train_targets_dir, 
        test_embeddings_dir, 
        experiment_name, 
        batch_size,
        patch_size,
        epochs,
        zip_output_name,
        device,
        predictions_subfolder,
        output_dir,
        train_embeddings_dir_arg,
        train_targets_dir_arg,
        test_embeddings_dir_arg,
        experiment_name_arg,
        batch_size_arg,
        patch_size_arg,
        epochs_arg,
        device_arg,
        random_seed
    ):
    lambdas = [1.0, 0.5, 0.5, 2.0]  # [MAE, SSIM, Gradient, Structure/Tversky]
    learning_rate = 2e-4
    weight_decay = 1e-4  # L2 Regularization
    val_split_fraction = 0.2
    set_seeds(random_seed)

    experiment_dir = os.path.join(base_runs_dir, experiment_name)
    predictions_dir = os.path.join(experiment_dir, predictions_subfolder)
    viz_output_dir = os.path.join(experiment_dir, "visualizations")
    best_model_path = os.path.join(experiment_dir, "model_best.pth")
    last_model_path = os.path.join(experiment_dir, "model_last.pth")
    config_log_path = os.path.join(experiment_dir, "training_params.txt")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)
    print(f"📁 Created experiment folder: {experiment_dir}")

    params_dict = {
        "model_type": model_type,
        "dataset_type": dataset_type,
        "base_dir": output_dir,
        "train_embeddings_dir": train_embeddings_dir_arg,
        "train_targets_dir": train_targets_dir_arg,
        "test_embeddings_dir": test_embeddings_dir_arg,
        "train_val_split": val_split_fraction,
        "predictions_subfolder": predictions_subfolder,
        "experiment_name": experiment_name_arg,
        "batch_size": batch_size_arg,
        "patch_size": patch_size_arg,
        "epochs": epochs_arg,
        "device": device_arg,
        "composite_loss_lambdas": lambdas,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "random_seed": random_seed,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau (factor=0.5, patience=2)",
        "gradient_clipping": "max_norm=1.0"
    }

    save_experiment_config(params_dict=params_dict, config_log_path=config_log_path)

    print("--- 1. Data Setup ---")
    
    all_train_pairs = find_file_pairs(train_embeddings_dir, train_targets_dir)

    train_pairs, val_pairs = train_test_split(all_train_pairs, test_size=val_split_fraction, random_state=random_seed)

    train_loader = build_dataloader(train_pairs, dataset_type, patch_size, batch_size, is_train=True)
    val_loader = build_dataloader(val_pairs, dataset_type, patch_size, batch_size, is_train=False)

    print("--- 2. Model Init ---")
    n_channels = train_loader.dataset[0][0].shape[0] # count of channels from the first sample in the dataset
    model, selected_model = build_model(model_type=model_type, n_channels=n_channels, n_classes=4)
    model = model.to(device)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    # Optimizer, scheduler, custom loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = ImprovedCompositeLoss(lambdas=lambdas).to(device)

    print(f"Starting training on {device}...")

    training_results = run_training_loop(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device,
        epochs=epochs,
        best_model_path=best_model_path
    )
    torch.save(training_results["model"].state_dict(), last_model_path)

    print("--- 3. Saving & Visualizing ---")

    generate_training_metrics_plots(
        train_losses=training_results["train_losses"],
        val_losses=training_results["val_losses"],
        train_mae_losses=training_results["train_mae_losses"],
        val_mae_losses=training_results["val_mae_losses"],
        train_ssim_losses=training_results["train_ssim_losses"],
        val_ssim_losses=training_results["val_ssim_losses"],
        train_grad_losses=training_results["train_grad_losses"],
        val_grad_losses=training_results["val_grad_losses"],
        train_tversky_losses=training_results["train_tversky_losses"],
        val_tversky_losses=training_results["val_tversky_losses"],
        experiment_name=experiment_name,
        exp_dir=experiment_dir
    )

    best_model = load_model(
        model_type=model_type,
        model_path=best_model_path,
        n_channels=n_channels,
        device=device
    )
    visualize_predictions(
        model=best_model,
        dataset=val_loader.dataset, 
        device=device,
        viz_output_dir=viz_output_dir,
        num_samples=5
    )

    if test_embeddings_dir != '' and os.path.exists(test_embeddings_dir):
        print("--- 4. Compute predictions for submission ---")
        run_prediction(
            device=device,
            model_path=best_model_path,
            predictions_dir=predictions_dir,
            test_embeddings_dir=test_embeddings_dir,
            patch_size=patch_size,
            dataset_type=dataset_type,
            max_samples=0,
            model_type=model_type,
            zip_output_path=zip_output_name
        )  


def main():
    print("Starting main() function")

    args = parse_args()

    run_training(
        model_type=args.model_type,
        dataset_type=args.dataset_type,
        base_runs_dir=args.output_dir,
        train_embeddings_dir=args.train_embeddings_dir,
        train_targets_dir=args.train_targets_dir,
        test_embeddings_dir=args.test_submission_embeddings_dir,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        epochs=args.epochs,
        zip_output_name=args.zip_output,
        device=get_torch_device(args.device),
        predictions_subfolder=args.predictions_subfolder,
        output_dir=args.output_dir,
        train_embeddings_dir_arg=args.train_embeddings_dir,
        train_targets_dir_arg=args.train_targets_dir,
        test_embeddings_dir_arg=args.test_submission_embeddings_dir,
        experiment_name_arg=args.experiment_name,
        batch_size_arg=args.batch_size,
        patch_size_arg=args.patch_size,
        epochs_arg=args.epochs,
        device_arg=args.device,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()