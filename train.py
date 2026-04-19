import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import PixelEmbeddingDataset, LatentTokenDataset, find_file_pairs, HEIGHT_NORM_CONSTANT
from core.losses import ImprovedCompositeLoss
from predict import get_prediction_dataset, run_inference, load_model, build_zip



def save_experiment_config(*, params_dict, config_log_path):
    """Logs all hyperparameters to a text file in the experiment folder."""
    with open(config_log_path, "w") as f:
        for key, value in params_dict.items():
            f.write(f"{key}: {value}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train emb2heights baseline models")
    parser.add_argument("--model-type", type=str, default="lightunet")
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


def run_training_loop(
        *,
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        device,
        epochs,
        best_model_path):
    train_losses, val_losses = [], []
    train_mae_losses, train_ssim_losses, train_grad_losses, train_tversky_losses = [], [], [], []

    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_samples_seen = 0
        train_components = torch.zeros(4).to(device)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
        for imgs, targets in train_pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)

            loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
            loss.backward()

            # NEW: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            bs = imgs.size(0)
            train_components[0] += l_mae * bs
            train_components[1] += l_ssim * bs
            train_components[2] += l_grad * bs
            train_components[3] += l_tversky * bs
            train_samples_seen += imgs.size(0)
            train_avg = running_loss / max(1, train_samples_seen)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_avg:.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        train_epoch_comp = train_components / len(train_loader)
        train_mae_losses.append(train_epoch_comp[0].item())
        train_ssim_losses.append(train_epoch_comp[1].item())
        train_grad_losses.append(train_epoch_comp[2].item())
        train_tversky_losses.append(train_epoch_comp[3].item())

        # --- VALIDATION LOOP ---
        model.eval()
        val_running_loss = 0.0
        val_components = torch.zeros(4).to(device)
        val_samples_seen = 0
        val_mae_losses, val_ssim_losses, val_grad_losses, val_tversky_losses = [], [], [], []
        best_val_loss = float('inf')

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False)
            for imgs, targets in val_pbar:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)

                loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
                val_running_loss += loss.item() * imgs.size(0)

                bs = imgs.size(0)
                val_components[0] += l_mae * bs
                val_components[1] += l_ssim * bs
                val_components[2] += l_grad * bs
                val_components[3] += l_tversky * bs
                val_samples_seen += bs
                val_avg_live = val_running_loss / max(1, val_samples_seen)
                val_pbar.set_postfix(avg=f"{val_avg_live:.4f}")

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_comp = val_components / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        val_mae_losses.append(epoch_comp[0].item())
        val_ssim_losses.append(epoch_comp[1].item())
        val_grad_losses.append(epoch_comp[2].item())
        val_tversky_losses.append(epoch_comp[3].item())

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   >> Model Saved! (New Best Val Loss: {best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        print(f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} |"
              f" SSIM:{epoch_comp[1]:.3f} |"
              f" Grad:{epoch_comp[2]:.3f} |"
              f" Tversky:{epoch_comp[3]:.3f}")
                    
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_mae_losses": train_mae_losses,
        "train_ssim_losses": train_ssim_losses,
        "train_grad_losses": train_grad_losses,
        "train_tversky_losses": train_tversky_losses,
        "val_mae_losses": val_mae_losses,
        "val_ssim_losses": val_ssim_losses,
        "val_grad_losses": val_grad_losses,
        "val_tversky_losses": val_tversky_losses
    }


def visualize_results(
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

def generate_plots(
        *,
        train_losses, 
        val_losses, 
        train_mae_losses, 
        val_mae_losses, 
        train_ssim_losses, 
        val_ssim_losses, 
        train_grad_losses, 
        val_grad_losses, 
        train_tversky_losses, 
        val_tversky_losses,
        experiment_name,
        exp_dir):
    
    combined_loss_output_path = os.path.join(exp_dir, "loss_curve.png")
    component_loss_output_path = os.path.join(exp_dir, "component_losses.png")

    # Plot combined loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training Loss Curve ({experiment_name})")
    plt.legend()
    plt.savefig(combined_loss_output_path)
    plt.close()

    # Plot individual loss components
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(train_mae_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_mae_losses, label='Val', linewidth=2)
    axes[0, 0].set_title('MAE Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(train_ssim_losses, label='Train', linewidth=2)
    axes[0, 1].plot(val_ssim_losses, label='Val', linewidth=2)
    axes[0, 1].set_title('SSIM Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(train_grad_losses, label='Train', linewidth=2)
    axes[1, 0].plot(val_grad_losses, label='Val', linewidth=2)
    axes[1, 0].set_title('Gradient Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(train_tversky_losses, label='Train', linewidth=2)
    axes[1, 1].plot(val_tversky_losses, label='Val', linewidth=2)
    axes[1, 1].set_title('Tversky Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Component Losses ({experiment_name})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(component_loss_output_path)
    plt.close()

def get_dataloaders(
        train_embeddings_dir, 
        train_targets_dir, 
        val_split, 
        random_seed, 
        model_type, 
        patch_size, 
        batch_size):
    all_train_pairs = find_file_pairs(train_embeddings_dir, train_targets_dir)
    if len(all_train_pairs) == 0:
        raise ValueError(
            "No training (embedding, label) pairs found. "
            f"train_embeddings_dir='{train_embeddings_dir}', "
            f"train_targets_dir='{train_targets_dir}'. "
            "Check filename conventions and directory paths."
        )
    train_pairs, val_pairs = train_test_split(
        all_train_pairs, test_size=val_split, random_state=random_seed
    )

    # In train.py:
    if model_type == "lightunet" or model_type == "pixelwise": # provided dataset have shapes 256x256x(64 or 128)
        train_ds = PixelEmbeddingDataset(train_pairs, patch_size=patch_size, is_train=True)
        val_ds = PixelEmbeddingDataset(val_pairs, patch_size=patch_size, is_train=False)
    elif model_type == "decoder_residual": # provided datasets have then shape 16x16x768
        # For the decoders (TerraMind/Thor)
        scale_factor = 16
        train_ds = LatentTokenDataset(train_pairs, patch_size=patch_size, scale_factor=scale_factor, is_train=True)
        val_ds = LatentTokenDataset(val_pairs, patch_size=patch_size, scale_factor=scale_factor, is_train=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    n_channels = train_loader.dataset[0][0].shape[0] # count of channels from the first sample in the dataset

    return train_loader, val_loader, n_channels

def set_device_and_seeds(device_str, random_seed):
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Pytorch device: ", device)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    return device

def main():
    print("Starting main() function")

    args = parse_args()

    model_type = args.model_type
    base_runs_dir = args.output_dir
    train_embeddings_dir = args.train_embeddings_dir
    train_targets_dir = args.train_targets_dir
    test_embeddings_dir = args.test_submission_embeddings_dir
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_size = args.patch_size
    epochs = args.epochs

    device = set_device_and_seeds(args.device, args.random_seed)

    lambdas = [1.0, 0.5, 0.5, 2.0]  # [MAE, SSIM, Gradient, Structure/Tversky]
    learning_rate = 2e-4
    weight_decay = 1e-4  # L2 Regularization
    val_split_fraction = 0.2
    random_seed = 42

    experiment_dir = os.path.join(base_runs_dir, experiment_name)
    predictions_dir = os.path.join(experiment_dir, args.predictions_subfolder)
    viz_output_dir = os.path.join(experiment_dir, "visualizations")
    best_model_path = os.path.join(experiment_dir, "model_best.pth")
    last_model_path = os.path.join(experiment_dir, "model_last.pth")
    config_log_path = os.path.join(experiment_dir, "training_params.txt")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)
    print(f"📁 Created experiment folder: {experiment_dir}")

    params_dict = {
        "model_type": args.model_type,
        "base_dir": args.output_dir,
        "train_embeddings_dir": args.train_embeddings_dir,
        "train_targets_dir": args.train_targets_dir,
        "test_embeddings_dir": args.test_submission_embeddings_dir,
        "train_val_split": val_split_fraction,
        "predictions_subfolder": args.predictions_subfolder,
        "experiment_name": args.experiment_name,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "epochs": args.epochs,
        "device": args.device,
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
    train_loader, val_loader, n_channels = get_dataloaders(
        train_embeddings_dir=train_embeddings_dir, 
        train_targets_dir=train_targets_dir, 
        val_split=val_split_fraction, 
        random_seed=random_seed, 
        model_type=model_type, 
        patch_size=patch_size, 
        batch_size=batch_size
    )

    print("--- 2. Model Init ---")
    n_classes = 4
    model, selected_model = build_model(model_type, n_channels, n_classes)
    model = model.to(device)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Aggressive Scheduler
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
    torch.save(model.state_dict(), last_model_path)

    print("--- 3. Saving & Visualizing ---")

    generate_plots(
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

    visualize_results(
        model=training_results["model"],
        dataset=val_loader.dataset, 
        device=device,
        viz_output_dir=viz_output_dir,
        num_samples=5
    )

    print("--- 4. Compute predictions for submission ---")
    if test_embeddings_dir != '' and os.path.exists(test_embeddings_dir):
        print("Generating predictions for submission...")
        test_ds = get_prediction_dataset(
            test_embeddings_dir=test_embeddings_dir, 
            patch_size=patch_size,
            model_type=model_type
        )
        best_model = load_model(
            dataset=test_ds,
            model_type=model_type,
            model_path=best_model_path,
            device=device
        )
        run_inference(best_model, test_ds, device, predictions_dir)
    
        zip_output_name = args.zip_output
        if zip_output_name:
            build_zip(predictions_dir, zip_output_name)

if __name__ == "__main__":
    main()