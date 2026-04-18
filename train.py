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

# --- 1. EXPERIMENT TRACKING ---
EXPERIMENT_NAME = "terramid_run02/"
BASE_DIR = "./runs"
EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")

# Paths for saving models and plots
BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best.pth")
LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

# --- 2. CONFIGURATION ---
# TRAIN_EMBEDDINGS_DIR = "../../emb2heights/data/gee_emb_aligned_v2/"

TRAIN_EMBEDDINGS_DIR = "../../emb2heights/data/gee_emb_aligned_v2"
TRAIN_TARGETS_DIR = "../../emb2heights/data/patches_labels_10m/"

TEST_EMBEDDINGS_DIR = ''

BATCH_SIZE = 32
PATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4  # L2 Regularization
VAL_SPLIT = 0.2
LAMBDAS = [1.0, 0.5, 0.5, 2.0]  # [MAE, SSIM, Gradient, Structure/Tversky]
RANDOM_SEED = 42
MODEL_TYPE = "auto"  # one of: auto, lightunet, decoder_residual

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def save_experiment_config():
    """Logs all hyperparameters to a text file in the experiment folder."""
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_LOG_PATH, "w") as f:
        f.write(f"--- EXPERIMENT: {EXPERIMENT_NAME} ---\n")
        f.write(f"OUTPUT_DIR: {BASE_DIR}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"LOSS LAMBDAS: {LAMBDAS}\n")
        f.write(f"MODEL_TYPE: {MODEL_TYPE}\n")
        f.write(f"TRAIN_EMBEDDINGS_DIR: {TRAIN_EMBEDDINGS_DIR}\n")
        f.write(f"TRAIN_TARGETS_DIR: {TRAIN_TARGETS_DIR}\n")
        f.write(f"VAL_SPLIT: {VAL_SPLIT}\n")
        f.write(f"OPTIMIZER: AdamW\n")
        f.write(f"SCHEDULER: ReduceLROnPlateau (factor=0.5, patience=2)\n")
        f.write(f"GRADIENT CLIPPING: max_norm=1.0\n")
    print(f"📁 Created experiment folder: {EXP_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train emb2heights baseline models")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, choices=["auto", "lightunet", "decoder_residual"])
    parser.add_argument("--output-dir", type=str, default=BASE_DIR)
    parser.add_argument("--train-embeddings-dir", type=str, default=TRAIN_EMBEDDINGS_DIR)
    parser.add_argument("--train-targets-dir", type=str, default=TRAIN_TARGETS_DIR)
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")


    parser.add_argument("--test-submission-embeddings-dir", type=str, default=TEST_EMBEDDINGS_DIR,
                        help="Directory containing embedding .tif files.")
    parser.add_argument("--predictions-subfolder", type=str, default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--zip-output", type=str, default=None, 
                        help="Zip name in submissions folder with all files from the predictions folder will be created.")

    return parser.parse_args()


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler):
    train_losses, val_losses = [], []
    train_mae_losses, train_ssim_losses, train_grad_losses, train_tversky_losses = [], [], [], []

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_samples_seen = 0
        train_components = torch.zeros(4).to(DEVICE)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False)
        for imgs, targets in train_pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
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
        val_components = torch.zeros(4).to(DEVICE)
        val_samples_seen = 0
        val_mae_losses, val_ssim_losses, val_grad_losses, val_tversky_losses = [], [], [], []
        best_val_loss = float('inf')

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=False)
            for imgs, targets in val_pbar:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
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
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   >> Model Saved! (New Best Val Loss: {best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        print(f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} |"
              " SSIM:{epoch_comp[1]:.3f} |"
              " Grad:{epoch_comp[2]:.3f} |"
              " Tversky:{epoch_comp[3]:.3f}")
        return {
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


def visualize_results(model, dataset, num_samples=3):
    """Generates sample visualizations from the dataset."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            target_batch = target_tensor.unsqueeze(0).to(DEVICE)

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
            plt.savefig(os.path.join(VIZ_OUTPUT_DIR, f"viz_{i}.png"))
            plt.close()

def generate_plots(train_losses, val_losses, train_mae_losses, val_mae_losses, train_ssim_losses, val_ssim_losses, train_grad_losses, val_grad_losses, train_tversky_losses, val_tversky_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training Loss Curve ({EXPERIMENT_NAME})")
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
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
    
    plt.suptitle(f"Component Losses ({EXPERIMENT_NAME})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    component_loss_path = os.path.join(EXP_DIR, "component_losses.png")
    plt.savefig(component_loss_path)
    plt.close()


def main():
    print("Starting main() function")
    global BASE_DIR, EXPERIMENT_NAME, EXP_DIR, VIZ_OUTPUT_DIR
    global BEST_MODEL_PATH, LAST_MODEL_PATH, LOSS_CURVE_PATH, CONFIG_LOG_PATH
    global TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR, TEST_EMBEDDINGS_DIR
    global MODEL_TYPE, EPOCHS, BATCH_SIZE, PATCH_SIZE, DEVICE

    args = parse_args()
    MODEL_TYPE = args.model_type
    BASE_DIR = args.output_dir
    TRAIN_EMBEDDINGS_DIR = args.train_embeddings_dir
    TRAIN_TARGETS_DIR = args.train_targets_dir
    TEST_EMBEDDINGS_DIR = args.test_submission_embeddings_dir
    EXPERIMENT_NAME = args.experiment_name
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    EPOCHS = args.epochs
    DEVICE = torch.device(args.device)

    EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
    predictions_dir = os.path.join(EXP_DIR, "predictions" if args.predictions_subfolder is None else args.predictions_subfolder)
    VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")
    BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best.pth")
    LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
    LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
    CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

    save_experiment_config()

    print("--- 1. Data Setup ---")
    all_train_pairs = find_file_pairs(TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR)
    if len(all_train_pairs) == 0:
        raise ValueError(
            "No training (embedding, label) pairs found. "
            f"train_embeddings_dir='{TRAIN_EMBEDDINGS_DIR}', "
            f"train_targets_dir='{TRAIN_TARGETS_DIR}'. "
            "Check filename conventions and directory paths."
        )
    train_pairs, val_pairs = train_test_split(
        all_train_pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    # In train.py:
    if MODEL_TYPE == "lightunet": # provided dataset have shapes 256x256x(64 or 128)
        train_ds = PixelEmbeddingDataset(train_pairs, patch_size=PATCH_SIZE, is_train=True)
        val_ds = PixelEmbeddingDataset(val_pairs, patch_size=PATCH_SIZE, is_train=False)
    else:
        # For the decoders (TerraMind/Thor) # provided datasets have then shape 16x16x768
        train_ds = LatentTokenDataset(train_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=True)
        val_ds = LatentTokenDataset(val_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    sample_img, _ = train_ds[0]
    n_channels, n_classes = sample_img.shape[0], 4

    print("--- 2. Model Init ---")
    model, selected_model = build_model(MODEL_TYPE, n_channels, n_classes)
    model = model.to(DEVICE)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    # NEW: AdamW with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Aggressive Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = ImprovedCompositeLoss(lambdas=LAMBDAS).to(DEVICE)

    print(f"Starting training on {DEVICE}...")

    training_results = run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler)

    print("--- 3. Saving & Visualizing ---")
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    generate_plots(train_losses=training_results["train_losses"],
                val_losses=training_results["val_losses"],
                train_mae_losses=training_results["train_mae_losses"],
                val_mae_losses=training_results["val_mae_losses"],
                train_ssim_losses=training_results["train_ssim_losses"],
                val_ssim_losses=training_results["val_ssim_losses"],
                train_grad_losses=training_results["train_grad_losses"],
                val_grad_losses=training_results["val_grad_losses"],
                train_tversky_losses=training_results["train_tversky_losses"],
                val_tversky_losses=training_results["val_tversky_losses"])

    visualize_results(model, val_ds, num_samples=5)

    print("--- 4. Compute predictions for submission ---")
    if TEST_EMBEDDINGS_DIR != '' and os.path.exists(TEST_EMBEDDINGS_DIR):
        print("Generating predictions for submission...")
        test_ds = get_prediction_dataset(
            predictions_dir=predictions_dir,
            test_embeddings_dir=TEST_EMBEDDINGS_DIR, 
            patch_size=PATCH_SIZE,
            model_type=MODEL_TYPE)
        best_model = load_model(
            dataset=test_ds,
            model_type=MODEL_TYPE,
            model_path=BEST_MODEL_PATH
        )
        run_inference(best_model, test_ds, DEVICE, predictions_dir)
    
        zip_output_name = args.zip_output
        if zip_output_name:
            build_zip(predictions_dir, zip_output_name)

if __name__ == "__main__":
    main()