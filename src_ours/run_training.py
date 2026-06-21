
import copy
import os
import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.dataset import _normalize_core_id
from src_ours.multi_folder_dataset import MultiFolderNpyDataset

from core.losses import ImprovedCompositeLoss
from core.training_utils import run_training_loop, generate_training_metrics_plots
from core.utils import get_torch_device, save_experiment_config, set_seeds

import random

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from src_ours.constants import HEIGHT_NORM_CONSTANT, TEST_DATASET_FOLDERS, TRAIN_DATASET_FOLDERS
from src_ours.models import PixelWiseBaseline

def resolve_folders(dataset_keys, mapping):
    folders = []

    for key in dataset_keys:
        if key not in mapping:
            raise ValueError(f"Unknown dataset: {key}")

        folders.append(mapping[key])

    return folders

def parse_dataset_keys(arg):
    return [x.strip() for x in arg.split(",") if x.strip()]

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
    model.to(device)
    best_val_loss = float('inf')
    
    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_samples_seen = 0
        train_components = torch.zeros(4).to(device)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
        for imgs, targets in train_pbar:
            imgs = {k: v.to(device) for k, v in imgs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            bs = next(iter(imgs.values())).size(0)
            running_loss += loss.item() * bs
            train_components[0] += l_mae * bs
            train_components[1] += l_ssim * bs
            train_components[2] += l_grad * bs
            train_components[3] += l_tversky * bs
            train_samples_seen += bs
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

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False)
            for imgs, targets in val_pbar:
                imgs = {k: v.to(device) for k, v in imgs.items()}
                targets = targets.to(device)

                outputs = model(imgs)

                loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
                bs = next(iter(imgs.values())).size(0)
                val_running_loss += loss.item() * bs
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
            best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_model_state_dict, best_model_path)
            print(f"   >> Model Saved! (New Best Val Loss: {best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        print(f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} |"
              f" SSIM:{epoch_comp[1]:.3f} |"
              f" Grad:{epoch_comp[2]:.3f} |"
              f" Tversky:{epoch_comp[3]:.3f}")
                    
    return {
        "model": model,
        "best_model_state_dict": best_model_state_dict,
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


def visualize_predictions(
        *,
        model, 
        dataset, 
        device,
        viz_output_dir,
        num_samples=3):
    """Generates sample visualizations from the dataset."""
    os.makedirs(viz_output_dir, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_dict, target_tensor = dataset[idx]
            img_dict_batch = {folder: image.unsqueeze(0).to(device) for folder, image in img_dict.items()}
            target_batch = target_tensor.unsqueeze(0).to(device)

            output_batch = model(img_dict_batch)

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



# ---------------- ARGPARSE ----------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--experiment-name", type=str, default="exp")
    parser.add_argument("--output-dir", type=str, default="./runs")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    
    parser.add_argument("--datasets",type=str,required=True,
        help=(
            "Comma-separated dataset names. "
            f"Available: {', '.join(TRAIN_DATASET_FOLDERS)}"
        ),
    )
    parser.add_argument("--save-zip", type=bool, default=True)
    parser.add_argument("--zip-output-name", type=str, default="output_predictions.zip")
    
    return parser.parse_args()

import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from src_ours.constants import HEIGHT_NORM_CONSTANT
from core.utils import build_zip

def run_prediction(*, model, test_loader, device, predictions_dir, zip_output_name=None):
    """Generate predictions for all test samples and optionally zip them."""
    os.makedirs(predictions_dir, exist_ok=True)
    model.eval()
    dataset = test_loader.dataset
    sample_index = 0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Predicting"):
            imgs = {k: v.to(device) for k, v in imgs.items()}
            outputs = model(imgs)
            preds = outputs.cpu().numpy().astype(np.float32)
            batch_size = preds.shape[0]
            for b in range(batch_size):
                pred = preds[b]
                pred[3] = pred[3] * HEIGHT_NORM_CONSTANT
                pred[0:3] = np.clip(pred[0:3], 0.0, 1.0)
                pred[3] = np.clip(pred[3], 0.0, 1000.0)
                sample_id = dataset.sample_ids[sample_index]
                output_filename = _normalize_core_id(sample_id, strip_year_suffix=False)
                save_path = Path(predictions_dir) / f"{output_filename}.npy"
                np.save(save_path, pred)
                sample_index += 1
    print(f"Saved predictions to: {predictions_dir}")
    if zip_output_name is None:
        zip_output_name = "output_predictions.zip"
    build_zip(predictions_dir, zip_output_name)

# ---------------- TRAIN ----------------
def run_training(
    data_root,
    experiment_name,
    output_dir,
    batch_size,
    epochs,
    device,
    random_seed,
    dataset_names,
    model,
    save_zip,
    zip_output_name
):
    set_seeds(random_seed)

    exp_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    best_path = os.path.join(exp_dir, "best.pth")
    last_path = os.path.join(exp_dir, "last.pth")
    viz_output_dir = os.path.join(exp_dir, "visualizations")

    print(f"📁 Experiment: {exp_dir}")

    # ---------------- DATASET ----------------

    train_folders = resolve_folders(dataset_names, TRAIN_DATASET_FOLDERS)
    test_folders = resolve_folders(dataset_names, TEST_DATASET_FOLDERS)

    print(f"Training folders: {train_folders}.")

    full_dataset = MultiFolderNpyDataset(
        root=data_root,
        split="train",
        input_folders=train_folders,
    )

    test_dataset = MultiFolderNpyDataset(
        root=data_root,
        split="test",
        input_folders=test_folders,
    )

    idx = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=random_seed)

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---------------- OPTIM ----------------
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = ImprovedCompositeLoss(lambdas=[1.0, 0.5, 0.5, 2.0]).to(device)

    # ---------------- TRAIN ----------------
    results = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        best_model_path=best_path,
    )

    torch.save(results["model"].state_dict(), last_path)

    generate_training_metrics_plots(
        train_losses=results["train_losses"],
        val_losses=results["val_losses"],
        train_mae_losses=results["train_mae_losses"],
        val_mae_losses=results["val_mae_losses"],
        train_ssim_losses=results["train_ssim_losses"],
        val_ssim_losses=results["val_ssim_losses"],
        train_grad_losses=results["train_grad_losses"],
        val_grad_losses=results["val_grad_losses"],
        train_tversky_losses=results["train_tversky_losses"],
        val_tversky_losses=results["val_tversky_losses"],
        experiment_name=experiment_name,
        exp_dir=exp_dir,
    )


    # to delete?
    # best_model = model.load_state_dict(results["best_model_state_dict"])
    
    model.load_state_dict(results["best_model_state_dict"])
    best_model = model


    visualize_predictions(
        model=best_model,
        dataset=val_loader.dataset, 
        device=device,
        viz_output_dir=viz_output_dir,
        num_samples=4
    )

    if save_zip:
        predictions_output_dir = os.path.join(exp_dir, "predictions")
        run_prediction(
            model=best_model,
            test_loader=test_loader,
            device=device,
            predictions_dir=predictions_output_dir,
            zip_output_name=zip_output_name
        )

# ---------------- MAIN ----------------
def main():
    args = parse_args()

    model = PixelWiseBaseline(
        in_channels=64,
        out_channels=4,
        input_key="alphaearth"
    )

    run_training(
        data_root=args.data_root,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=get_torch_device(args.device),
        random_seed=args.random_seed,
        dataset_names=parse_dataset_keys(args.datasets),
        model=model,
        save_zip=args.save_zip,
        zip_output_name=args.zip_output_name
    )


if __name__ == "__main__":
    main()