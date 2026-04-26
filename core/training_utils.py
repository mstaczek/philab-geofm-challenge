
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from core.dataset import HEIGHT_NORM_CONSTANT, _normalize_core_id, build_dataset, find_file_pairs
from core.model import load_model
from core.utils import build_zip


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



def generate_training_metrics_plots(
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

def run_prediction(
        device,
        model_path,
        predictions_dir,
        test_embeddings_dir,
        patch_size,
        dataset_type,
        max_samples,
        model_type,
        zip_output_path
    ):
    os.makedirs(predictions_dir, exist_ok=True)

    all_file_pairs = find_file_pairs(emb_dir=test_embeddings_dir, tar_dir=None)

    test_ds = build_dataset(
        pairs=all_file_pairs,
        dataset_type=dataset_type,
        patch_size=patch_size,
        is_train=False,
        max_samples=max_samples
    )
    n_channels = test_ds[0][0].shape[0] # count of channels from the first sample in the dataset

    model = load_model(
        model_type=model_type,
        model_path=model_path,
        n_channels=n_channels,
        device=device
    )
    print("Generating predictions...")
    run_inference(model, test_ds, device, predictions_dir)

    if zip_output_path:
        print("Compressing predictions to ZIP...")
        build_zip(predictions_dir, zip_output_path)

