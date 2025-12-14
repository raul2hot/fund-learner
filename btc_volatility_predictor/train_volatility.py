"""Training script for BTC Volatility Prediction with SPH-Net"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import Config
from models import SPHNet
from data.dataset import create_dataloaders


class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood loss for Gaussian with predicted variance."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_mean, pred_var, target):
        """
        Args:
            pred_mean: Predicted mean [batch, horizon]
            pred_var: Predicted variance [batch, horizon] (must be positive)
            target: Ground truth [batch, horizon]
        """
        # Clamp variance for numerical stability
        pred_var = pred_var.clamp(min=1e-6)

        # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
        nll = 0.5 * (torch.log(pred_var) + (target - pred_mean)**2 / pred_var)
        return nll.mean()


def train_epoch(model, dataloader, optimizer, criterion_reg, criterion_cls,
                criterion_nll, config, device, scaler=None):
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    total_mse = 0
    total_bce = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].to(device).unsqueeze(-1)
        target_dir = batch['target_direction'].to(device).unsqueeze(-1)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(prices, features)

            # Compute losses
            loss_mse = criterion_reg(outputs['volatility_pred'], target_vol)
            loss_bce = criterion_cls(outputs['direction_pred'], target_dir)

            # Combined loss
            loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce

            # Add uncertainty loss if available
            if 'uncertainty' in outputs and config.use_uncertainty:
                loss_nll = criterion_nll(
                    outputs['volatility_pred'],
                    outputs['uncertainty'],
                    target_vol
                )
                loss += config.gamma_uncertainty * loss_nll

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_bce += loss_bce.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'bce': total_bce / n_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion_reg, criterion_cls, criterion_nll,
             config, device, target_scaler=None):
    """Validation with metrics calculation."""
    model.eval()

    total_loss = 0
    total_mse = 0
    all_preds = []
    all_targets = []
    all_pred_dirs = []
    all_target_dirs = []
    n_batches = 0

    for batch in dataloader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].to(device).unsqueeze(-1)
        target_dir = batch['target_direction'].to(device).unsqueeze(-1)

        outputs = model(prices, features)

        loss_mse = criterion_reg(outputs['volatility_pred'], target_vol)
        loss_bce = criterion_cls(outputs['direction_pred'], target_dir)
        loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce

        total_loss += loss.item()
        total_mse += loss_mse.item()

        all_preds.append(outputs['volatility_pred'].cpu())
        all_targets.append(target_vol.cpu())
        all_pred_dirs.append(torch.sigmoid(outputs['direction_pred']).cpu())
        all_target_dirs.append(target_dir.cpu())

        n_batches += 1

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_pred_dirs = torch.cat(all_pred_dirs, dim=0).numpy()
    all_target_dirs = torch.cat(all_target_dirs, dim=0).numpy()

    # Calculate direction accuracy
    dir_acc = ((all_pred_dirs > 0.5) == all_target_dirs).mean()

    # Inverse transform for interpretable metrics
    if target_scaler is not None:
        all_preds_orig = target_scaler.inverse_transform(all_preds)
        all_targets_orig = target_scaler.inverse_transform(all_targets)

        # RMSE in original scale
        rmse_orig = np.sqrt(np.mean((all_preds_orig - all_targets_orig)**2))
        mae_orig = np.mean(np.abs(all_preds_orig - all_targets_orig))

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((all_targets_orig - all_preds_orig) /
                             (all_targets_orig + 1e-8))) * 100
    else:
        rmse_orig = np.sqrt(np.mean((all_preds - all_targets)**2))
        mae_orig = np.mean(np.abs(all_preds - all_targets))
        mape = 0

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'rmse': rmse_orig,
        'mae': mae_orig,
        'mape': mape,
        'direction_acc': dir_acc
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                     steps_per_epoch):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        if step < warmup_steps:
            return step / warmup_steps

        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # Configuration
    config = Config()

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_path=config.data_path,
        window_size=config.window_size,
        batch_size=config.batch_size
    )

    # Update config from data metadata
    config.update_from_metadata(metadata)
    print(f"Price features: {config.price_features}")
    print(f"Engineered features: {config.engineered_features}")

    # Model
    model = SPHNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer & Schedulers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        config.warmup_epochs,
        config.epochs,
        len(train_loader)
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Loss functions
    criterion_reg = nn.HuberLoss()  # Robust to outliers
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_nll = GaussianNLLLoss()

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{config.log_dir}/run_{timestamp}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion_reg, criterion_cls, criterion_nll,
            config, device, scaler
        )

        # Step scheduler
        scheduler.step()

        print(f"Train - Loss: {train_metrics['loss']:.6f}, "
              f"MSE: {train_metrics['mse']:.6f}")

        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_reg, criterion_cls, criterion_nll,
            config, device, metadata['target_scaler']
        )

        print(f"Val - Loss: {val_metrics['loss']:.6f}, "
              f"RMSE: {val_metrics['rmse']:.6f}, "
              f"MAE: {val_metrics['mae']:.6f}, "
              f"MAPE: {val_metrics['mape']:.2f}%, "
              f"Dir Acc: {val_metrics['direction_acc']:.4f}")

        # Log to TensorBoard
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalar('Val/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Val/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Val/MAPE', val_metrics['mape'], epoch)
        writer.add_scalar('Val/Direction_Acc', val_metrics['direction_acc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config.__dict__
            }
            torch.save(checkpoint, f"{config.checkpoint_dir}/best_model.pt")
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(f"{config.checkpoint_dir}/best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(
        model, test_loader,
        criterion_reg, criterion_cls, criterion_nll,
        config, device, metadata['target_scaler']
    )

    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"Test Direction Accuracy: {test_metrics['direction_acc']:.4f}")

    # Save test results
    results = {
        'test_metrics': test_metrics,
        'best_val_metrics': checkpoint['val_metrics'],
        'config': config.__dict__,
        'n_params': n_params
    }

    with open(f"{config.checkpoint_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
