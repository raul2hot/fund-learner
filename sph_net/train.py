"""Training script for SPH-Net"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from config import Config
from models import SPHNet
from data.synthetic import create_dataloaders


def train_epoch(model, dataloader, optimizer, criterion_mse, criterion_bce, config, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_return = batch['target_return'].to(device)
        target_direction = batch['target_direction'].to(device)

        optimizer.zero_grad()

        outputs = model(prices, features)

        # Compute losses
        loss_mse = criterion_mse(outputs['return_pred'], target_return)
        loss_bce = criterion_bce(outputs['direction_pred'], target_direction)

        # Combined loss
        loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, criterion_mse, criterion_bce, config, device):
    """Validation pass"""
    model.eval()
    total_loss = 0
    total_mse = 0
    correct_direction = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            prices = batch['prices'].to(device)
            features = batch['features'].to(device)
            target_return = batch['target_return'].to(device)
            target_direction = batch['target_direction'].to(device)

            outputs = model(prices, features)

            loss_mse = criterion_mse(outputs['return_pred'], target_return)
            loss_bce = criterion_bce(outputs['direction_pred'], target_direction)
            loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce

            total_loss += loss.item()
            total_mse += loss_mse.item()

            # Direction accuracy
            pred_dir = (torch.sigmoid(outputs['direction_pred']) > 0.5).float()
            correct_direction += (pred_dir == target_direction).sum().item()
            total_samples += target_direction.numel()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'direction_acc': correct_direction / total_samples
    }


def main():
    # Configuration
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Model
    model = SPHNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion_mse = nn.HuberLoss()  # More robust than MSE
    criterion_bce = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_bce, config, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion_mse, criterion_bce, config, device)
        print(f"Val Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}, Dir Acc: {val_metrics['direction_acc']:.4f}")

        scheduler.step(val_metrics['loss'])

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)

    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = validate(model, test_loader, criterion_mse, criterion_bce, config, device)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test Direction Accuracy: {test_metrics['direction_acc']:.4f}")


if __name__ == "__main__":
    main()
