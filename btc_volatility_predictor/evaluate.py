"""Evaluation and visualization for BTC volatility predictor."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from config import Config
from models import SPHNet
from data.dataset import create_dataloaders


def load_model(checkpoint_path: str, config: Config, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Update config if stored
    if 'config' in checkpoint:
        for k, v in checkpoint['config'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = SPHNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


@torch.no_grad()
def predict_all(model, dataloader, device, target_scaler=None):
    """Get all predictions from dataloader."""
    all_preds = []
    all_targets = []
    all_uncertainties = []

    for batch in dataloader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].numpy()

        outputs = model(prices, features)
        preds = outputs['volatility_pred'].cpu().numpy()

        all_preds.append(preds)
        all_targets.append(target_vol)

        if 'uncertainty' in outputs:
            all_uncertainties.append(outputs['uncertainty'].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Inverse transform
    if target_scaler is not None:
        all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    results = {
        'predictions': all_preds,
        'targets': all_targets
    }

    if all_uncertainties:
        all_uncertainties = np.concatenate(all_uncertainties, axis=0).flatten()
        results['uncertainties'] = all_uncertainties

    return results


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Directional accuracy
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    dir_acc = np.mean(true_direction == pred_direction)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Direction_Accuracy': dir_acc
    }


def plot_predictions(results, save_path: str = "figures/predictions.png"):
    """Plot actual vs predicted volatility."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    preds = results['predictions']
    targets = results['targets']

    # Time series plot (last 100 points)
    n_show = min(240, len(preds))  # 10 days
    ax = axes[0, 0]
    ax.plot(targets[-n_show:], label='Actual', alpha=0.8)
    ax.plot(preds[-n_show:], label='Predicted', alpha=0.8)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Volatility')
    ax.set_title('Volatility Prediction (Last 10 Days)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[0, 1]
    ax.scatter(targets, preds, alpha=0.5, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Actual Volatility')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1, 0]
    errors = preds - targets
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (Mean: {errors.mean():.4f}, Std: {errors.std():.4f})')
    ax.grid(True, alpha=0.3)

    # Uncertainty vs error (if available)
    ax = axes[1, 1]
    if 'uncertainties' in results:
        uncertainties = results['uncertainties']
        abs_errors = np.abs(errors)
        ax.scatter(uncertainties, abs_errors, alpha=0.5, s=10)
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Uncertainty Calibration')
    else:
        # Rolling error
        window = 24
        rolling_mae = pd.Series(np.abs(errors)).rolling(window).mean()
        ax.plot(rolling_mae)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Rolling MAE (24h)')
        ax.set_title('Error Over Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved predictions plot to {save_path}")


def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader, metadata = create_dataloaders(
        data_path=config.data_path,
        window_size=config.window_size,
        batch_size=config.batch_size
    )
    config.update_from_metadata(metadata)

    # Load model
    model = load_model(f"{config.checkpoint_dir}/best_model.pt", config, device)

    # Predict
    print("Generating predictions...")
    results = predict_all(model, test_loader, device, metadata['target_scaler'])

    # Calculate metrics
    metrics = calculate_metrics(results['targets'], results['predictions'])

    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    # Plot
    os.makedirs("figures", exist_ok=True)
    plot_predictions(results)

    # Save predictions
    df = pd.DataFrame({
        'actual': results['targets'],
        'predicted': results['predictions']
    })
    if 'uncertainties' in results:
        df['uncertainty'] = results['uncertainties']

    df.to_csv("figures/test_predictions.csv", index=False)
    print("\nSaved predictions to figures/test_predictions.csv")


if __name__ == "__main__":
    main()
