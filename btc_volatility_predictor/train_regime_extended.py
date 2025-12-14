"""
Train regime classifier with extended data and 90-day test period.

Key changes from train_regime.py:
1. test_days = 90 (2160 hours) for statistical significance
2. data_path = "data/processed/features_365d.csv"
3. Saves to checkpoints/best_regime_model_90d.pt
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd

from config import Config
from models import SPHNet


class VolatilityRegimeDataset(Dataset):
    """Dataset for high/low volatility classification."""

    PRICE_COLS = ['open', 'high', 'low', 'close', 'volume', 'log_return',
                  'hl_range', 'oc_range', 'log_return_abs']

    ENGINEERED_COLS = [
        'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
        'vol_park_1h', 'vol_park_24h',
        'vol_rs_1h', 'vol_rs_24h',
        'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
        'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
        'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
        'atr_14', 'atr_24', 'macd_hist',
        'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]

    def __init__(self, df, window_size=48, percentile=50,
                 price_scaler=None, feature_scaler=None, fit_scalers=False,
                 vol_threshold=None):
        self.window_size = window_size

        self.price_cols = [c for c in self.PRICE_COLS if c in df.columns]
        self.eng_cols = [c for c in self.ENGINEERED_COLS if c in df.columns]

        self.prices = df[self.price_cols].values.astype(np.float32)
        self.features = df[self.eng_cols].values.astype(np.float32)

        # Target: Is next hour's volatility above threshold?
        target_vol = df['target_volatility'].values

        if vol_threshold is None:
            # Use percentile of this dataset
            self.vol_threshold = np.percentile(target_vol[~np.isnan(target_vol)], percentile)
        else:
            self.vol_threshold = vol_threshold

        self.targets = (target_vol > self.vol_threshold).astype(np.float32)

        # Scale features
        if fit_scalers:
            self.price_scaler = RobustScaler()
            self.feature_scaler = RobustScaler()
            self.prices = self.price_scaler.fit_transform(self.prices)
            self.features = self.feature_scaler.fit_transform(self.features)
        else:
            self.price_scaler = price_scaler
            self.feature_scaler = feature_scaler
            if price_scaler:
                self.prices = price_scaler.transform(self.prices)
            if feature_scaler:
                self.features = feature_scaler.transform(self.features)

        self.n_samples = len(self.prices) - window_size - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        price_window = self.prices[idx:idx + self.window_size]
        feat_window = self.features[idx:idx + self.window_size]
        target = self.targets[idx + self.window_size]

        return {
            'prices': torch.tensor(price_window, dtype=torch.float32),
            'features': torch.tensor(feat_window, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }


def create_regime_dataloaders_extended(
    data_path: str = "data/processed/features_365d.csv",
    window_size: int = 48,
    batch_size: int = 32,
    test_days: int = 90,  # CHANGED: 90 days = 2160 hours
    val_ratio: float = 0.15,
    percentile: int = 50
):
    """Create dataloaders with 90-day test period."""
    df = pd.read_csv(data_path)

    n_total = len(df)
    n_test = test_days * 24  # 2160 hours
    n_trainval = n_total - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val

    print(f"Total: {n_total} samples ({n_total/24:.0f} days)")
    print(f"Train: {n_train} ({n_train/24:.0f} days)")
    print(f"Val: {n_val} ({n_val/24:.0f} days)")
    print(f"Test: {n_test} ({n_test/24:.0f} days)")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    # Fit on training data
    train_dataset = VolatilityRegimeDataset(
        train_df, window_size=window_size, percentile=percentile, fit_scalers=True
    )

    # Use training threshold and scalers for val/test
    val_dataset = VolatilityRegimeDataset(
        val_df, window_size=window_size,
        price_scaler=train_dataset.price_scaler,
        feature_scaler=train_dataset.feature_scaler,
        vol_threshold=train_dataset.vol_threshold
    )

    test_dataset = VolatilityRegimeDataset(
        test_df, window_size=window_size,
        price_scaler=train_dataset.price_scaler,
        feature_scaler=train_dataset.feature_scaler,
        vol_threshold=train_dataset.vol_threshold
    )

    print(f"Volatility threshold (p{percentile}): {train_dataset.vol_threshold:.6f}")
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Check class balance
    train_pos = train_dataset.targets[train_dataset.window_size:].mean()
    print(f"Train class balance: {train_pos:.1%} high volatility")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        'n_price_features': len(train_dataset.price_cols),
        'n_engineered_features': len(train_dataset.eng_cols),
        'window_size': window_size,
        'vol_threshold': train_dataset.vol_threshold,
        'price_scaler': train_dataset.price_scaler,
        'feature_scaler': train_dataset.feature_scaler,
        'test_days': test_days
    }

    return train_loader, val_loader, test_loader, metadata


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate classification metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    for batch in dataloader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        targets = batch['target'].numpy()

        outputs = model(prices, features)
        probs = torch.sigmoid(outputs['direction_pred']).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(float)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    return {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    }


def main():
    config = Config()
    config.beta_ce = 1.0  # Focus on classification
    config.alpha_mse = 0.0  # Disable regression
    config.data_path = "data/processed/features_365d.csv"

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data - use median (50th percentile) as threshold, 90-day test
    train_loader, val_loader, test_loader, metadata = create_regime_dataloaders_extended(
        config.data_path,
        window_size=config.window_size,
        batch_size=config.batch_size,
        test_days=90,  # Extended test period
        percentile=50  # Median split
    )

    config.update_from_metadata(metadata)

    # Model
    model = SPHNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    criterion = nn.BCEWithLogitsLoss()

    # Ensure checkpoint directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training
    best_val_auc = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            prices = batch['prices'].to(device)
            features = batch['features'].to(device)
            targets = batch['target'].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(prices, features)
            loss = criterion(outputs['direction_pred'], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Loss={total_loss/n_batches:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.1%}, "
              f"Val AUC={val_metrics['auc']:.4f}, "
              f"Val F1={val_metrics['f1']:.4f}")

        # Early stopping on AUC
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config.__dict__,
                'vol_threshold': metadata['vol_threshold'],
                'test_days': 90
            }, f"{config.checkpoint_dir}/best_regime_model_90d.pt")
            print(f"  Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION (90-Day Test Period)")
    print("="*60)

    checkpoint = torch.load(f"{config.checkpoint_dir}/best_regime_model_90d.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device)

    print(f"Test Accuracy:  {test_metrics['accuracy']:.1%}")
    print(f"Test Precision: {test_metrics['precision']:.1%}")
    print(f"Test Recall:    {test_metrics['recall']:.1%}")
    print(f"Test F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Test AUC:       {test_metrics['auc']:.4f}")

    # Save results
    with open(f"{config.checkpoint_dir}/regime_results_90d.json", 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'vol_threshold': float(metadata['vol_threshold']),
            'test_days': 90
        }, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
