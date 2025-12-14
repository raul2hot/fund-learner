"""
PyTorch Dataset for BTC volatility prediction.
Walk-forward split for time series integrity.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict
import os


class BTCVolatilityDataset(Dataset):
    """
    Dataset for BTC/USDT volatility prediction.

    Features are split into:
    - price_features: OHLCV and derived price metrics
    - engineered_features: Technical indicators and volume metrics

    Target: Next-hour Garman-Klass volatility
    """

    # Define feature groups
    PRICE_COLS = ['open', 'high', 'low', 'close', 'volume', 'log_return',
                  'hl_range', 'oc_range', 'log_return_abs']

    ENGINEERED_COLS = [
        # Volatility features
        'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
        'vol_park_1h', 'vol_park_24h',
        'vol_rs_1h', 'vol_rs_24h',
        'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
        # Momentum
        'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
        # Technical indicators
        'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
        'atr_14', 'atr_24', 'macd_hist',
        # Volume features
        'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
        # Time features
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 72,  # 3 days of hourly data
        horizon: int = 1,
        price_scaler=None,
        feature_scaler=None,
        target_scaler=None,
        fit_scalers: bool = False
    ):
        """
        Args:
            df: DataFrame with all features and targets
            window_size: Number of past candles to use as input
            horizon: Forecast horizon (1 = next hour)
            price_scaler: Fitted scaler for price features (or None to create)
            feature_scaler: Fitted scaler for engineered features
            target_scaler: Fitted scaler for target volatility
            fit_scalers: Whether to fit scalers on this data
        """
        self.window_size = window_size
        self.horizon = horizon

        # Get available columns
        self.price_cols = [c for c in self.PRICE_COLS if c in df.columns]
        self.eng_cols = [c for c in self.ENGINEERED_COLS if c in df.columns]

        # Extract arrays
        self.prices = df[self.price_cols].values.astype(np.float32)
        self.features = df[self.eng_cols].values.astype(np.float32)
        self.targets = df['target_volatility'].values.astype(np.float32)
        self.target_direction = df['target_vol_direction'].values.astype(np.float32)

        # Handle scalers
        if fit_scalers:
            self.price_scaler = RobustScaler()
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()

            self.prices = self.price_scaler.fit_transform(self.prices)
            self.features = self.feature_scaler.fit_transform(self.features)
            self.targets = self.target_scaler.fit_transform(
                self.targets.reshape(-1, 1)
            ).flatten()
        else:
            self.price_scaler = price_scaler
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

            if price_scaler is not None:
                self.prices = self.price_scaler.transform(self.prices)
            if feature_scaler is not None:
                self.features = self.feature_scaler.transform(self.features)
            if target_scaler is not None:
                self.targets = self.target_scaler.transform(
                    self.targets.reshape(-1, 1)
                ).flatten()

        # Valid indices
        self.n_samples = len(self.prices) - window_size - horizon

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Input window
        price_window = self.prices[idx:idx + self.window_size]
        feat_window = self.features[idx:idx + self.window_size]

        # Target (next timestep after window)
        target_idx = idx + self.window_size
        target_vol = self.targets[target_idx]
        target_dir = self.target_direction[target_idx]

        return {
            'prices': torch.tensor(price_window, dtype=torch.float32),
            'features': torch.tensor(feat_window, dtype=torch.float32),
            'target_volatility': torch.tensor(target_vol, dtype=torch.float32),
            'target_direction': torch.tensor(target_dir, dtype=torch.float32)
        }

    def get_scalers(self):
        """Return fitted scalers for use in validation/test sets."""
        return self.price_scaler, self.feature_scaler, self.target_scaler


def create_dataloaders(
    data_path: str = "data/processed/features.csv",
    window_size: int = 72,
    batch_size: int = 64,
    test_days: int = 10,  # Last 10 days for testing
    val_ratio: float = 0.15,  # 15% of training data for validation
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test dataloaders with walk-forward split.

    Split strategy:
    - Test: Last 10 days (240 hours)
    - Val: 15% of remaining data (before test)
    - Train: Everything else

    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    n_total = len(df)
    n_test = test_days * 24  # 10 days = 240 hourly candles
    n_trainval = n_total - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val

    print(f"Total samples: {n_total}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Split data (walk-forward: train -> val -> test)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    # Create datasets (fit scalers only on training data)
    train_dataset = BTCVolatilityDataset(
        train_df, window_size=window_size, fit_scalers=True
    )

    # Get scalers from training set
    price_scaler, feature_scaler, target_scaler = train_dataset.get_scalers()

    val_dataset = BTCVolatilityDataset(
        val_df, window_size=window_size,
        price_scaler=price_scaler,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )

    test_dataset = BTCVolatilityDataset(
        test_df, window_size=window_size,
        price_scaler=price_scaler,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Metadata for model config
    metadata = {
        'n_price_features': len(train_dataset.price_cols),
        'n_engineered_features': len(train_dataset.eng_cols),
        'window_size': window_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'price_scaler': price_scaler,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, test_loader, metadata = create_dataloaders()

    print(f"\nMetadata: {metadata}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
