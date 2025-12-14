"""Synthetic data generator for testing SPH-Net"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_synthetic_prices(n_samples: int = 10000, seed: int = 42):
    """
    Generate synthetic OHLCV data with trend + noise.
    Returns: prices [n_samples, 5], features [n_samples, 10]
    """
    np.random.seed(seed)

    # Base price with trend and mean reversion
    t = np.arange(n_samples)
    trend = 0.0001 * t
    cycles = 0.02 * np.sin(2 * np.pi * t / 252)  # Yearly cycle
    noise = np.random.randn(n_samples) * 0.01

    log_returns = trend + cycles + noise
    close = 100 * np.exp(np.cumsum(log_returns))

    # Generate OHLCV from close
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_price = np.roll(close, 1) * (1 + np.random.randn(n_samples) * 0.002)
    open_price[0] = close[0]
    volume = np.random.lognormal(10, 0.5, n_samples)

    prices = np.stack([open_price, high, low, close, volume], axis=1)

    # Generate synthetic engineered features (e.g., mock indicators)
    features = np.random.randn(n_samples, 10) * 0.1
    # Add some signal correlated with future returns
    features[:, 0] = np.roll(log_returns, -1) + np.random.randn(n_samples) * 0.005

    return prices.astype(np.float32), features.astype(np.float32), log_returns.astype(np.float32)


class TimeSeriesDataset(Dataset):
    """Walk-forward time series dataset"""

    def __init__(self, prices, features, returns, window_size=64, horizon=1):
        self.prices = prices
        self.features = features
        self.returns = returns
        self.window_size = window_size
        self.horizon = horizon

        # Valid indices (ensure we have enough data for window + horizon)
        self.valid_indices = len(prices) - window_size - horizon

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # Input window
        price_window = self.prices[idx:idx + self.window_size]
        feat_window = self.features[idx:idx + self.window_size]

        # Target: next return(s) and direction
        target_idx = idx + self.window_size
        target_return = self.returns[target_idx:target_idx + self.horizon]
        target_direction = (target_return > 0).astype(np.float32)

        return {
            'prices': torch.tensor(price_window),
            'features': torch.tensor(feat_window),
            'target_return': torch.tensor(target_return),
            'target_direction': torch.tensor(target_direction)
        }


def create_dataloaders(config, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders with walk-forward split"""

    prices, features, returns = generate_synthetic_prices()
    n = len(prices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_dataset = TimeSeriesDataset(
        prices[:train_end], features[:train_end], returns[:train_end],
        config.window_size, config.forecast_horizon
    )
    val_dataset = TimeSeriesDataset(
        prices[train_end:val_end], features[train_end:val_end], returns[train_end:val_end],
        config.window_size, config.forecast_horizon
    )
    test_dataset = TimeSeriesDataset(
        prices[val_end:], features[val_end:], returns[val_end:],
        config.window_size, config.forecast_horizon
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
