"""
PyTorch Dataset for SPH-Net

Handles windowing and sequence creation for time series.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading classification.

    Creates windows of historical data for prediction.

    CRITICAL:
    - Window at index i uses data from [i, i+window_size)
    - Label at index i is for predicting what happens AFTER window
    - The label was already shifted during labeling step
    """

    def __init__(
        self,
        df: pd.DataFrame,
        price_columns: List[str],
        feature_columns: List[str],
        window_size: int = 64,
        label_column: str = 'label'
    ):
        """
        Args:
            df: Labeled and feature-engineered DataFrame
            price_columns: Columns for price encoder
            feature_columns: Columns for feature encoder
            window_size: Sequence length for model input
            label_column: Column containing labels
        """
        self.window_size = window_size
        self.label_column = label_column

        # Filter to available columns
        available_price_cols = [c for c in price_columns if c in df.columns]
        available_feature_cols = [c for c in feature_columns if c in df.columns]

        if len(available_price_cols) < len(price_columns):
            logger.warning(f"Missing price columns: {set(price_columns) - set(available_price_cols)}")
        if len(available_feature_cols) < len(feature_columns):
            logger.warning(f"Missing feature columns: {set(feature_columns) - set(available_feature_cols)}")

        self.price_columns = available_price_cols
        self.feature_columns = available_feature_cols

        # Store data as numpy arrays for efficiency
        self.prices = df[available_price_cols].values.astype(np.float32)
        self.features = df[available_feature_cols].values.astype(np.float32)
        self.labels = df[label_column].values.astype(np.float32)

        # Store additional columns for analysis
        self.next_return = df['next_return'].values.astype(np.float32) if 'next_return' in df.columns else np.zeros(len(df), dtype=np.float32)
        self.next_mae_long = df['next_mae_long'].values.astype(np.float32) if 'next_mae_long' in df.columns else np.zeros(len(df), dtype=np.float32)
        self.next_mae_short = df['next_mae_short'].values.astype(np.float32) if 'next_mae_short' in df.columns else np.zeros(len(df), dtype=np.float32)

        # Valid indices: need full window AND valid label
        self.valid_indices = self._get_valid_indices()

        logger.info(f"Dataset created: {len(self.valid_indices)} valid samples "
                   f"from {len(df)} rows")

    def _get_valid_indices(self) -> np.ndarray:
        """
        Get indices where we have:
        1. Full window of data
        2. Valid label (not NaN)
        3. No NaN in features within window
        """
        valid = []
        n = len(self.labels)

        for i in range(n - self.window_size):
            # Check label is valid
            label_idx = i + self.window_size - 1
            if np.isnan(self.labels[label_idx]):
                continue

            # Check window has no NaN in critical columns
            window_prices = self.prices[i:i + self.window_size]

            if np.any(np.isnan(window_prices[:, :4])):  # OHLC must be valid
                continue

            valid.append(i)

        return np.array(valid)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns dict with:
        - prices: [window_size, n_price_features]
        - features: [window_size, n_engineered_features]
        - label: scalar (0-4)
        - next_return: scalar (for evaluation)
        - next_mae_long: scalar (for evaluation)
        - next_mae_short: scalar (for evaluation)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        label_idx = end_idx - 1

        # Replace NaN with 0 in features (for stability)
        prices = np.nan_to_num(self.prices[start_idx:end_idx], nan=0.0)
        features = np.nan_to_num(self.features[start_idx:end_idx], nan=0.0)

        return {
            'prices': torch.tensor(prices, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(self.labels[label_idx], dtype=torch.long),
            'next_return': torch.tensor(self.next_return[label_idx], dtype=torch.float32),
            'next_mae_long': torch.tensor(self.next_mae_long[label_idx], dtype=torch.float32),
            'next_mae_short': torch.tensor(self.next_mae_short[label_idx], dtype=torch.float32),
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    price_columns: List[str],
    feature_columns: List[str],
    window_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test.

    NOTE: train_loader shuffles, val/test do not.
    """
    train_dataset = TradingDataset(
        train_df, price_columns, feature_columns, window_size
    )
    val_dataset = TradingDataset(
        val_df, price_columns, feature_columns, window_size
    )
    test_dataset = TradingDataset(
        test_df, price_columns, feature_columns, window_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Only train shuffles
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
