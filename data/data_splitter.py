"""
Time-Series Data Splitting

CRITICAL: Must use temporal splits, NOT random splits.
Future data must never leak into training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 0.001, f"Ratios must sum to 1, got {total}"


class TemporalSplitter:
    """
    Splits time series data chronologically.

    CRITICAL: No shuffling, no random sampling.
    Train -> Validation -> Test in chronological order.
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()

    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.

        Args:
            df: DataFrame sorted by timestamp
            timestamp_col: Name of timestamp column

        Returns:
            (train_df, val_df, test_df)
        """
        # Ensure sorted
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # Log split info
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_df):,} rows "
                   f"({train_df[timestamp_col].min()} to {train_df[timestamp_col].max()})")
        logger.info(f"  Val:   {len(val_df):,} rows "
                   f"({val_df[timestamp_col].min()} to {val_df[timestamp_col].max()})")
        logger.info(f"  Test:  {len(test_df):,} rows "
                   f"({test_df[timestamp_col].min()} to {test_df[timestamp_col].max()})")

        return train_df, val_df, test_df

    def verify_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> bool:
        """
        Verify no temporal leakage between splits.
        """
        train_max = train_df[timestamp_col].max()
        val_min = val_df[timestamp_col].min()
        val_max = val_df[timestamp_col].max()
        test_min = test_df[timestamp_col].min()

        no_leakage = (train_max < val_min) and (val_max < test_min)

        if not no_leakage:
            logger.error("TEMPORAL LEAKAGE DETECTED!")
            logger.error(f"Train max: {train_max}")
            logger.error(f"Val min: {val_min}, max: {val_max}")
            logger.error(f"Test min: {test_min}")

        return no_leakage

    def get_split_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """Get summary statistics for each split."""
        summary = {}

        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            label_counts = split_df['label'].value_counts().sort_index()

            summary[name] = {
                'n_samples': len(split_df),
                'n_valid_labels': split_df['label'].notna().sum(),
                'label_distribution': label_counts.to_dict(),
                'label_pct': (label_counts / len(split_df) * 100).round(2).to_dict()
            }

        return summary
