"""
Walk-forward train/val/test split.
CRITICAL: Chronological order, NO shuffling!
"""

import pandas as pd
from typing import Tuple


class DataSplitter:
    """
    Chronological data splitter.

    CRITICAL: Data must remain in time order. Never shuffle time series data!
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize DataSplitter.

        Args:
            train_ratio: Fraction of data for training (default 0.70)
            val_ratio: Fraction of data for validation (default 0.15)
            test_ratio: Fraction of data for testing (default 0.15)
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.

        CRITICAL: No shuffling! Train is earliest, test is most recent.

        Args:
            df: DataFrame to split (must have 'timestamp' column or datetime index)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure sorted by time
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        n = len(df)

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # Print summary
        self._print_summary(df, train_df, val_df, test_df)

        # Validate no overlap
        self._validate_split(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _get_timestamp_col(self, df: pd.DataFrame):
        """Get timestamp from column or index."""
        if 'timestamp' in df.columns:
            return df['timestamp']
        elif isinstance(df.index, pd.DatetimeIndex):
            return df.index
        else:
            return None

    def _print_summary(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Print split summary."""
        n = len(full_df)

        print("\n" + "=" * 60)
        print("DATA SPLIT SUMMARY")
        print("=" * 60)
        print(f"Total samples: {n}")
        print("-" * 60)

        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            ts = self._get_timestamp_col(split_df)
            pct = len(split_df) / n * 100

            print(f"{name}: {len(split_df):>8} samples ({pct:>5.1f}%)")
            if ts is not None and len(ts) > 0:
                print(f"      Date range: {ts.min()} -> {ts.max()}")

        print("=" * 60 + "\n")

    def _validate_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Validate that splits don't overlap."""
        train_ts = self._get_timestamp_col(train_df)
        val_ts = self._get_timestamp_col(val_df)
        test_ts = self._get_timestamp_col(test_df)

        if train_ts is not None and val_ts is not None and len(train_ts) > 0 and len(val_ts) > 0:
            if train_ts.max() >= val_ts.min():
                raise ValueError(
                    f"Train/Val overlap detected! "
                    f"Train ends at {train_ts.max()}, Val starts at {val_ts.min()}"
                )

        if val_ts is not None and test_ts is not None and len(val_ts) > 0 and len(test_ts) > 0:
            if val_ts.max() >= test_ts.min():
                raise ValueError(
                    f"Val/Test overlap detected! "
                    f"Val ends at {val_ts.max()}, Test starts at {test_ts.min()}"
                )

        print("Split validation: PASSED (no overlaps)")
