"""
5-Class Candle Classification with Path Quality Metrics

Labels:
  0: HIGH_BULL   - >+1.5% move, clean path (low never < prev_close)
  1: BULL        - +0.5% to +1.5% move, or HIGH_BULL with messy path
  2: RANGE_BOUND - -0.5% to +0.5% move
  3: BEAR        - -0.5% to -1.5% move, or LOW_BEAR with messy path
  4: LOW_BEAR    - <-1.5% move, clean path (high never > prev_close)
"""

import pandas as pd
import numpy as np
from typing import Dict


class CandleLabeler:
    """
    Generate labels and path quality metrics.

    CRITICAL: Labels represent NEXT candle prediction, not current candle.
    This ensures no look-ahead bias in label generation.
    """

    LABELS = {
        0: "HIGH_BULL",
        1: "BULL",
        2: "RANGE_BOUND",
        3: "BEAR",
        4: "LOW_BEAR"
    }

    def __init__(
        self,
        high_threshold: float = 1.5,  # %
        low_threshold: float = 0.5,   # %
    ):
        """
        Initialize CandleLabeler.

        Args:
            high_threshold: Threshold for HIGH_BULL/LOW_BEAR (default 1.5%)
            low_threshold: Threshold for BULL/BEAR (default 0.5%)
        """
        self.high_threshold = high_threshold / 100
        self.low_threshold = low_threshold / 100

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for NEXT candle based on CURRENT candle's close.

        CRITICAL: Labels are shifted forward - we predict what WILL happen.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with labels and path metrics added
        """
        df = df.copy()

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Previous close (for next-candle prediction, this is current close)
        prev_close = df['close'].shift(1)

        # Next candle OHLC (what we're predicting)
        next_open = df['open'].shift(-1)
        next_high = df['high'].shift(-1)
        next_low = df['low'].shift(-1)
        next_close = df['close'].shift(-1)

        # Calculate move percentage
        move_pct = (next_close - prev_close) / prev_close

        # Path quality checks
        low_breach = next_low < prev_close   # For longs: adverse
        high_breach = next_high > prev_close  # For shorts: adverse

        # Path metrics (for analysis)
        # MAE = Maximum Adverse Excursion
        # MFE = Maximum Favorable Excursion
        df['next_mae_long'] = (df['close'] - next_low) / df['close']   # Max adverse if long
        df['next_mfe_long'] = (next_high - df['close']) / df['close']  # Max favorable if long
        df['next_mae_short'] = (next_high - df['close']) / df['close'] # Max adverse if short
        df['next_mfe_short'] = (df['close'] - next_low) / df['close']  # Max favorable if short

        # Classification
        df['label'] = np.nan

        # HIGH_BULL: > +1.5% AND clean path (low never breached prev close)
        mask_high_bull = (move_pct > self.high_threshold) & (~low_breach)
        df.loc[mask_high_bull, 'label'] = 0

        # LOW_BEAR: < -1.5% AND clean path (high never breached prev close)
        mask_low_bear = (move_pct < -self.high_threshold) & (~high_breach)
        df.loc[mask_low_bear, 'label'] = 4

        # BULL: +0.5% to +1.5% OR (>+1.5% but messy path)
        mask_bull = (
            ((move_pct > self.low_threshold) & (move_pct <= self.high_threshold)) |
            ((move_pct > self.high_threshold) & low_breach)
        )
        df.loc[mask_bull, 'label'] = 1

        # BEAR: -0.5% to -1.5% OR (<-1.5% but messy path)
        mask_bear = (
            ((move_pct < -self.low_threshold) & (move_pct >= -self.high_threshold)) |
            ((move_pct < -self.high_threshold) & high_breach)
        )
        df.loc[mask_bear, 'label'] = 3

        # RANGE_BOUND: everything else
        mask_range = df['label'].isna() & move_pct.notna()
        df.loc[mask_range, 'label'] = 2

        # Additional target columns
        df['next_return'] = move_pct
        df['next_direction'] = np.sign(move_pct)

        # Convert label to int where valid
        df['label'] = df['label'].astype('Int64')  # Nullable int

        # Remove last row (no future data to predict)
        df = df.iloc[:-1]

        print(f"Generated labels for {len(df)} samples")

        return df

    def get_label_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of labels.

        Args:
            df: DataFrame with 'label' column

        Returns:
            DataFrame with label distribution
        """
        if 'label' not in df.columns:
            raise ValueError("No 'label' column found. Run generate_labels() first.")

        counts = df['label'].value_counts().sort_index()
        pcts = (counts / len(df) * 100).round(2)

        dist = pd.DataFrame({
            'Label': [self.LABELS.get(i, f"Unknown_{i}") for i in counts.index],
            'Count': counts.values,
            'Percentage': pcts.values
        })

        return dist

    def get_label_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get detailed statistics for each label class.

        Args:
            df: DataFrame with labels and path metrics

        Returns:
            Dict with statistics per label
        """
        stats = {}

        for label_id, label_name in self.LABELS.items():
            mask = df['label'] == label_id
            label_df = df[mask]

            if len(label_df) == 0:
                continue

            stats[label_name] = {
                'count': len(label_df),
                'percentage': len(label_df) / len(df) * 100,
                'avg_return': label_df['next_return'].mean() * 100,
                'std_return': label_df['next_return'].std() * 100,
                'avg_mae_long': label_df['next_mae_long'].mean() * 100 if 'next_mae_long' in df else None,
                'avg_mfe_long': label_df['next_mfe_long'].mean() * 100 if 'next_mfe_long' in df else None,
            }

        return stats

    def print_distribution(self, df: pd.DataFrame):
        """Print formatted label distribution."""
        dist = self.get_label_distribution(df)

        print("\n" + "=" * 50)
        print("LABEL DISTRIBUTION")
        print("=" * 50)
        print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
        print("-" * 50)

        for _, row in dist.iterrows():
            print(f"{row['Label']:<15} {row['Count']:>10} {row['Percentage']:>11.2f}%")

        print("-" * 50)
        print(f"{'Total':<15} {df['label'].notna().sum():>10} {'100.00':>11}%")
        print("=" * 50 + "\n")
