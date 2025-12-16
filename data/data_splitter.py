"""
Time-Series Data Splitting

CRITICAL: Must use temporal splits, NOT random splits.
Future data must never leak into training.

Supports:
1. Standard chronological splits (70/15/15)
2. Volatile period splits (for testing on similar market conditions)
3. Date-based splits
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Date-based splitting (overrides ratios if set)
    train_end_date: Optional[str] = None  # e.g., "2022-06-01"
    val_end_date: Optional[str] = None    # e.g., "2022-11-01"
    # Volatile period preset
    preset: Optional[str] = None  # "volatile_2022", "volatile_2021", "standard"

    def __post_init__(self):
        if self.preset is None and self.train_end_date is None:
            total = self.train_ratio + self.val_ratio + self.test_ratio
            assert abs(total - 1.0) < 0.001, f"Ratios must sum to 1, got {total}"


# Predefined volatile periods for testing on similar market conditions
VOLATILE_PRESETS = {
    # Test on 2022 bear market (most volatile period)
    "volatile_2022": {
        "train_end": "2022-06-01",
        "val_end": "2022-11-01",
        "description": "Train 2019-2022H1, Val 2022H2 (bear), Test 2022-end to 2023H1 (recovery)"
    },
    # Test on 2021 (bull run with crashes)
    "volatile_2021": {
        "train_end": "2021-01-01",
        "val_end": "2021-06-01",
        "description": "Train 2019-2020, Val 2021H1 (bull), Test 2021H2 (crash+recovery)"
    },
    # Test on COVID crash and recovery
    "volatile_2020": {
        "train_end": "2020-01-01",
        "val_end": "2020-06-01",
        "description": "Train 2019, Val 2020H1 (COVID crash), Test 2020H2 (recovery)"
    },
    # Standard chronological (tests on most recent data)
    "standard": {
        "train_end": None,
        "val_end": None,
        "description": "Standard 70/15/15 chronological split"
    }
}


class TemporalSplitter:
    """
    Splits time series data chronologically.

    CRITICAL: No shuffling, no random sampling.
    Train -> Validation -> Test in chronological order.

    Supports:
    1. Ratio-based splitting (default 70/15/15)
    2. Date-based splitting (for specific period testing)
    3. Preset volatile periods (for testing on similar market conditions)
    """

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()

        # Apply preset if specified
        if self.config.preset and self.config.preset in VOLATILE_PRESETS:
            preset = VOLATILE_PRESETS[self.config.preset]
            self.config.train_end_date = preset["train_end"]
            self.config.val_end_date = preset["val_end"]
            logger.info(f"Using preset '{self.config.preset}': {preset['description']}")

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

        # Convert timestamp column if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Use date-based splitting if dates are provided
        if self.config.train_end_date is not None:
            return self._split_by_date(df, timestamp_col)
        else:
            return self._split_by_ratio(df, timestamp_col)

    def _split_by_ratio(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split using ratio-based approach."""
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        self._log_split_info(train_df, val_df, test_df, timestamp_col)
        return train_df, val_df, test_df

    def _split_by_date(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split using date-based approach."""
        train_end_date = pd.to_datetime(self.config.train_end_date)
        val_end_date = pd.to_datetime(self.config.val_end_date)

        # Handle timezone-aware timestamps
        if df[timestamp_col].dt.tz is not None:
            train_end_date = train_end_date.tz_localize('UTC')
            val_end_date = val_end_date.tz_localize('UTC')

        train_df = df[df[timestamp_col] < train_end_date].copy()
        val_df = df[(df[timestamp_col] >= train_end_date) &
                    (df[timestamp_col] < val_end_date)].copy()
        test_df = df[df[timestamp_col] >= val_end_date].copy()

        self._log_split_info(train_df, val_df, test_df, timestamp_col)
        return train_df, val_df, test_df

    def _log_split_info(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp_col: str
    ):
        """Log split information."""
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_df):,} rows "
                   f"({train_df[timestamp_col].min()} to {train_df[timestamp_col].max()})")
        logger.info(f"  Val:   {len(val_df):,} rows "
                   f"({val_df[timestamp_col].min()} to {val_df[timestamp_col].max()})")
        logger.info(f"  Test:  {len(test_df):,} rows "
                   f"({test_df[timestamp_col].min()} to {test_df[timestamp_col].max()})")

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

    def analyze_distribution_shift(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        return_col: str = 'return_1h'
    ) -> Dict:
        """
        Analyze distribution shift between splits.

        This helps diagnose if test set has fundamentally different
        market conditions than training data.

        Returns:
            Dict with volatility metrics and distribution comparison
        """
        analysis = {}

        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Compute return if not present
            if return_col not in split_df.columns and 'close' in split_df.columns:
                returns = split_df['close'].pct_change()
            elif return_col in split_df.columns:
                returns = split_df[return_col]
            else:
                returns = pd.Series([0], index=split_df.index[:1])

            # Basic statistics
            analysis[name] = {
                'n_samples': len(split_df),
                'volatility': float(returns.std() * 100),  # as percentage
                'mean_return': float(returns.mean() * 100),
                'skewness': float(returns.skew()) if len(returns) > 2 else 0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0,
                'max_drawdown': float(self._compute_max_drawdown(split_df)),
            }

            # Label distribution
            if 'label' in split_df.columns:
                label_pct = split_df['label'].value_counts(normalize=True).sort_index() * 100
                analysis[name]['label_pct'] = label_pct.to_dict()

                # Tradeable percentage (labels 0 and 4)
                tradeable_pct = ((split_df['label'] == 0) | (split_df['label'] == 4)).mean() * 100
                analysis[name]['tradeable_pct'] = float(tradeable_pct)

        # Compute shift metrics
        if 'train' in analysis and 'test' in analysis:
            analysis['shift_metrics'] = {
                'volatility_ratio': analysis['test']['volatility'] / (analysis['train']['volatility'] + 1e-8),
                'tradeable_ratio': analysis['test'].get('tradeable_pct', 0) / (analysis['train'].get('tradeable_pct', 1) + 1e-8),
            }

            # Flag severe distribution shift
            vol_ratio = analysis['shift_metrics']['volatility_ratio']
            trade_ratio = analysis['shift_metrics']['tradeable_ratio']

            if vol_ratio < 0.7 or vol_ratio > 1.5:
                logger.warning(f"DISTRIBUTION SHIFT WARNING: Volatility ratio = {vol_ratio:.2f}")
                logger.warning("Test period has significantly different volatility than training!")

            if trade_ratio < 0.7 or trade_ratio > 1.5:
                logger.warning(f"DISTRIBUTION SHIFT WARNING: Tradeable ratio = {trade_ratio:.2f}")
                logger.warning("Test period has different tradeable opportunity frequency!")

        return analysis

    def _compute_max_drawdown(self, df: pd.DataFrame) -> float:
        """Compute max drawdown for the period."""
        if 'close' not in df.columns:
            return 0.0

        prices = df['close'].values
        peak = prices[0]
        max_dd = 0

        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd * 100  # as percentage

    def print_distribution_analysis(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Print formatted distribution analysis."""
        analysis = self.analyze_distribution_shift(train_df, val_df, test_df)

        print("\n" + "="*70)
        print("DISTRIBUTION SHIFT ANALYSIS")
        print("="*70)

        # Header
        print(f"\n{'Metric':<25} {'Train':>12} {'Val':>12} {'Test':>12}")
        print("-"*70)

        # Basic metrics
        for metric in ['n_samples', 'volatility', 'mean_return', 'tradeable_pct']:
            train_val = analysis['train'].get(metric, 0)
            val_val = analysis['val'].get(metric, 0)
            test_val = analysis['test'].get(metric, 0)

            if metric == 'n_samples':
                print(f"{metric:<25} {train_val:>12,} {val_val:>12,} {test_val:>12,}")
            else:
                print(f"{metric:<25} {train_val:>12.2f} {val_val:>12.2f} {test_val:>12.2f}")

        # Label distribution
        print("\nLabel Distribution (%):")
        print("-"*70)
        label_names = {0: 'HIGH_BULL', 1: 'BULL', 2: 'RANGE_BOUND', 3: 'BEAR', 4: 'LOW_BEAR'}

        for label in range(5):
            name = label_names[label]
            train_pct = analysis['train'].get('label_pct', {}).get(label, 0)
            val_pct = analysis['val'].get('label_pct', {}).get(label, 0)
            test_pct = analysis['test'].get('label_pct', {}).get(label, 0)
            print(f"  {name:<20} {train_pct:>12.2f} {val_pct:>12.2f} {test_pct:>12.2f}")

        # Shift metrics
        if 'shift_metrics' in analysis:
            print("\nDistribution Shift Metrics:")
            print("-"*70)
            for metric, value in analysis['shift_metrics'].items():
                status = "OK" if 0.7 <= value <= 1.5 else "WARNING"
                print(f"  {metric:<25}: {value:.2f} [{status}]")

        print("="*70 + "\n")
