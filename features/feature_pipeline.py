"""
Feature Engineering Pipeline

Orchestrates all feature computation and ensures no look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    CRITICAL RULES:
    1. All features use only past/current data
    2. No future information leaks
    3. NaN handling must be explicit
    """

    # Define feature groups for model input
    PRICE_FEATURES = ['open', 'high', 'low', 'close', 'volume']

    # Engineered features that go into the feature encoder
    ENGINEERED_FEATURE_GROUPS = {
        'returns': [
            'return_1h', 'return_4h', 'return_24h', 'log_return_1h'
        ],
        'volatility': [
            'volatility_12h', 'volatility_24h', 'atr_24h_pct'
        ],
        'momentum': [
            'rsi_14', 'roc_6h', 'roc_12h', 'macd_hist'
        ],
        'trend': [
            'close_to_sma_20', 'close_to_sma_50', 'trend_slope_20'
        ],
        'volume': [
            'relative_volume', 'obv_slope', 'close_to_vwap'
        ],
        'orderflow': [
            'order_flow_imbalance', 'ofi_ma_12', 'tbr_ma_12',
            'funding_rate', 'funding_rate_zscore'
        ],
        'session': [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ],
        'sentiment': [
            'fng_normalized', 'fng_deviation'
        ],
        'regime': [
            'vol_percentile', 'vol_ratio', 'trend_efficiency',
            'trend_efficiency_72h', 'bb_width_pct', 'bb_width_percentile',
            'atr_percentile', 'regime_low_vol', 'regime_high_vol'
        ]
    }

    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        self.ti = TechnicalIndicators()

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features.

        Order matters - some features depend on others.
        """
        logger.info("Computing features...")

        result = df.copy()

        # 1. Returns (needed for volatility)
        result = self.ti.compute_returns(result)

        # 2. Volatility
        result = self.ti.compute_volatility(result)

        # 3. Momentum
        result = self.ti.compute_momentum(result)

        # 4. Trend
        result = self.ti.compute_trend(result)

        # 5. Volume features
        result = self.ti.compute_volume_features(result)

        # 6. Order flow features
        result = self.ti.compute_orderflow_features(result)

        # 7. Session features
        result = self.ti.compute_session_features(result)

        # 8. Sentiment features
        result = self.ti.compute_sentiment_features(result)

        # 9. Regime features (helps with distribution shift awareness)
        result = self.ti.compute_regime_features(result)

        logger.info(f"Computed {len(result.columns)} total columns")

        return result

    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """
        Get column names for price and engineered features.

        Returns:
            (price_columns, engineered_columns)
        """
        engineered = []
        for group_features in self.ENGINEERED_FEATURE_GROUPS.values():
            engineered.extend(group_features)

        return self.PRICE_FEATURES.copy(), engineered

    def normalize_features(
        self,
        df: pd.DataFrame,
        fit_data: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using z-score normalization.

        CRITICAL: When normalizing test data, use statistics from training data!

        Args:
            df: Data to normalize
            fit_data: Data to compute statistics from (use training set)
                     If None, uses df itself

        Returns:
            (normalized_df, normalization_stats)
        """
        if fit_data is None:
            fit_data = df

        result = df.copy()
        stats = {}

        price_cols, eng_cols = self.get_feature_columns()
        all_feature_cols = price_cols + eng_cols

        for col in all_feature_cols:
            if col not in result.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            # Compute stats from fit_data
            mean = fit_data[col].mean()
            std = fit_data[col].std()

            if std == 0 or pd.isna(std):
                std = 1.0  # Avoid division by zero

            stats[col] = {'mean': float(mean), 'std': float(std)}

            # Normalize
            result[col] = (result[col] - mean) / std

        return result, stats

    def apply_normalization(
        self,
        df: pd.DataFrame,
        stats: Dict
    ) -> pd.DataFrame:
        """Apply pre-computed normalization statistics."""
        result = df.copy()

        for col, col_stats in stats.items():
            if col in result.columns:
                result[col] = (
                    (result[col] - col_stats['mean']) / col_stats['std']
                )

        return result

    def validate_features(self, df: pd.DataFrame) -> Dict:
        """
        Validate feature quality.

        Checks:
        - NaN percentages
        - Infinite values
        - Reasonable ranges
        """
        price_cols, eng_cols = self.get_feature_columns()
        all_cols = price_cols + eng_cols

        issues = {}

        for col in all_cols:
            if col not in df.columns:
                issues[col] = 'MISSING'
                continue

            nan_pct = df[col].isna().mean() * 100
            inf_pct = np.isinf(df[col]).mean() * 100

            if nan_pct > 20:
                issues[col] = f'HIGH_NAN ({nan_pct:.1f}%)'
            elif inf_pct > 0:
                issues[col] = f'HAS_INF ({inf_pct:.1f}%)'

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_features': len(all_cols),
            'available_features': sum(1 for c in all_cols if c in df.columns)
        }

    def get_warmup_periods(self) -> int:
        """
        Get number of periods needed for feature warmup.

        This is the minimum number of rows before features are valid.
        Based on the longest lookback period used.
        """
        # SMA 100 is the longest indicator
        # Add buffer for safety
        return 100 + 10
