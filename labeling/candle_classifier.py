"""
Candle Classification Based on Tradeability

CRITICAL: This module labels candles based on NEXT candle's behavior.
At time T, we label using data from T+1 to create prediction targets.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict


class CandleLabel(IntEnum):
    """5-class candle classification."""
    HIGH_BULL = 0    # Strong bullish + clean path -> Ideal LONG
    BULL = 1         # Moderate bullish or messy strong bull
    RANGE_BOUND = 2  # No clear direction -> NO TRADE
    BEAR = 3         # Moderate bearish or messy strong bear
    LOW_BEAR = 4     # Strong bearish + clean path -> Ideal SHORT


@dataclass
class LabelingConfig:
    """
    Labeling thresholds - tune based on asset volatility.

    For BTC hourly data (typical 0.5-2% hourly moves):
    - strong_move: 1.5% captures significant moves
    - weak_move: 0.5% filters out noise
    - clean_path_mae: 0.5% matches typical tight stop-loss
    """
    strong_move_threshold: float = 0.015    # 1.5%
    weak_move_threshold: float = 0.005      # 0.5%
    clean_path_mae_threshold: float = 0.005 # 0.5%

    def validate(self):
        assert self.strong_move_threshold > self.weak_move_threshold
        assert self.clean_path_mae_threshold > 0
        assert self.weak_move_threshold > 0


class CandleLabeler:
    """
    Labels candles based on return magnitude AND path quality.

    Key Concept:
    - MAE (Maximum Adverse Excursion): Worst drawdown during candle
    - MFE (Maximum Favorable Excursion): Best unrealized profit

    A "clean" path means price didn't spike against your position
    before reaching its destination.
    """

    def __init__(self, config: LabelingConfig = None):
        self.config = config or LabelingConfig()
        self.config.validate()

    def compute_mae_mfe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MAE/MFE for each candle.

        For LONG at open:
        - MAE = (open - low) / open  -> How much it dropped
        - MFE = (high - open) / open -> How much it rose

        For SHORT at open:
        - MAE = (high - open) / open -> How much it rose (bad)
        - MFE = (open - low) / open  -> How much it dropped (good)
        """
        result = df.copy()

        # Validate OHLC consistency first
        mask_valid = (
            (result['high'] >= result['low']) &
            (result['high'] >= result['open']) &
            (result['high'] >= result['close']) &
            (result['low'] <= result['open']) &
            (result['low'] <= result['close']) &
            (result['open'] > 0)
        )

        # MAE/MFE for LONG positions
        result['mae_long'] = np.where(
            mask_valid & (result['open'] > 0),
            (result['open'] - result['low']) / result['open'],
            np.nan
        )
        result['mfe_long'] = np.where(
            mask_valid & (result['open'] > 0),
            (result['high'] - result['open']) / result['open'],
            np.nan
        )

        # MAE/MFE for SHORT positions
        result['mae_short'] = np.where(
            mask_valid & (result['open'] > 0),
            (result['high'] - result['open']) / result['open'],
            np.nan
        )
        result['mfe_short'] = np.where(
            mask_valid & (result['open'] > 0),
            (result['open'] - result['low']) / result['open'],
            np.nan
        )

        # Candle return (open to close)
        result['candle_return'] = np.where(
            result['open'] > 0,
            (result['close'] - result['open']) / result['open'],
            np.nan
        )

        return result

    def _classify_candle(
        self,
        candle_return: float,
        mae_long: float,
        mae_short: float
    ) -> int:
        """
        Classify single candle based on return and path quality.

        Decision Tree:
        1. If |return| < weak_threshold -> RANGE_BOUND
        2. If return > 0:
           - If return >= strong AND mae_long < clean -> HIGH_BULL
           - Else -> BULL
        3. If return < 0:
           - If return <= -strong AND mae_short < clean -> LOW_BEAR
           - Else -> BEAR
        """
        if pd.isna(candle_return) or pd.isna(mae_long) or pd.isna(mae_short):
            return np.nan

        cfg = self.config

        # RANGE BOUND: Insignificant move
        if abs(candle_return) < cfg.weak_move_threshold:
            return CandleLabel.RANGE_BOUND

        # BULLISH
        if candle_return >= cfg.weak_move_threshold:
            is_strong = candle_return >= cfg.strong_move_threshold
            is_clean = mae_long < cfg.clean_path_mae_threshold

            if is_strong and is_clean:
                return CandleLabel.HIGH_BULL
            return CandleLabel.BULL

        # BEARISH
        if candle_return <= -cfg.weak_move_threshold:
            is_strong = candle_return <= -cfg.strong_move_threshold
            is_clean = mae_short < cfg.clean_path_mae_threshold

            if is_strong and is_clean:
                return CandleLabel.LOW_BEAR
            return CandleLabel.BEAR

        return CandleLabel.RANGE_BOUND

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label dataset with forward-looking targets.

        CRITICAL: Labels use NEXT candle's data!
        - Features at row T describe state at T
        - Labels at row T describe what happens at T+1
        - Last row will have NaN label (no future data)

        Returns DataFrame with added columns:
        - next_* columns for next candle metrics
        - label (0-4)
        - label_name (string)
        """
        result = self.compute_mae_mfe(df)

        # Shift NEXT candle data - THIS IS THE TARGET
        result['next_return'] = result['candle_return'].shift(-1)
        result['next_mae_long'] = result['mae_long'].shift(-1)
        result['next_mfe_long'] = result['mfe_long'].shift(-1)
        result['next_mae_short'] = result['mae_short'].shift(-1)
        result['next_mfe_short'] = result['mfe_short'].shift(-1)
        result['next_open'] = result['open'].shift(-1)
        result['next_high'] = result['high'].shift(-1)
        result['next_low'] = result['low'].shift(-1)
        result['next_close'] = result['close'].shift(-1)

        # Classify based on NEXT candle
        result['label'] = result.apply(
            lambda row: self._classify_candle(
                row['next_return'],
                row['next_mae_long'],
                row['next_mae_short']
            ),
            axis=1
        )

        # Human-readable names
        label_names = {
            0: 'HIGH_BULL',
            1: 'BULL',
            2: 'RANGE_BOUND',
            3: 'BEAR',
            4: 'LOW_BEAR'
        }
        result['label_name'] = result['label'].map(label_names)

        return result

    def get_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze label distribution."""
        if 'label' not in df.columns:
            raise ValueError("DataFrame must be labeled first")

        valid_labels = df['label'].dropna()
        total = len(valid_labels)

        counts = valid_labels.value_counts().sort_index()

        return {
            'counts': counts.to_dict(),
            'percentages': (counts / total * 100).round(2).to_dict(),
            'total_samples': total,
            'tradeable_high_conf': int((valid_labels.isin([0, 4])).sum()),
            'tradeable_pct': round((valid_labels.isin([0, 4])).mean() * 100, 2)
        }
