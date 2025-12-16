# SPH-Net 5-Class Classifier Implementation Instructions

## Executive Summary

This document provides precise instructions for implementing a trading classification system using the SPH-Net architecture. The system predicts **tradeability** (not just direction) using a 5-class labeling scheme based on return magnitude AND path quality (MAE/MFE).

**Key Insight**: We predict whether a trade can be profitably executed, considering stop-loss survival, not just price direction.

---

## 1. Project Structure

Create the following directory structure:

```
fund-learner/
├── data_pipleine/           # Existing - data fetching
│   └── ml_data/
│       ├── BTCUSDT_ml_data.parquet
│       └── BTCUSDT_metadata.json
├── labeling/                # NEW - labeling system
│   ├── __init__.py
│   ├── candle_classifier.py
│   └── label_analyzer.py
├── features/                # NEW - feature engineering
│   ├── __init__.py
│   ├── technical_indicators.py
│   └── feature_pipeline.py
├── data/                    # NEW - dataset management
│   ├── __init__.py
│   ├── dataset.py
│   └── data_splitter.py
├── sph_net/                 # MODIFY - model architecture
│   ├── config.py
│   ├── models/
│   │   ├── sph_net.py
│   │   ├── encoders.py
│   │   ├── attention.py
│   │   └── heads.py
│   └── data/                # Remove synthetic.py dependency
├── training/                # NEW - training pipeline
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── evaluation/              # NEW - evaluation & reporting
│   ├── __init__.py
│   ├── evaluator.py
│   └── trading_metrics.py
├── scripts/                 # NEW - entry points
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
└── configs/                 # NEW - configuration files
    └── experiment_config.yaml
```

---

## 2. Labeling System Implementation

### 2.1 File: `labeling/candle_classifier.py`

```python
"""
Candle Classification Based on Tradeability

CRITICAL: This module labels candles based on NEXT candle's behavior.
At time T, we label using data from T+1 to create prediction targets.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Dict


class CandleLabel(IntEnum):
    """5-class candle classification."""
    HIGH_BULL = 0    # Strong bullish + clean path → Ideal LONG
    BULL = 1         # Moderate bullish or messy strong bull
    RANGE_BOUND = 2  # No clear direction → NO TRADE
    BEAR = 3         # Moderate bearish or messy strong bear
    LOW_BEAR = 4     # Strong bearish + clean path → Ideal SHORT


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
        - MAE = (open - low) / open  → How much it dropped
        - MFE = (high - open) / open → How much it rose
        
        For SHORT at open:
        - MAE = (high - open) / open → How much it rose (bad)
        - MFE = (open - low) / open  → How much it dropped (good)
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
        1. If |return| < weak_threshold → RANGE_BOUND
        2. If return > 0:
           - If return >= strong AND mae_long < clean → HIGH_BULL
           - Else → BULL
        3. If return < 0:
           - If return <= -strong AND mae_short < clean → LOW_BEAR
           - Else → BEAR
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
```

### 2.2 File: `labeling/label_analyzer.py`

```python
"""
Label Distribution Analysis and Visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LabelAnalyzer:
    """Analyzes label quality and distribution."""
    
    def __init__(self, df: pd.DataFrame):
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        self.df = df
        
    def compute_statistics(self) -> Dict:
        """Comprehensive statistics about labels."""
        valid = self.df[self.df['label'].notna()].copy()
        
        stats = {
            'total_rows': len(self.df),
            'valid_labels': len(valid),
            'nan_labels': self.df['label'].isna().sum(),
        }
        
        # Distribution
        for label in range(5):
            count = (valid['label'] == label).sum()
            stats[f'class_{label}_count'] = count
            stats[f'class_{label}_pct'] = round(count / len(valid) * 100, 2)
        
        # Return statistics per class
        for label in range(5):
            class_data = valid[valid['label'] == label]
            if len(class_data) > 0:
                stats[f'class_{label}_mean_return'] = round(
                    class_data['next_return'].mean() * 100, 4
                )
                stats[f'class_{label}_std_return'] = round(
                    class_data['next_return'].std() * 100, 4
                )
        
        # MAE statistics for tradeable classes
        for label, col in [(0, 'next_mae_long'), (4, 'next_mae_short')]:
            class_data = valid[valid['label'] == label]
            if len(class_data) > 0:
                stats[f'class_{label}_mean_mae'] = round(
                    class_data[col].mean() * 100, 4
                )
        
        return stats
    
    def check_class_balance(self) -> Tuple[bool, str]:
        """
        Check if class distribution is reasonable.
        
        Expected distribution (roughly):
        - HIGH_BULL: 5-10%
        - BULL: 20-25%
        - RANGE_BOUND: 35-45%
        - BEAR: 20-25%
        - LOW_BEAR: 5-10%
        """
        valid = self.df[self.df['label'].notna()]
        total = len(valid)
        
        warnings = []
        
        for label in range(5):
            pct = (valid['label'] == label).sum() / total * 100
            
            if label in [0, 4]:  # HIGH_BULL, LOW_BEAR
                if pct < 2:
                    warnings.append(f"Class {label} severely underrepresented ({pct:.1f}%)")
                elif pct > 15:
                    warnings.append(f"Class {label} overrepresented ({pct:.1f}%)")
            elif label == 2:  # RANGE_BOUND
                if pct < 25:
                    warnings.append(f"RANGE_BOUND too low ({pct:.1f}%)")
                elif pct > 60:
                    warnings.append(f"RANGE_BOUND too high ({pct:.1f}%)")
        
        if warnings:
            return False, "; ".join(warnings)
        return True, "Distribution looks reasonable"
    
    def validate_label_correctness(self) -> Dict:
        """
        Validate that labels are correctly assigned.
        Spot-check that thresholds are applied correctly.
        """
        valid = self.df[self.df['label'].notna()].copy()
        
        # HIGH_BULL should have: return >= 1.5% AND mae_long < 0.5%
        high_bull = valid[valid['label'] == 0]
        if len(high_bull) > 0:
            correct_return = (high_bull['next_return'] >= 0.015).all()
            correct_mae = (high_bull['next_mae_long'] < 0.005).all()
        else:
            correct_return, correct_mae = True, True
        
        # LOW_BEAR should have: return <= -1.5% AND mae_short < 0.5%
        low_bear = valid[valid['label'] == 4]
        if len(low_bear) > 0:
            correct_return_bear = (low_bear['next_return'] <= -0.015).all()
            correct_mae_bear = (low_bear['next_mae_short'] < 0.005).all()
        else:
            correct_return_bear, correct_mae_bear = True, True
        
        return {
            'high_bull_return_check': correct_return,
            'high_bull_mae_check': correct_mae,
            'low_bear_return_check': correct_return_bear,
            'low_bear_mae_check': correct_mae_bear,
            'all_valid': all([
                correct_return, correct_mae,
                correct_return_bear, correct_mae_bear
            ])
        }
```

---

## 3. Feature Engineering

### 3.1 File: `features/technical_indicators.py`

```python
"""
Technical Indicator Computation

CRITICAL: All indicators must be computed using ONLY past data.
NO LOOK-AHEAD BIAS - never use future values.

Convention: At index i, indicator uses data from indices <= i
"""

import pandas as pd
import numpy as np
from typing import List


class TechnicalIndicators:
    """
    Computes technical indicators without look-ahead bias.
    
    All functions take a DataFrame and return it with new columns added.
    """
    
    @staticmethod
    def validate_no_lookahead(df: pd.DataFrame, new_cols: List[str]) -> bool:
        """
        Validate that new columns don't have look-ahead bias.
        Simple check: first N rows should be NaN where N = longest lookback.
        """
        # This is a heuristic check, manual review is still needed
        for col in new_cols:
            if col in df.columns:
                first_valid = df[col].first_valid_index()
                if first_valid is not None and first_valid < 5:
                    # Warning: possibly no lookback
                    return False
        return True
    
    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various return metrics.
        
        Returns use PAST data only: return at t = (close[t] - close[t-1]) / close[t-1]
        """
        result = df.copy()
        
        # Simple returns (using past data)
        result['return_1h'] = result['close'].pct_change(1)
        result['return_4h'] = result['close'].pct_change(4)
        result['return_24h'] = result['close'].pct_change(24)
        
        # Log returns
        result['log_return_1h'] = np.log(result['close'] / result['close'].shift(1))
        
        return result
    
    @staticmethod
    def compute_volatility(df: pd.DataFrame, windows: List[int] = [12, 24, 72]) -> pd.DataFrame:
        """
        Compute volatility metrics using rolling windows of PAST data.
        
        At time t, window [t-n+1, t] is used.
        """
        result = df.copy()
        
        for w in windows:
            # Rolling standard deviation of returns
            result[f'volatility_{w}h'] = (
                result['log_return_1h']
                .rolling(window=w, min_periods=w)
                .std()
            )
            
            # ATR-style volatility (using TR)
            tr = pd.concat([
                result['high'] - result['low'],
                (result['high'] - result['close'].shift(1)).abs(),
                (result['low'] - result['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            
            result[f'atr_{w}h'] = tr.rolling(window=w, min_periods=w).mean()
            
            # Normalized ATR
            result[f'atr_{w}h_pct'] = result[f'atr_{w}h'] / result['close']
        
        return result
    
    @staticmethod
    def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum indicators.
        
        RSI, ROC, etc. - all use past data only.
        """
        result = df.copy()
        
        # RSI (14-period default)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Rate of Change
        for period in [6, 12, 24]:
            result[f'roc_{period}h'] = result['close'].pct_change(period)
        
        # MACD
        ema_12 = result['close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = result['close'].ewm(span=26, adjust=False, min_periods=26).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        return result
    
    @staticmethod
    def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend indicators.
        
        Moving averages, crossovers - all backward-looking.
        """
        result = df.copy()
        
        # Simple Moving Averages
        for period in [20, 50, 100]:
            result[f'sma_{period}'] = (
                result['close']
                .rolling(window=period, min_periods=period)
                .mean()
            )
            # Price relative to SMA
            result[f'close_to_sma_{period}'] = (
                result['close'] / result[f'sma_{period}'] - 1
            )
        
        # EMA
        for period in [12, 26]:
            result[f'ema_{period}'] = (
                result['close']
                .ewm(span=period, adjust=False, min_periods=period)
                .mean()
            )
        
        # Trend strength: slope of SMA
        result['trend_slope_20'] = result['sma_20'].diff(5) / 5
        
        return result
    
    @staticmethod
    def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.
        
        Volume ratios, OBV, etc.
        """
        result = df.copy()
        
        # Volume SMA
        for period in [12, 24]:
            result[f'volume_sma_{period}'] = (
                result['volume']
                .rolling(window=period, min_periods=period)
                .mean()
            )
        
        # Relative volume
        result['relative_volume'] = result['volume'] / result['volume_sma_24']
        
        # OBV (On-Balance Volume)
        obv = np.where(
            result['close'] > result['close'].shift(1),
            result['volume'],
            np.where(
                result['close'] < result['close'].shift(1),
                -result['volume'],
                0
            )
        )
        result['obv'] = np.cumsum(obv)
        result['obv_slope'] = result['obv'].diff(5)
        
        # VWAP-style metric (rolling)
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        result['vwap_24h'] = (
            (typical_price * result['volume']).rolling(24).sum() /
            result['volume'].rolling(24).sum()
        )
        result['close_to_vwap'] = result['close'] / result['vwap_24h'] - 1
        
        return result
    
    @staticmethod
    def compute_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute order flow features.
        
        Uses taker_buy_volume, funding_rate from data pipeline.
        """
        result = df.copy()
        
        # Rolling order flow imbalance
        for period in [6, 12, 24]:
            result[f'ofi_ma_{period}'] = (
                result['order_flow_imbalance']
                .rolling(window=period, min_periods=period)
                .mean()
            )
        
        # Taker buy ratio moving average
        for period in [12, 24]:
            result[f'tbr_ma_{period}'] = (
                result['taker_buy_ratio']
                .rolling(window=period, min_periods=period)
                .mean()
            )
        
        # Funding rate features (if available)
        if 'funding_rate' in result.columns:
            result['funding_rate_ma'] = (
                result['funding_rate']
                .rolling(window=24, min_periods=1)
                .mean()
            )
            result['funding_rate_zscore'] = (
                (result['funding_rate'] - result['funding_rate_ma']) /
                result['funding_rate'].rolling(window=72, min_periods=1).std()
            )
        
        return result
    
    @staticmethod
    def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute session-based features.
        
        Cyclical encoding for hour and day_of_week.
        """
        result = df.copy()
        
        # Cyclical encoding for hour (0-23)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        # Cyclical encoding for day of week (0-6)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Session one-hot (already have 'session' column)
        result['is_asian'] = (result['session'] == 'asian').astype(int)
        result['is_london'] = (result['session'] == 'london').astype(int)
        result['is_newyork'] = (result['session'] == 'new_york').astype(int)
        
        return result
    
    @staticmethod
    def compute_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment-based features.
        
        Uses fear_greed_value from data pipeline.
        """
        result = df.copy()
        
        if 'fear_greed_value' in result.columns:
            # Normalized to 0-1
            result['fng_normalized'] = result['fear_greed_value'] / 100
            
            # Rolling average
            result['fng_ma_7d'] = (
                result['fear_greed_value']
                .rolling(window=24*7, min_periods=1)
                .mean()
            )
            
            # Deviation from average
            result['fng_deviation'] = (
                result['fear_greed_value'] - result['fng_ma_7d']
            )
        
        return result
```

### 3.2 File: `features/feature_pipeline.py`

```python
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
            
            stats[col] = {'mean': mean, 'std': std}
            
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
```

---

## 4. Data Splitting

### 4.1 File: `data/data_splitter.py`

```python
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
```

### 4.2 File: `data/dataset.py`

```python
"""
PyTorch Dataset for SPH-Net

Handles windowing and sequence creation for time series.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
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
        
        # Store data as numpy arrays for efficiency
        self.prices = df[price_columns].values.astype(np.float32)
        self.features = df[feature_columns].values.astype(np.float32)
        self.labels = df[label_column].values.astype(np.float32)
        
        # Store additional columns for analysis
        self.next_return = df['next_return'].values.astype(np.float32)
        self.next_mae_long = df['next_mae_long'].values.astype(np.float32)
        self.next_mae_short = df['next_mae_short'].values.astype(np.float32)
        
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
            window_features = self.features[i:i + self.window_size]
            
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
```

---

## 5. Model Architecture Updates

### 5.1 File: `sph_net/config.py` (UPDATED)

```python
"""SPH-Net Configuration for 5-Class Trading Classification"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SPHNetConfig:
    """Configuration for SPH-Net model."""
    
    # Data
    window_size: int = 64
    n_price_features: int = 5          # OHLCV
    n_engineered_features: int = 25    # Technical indicators
    n_classes: int = 5                  # 5-class classification
    
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_feedforward: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15
    
    # Loss weights
    class_weights: List[float] = field(default_factory=lambda: [
        2.0,   # HIGH_BULL - upweight rare class
        1.0,   # BULL
        0.5,   # RANGE_BOUND - downweight majority
        1.0,   # BEAR
        2.0    # LOW_BEAR - upweight rare class
    ])
    
    # Focal loss parameters (handles class imbalance)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Device
    device: str = "cuda"


@dataclass 
class ExperimentConfig:
    """Full experiment configuration."""
    
    # Paths
    data_path: str = "data_pipleine/ml_data/BTCUSDT_ml_data.parquet"
    output_dir: str = "experiments"
    
    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Labeling
    strong_move_threshold: float = 0.015
    weak_move_threshold: float = 0.005
    clean_path_mae_threshold: float = 0.005
    
    # Model
    model: SPHNetConfig = field(default_factory=SPHNetConfig)
    
    # Random seed
    seed: int = 42
```

### 5.2 File: `sph_net/models/heads.py` (UPDATED)

```python
"""
Prediction Heads for SPH-Net

Updated for 5-class classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    5-Class classification head.
    
    Outputs logits for each class:
    0: HIGH_BULL
    1: BULL
    2: RANGE_BOUND
    3: BEAR
    4: LOW_BEAR
    """
    
    def __init__(
        self,
        d_model: int,
        n_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]
        Returns:
            logits: [batch, n_classes]
        """
        return self.head(x)


class AuxiliaryRegressionHead(nn.Module):
    """
    Auxiliary head for return prediction.
    
    Helps with feature learning - predicts expected return.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]
        Returns:
            return_pred: [batch, 1]
        """
        return self.head(x).squeeze(-1)
```

### 5.3 File: `sph_net/models/sph_net.py` (UPDATED)

```python
"""
SPH-Net: Hybrid Transformer for Trading Classification

Updated for 5-class classification with auxiliary regression.
"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import ClassificationHead, AuxiliaryRegressionHead


class SPHNet(nn.Module):
    """
    SPH-Net for 5-Class Trading Classification.
    
    Architecture:
    1. Temporal Encoder (Transformer) - processes OHLCV
    2. Feature Encoder (MLP) - processes engineered features
    3. Co-Attention Fusion - combines both streams
    4. Classification Head - 5-class output
    5. Auxiliary Regression Head - return prediction (optional)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.n_price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )
        
        self.feature_encoder = FeatureEncoder(
            input_dim=config.n_engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Post-fusion transformer layer
        self.decoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Pooling: use last token
        self.pool = nn.Identity()  # Can replace with attention pooling
        
        # Heads
        self.classifier = ClassificationHead(
            config.d_model,
            n_classes=config.n_classes,
            dropout=config.dropout
        )
        
        self.aux_regressor = AuxiliaryRegressionHead(
            config.d_model,
            dropout=config.dropout
        )
    
    def forward(
        self,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> dict:
        """
        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]
        
        Returns:
            dict with:
            - logits: [batch, n_classes]
            - return_pred: [batch]
        """
        # Encode
        temporal_tokens = self.temporal_encoder(prices)
        feature_tokens = self.feature_encoder(features)
        
        # Fuse
        fused = self.co_attention(temporal_tokens, feature_tokens)
        
        # Decode
        decoded = self.decoder(fused)
        
        # Pool: use last token
        pooled = decoded[:, -1, :]
        
        # Predictions
        logits = self.classifier(pooled)
        return_pred = self.aux_regressor(pooled)
        
        return {
            'logits': logits,
            'return_pred': return_pred
        }
    
    def predict(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        outputs = self.forward(prices, features)
        return torch.argmax(outputs['logits'], dim=-1)
    
    def predict_proba(self, prices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        outputs = self.forward(prices, features)
        return torch.softmax(outputs['logits'], dim=-1)
```

---

## 6. Training Infrastructure

### 6.1 File: `training/losses.py`

```python
"""
Loss Functions for Trading Classification

Handles class imbalance with focal loss and class weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma > 0, reduces loss for well-classified examples,
    focusing training on hard negatives.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: [batch, n_classes] logits
            targets: [batch] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
            focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TradingLoss(nn.Module):
    """
    Combined loss for trading classification.
    
    Components:
    1. Focal loss for classification (handles imbalance)
    2. MSE loss for auxiliary return prediction
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.1
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights
        )
        self.mse_loss = nn.MSELoss()
        self.aux_weight = aux_weight
    
    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        next_return: torch.Tensor
    ) -> dict:
        """
        Args:
            outputs: dict with 'logits' and 'return_pred'
            targets: [batch] class labels
            next_return: [batch] actual returns
        
        Returns:
            dict with individual losses and total
        """
        # Classification loss
        cls_loss = self.focal_loss(outputs['logits'], targets)
        
        # Auxiliary regression loss
        aux_loss = self.mse_loss(outputs['return_pred'], next_return)
        
        # Total loss
        total_loss = cls_loss + self.aux_weight * aux_loss
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'auxiliary': aux_loss
        }
```

### 6.2 File: `training/metrics.py`

```python
"""
Training Metrics

Standard classification metrics + trading-specific metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import pandas as pd


class MetricTracker:
    """Tracks and computes metrics during training."""
    
    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        self.all_returns = []
        self.all_mae_long = []
        self.all_mae_short = []
        self.losses = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor = None,
        loss: float = None,
        next_return: torch.Tensor = None,
        next_mae_long: torch.Tensor = None,
        next_mae_short: torch.Tensor = None
    ):
        """Add batch results."""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
        if loss is not None:
            self.losses.append(loss)
        if next_return is not None:
            self.all_returns.extend(next_return.cpu().numpy())
        if next_mae_long is not None:
            self.all_mae_long.extend(next_mae_long.cpu().numpy())
        if next_mae_short is not None:
            self.all_mae_short.extend(next_mae_short.cpu().numpy())
    
    def compute(self) -> Dict:
        """Compute all metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=list(range(self.n_classes)), zero_division=0
        )
        
        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']
        for i, name in enumerate(class_names):
            metrics[f'{name}_precision'] = precision[i]
            metrics[f'{name}_recall'] = recall[i]
            metrics[f'{name}_f1'] = f1[i]
            metrics[f'{name}_support'] = int(support[i])
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Tradeable class accuracy (classes 0 and 4)
        tradeable_mask = np.isin(targets, [0, 4])
        if tradeable_mask.sum() > 0:
            metrics['tradeable_accuracy'] = accuracy_score(
                targets[tradeable_mask], preds[tradeable_mask]
            )
        else:
            metrics['tradeable_accuracy'] = 0.0
        
        # Average loss
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            targets, preds, labels=list(range(self.n_classes))
        )
        
        return metrics
    
    def compute_trading_metrics(self) -> Dict:
        """
        Compute trading-specific metrics.
        
        These evaluate whether the model's predictions
        would lead to profitable trades.
        """
        if not self.all_returns:
            return {}
        
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        returns = np.array(self.all_returns)
        mae_long = np.array(self.all_mae_long) if self.all_mae_long else None
        mae_short = np.array(self.all_mae_short) if self.all_mae_short else None
        
        metrics = {}
        
        # === LONG TRADES (predicted HIGH_BULL) ===
        long_mask = preds == 0
        if long_mask.sum() > 0:
            long_returns = returns[long_mask]
            long_targets = targets[long_mask]
            
            metrics['long_trades'] = int(long_mask.sum())
            metrics['long_avg_return'] = float(long_returns.mean() * 100)
            metrics['long_total_return'] = float(long_returns.sum() * 100)
            metrics['long_win_rate'] = float((long_returns > 0).mean() * 100)
            metrics['long_correct_class'] = float((long_targets == 0).mean() * 100)
            
            if mae_long is not None:
                long_mae = mae_long[long_mask]
                # Survival rate: would not have been stopped at 0.5%
                metrics['long_survival_rate'] = float((long_mae < 0.005).mean() * 100)
        
        # === SHORT TRADES (predicted LOW_BEAR) ===
        short_mask = preds == 4
        if short_mask.sum() > 0:
            short_returns = -returns[short_mask]  # Invert for shorts
            short_targets = targets[short_mask]
            
            metrics['short_trades'] = int(short_mask.sum())
            metrics['short_avg_return'] = float(short_returns.mean() * 100)
            metrics['short_total_return'] = float(short_returns.sum() * 100)
            metrics['short_win_rate'] = float((short_returns > 0).mean() * 100)
            metrics['short_correct_class'] = float((short_targets == 4).mean() * 100)
            
            if mae_short is not None:
                short_mae = mae_short[short_mask]
                metrics['short_survival_rate'] = float((short_mae < 0.005).mean() * 100)
        
        # === OVERALL TRADING ===
        total_trades = metrics.get('long_trades', 0) + metrics.get('short_trades', 0)
        total_samples = len(preds)
        
        metrics['trade_frequency'] = float(total_trades / total_samples * 100)
        
        if total_trades > 0:
            combined_return = (
                metrics.get('long_total_return', 0) + 
                metrics.get('short_total_return', 0)
            )
            metrics['combined_total_return'] = combined_return
            metrics['avg_return_per_trade'] = combined_return / total_trades
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get sklearn classification report as string."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']
        
        return classification_report(
            targets, preds,
            labels=list(range(self.n_classes)),
            target_names=class_names,
            zero_division=0
        )
```

### 6.3 File: `training/trainer.py`

```python
"""
Training Loop for SPH-Net

Handles training, validation, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple

from .losses import TradingLoss
from .metrics import MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for SPH-Net trading classifier.
    
    Features:
    - Training and validation loops
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    - Gradient clipping
    - Detailed logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "experiments"
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # Loss function
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32)
        self.criterion = TradingLoss(
            class_weights=class_weights,
            focal_gamma=config.focal_gamma
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_tradeable_accuracy': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Dict:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(prices, features)
            
            # Compute loss
            loss_dict = self.criterion(outputs, labels, next_return)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            preds = torch.argmax(outputs['logits'], dim=-1)
            tracker.update(
                preds=preds,
                targets=labels,
                loss=loss.item()
            )
            
            pbar.set_postfix({'loss': loss.item()})
        
        return tracker.compute()
    
    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Dict:
        """Run validation."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        tracker = MetricTracker()
        
        for batch in loader:
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            next_mae_long = batch['next_mae_long'].to(self.device)
            next_mae_short = batch['next_mae_short'].to(self.device)
            
            outputs = self.model(prices, features)
            loss_dict = self.criterion(outputs, labels, next_return)
            
            preds = torch.argmax(outputs['logits'], dim=-1)
            probs = torch.softmax(outputs['logits'], dim=-1)
            
            tracker.update(
                preds=preds,
                targets=labels,
                probs=probs,
                loss=loss_dict['total'].item(),
                next_return=next_return,
                next_mae_long=next_mae_long,
                next_mae_short=next_mae_short
            )
        
        metrics = tracker.compute()
        trading_metrics = tracker.compute_trading_metrics()
        metrics.update(trading_metrics)
        
        return metrics
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping.
        
        Returns final metrics.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"Val Tradeable Accuracy: {val_metrics['tradeable_accuracy']:.4f}")
            
            # Trading metrics
            if 'trade_frequency' in val_metrics:
                logger.info(f"Trade Frequency: {val_metrics['trade_frequency']:.2f}%")
                if 'long_survival_rate' in val_metrics:
                    logger.info(f"Long Survival Rate: {val_metrics['long_survival_rate']:.2f}%")
                if 'short_survival_rate' in val_metrics:
                    logger.info(f"Short Survival Rate: {val_metrics['short_survival_rate']:.2f}%")
            
            # Update history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.history['val_tradeable_accuracy'].append(val_metrics['tradeable_accuracy'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['avg_loss'])
            
            # Early stopping check
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt', val_metrics)
                logger.info("✓ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)
        
        # Save final model and history
        self._save_checkpoint('final_model.pt', val_metrics)
        self._save_history()
        
        return val_metrics
    
    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def _save_history(self):
        """Save training history."""
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_best_model(self):
        """Load best model checkpoint."""
        checkpoint = torch.load(self.output_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['metrics']
```

---

## 7. Evaluation

### 7.1 File: `evaluation/evaluator.py`

```python
"""
Comprehensive Model Evaluation

Produces detailed analysis of model performance.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

from training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive model evaluator.
    
    Produces:
    - Classification metrics (per class and overall)
    - Confusion matrix analysis
    - Trading performance simulation
    - Detailed report
    """
    
    def __init__(
        self,
        model,
        test_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run full evaluation.
        
        Returns:
            (metrics_dict, predictions_df)
        """
        tracker = MetricTracker()
        all_predictions = []
        
        for batch in self.test_loader:
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            next_mae_long = batch['next_mae_long'].to(self.device)
            next_mae_short = batch['next_mae_short'].to(self.device)
            
            outputs = self.model(prices, features)
            preds = torch.argmax(outputs['logits'], dim=-1)
            probs = torch.softmax(outputs['logits'], dim=-1)
            
            tracker.update(
                preds=preds,
                targets=labels,
                probs=probs,
                next_return=next_return,
                next_mae_long=next_mae_long,
                next_mae_short=next_mae_short
            )
            
            # Store predictions for detailed analysis
            for i in range(len(preds)):
                all_predictions.append({
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'prob_class_0': probs[i, 0].item(),
                    'prob_class_1': probs[i, 1].item(),
                    'prob_class_2': probs[i, 2].item(),
                    'prob_class_3': probs[i, 3].item(),
                    'prob_class_4': probs[i, 4].item(),
                    'next_return': next_return[i].item(),
                    'next_mae_long': next_mae_long[i].item(),
                    'next_mae_short': next_mae_short[i].item(),
                })
        
        metrics = tracker.compute()
        trading_metrics = tracker.compute_trading_metrics()
        metrics.update(trading_metrics)
        
        predictions_df = pd.DataFrame(all_predictions)
        
        return metrics, predictions_df
    
    def generate_report(
        self,
        metrics: Dict,
        predictions_df: pd.DataFrame,
        output_path: Path
    ) -> str:
        """Generate comprehensive evaluation report."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("SPH-NET EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # === OVERALL METRICS ===
        lines.append("OVERALL CLASSIFICATION METRICS")
        lines.append("-" * 40)
        lines.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        lines.append(f"Macro Precision:    {metrics['macro_precision']:.4f}")
        lines.append(f"Macro Recall:       {metrics['macro_recall']:.4f}")
        lines.append(f"Macro F1:           {metrics['macro_f1']:.4f}")
        lines.append(f"Tradeable Accuracy: {metrics['tradeable_accuracy']:.4f}")
        lines.append("")
        
        # === PER-CLASS METRICS ===
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 40)
        class_names = ['HIGH_BULL', 'BULL', 'RANGE_BOUND', 'BEAR', 'LOW_BEAR']
        
        header = f"{'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}"
        lines.append(header)
        lines.append("-" * 50)
        
        for name in class_names:
            prec = metrics[f'{name}_precision']
            rec = metrics[f'{name}_recall']
            f1 = metrics[f'{name}_f1']
            sup = metrics[f'{name}_support']
            lines.append(f"{name:<15} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {sup:>8}")
        lines.append("")
        
        # === CONFUSION MATRIX ===
        lines.append("CONFUSION MATRIX")
        lines.append("-" * 40)
        cm = metrics['confusion_matrix']
        lines.append("Rows: True | Cols: Predicted")
        lines.append("")
        
        header = "          " + " ".join([f"{n[:7]:>8}" for n in class_names])
        lines.append(header)
        
        for i, name in enumerate(class_names):
            row = f"{name[:9]:<10}" + " ".join([f"{cm[i,j]:>8}" for j in range(5)])
            lines.append(row)
        lines.append("")
        
        # === TRADING METRICS ===
        lines.append("TRADING PERFORMANCE METRICS")
        lines.append("-" * 40)
        
        if 'trade_frequency' in metrics:
            lines.append(f"Trade Frequency: {metrics['trade_frequency']:.2f}%")
            lines.append("")
            
            lines.append("LONG TRADES (predicted HIGH_BULL):")
            if 'long_trades' in metrics:
                lines.append(f"  Total Trades:    {metrics['long_trades']}")
                lines.append(f"  Avg Return:      {metrics['long_avg_return']:.4f}%")
                lines.append(f"  Total Return:    {metrics['long_total_return']:.2f}%")
                lines.append(f"  Win Rate:        {metrics['long_win_rate']:.2f}%")
                lines.append(f"  Correct Class:   {metrics['long_correct_class']:.2f}%")
                if 'long_survival_rate' in metrics:
                    lines.append(f"  Survival Rate:   {metrics['long_survival_rate']:.2f}%")
            else:
                lines.append("  No long trades made")
            lines.append("")
            
            lines.append("SHORT TRADES (predicted LOW_BEAR):")
            if 'short_trades' in metrics:
                lines.append(f"  Total Trades:    {metrics['short_trades']}")
                lines.append(f"  Avg Return:      {metrics['short_avg_return']:.4f}%")
                lines.append(f"  Total Return:    {metrics['short_total_return']:.2f}%")
                lines.append(f"  Win Rate:        {metrics['short_win_rate']:.2f}%")
                lines.append(f"  Correct Class:   {metrics['short_correct_class']:.2f}%")
                if 'short_survival_rate' in metrics:
                    lines.append(f"  Survival Rate:   {metrics['short_survival_rate']:.2f}%")
            else:
                lines.append("  No short trades made")
            lines.append("")
            
            if 'combined_total_return' in metrics:
                lines.append("COMBINED:")
                lines.append(f"  Total Return:       {metrics['combined_total_return']:.2f}%")
                lines.append(f"  Return per Trade:   {metrics['avg_return_per_trade']:.4f}%")
        
        lines.append("")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save report
        with open(output_path / "evaluation_report.txt", 'w') as f:
            f.write(report_text)
        
        # Save metrics as JSON
        metrics_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Save predictions
        predictions_df.to_csv(output_path / "predictions.csv", index=False)
        
        return report_text
```

---

## 8. Main Scripts

### 8.1 File: `scripts/prepare_data.py`

```python
#!/usr/bin/env python
"""
Data Preparation Script

Loads raw data, computes labels and features, saves prepared datasets.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import json

from labeling.candle_classifier import CandleLabeler, LabelingConfig
from labeling.label_analyzer import LabelAnalyzer
from features.feature_pipeline import FeaturePipeline
from data.data_splitter import TemporalSplitter, SplitConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")
    OUTPUT_DIR = Path("prepared_data")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # === Load Data ===
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows")
    
    # Drop rows with NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna(subset=critical_cols)
    logger.info(f"After dropping NaN: {len(df)} rows")
    
    # === Labeling ===
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Labeling")
    logger.info("="*50)
    
    label_config = LabelingConfig(
        strong_move_threshold=0.015,
        weak_move_threshold=0.005,
        clean_path_mae_threshold=0.005
    )
    labeler = CandleLabeler(label_config)
    
    df_labeled = labeler.label_dataset(df)
    
    # Analyze labels
    analyzer = LabelAnalyzer(df_labeled)
    stats = analyzer.compute_statistics()
    
    logger.info("\nLabel Distribution:")
    for i in range(5):
        logger.info(f"  Class {i}: {stats[f'class_{i}_count']:,} ({stats[f'class_{i}_pct']:.2f}%)")
    
    balanced, msg = analyzer.check_class_balance()
    logger.info(f"\nBalance check: {msg}")
    
    validation = analyzer.validate_label_correctness()
    logger.info(f"Label validation: {'PASSED' if validation['all_valid'] else 'FAILED'}")
    
    # === Feature Engineering ===
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*50)
    
    feature_pipeline = FeaturePipeline()
    df_features = feature_pipeline.compute_all_features(df_labeled)
    
    # Validate features
    validation = feature_pipeline.validate_features(df_features)
    logger.info(f"Feature validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['issues']:
        for col, issue in validation['issues'].items():
            logger.warning(f"  {col}: {issue}")
    
    # === Drop Warmup Period ===
    warmup = feature_pipeline.get_warmup_periods()
    df_clean = df_features.iloc[warmup:].copy()
    logger.info(f"Dropped {warmup} warmup rows, remaining: {len(df_clean)}")
    
    # Drop rows with NaN labels (last row)
    df_clean = df_clean[df_clean['label'].notna()]
    logger.info(f"After dropping NaN labels: {len(df_clean)}")
    
    # === Split Data ===
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Data Splitting")
    logger.info("="*50)
    
    splitter = TemporalSplitter(SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    ))
    
    train_df, val_df, test_df = splitter.split(df_clean, 'timestamp')
    
    # Verify no leakage
    no_leakage = splitter.verify_no_leakage(train_df, val_df, test_df, 'timestamp')
    assert no_leakage, "TEMPORAL LEAKAGE DETECTED!"
    
    # Get split summary
    summary = splitter.get_split_summary(train_df, val_df, test_df)
    
    logger.info("\nSplit Summary:")
    for split_name, split_stats in summary.items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Samples: {split_stats['n_samples']}")
        logger.info(f"  Label Distribution: {split_stats['label_pct']}")
    
    # === Normalize Features ===
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Feature Normalization")
    logger.info("="*50)
    
    # Normalize using training data statistics
    price_cols, eng_cols = feature_pipeline.get_feature_columns()
    
    train_normalized, norm_stats = feature_pipeline.normalize_features(
        train_df, fit_data=train_df
    )
    val_normalized = feature_pipeline.apply_normalization(val_df, norm_stats)
    test_normalized = feature_pipeline.apply_normalization(test_df, norm_stats)
    
    logger.info(f"Normalized {len(norm_stats)} features")
    
    # === Save Prepared Data ===
    logger.info("\n" + "="*50)
    logger.info("STEP 5: Saving Prepared Data")
    logger.info("="*50)
    
    train_normalized.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_normalized.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test_normalized.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    
    # Save normalization stats
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # Save feature columns
    feature_info = {
        'price_columns': price_cols,
        'engineered_columns': eng_cols,
        'all_columns': list(train_normalized.columns)
    }
    with open(OUTPUT_DIR / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save label statistics
    with open(OUTPUT_DIR / "label_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nSaved prepared data to {OUTPUT_DIR}/")
    logger.info("Files created:")
    logger.info("  - train.parquet")
    logger.info("  - val.parquet")
    logger.info("  - test.parquet")
    logger.info("  - normalization_stats.json")
    logger.info("  - feature_info.json")
    logger.info("  - label_stats.json")
    
    logger.info("\n" + "="*50)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*50)


if __name__ == "__main__":
    main()
```

### 8.2 File: `scripts/train.py`

```python
#!/usr/bin/env python
"""
Training Script for SPH-Net

Trains the model on prepared data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import json
import logging
import torch

from sph_net.config import SPHNetConfig, ExperimentConfig
from sph_net.models.sph_net import SPHNet
from data.dataset import create_dataloaders
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    OUTPUT_DIR = Path("experiments/run_001")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # === Load Prepared Data ===
    logger.info("Loading prepared data...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    
    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)
    
    price_cols = feature_info['price_columns']
    eng_cols = feature_info['engineered_columns']
    
    # Filter to available columns
    eng_cols = [c for c in eng_cols if c in train_df.columns]
    
    logger.info(f"Price features: {len(price_cols)}")
    logger.info(f"Engineered features: {len(eng_cols)}")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    
    # === Create Model Config ===
    model_config = SPHNetConfig(
        n_price_features=len(price_cols),
        n_engineered_features=len(eng_cols),
        n_classes=5,
        window_size=64,
        d_model=128,
        n_heads=8,
        n_encoder_layers=3,
        dropout=0.1,
        batch_size=64,
        learning_rate=1e-4,
        epochs=100,
        patience=15,
        class_weights=[2.0, 1.0, 0.5, 1.0, 2.0],
        focal_gamma=2.0,
        device='cuda'
    )
    
    # === Create DataLoaders ===
    logger.info("Creating DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, val_df,  # Use val as placeholder for test
        price_columns=price_cols,
        feature_columns=eng_cols,
        window_size=model_config.window_size,
        batch_size=model_config.batch_size
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # === Create Model ===
    logger.info("Creating model...")
    model = SPHNet(model_config)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    
    # === Train ===
    logger.info("\nStarting training...")
    trainer = Trainer(
        model=model,
        config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(OUTPUT_DIR)
    )
    
    final_metrics = trainer.train()
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Final macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### 8.3 File: `scripts/evaluate.py`

```python
#!/usr/bin/env python
"""
Evaluation Script for SPH-Net

Evaluates trained model on test set.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import json
import logging
import torch

from sph_net.config import SPHNetConfig
from sph_net.models.sph_net import SPHNet
from data.dataset import TradingDataset
from torch.utils.data import DataLoader
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    MODEL_DIR = Path("experiments/run_001")
    OUTPUT_DIR = MODEL_DIR / "evaluation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # === Load Test Data ===
    logger.info("Loading test data...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    
    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)
    
    price_cols = feature_info['price_columns']
    eng_cols = [c for c in feature_info['engineered_columns'] if c in test_df.columns]
    
    logger.info(f"Test samples: {len(test_df)}")
    
    # === Load Model ===
    logger.info("Loading model...")
    checkpoint = torch.load(MODEL_DIR / "best_model.pt")
    config = checkpoint['config']
    
    model = SPHNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # === Create Test DataLoader ===
    test_dataset = TradingDataset(
        test_df, price_cols, eng_cols,
        window_size=config.window_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # === Evaluate ===
    logger.info("\nEvaluating model...")
    evaluator = Evaluator(model, test_loader, device=config.device)
    metrics, predictions_df = evaluator.evaluate()
    
    # === Generate Report ===
    logger.info("\nGenerating report...")
    report = evaluator.generate_report(metrics, predictions_df, OUTPUT_DIR)
    
    print("\n" + report)
    
    logger.info(f"\nEvaluation results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

---

## 9. Execution Order

Run the scripts in this order:

```bash
# 1. Prepare data (labeling, features, splitting)
python scripts/prepare_data.py

# 2. Train model
python scripts/train.py

# 3. Evaluate on test set
python scripts/evaluate.py
```

---

## 10. Key Implementation Notes

### 10.1 NO LOOK-AHEAD BIAS

The most critical aspect of this implementation:

1. **Labeling**: Labels at row T describe what happens at T+1 (using `shift(-1)`)
2. **Features**: All technical indicators use only data from `<= current time`
3. **Normalization**: Test data is normalized using TRAINING statistics
4. **Splitting**: Chronological order, no shuffling between splits

### 10.2 Class Imbalance Handling

- **Focal Loss**: Down-weights easy examples, focuses on hard ones
- **Class Weights**: Up-weight rare classes (HIGH_BULL, LOW_BEAR)
- **Stratified Logging**: Track per-class metrics, not just overall accuracy

### 10.3 Trading-Relevant Metrics

Beyond classification metrics, track:

- **Survival Rate**: Would trades have survived stop-loss?
- **Trade Frequency**: Is model selective or trading too often?
- **Return Analysis**: Actual P&L if predictions were traded

### 10.4 Feature Groups

The engineered features cover:

1. **Returns**: Historical price changes
2. **Volatility**: ATR, rolling std
3. **Momentum**: RSI, MACD, ROC
4. **Trend**: Moving averages, slopes
5. **Volume**: Relative volume, OBV
6. **Order Flow**: Taker buy ratio, funding rate
7. **Session**: Hour, day of week (cyclically encoded)
8. **Sentiment**: Fear & Greed index

### 10.5 Expected Results

With 1 year of hourly BTC data (~8,700 samples):

- **Train**: ~6,000 samples
- **Val**: ~1,300 samples  
- **Test**: ~1,300 samples

Expected class distribution:
- HIGH_BULL: ~5-8%
- BULL: ~20-25%
- RANGE_BOUND: ~40-45%
- BEAR: ~20-25%
- LOW_BEAR: ~5-8%

Target metrics:
- Overall accuracy: 45-55% (5-class is hard!)
- Macro F1: 0.35-0.45
- Tradeable accuracy: 50-65%
- Survival rate: >70%

---

## 11. Troubleshooting

### Issue: Too few HIGH_BULL/LOW_BEAR samples

**Solution**: Lower thresholds in `LabelingConfig`:
```python
LabelingConfig(
    strong_move_threshold=0.012,  # Lower from 0.015
    weak_move_threshold=0.004,   # Lower from 0.005
)
```

### Issue: Model always predicts RANGE_BOUND

**Solution**: 
1. Increase class weights for HIGH_BULL/LOW_BEAR
2. Lower focal_gamma to reduce focus on hard examples
3. Check if features have signal (plot feature distributions by class)

### Issue: NaN in loss during training

**Solution**:
1. Ensure all features are normalized
2. Replace NaN with 0 in dataset `__getitem__`
3. Add gradient clipping (already included)
4. Reduce learning rate

### Issue: Validation loss not decreasing

**Solution**:
1. Check for data leakage (verify split timestamps)
2. Ensure train/val have similar class distributions
3. Try simpler model (fewer layers, smaller d_model)

---

## 12. File Checklist

Ensure all these files are created:

```
✓ labeling/__init__.py
✓ labeling/candle_classifier.py
✓ labeling/label_analyzer.py
✓ features/__init__.py
✓ features/technical_indicators.py
✓ features/feature_pipeline.py
✓ data/__init__.py
✓ data/data_splitter.py
✓ data/dataset.py
✓ sph_net/config.py (updated)
✓ sph_net/models/heads.py (updated)
✓ sph_net/models/sph_net.py (updated)
✓ training/__init__.py
✓ training/losses.py
✓ training/metrics.py
✓ training/trainer.py
✓ evaluation/__init__.py
✓ evaluation/evaluator.py
✓ scripts/prepare_data.py
✓ scripts/train.py
✓ scripts/evaluate.py
```

---

This implementation is designed to be:
- **Rigorous**: No look-ahead bias, proper temporal splitting
- **Interpretable**: Detailed metrics and reports
- **Trading-focused**: Evaluates tradeability, not just direction
- **Production-ready**: Clean code, proper logging, checkpointing