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
            if 'log_return_1h' in result.columns:
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
        if 'order_flow_imbalance' in result.columns:
            for period in [6, 12, 24]:
                result[f'ofi_ma_{period}'] = (
                    result['order_flow_imbalance']
                    .rolling(window=period, min_periods=period)
                    .mean()
                )
        else:
            # Create placeholder if not available
            for period in [6, 12, 24]:
                result[f'ofi_ma_{period}'] = 0.0

        # Taker buy ratio moving average
        if 'taker_buy_ratio' in result.columns:
            for period in [12, 24]:
                result[f'tbr_ma_{period}'] = (
                    result['taker_buy_ratio']
                    .rolling(window=period, min_periods=period)
                    .mean()
                )
        else:
            for period in [12, 24]:
                result[f'tbr_ma_{period}'] = 0.5

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
        else:
            result['funding_rate'] = 0.0
            result['funding_rate_ma'] = 0.0
            result['funding_rate_zscore'] = 0.0

        return result

    @staticmethod
    def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute session-based features.

        Cyclical encoding for hour and day_of_week.
        """
        result = df.copy()

        # Extract hour and day_of_week if not present
        if 'hour' not in result.columns:
            if 'timestamp' in result.columns:
                result['hour'] = pd.to_datetime(result['timestamp']).dt.hour
            else:
                result['hour'] = 0

        if 'day_of_week' not in result.columns:
            if 'timestamp' in result.columns:
                result['day_of_week'] = pd.to_datetime(result['timestamp']).dt.dayofweek
            else:
                result['day_of_week'] = 0

        # Cyclical encoding for hour (0-23)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

        # Cyclical encoding for day of week (0-6)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

        # Session one-hot (if session column exists)
        if 'session' in result.columns:
            result['is_asian'] = (result['session'] == 'asian').astype(int)
            result['is_london'] = (result['session'] == 'london').astype(int)
            result['is_newyork'] = (result['session'] == 'new_york').astype(int)
        else:
            result['is_asian'] = 0
            result['is_london'] = 0
            result['is_newyork'] = 0

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
        else:
            result['fng_normalized'] = 0.5
            result['fng_ma_7d'] = 50.0
            result['fng_deviation'] = 0.0

        return result
