#!/usr/bin/env python3
"""
Feature engineering for trade quality prediction.

This module defines the features used for training and inference
of the XGBoost trade quality classifier.
"""

from typing import Dict, List
import numpy as np


# Core features from the existing feature set (from features_365d.csv)
CORE_FEATURES = [
    'rsi_14', 'rsi_6',
    'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24',
    'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h',
    'vol_realized_24h', 'vol_of_vol',
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h',
    'macd_hist',
    'volume_ma_ratio', 'volume_change',
    'adx_14', 'plus_di_14', 'minus_di_14',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

# Additional computed features for trade entry
COMPUTED_FEATURES = [
    'trend_strength',       # SMA_fast / SMA_slow ratio
    'price_vs_sma_fast',    # close / SMA_72
    'price_vs_sma_slow',    # close / SMA_168
    'rsi_slope_3h',         # RSI change over last 3 hours
    'vol_regime_prob',      # Model's confidence in LOW vol
]

# All features for training/inference
TRADE_FEATURES = CORE_FEATURES + COMPUTED_FEATURES


def compute_additional_features(row: dict, history: List[dict]) -> Dict[str, float]:
    """
    Compute additional features for trade entry point.

    These features are computed at runtime from the current bar
    and price history, providing additional context for the ML model.

    Args:
        row: Current bar data (dict with OHLCV and indicators)
        history: List of previous bars (most recent last)

    Returns:
        Dictionary of computed feature values
    """
    features = {}

    close = row.get('close', 0)

    # Trend strength features
    if len(history) >= 168:
        # Fast SMA (72 hours = 3 days)
        fast_prices = [h.get('close', 0) for h in history[-72:]]
        sma_72 = np.mean(fast_prices) if fast_prices else 0

        # Slow SMA (168 hours = 7 days)
        slow_prices = [h.get('close', 0) for h in history[-168:]]
        sma_168 = np.mean(slow_prices) if slow_prices else 0

        # Trend strength: ratio of fast/slow SMA
        features['trend_strength'] = sma_72 / sma_168 if sma_168 > 0 else 1.0

        # Price position relative to SMAs
        features['price_vs_sma_fast'] = close / sma_72 if sma_72 > 0 else 1.0
        features['price_vs_sma_slow'] = close / sma_168 if sma_168 > 0 else 1.0
    else:
        features['trend_strength'] = 1.0
        features['price_vs_sma_fast'] = 1.0
        features['price_vs_sma_slow'] = 1.0

    # RSI slope (change over last 3 hours)
    if len(history) >= 3:
        rsi_now = row.get('rsi_14', 50)
        rsi_3h_ago = history[-3].get('rsi_14', 50)
        features['rsi_slope_3h'] = rsi_now - rsi_3h_ago
    else:
        features['rsi_slope_3h'] = 0.0

    # Volatility regime probability (from predictions if available)
    # Default to 0.5 (neutral) if not available
    features['vol_regime_prob'] = row.get('prediction_prob', 0.5)

    return features


def extract_all_features(row: dict, history: List[dict]) -> Dict[str, float]:
    """
    Extract all features (core + computed) for a single data point.

    Args:
        row: Current bar data
        history: Previous bars

    Returns:
        Dictionary with all feature values
    """
    features = {}

    # Extract core features
    for feat in CORE_FEATURES:
        features[feat] = row.get(feat, 0.0)

    # Add computed features
    computed = compute_additional_features(row, history)
    features.update(computed)

    return features


def features_to_array(features: Dict[str, float], feature_names: List[str] = None) -> np.ndarray:
    """
    Convert feature dictionary to numpy array in correct order.

    Args:
        features: Dictionary of feature values
        feature_names: List of feature names in desired order
                      (defaults to TRADE_FEATURES)

    Returns:
        Numpy array of feature values
    """
    if feature_names is None:
        feature_names = TRADE_FEATURES

    return np.array([features.get(f, 0.0) for f in feature_names])
