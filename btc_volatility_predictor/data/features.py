"""
Feature engineering for BTC volatility prediction.
Includes volatility estimators and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import os


# =============================================================================
# VOLATILITY TARGETS (What we're predicting)
# =============================================================================

def calc_garman_klass_volatility(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Garman-Klass volatility estimator - 7.4x more efficient than close-to-close.
    Uses OHLC data for better intraday volatility estimation.

    Formula: sigma^2 = 0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    if window > 1:
        gk_var = gk_var.rolling(window=window).mean()

    return np.sqrt(gk_var)


def calc_parkinson_volatility(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Parkinson volatility - uses high-low range.
    Formula: sigma^2 = ln(H/L)^2 / (4 * ln(2))
    """
    log_hl = np.log(df['high'] / df['low'])
    park_var = log_hl**2 / (4 * np.log(2))

    if window > 1:
        park_var = park_var.rolling(window=window).mean()

    return np.sqrt(park_var)


def calc_rogers_satchell_volatility(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Rogers-Satchell volatility - handles non-zero drift.
    Formula: sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])

    rs_var = log_hc * log_ho + log_lc * log_lo

    if window > 1:
        rs_var = rs_var.rolling(window=window).mean()

    return np.sqrt(np.abs(rs_var))


def calc_yang_zhang_volatility(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Yang-Zhang volatility - most robust estimator.
    Handles overnight gaps and drift. Minimum estimation error.
    """
    log_oc = np.log(df['open'] / df['close'].shift(1))  # Overnight return
    log_co = np.log(df['close'] / df['open'])  # Open-to-close

    # Rogers-Satchell component
    rs = calc_rogers_satchell_volatility(df, window=1)**2

    # Overnight variance
    overnight_var = log_oc.rolling(window=window).var()

    # Open-to-close variance
    oc_var = log_co.rolling(window=window).var()

    # Yang-Zhang combination (k = 0.34 is optimal)
    k = 0.34
    yz_var = overnight_var + k * oc_var + (1 - k) * rs.rolling(window=window).mean()

    return np.sqrt(yz_var)


def calc_realized_volatility(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Realized volatility from log returns.
    Standard deviation of returns over rolling window.
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std()


# =============================================================================
# TECHNICAL INDICATORS (Features)
# =============================================================================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands - returns middle, upper, lower"""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def calc_bollinger_bandwidth(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bandwidth - volatility measure"""
    middle, upper, lower = calc_bollinger_bands(close, period, std_dev)
    return (upper - lower) / (middle + 1e-10)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - volatility indicator"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD - trend and momentum"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume - volume trend"""
    obv = np.where(
        df['close'] > df['close'].shift(1),
        df['volume'],
        np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)
    )
    return pd.Series(obv, index=df.index).cumsum()


def calc_vwap(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)


def calc_price_momentum(close: pd.Series, periods: list = [1, 6, 12, 24]) -> pd.DataFrame:
    """Price momentum at different lookbacks"""
    momentum = pd.DataFrame()
    for p in periods:
        momentum[f'momentum_{p}h'] = close.pct_change(p)
    return momentum


def calc_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features"""
    features = pd.DataFrame(index=df.index)

    # Volume momentum
    features['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-10)

    # Volume change
    features['volume_change'] = df['volume'].pct_change()

    # Quote volume ratio (buying pressure proxy)
    if 'quote_volume' in df.columns:
        features['quote_volume_ratio'] = df['quote_volume'] / (df['volume'] * df['close'] + 1e-10)

    return features


# =============================================================================
# FULL FEATURE PIPELINE
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Returns DataFrame with all features + target volatility.
    """
    features = df.copy()

    # --- PRICE FEATURES ---
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    features['log_return_abs'] = np.abs(features['log_return'])

    # Price momentum at multiple horizons
    momentum = calc_price_momentum(df['close'], [1, 6, 12, 24, 48])
    features = pd.concat([features, momentum], axis=1)

    # High-Low range (normalized)
    features['hl_range'] = (df['high'] - df['low']) / df['close']
    features['oc_range'] = np.abs(df['close'] - df['open']) / df['open']

    # --- VOLATILITY FEATURES (Historical) ---
    features['vol_gk_1h'] = calc_garman_klass_volatility(df, window=1)
    features['vol_gk_6h'] = calc_garman_klass_volatility(df, window=6)
    features['vol_gk_24h'] = calc_garman_klass_volatility(df, window=24)

    features['vol_park_1h'] = calc_parkinson_volatility(df, window=1)
    features['vol_park_24h'] = calc_parkinson_volatility(df, window=24)

    features['vol_rs_1h'] = calc_rogers_satchell_volatility(df, window=1)
    features['vol_rs_24h'] = calc_rogers_satchell_volatility(df, window=24)

    features['vol_yz_24h'] = calc_yang_zhang_volatility(df, window=24)
    features['vol_realized_24h'] = calc_realized_volatility(df, window=24)

    # Volatility of volatility
    features['vol_of_vol'] = features['vol_gk_1h'].rolling(24).std()

    # --- TECHNICAL INDICATORS ---
    features['rsi_14'] = calc_rsi(df['close'], 14)
    features['rsi_6'] = calc_rsi(df['close'], 6)

    # Bollinger features
    features['bb_bandwidth_20'] = calc_bollinger_bandwidth(df['close'], 20)
    _, upper, lower = calc_bollinger_bands(df['close'], 20)
    features['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)

    # ATR (normalized)
    features['atr_14'] = calc_atr(df, 14) / df['close']
    features['atr_24'] = calc_atr(df, 24) / df['close']

    # MACD
    macd, signal, hist = calc_macd(df['close'])
    features['macd_hist'] = hist / df['close']

    # --- VOLUME FEATURES ---
    vol_features = calc_volume_features(df)
    features = pd.concat([features, vol_features], axis=1)

    # OBV momentum
    obv = calc_obv(df)
    features['obv_momentum'] = obv.pct_change(24)

    # VWAP deviation
    features['vwap_deviation'] = (df['close'] - calc_vwap(df, 24)) / df['close']

    # --- TIME FEATURES ---
    if 'timestamp' in df.columns:
        features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

    # --- TARGET: Next hour Garman-Klass volatility ---
    features['target_volatility'] = calc_garman_klass_volatility(df, window=1).shift(-1)

    # Also include direction for auxiliary classification
    features['target_vol_direction'] = (features['target_volatility'] > features['vol_gk_1h']).astype(float)

    return features


def prepare_dataset(csv_path: str, output_path: str = "data/processed/features.csv"):
    """Load raw data, engineer features, save processed dataset."""

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Engineering features...")
    features = engineer_features(df)

    # Drop rows with NaN (from rolling calculations)
    initial_len = len(features)
    features = features.dropna()
    print(f"Dropped {initial_len - len(features)} rows with NaN values")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} samples to {output_path}")

    return features


if __name__ == "__main__":
    prepare_dataset("data/raw/btcusdt_1h.csv")
