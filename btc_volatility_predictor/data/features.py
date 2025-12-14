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


def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) - Trend Strength Indicator.

    ADX values interpretation:
    - 0-20: Weak or absent trend (ranging market)
    - 20-25: Emerging trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)

    Returns:
        tuple: (adx, plus_di, minus_di)
            - adx: Average Directional Index (0-100)
            - plus_di: Positive Directional Indicator
            - minus_di: Negative Directional Indicator
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed True Range (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smoothed +DM and -DM
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Directional Indicators (+DI and -DI)
    plus_di = 100 * (plus_dm_smooth / (atr + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (atr + 1e-10))

    # DX (Directional Movement Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    # ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


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

    # --- ADX (Trend Strength) ---
    features['adx_14'], features['plus_di_14'], features['minus_di_14'] = calc_adx(df, 14)
    features['adx_21'], _, _ = calc_adx(df, 21)

    # ADX-derived features
    features['adx_trend_strength'] = features['adx_14'] / 50  # Normalized to ~0-2 range
    features['di_diff'] = features['plus_di_14'] - features['minus_di_14']  # Directional bias

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


def prepare_all_historical_datasets(
    input_dir: str = "data/raw/historical",
    output_dir: str = "data/processed/historical"
) -> dict:
    """
    Prepare feature datasets for all historical periods.

    Processes raw historical CSV files into feature-engineered datasets
    ready for backtesting.

    Args:
        input_dir: Directory containing raw historical CSV files
        output_dir: Directory to save processed feature files

    Returns:
        Dict mapping period names to output paths
    """
    periods = ['2018_bear', '2019_2020_recovery', '2023_2024_bull']

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("=" * 60)
    print("HISTORICAL FEATURE ENGINEERING")
    print("=" * 60)

    for period in periods:
        input_path = os.path.join(input_dir, f"btcusdt_1h_{period}.csv")
        output_path = os.path.join(output_dir, f"features_{period}.csv")

        print(f"\nProcessing {period}...")

        if os.path.exists(input_path):
            try:
                features = prepare_dataset(input_path, output_path)
                results[period] = {
                    'path': output_path,
                    'samples': len(features),
                    'status': 'success'
                }
                print(f"  Successfully processed {len(features)} samples")
            except Exception as e:
                results[period] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"  ERROR: {e}")
        else:
            results[period] = {
                'status': 'not_found',
                'error': f'Input file not found: {input_path}'
            }
            print(f"  SKIPPED: Input file not found at {input_path}")

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    print(f"Successfully processed: {successful}/{len(periods)} periods")

    for period, result in results.items():
        if result.get('status') == 'success':
            print(f"  ✓ {period}: {result['samples']} samples")
        else:
            print(f"  ✗ {period}: {result.get('error', 'unknown error')}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--historical":
        # Process all historical datasets
        prepare_all_historical_datasets()
    else:
        # Default: process current data
        prepare_dataset("data/raw/btcusdt_1h.csv")
