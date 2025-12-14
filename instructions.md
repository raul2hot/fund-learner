# BTC/USDT 1-Hour Volatility Predictor - Claude Code Instructions

## Project Overview

Build an accurate BTC/USDT volatility predictor using the SPH-Net Hybrid Transformer architecture. The model should:

- Use **180 days of 1-hour candle data** (~4,320 candles)
- Predict **1-hour ahead volatility** using n-1 candles
- Evaluate on **last 10 days** (240 candles) as unseen test set
- Train on first 170 days (~4,080 candles)

---

## Phase 1: Project Setup

### 1.1 Copy Boilerplate Structure

```bash
# Create project directory
mkdir -p btc_volatility_predictor
cd btc_volatility_predictor

# Copy existing SPH-Net boilerplate
cp -r /path/to/sph_net/* .

# Create additional directories
mkdir -p data/raw data/processed checkpoints logs
```

### 1.2 Update Requirements

Create `requirements.txt`:

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
requests>=2.28.0
ta>=0.10.2
python-binance>=1.0.19
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Phase 2: Data Pipeline

### 2.1 Create Data Fetcher

Create `data/fetch_binance.py`:

```python
"""
Fetch BTC/USDT 1h OHLCV data from Binance public API.
No API key required for historical klines.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180,
    save_path: str = "data/raw/btcusdt_1h.csv"
):
    """
    Fetch historical klines from Binance API.
    
    Args:
        symbol: Trading pair (default BTCUSDT)
        interval: Candle interval (1h for hourly)
        days: Number of days of history to fetch
        save_path: Where to save the CSV
    
    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    print(f"Fetching {days} days of {interval} data for {symbol}...")
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000  # Max per request
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        
        if not data:
            break
            
        all_klines.extend(data)
        current_start = data[-1][0] + 1  # Next millisecond after last candle
        
        print(f"  Fetched {len(all_klines)} candles...")
        time.sleep(0.1)  # Rate limiting
    
    # Convert to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]
    
    df = pd.DataFrame(all_klines, columns=columns)
    
    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
    
    df['trades'] = df['trades'].astype(int)
    
    # Keep essential columns
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'trades', 'quote_volume']]
    df = df.rename(columns={'open_time': 'timestamp'})
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} candles to {save_path}")
    
    return df


if __name__ == "__main__":
    df = fetch_binance_klines(days=180)
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### 2.2 Create Feature Engineering Module

Create `data/features.py`:

```python
"""
Feature engineering for BTC volatility prediction.
Includes volatility estimators and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# =============================================================================
# VOLATILITY TARGETS (What we're predicting)
# =============================================================================

def calc_garman_klass_volatility(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Garman-Klass volatility estimator - 7.4x more efficient than close-to-close.
    Uses OHLC data for better intraday volatility estimation.
    
    Formula: σ² = 0.5 * ln(H/L)² - (2ln2 - 1) * ln(C/O)²
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
    Formula: σ² = ln(H/L)² / (4 * ln(2))
    """
    log_hl = np.log(df['high'] / df['low'])
    park_var = log_hl**2 / (4 * np.log(2))
    
    if window > 1:
        park_var = park_var.rolling(window=window).mean()
    
    return np.sqrt(park_var)


def calc_rogers_satchell_volatility(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """
    Rogers-Satchell volatility - handles non-zero drift.
    Formula: σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
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
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} samples to {output_path}")
    
    return features


if __name__ == "__main__":
    prepare_dataset("data/raw/btcusdt_1h.csv")
```

### 2.3 Create Dataset Class

Update `data/dataset.py`:

```python
"""
PyTorch Dataset for BTC volatility prediction.
Walk-forward split for time series integrity.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, List
import pickle
import os


class BTCVolatilityDataset(Dataset):
    """
    Dataset for BTC/USDT volatility prediction.
    
    Features are split into:
    - price_features: OHLCV and derived price metrics
    - engineered_features: Technical indicators and volume metrics
    
    Target: Next-hour Garman-Klass volatility
    """
    
    # Define feature groups
    PRICE_COLS = ['open', 'high', 'low', 'close', 'volume', 'log_return', 
                  'hl_range', 'oc_range', 'log_return_abs']
    
    ENGINEERED_COLS = [
        # Volatility features
        'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
        'vol_park_1h', 'vol_park_24h',
        'vol_rs_1h', 'vol_rs_24h',
        'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
        # Momentum
        'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
        # Technical indicators
        'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
        'atr_14', 'atr_24', 'macd_hist',
        # Volume features
        'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
        # Time features
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 72,  # 3 days of hourly data
        horizon: int = 1,
        price_scaler=None,
        feature_scaler=None,
        target_scaler=None,
        fit_scalers: bool = False
    ):
        """
        Args:
            df: DataFrame with all features and targets
            window_size: Number of past candles to use as input
            horizon: Forecast horizon (1 = next hour)
            price_scaler: Fitted scaler for price features (or None to create)
            feature_scaler: Fitted scaler for engineered features
            target_scaler: Fitted scaler for target volatility
            fit_scalers: Whether to fit scalers on this data
        """
        self.window_size = window_size
        self.horizon = horizon
        
        # Get available columns
        self.price_cols = [c for c in self.PRICE_COLS if c in df.columns]
        self.eng_cols = [c for c in self.ENGINEERED_COLS if c in df.columns]
        
        # Extract arrays
        self.prices = df[self.price_cols].values.astype(np.float32)
        self.features = df[self.eng_cols].values.astype(np.float32)
        self.targets = df['target_volatility'].values.astype(np.float32)
        self.target_direction = df['target_vol_direction'].values.astype(np.float32)
        
        # Handle scalers
        if fit_scalers:
            self.price_scaler = RobustScaler()
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
            
            self.prices = self.price_scaler.fit_transform(self.prices)
            self.features = self.feature_scaler.fit_transform(self.features)
            self.targets = self.target_scaler.fit_transform(
                self.targets.reshape(-1, 1)
            ).flatten()
        else:
            self.price_scaler = price_scaler
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            
            if price_scaler is not None:
                self.prices = self.price_scaler.transform(self.prices)
            if feature_scaler is not None:
                self.features = self.feature_scaler.transform(self.features)
            if target_scaler is not None:
                self.targets = self.target_scaler.transform(
                    self.targets.reshape(-1, 1)
                ).flatten()
        
        # Valid indices
        self.n_samples = len(self.prices) - window_size - horizon
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Input window
        price_window = self.prices[idx:idx + self.window_size]
        feat_window = self.features[idx:idx + self.window_size]
        
        # Target (next timestep after window)
        target_idx = idx + self.window_size
        target_vol = self.targets[target_idx]
        target_dir = self.target_direction[target_idx]
        
        return {
            'prices': torch.tensor(price_window, dtype=torch.float32),
            'features': torch.tensor(feat_window, dtype=torch.float32),
            'target_volatility': torch.tensor(target_vol, dtype=torch.float32),
            'target_direction': torch.tensor(target_dir, dtype=torch.float32)
        }
    
    def get_scalers(self):
        """Return fitted scalers for use in validation/test sets."""
        return self.price_scaler, self.feature_scaler, self.target_scaler


def create_dataloaders(
    data_path: str = "data/processed/features.csv",
    window_size: int = 72,
    batch_size: int = 64,
    test_days: int = 10,  # Last 10 days for testing
    val_ratio: float = 0.15,  # 15% of training data for validation
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test dataloaders with walk-forward split.
    
    Split strategy:
    - Test: Last 10 days (240 hours)
    - Val: 15% of remaining data (before test)
    - Train: Everything else
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    n_total = len(df)
    n_test = test_days * 24  # 10 days = 240 hourly candles
    n_trainval = n_total - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val
    
    print(f"Total samples: {n_total}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Split data (walk-forward: train -> val -> test)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    # Create datasets (fit scalers only on training data)
    train_dataset = BTCVolatilityDataset(
        train_df, window_size=window_size, fit_scalers=True
    )
    
    # Get scalers from training set
    price_scaler, feature_scaler, target_scaler = train_dataset.get_scalers()
    
    val_dataset = BTCVolatilityDataset(
        val_df, window_size=window_size,
        price_scaler=price_scaler,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )
    
    test_dataset = BTCVolatilityDataset(
        test_df, window_size=window_size,
        price_scaler=price_scaler,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Metadata for model config
    metadata = {
        'n_price_features': len(train_dataset.price_cols),
        'n_engineered_features': len(train_dataset.eng_cols),
        'window_size': window_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'price_scaler': price_scaler,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, test_loader, metadata = create_dataloaders()
    
    print(f"\nMetadata: {metadata}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
```

---

## Phase 3: Model Architecture Updates

### 3.1 Update Config

Update `config.py`:

```python
"""SPH-Net Configuration for BTC Volatility Prediction"""

from dataclasses import dataclass

@dataclass
class Config:
    # Data
    window_size: int = 72           # 3 days of hourly data (72 hours)
    n_assets: int = 1               # Single asset (BTC/USDT)
    price_features: int = 9         # OHLCV + derived (will be set from data)
    engineered_features: int = 32   # Technical indicators (will be set from data)
    forecast_horizon: int = 1       # Predict 1 hour ahead

    # Model architecture
    d_model: int = 128              # Embedding dimension (increased for complexity)
    n_heads: int = 8                # Attention heads
    n_encoder_layers: int = 4       # Transformer layers (deeper)
    dropout: float = 0.15           # Slightly higher for regularization
    
    # Prediction heads
    use_uncertainty: bool = True    # Predict uncertainty for volatility

    # Training
    batch_size: int = 64
    learning_rate: float = 5e-4     # Lower LR for stability
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15              # Early stopping patience
    
    # Loss weights
    alpha_mse: float = 1.0          # Regression loss weight
    beta_ce: float = 0.3            # Classification loss weight (auxiliary)
    gamma_uncertainty: float = 0.1  # Uncertainty loss weight
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"    # "cosine" or "plateau"
    warmup_epochs: int = 5

    # Device
    device: str = "cuda"            # or "cpu"
    
    # Paths
    data_path: str = "data/processed/features.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def update_from_metadata(self, metadata: dict):
        """Update config from dataset metadata."""
        self.price_features = metadata.get('n_price_features', self.price_features)
        self.engineered_features = metadata.get('n_engineered_features', self.engineered_features)
        self.window_size = metadata.get('window_size', self.window_size)
```

### 3.2 Update SPH-Net Model

The existing model architecture is mostly suitable. Update `models/sph_net.py` to handle volatility prediction better:

```python
"""SPH-Net: Hybrid Transformer for BTC Volatility Prediction"""

import torch
import torch.nn as nn

from .encoders import TemporalEncoder, FeatureEncoder
from .attention import CoAttentionFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead


class SPHNet(nn.Module):
    """
    SPH-Net Hybrid Transformer for Volatility Prediction

    Architecture:
    1. Temporal Encoder (Transformer) for price sequences
    2. Feature Encoder (MLP) for engineered features  
    3. Co-Attention Fusion
    4. Prediction Heads (volatility regression, direction, uncertainty)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.price_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout
        )

        self.feature_encoder = FeatureEncoder(
            input_dim=config.engineered_features,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Co-attention fusion
        self.co_attention = CoAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Additional transformer block after fusion
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Temporal attention pooling (instead of just last token)
        self.temporal_pool = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Softmax(dim=1)
        )

        # Prediction heads
        self.regression_head = RegressionHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        self.classification_head = ClassificationHead(
            config.d_model, config.forecast_horizon, config.dropout
        )
        
        if config.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                config.d_model, config.forecast_horizon, config.dropout
            )
        else:
            self.uncertainty_head = None

    def forward(self, prices, features):
        """
        Args:
            prices: [batch, T, n_price_features]
            features: [batch, T, n_engineered_features]

        Returns:
            dict with 'volatility_pred', 'direction_pred', 'uncertainty'
        """
        # Encode both streams
        temporal_tokens = self.temporal_encoder(prices)      # [batch, T, d_model]
        feature_tokens = self.feature_encoder(features)      # [batch, T, d_model]

        # Fuse with co-attention
        fused = self.co_attention(temporal_tokens, feature_tokens)  # [batch, T, d_model]

        # Decode
        decoded = self.decoder(fused)  # [batch, T, d_model]

        # Attention-weighted pooling (focus on important timesteps)
        attn_weights = self.temporal_pool(decoded)  # [batch, T, 1]
        pooled = (decoded * attn_weights).sum(dim=1)  # [batch, d_model]
        
        # Also keep last token for comparison
        last_token = decoded[:, -1, :]  # [batch, d_model]
        
        # Combine pooled and last token
        combined = pooled + last_token  # Simple residual combination

        # Predictions
        volatility_pred = self.regression_head(combined)
        direction_pred = self.classification_head(combined)
        
        output = {
            'volatility_pred': volatility_pred,
            'direction_pred': direction_pred,
        }
        
        if self.uncertainty_head is not None:
            output['uncertainty'] = self.uncertainty_head(combined)

        return output
```

---

## Phase 4: Training Pipeline

### 4.1 Create Training Script

Create `train_volatility.py`:

```python
"""Training script for BTC Volatility Prediction with SPH-Net"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import Config
from models import SPHNet
from data.dataset import create_dataloaders


class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood loss for Gaussian with predicted variance."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_mean, pred_var, target):
        """
        Args:
            pred_mean: Predicted mean [batch, horizon]
            pred_var: Predicted variance [batch, horizon] (must be positive)
            target: Ground truth [batch, horizon]
        """
        # Clamp variance for numerical stability
        pred_var = pred_var.clamp(min=1e-6)
        
        # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
        nll = 0.5 * (torch.log(pred_var) + (target - pred_mean)**2 / pred_var)
        return nll.mean()


def train_epoch(model, dataloader, optimizer, criterion_reg, criterion_cls, 
                criterion_nll, config, device, scaler=None):
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    total_mse = 0
    total_bce = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].to(device).unsqueeze(-1)
        target_dir = batch['target_direction'].to(device).unsqueeze(-1)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(prices, features)

            # Compute losses
            loss_mse = criterion_reg(outputs['volatility_pred'], target_vol)
            loss_bce = criterion_cls(outputs['direction_pred'], target_dir)
            
            # Combined loss
            loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce
            
            # Add uncertainty loss if available
            if 'uncertainty' in outputs and config.use_uncertainty:
                loss_nll = criterion_nll(
                    outputs['volatility_pred'], 
                    outputs['uncertainty'], 
                    target_vol
                )
                loss += config.gamma_uncertainty * loss_nll

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_bce += loss_bce.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'bce': total_bce / n_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion_reg, criterion_cls, criterion_nll, 
             config, device, target_scaler=None):
    """Validation with metrics calculation."""
    model.eval()
    
    total_loss = 0
    total_mse = 0
    all_preds = []
    all_targets = []
    all_pred_dirs = []
    all_target_dirs = []
    n_batches = 0

    for batch in dataloader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].to(device).unsqueeze(-1)
        target_dir = batch['target_direction'].to(device).unsqueeze(-1)

        outputs = model(prices, features)

        loss_mse = criterion_reg(outputs['volatility_pred'], target_vol)
        loss_bce = criterion_cls(outputs['direction_pred'], target_dir)
        loss = config.alpha_mse * loss_mse + config.beta_ce * loss_bce

        total_loss += loss.item()
        total_mse += loss_mse.item()
        
        all_preds.append(outputs['volatility_pred'].cpu())
        all_targets.append(target_vol.cpu())
        all_pred_dirs.append(torch.sigmoid(outputs['direction_pred']).cpu())
        all_target_dirs.append(target_dir.cpu())
        
        n_batches += 1

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_pred_dirs = torch.cat(all_pred_dirs, dim=0).numpy()
    all_target_dirs = torch.cat(all_target_dirs, dim=0).numpy()
    
    # Calculate direction accuracy
    dir_acc = ((all_pred_dirs > 0.5) == all_target_dirs).mean()
    
    # Inverse transform for interpretable metrics
    if target_scaler is not None:
        all_preds_orig = target_scaler.inverse_transform(all_preds)
        all_targets_orig = target_scaler.inverse_transform(all_targets)
        
        # RMSE in original scale
        rmse_orig = np.sqrt(np.mean((all_preds_orig - all_targets_orig)**2))
        mae_orig = np.mean(np.abs(all_preds_orig - all_targets_orig))
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((all_targets_orig - all_preds_orig) / 
                             (all_targets_orig + 1e-8))) * 100
    else:
        rmse_orig = np.sqrt(np.mean((all_preds - all_targets)**2))
        mae_orig = np.mean(np.abs(all_preds - all_targets))
        mape = 0

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'rmse': rmse_orig,
        'mae': mae_orig,
        'mape': mape,
        'direction_acc': dir_acc
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, 
                                     steps_per_epoch):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        
        if step < warmup_steps:
            return step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # Configuration
    config = Config()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_path=config.data_path,
        window_size=config.window_size,
        batch_size=config.batch_size
    )
    
    # Update config from data metadata
    config.update_from_metadata(metadata)
    print(f"Price features: {config.price_features}")
    print(f"Engineered features: {config.engineered_features}")
    
    # Model
    model = SPHNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer & Schedulers
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config.warmup_epochs, 
        config.epochs,
        len(train_loader)
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Loss functions
    criterion_reg = nn.HuberLoss()  # Robust to outliers
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_nll = GaussianNLLLoss()
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{config.log_dir}/run_{timestamp}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, 
            criterion_reg, criterion_cls, criterion_nll,
            config, device, scaler
        )
        
        # Step scheduler
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.6f}, "
              f"MSE: {train_metrics['mse']:.6f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_reg, criterion_cls, criterion_nll,
            config, device, metadata['target_scaler']
        )
        
        print(f"Val - Loss: {val_metrics['loss']:.6f}, "
              f"RMSE: {val_metrics['rmse']:.6f}, "
              f"MAE: {val_metrics['mae']:.6f}, "
              f"MAPE: {val_metrics['mape']:.2f}%, "
              f"Dir Acc: {val_metrics['direction_acc']:.4f}")
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalar('Val/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Val/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Val/MAPE', val_metrics['mape'], epoch)
        writer.add_scalar('Val/Direction_Acc', val_metrics['direction_acc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config.__dict__
            }
            torch.save(checkpoint, f"{config.checkpoint_dir}/best_model.pt")
            print("✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(f"{config.checkpoint_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(
        model, test_loader,
        criterion_reg, criterion_cls, criterion_nll,
        config, device, metadata['target_scaler']
    )
    
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"Test Direction Accuracy: {test_metrics['direction_acc']:.4f}")
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'best_val_metrics': checkpoint['val_metrics'],
        'config': config.__dict__,
        'n_params': n_params
    }
    
    with open(f"{config.checkpoint_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## Phase 5: Evaluation & Visualization

### 5.1 Create Evaluation Script

Create `evaluate.py`:

```python
"""Evaluation and visualization for BTC volatility predictor."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from config import Config
from models import SPHNet
from data.dataset import create_dataloaders


def load_model(checkpoint_path: str, config: Config, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Update config if stored
    if 'config' in checkpoint:
        for k, v in checkpoint['config'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    model = SPHNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


@torch.no_grad()
def predict_all(model, dataloader, device, target_scaler=None):
    """Get all predictions from dataloader."""
    all_preds = []
    all_targets = []
    all_uncertainties = []
    
    for batch in dataloader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        target_vol = batch['target_volatility'].numpy()
        
        outputs = model(prices, features)
        preds = outputs['volatility_pred'].cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(target_vol)
        
        if 'uncertainty' in outputs:
            all_uncertainties.append(outputs['uncertainty'].cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Inverse transform
    if target_scaler is not None:
        all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    results = {
        'predictions': all_preds,
        'targets': all_targets
    }
    
    if all_uncertainties:
        all_uncertainties = np.concatenate(all_uncertainties, axis=0).flatten()
        results['uncertainties'] = all_uncertainties
    
    return results


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Directional accuracy
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    dir_acc = np.mean(true_direction == pred_direction)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Direction_Accuracy': dir_acc
    }


def plot_predictions(results, save_path: str = "figures/predictions.png"):
    """Plot actual vs predicted volatility."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    preds = results['predictions']
    targets = results['targets']
    
    # Time series plot (last 100 points)
    n_show = min(240, len(preds))  # 10 days
    ax = axes[0, 0]
    ax.plot(targets[-n_show:], label='Actual', alpha=0.8)
    ax.plot(preds[-n_show:], label='Predicted', alpha=0.8)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Volatility')
    ax.set_title('Volatility Prediction (Last 10 Days)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot
    ax = axes[0, 1]
    ax.scatter(targets, preds, alpha=0.5, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Actual Volatility')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 0]
    errors = preds - targets
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (Mean: {errors.mean():.4f}, Std: {errors.std():.4f})')
    ax.grid(True, alpha=0.3)
    
    # Uncertainty vs error (if available)
    ax = axes[1, 1]
    if 'uncertainties' in results:
        uncertainties = results['uncertainties']
        abs_errors = np.abs(errors)
        ax.scatter(uncertainties, abs_errors, alpha=0.5, s=10)
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Uncertainty Calibration')
    else:
        # Rolling error
        window = 24
        rolling_mae = pd.Series(np.abs(errors)).rolling(window).mean()
        ax.plot(rolling_mae)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Rolling MAE (24h)')
        ax.set_title('Error Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved predictions plot to {save_path}")


def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load data
    _, _, test_loader, metadata = create_dataloaders(
        data_path=config.data_path,
        window_size=config.window_size,
        batch_size=config.batch_size
    )
    config.update_from_metadata(metadata)
    
    # Load model
    model = load_model(f"{config.checkpoint_dir}/best_model.pt", config, device)
    
    # Predict
    print("Generating predictions...")
    results = predict_all(model, test_loader, device, metadata['target_scaler'])
    
    # Calculate metrics
    metrics = calculate_metrics(results['targets'], results['predictions'])
    
    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    
    # Plot
    os.makedirs("figures", exist_ok=True)
    plot_predictions(results)
    
    # Save predictions
    df = pd.DataFrame({
        'actual': results['targets'],
        'predicted': results['predictions']
    })
    if 'uncertainties' in results:
        df['uncertainty'] = results['uncertainties']
    
    df.to_csv("figures/test_predictions.csv", index=False)
    print("\nSaved predictions to figures/test_predictions.csv")


if __name__ == "__main__":
    main()
```

---

## Phase 6: Execution Steps

Run these commands in order:

```bash
# 1. Fetch data (180 days of hourly BTC/USDT)
python data/fetch_binance.py

# 2. Engineer features and prepare dataset
python data/features.py

# 3. Create dataset and verify
python data/dataset.py

# 4. Train the model
python train_volatility.py

# 5. Evaluate on test set (last 10 days)
python evaluate.py
```

---

## Phase 7: Expected Output Structure

```
btc_volatility_predictor/
├── config.py
├── requirements.txt
├── train_volatility.py
├── evaluate.py
├── data/
│   ├── __init__.py
│   ├── fetch_binance.py
│   ├── features.py
│   ├── dataset.py
│   ├── raw/
│   │   └── btcusdt_1h.csv
│   └── processed/
│       └── features.csv
├── models/
│   ├── __init__.py
│   ├── sph_net.py
│   ├── encoders.py
│   ├── attention.py
│   └── heads.py
├── checkpoints/
│   ├── best_model.pt
│   └── results.json
├── logs/
│   └── run_YYYYMMDD_HHMMSS/
└── figures/
    ├── predictions.png
    └── test_predictions.csv
```

---

## Key Hyperparameters to Tune

| Parameter | Default | Range to Try | Notes |
|-----------|---------|--------------|-------|
| `window_size` | 72 | 48, 72, 96, 168 | More history = better patterns but slower |
| `d_model` | 128 | 64, 128, 256 | Model capacity |
| `n_encoder_layers` | 4 | 2, 4, 6 | Depth vs. overfitting |
| `n_heads` | 8 | 4, 8 | Must divide d_model |
| `dropout` | 0.15 | 0.1, 0.15, 0.2 | Regularization |
| `learning_rate` | 5e-4 | 1e-4 to 1e-3 | Critical for convergence |
| `batch_size` | 64 | 32, 64, 128 | Memory vs. gradient noise |

---

## Potential Improvements

1. **Add more features**: Fear & Greed Index, funding rates, liquidation data
2. **Multi-timeframe**: Combine 1h with 4h and daily aggregations
3. **Ensemble**: Train multiple models with different seeds
4. **Quantile regression**: Predict volatility distribution, not just point estimate
5. **Online learning**: Update model with recent data
6. **Attention visualization**: Interpret which timesteps matter most

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or `window_size` |
| Loss not decreasing | Lower `learning_rate`, check data normalization |
| Validation loss increases | Increase `dropout`, reduce `n_encoder_layers` |
| API rate limiting | Add `time.sleep(1)` between requests |
| NaN in features | Check for division by zero in feature engineering |