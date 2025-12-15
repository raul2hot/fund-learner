# Data Alignment & Feature Engineering System
## Instructions for Claude Code Opus

---

## 🎯 Project Objective

Build a **robust, point-in-time correct data alignment system** that:
1. Fetches multi-source trading data (OHLCV, funding rates, sentiment, etc.)
2. Aligns all data to a 1-hour base frequency without look-ahead bias
3. Engineers features from 10+ filter categories
4. Labels data for 5-class candle classification
5. Produces train/val/test splits for ML model training

**This data layer is CRITICAL** - all downstream ML models depend on it being correct. Any look-ahead bias will produce models that backtest well but fail in production.

---

## 📅 Configuration Parameters

### Time Range (Dynamic - Future-Proof Design)

```python
# config/settings.py

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TimeConfig:
    """
    NON-FRAGILE TIME CONFIGURATION
    
    Design Principles:
    1. Use relative dates where possible (e.g., "last 5 years")
    2. Hard floor based on data availability (Binance Futures launch)
    3. Dynamic end date (always "now" unless specified)
    4. Graceful handling of future dates
    """
    
    # === HARD CONSTRAINTS (Data Availability) ===
    # Binance Futures launched Sept 13, 2019
    # We use Sept 15 to ensure stable data
    BINANCE_FUTURES_LAUNCH = "2019-09-15"
    
    # Minimum required for full feature set
    DATA_FLOOR = BINANCE_FUTURES_LAUNCH
    
    # === DEFAULT CONFIGURATION ===
    # Uses relative dates for robustness
    
    @classmethod
    def get_default_start(cls) -> str:
        """
        Default: Maximum available history with full features.
        This automatically grows as time passes.
        """
        return cls.DATA_FLOOR
    
    @classmethod  
    def get_default_end(cls) -> str:
        """
        Default: Today's date.
        Always fresh, never stale.
        """
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    @classmethod
    def get_relative_start(cls, years_back: int = 5) -> str:
        """
        Alternative: Rolling window (e.g., last 5 years).
        Useful for strategies that don't need ancient data.
        """
        start = datetime.utcnow() - timedelta(days=365 * years_back)
        
        # Never go before data floor
        floor = datetime.strptime(cls.DATA_FLOOR, "%Y-%m-%d")
        if start < floor:
            start = floor
            
        return start.strftime("%Y-%m-%d")
    
    @classmethod
    def validate_dates(cls, start: str, end: str) -> tuple:
        """
        Validate and adjust dates to ensure data availability.
        Returns adjusted (start, end) with warnings if modified.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        floor_dt = datetime.strptime(cls.DATA_FLOOR, "%Y-%m-%d")
        now = datetime.utcnow()
        
        warnings = []
        
        # Adjust start if before data floor
        if start_dt < floor_dt:
            warnings.append(
                f"⚠️ Start date {start} is before Binance Futures launch. "
                f"Adjusted to {cls.DATA_FLOOR}"
            )
            start_dt = floor_dt
        
        # Adjust end if in future
        if end_dt > now:
            new_end = now.strftime("%Y-%m-%d")
            warnings.append(
                f"⚠️ End date {end} is in the future. "
                f"Adjusted to {new_end}"
            )
            end_dt = now
        
        # Ensure start < end
        if start_dt >= end_dt:
            raise ValueError(
                f"Start date ({start_dt.date()}) must be before "
                f"end date ({end_dt.date()})"
            )
        
        for w in warnings:
            print(w)
        
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# === EXPORTED DEFAULTS ===
# These are used throughout the codebase

DEFAULT_START_DATE = TimeConfig.get_default_start()  # "2019-09-15"
DEFAULT_END_DATE = TimeConfig.get_default_end()      # Dynamic: today
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"

# For strategies that want rolling window instead of full history
ROLLING_WINDOW_YEARS = 3  # Configurable
```

### Data Split Ratios

```python
# IMPORTANT: Walk-forward split (chronological, NO shuffling)
TRAIN_RATIO = 0.70   # 70% for training (earliest data)
VAL_RATIO = 0.15     # 15% for validation (middle)
TEST_RATIO = 0.15    # 15% for testing (most recent)

# Future-proof: Test set is always the MOST RECENT data
# This means as you add more data, test set naturally updates
```

### CLI Interface (Flexible)

```bash
# Full history (recommended for initial training)
python fetch_data.py 
# Uses: 2019-09-15 to today

# Custom absolute dates
python fetch_data.py --start 2021-01-01 --end 2023-12-31

# Rolling window (last N years from today)
python fetch_data.py --rolling-years 3
# Uses: (today - 3 years) to today

# Different symbol
python fetch_data.py --symbol ETHUSDT

# Config file (for reproducibility)
python fetch_data.py --config experiments/bear_market_test.yaml
```

### Feature Availability Timeline

```python
# config/feature_availability.py

"""
Track when each feature becomes available.
This allows graceful degradation for older date ranges.
"""

FEATURE_AVAILABILITY = {
    # Core features (always available for our date range)
    'ohlcv': '2017-01-01',           # Binance spot launch
    'volume': '2017-01-01',
    
    # Sentiment features
    'fear_greed': '2018-02-01',       # Alternative.me launch
    
    # Futures-specific features (our primary data floor)
    'funding_rate': '2019-09-13',
    'open_interest': '2019-09-13',
    'long_short_ratio': '2020-01-01', # Slightly later availability
    'taker_volume': '2020-01-01',
    
    # Derived features (computed, always available if inputs exist)
    'volatility': 'derived',
    'trend_strength': 'derived',
    'momentum': 'derived',
    'kmeans_sr': 'derived',
    'hmm_regime': 'derived',
    'session_time': 'derived',
}

def get_available_features(start_date: str) -> list:
    """
    Returns list of features available for a given start date.
    Allows model to adapt to available data.
    """
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    available = []
    for feature, avail_date in FEATURE_AVAILABILITY.items():
        if avail_date == 'derived':
            available.append(feature)
        elif datetime.strptime(avail_date, "%Y-%m-%d") <= start:
            available.append(feature)
    
    return available

def check_feature_coverage(start_date: str) -> dict:
    """
    Returns coverage report for transparency.
    """
    available = get_available_features(start_date)
    all_features = list(FEATURE_AVAILABILITY.keys())
    missing = set(all_features) - set(available) - {'derived'}
    
    return {
        'start_date': start_date,
        'available_features': available,
        'missing_features': list(missing),
        'coverage_pct': len(available) / len(all_features) * 100,
        'full_coverage': len(missing) == 0
    }
```

### Why This Design is Non-Fragile

| Principle | Implementation |
|-----------|---------------|
| **No hardcoded "now"** | `get_default_end()` always returns current date |
| **Hard floor protection** | Can never accidentally request unavailable data |
| **Graceful degradation** | Feature availability tracking allows partial operation |
| **Rolling window option** | Strategies can use "last N years" instead of fixed dates |
| **Reproducibility** | Config files can lock specific dates for experiments |
| **Self-documenting** | Feature availability is explicit, not hidden |
| **Future expansion** | Easy to add new features with their availability dates |

---

## 📊 Data Sources & API Endpoints

### 1. OHLCV Data (Binance Futures)
**Primary data source - base timeline**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://fapi.binance.com/fapi/v1/klines` |
| Frequency | 1 hour |
| Publication Delay | ~5 seconds after candle close |
| Rate Limit | 2400 requests/min |
| Max per request | 1500 candles |

```python
# API Parameters
params = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "startTime": start_ms,  # Unix milliseconds
    "endTime": end_ms,
    "limit": 1500
}

# Response format (array of arrays):
# [
#   [
#     1499040000000,      # Open time (ms)
#     "0.01634790",       # Open
#     "0.80000000",       # High
#     "0.01575800",       # Low
#     "0.01577100",       # Close
#     "148976.11427815",  # Volume
#     1499644799999,      # Close time (ms)
#     "2434.19055334",    # Quote asset volume
#     308,                # Number of trades
#     "1756.87402397",    # Taker buy base volume
#     "28.46694368",      # Taker buy quote volume
#     "0"                 # Ignore
#   ]
# ]
```

**Implementation Notes:**
- Paginate through date range (max 1500 candles per request)
- Handle rate limits with exponential backoff
- Validate no gaps in hourly data
- Store `taker_buy_base_volume` and `taker_buy_quote_volume` for Order Flow features

---

### 2. Funding Rate (Binance Futures)
**8-hour frequency, critical for sentiment**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://fapi.binance.com/fapi/v1/fundingRate` |
| Frequency | Every 8 hours (00:00, 08:00, 16:00 UTC) |
| Publication Delay | ~1 minute after funding time |
| Rate Limit | 500 requests per 5 min |
| Max per request | 1000 records |
| Historical Limit | Full history available |

```python
params = {
    "symbol": "BTCUSDT",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 1000
}

# Response:
# [
#   {
#     "symbol": "BTCUSDT",
#     "fundingRate": "-0.03750000",
#     "fundingTime": 1570608000000,
#     "markPrice": "34287.54619963"
#   }
# ]
```

**Alignment Strategy:**
- Forward fill to 1hr frequency
- Fill limit: 8 hours (next funding rate should arrive)
- Add `funding_rate_age_hours` feature

---

### 3. Fear & Greed Index (Alternative.me)
**Daily frequency, sentiment indicator**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://api.alternative.me/fng/` |
| Frequency | Daily (updated ~09:00 UTC) |
| Publication Delay | ~9 hours (dated for "today" but published ~09:00) |
| Rate Limit | Be respectful, ~1 req/sec |
| Historical | Use `?limit=0` for all history |

```python
# Get all historical data
url = "https://api.alternative.me/fng/?limit=0"

# Response:
# {
#   "data": [
#     {
#       "value": "26",
#       "value_classification": "Fear",
#       "timestamp": "1671321600",  # Unix seconds
#       "time_until_update": "52844"  # Only for latest
#     }
#   ]
# }
```

**Alignment Strategy:**
- Publication delay: 9 hours (available from 09:00 UTC onwards)
- Forward fill to 1hr with limit of 24 hours
- Add `fear_greed_age_hours` feature
- CRITICAL: Data dated "2024-01-15" is NOT available at 00:00 on Jan 15!

---

### 4. Open Interest (Binance Futures) ⚠️ HYBRID APPROACH
**5-minute to daily aggregations available**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://fapi.binance.com/futures/data/openInterestHist` |
| Frequency | 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d |
| Publication Delay | ~1 minute |
| Rate Limit | 1000 requests per 5 min |
| API Historical Limit | **Only 30 days via API!** |
| Kaggle Historical | Full history available (updated quarterly) |

```python
params = {
    "symbol": "BTCUSDT",
    "period": "1h",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 500
}

# Response:
# [
#   {
#     "symbol": "BTCUSDT",
#     "sumOpenInterest": "20403.63700000",
#     "sumOpenInterestValue": "150570784.07809979",
#     "timestamp": "1583127900000"
#   }
# ]
```

**⚠️ USES HYBRID DATA STRATEGY** - See [Section 8: Hybrid Data Strategy](#8-hybrid-data-strategy-for-30-day-limited-apis) below.

---

### 5. Long/Short Ratio (Binance Futures) ⚠️ HYBRID APPROACH
**Multiple ratio types available**

| Endpoint | Description |
|----------|-------------|
| `/futures/data/globalLongShortAccountRatio` | All accounts L/S ratio |
| `/futures/data/topLongShortAccountRatio` | Top 20% traders by account |
| `/futures/data/topLongShortPositionRatio` | Top 20% traders by position |

| Field | Details |
|-------|---------|
| Frequency | 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d |
| Publication Delay | ~1 minute |
| API Historical Limit | **Only 30 days via API!** |
| Kaggle Historical | Full history available (updated quarterly) |

```python
params = {
    "symbol": "BTCUSDT",
    "period": "1h",
    "limit": 500
}

# Response (globalLongShortAccountRatio):
# [
#   {
#     "symbol": "BTCUSDT",
#     "longShortRatio": "1.8105",
#     "longAccount": "0.6442",
#     "shortAccount": "0.3558",
#     "timestamp": "1583139600000"
#   }
# ]
```

**⚠️ USES HYBRID DATA STRATEGY** - See [Section 8: Hybrid Data Strategy](#8-hybrid-data-strategy-for-30-day-limited-apis) below.

---

### 6. Taker Buy/Sell Volume (Binance Futures) ⚠️ HYBRID APPROACH
**Order flow proxy**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://fapi.binance.com/futures/data/takerlongshortRatio` |
| Frequency | 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d |
| API Historical Limit | **Only 30 days via API!** |
| Kaggle Historical | Full history available (updated quarterly) |

```python
# Response:
# [
#   {
#     "buySellRatio": "1.0234",
#     "buyVol": "234532.234",
#     "sellVol": "229145.123",
#     "timestamp": 1583139600000
#   }
# ]
```

**⚠️ USES HYBRID DATA STRATEGY** - See [Section 8: Hybrid Data Strategy](#8-hybrid-data-strategy-for-30-day-limited-apis) below.

---

### 7. Order Book Depth (Binance)
**Real-time only - must collect ourselves**

| Field | Details |
|-------|---------|
| Endpoint | `GET https://fapi.binance.com/fapi/v1/depth` |
| Type | Snapshot (not historical) |
| Limits | 5, 10, 20, 50, 100, 500, 1000 levels |

```python
params = {"symbol": "BTCUSDT", "limit": 20}

# Response:
# {
#   "lastUpdateId": 1027024,
#   "bids": [["4.00000000", "431.00000000"]],
#   "asks": [["4.00000200", "12.00000000"]]
# }
```

**⚠️ NOTE:** Historical order book data requires:
- Binance VIP 1+ status for historical API access
- Or third-party: Tardis.dev, Amberdata, Kaiko
- Or start collecting now via websocket

**For MVP:** Skip order book, derive liquidity proxy from volume/spread

---

## 8. Hybrid Data Strategy for 30-Day Limited APIs

### The Problem

Binance API only provides **30 days** of historical data for:
- Open Interest
- Long/Short Ratio  
- Taker Buy/Sell Volume

But we need **12+ months** of data for proper backtesting.

### The Solution: Kaggle + API Hybrid

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       HYBRID DATA PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   KAGGLE CSV     │    │  GAP FILLER  │    │   BINANCE API        │   │
│  │   (Historical)   │ -> │  (if needed) │ -> │   (Last 30 days)     │   │
│  └──────────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                          │
│  Coverage:               Coverage:           Coverage:                   │
│  2020-01 to last         Kaggle end to       API_start to today          │
│  quarterly update        API_start                                       │
│                                                                          │
│  Source:                 Source:             Source:                     │
│  Free Kaggle download    Mark as missing     Binance REST API            │
│                          OR interpolate                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Free Kaggle Datasets (by jesusgraterol)

| Dataset | URL | Update Frequency |
|---------|-----|------------------|
| **Open Interest** | `kaggle.com/datasets/jesusgraterol/bitcoin-open-interest-binance-futures` | Quarterly |
| **Long/Short Ratio** | `kaggle.com/datasets/jesusgraterol/bitcoin-longshort-ratio-binance-futures` | Quarterly |
| **Taker Buy/Sell Volume** | `kaggle.com/datasets/jesusgraterol/bitcoin-taker-buysell-volume-binance-futures` | Quarterly |
| **Funding Rate** | `kaggle.com/datasets/jesusgraterol/bitcoin-funding-rate-binance-futures` | Quarterly |

**GitHub Builder Script:** `github.com/jesusgraterol/binance-futures-dataset-builder`

### Implementation

```python
# fetchers/hybrid_fetcher.py

"""
Hybrid fetcher that combines Kaggle historical data with Binance API recent data.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings


class HybridDataFetcher:
    """
    Fetches data using hybrid strategy:
    1. Load historical data from Kaggle CSV
    2. Fetch recent data from Binance API (last 30 days)
    3. Merge and deduplicate
    """
    
    KAGGLE_DATASETS = {
        'open_interest': {
            'kaggle_slug': 'jesusgraterol/bitcoin-open-interest-binance-futures',
            'filename': 'open_interest.csv',
            'timestamp_col': 'ot',  # Open time
            'value_cols': ['soi', 'soiv'],  # Sum OI, Sum OI Value
        },
        'long_short_ratio': {
            'kaggle_slug': 'jesusgraterol/bitcoin-longshort-ratio-binance-futures',
            'filename': 'long_short_ratio.csv',
            'timestamp_col': 'ot',
            'value_cols': ['lsr', 'la', 'sa'],  # LS Ratio, Long Acct, Short Acct
        },
        'taker_volume': {
            'kaggle_slug': 'jesusgraterol/bitcoin-taker-buysell-volume-binance-futures',
            'filename': 'taker_buy_sell_volume.csv',
            'timestamp_col': 'ot',
            'value_cols': ['bsr', 'bv', 'sv'],  # Buy/Sell Ratio, Buy Vol, Sell Vol
        },
    }
    
    def __init__(self, kaggle_data_dir: Path, binance_client):
        """
        Args:
            kaggle_data_dir: Directory containing downloaded Kaggle CSVs
            binance_client: Initialized Binance API client
        """
        self.kaggle_dir = Path(kaggle_data_dir)
        self.client = binance_client
        
    def fetch(
        self, 
        data_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data using hybrid strategy.
        
        Args:
            data_type: 'open_interest', 'long_short_ratio', or 'taker_volume'
            start_date: Start of date range
            end_date: End of date range (usually today)
            
        Returns:
            DataFrame with complete data coverage
        """
        if data_type not in self.KAGGLE_DATASETS:
            raise ValueError(f"Unknown data type: {data_type}")
        
        config = self.KAGGLE_DATASETS[data_type]
        
        # Step 1: Load Kaggle historical data
        kaggle_df = self._load_kaggle_data(config)
        
        # Step 2: Determine gap
        kaggle_end = kaggle_df['timestamp'].max()
        api_start = end_date - timedelta(days=29)  # API gives 30 days
        
        # Step 3: Fetch from API
        api_df = self._fetch_from_api(data_type, api_start, end_date)
        
        # Step 4: Merge
        merged_df = self._merge_data(kaggle_df, api_df, config)
        
        # Step 5: Filter to requested range
        merged_df = merged_df[
            (merged_df['timestamp'] >= start_date) &
            (merged_df['timestamp'] <= end_date)
        ]
        
        # Step 6: Check for gaps and warn
        self._check_gaps(merged_df, kaggle_end, api_start, data_type)
        
        return merged_df
    
    def _load_kaggle_data(self, config: dict) -> pd.DataFrame:
        """Load and standardize Kaggle CSV data"""
        filepath = self.kaggle_dir / config['filename']
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Kaggle data not found: {filepath}\n"
                f"Download from: kaggle.com/datasets/{config['kaggle_slug']}"
            )
        
        df = pd.read_csv(filepath)
        
        # Standardize timestamp
        df['timestamp'] = pd.to_datetime(df[config['timestamp_col']], unit='ms', utc=True)
        
        return df
    
    def _fetch_from_api(
        self, 
        data_type: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Fetch recent data from Binance API"""
        # Implementation depends on data_type
        # This is a placeholder - actual implementation uses Binance client
        pass
    
    def _merge_data(
        self, 
        kaggle_df: pd.DataFrame,
        api_df: pd.DataFrame,
        config: dict
    ) -> pd.DataFrame:
        """Merge Kaggle and API data, preferring API for overlapping periods"""
        # Combine
        combined = pd.concat([kaggle_df, api_df], ignore_index=True)
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp')
        
        # Remove duplicates, keeping last (API data preferred)
        combined = combined.drop_duplicates(subset='timestamp', keep='last')
        
        return combined
    
    def _check_gaps(
        self, 
        df: pd.DataFrame,
        kaggle_end: datetime,
        api_start: datetime,
        data_type: str
    ):
        """Warn if there's a gap between Kaggle and API data"""
        gap_days = (api_start - kaggle_end).days
        
        if gap_days > 1:
            warnings.warn(
                f"\n⚠️ DATA GAP DETECTED for {data_type}:\n"
                f"   Kaggle data ends: {kaggle_end.strftime('%Y-%m-%d')}\n"
                f"   API data starts:  {api_start.strftime('%Y-%m-%d')}\n"
                f"   Gap: {gap_days} days\n"
                f"   \n"
                f"   Options:\n"
                f"   1. Update Kaggle CSV from: github.com/jesusgraterol/binance-futures-dataset-builder\n"
                f"   2. Mark gap period as missing (current behavior)\n"
                f"   3. Interpolate (not recommended for trading)\n"
            )
```

### Data Directory Setup

```
data/
├── kaggle/                          # Downloaded Kaggle CSVs
│   ├── open_interest.csv
│   ├── long_short_ratio.csv
│   ├── taker_buy_sell_volume.csv
│   └── funding_rate.csv             # Optional (API has full history)
│
├── raw/                             # Raw fetched data
│   ├── ohlcv_BTCUSDT_1h.parquet
│   └── ...
```

### CLI Download Helper

```bash
# Download Kaggle datasets (requires kaggle CLI configured)
kaggle datasets download -d jesusgraterol/bitcoin-open-interest-binance-futures -p data/kaggle/
kaggle datasets download -d jesusgraterol/bitcoin-longshort-ratio-binance-futures -p data/kaggle/
kaggle datasets download -d jesusgraterol/bitcoin-taker-buysell-volume-binance-futures -p data/kaggle/

# Unzip
cd data/kaggle && unzip -o "*.zip" && rm *.zip
```

### Handling the Gap

If there's a gap between Kaggle's last update and API's 30-day window:

```python
# Option 1: Mark as missing (RECOMMENDED)
# The model learns to handle missing data via missingness features
df.loc[gap_mask, 'open_interest'] = np.nan
df.loc[gap_mask, 'open_interest_missing'] = 1

# Option 2: Forward fill with staleness tracking (ACCEPTABLE)
# But limit how far we fill
df['open_interest'] = df['open_interest'].ffill(limit=72)  # Max 3 days
df['open_interest_stale'] = (hours_since_last_valid > 24).astype(int)

# Option 3: Interpolate (NOT RECOMMENDED for trading)
# Creates fake data that looks real - dangerous!
```

### Future-Proofing: Daily Collection Job

To avoid gaps in the future, set up automated daily collection:

```python
# scripts/daily_collector.py
"""
Run daily via cron to collect fresh data.
Stores to local parquet files that can supplement Kaggle data.
"""

import schedule
import time
from datetime import datetime

def collect_daily():
    """Collect today's data from Binance API"""
    today = datetime.utcnow().date()
    
    # Fetch last 24h of each metric
    for data_type in ['open_interest', 'long_short_ratio', 'taker_volume']:
        df = fetch_from_binance_api(data_type, hours=24)
        df.to_parquet(f'data/daily/{data_type}_{today}.parquet')
    
    print(f"✓ Collected data for {today}")

# Run at 00:05 UTC daily
schedule.every().day.at("00:05").do(collect_daily)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Cron Setup (Linux/Mac)

```bash
# Add to crontab (crontab -e)
5 0 * * * cd /path/to/project && python scripts/daily_collector.py >> logs/collector.log 2>&1
```

### Important Notes

1. **Kaggle data is updated quarterly** - expect up to 3 months of potential gaps
2. **Always prefer API data when available** - it's more recent and accurate
3. **Track data source in features** - add `data_source` column ('kaggle' or 'api')
4. **Validate after merge** - check for duplicates, gaps, and outliers
5. **Gap < 30 days is manageable** - model can learn from missingness
6. **Gap > 30 days requires Kaggle update** - run the builder script yourself

---

## 🏗️ Project Structure

```
trading_data_pipeline/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration dataclasses
│   └── data_sources.py      # DataSource definitions
│
├── fetchers/
│   ├── __init__.py
│   ├── base.py              # Abstract base fetcher
│   ├── binance_ohlcv.py     # OHLCV fetcher
│   ├── binance_funding.py   # Funding rate fetcher
│   ├── binance_futures.py   # OI, L/S ratio, taker volume (API)
│   ├── hybrid_fetcher.py    # 🆕 Kaggle + API hybrid fetcher
│   ├── kaggle_loader.py     # 🆕 Kaggle CSV loader
│   ├── fear_greed.py        # Alternative.me fetcher
│   └── rate_limiter.py      # Rate limiting utilities
│   ├── fear_greed.py        # Alternative.me fetcher
│   └── rate_limiter.py      # Rate limiting utilities
│
├── alignment/
│   ├── __init__.py
│   ├── point_in_time.py     # PointInTimeDatabase class
│   ├── resampling.py        # Resampling strategies
│   └── validation.py        # Alignment validation
│
├── features/
│   ├── __init__.py
│   ├── volatility.py        # ATR, Bollinger, realized vol
│   ├── trend.py             # ADX, MA slopes, trend strength
│   ├── momentum.py          # RSI, MACD, momentum exhaustion
│   ├── support_resistance.py # K-Means S/R levels
│   ├── regime.py            # HMM regime detection
│   ├── session.py           # Time/session features
│   ├── sentiment.py         # Fear/greed, funding derived
│   └── order_flow.py        # Taker volume, OFI proxy
│
├── labeling/
│   ├── __init__.py
│   ├── candle_classifier.py # 5-class labeling
│   └── path_metrics.py      # MAE/MFE calculations
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py      # Main pipeline runner
│   └── splitter.py          # Train/val/test split
│
├── storage/
│   ├── __init__.py
│   ├── parquet_store.py     # Parquet file storage
│   └── cache.py             # API response caching
│
├── tests/
│   ├── test_alignment.py
│   ├── test_features.py
│   ├── test_labeling.py
│   └── test_no_lookahead.py # Critical validation tests
│
├── scripts/
│   ├── fetch_all_data.py    # CLI entry point
│   └── validate_dataset.py  # Dataset validation script
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Core Implementation Requirements

### 1. DataSource Configuration

```python
# config/data_sources.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import pandas as pd


class ResampleMethod(Enum):
    FORWARD_FILL = "ffill"
    FORWARD_FILL_LIMIT = "ffill_limit"
    AGGREGATE_LAST = "last"
    AGGREGATE_MEAN = "mean"
    POINT_IN_TIME = "pit"


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    frequency: str                    # '1h', '8h', '1D', '5min'
    timestamp_col: str                # Column name for timestamp
    timestamp_format: str             # 'unix_ms', 'unix_s', 'iso', 'date'
    timezone: str                     # Source timezone
    publication_delay: pd.Timedelta   # Time until data is available
    resample_method: ResampleMethod
    fill_limit_hours: Optional[int] = None
    value_columns: List[str] = None
    api_historical_limit_days: Optional[int] = None  # API constraint


# Define all sources
OHLCV_SOURCE = DataSource(
    name='ohlcv',
    frequency='1h',
    timestamp_col='open_time',
    timestamp_format='unix_ms',
    timezone='UTC',
    publication_delay=pd.Timedelta(seconds=5),
    resample_method=ResampleMethod.FORWARD_FILL,
    value_columns=['open', 'high', 'low', 'close', 'volume', 
                   'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
)

FUNDING_RATE_SOURCE = DataSource(
    name='funding_rate',
    frequency='8h',
    timestamp_col='fundingTime',
    timestamp_format='unix_ms',
    timezone='UTC',
    publication_delay=pd.Timedelta(minutes=1),
    resample_method=ResampleMethod.FORWARD_FILL_LIMIT,
    fill_limit_hours=9,  # Next funding in 8h, allow 1h buffer
    value_columns=['fundingRate', 'markPrice']
)

FEAR_GREED_SOURCE = DataSource(
    name='fear_greed',
    frequency='1D',
    timestamp_col='timestamp',
    timestamp_format='unix_s',
    timezone='UTC',
    publication_delay=pd.Timedelta(hours=9),  # Published ~09:00 UTC
    resample_method=ResampleMethod.FORWARD_FILL_LIMIT,
    fill_limit_hours=36,  # Max ~1.5 days stale
    value_columns=['value', 'value_classification']
)

OPEN_INTEREST_SOURCE = DataSource(
    name='open_interest',
    frequency='1h',
    timestamp_col='timestamp',
    timestamp_format='unix_ms',
    timezone='UTC',
    publication_delay=pd.Timedelta(minutes=1),
    resample_method=ResampleMethod.FORWARD_FILL_LIMIT,
    fill_limit_hours=2,
    value_columns=['sumOpenInterest', 'sumOpenInterestValue'],
    api_historical_limit_days=30  # CRITICAL: Only 30 days available!
)

LONG_SHORT_RATIO_SOURCE = DataSource(
    name='long_short_ratio',
    frequency='1h',
    timestamp_col='timestamp',
    timestamp_format='unix_ms',
    timezone='UTC',
    publication_delay=pd.Timedelta(minutes=1),
    resample_method=ResampleMethod.FORWARD_FILL_LIMIT,
    fill_limit_hours=2,
    value_columns=['longShortRatio', 'longAccount', 'shortAccount'],
    api_historical_limit_days=30
)

TAKER_VOLUME_SOURCE = DataSource(
    name='taker_volume',
    frequency='1h',
    timestamp_col='timestamp',
    timestamp_format='unix_ms',
    timezone='UTC',
    publication_delay=pd.Timedelta(minutes=1),
    resample_method=ResampleMethod.FORWARD_FILL_LIMIT,
    fill_limit_hours=2,
    value_columns=['buySellRatio', 'buyVol', 'sellVol'],
    api_historical_limit_days=30
)
```

---

### 2. Point-In-Time Database (CRITICAL)

```python
# alignment/point_in_time.py

"""
CRITICAL: This class ensures no look-ahead bias.

Key principle: For any timestamp T, we can only use data that was
ACTUALLY AVAILABLE at time T, not data timestamped at T.

Example:
- Candle closes at 14:00:00
- We receive candle data at 14:00:05
- Therefore, at 14:00:00, we could NOT have known the 14:00 candle!
- For 14:00 prediction, we can only use data available_at <= 14:00
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


class PointInTimeDatabase:
    """
    Stores all data with availability timestamps.
    Ensures no look-ahead bias in feature construction.
    """
    
    def __init__(self, base_frequency: str = '1h'):
        self.base_frequency = base_frequency
        self.sources: Dict[str, DataSource] = {}
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None
        
    def register_source(self, config: DataSource):
        """Register a data source configuration"""
        self.sources[config.name] = config
        print(f"✓ Registered: {config.name} ({config.frequency}, "
              f"delay={config.publication_delay})")
    
    def ingest(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ingest raw data with proper timestamp handling.
        Adds _available_at column for point-in-time queries.
        """
        if name not in self.sources:
            raise ValueError(f"Unknown source: {name}. Register first.")
        
        config = self.sources[name]
        df = df.copy()
        
        # 1. Standardize timestamp
        df = self._standardize_timestamp(df, config)
        
        # 2. Add availability timestamp (CRITICAL!)
        df['_available_at'] = df['timestamp'] + config.publication_delay
        
        # 3. Sort and validate
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        self._validate_timestamps(df, name)
        
        # 4. Store
        self.raw_data[name] = df
        print(f"✓ Ingested {name}: {len(df)} rows, "
              f"{df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
    
    def _standardize_timestamp(self, df: pd.DataFrame, config: DataSource) -> pd.DataFrame:
        """Convert all timestamps to UTC pandas Timestamp"""
        ts_col = config.timestamp_col
        
        if config.timestamp_format == 'unix_ms':
            df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
        elif config.timestamp_format == 'unix_s':
            df['timestamp'] = pd.to_datetime(df[ts_col], unit='s', utc=True)
        elif config.timestamp_format == 'iso':
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
        else:
            raise ValueError(f"Unknown format: {config.timestamp_format}")
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame, name: str):
        """Validate timestamp quality"""
        nat_count = df['timestamp'].isna().sum()
        if nat_count > 0:
            raise ValueError(f"{name}: {nat_count} NaT timestamps!")
        
        if not df['timestamp'].is_monotonic_increasing:
            dup_count = df['timestamp'].duplicated().sum()
            if dup_count > 0:
                warnings.warn(f"{name}: {dup_count} duplicate timestamps removed")
    
    def get_point_in_time(self, as_of: pd.Timestamp, source: str) -> Optional[pd.Series]:
        """
        Get the most recent data available at a specific time.
        
        THIS IS THE KEY FUNCTION FOR PREVENTING LOOK-AHEAD BIAS.
        """
        if source not in self.raw_data:
            return None
        
        df = self.raw_data[source]
        
        # Only data that was AVAILABLE at as_of time
        available = df[df['_available_at'] <= as_of]
        
        if len(available) == 0:
            return None
        
        return available.iloc[-1]
    
    def build_aligned_dataset(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Build aligned dataset at base frequency.
        All features are point-in-time correct.
        """
        # Create base timeline from OHLCV
        if 'ohlcv' not in self.raw_data:
            raise ValueError("OHLCV data required as base timeline")
        
        base_df = self.raw_data['ohlcv']
        base_df = base_df[(base_df['timestamp'] >= start) & 
                          (base_df['timestamp'] <= end)]
        
        # Initialize aligned dataframe
        aligned = pd.DataFrame(index=base_df['timestamp'])
        aligned.index.name = 'timestamp'
        
        # Add OHLCV columns directly (same frequency)
        for col in self.sources['ohlcv'].value_columns:
            if col in base_df.columns:
                aligned[col] = base_df.set_index('timestamp')[col]
        
        # Align other sources
        for name, config in self.sources.items():
            if name == 'ohlcv':
                continue
            print(f"Aligning {name}...")
            aligned = self._align_source(aligned, name, config)
        
        # Final validation
        self._validate_no_lookahead(aligned)
        
        self.aligned_data = aligned
        return aligned
    
    def _align_source(
        self,
        aligned: pd.DataFrame,
        name: str,
        config: DataSource
    ) -> pd.DataFrame:
        """Align a source to base timeline using point-in-time logic"""
        if name not in self.raw_data:
            warnings.warn(f"Source {name} not ingested, marking as missing")
            for col in config.value_columns or []:
                aligned[f"{name}_{col}"] = np.nan
                aligned[f"{name}_{col}_missing"] = 1
            return aligned
        
        source_df = self.raw_data[name]
        value_cols = config.value_columns or []
        
        # For each base timestamp, get latest available data
        for col in value_cols:
            aligned[f"{name}_{col}"] = np.nan
            aligned[f"{name}_{col}_age_hours"] = np.nan
        
        # Efficient merge_asof approach
        source_pit = source_df.copy()
        source_pit = source_pit.set_index('_available_at').sort_index()
        
        for col in value_cols:
            # Merge based on availability time
            merged = pd.merge_asof(
                aligned.reset_index()[['timestamp']],
                source_pit.reset_index()[['_available_at', 'timestamp', col]].rename(
                    columns={'timestamp': 'source_ts', '_available_at': 'avail_ts'}
                ),
                left_on='timestamp',
                right_on='avail_ts',
                direction='backward'
            )
            
            # Calculate age
            age_hours = (merged['timestamp'] - merged['source_ts']).dt.total_seconds() / 3600
            
            # Apply fill limit if specified
            if config.fill_limit_hours:
                mask = age_hours > config.fill_limit_hours
                merged.loc[mask, col] = np.nan
                age_hours[mask] = np.nan
            
            aligned[f"{name}_{col}"] = merged[col].values
            aligned[f"{name}_{col}_age_hours"] = age_hours.values
        
        # Add missingness indicator
        for col in value_cols:
            aligned[f"{name}_{col}_missing"] = aligned[f"{name}_{col}"].isna().astype(int)
        
        return aligned
    
    def _validate_no_lookahead(self, aligned: pd.DataFrame):
        """
        CRITICAL: Validate no look-ahead bias exists.
        """
        print("\n" + "="*50)
        print("LOOK-AHEAD BIAS VALIDATION")
        print("="*50)
        
        issues = []
        
        # Check age columns are all non-negative
        age_cols = [c for c in aligned.columns if '_age_hours' in c]
        for col in age_cols:
            min_age = aligned[col].min()
            if pd.notna(min_age) and min_age < 0:
                issues.append(f"❌ {col}: negative age ({min_age:.2f}h) = FUTURE DATA!")
        
        if issues:
            for issue in issues:
                print(issue)
            raise ValueError("LOOK-AHEAD BIAS DETECTED! Fix before proceeding.")
        
        print("✓ No look-ahead bias detected")
        print("="*50 + "\n")
```

---

### 3. Feature Engineering (10 Filters)

```python
# features/feature_engineer.py

"""
Feature engineering for all 10 filter categories.
All features are computed using only past data (no future leakage).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from hmmlearn import hmm


class FeatureEngineer:
    """Compute all filter features from aligned data"""
    
    def __init__(self, aligned_data: pd.DataFrame):
        self.data = aligned_data.copy()
        
    def compute_all_features(self) -> pd.DataFrame:
        """Compute all feature categories"""
        
        # 1. Volatility Features
        self._add_volatility_features()
        
        # 2. K-Means S/R Levels
        self._add_support_resistance_features()
        
        # 3. HMM Regime Detection
        self._add_regime_features()
        
        # 4. Order Flow Imbalance (proxy from taker data)
        self._add_order_flow_features()
        
        # 5. Funding Rate Features
        self._add_funding_features()
        
        # 6. Liquidity Score (proxy)
        self._add_liquidity_features()
        
        # 7. Fear & Greed Features
        self._add_sentiment_features()
        
        # 8. Session/Time Features
        self._add_session_features()
        
        # 9. Trend Strength Features
        self._add_trend_features()
        
        # 10. Momentum Exhaustion Features
        self._add_momentum_features()
        
        return self.data
    
    # =========================================================================
    # 1. VOLATILITY FEATURES
    # =========================================================================
    def _add_volatility_features(self):
        """ATR, Bollinger Width, Realized Volatility"""
        
        df = self.data
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # ATR (14 period)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']  # Normalized
        
        # Bollinger Bands Width
        df['sma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Realized Volatility (hourly returns std)
        df['returns'] = df['close'].pct_change()
        df['realized_vol_24h'] = df['returns'].rolling(24).std() * np.sqrt(24)
        df['realized_vol_168h'] = df['returns'].rolling(168).std() * np.sqrt(168)
        
        # Volatility regime (percentile-based)
        df['vol_percentile'] = df['atr_ratio'].rolling(168).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
    
    # =========================================================================
    # 2. K-MEANS SUPPORT/RESISTANCE
    # =========================================================================
    def _add_support_resistance_features(self, lookback: int = 168, n_clusters: int = 5):
        """Dynamic S/R levels using K-Means clustering"""
        
        df = self.data
        
        # Initialize columns
        df['nearest_sr_distance'] = np.nan
        df['nearest_sr_strength'] = np.nan
        df['price_vs_sr'] = np.nan  # Above (+1) or below (-1) nearest level
        
        for i in range(lookback, len(df)):
            # Use only past data for clustering
            window = df.iloc[i-lookback:i]
            
            # Cluster on highs, lows, closes (pivot points)
            pivots = np.concatenate([
                window['high'].values,
                window['low'].values,
                window['close'].values
            ]).reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pivots)
            
            # Get cluster centers (S/R levels)
            levels = kmeans.cluster_centers_.flatten()
            
            # Current price
            current_price = df.iloc[i]['close']
            
            # Distance to nearest level
            distances = np.abs(levels - current_price)
            nearest_idx = np.argmin(distances)
            nearest_level = levels[nearest_idx]
            
            # Strength = how many points in that cluster
            labels = kmeans.labels_
            strength = np.sum(labels == nearest_idx) / len(labels)
            
            df.iloc[i, df.columns.get_loc('nearest_sr_distance')] = \
                (current_price - nearest_level) / current_price
            df.iloc[i, df.columns.get_loc('nearest_sr_strength')] = strength
            df.iloc[i, df.columns.get_loc('price_vs_sr')] = \
                1 if current_price > nearest_level else -1
    
    # =========================================================================
    # 3. HMM REGIME DETECTION
    # =========================================================================
    def _add_regime_features(self, lookback: int = 168, n_regimes: int = 3):
        """Hidden Markov Model for regime detection"""
        
        df = self.data
        
        # Initialize
        df['hmm_regime'] = np.nan
        df['hmm_regime_prob_bull'] = np.nan
        df['hmm_regime_prob_bear'] = np.nan
        df['hmm_regime_prob_sideways'] = np.nan
        
        for i in range(lookback, len(df)):
            # Features for HMM: returns and volatility
            window = df.iloc[i-lookback:i]
            
            features = np.column_stack([
                window['returns'].fillna(0).values,
                window['atr_ratio'].fillna(window['atr_ratio'].mean()).values
            ])
            
            try:
                model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42
                )
                model.fit(features)
                
                # Get regime probabilities for current state
                current_features = features[-1:].reshape(1, -1)
                probs = model.predict_proba(features)[-1]
                regime = model.predict(features)[-1]
                
                df.iloc[i, df.columns.get_loc('hmm_regime')] = regime
                
                # Sort regimes by mean return to identify bull/bear/sideways
                regime_returns = []
                for r in range(n_regimes):
                    mask = model.predict(features) == r
                    if mask.any():
                        regime_returns.append(window['returns'].values[mask].mean())
                    else:
                        regime_returns.append(0)
                
                sorted_regimes = np.argsort(regime_returns)
                bear_idx, sideways_idx, bull_idx = sorted_regimes
                
                df.iloc[i, df.columns.get_loc('hmm_regime_prob_bull')] = probs[bull_idx]
                df.iloc[i, df.columns.get_loc('hmm_regime_prob_bear')] = probs[bear_idx]
                df.iloc[i, df.columns.get_loc('hmm_regime_prob_sideways')] = probs[sideways_idx]
                
            except Exception:
                # HMM failed to converge, use NaN
                pass
    
    # =========================================================================
    # 4. ORDER FLOW IMBALANCE (Proxy)
    # =========================================================================
    def _add_order_flow_features(self):
        """Order flow features from taker buy/sell data"""
        
        df = self.data
        
        # If we have taker volume data
        if 'taker_volume_buySellRatio' in df.columns:
            df['ofi_proxy'] = df['taker_volume_buySellRatio'] - 1  # >0 = buy pressure
            df['ofi_proxy_ma'] = df['ofi_proxy'].rolling(6).mean()
            df['ofi_proxy_std'] = df['ofi_proxy'].rolling(24).std()
        
        # Alternative: use taker buy from OHLCV
        if 'taker_buy_base' in df.columns and 'volume' in df.columns:
            df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume']
            df['buy_pressure'] = df['taker_buy_ratio'] - 0.5  # Center around 0
            df['buy_pressure_ma'] = df['buy_pressure'].rolling(6).mean()
    
    # =========================================================================
    # 5. FUNDING RATE FEATURES
    # =========================================================================
    def _add_funding_features(self):
        """Funding rate sentiment features"""
        
        df = self.data
        
        if 'funding_rate_fundingRate' in df.columns:
            fr = df['funding_rate_fundingRate']
            
            # Raw and normalized
            df['funding_rate'] = fr
            df['funding_rate_abs'] = fr.abs()
            
            # Rolling averages
            df['funding_rate_ma_24h'] = fr.rolling(24).mean()
            
            # Extreme funding detection
            df['funding_extreme_long'] = (fr > 0.001).astype(int)  # >0.1%
            df['funding_extreme_short'] = (fr < -0.001).astype(int)
            
            # Z-score of funding
            df['funding_zscore'] = (fr - fr.rolling(168).mean()) / fr.rolling(168).std()
    
    # =========================================================================
    # 6. LIQUIDITY SCORE (Proxy)
    # =========================================================================
    def _add_liquidity_features(self):
        """Liquidity proxy from available data"""
        
        df = self.data
        
        # Volume-based liquidity proxy
        df['volume_ma_24h'] = df['volume'].rolling(24).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_24h']
        
        # Spread proxy: (high - low) / close normalized by ATR
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_vs_atr'] = df['spread_proxy'] / df['atr_ratio']
        
        # Composite liquidity score
        df['liquidity_score'] = df['volume_ratio'] / (df['spread_vs_atr'] + 1)
        
        # Low liquidity flag
        df['low_liquidity'] = (df['liquidity_score'] < 
                               df['liquidity_score'].rolling(168).quantile(0.25)).astype(int)
    
    # =========================================================================
    # 7. FEAR & GREED FEATURES
    # =========================================================================
    def _add_sentiment_features(self):
        """Fear & Greed index features"""
        
        df = self.data
        
        if 'fear_greed_value' in df.columns:
            fg = df['fear_greed_value'].astype(float)
            
            df['fear_greed'] = fg
            df['fear_greed_ma_7d'] = fg.rolling(168).mean()  # 7 days
            
            # Sentiment zones
            df['extreme_fear'] = (fg < 25).astype(int)
            df['extreme_greed'] = (fg > 75).astype(int)
            
            # Rate of change
            df['fear_greed_roc'] = fg.diff(24)  # 24h change
    
    # =========================================================================
    # 8. SESSION/TIME FEATURES
    # =========================================================================
    def _add_session_features(self):
        """Time-based features"""
        
        df = self.data
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Hour of day (cyclical encoding)
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclical encoding)
        dow = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Trading sessions (UTC)
        df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_london'] = ((hour >= 8) & (hour < 14)).astype(int)
        df['session_newyork'] = ((hour >= 14) & (hour < 21)).astype(int)
        df['session_overlap'] = ((hour >= 14) & (hour < 17)).astype(int)  # London/NY
        
        # Weekend
        df['is_weekend'] = (dow >= 5).astype(int)
    
    # =========================================================================
    # 9. TREND STRENGTH FEATURES
    # =========================================================================
    def _add_trend_features(self):
        """ADX and trend strength indicators"""
        
        df = self.data
        
        # Directional Movement
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        df['plus_dm'] = np.where(
            (high_diff > low_diff.abs()) & (high_diff > 0),
            high_diff, 0
        )
        df['minus_dm'] = np.where(
            (low_diff.abs() > high_diff) & (low_diff < 0),
            low_diff.abs(), 0
        )
        
        # Smoothed DI
        atr = df['atr_14']
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).sum() / (atr * 14))
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).sum() / (atr * 14))
        
        # ADX
        di_diff = (df['plus_di'] - df['minus_di']).abs()
        di_sum = df['plus_di'] + df['minus_di']
        df['dx'] = 100 * di_diff / di_sum.replace(0, np.nan)
        df['adx'] = df['dx'].rolling(14).mean()
        
        # MA slopes
        df['sma_slope_20'] = df['sma_20'].diff(5) / df['sma_20'].shift(5)
        df['sma_slope_50'] = df['close'].rolling(50).mean().diff(10) / \
                            df['close'].rolling(50).mean().shift(10)
        
        # Higher highs / lower lows count
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['hh_count_10'] = df['higher_high'].rolling(10).sum()
        df['ll_count_10'] = df['lower_low'].rolling(10).sum()
        
        # Trend score
        df['trend_score'] = (df['hh_count_10'] - df['ll_count_10']) / 10
    
    # =========================================================================
    # 10. MOMENTUM EXHAUSTION FEATURES
    # =========================================================================
    def _add_momentum_features(self):
        """RSI, MACD, and exhaustion signals"""
        
        df = self.data
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Price-Volume Divergence (exhaustion signal)
        price_slope = df['close'].diff(5).rolling(5).mean()
        volume_slope = df['volume'].diff(5).rolling(5).mean()
        
        df['bullish_exhaustion'] = ((price_slope > 0) & (volume_slope < 0)).astype(int)
        df['bearish_exhaustion'] = ((price_slope < 0) & (volume_slope < 0)).astype(int)
        
        # Momentum (rate of change)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_24'] = df['close'].pct_change(24)
```

---

### 4. Enhanced Candle Labeling

```python
# labeling/candle_classifier.py

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
from typing import Tuple


class CandleLabeler:
    """Generate labels and path quality metrics"""
    
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
        self.high_threshold = high_threshold / 100
        self.low_threshold = low_threshold / 100
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for NEXT candle based on CURRENT candle's close.
        
        CRITICAL: Labels are shifted forward - we predict what WILL happen.
        """
        df = df.copy()
        
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
        df['next_mae_long'] = (df['open'] - next_low) / df['open']   # Max adverse if long
        df['next_mfe_long'] = (next_high - df['open']) / df['open']  # Max favorable if long
        df['next_mae_short'] = (next_high - df['open']) / df['open'] # Max adverse if short
        df['next_mfe_short'] = (df['open'] - next_low) / df['open']  # Max favorable if short
        
        # Classification
        df['label'] = np.nan
        
        # HIGH_BULL: > +1.5% AND clean path
        mask_high_bull = (move_pct > self.high_threshold) & (~low_breach)
        df.loc[mask_high_bull, 'label'] = 0
        
        # LOW_BEAR: < -1.5% AND clean path
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
        
        # Remove last row (no future data)
        df = df.iloc[:-1]
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get distribution of labels"""
        counts = df['label'].value_counts().sort_index()
        pcts = (counts / len(df) * 100).round(2)
        
        dist = pd.DataFrame({
            'Label': [self.LABELS[i] for i in counts.index],
            'Count': counts.values,
            'Percentage': pcts.values
        })
        
        return dist
```

---

### 5. Data Splitting

```python
# pipeline/splitter.py

"""
Walk-forward train/val/test split.
CRITICAL: Chronological order, NO shuffling!
"""

import pandas as pd
from typing import Tuple


class DataSplitter:
    """Chronological data splitter"""
    
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.
        
        Returns: (train_df, val_df, test_df)
        """
        n = len(df)
        
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"\n{'='*50}")
        print("DATA SPLIT SUMMARY")
        print(f"{'='*50}")
        print(f"Total samples: {n}")
        print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"  Date range: {train_df.index.min()} → {train_df.index.max()}")
        print(f"Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"  Date range: {val_df.index.min()} → {val_df.index.max()}")
        print(f"Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        print(f"  Date range: {test_df.index.min()} → {test_df.index.max()}")
        print(f"{'='*50}\n")
        
        # Validate no overlap
        assert train_df.index.max() < val_df.index.min(), "Train/Val overlap!"
        assert val_df.index.max() < test_df.index.min(), "Val/Test overlap!"
        
        return train_df, val_df, test_df
```

---

## 🧪 Testing Requirements

### Critical Tests to Implement

```python
# tests/test_no_lookahead.py

"""
CRITICAL: These tests MUST pass before using data for training.
"""

import pytest
import pandas as pd
import numpy as np


class TestNoLookahead:
    """Tests to ensure no look-ahead bias"""
    
    def test_feature_timestamps_before_labels(self, aligned_data):
        """All features must be computed from data before label timestamp"""
        # For each row, all age columns should be >= 0
        age_cols = [c for c in aligned_data.columns if '_age_hours' in c]
        for col in age_cols:
            min_age = aligned_data[col].min()
            assert pd.isna(min_age) or min_age >= 0, \
                f"{col} has negative age (future data used!)"
    
    def test_label_is_future_data(self, labeled_data):
        """Labels should represent NEXT candle, not current"""
        # Label at time T should predict candle at T+1
        # Verify by checking label matches next candle's metrics
        for i in range(len(labeled_data) - 1):
            current = labeled_data.iloc[i]
            next_row = labeled_data.iloc[i + 1]
            
            # Verify next_return matches actual next close
            expected_return = (next_row['close'] - current['close']) / current['close']
            assert abs(current['next_return'] - expected_return) < 1e-6
    
    def test_no_shuffling_in_split(self, train_df, val_df, test_df):
        """Train/val/test must be strictly chronological"""
        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()
    
    def test_publication_delays_applied(self, pit_db):
        """Verify publication delays are correctly applied"""
        for name, source in pit_db.sources.items():
            df = pit_db.raw_data.get(name)
            if df is None:
                continue
            
            # _available_at should be > timestamp
            assert (df['_available_at'] >= df['timestamp']).all(), \
                f"{name}: available_at before timestamp!"
            
            # Check delay matches config
            expected_delay = source.publication_delay
            actual_delay = (df['_available_at'] - df['timestamp']).iloc[0]
            assert actual_delay == expected_delay, \
                f"{name}: delay mismatch"
```

---

## 📋 Implementation Checklist

### Phase 0: Kaggle Data Setup (NEW - Do First!)
- [ ] Install and configure Kaggle CLI
- [ ] Download historical datasets (Open Interest, L/S Ratio, Taker Volume)
- [ ] Verify CSV files in data/kaggle/ directory
- [ ] Document Kaggle data end date for gap detection

### Phase 1: Data Fetchers
- [ ] Implement rate-limited HTTP client with exponential backoff
- [ ] OHLCV fetcher (Binance Futures /fapi/v1/klines)
- [ ] Funding rate fetcher (/fapi/v1/fundingRate)
- [ ] Fear & Greed fetcher (Alternative.me API)
- [ ] **Hybrid fetcher for Open Interest** (Kaggle CSV + API last 30 days)
- [ ] **Hybrid fetcher for Long/Short Ratio** (Kaggle CSV + API last 30 days)
- [ ] **Hybrid fetcher for Taker Volume** (Kaggle CSV + API last 30 days)
- [ ] Implement Kaggle CSV loader with schema validation
- [ ] Implement data merge logic (Kaggle + API, deduplicate)
- [ ] Implement gap detection and warning system
- [ ] Implement response caching (Parquet files)
- [ ] Add retry logic for failed requests

### Phase 1.5: Daily Collection Setup (Optional but Recommended)
- [ ] Implement daily_collector.py script
- [ ] Set up cron job for automated collection
- [ ] Implement merge logic for daily files + Kaggle baseline

### Phase 2: Data Alignment
- [ ] Implement PointInTimeDatabase class
- [ ] Implement DataSource configuration
- [ ] Implement timestamp standardization (all to UTC)
- [ ] Implement publication delay logic
- [ ] Implement point-in-time merge (merge_asof)
- [ ] Implement fill limit logic
- [ ] Implement missingness features
- [ ] **Track data source ('kaggle' vs 'api') as feature**
- [ ] Implement alignment validation

### Phase 3: Feature Engineering
- [ ] Volatility features (ATR, BB, realized vol)
- [ ] K-Means S/R levels (rolling window)
- [ ] HMM regime detection (rolling window)
- [ ] Order flow proxy features
- [ ] Funding rate features
- [ ] Liquidity proxy features
- [ ] Fear & Greed features
- [ ] Session/time features (cyclical encoding)
- [ ] Trend strength features (ADX)
- [ ] Momentum exhaustion features (RSI, MACD, divergence)

### Phase 4: Labeling
- [ ] Implement 5-class candle labeling
- [ ] Implement path quality metrics (MAE/MFE)
- [ ] Verify label shift is correct (predicting NEXT candle)
- [ ] Implement label distribution analysis

### Phase 5: Pipeline & Splitting
- [ ] Implement main orchestrator
- [ ] Implement CLI interface
- [ ] Implement chronological splitter
- [ ] Implement final dataset export (Parquet)
- [ ] Generate dataset summary report
- [ ] **Generate data coverage/gaps report**

### Phase 6: Testing & Validation
- [ ] Unit tests for each component
- [ ] Integration tests for full pipeline
- [ ] **CRITICAL: No look-ahead bias tests**
- [ ] **Test hybrid data merge correctness**
- [ ] Data quality report generation

---

## 📁 Output Format

### Final Dataset Schema

```python
# Output Parquet file columns

# Index
'timestamp'                    # datetime64[ns, UTC]

# OHLCV (raw)
'open', 'high', 'low', 'close', 'volume'
'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'

# Funding Rate
'funding_rate_fundingRate', 'funding_rate_fundingRate_age_hours'
'funding_rate_fundingRate_missing'

# Fear & Greed
'fear_greed_value', 'fear_greed_value_age_hours'
'fear_greed_value_missing'

# Open Interest (may be missing for older data)
'open_interest_sumOpenInterest', 'open_interest_sumOpenInterest_age_hours'
'open_interest_sumOpenInterest_missing'

# Long/Short Ratio (may be missing for older data)
'long_short_ratio_longShortRatio', 'long_short_ratio_longShortRatio_age_hours'
'long_short_ratio_longShortRatio_missing'

# Volatility Features
'atr_14', 'atr_ratio', 'bb_width', 'realized_vol_24h', 'vol_percentile'

# S/R Features
'nearest_sr_distance', 'nearest_sr_strength', 'price_vs_sr'

# Regime Features
'hmm_regime', 'hmm_regime_prob_bull', 'hmm_regime_prob_bear', 'hmm_regime_prob_sideways'

# Order Flow Features
'taker_buy_ratio', 'buy_pressure', 'buy_pressure_ma'

# Funding Features
'funding_rate', 'funding_rate_ma_24h', 'funding_zscore'
'funding_extreme_long', 'funding_extreme_short'

# Liquidity Features
'volume_ratio', 'spread_proxy', 'liquidity_score', 'low_liquidity'

# Sentiment Features
'fear_greed', 'fear_greed_ma_7d', 'extreme_fear', 'extreme_greed'

# Session Features
'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
'session_asia', 'session_london', 'session_newyork', 'session_overlap'
'is_weekend'

# Trend Features
'adx', 'plus_di', 'minus_di', 'sma_slope_20', 'trend_score'
'hh_count_10', 'll_count_10'

# Momentum Features
'rsi', 'rsi_oversold', 'rsi_overbought'
'macd', 'macd_signal', 'macd_hist'
'bullish_exhaustion', 'bearish_exhaustion'
'momentum_10', 'momentum_24'

# Labels (targets)
'label'                        # 0-4 (5 classes)
'next_return'                  # Continuous return
'next_direction'               # -1, 0, +1
'next_mae_long'                # Path quality
'next_mfe_long'
'next_mae_short'
'next_mfe_short'
```

### File Output

```
data/
├── kaggle/                          # 🆕 Downloaded Kaggle CSVs (hybrid source)
│   ├── open_interest.csv
│   ├── long_short_ratio.csv
│   ├── taker_buy_sell_volume.csv
│   └── README.md                    # Download instructions
│
├── daily/                           # 🆕 Daily collected data (future-proofing)
│   ├── open_interest_2025-01-15.parquet
│   ├── long_short_ratio_2025-01-15.parquet
│   └── ...
│
├── raw/
│   ├── ohlcv_BTCUSDT_1h.parquet
│   ├── funding_rate_BTCUSDT.parquet
│   ├── fear_greed.parquet
│   ├── open_interest_BTCUSDT_1h.parquet      # Merged: Kaggle + API
│   ├── long_short_ratio_BTCUSDT_1h.parquet   # Merged: Kaggle + API
│   └── taker_volume_BTCUSDT_1h.parquet       # Merged: Kaggle + API
│
├── aligned/
│   └── aligned_BTCUSDT_1h.parquet
│
├── featured/
│   └── featured_BTCUSDT_1h.parquet
│
├── final/
│   ├── train_BTCUSDT_1h.parquet
│   ├── val_BTCUSDT_1h.parquet
│   ├── test_BTCUSDT_1h.parquet
│   └── metadata.json          # Contains date ranges, feature list, data gaps, etc.
│
└── reports/
    ├── data_quality_report.html
    ├── data_coverage_gaps.png       # 🆕 Visualize any gaps
    ├── feature_correlations.png
    └── label_distribution.png
```

---

## 🛡️ Non-Fragile Architecture (Long-Term Robustness)

This section outlines design principles that ensure the system remains robust and maintainable for years, not months.

### Core Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NON-FRAGILE DESIGN PRINCIPLES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. GRACEFUL DEGRADATION                                                │
│     - System works with partial data                                    │
│     - Missing features don't crash the pipeline                         │
│     - Model adapts to available features                                │
│                                                                          │
│  2. EXPLICIT OVER IMPLICIT                                              │
│     - All data sources documented with availability dates               │
│     - Feature provenance tracked (which source, when)                   │
│     - No hidden assumptions about data                                  │
│                                                                          │
│  3. RELATIVE OVER ABSOLUTE                                              │
│     - Use "last N years" instead of hardcoded dates                     │
│     - Dynamic end date (today, not a fixed date)                        │
│     - Rolling windows that move with time                               │
│                                                                          │
│  4. SCHEMA EVOLUTION                                                    │
│     - New features can be added without breaking old models             │
│     - Versioned feature sets                                            │
│     - Backward-compatible data format                                   │
│                                                                          │
│  5. SOURCE ABSTRACTION                                                  │
│     - Easy to swap data sources (e.g., Binance → another exchange)     │
│     - Common interface for all fetchers                                 │
│     - No tight coupling to specific APIs                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Feature Registry Pattern

```python
# config/feature_registry.py

"""
Central registry of all features with metadata.
This enables:
- Automatic feature selection based on date range
- Versioned feature sets for reproducibility
- Easy addition of new features
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class FeatureCategory(Enum):
    CORE = "core"           # Always required
    SENTIMENT = "sentiment" # Market sentiment
    FUTURES = "futures"     # Futures-specific
    DERIVED = "derived"     # Computed from other features


@dataclass
class FeatureDefinition:
    name: str
    category: FeatureCategory
    source: str
    available_from: str        # ISO date or 'derived'
    dependencies: List[str]    # Other features this depends on
    version: str = "1.0"
    description: str = ""
    deprecated: bool = False
    deprecated_reason: Optional[str] = None


# Central Feature Registry
FEATURE_REGISTRY = {
    # === CORE FEATURES (Always Available) ===
    'open': FeatureDefinition(
        name='open', category=FeatureCategory.CORE,
        source='binance_ohlcv', available_from='2017-01-01',
        dependencies=[], description='Candle open price'
    ),
    'high': FeatureDefinition(
        name='high', category=FeatureCategory.CORE,
        source='binance_ohlcv', available_from='2017-01-01',
        dependencies=[], description='Candle high price'
    ),
    # ... (all OHLCV features)
    
    # === FUTURES FEATURES (From Sept 2019) ===
    'funding_rate': FeatureDefinition(
        name='funding_rate', category=FeatureCategory.FUTURES,
        source='binance_funding', available_from='2019-09-15',
        dependencies=[], description='8-hour funding rate'
    ),
    'open_interest': FeatureDefinition(
        name='open_interest', category=FeatureCategory.FUTURES,
        source='hybrid_kaggle_api', available_from='2019-09-15',
        dependencies=[], description='Total open interest'
    ),
    
    # === DERIVED FEATURES (Computed) ===
    'atr_14': FeatureDefinition(
        name='atr_14', category=FeatureCategory.DERIVED,
        source='computed', available_from='derived',
        dependencies=['high', 'low', 'close'],
        description='14-period Average True Range'
    ),
    'hmm_regime': FeatureDefinition(
        name='hmm_regime', category=FeatureCategory.DERIVED,
        source='computed', available_from='derived',
        dependencies=['close', 'atr_14'],
        description='HMM-detected market regime'
    ),
}


class FeatureSelector:
    """
    Selects features based on date range and requirements.
    """
    
    def __init__(self, start_date: str, required_categories: List[FeatureCategory] = None):
        self.start_date = start_date
        self.required = required_categories or [FeatureCategory.CORE]
    
    def get_available_features(self) -> List[str]:
        """Returns features available for the date range."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        
        available = []
        for name, defn in FEATURE_REGISTRY.items():
            if defn.deprecated:
                continue
            
            if defn.available_from == 'derived':
                # Check if all dependencies are available
                deps_available = all(
                    d in available or FEATURE_REGISTRY[d].available_from == 'derived'
                    for d in defn.dependencies
                )
                if deps_available:
                    available.append(name)
            else:
                avail_date = datetime.strptime(defn.available_from, "%Y-%m-%d")
                if avail_date <= start:
                    available.append(name)
        
        return available
    
    def get_feature_version_hash(self) -> str:
        """
        Returns a hash of the feature set for reproducibility.
        Two runs with same hash = same features.
        """
        import hashlib
        features = sorted(self.get_available_features())
        versions = [FEATURE_REGISTRY[f].version for f in features]
        content = str(list(zip(features, versions)))
        return hashlib.md5(content.encode()).hexdigest()[:8]
```

### 2. Dataset Metadata (Reproducibility)

```python
# Every dataset output includes comprehensive metadata

METADATA_SCHEMA = {
    # === Identification ===
    "dataset_id": "uuid",
    "created_at": "ISO timestamp",
    "created_by": "pipeline version",
    
    # === Date Range ===
    "date_range": {
        "start": "2019-09-15",
        "end": "2025-12-14",
        "total_hours": 45000,
        "total_days": 1875
    },
    
    # === Data Sources ===
    "sources": {
        "ohlcv": {"source": "binance_api", "rows": 45000},
        "funding_rate": {"source": "binance_api", "rows": 5625},
        "open_interest": {
            "source": "hybrid",
            "kaggle_rows": 40000,
            "api_rows": 720,
            "gap_hours": 48  # Gap between Kaggle and API
        }
    },
    
    # === Features ===
    "features": {
        "version_hash": "a1b2c3d4",
        "total_features": 87,
        "categories": {
            "core": 9,
            "futures": 12,
            "derived": 66
        },
        "list": ["open", "high", "low", ...]
    },
    
    # === Quality Metrics ===
    "quality": {
        "missing_pct": {"funding_rate": 0.1, "open_interest": 2.3},
        "gaps_detected": [
            {"feature": "open_interest", "start": "2024-10-01", "end": "2024-10-15"}
        ],
        "lookahead_bias_check": "PASSED"
    },
    
    # === Labels ===
    "labels": {
        "distribution": {
            "HIGH_BULL": 0.08,
            "BULL": 0.22,
            "RANGE_BOUND": 0.40,
            "BEAR": 0.22,
            "LOW_BEAR": 0.08
        }
    },
    
    # === Split Info ===
    "splits": {
        "train": {"start": "2019-09-15", "end": "2023-06-01", "rows": 31500},
        "val": {"start": "2023-06-01", "end": "2024-04-15", "rows": 6750},
        "test": {"start": "2024-04-15", "end": "2025-12-14", "rows": 6750}
    }
}
```

### 3. Graceful Degradation

```python
# pipeline/graceful_degradation.py

"""
Handle missing or unavailable data gracefully.
Never crash - always produce usable output.
"""

class GracefulPipeline:
    """
    Pipeline that degrades gracefully when data is missing.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.warnings = []
        self.degradations = []
    
    def fetch_with_fallback(self, primary_source, fallback_source, name: str):
        """
        Try primary source, fall back if it fails.
        """
        try:
            data = primary_source.fetch()
            return data, "primary"
        except Exception as e:
            self.warnings.append(f"{name}: Primary source failed ({e})")
            
            if fallback_source:
                try:
                    data = fallback_source.fetch()
                    self.degradations.append(f"{name}: Using fallback source")
                    return data, "fallback"
                except Exception as e2:
                    self.warnings.append(f"{name}: Fallback also failed ({e2})")
            
            # Return empty with missingness markers
            self.degradations.append(f"{name}: Marking as missing")
            return self._create_missing_placeholder(name), "missing"
    
    def _create_missing_placeholder(self, feature_name: str):
        """
        Create a placeholder for missing data.
        Model will see _missing=1 feature.
        """
        return {
            f"{feature_name}": np.nan,
            f"{feature_name}_missing": 1,
            f"{feature_name}_source": "unavailable"
        }
    
    def get_degradation_report(self) -> dict:
        """
        Report what degradations occurred.
        Important for understanding model limitations.
        """
        return {
            "warnings": self.warnings,
            "degradations": self.degradations,
            "fully_operational": len(self.degradations) == 0
        }
```

### 4. Source Abstraction (Easy to Swap)

```python
# fetchers/base.py

from abc import ABC, abstractmethod
from typing import Protocol


class DataFetcher(Protocol):
    """
    All data fetchers implement this interface.
    This allows easy swapping of sources.
    """
    
    @abstractmethod
    def fetch(self, start: str, end: str, symbol: str) -> pd.DataFrame:
        """Fetch data for date range."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return source identifier."""
        pass
    
    @abstractmethod
    def get_rate_limit(self) -> dict:
        """Return rate limit info."""
        pass


# Example: Easy to add new exchange
class BinanceFetcher(DataFetcher):
    def get_source_name(self) -> str:
        return "binance"

class BybitFetcher(DataFetcher):  # Future expansion
    def get_source_name(self) -> str:
        return "bybit"

class OKXFetcher(DataFetcher):  # Future expansion
    def get_source_name(self) -> str:
        return "okx"


# Factory pattern for source selection
def get_fetcher(source: str, **kwargs) -> DataFetcher:
    """
    Factory to create fetchers.
    Easy to add new sources without changing existing code.
    """
    fetchers = {
        "binance": BinanceFetcher,
        "bybit": BybitFetcher,
        "okx": OKXFetcher,
    }
    
    if source not in fetchers:
        raise ValueError(f"Unknown source: {source}. Available: {list(fetchers.keys())}")
    
    return fetchers[source](**kwargs)
```

### 5. Version Pinning for Reproducibility

```python
# config/versions.py

"""
Pin versions of external dependencies and data schemas.
Ensures reproducibility across time.
"""

SCHEMA_VERSION = "1.0.0"
FEATURE_SET_VERSION = "2024.12"

DATA_SOURCE_VERSIONS = {
    "binance_api": "v3",
    "alternative_me_fng": "v1", 
    "kaggle_jesusgraterol": "2024-Q4"
}

# Changelog for breaking changes
CHANGELOG = {
    "1.0.0": "Initial release",
    # Future:
    # "1.1.0": "Added new sentiment features",
    # "2.0.0": "BREAKING: Changed label definitions"
}
```

### 6. Self-Healing Data Collection

```python
# scripts/self_healing_collector.py

"""
Data collector that automatically handles issues.
"""

class SelfHealingCollector:
    """
    Collector that:
    - Detects and fills gaps
    - Retries failed requests
    - Alerts on persistent issues
    - Maintains data quality
    """
    
    def run_health_check(self) -> dict:
        """
        Daily health check for data pipeline.
        """
        issues = []
        
        # Check for gaps
        gaps = self.detect_gaps()
        if gaps:
            issues.append({"type": "gap", "details": gaps})
            self.attempt_gap_fill(gaps)
        
        # Check for stale data
        staleness = self.check_staleness()
        if staleness['hours_since_update'] > 24:
            issues.append({"type": "stale", "details": staleness})
        
        # Check data quality
        quality = self.check_quality()
        if quality['anomalies']:
            issues.append({"type": "quality", "details": quality})
        
        return {
            "status": "healthy" if not issues else "issues_detected",
            "issues": issues,
            "last_check": datetime.utcnow().isoformat()
        }
```

### Summary: Why This Architecture Survives

| Threat | Mitigation |
|--------|------------|
| API changes | Source abstraction, easy to swap |
| New features needed | Feature registry, versioned additions |
| Data gaps | Graceful degradation, missingness tracking |
| Reproducibility | Metadata, version pinning, config files |
| Stale data | Self-healing collector, daily jobs |
| Date handling | Relative dates, dynamic end date |
| Breaking changes | Schema versioning, changelog |

---

## ⚠️ Critical Reminders

1. **NO LOOK-AHEAD BIAS** - This is the #1 priority. All features must only use data that was available at decision time.

2. **Publication delays matter** - Fear & Greed dated "Jan 15" isn't available until ~09:00 UTC on Jan 15.

3. **HYBRID DATA STRATEGY** - Open Interest, L/S Ratio, Taker Volume use Kaggle (historical) + API (recent 30 days). Check for gaps!

4. **Walk-forward split** - NEVER shuffle the data. Train must come before Val, Val before Test.

5. **Missing data is information** - Track missingness as features, don't just drop rows.

6. **Validate everything** - Run the no-lookahead tests before ANY model training.

7. **Cache API responses** - Store raw responses to avoid re-fetching during development.

---

## 🚀 Getting Started

```bash
# 1. Clone and setup
cd trading_data_pipeline
pip install -r requirements.txt

# 2. Setup Kaggle CLI (REQUIRED for hybrid data)
# Get API key from: kaggle.com/account -> Create New Token
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Download Kaggle historical datasets
mkdir -p data/kaggle
kaggle datasets download -d jesusgraterol/bitcoin-open-interest-binance-futures -p data/kaggle/ --unzip
kaggle datasets download -d jesusgraterol/bitcoin-longshort-ratio-binance-futures -p data/kaggle/ --unzip
kaggle datasets download -d jesusgraterol/bitcoin-taker-buysell-volume-binance-futures -p data/kaggle/ --unzip

# 4. Fetch all data (default: 2024-12-14 to 2025-12-14)
# This will merge Kaggle historical data with recent API data
python scripts/fetch_all_data.py

# 5. With custom date range
python scripts/fetch_all_data.py --start 2023-01-01 --end 2024-01-01

# 6. Validate dataset (checks for gaps, look-ahead bias, etc.)
python scripts/validate_dataset.py

# 7. View generated files
ls -la data/final/

# 8. (Optional) Setup daily collection for future-proofing
python scripts/setup_daily_collector.py
```

---

## Dependencies

```
# requirements.txt

# Data fetching
requests>=2.28.0
aiohttp>=3.8.0
kaggle>=1.6.0        # 🆕 For downloading Kaggle datasets

# Data processing
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0  # For Parquet

# ML features
scikit-learn>=1.3.0  # For K-Means
hmmlearn>=0.3.0      # For HMM

# Utilities
python-dateutil>=2.8.0
pytz>=2023.3
tqdm>=4.65.0
click>=8.1.0         # For CLI
schedule>=1.2.0      # 🆕 For daily collection job

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Optional: visualization for reports
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Kaggle CLI Setup

```bash
# 1. Install kaggle CLI
pip install kaggle

# 2. Get API credentials from kaggle.com/account
# 3. Create credentials file
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 4. Download datasets
kaggle datasets download -d jesusgraterol/bitcoin-open-interest-binance-futures -p data/kaggle/ --unzip
kaggle datasets download -d jesusgraterol/bitcoin-longshort-ratio-binance-futures -p data/kaggle/ --unzip
kaggle datasets download -d jesusgraterol/bitcoin-taker-buysell-volume-binance-futures -p data/kaggle/ --unzip
```

---

**Remember: The data layer is the foundation. Get it RIGHT.**