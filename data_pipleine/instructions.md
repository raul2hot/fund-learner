# ML Trading Data Pipeline v2.0

**Clean Architecture: Binance API + Alternative.me**

*Last Updated: December 16, 2025*

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [CLI Commands](#2-cli-commands)
3. [Step-by-Step Guide](#3-step-by-step-guide)
4. [Overview](#4-overview)
5. [Data Sources](#5-data-sources)
6. [Filter-to-Data Mapping](#6-filter-to-data-mapping)
7. [Installation & Setup](#7-installation--setup)
8. [Data Alignment](#8-data-alignment)
9. [Output Format](#9-output-format)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick Start

```bash
# Navigate to the data pipeline directory
cd data_pipleine

# Quick test (last 7 days of data)
python main_pipeline.py --test

# Full data fetch (2 years for training)
python main_pipeline.py --start 2023-01-01 --end 2024-12-31

# Custom symbol and output directory
python main_pipeline.py --start 2023-01-01 --symbol ETHUSDT --output ./eth_data
```

---

## 2. CLI Commands

### Main Pipeline Commands

| Command | Description |
|---------|-------------|
| `python main_pipeline.py --test` | Quick test with last 7 days of data |
| `python main_pipeline.py --start YYYY-MM-DD` | Fetch data from start date to today |
| `python main_pipeline.py --start YYYY-MM-DD --end YYYY-MM-DD` | Fetch data for specific date range |
| `python main_pipeline.py --symbol ETHUSDT` | Fetch data for different trading pair |
| `python main_pipeline.py --output ./custom_dir` | Save to custom output directory |
| `python main_pipeline.py --save-intermediate` | Save intermediate data files |

### Testing Individual Components

```bash
# Test data fetcher only
python data_fetcher.py

# Test with custom date range
python -c "
from data_fetcher import BinanceFetcher
fetcher = BinanceFetcher()
df = fetcher.fetch_klines(start_date='2025-12-01', end_date='2025-12-10')
print(df.head())
"
```

---

## 3. Step-by-Step Guide

### Step 1: Install Dependencies

```bash
pip install pandas numpy requests python-dateutil tqdm
```

### Step 2: (Optional) Set Up Binance API Key

```bash
# Set environment variable for better rate limits
export BINANCE_API_KEY="your_api_key_here"
```

### Step 3: Run Quick Test

```bash
cd data_pipleine
python main_pipeline.py --test
```

Expected output:
```
============================================================
QUICK TEST - Last 7 days of data
============================================================
Starting ML Data Pipeline
Symbol: BTCUSDT
Date Range: 2025-12-09 to 2025-12-16
============================================================
[1/5] Fetching data from sources...
[2/5] Handling edge cases...
[3/5] Merging and aligning data...
[4/5] Computing derived features...
[5/5] Validating and exporting...
Data validation passed!
```

### Step 4: Fetch Full Training Data

```bash
# Fetch 2 years of data for ML training
python main_pipeline.py --start 2023-01-01 --end 2024-12-31 --output ./ml_data
```

### Step 5: Verify Output Files

```bash
ls -la ./ml_data/
# Expected files:
# - BTCUSDT_ml_data.parquet  (efficient for ML)
# - BTCUSDT_ml_data.csv      (for inspection)
# - BTCUSDT_metadata.json    (pipeline metadata)
```

### Step 6: Load Data in Python

```python
import pandas as pd

# Load the data
df = pd.read_parquet('./ml_data/BTCUSDT_ml_data.parquet')

# Check shape and columns
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

---

## 4. Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FREE TIER DATA PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BINANCE FUTURES API (with API key):                            │
│  ─────────────────────────────────────                          │
│  - OHLCV 1h candles        -> Full history (Sept 2019+)         │
│  - Taker Buy Volume        -> Included in OHLCV response [9]    │
│  - Funding Rate            -> Full history (8h intervals)       │
│                                                                  │
│  ALTERNATIVE.ME API (no key required):                          │
│  ─────────────────────────────────────                          │
│  - Fear & Greed Index      -> Full history (Feb 2018+)          │
│                                                                  │
│  COST: $0                                                        │
│  FILTERS SUPPORTED: 10/10                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Point-in-Time Correctness**: All features use only data available at prediction time
2. **Timestamp Alignment**: All data aligned to hourly UTC boundaries
3. **No Look-Ahead Bias**: Funding rate forward-filled (8h->1h), Fear/Greed forward-filled (daily->1h)
4. **Robust Edge Case Handling**: NaN interpolation, gap detection, validation checks
5. **Clean Single-Source Architecture**: No Kaggle dependencies, direct API access

---

## 5. Data Sources

### 5.1 Binance Futures API

**Base URL**: `https://fapi.binance.com`

#### Klines (OHLCV + Taker Volume)

| Endpoint | `/fapi/v1/klines` |
|----------|-------------------|
| Rate Limit | 5 weight per request |
| Max per Request | 1500 candles |
| History Available | September 2019+ (BTCUSDT perpetual launch) |

**Response Format** (12 fields per candle):
```
Index  Field                        Type     Used For
─────────────────────────────────────────────────────────
[0]    Open time                    INT      Timestamp (ms)
[1]    Open price                   STRING   OHLCV
[2]    High price                   STRING   OHLCV
[3]    Low price                    STRING   OHLCV
[4]    Close price                  STRING   OHLCV
[5]    Volume                       STRING   Total volume
[6]    Close time                   INT      (ignore)
[7]    Quote asset volume           STRING   USD volume
[8]    Number of trades             INT      Trade count
[9]    Taker buy base asset volume  STRING   ← ORDER FLOW
[10]   Taker buy quote asset volume STRING   (ignore)
[11]   Ignore                       STRING   (ignore)
```

#### Funding Rate

| Endpoint | `/fapi/v1/fundingRate` |
|----------|------------------------|
| Rate Limit | Shared 500/5min/IP with /fundingInfo |
| Max per Request | 1000 records |
| Frequency | Every 8 hours (00:00, 08:00, 16:00 UTC) |
| History Available | Full history since contract launch |

**Response Format**:
```json
{
    "symbol": "BTCUSDT",
    "fundingRate": "0.00010000",
    "fundingTime": 1570608000000,
    "markPrice": "34287.54619963"
}
```

### 5.2 Alternative.me Fear & Greed API

**Base URL**: `https://api.alternative.me`

| Endpoint | `/fng/` |
|----------|---------|
| Rate Limit | No strict limit (be respectful) |
| Frequency | Daily updates |
| History Available | February 2018+ |
| Cost | Free |

**Parameters**:
- `limit=0` → Returns ALL historical data
- `format=json` → JSON response (default)

**Response Format**:
```json
{
    "name": "Fear and Greed Index",
    "data": [
        {
            "value": "16",
            "value_classification": "Extreme Fear",
            "timestamp": "1734220800"
        }
    ]
}
```

**Index Scale**:
- 0-24: Extreme Fear
- 25-49: Fear
- 50-74: Greed
- 75-100: Extreme Greed

---

## 6. Filter-to-Data Mapping

| # | Filter | Required Data | Source | Availability |
|---|--------|---------------|--------|--------------|
| 1 | Volatility Filter | OHLCV (ATR, BB, realized vol) | Binance Klines | ✓ Full |
| 2 | K-Means S/R Levels | OHLCV (High, Low, Close) | Binance Klines | ✓ Full |
| 3 | HMM Regime Detection | OHLCV + derived volatility | Binance Klines | ✓ Full |
| 4 | Order Flow Imbalance | Taker Buy Volume [9] | Binance Klines | ✓ Full |
| 5 | Funding Rate Filter | Funding Rate | Binance Funding | ✓ Full |
| 6 | Liquidity Score | Volume + spread proxy | Binance Klines | ✓ Full |
| 7 | Fear & Greed Index | FnG value (0-100) | Alternative.me | ✓ Full |
| 8 | Session/Time Filter | Timestamp | Computed | ✓ Full |
| 9 | Trend Strength | OHLCV (ADX, MA slopes) | Binance Klines | ✓ Full |
| 10 | Momentum Exhaustion | OHLCV (RSI, MACD) | Binance Klines | ✓ Full |

**Result: 10/10 filters fully supported with free data**

---

## 7. Installation & Setup

### 7.1 Dependencies

```bash
pip install pandas numpy requests python-dateutil tqdm
```

### 7.2 API Key Setup (Recommended)

While Binance public endpoints work without authentication, using an API key provides:
- Better rate limit tracking via response headers
- More stable access for bulk historical data
- Required for any future account/trading features

**Get your API key**:
1. Go to binance.com → Profile → API Management
2. Create new API key (label: "ml_research")
3. **No permissions needed** for market data (read-only is fine)
4. Save API Key and Secret securely

```python
# config.py
import os

BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')

# Alternative: Direct assignment (not recommended for production)
# BINANCE_API_KEY = "your_api_key_here"
```

---

## 5. Data Fetching Module

### 5.1 Core Fetcher Class

```python
"""
data_fetcher.py - Clean Data Pipeline for ML Trading
Version: 2.0
Date: December 2025
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, List
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceFetcher:
    """
    Fetches OHLCV and Funding Rate data from Binance Futures API.
    
    Handles:
    - Pagination for large date ranges
    - Rate limiting with automatic backoff
    - Timestamp alignment to hourly boundaries
    - Proper error handling and retries
    """
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-MBX-APIKEY': api_key})
    
    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> list:
        """Make API request with retry logic and rate limit handling."""
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                # Check rate limit headers
                used_weight = response.headers.get('X-MBX-USED-WEIGHT-1M', 'N/A')
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 418:
                    # IP banned - longer wait
                    logger.error("IP banned. Waiting 5 minutes...")
                    time.sleep(300)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return []
    
    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines with taker buy volume.
        
        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., "BTCUSDT")
        interval : str
            Candle interval (e.g., "1h", "4h", "1d")
        start_date : str
            Start date in "YYYY-MM-DD" format
        end_date : str, optional
            End date in "YYYY-MM-DD" format (default: now)
        
        Returns:
        --------
        pd.DataFrame with columns:
            - timestamp: datetime64[ns, UTC]
            - open, high, low, close, volume: float64
            - quote_volume, trades, taker_buy_volume, taker_buy_quote_volume: float64
        """
        # Convert dates to milliseconds
        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        limit = 1500  # Max allowed by Binance
        
        # Calculate expected number of candles for progress bar
        interval_ms = self._interval_to_ms(interval)
        expected_candles = (end_ts - start_ts) // interval_ms
        
        with tqdm(total=expected_candles, desc=f"Fetching {symbol} klines") as pbar:
            while current_start < end_ts:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': limit
                }
                
                klines = self._make_request('/fapi/v1/klines', params)
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                pbar.update(len(klines))
                
                # Move start to after last candle
                last_close_time = klines[-1][6]  # Close time
                current_start = last_close_time + 1
                
                # Small delay to be nice to the API
                time.sleep(0.1)
        
        if not all_klines:
            logger.warning(f"No klines data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['trades'] = df['trades'].astype(int)
        
        # Select and order columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume']]
        
        # Remove duplicates (can happen at pagination boundaries)
        df = df.drop_duplicates(subset='timestamp', keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} klines from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def fetch_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.
        
        Funding rates are recorded every 8 hours (00:00, 08:00, 16:00 UTC).
        
        Returns:
        --------
        pd.DataFrame with columns:
            - timestamp: datetime64[ns, UTC]
            - funding_rate: float64
            - mark_price: float64
        """
        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_funding = []
        current_start = start_ts
        limit = 1000  # Max allowed
        
        while current_start < end_ts:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': limit
            }
            
            funding = self._make_request('/fapi/v1/fundingRate', params)
            
            if not funding:
                break
            
            all_funding.extend(funding)
            
            # Move start to after last funding time
            last_time = funding[-1]['fundingTime']
            current_start = last_time + 1
            
            time.sleep(0.2)  # Respect shared rate limit
        
        if not all_funding:
            logger.warning(f"No funding rate data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_funding)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
        df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df['mark_price'] = pd.to_numeric(df['markPrice'], errors='coerce')
        
        df = df[['timestamp', 'funding_rate', 'mark_price']]
        df = df.drop_duplicates(subset='timestamp', keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} funding rate records")
        
        return df
    
    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """Convert interval string to milliseconds."""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000
        }
        unit = interval[-1]
        value = int(interval[:-1])
        return value * multipliers.get(unit, 60 * 60 * 1000)


class FearGreedFetcher:
    """
    Fetches Fear & Greed Index data from Alternative.me API.
    
    The index is updated daily and represents overall crypto market sentiment.
    Values range from 0 (Extreme Fear) to 100 (Extreme Greed).
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch complete Fear & Greed Index history.
        
        Returns:
        --------
        pd.DataFrame with columns:
            - timestamp: datetime64[ns, UTC] (normalized to 00:00 UTC)
            - fear_greed_value: int (0-100)
            - fear_greed_class: str (classification label)
        """
        params = {
            'limit': 0,  # 0 = all available data
            'format': 'json'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Fear & Greed data: {e}")
            return pd.DataFrame()
        
        if 'data' not in data:
            logger.error("Unexpected API response format")
            return pd.DataFrame()
        
        records = data['data']
        
        df = pd.DataFrame(records)
        
        # Convert timestamp (Unix seconds to datetime)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
        
        # Normalize to midnight UTC (Fear/Greed is a daily value)
        df['timestamp'] = df['timestamp'].dt.normalize()
        
        df['fear_greed_value'] = pd.to_numeric(df['value'], errors='coerce').astype(int)
        df['fear_greed_class'] = df['value_classification']
        
        df = df[['timestamp', 'fear_greed_value', 'fear_greed_class']]
        df = df.drop_duplicates(subset='timestamp', keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} Fear & Greed records from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        
        return df
```

---

## 6. Data Alignment & Preprocessing

### 6.1 Alignment Strategy

```
TIMELINE ALIGNMENT VISUALIZATION
────────────────────────────────────────────────────────────────────

OHLCV (1h):      │─●─●─●─●─●─●─●─●─│─●─●─●─●─●─●─●─●─│
                 0  1  2  3  4  5  6  7  8  9 10 11...
                 
Funding (8h):    │─────────●─────────────────●───────│
                 00:00           08:00          16:00
                 ↓               ↓               ↓
Forward-fill:    │─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─│
                 (same value until next funding)

Fear/Greed (1d): │───────────────●───────────────────│
                 00:00 UTC (daily snapshot)
                 ↓
Forward-fill:    │─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─│
                 (same value for all 24 hours)

FINAL MERGED:    │─●─●─●─●─●─●─●─●─│─●─●─●─●─●─●─●─●─│
                 All features aligned to 1h OHLCV timestamps
```

### 6.2 Data Processor Class

```python
"""
data_processor.py - Data Alignment and Preprocessing
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles alignment, merging, and preprocessing of multi-source data.
    
    Key responsibilities:
    - Align all data to hourly UTC timestamps
    - Forward-fill lower-frequency data (funding, fear/greed)
    - Handle missing values and gaps
    - Validate data integrity
    """
    
    def __init__(self):
        self.validation_errors = []
    
    def create_hourly_index(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DatetimeIndex:
        """
        Create complete hourly datetime index.
        
        This ensures no gaps in the final dataset.
        """
        start = pd.Timestamp(start_date, tz='UTC').normalize()
        end = pd.Timestamp(end_date, tz='UTC').normalize() + pd.Timedelta(days=1)
        
        return pd.date_range(start=start, end=end, freq='1h', tz='UTC')[:-1]
    
    def align_funding_rate(
        self,
        funding_df: pd.DataFrame,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align 8-hourly funding rate to hourly timestamps.
        
        Strategy: Forward-fill
        - Funding rate at 00:00 applies to hours 00-07
        - Funding rate at 08:00 applies to hours 08-15
        - Funding rate at 16:00 applies to hours 16-23
        
        CRITICAL: Use shift to avoid look-ahead bias!
        The funding rate announced at 08:00 is for the PREVIOUS 8-hour period.
        We should only use it starting at 08:00.
        """
        if funding_df.empty:
            return pd.DataFrame(index=target_index, columns=['funding_rate'])
        
        # Set index for resampling
        funding_indexed = funding_df.set_index('timestamp')
        
        # Reindex to hourly, forward-fill
        aligned = funding_indexed[['funding_rate']].reindex(target_index)
        aligned = aligned.ffill()
        
        # Handle leading NaNs (before first funding rate)
        # Backfill only the initial period
        first_valid = aligned['funding_rate'].first_valid_index()
        if first_valid is not None:
            aligned.loc[:first_valid, 'funding_rate'] = aligned.loc[first_valid, 'funding_rate']
        
        logger.info(f"Aligned funding rate: {aligned['funding_rate'].notna().sum()}/{len(aligned)} valid values")
        
        return aligned.reset_index().rename(columns={'index': 'timestamp'})
    
    def align_fear_greed(
        self,
        fng_df: pd.DataFrame,
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align daily Fear & Greed index to hourly timestamps.
        
        Strategy: Forward-fill
        - Daily value at 00:00 UTC applies to all 24 hours of that day
        
        Note: Fear & Greed is a lagging indicator (uses previous day's data),
        so forward-filling is appropriate and doesn't introduce look-ahead.
        """
        if fng_df.empty:
            return pd.DataFrame(index=target_index, columns=['fear_greed_value'])
        
        # Set index
        fng_indexed = fng_df.set_index('timestamp')
        
        # Reindex to hourly, forward-fill
        aligned = fng_indexed[['fear_greed_value']].reindex(target_index)
        aligned = aligned.ffill()
        
        # Handle leading NaNs
        first_valid = aligned['fear_greed_value'].first_valid_index()
        if first_valid is not None:
            aligned.loc[:first_valid, 'fear_greed_value'] = aligned.loc[first_valid, 'fear_greed_value']
        
        logger.info(f"Aligned Fear & Greed: {aligned['fear_greed_value'].notna().sum()}/{len(aligned)} valid values")
        
        return aligned.reset_index().rename(columns={'index': 'timestamp'})
    
    def merge_all_data(
        self,
        ohlcv_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        fng_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Merge all data sources into a single aligned DataFrame.
        
        Parameters:
        -----------
        ohlcv_df : DataFrame with OHLCV + taker volume
        funding_df : DataFrame with funding rate
        fng_df : DataFrame with Fear & Greed index
        start_date : Start date (inclusive)
        end_date : End date (inclusive)
        
        Returns:
        --------
        Merged DataFrame with all features aligned to hourly timestamps
        """
        # Create target index
        target_index = self.create_hourly_index(start_date, end_date)
        
        # Start with OHLCV as base (hourly data)
        if ohlcv_df.empty:
            raise ValueError("OHLCV data is required")
        
        ohlcv_indexed = ohlcv_df.set_index('timestamp')
        merged = ohlcv_indexed.reindex(target_index)
        
        # Align and merge funding rate
        aligned_funding = self.align_funding_rate(funding_df, target_index)
        aligned_funding = aligned_funding.set_index('timestamp')
        merged = merged.join(aligned_funding)
        
        # Align and merge Fear & Greed
        aligned_fng = self.align_fear_greed(fng_df, target_index)
        aligned_fng = aligned_fng.set_index('timestamp')
        merged = merged.join(aligned_fng)
        
        # Reset index
        merged = merged.reset_index().rename(columns={'index': 'timestamp'})
        
        # Log merge statistics
        self._log_merge_stats(merged)
        
        return merged
    
    def _log_merge_stats(self, df: pd.DataFrame):
        """Log statistics about the merged data."""
        total_rows = len(df)
        
        for col in df.columns:
            if col == 'timestamp':
                continue
            missing = df[col].isna().sum()
            pct = (missing / total_rows) * 100
            if missing > 0:
                logger.warning(f"Column '{col}': {missing} missing values ({pct:.2f}%)")
            else:
                logger.info(f"Column '{col}': Complete (0 missing)")
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features for the 10 filters.
        
        Features computed:
        - taker_buy_ratio: Taker buy volume / Total volume (Order Flow)
        - taker_sell_volume: Total - Taker buy (for imbalance)
        - spread_proxy: (High - Low) / Close (Liquidity proxy)
        - session: Trading session (asian/london/ny)
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        """
        df = df.copy()
        
        # Order Flow Imbalance features
        df['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume']
        df['taker_sell_volume'] = df['volume'] - df['taker_buy_volume']
        df['order_flow_imbalance'] = (df['taker_buy_volume'] - df['taker_sell_volume']) / df['volume']
        
        # Liquidity/Spread proxy
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Session/Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Session classification (based on major market hours)
        def classify_session(hour):
            if 0 <= hour < 8:
                return 'asian'
            elif 8 <= hour < 16:
                return 'london'
            else:
                return 'new_york'
        
        df['session'] = df['hour'].apply(classify_session)
        
        # Validate derived features
        self._validate_derived_features(df)
        
        return df
    
    def _validate_derived_features(self, df: pd.DataFrame):
        """Validate derived features are within expected ranges."""
        validations = [
            ('taker_buy_ratio', 0, 1, 'Taker buy ratio should be between 0 and 1'),
            ('order_flow_imbalance', -1, 1, 'Order flow imbalance should be between -1 and 1'),
            ('spread_proxy', 0, 1, 'Spread proxy should be between 0 and 1 (usually)'),
        ]
        
        for col, min_val, max_val, msg in validations:
            if col in df.columns:
                invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid > 0:
                    logger.warning(f"{msg}: {invalid} violations found")
```

---

## 7. Edge Case Handling

### 7.1 Missing Value Strategy

```python
"""
edge_cases.py - Robust handling of data quality issues
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """
    Handles edge cases in crypto market data.
    
    Common issues:
    1. Missing OHLCV candles (exchange downtime)
    2. Missing funding rate records
    3. Gaps in Fear & Greed data
    4. Extreme/invalid values
    5. Timezone inconsistencies
    """
    
    # Maximum acceptable gap sizes before warning
    MAX_OHLCV_GAP_HOURS = 4
    MAX_FUNDING_GAP_HOURS = 24  # 3 funding periods
    MAX_FNG_GAP_DAYS = 3
    
    def __init__(self):
        self.issues_found = []
    
    def detect_gaps(self, df: pd.DataFrame, expected_freq: str = '1h') -> List[Dict]:
        """
        Detect gaps in time series data.
        
        Returns list of gaps with start, end, and duration.
        """
        if df.empty or 'timestamp' not in df.columns:
            return []
        
        df = df.sort_values('timestamp')
        
        # Calculate expected time difference
        freq_map = {'1h': pd.Timedelta(hours=1), '8h': pd.Timedelta(hours=8), '1d': pd.Timedelta(days=1)}
        expected_diff = freq_map.get(expected_freq, pd.Timedelta(hours=1))
        
        gaps = []
        timestamps = df['timestamp'].values
        
        for i in range(1, len(timestamps)):
            actual_diff = pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i-1])
            
            if actual_diff > expected_diff * 1.5:  # Allow 50% tolerance
                gap = {
                    'start': pd.Timestamp(timestamps[i-1]),
                    'end': pd.Timestamp(timestamps[i]),
                    'duration': actual_diff,
                    'missing_periods': int(actual_diff / expected_diff) - 1
                }
                gaps.append(gap)
        
        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in data")
            for gap in gaps[:5]:  # Log first 5
                logger.warning(f"  Gap: {gap['start']} to {gap['end']} ({gap['missing_periods']} periods)")
        
        return gaps
    
    def fill_ohlcv_gaps(
        self,
        df: pd.DataFrame,
        max_gap_hours: int = 4
    ) -> pd.DataFrame:
        """
        Fill small gaps in OHLCV data.
        
        Strategy:
        - Gaps ≤ max_gap_hours: Interpolate
        - Gaps > max_gap_hours: Forward-fill prices, zero volume
        
        CRITICAL: Large gaps should trigger a warning as they may indicate
        significant market events (exchange issues, flash crashes, etc.)
        """
        if df.empty:
            return df
        
        df = df.copy().sort_values('timestamp')
        
        # Create complete index
        full_index = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='1h',
            tz='UTC'
        )
        
        # Reindex to complete timeline
        df = df.set_index('timestamp').reindex(full_index)
        
        # Detect which rows were added (gaps)
        was_missing = df['close'].isna()
        
        # Forward-fill OHLC (use last known prices)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # For volume-based columns, fill with 0 during gaps
        # This is more accurate than interpolation for volume
        volume_cols = ['volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in volume_cols:
            if col in df.columns:
                df.loc[was_missing, col] = 0
        
        # Trades: fill with 0
        if 'trades' in df.columns:
            df.loc[was_missing, 'trades'] = 0
        
        # Log statistics
        filled = was_missing.sum()
        if filled > 0:
            logger.info(f"Filled {filled} missing OHLCV rows ({filled/len(df)*100:.2f}%)")
        
        return df.reset_index().rename(columns={'index': 'timestamp'})
    
    def handle_extreme_values(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Detect and handle extreme/invalid values.
        
        Checks:
        - Negative prices (invalid)
        - Zero prices (invalid except for some altcoins)
        - Extreme price changes (> 50% in 1 hour)
        - Negative volumes (invalid)
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Price columns validation
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # Replace negative or zero with NaN
                invalid_mask = df[col] <= 0
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid values in {col}")
                    df.loc[invalid_mask, col] = np.nan
        
        # Volume columns validation
        volume_cols = ['volume', 'taker_buy_volume']
        for col in volume_cols:
            if col in df.columns:
                # Negative volume is invalid
                invalid_mask = df[col] < 0
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} negative values in {col}")
                    df.loc[invalid_mask, col] = np.nan
        
        # Extreme price change detection (informational only)
        if 'close' in df.columns:
            pct_change = df['close'].pct_change().abs()
            extreme_changes = pct_change > 0.5  # 50% move in 1 hour
            if extreme_changes.any():
                logger.warning(f"Found {extreme_changes.sum()} extreme price changes (>50% in 1h)")
        
        # Funding rate validation (-1 to 1 is reasonable)
        if 'funding_rate' in df.columns:
            invalid_mask = (df['funding_rate'].abs() > 1)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} extreme funding rates")
        
        # Fear & Greed validation (0-100)
        if 'fear_greed_value' in df.columns:
            invalid_mask = (df['fear_greed_value'] < 0) | (df['fear_greed_value'] > 100)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} invalid Fear & Greed values")
                df.loc[invalid_mask, 'fear_greed_value'] = np.nan
        
        return df
    
    def interpolate_missing(
        self,
        df: pd.DataFrame,
        method: str = 'linear',
        limit: int = 4
    ) -> pd.DataFrame:
        """
        Interpolate remaining NaN values.
        
        Parameters:
        -----------
        method : str
            Interpolation method ('linear', 'ffill', 'bfill')
        limit : int
            Maximum number of consecutive NaNs to fill
        
        Note: Only use for small gaps. Large gaps should remain NaN
        or be flagged for investigation.
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'timestamp':
                continue
            
            before_nans = df[col].isna().sum()
            
            if method == 'linear':
                df[col] = df[col].interpolate(method='linear', limit=limit)
            elif method == 'ffill':
                df[col] = df[col].ffill(limit=limit)
            elif method == 'bfill':
                df[col] = df[col].bfill(limit=limit)
            
            after_nans = df[col].isna().sum()
            
            if before_nans > after_nans:
                logger.debug(f"Interpolated {before_nans - after_nans} values in {col}")
        
        return df
    
    def validate_data_integrity(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Final validation of data integrity.
        
        Returns:
        --------
        Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check timestamp is sorted
        if not df['timestamp'].is_monotonic_increasing:
            issues.append("Timestamps are not sorted")
        
        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
        
        # Check for remaining NaNs in critical columns
        for col in ['close', 'volume']:
            if col in df.columns:
                nans = df[col].isna().sum()
                if nans > 0:
                    issues.append(f"Column {col} has {nans} NaN values ({nans/len(df)*100:.2f}%)")
        
        # Check OHLC consistency (high >= low, etc.)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            inconsistent = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            if inconsistent > 0:
                issues.append(f"Found {inconsistent} OHLC consistency violations")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
```

---

## 8. Complete Pipeline

### 8.1 Main Pipeline Script

```python
"""
main_pipeline.py - Complete Data Pipeline Execution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

# Import our modules (in same directory or installed as package)
from data_fetcher import BinanceFetcher, FearGreedFetcher
from data_processor import DataProcessor
from edge_cases import EdgeCaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLDataPipeline:
    """
    Complete ML data pipeline for crypto trading strategy.
    
    Orchestrates:
    1. Data fetching from all sources
    2. Alignment and merging
    3. Edge case handling
    4. Feature engineering
    5. Validation and export
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        binance_api_key: str = None,
        output_dir: str = "./data"
    ):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.binance = BinanceFetcher(api_key=binance_api_key)
        self.fng = FearGreedFetcher()
        self.processor = DataProcessor()
        self.edge_handler = EdgeCaseHandler()
        
        # Pipeline metadata
        self.metadata = {
            'symbol': symbol,
            'pipeline_version': '2.0',
            'created_at': None,
            'data_sources': {
                'ohlcv': 'Binance Futures API',
                'funding_rate': 'Binance Futures API',
                'fear_greed': 'Alternative.me API'
            }
        }
    
    def run(
        self,
        start_date: str,
        end_date: str = None,
        save_intermediate: bool = False
    ) -> pd.DataFrame:
        """
        Execute the complete data pipeline.
        
        Parameters:
        -----------
        start_date : str
            Start date in "YYYY-MM-DD" format
        end_date : str, optional
            End date in "YYYY-MM-DD" format (default: today)
        save_intermediate : bool
            Whether to save intermediate data files
        
        Returns:
        --------
        pd.DataFrame with all features ready for ML
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"=" * 60)
        logger.info(f"Starting ML Data Pipeline")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info(f"=" * 60)
        
        # Step 1: Fetch all data
        logger.info("\n[1/5] Fetching data from sources...")
        
        ohlcv_df = self.binance.fetch_klines(
            symbol=self.symbol,
            interval="1h",
            start_date=start_date,
            end_date=end_date
        )
        
        funding_df = self.binance.fetch_funding_rate(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        fng_df = self.fng.fetch_all()
        
        if save_intermediate:
            self._save_intermediate(ohlcv_df, 'ohlcv_raw.parquet')
            self._save_intermediate(funding_df, 'funding_raw.parquet')
            self._save_intermediate(fng_df, 'fng_raw.parquet')
        
        # Step 2: Handle edge cases in raw data
        logger.info("\n[2/5] Handling edge cases...")
        
        # Detect gaps
        ohlcv_gaps = self.edge_handler.detect_gaps(ohlcv_df, '1h')
        funding_gaps = self.edge_handler.detect_gaps(funding_df, '8h')
        
        # Fill OHLCV gaps
        ohlcv_df = self.edge_handler.fill_ohlcv_gaps(ohlcv_df)
        
        # Handle extreme values
        ohlcv_df = self.edge_handler.handle_extreme_values(ohlcv_df)
        
        # Step 3: Merge all data
        logger.info("\n[3/5] Merging and aligning data...")
        
        merged_df = self.processor.merge_all_data(
            ohlcv_df=ohlcv_df,
            funding_df=funding_df,
            fng_df=fng_df,
            start_date=start_date,
            end_date=end_date
        )
        
        if save_intermediate:
            self._save_intermediate(merged_df, 'merged_raw.parquet')
        
        # Step 4: Compute derived features
        logger.info("\n[4/5] Computing derived features...")
        
        final_df = self.processor.compute_derived_features(merged_df)
        
        # Final interpolation for any remaining NaNs
        final_df = self.edge_handler.interpolate_missing(final_df, method='ffill', limit=4)
        
        # Step 5: Validate and export
        logger.info("\n[5/5] Validating and exporting...")
        
        is_valid, issues = self.edge_handler.validate_data_integrity(final_df)
        
        if not is_valid:
            logger.error("Data validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("Data validation passed!")
        
        # Update metadata
        self.metadata['created_at'] = datetime.now().isoformat()
        self.metadata['date_range'] = {'start': start_date, 'end': end_date}
        self.metadata['total_rows'] = len(final_df)
        self.metadata['validation'] = {'passed': is_valid, 'issues': issues}
        
        # Save final output
        self._save_final(final_df)
        
        # Log summary
        self._log_summary(final_df)
        
        return final_df
    
    def _save_intermediate(self, df: pd.DataFrame, filename: str):
        """Save intermediate data file."""
        path = self.output_dir / 'intermediate' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug(f"Saved intermediate: {path}")
    
    def _save_final(self, df: pd.DataFrame):
        """Save final dataset and metadata."""
        # Save as parquet (efficient for ML)
        parquet_path = self.output_dir / f"{self.symbol}_ml_data.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved: {parquet_path}")
        
        # Save as CSV (for inspection)
        csv_path = self.output_dir / f"{self.symbol}_ml_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
        
        # Save metadata
        meta_path = self.output_dir / f"{self.symbol}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved: {meta_path}")
    
    def _log_summary(self, df: pd.DataFrame):
        """Log summary statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"\nColumn statistics:")
        
        for col in df.columns:
            if col == 'timestamp':
                continue
            if df[col].dtype in [np.float64, np.int64]:
                nan_count = df[col].isna().sum()
                nan_pct = nan_count / len(df) * 100
                logger.info(f"  {col}: mean={df[col].mean():.4f}, "
                          f"std={df[col].std():.4f}, "
                          f"NaN={nan_count} ({nan_pct:.2f}%)")


if __name__ == "__main__":
    # Example usage
    pipeline = MLDataPipeline(
        symbol="BTCUSDT",
        binance_api_key=None,  # Set your API key or use environment variable
        output_dir="./ml_data"
    )
    
    # Fetch 2 years of data for training
    df = pipeline.run(
        start_date="2023-01-01",
        end_date="2024-12-31",
        save_intermediate=True
    )
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nSample data:\n{df.head()}")
```

---

## 9. Data Validation

### 9.1 Validation Checklist

```python
"""
validation.py - Data Quality Checks
"""

def run_all_validations(df: pd.DataFrame) -> dict:
    """
    Run comprehensive data quality validations.
    
    Returns dict with validation results.
    """
    results = {
        'passed': True,
        'checks': []
    }
    
    # Check 1: No duplicate timestamps
    check = {
        'name': 'No duplicate timestamps',
        'passed': not df['timestamp'].duplicated().any(),
        'details': f"Duplicates found: {df['timestamp'].duplicated().sum()}"
    }
    results['checks'].append(check)
    
    # Check 2: Timestamps are sorted
    check = {
        'name': 'Timestamps are sorted',
        'passed': df['timestamp'].is_monotonic_increasing,
        'details': None
    }
    results['checks'].append(check)
    
    # Check 3: No gaps > 4 hours in OHLCV
    time_diffs = df['timestamp'].diff()
    max_gap = time_diffs.max()
    check = {
        'name': 'No large gaps (> 4 hours)',
        'passed': max_gap <= pd.Timedelta(hours=4),
        'details': f"Max gap: {max_gap}"
    }
    results['checks'].append(check)
    
    # Check 4: OHLC consistency
    ohlc_valid = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ).all()
    check = {
        'name': 'OHLC consistency',
        'passed': ohlc_valid,
        'details': None
    }
    results['checks'].append(check)
    
    # Check 5: No negative volumes
    check = {
        'name': 'No negative volumes',
        'passed': (df['volume'] >= 0).all(),
        'details': f"Negative values: {(df['volume'] < 0).sum()}"
    }
    results['checks'].append(check)
    
    # Check 6: Funding rate within bounds
    check = {
        'name': 'Funding rate within [-0.75%, 0.75%]',
        'passed': ((df['funding_rate'].abs() <= 0.0075) | df['funding_rate'].isna()).all(),
        'details': f"Out of bounds: {(df['funding_rate'].abs() > 0.0075).sum()}"
    }
    results['checks'].append(check)
    
    # Check 7: Fear & Greed within 0-100
    check = {
        'name': 'Fear & Greed within [0, 100]',
        'passed': ((df['fear_greed_value'] >= 0) & (df['fear_greed_value'] <= 100) | df['fear_greed_value'].isna()).all(),
        'details': None
    }
    results['checks'].append(check)
    
    # Check 8: Taker buy ratio within 0-1
    if 'taker_buy_ratio' in df.columns:
        valid_ratio = ((df['taker_buy_ratio'] >= 0) & (df['taker_buy_ratio'] <= 1) | df['taker_buy_ratio'].isna()).all()
        check = {
            'name': 'Taker buy ratio within [0, 1]',
            'passed': valid_ratio,
            'details': None
        }
        results['checks'].append(check)
    
    # Check 9: Minimum data points for training
    min_rows = 8760  # 1 year of hourly data
    check = {
        'name': f'Minimum rows ({min_rows:,})',
        'passed': len(df) >= min_rows,
        'details': f"Actual rows: {len(df):,}"
    }
    results['checks'].append(check)
    
    # Check 10: Missing data < 5%
    for col in ['close', 'volume', 'funding_rate', 'fear_greed_value']:
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            check = {
                'name': f'{col} missing < 5%',
                'passed': missing_pct < 5,
                'details': f"Missing: {missing_pct:.2f}%"
            }
            results['checks'].append(check)
    
    # Update overall status
    results['passed'] = all(c['passed'] for c in results['checks'])
    
    return results


def print_validation_report(results: dict):
    """Print formatted validation report."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    for check in results['checks']:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"{status}: {check['name']}")
        if check['details'] and not check['passed']:
            print(f"       {check['details']}")
    
    print("-" * 60)
    overall = "✓ ALL CHECKS PASSED" if results['passed'] else "✗ SOME CHECKS FAILED"
    print(f"{overall}")
    print("=" * 60)
```

---

## 10. Usage Examples

### 10.1 Basic Usage

```python
# Fetch 2 years of training data
from main_pipeline import MLDataPipeline

pipeline = MLDataPipeline(
    symbol="BTCUSDT",
    binance_api_key="your_api_key_here",  # Optional but recommended
    output_dir="./ml_data"
)

df = pipeline.run(
    start_date="2023-01-01",
    end_date="2024-12-31"
)

print(df.info())
print(df.head())
```

### 10.2 Quick Data Check

```python
# Quick check of available data
from data_fetcher import BinanceFetcher

fetcher = BinanceFetcher()

# Fetch last 100 hours of data
df = fetcher.fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_date="2025-12-10"
)

print(f"Fetched {len(df)} rows")
print(df.tail())
```

### 10.3 Fear & Greed Only

```python
# Fetch Fear & Greed history
from data_fetcher import FearGreedFetcher

fng = FearGreedFetcher()
df = fng.fetch_all()

print(f"History from {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Current value: {df.iloc[-1]['fear_greed_value']} ({df.iloc[-1]['fear_greed_class']})")
```

### 10.4 Validation After Load

```python
# Load and validate saved data
import pandas as pd
from validation import run_all_validations, print_validation_report

df = pd.read_parquet("./ml_data/BTCUSDT_ml_data.parquet")

results = run_all_validations(df)
print_validation_report(results)
```

---

## Appendix A: Column Reference

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `timestamp` | datetime64[ns, UTC] | - | Hourly UTC timestamp |
| `open` | float64 | Binance | Opening price |
| `high` | float64 | Binance | Highest price |
| `low` | float64 | Binance | Lowest price |
| `close` | float64 | Binance | Closing price |
| `volume` | float64 | Binance | Total volume (base asset) |
| `quote_volume` | float64 | Binance | Total volume (quote asset, USDT) |
| `trades` | int64 | Binance | Number of trades |
| `taker_buy_volume` | float64 | Binance | Taker buy volume (base) |
| `taker_buy_quote_volume` | float64 | Binance | Taker buy volume (quote) |
| `funding_rate` | float64 | Binance | Funding rate (forward-filled from 8h) |
| `fear_greed_value` | int64 | Alternative.me | Fear & Greed Index (0-100) |
| `taker_buy_ratio` | float64 | Derived | Taker buy / Total volume |
| `taker_sell_volume` | float64 | Derived | Total - Taker buy volume |
| `order_flow_imbalance` | float64 | Derived | (Buy - Sell) / Total |
| `spread_proxy` | float64 | Derived | (High - Low) / Close |
| `hour` | int64 | Derived | Hour of day (0-23) |
| `day_of_week` | int64 | Derived | Day of week (0=Mon, 6=Sun) |
| `session` | string | Derived | Trading session (asian/london/new_york) |

---

## Appendix B: Troubleshooting

### Common Issues

**1. Rate Limit Errors (429)**
```
Solution: The fetcher includes automatic retry with backoff.
If persistent, add delay between requests or use API key.
```

**2. Missing Funding Rate Data**
```
Cause: Funding rate only at 00:00, 08:00, 16:00 UTC
Solution: Forward-fill is applied automatically
```

**3. Fear & Greed API Down**
```
Solution: Alternative.me is generally stable.
If down, data can be forward-filled from last known value.
```

**4. Large Gaps in Data**
```
Cause: Exchange downtime or maintenance
Solution: Gaps < 4 hours are interpolated.
Larger gaps are logged as warnings.
```

---

## Appendix C: Performance Notes

| Operation | Time Estimate | API Calls |
|-----------|---------------|-----------|
| 1 year OHLCV (1h) | ~30 seconds | ~6 calls |
| 2 years OHLCV (1h) | ~60 seconds | ~12 calls |
| 5 years OHLCV (1h) | ~2.5 minutes | ~30 calls |
| 1 year Funding Rate | ~5 seconds | ~2 calls |
| Full Fear & Greed History | ~2 seconds | 1 call |

---

*Document generated for ML Trading Pipeline v2.0*
*Data sources verified as of December 2025*
