"""
data_fetcher.py - Clean Data Fetching for ML Trading Pipeline
Version: 2.0
Date: December 2025

Data Sources:
- Binance Futures API: OHLCV, Taker Volume, Funding Rate
- Alternative.me API: Fear & Greed Index
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from typing import Optional, List
from tqdm import tqdm
import logging

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
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 418:
                    logger.error("IP banned. Waiting 5 minutes...")
                    time.sleep(300)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
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
        symbol : str - Trading pair (e.g., "BTCUSDT")
        interval : str - Candle interval (e.g., "1h", "4h", "1d")
        start_date : str - Start date in "YYYY-MM-DD" format
        end_date : str - End date in "YYYY-MM-DD" format (default: now)
        
        Returns:
        --------
        pd.DataFrame with columns:
            timestamp, open, high, low, close, volume,
            quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
        """
        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        limit = 1500
        
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
                
                last_close_time = klines[-1][6]
                current_start = last_close_time + 1
                
                time.sleep(0.1)
        
        if not all_klines:
            logger.warning(f"No klines data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['trades'] = df['trades'].astype(int)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume']]
        
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
        pd.DataFrame with columns: timestamp, funding_rate, mark_price
        """
        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_funding = []
        current_start = start_ts
        limit = 1000
        
        logger.info(f"Fetching funding rate for {symbol}...")
        
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
            
            last_time = funding[-1]['fundingTime']
            current_start = last_time + 1
            
            time.sleep(0.2)
        
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
            timestamp (datetime64[ns, UTC] normalized to 00:00 UTC)
            fear_greed_value (int 0-100)
            fear_greed_class (str classification label)
        """
        params = {'limit': 0, 'format': 'json'}
        
        logger.info("Fetching Fear & Greed Index history...")
        
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
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
        df['timestamp'] = df['timestamp'].dt.normalize()
        
        df['fear_greed_value'] = pd.to_numeric(df['value'], errors='coerce').astype(int)
        df['fear_greed_class'] = df['value_classification']
        
        df = df[['timestamp', 'fear_greed_value', 'fear_greed_class']]
        df = df.drop_duplicates(subset='timestamp', keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} Fear & Greed records from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        
        return df


if __name__ == "__main__":
    # Test fetchers
    binance = BinanceFetcher()
    fng = FearGreedFetcher()
    
    # Quick test
    print("Testing Binance Klines...")
    df = binance.fetch_klines(symbol="BTCUSDT", start_date="2025-12-01", end_date="2025-12-10")
    print(f"Klines shape: {df.shape}")
    print(df.head())
    
    print("\nTesting Fear & Greed...")
    fng_df = fng.fetch_all()
    print(f"FnG shape: {fng_df.shape}")
    print(fng_df.tail())
