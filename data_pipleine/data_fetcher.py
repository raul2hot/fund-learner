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


class DemoDataGenerator:
    """
    Generates synthetic demo data for testing when APIs are unavailable.
    """

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        logger.info(f"Generating demo data for {symbol}...")

        start = pd.Timestamp(start_date, tz='UTC')
        if end_date:
            end = pd.Timestamp(end_date, tz='UTC')
        else:
            end = pd.Timestamp(datetime.now(timezone.utc))

        # Generate hourly timestamps
        timestamps = pd.date_range(start=start, end=end, freq='1h', tz='UTC')

        # Generate realistic-looking price data
        np.random.seed(42)
        base_price = 50000  # BTC-like base price
        returns = np.random.normal(0, 0.005, len(timestamps))  # 0.5% hourly volatility
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLCV data
        open_prices = np.roll(prices, 1)
        open_prices[0] = prices[0]

        high_mult = 1 + np.abs(np.random.normal(0, 0.002, len(timestamps)))
        low_mult = 1 - np.abs(np.random.normal(0, 0.002, len(timestamps)))

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'close': prices,
            'volume': np.random.exponential(1000, len(timestamps)),
        })

        # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
        df['high'] = np.maximum(df['open'], df['close']) * high_mult
        df['low'] = np.minimum(df['open'], df['close']) * low_mult
        df['quote_volume'] = df['volume'] * df['close']
        df['trades'] = np.random.poisson(500, len(timestamps))
        df['taker_buy_volume'] = df['volume'] * np.random.uniform(0.4, 0.6, len(timestamps))
        df['taker_buy_quote_volume'] = df['taker_buy_volume'] * df['close']

        logger.info(f"Generated {len(df)} demo data rows")
        return df

    def fetch_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic funding rate data."""
        start = pd.Timestamp(start_date, tz='UTC')
        if end_date:
            end = pd.Timestamp(end_date, tz='UTC')
        else:
            end = pd.Timestamp(datetime.now(timezone.utc))

        # Generate 8-hourly timestamps
        timestamps = pd.date_range(start=start, end=end, freq='8h', tz='UTC')

        np.random.seed(43)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'funding_rate': np.random.normal(0.0001, 0.0002, len(timestamps)),
            'mark_price': 50000 * np.cumprod(1 + np.random.normal(0, 0.001, len(timestamps)))
        })

        logger.info(f"Generated {len(df)} demo funding rate rows")
        return df


class CoinGeckoFetcher:
    """
    Fetches OHLCV data from CoinGecko API (free, no API key required).

    Use as fallback when Binance is geo-blocked.
    Note: Limited to OHLC data, no taker volume or funding rate.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Map common trading pairs to CoinGecko coin IDs
    COIN_MAP = {
        'BTCUSDT': 'bitcoin',
        'ETHUSDT': 'ethereum',
        'BNBUSDT': 'binancecoin',
        'SOLUSDT': 'solana',
        'XRPUSDT': 'ripple',
        'ADAUSDT': 'cardano',
        'DOGEUSDT': 'dogecoin',
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.

        Note: CoinGecko free tier has limited historical data.
        For extensive history, Binance API is recommended.
        """
        coin_id = self.COIN_MAP.get(symbol.upper(), 'bitcoin')
        vs_currency = 'usd'

        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp())
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp())
        else:
            end_ts = int(datetime.now(timezone.utc).timestamp())

        # CoinGecko market_chart/range endpoint
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': vs_currency,
            'from': start_ts,
            'to': end_ts
        }

        logger.info(f"Fetching {symbol} data from CoinGecko...")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko request failed: {e}")
            return pd.DataFrame()

        if 'prices' not in data:
            logger.warning("No price data in CoinGecko response")
            return pd.DataFrame()

        # Convert to DataFrame
        prices = pd.DataFrame(data['prices'], columns=['timestamp_ms', 'close'])
        volumes = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp_ms', 'volume'])

        df = prices.merge(volumes, on='timestamp_ms', how='left')
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)

        # CoinGecko doesn't provide OHLC breakdown, so we approximate
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df['close'] * 1.001  # Approximate
        df['low'] = df['close'] * 0.999   # Approximate
        df['volume'] = df['volume'].fillna(0)

        # Add placeholder columns for compatibility
        df['quote_volume'] = df['volume']
        df['trades'] = 0
        df['taker_buy_volume'] = df['volume'] * 0.5  # Approximate as 50%
        df['taker_buy_quote_volume'] = df['quote_volume'] * 0.5

        # Resample to hourly if needed
        df = df.set_index('timestamp')
        df = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
            'trades': 'sum',
            'taker_buy_volume': 'sum',
            'taker_buy_quote_volume': 'sum'
        }).dropna()

        df = df.reset_index()
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Fetched {len(df)} rows from CoinGecko")
        return df


class BinanceFetcher:
    """
    Fetches OHLCV and Funding Rate data from Binance API.

    Handles:
    - Pagination for large date ranges
    - Rate limiting with automatic backoff
    - Timestamp alignment to hourly boundaries
    - Proper error handling and retries
    - Automatic fallback from Futures to Spot API if geo-blocked
    """

    FUTURES_URL = "https://fapi.binance.com"
    SPOT_URL = "https://api.binance.com"

    def __init__(self, api_key: Optional[str] = None, use_spot: bool = False):
        self.api_key = api_key
        self.use_spot = use_spot
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-MBX-APIKEY': api_key})

    @property
    def BASE_URL(self):
        return self.SPOT_URL if self.use_spot else self.FUTURES_URL
    
    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> list:
        """Make API request with retry logic and rate limit handling."""
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 403:
                    # Geo-blocked - try spot API if not already using it
                    if not self.use_spot:
                        logger.warning("Futures API blocked (403). Switching to Spot API...")
                        self.use_spot = True
                        url = f"{self.BASE_URL}{endpoint}"
                        # Adjust endpoint for spot API
                        if '/fapi/v1/klines' in endpoint:
                            url = f"{self.SPOT_URL}/api/v3/klines"
                        continue
                    raise requests.exceptions.HTTPError(f"403 Forbidden: {url}")

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
    
    def _get_klines_endpoint(self) -> str:
        """Get the correct klines endpoint based on API type."""
        return '/api/v3/klines' if self.use_spot else '/fapi/v1/klines'

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
        limit = 1000 if self.use_spot else 1500  # Spot API max is 1000

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

                endpoint = self._get_klines_endpoint()
                klines = self._make_request(endpoint, params)

                if not klines:
                    # Check if we switched to spot API during the request
                    if self.use_spot and limit == 1500:
                        limit = 1000
                        continue
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
        Note: Only available on Futures API. Returns empty DataFrame if using Spot API.

        Returns:
        --------
        pd.DataFrame with columns: timestamp, funding_rate, mark_price
        """
        # Funding rate is only available on Futures API
        if self.use_spot:
            logger.warning("Funding rate not available on Spot API. Returning empty DataFrame.")
            return pd.DataFrame(columns=['timestamp', 'funding_rate', 'mark_price'])

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

            try:
                funding = self._make_request('/fapi/v1/fundingRate', params)
            except requests.exceptions.HTTPError as e:
                if '403' in str(e):
                    logger.warning("Funding rate API blocked. Returning empty DataFrame.")
                    return pd.DataFrame(columns=['timestamp', 'funding_rate', 'mark_price'])
                raise

            if not funding:
                break

            all_funding.extend(funding)

            last_time = funding[-1]['fundingTime']
            current_start = last_time + 1

            time.sleep(0.2)

        if not all_funding:
            logger.warning(f"No funding rate data returned for {symbol}")
            return pd.DataFrame(columns=['timestamp', 'funding_rate', 'mark_price'])
        
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
