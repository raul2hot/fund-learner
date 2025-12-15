"""
Binance Futures OHLCV data fetcher.

Primary data source - base timeline.
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from .base import BinanceFetcher, register_fetcher
from .rate_limiter import RateLimiter


@register_fetcher('binance_ohlcv')
class BinanceOHLCVFetcher(BinanceFetcher):
    """
    Fetch OHLCV data from Binance Futures.

    Endpoint: GET https://fapi.binance.com/fapi/v1/klines
    Max per request: 1500 candles
    """

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    MAX_CANDLES_PER_REQUEST = 1500

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=2400)

    def get_rate_limit(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 2400,
            "weight_per_request": 5,
            "max_per_request": self.MAX_CANDLES_PER_REQUEST,
        }

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for date range.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol
            interval: Candle interval

        Returns:
            DataFrame with OHLCV columns
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        print(f"Fetching OHLCV for {symbol} from {start} to {end}...")

        while current_start < end_ms:
            data = self._fetch_batch(symbol, interval, current_start, end_ms)

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_timestamp = data[-1][0]
            current_start = last_timestamp + 1

            if len(data) < self.MAX_CANDLES_PER_REQUEST:
                break

        if not all_data:
            return pd.DataFrame()

        df = self._parse_response(all_data)
        print(f"Fetched {len(df)} OHLCV candles")

        return df

    def _fetch_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int
    ) -> list:
        """Fetch a single batch of candles."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.MAX_CANDLES_PER_REQUEST
        }

        def make_request():
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        return self.rate_limiter.execute_with_retry(make_request)

    def _parse_response(self, data: list) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]

        df = pd.DataFrame(data, columns=columns)

        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['trades'] = df['trades'].astype(int)

        # Drop unnecessary columns
        df = df.drop(['close_time', 'ignore'], axis=1)

        # Convert timestamp
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

        # Sort and remove duplicates
        df = df.sort_values('open_time').drop_duplicates('open_time')

        return df.reset_index(drop=True)
