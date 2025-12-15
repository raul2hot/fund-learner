"""
Binance Futures data fetchers for Open Interest, Long/Short Ratio, and Taker Volume.

IMPORTANT: These endpoints only provide 30 days of historical data via API.
Use HybridDataFetcher with Kaggle data for full historical coverage.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .base import BinanceFetcher, register_fetcher
from .rate_limiter import RateLimiter


class BinanceFuturesDataFetcher(BinanceFetcher):
    """Base class for Binance Futures data endpoints with 30-day limit."""

    MAX_RECORDS_PER_REQUEST = 500
    API_HISTORICAL_LIMIT_DAYS = 30

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=1000)

    def get_rate_limit(self) -> Dict[str, Any]:
        return {
            "requests_per_5min": 1000,
            "weight_per_request": 1,
            "max_per_request": self.MAX_RECORDS_PER_REQUEST,
        }

    def get_api_historical_limit_days(self) -> int:
        return self.API_HISTORICAL_LIMIT_DAYS

    def _get_valid_date_range(self, start: str, end: str) -> tuple:
        """
        Validate and adjust date range to API limits.
        Returns (start_ms, end_ms, was_adjusted).
        """
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start_dt = datetime.strptime(start, "%Y-%m-%d")

        # API only provides last 30 days
        min_start = datetime.utcnow() - timedelta(days=self.API_HISTORICAL_LIMIT_DAYS)

        was_adjusted = False
        if start_dt < min_start:
            print(f"WARNING: API only provides {self.API_HISTORICAL_LIMIT_DAYS} days of history. "
                  f"Adjusting start from {start} to {min_start.strftime('%Y-%m-%d')}")
            start_dt = min_start
            was_adjusted = True

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        return start_ms, end_ms, was_adjusted


@register_fetcher('binance_open_interest')
class BinanceOpenInterestFetcher(BinanceFuturesDataFetcher):
    """
    Fetch Open Interest data from Binance Futures.

    Endpoint: GET https://fapi.binance.com/futures/data/openInterestHist
    API Historical Limit: 30 days
    """

    BASE_URL = "https://fapi.binance.com/futures/data/openInterestHist"

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT",
        period: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch open interest data.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)

        Returns:
            DataFrame with open interest data
        """
        start_ms, end_ms, _ = self._get_valid_date_range(start, end)

        all_data = []
        current_start = start_ms

        print(f"Fetching open interest for {symbol} (API limit: last 30 days)...")

        while current_start < end_ms:
            data = self._fetch_batch(symbol, period, current_start, end_ms)

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_timestamp = data[-1]['timestamp']
            current_start = last_timestamp + 1

            if len(data) < self.MAX_RECORDS_PER_REQUEST:
                break

        if not all_data:
            return pd.DataFrame()

        df = self._parse_response(all_data)
        print(f"Fetched {len(df)} open interest records")

        return df

    def _fetch_batch(
        self,
        symbol: str,
        period: str,
        start_ms: int,
        end_ms: int
    ) -> list:
        """Fetch a single batch."""
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.MAX_RECORDS_PER_REQUEST
        }

        def make_request():
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        return self.rate_limiter.execute_with_retry(make_request)

    def _parse_response(self, data: list) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        df = pd.DataFrame(data)

        # Convert types
        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Sort and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        return df.reset_index(drop=True)


@register_fetcher('binance_long_short')
class BinanceLongShortFetcher(BinanceFuturesDataFetcher):
    """
    Fetch Long/Short Ratio data from Binance Futures.

    Endpoint: GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio
    API Historical Limit: 30 days
    """

    BASE_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT",
        period: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch long/short ratio data.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol
            period: Time period

        Returns:
            DataFrame with long/short ratio data
        """
        start_ms, end_ms, _ = self._get_valid_date_range(start, end)

        all_data = []
        current_start = start_ms

        print(f"Fetching long/short ratio for {symbol} (API limit: last 30 days)...")

        while current_start < end_ms:
            data = self._fetch_batch(symbol, period, current_start, end_ms)

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_timestamp = data[-1]['timestamp']
            current_start = last_timestamp + 1

            if len(data) < self.MAX_RECORDS_PER_REQUEST:
                break

        if not all_data:
            return pd.DataFrame()

        df = self._parse_response(all_data)
        print(f"Fetched {len(df)} long/short ratio records")

        return df

    def _fetch_batch(
        self,
        symbol: str,
        period: str,
        start_ms: int,
        end_ms: int
    ) -> list:
        """Fetch a single batch."""
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.MAX_RECORDS_PER_REQUEST
        }

        def make_request():
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        return self.rate_limiter.execute_with_retry(make_request)

    def _parse_response(self, data: list) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        df = pd.DataFrame(data)

        # Convert types
        df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
        df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
        df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Sort and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        return df.reset_index(drop=True)


@register_fetcher('binance_taker_volume')
class BinanceTakerVolumeFetcher(BinanceFuturesDataFetcher):
    """
    Fetch Taker Buy/Sell Volume data from Binance Futures.

    Endpoint: GET https://fapi.binance.com/futures/data/takerlongshortRatio
    API Historical Limit: 30 days
    """

    BASE_URL = "https://fapi.binance.com/futures/data/takerlongshortRatio"

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT",
        period: str = "1h"
    ) -> pd.DataFrame:
        """
        Fetch taker buy/sell volume data.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol
            period: Time period

        Returns:
            DataFrame with taker volume data
        """
        start_ms, end_ms, _ = self._get_valid_date_range(start, end)

        all_data = []
        current_start = start_ms

        print(f"Fetching taker volume for {symbol} (API limit: last 30 days)...")

        while current_start < end_ms:
            data = self._fetch_batch(symbol, period, current_start, end_ms)

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_timestamp = data[-1]['timestamp']
            current_start = last_timestamp + 1

            if len(data) < self.MAX_RECORDS_PER_REQUEST:
                break

        if not all_data:
            return pd.DataFrame()

        df = self._parse_response(all_data)
        print(f"Fetched {len(df)} taker volume records")

        return df

    def _fetch_batch(
        self,
        symbol: str,
        period: str,
        start_ms: int,
        end_ms: int
    ) -> list:
        """Fetch a single batch."""
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.MAX_RECORDS_PER_REQUEST
        }

        def make_request():
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        return self.rate_limiter.execute_with_retry(make_request)

    def _parse_response(self, data: list) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        df = pd.DataFrame(data)

        # Convert types
        df['buySellRatio'] = pd.to_numeric(df['buySellRatio'], errors='coerce')
        df['buyVol'] = pd.to_numeric(df['buyVol'], errors='coerce')
        df['sellVol'] = pd.to_numeric(df['sellVol'], errors='coerce')

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Sort and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        return df.reset_index(drop=True)
