"""
Binance Futures Funding Rate fetcher.

8-hour frequency, critical for sentiment.
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from .base import BinanceFetcher, register_fetcher
from .rate_limiter import RateLimiter


@register_fetcher('binance_funding')
class BinanceFundingFetcher(BinanceFetcher):
    """
    Fetch funding rate data from Binance Futures.

    Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate
    Frequency: Every 8 hours (00:00, 08:00, 16:00 UTC)
    Max per request: 1000 records
    Full history available.
    """

    BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
    MAX_RECORDS_PER_REQUEST = 1000

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=500)

    def get_rate_limit(self) -> Dict[str, Any]:
        return {
            "requests_per_5min": 500,
            "weight_per_request": 1,
            "max_per_request": self.MAX_RECORDS_PER_REQUEST,
        }

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT"
    ) -> pd.DataFrame:
        """
        Fetch funding rate data for date range.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol

        Returns:
            DataFrame with funding rate data
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        print(f"Fetching funding rates for {symbol} from {start} to {end}...")

        while current_start < end_ms:
            data = self._fetch_batch(symbol, current_start, end_ms)

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_timestamp = data[-1]['fundingTime']
            current_start = last_timestamp + 1

            if len(data) < self.MAX_RECORDS_PER_REQUEST:
                break

        if not all_data:
            return pd.DataFrame()

        df = self._parse_response(all_data)
        print(f"Fetched {len(df)} funding rate records")

        return df

    def _fetch_batch(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int
    ) -> list:
        """Fetch a single batch of funding rates."""
        params = {
            "symbol": symbol,
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
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')

        # Convert timestamp
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)

        # Rename for consistency
        df = df.rename(columns={'fundingTime': 'timestamp'})

        # Sort and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        return df.reset_index(drop=True)
