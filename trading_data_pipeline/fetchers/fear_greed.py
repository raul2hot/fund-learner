"""
Fear & Greed Index fetcher from Alternative.me.

Daily frequency, sentiment indicator.
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from .base import DataFetcher, register_fetcher
from .rate_limiter import RateLimiter


@register_fetcher('fear_greed')
class FearGreedFetcher(DataFetcher):
    """
    Fetch Fear & Greed Index from Alternative.me.

    Endpoint: GET https://api.alternative.me/fng/
    Frequency: Daily (updated ~09:00 UTC)
    Publication Delay: ~9 hours (dated for "today" but published ~09:00)

    CRITICAL: Data dated "2024-01-15" is NOT available at 00:00 on Jan 15!
    It becomes available around 09:00 UTC.
    """

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        # Be respectful - ~1 req/sec
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=60)

    def get_source_name(self) -> str:
        return "alternative_me"

    def get_rate_limit(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 60,
            "weight_per_request": 1,
            "note": "Be respectful, ~1 req/sec",
        }

    def fetch(
        self,
        start: str,
        end: str,
        symbol: str = "BTCUSDT"  # Symbol ignored for this endpoint
    ) -> pd.DataFrame:
        """
        Fetch Fear & Greed index data.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Ignored (F&G is for crypto market overall)

        Returns:
            DataFrame with Fear & Greed data
        """
        print(f"Fetching Fear & Greed index from {start} to {end}...")

        # Fetch all historical data (API returns most recent first)
        data = self._fetch_all_history()

        if not data:
            return pd.DataFrame()

        df = self._parse_response(data)

        # Filter to date range
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')

        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

        print(f"Fetched {len(df)} Fear & Greed records")

        return df.reset_index(drop=True)

    def _fetch_all_history(self) -> list:
        """Fetch all historical Fear & Greed data."""
        params = {"limit": 0}  # 0 = all history

        def make_request():
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        result = self.rate_limiter.execute_with_retry(make_request)
        return result.get('data', [])

    def _parse_response(self, data: list) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        df = pd.DataFrame(data)

        # Convert timestamp (unix seconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)

        # Convert value to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Rename for consistency
        df = df.rename(columns={'value_classification': 'classification'})

        # Drop time_until_update if present (only for latest)
        if 'time_until_update' in df.columns:
            df = df.drop('time_until_update', axis=1)

        # Sort by timestamp (ascending)
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        return df
