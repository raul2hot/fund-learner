"""
Hybrid fetcher that combines Kaggle historical data with Binance API recent data.

Used for Open Interest, Long/Short Ratio, and Taker Volume which have
only 30 days of API history but full history available via Kaggle.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings

from .kaggle_loader import KaggleLoader
from .binance_futures import (
    BinanceOpenInterestFetcher,
    BinanceLongShortFetcher,
    BinanceTakerVolumeFetcher,
)
from .rate_limiter import RateLimiter


class HybridDataFetcher:
    """
    Fetches data using hybrid strategy:
    1. Load historical data from Kaggle CSV
    2. Fetch recent data from Binance API (last 30 days)
    3. Merge and deduplicate

    Architecture:
    ┌──────────────────┐    ┌──────────────┐    ┌──────────────────────┐
    │   KAGGLE CSV     │    │  GAP FILLER  │    │   BINANCE API        │
    │   (Historical)   │ -> │  (if needed) │ -> │   (Last 30 days)     │
    └──────────────────┘    └──────────────┘    └──────────────────────┘
    """

    DATA_TYPES = {
        'open_interest': {
            'fetcher_class': BinanceOpenInterestFetcher,
            'value_cols': ['sumOpenInterest', 'sumOpenInterestValue'],
        },
        'long_short_ratio': {
            'fetcher_class': BinanceLongShortFetcher,
            'value_cols': ['longShortRatio', 'longAccount', 'shortAccount'],
        },
        'taker_volume': {
            'fetcher_class': BinanceTakerVolumeFetcher,
            'value_cols': ['buySellRatio', 'buyVol', 'sellVol'],
        },
    }

    def __init__(
        self,
        kaggle_data_dir: str = "data/kaggle",
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize HybridDataFetcher.

        Args:
            kaggle_data_dir: Directory containing downloaded Kaggle CSVs
            rate_limiter: Optional rate limiter for API requests
        """
        self.kaggle_loader = KaggleLoader(kaggle_data_dir)
        self.rate_limiter = rate_limiter

    def fetch(
        self,
        data_type: str,
        start: str,
        end: str,
        symbol: str = "BTCUSDT"
    ) -> pd.DataFrame:
        """
        Fetch data using hybrid strategy.

        Args:
            data_type: 'open_interest', 'long_short_ratio', or 'taker_volume'
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Trading symbol

        Returns:
            DataFrame with complete data coverage
        """
        if data_type not in self.DATA_TYPES:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available: {list(self.DATA_TYPES.keys())}"
            )

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        config = self.DATA_TYPES[data_type]

        print(f"\n{'='*50}")
        print(f"HYBRID FETCH: {data_type}")
        print(f"{'='*50}")

        # Step 1: Load Kaggle historical data
        try:
            kaggle_df = self.kaggle_loader.load(data_type)
            kaggle_available = True
            kaggle_end = kaggle_df['timestamp'].max()
            print(f"Kaggle data: {len(kaggle_df)} rows, ends at {kaggle_end}")
        except FileNotFoundError as e:
            warnings.warn(str(e))
            kaggle_df = pd.DataFrame()
            kaggle_available = False
            kaggle_end = None
            print("Kaggle data: NOT AVAILABLE")

        # Step 2: Determine API date range (last 30 days)
        api_start_dt = end_dt - timedelta(days=29)
        api_start = api_start_dt.strftime("%Y-%m-%d")

        # Step 3: Fetch from Binance API
        fetcher = config['fetcher_class'](rate_limiter=self.rate_limiter)
        api_df = fetcher.fetch(api_start, end, symbol)

        if not api_df.empty:
            api_df['data_source'] = 'api'
            print(f"API data: {len(api_df)} rows")
        else:
            print("API data: EMPTY")

        # Step 4: Merge data
        merged_df = self._merge_data(kaggle_df, api_df, config['value_cols'])

        # Step 5: Filter to requested range
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')
        merged_df = merged_df[
            (merged_df['timestamp'] >= start_ts) &
            (merged_df['timestamp'] <= end_ts)
        ]

        # Step 6: Check for gaps and warn
        if kaggle_available and not api_df.empty:
            self._check_gaps(merged_df, kaggle_end, api_start_dt, data_type)

        print(f"Final merged data: {len(merged_df)} rows")
        print(f"{'='*50}\n")

        return merged_df.reset_index(drop=True)

    def _merge_data(
        self,
        kaggle_df: pd.DataFrame,
        api_df: pd.DataFrame,
        value_cols: list
    ) -> pd.DataFrame:
        """
        Merge Kaggle and API data, preferring API for overlapping periods.
        """
        if kaggle_df.empty and api_df.empty:
            return pd.DataFrame(columns=['timestamp', 'data_source'] + value_cols)

        if kaggle_df.empty:
            return api_df

        if api_df.empty:
            return kaggle_df

        # Combine
        combined = pd.concat([kaggle_df, api_df], ignore_index=True)

        # Sort by timestamp
        combined = combined.sort_values('timestamp')

        # Remove duplicates, keeping API data (last) for overlapping periods
        combined = combined.drop_duplicates(subset='timestamp', keep='last')

        return combined

    def _check_gaps(
        self,
        df: pd.DataFrame,
        kaggle_end: pd.Timestamp,
        api_start: datetime,
        data_type: str
    ):
        """Warn if there's a gap between Kaggle and API data."""
        api_start_ts = pd.Timestamp(api_start, tz='UTC')
        gap_hours = (api_start_ts - kaggle_end).total_seconds() / 3600

        if gap_hours > 24:  # More than 1 day gap
            gap_days = gap_hours / 24
            warnings.warn(
                f"\nDATA GAP DETECTED for {data_type}:\n"
                f"   Kaggle data ends: {kaggle_end.strftime('%Y-%m-%d %H:%M')}\n"
                f"   API data starts:  {api_start_ts.strftime('%Y-%m-%d %H:%M')}\n"
                f"   Gap: {gap_days:.1f} days ({gap_hours:.0f} hours)\n"
                f"   \n"
                f"   Options:\n"
                f"   1. Update Kaggle CSV from: github.com/jesusgraterol/binance-futures-dataset-builder\n"
                f"   2. Gap will be marked as missing (current behavior)\n"
            )

            # Mark gap in data
            gap_mask = (df['timestamp'] > kaggle_end) & (df['timestamp'] < api_start_ts)
            gap_count = gap_mask.sum()
            if gap_count > 0:
                print(f"   Gap contains {gap_count} expected data points (marked as missing)")

    def get_gap_report(self, end_date: str = None) -> Dict[str, Any]:
        """
        Generate a report of data gaps for all data types.

        Args:
            end_date: End date to check against (default: today)

        Returns:
            Dict with gap information for each data type
        """
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        api_start = end_dt - timedelta(days=29)

        report = {}

        for data_type in self.DATA_TYPES:
            try:
                kaggle_end = self.kaggle_loader.get_data_end_date(data_type)
                if kaggle_end:
                    gap_days = (api_start - kaggle_end.to_pydatetime().replace(tzinfo=None)).days
                    report[data_type] = {
                        'kaggle_end': str(kaggle_end),
                        'api_start': str(api_start),
                        'gap_days': max(0, gap_days),
                        'status': 'OK' if gap_days <= 1 else 'GAP',
                    }
                else:
                    report[data_type] = {
                        'kaggle_end': None,
                        'status': 'NO_KAGGLE_DATA',
                    }
            except Exception as e:
                report[data_type] = {
                    'status': 'ERROR',
                    'error': str(e),
                }

        return report
