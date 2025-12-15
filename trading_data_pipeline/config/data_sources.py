"""
DataSource Configuration

Defines configuration for each data source including frequency,
timestamp handling, publication delays, and resampling methods.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import pandas as pd


class ResampleMethod(Enum):
    """Methods for resampling lower-frequency data to base frequency."""
    FORWARD_FILL = "ffill"
    FORWARD_FILL_LIMIT = "ffill_limit"
    AGGREGATE_LAST = "last"
    AGGREGATE_MEAN = "mean"
    POINT_IN_TIME = "pit"


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    frequency: str                    # '1h', '8h', '1D', '5min'
    timestamp_col: str                # Column name for timestamp
    timestamp_format: str             # 'unix_ms', 'unix_s', 'iso', 'date'
    timezone: str                     # Source timezone
    publication_delay: pd.Timedelta   # Time until data is available
    resample_method: ResampleMethod
    fill_limit_hours: Optional[int] = None
    value_columns: Optional[List[str]] = None
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


# All sources as a dict for easy access
ALL_SOURCES = {
    'ohlcv': OHLCV_SOURCE,
    'funding_rate': FUNDING_RATE_SOURCE,
    'fear_greed': FEAR_GREED_SOURCE,
    'open_interest': OPEN_INTEREST_SOURCE,
    'long_short_ratio': LONG_SHORT_RATIO_SOURCE,
    'taker_volume': TAKER_VOLUME_SOURCE,
}
