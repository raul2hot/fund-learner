"""Configuration module for trading data pipeline."""

from .settings import (
    TimeConfig,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    ROLLING_WINDOW_YEARS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from .data_sources import (
    DataSource,
    ResampleMethod,
    OHLCV_SOURCE,
    FUNDING_RATE_SOURCE,
    FEAR_GREED_SOURCE,
    OPEN_INTEREST_SOURCE,
    LONG_SHORT_RATIO_SOURCE,
    TAKER_VOLUME_SOURCE,
)
from .feature_availability import (
    FEATURE_AVAILABILITY,
    get_available_features,
    check_feature_coverage,
)

__all__ = [
    'TimeConfig',
    'DEFAULT_START_DATE',
    'DEFAULT_END_DATE',
    'DEFAULT_SYMBOL',
    'DEFAULT_TIMEFRAME',
    'ROLLING_WINDOW_YEARS',
    'TRAIN_RATIO',
    'VAL_RATIO',
    'TEST_RATIO',
    'DataSource',
    'ResampleMethod',
    'OHLCV_SOURCE',
    'FUNDING_RATE_SOURCE',
    'FEAR_GREED_SOURCE',
    'OPEN_INTEREST_SOURCE',
    'LONG_SHORT_RATIO_SOURCE',
    'TAKER_VOLUME_SOURCE',
    'FEATURE_AVAILABILITY',
    'get_available_features',
    'check_feature_coverage',
]
