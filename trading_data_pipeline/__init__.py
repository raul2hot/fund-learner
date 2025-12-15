"""
Trading Data Pipeline

A robust, point-in-time correct data alignment system for trading ML models.

Key Features:
- Multi-source data fetching (Binance, Fear & Greed, Kaggle)
- Hybrid data strategy for 30-day limited APIs
- Point-in-time alignment to prevent look-ahead bias
- 10 filter categories of engineered features
- 5-class candle labeling
- Walk-forward train/val/test splitting

Usage:
    from trading_data_pipeline import PipelineOrchestrator

    pipeline = PipelineOrchestrator(
        symbol="BTCUSDT",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
    df = pipeline.run()
"""

from .config import (
    TimeConfig,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SYMBOL,
    DataSource,
)
from .alignment import PointInTimeDatabase
from .features import FeatureEngineer
from .labeling import CandleLabeler
from .pipeline import PipelineOrchestrator, DataSplitter

__version__ = "1.0.0"

__all__ = [
    'TimeConfig',
    'DEFAULT_START_DATE',
    'DEFAULT_END_DATE',
    'DEFAULT_SYMBOL',
    'DataSource',
    'PointInTimeDatabase',
    'FeatureEngineer',
    'CandleLabeler',
    'PipelineOrchestrator',
    'DataSplitter',
]
