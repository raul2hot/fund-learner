"""
ML Data Pipeline for Crypto Trading
Version: 2.0

Modules:
- data_fetcher: Fetches data from Binance and Alternative.me APIs
- data_processor: Aligns and preprocesses multi-source data
- edge_cases: Handles data quality issues and validation
- main_pipeline: Complete pipeline orchestration with CLI
"""

from .data_fetcher import BinanceFetcher, FearGreedFetcher, CoinGeckoFetcher
from .data_processor import DataProcessor
from .edge_cases import EdgeCaseHandler, run_all_validations, print_validation_report
from .main_pipeline import MLDataPipeline

__version__ = "2.0.0"
__all__ = [
    "BinanceFetcher",
    "FearGreedFetcher",
    "CoinGeckoFetcher",
    "DataProcessor",
    "EdgeCaseHandler",
    "MLDataPipeline",
    "run_all_validations",
    "print_validation_report",
]
