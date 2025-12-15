"""
Base fetcher interface and factory.

All data fetchers implement this interface, enabling easy source swapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class DataFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    This allows easy swapping of sources.
    """

    @abstractmethod
    def fetch(self, start: str, end: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Fetch data for date range.

        Args:
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            symbol: Trading symbol (default BTCUSDT)

        Returns:
            DataFrame with standardized columns
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return source identifier."""
        pass

    @abstractmethod
    def get_rate_limit(self) -> Dict[str, Any]:
        """Return rate limit info."""
        pass

    def get_api_historical_limit_days(self) -> Optional[int]:
        """Return the API historical limit in days if applicable."""
        return None


class BinanceFetcher(DataFetcher):
    """Base class for all Binance fetchers."""

    def get_source_name(self) -> str:
        return "binance"

    def get_rate_limit(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": 2400,
            "weight_per_request": 1,
        }


# Registry of available fetchers
_FETCHER_REGISTRY: Dict[str, type] = {}


def register_fetcher(name: str):
    """Decorator to register a fetcher class."""
    def decorator(cls):
        _FETCHER_REGISTRY[name] = cls
        return cls
    return decorator


def get_fetcher(source: str, **kwargs) -> DataFetcher:
    """
    Factory to create fetchers.
    Easy to add new sources without changing existing code.

    Args:
        source: Name of the data source
        **kwargs: Arguments to pass to the fetcher constructor

    Returns:
        Initialized DataFetcher instance
    """
    if source not in _FETCHER_REGISTRY:
        raise ValueError(
            f"Unknown source: {source}. Available: {list(_FETCHER_REGISTRY.keys())}"
        )

    return _FETCHER_REGISTRY[source](**kwargs)
