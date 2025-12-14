"""Backtesting framework for BTC volatility regime strategies."""

from .engine import BacktestEngine, BacktestResult, Trade
from .generate_predictions import generate_test_predictions

__all__ = ['BacktestEngine', 'BacktestResult', 'Trade', 'generate_test_predictions']
