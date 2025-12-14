"""
Machine Learning module for trade quality prediction.

This module provides:
- TradeQualityFilter: XGBoost-based filter to predict WIN/LOSS trades
- generate_trade_labels: Generate labeled data from historical backtests
- train_xgboost: Train the XGBoost classifier
"""

from .trade_filter import TradeQualityFilter, get_trade_filter
from .feature_engineering import compute_additional_features, CORE_FEATURES, TRADE_FEATURES

__all__ = [
    'TradeQualityFilter',
    'get_trade_filter',
    'compute_additional_features',
    'CORE_FEATURES',
    'TRADE_FEATURES',
]
