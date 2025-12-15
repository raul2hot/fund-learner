"""
Track when each feature becomes available.
This allows graceful degradation for older date ranges.
"""

from datetime import datetime
from typing import Dict, List


FEATURE_AVAILABILITY: Dict[str, str] = {
    # Core features (always available for our date range)
    'ohlcv': '2017-01-01',           # Binance spot launch
    'volume': '2017-01-01',

    # Sentiment features
    'fear_greed': '2018-02-01',       # Alternative.me launch

    # Futures-specific features (our primary data floor)
    'funding_rate': '2019-09-13',
    'open_interest': '2019-09-13',
    'long_short_ratio': '2020-01-01', # Slightly later availability
    'taker_volume': '2020-01-01',

    # Derived features (computed, always available if inputs exist)
    'volatility': 'derived',
    'trend_strength': 'derived',
    'momentum': 'derived',
    'kmeans_sr': 'derived',
    'hmm_regime': 'derived',
    'session_time': 'derived',
}


def get_available_features(start_date: str) -> List[str]:
    """
    Returns list of features available for a given start date.
    Allows model to adapt to available data.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")

    available = []
    for feature, avail_date in FEATURE_AVAILABILITY.items():
        if avail_date == 'derived':
            available.append(feature)
        elif datetime.strptime(avail_date, "%Y-%m-%d") <= start:
            available.append(feature)

    return available


def check_feature_coverage(start_date: str) -> dict:
    """
    Returns coverage report for transparency.
    """
    available = get_available_features(start_date)
    all_features = list(FEATURE_AVAILABILITY.keys())

    # Find missing non-derived features
    missing = []
    for f in all_features:
        if f not in available and FEATURE_AVAILABILITY[f] != 'derived':
            missing.append(f)

    return {
        'start_date': start_date,
        'available_features': available,
        'missing_features': missing,
        'coverage_pct': len(available) / len(all_features) * 100,
        'full_coverage': len(missing) == 0
    }
