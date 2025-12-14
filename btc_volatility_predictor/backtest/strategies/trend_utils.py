"""
Trend detection utilities for V3 strategies.

Trend Classification:
- UPTREND: Price > SMA_168 (7-day) AND SMA_168 > SMA_720 (30-day)
- DOWNTREND: Price < SMA_168 AND SMA_168 < SMA_720
- SIDEWAYS: Mixed conditions
"""

from typing import Literal

TrendType = Literal['UPTREND', 'DOWNTREND', 'SIDEWAYS']


def calculate_sma(history: list[dict], period: int, price_key: str = 'close') -> float:
    """Calculate Simple Moving Average from history."""
    if len(history) < period:
        return None

    prices = [h.get(price_key, 0) for h in history[-period:]]
    return sum(prices) / len(prices)


def calculate_ema(history: list[dict], period: int, price_key: str = 'close') -> float:
    """Calculate Exponential Moving Average from history."""
    if len(history) < period:
        return None

    prices = [h.get(price_key, 0) for h in history]
    multiplier = 2 / (period + 1)

    # Initialize with SMA
    ema = sum(prices[:period]) / period

    # Apply EMA formula
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def detect_trend(
    current_price: float,
    history: list[dict],
    fast_period: int = 168,   # 7 days (168 hours)
    slow_period: int = 720,   # 30 days (720 hours)
    threshold: float = 0.005  # 0.5% buffer for sideways
) -> TrendType:
    """
    Detect market trend based on price vs moving averages.

    Args:
        current_price: Current close price
        history: List of historical bars
        fast_period: Short-term MA period (default 7 days)
        slow_period: Long-term MA period (default 30 days)
        threshold: Percentage buffer for sideways detection

    Returns:
        'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
    """
    if len(history) < slow_period:
        return 'SIDEWAYS'  # Not enough data

    sma_fast = calculate_sma(history, fast_period)
    sma_slow = calculate_sma(history, slow_period)

    if sma_fast is None or sma_slow is None:
        return 'SIDEWAYS'

    # Conditions for uptrend
    price_above_fast = current_price > sma_fast * (1 + threshold)
    fast_above_slow = sma_fast > sma_slow * (1 + threshold)

    # Conditions for downtrend
    price_below_fast = current_price < sma_fast * (1 - threshold)
    fast_below_slow = sma_fast < sma_slow * (1 - threshold)

    if price_above_fast and fast_above_slow:
        return 'UPTREND'
    elif price_below_fast and fast_below_slow:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'


def get_trend_strength(
    current_price: float,
    history: list[dict],
    fast_period: int = 168,
    slow_period: int = 720
) -> float:
    """
    Calculate trend strength (-1 to +1).

    Returns:
        -1.0 = Strong downtrend
         0.0 = Sideways
        +1.0 = Strong uptrend
    """
    if len(history) < slow_period:
        return 0.0

    sma_fast = calculate_sma(history, fast_period)
    sma_slow = calculate_sma(history, slow_period)

    if sma_fast is None or sma_slow is None:
        return 0.0

    # Price deviation from fast MA (normalized)
    price_dev = (current_price - sma_fast) / sma_fast

    # Fast MA deviation from slow MA (normalized)
    ma_dev = (sma_fast - sma_slow) / sma_slow

    # Combined strength (clamped to -1, +1)
    strength = (price_dev + ma_dev) / 2
    return max(-1.0, min(1.0, strength * 10))  # Scale up for sensitivity
