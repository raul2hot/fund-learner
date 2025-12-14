"""Trading strategies for volatility regime backtesting."""

from .base import BaseStrategy, Position, Signal
from .baseline import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .momentum import MomentumStrategy
from .regime_switch import RegimeSwitchStrategy

__all__ = [
    'BaseStrategy',
    'Position',
    'Signal',
    'BuyAndHoldStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'MomentumStrategy',
    'RegimeSwitchStrategy',
]
