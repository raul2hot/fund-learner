"""Trading strategies for volatility regime backtesting."""

from .base import BaseStrategy, Position, Signal
from .baseline import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .momentum import MomentumStrategy
from .regime_switch import RegimeSwitchStrategy

# V2 Strategies - based on proven approaches
from .mean_reversion_v2 import MeanReversionV2Strategy
from .defensive import DefensiveStrategy, DefensiveWithTPSL
from .direction_aware import DirectionAwareMeanReversion, DirectionOnlyStrategy

__all__ = [
    # Base
    'BaseStrategy',
    'Position',
    'Signal',
    # V1 Strategies
    'BuyAndHoldStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'MomentumStrategy',
    'RegimeSwitchStrategy',
    # V2 Strategies (improved)
    'MeanReversionV2Strategy',
    'DefensiveStrategy',
    'DefensiveWithTPSL',
    'DirectionAwareMeanReversion',
    'DirectionOnlyStrategy',
]
