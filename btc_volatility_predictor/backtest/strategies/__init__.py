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

# V3 Strategies - Trend Filtered
from .trend_utils import detect_trend, get_trend_strength, TrendType
from .trend_filtered_mr import TrendFilteredMeanReversion
from .trend_defensive import TrendAdaptiveDefensive
from .trend_follower import TrendFollowerStrategy

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
    # V3 Strategies (trend filtered)
    'detect_trend',
    'get_trend_strength',
    'TrendType',
    'TrendFilteredMeanReversion',
    'TrendAdaptiveDefensive',
    'TrendFollowerStrategy',
]
