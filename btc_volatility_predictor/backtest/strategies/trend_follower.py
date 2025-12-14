"""
Pure Trend Following Strategy (for comparison).

This strategy ignores volatility prediction and uses only trend.
Useful for comparing: Is vol filtering adding value on top of trend?
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, TrendType


class TrendFollowerStrategy(BaseStrategy):
    """
    Pure trend following without volatility filter.

    - UPTREND: Hold long
    - DOWNTREND: Hold short or flat
    - SIDEWAYS: Flat

    Comparison baseline for trend + vol strategies.
    """

    def __init__(
        self,
        fast_ma_period: int = 168,
        slow_ma_period: int = 720,
        allow_short: bool = False,
    ):
        super().__init__(name="TrendFollower")
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.allow_short = allow_short
        self._current_trend: TrendType = 'SIDEWAYS'
        self._prev_trend: Optional[TrendType] = None

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,  # Ignored
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)

        if len(history) < self.slow_ma_period:
            return 'HOLD'

        self._prev_trend = self._current_trend
        self._current_trend = detect_trend(
            close, history,
            self.fast_ma_period,
            self.slow_ma_period
        )

        # Exit on trend change
        if self.has_position():
            direction = self.position.direction

            if direction == 'LONG' and self._current_trend != 'UPTREND':
                return 'CLOSE'
            if direction == 'SHORT' and self._current_trend != 'DOWNTREND':
                return 'CLOSE'

            return 'HOLD'

        # Entry
        if self._current_trend == 'UPTREND':
            return 'BUY'

        if self._current_trend == 'DOWNTREND' and self.allow_short:
            return 'SELL'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Pure Trend Follower',
            'fast_ma': self.fast_ma_period,
            'slow_ma': self.slow_ma_period,
            'allow_short': self.allow_short,
            'vol_filter': 'NONE'
        }

    def reset(self):
        super().reset()
        self._current_trend = 'SIDEWAYS'
        self._prev_trend = None
