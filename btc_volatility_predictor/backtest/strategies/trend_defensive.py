"""
Trend-Adaptive Defensive Strategy.

Improvement over V2 Defensive:
1. In UPTREND: Hold long during LOW vol, exit during HIGH vol (original defensive)
2. In DOWNTREND: Stay flat entirely OR go short during LOW vol
3. In SIDEWAYS: Reduced position size or flat
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, TrendType


class TrendAdaptiveDefensive(BaseStrategy):
    """
    Defensive strategy that adapts to the macro trend.

    - UPTREND + LOW: Hold long
    - UPTREND + HIGH: Exit to flat
    - DOWNTREND: Stay flat (protect capital in bear market)
    - SIDEWAYS + LOW: Optional cautious long
    - SIDEWAYS + HIGH: Exit
    """

    def __init__(
        self,
        reentry_delay: int = 2,
        fast_ma_period: int = 168,
        slow_ma_period: int = 720,
        trade_downtrend: bool = False,  # If True, short in downtrend
        trade_sideways: bool = True,
    ):
        super().__init__(name="TrendDefensive")
        self.reentry_delay = reentry_delay
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trade_downtrend = trade_downtrend
        self.trade_sideways = trade_sideways

        self._bars_since_low: int = 0
        self._prev_regime: Optional[str] = None
        self._current_trend: TrendType = 'SIDEWAYS'

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)

        # Track regime
        if predicted_regime == 'LOW':
            self._bars_since_low += 1
        else:
            self._bars_since_low = 0

        self._prev_regime = predicted_regime

        # Detect trend
        if len(history) >= self.slow_ma_period:
            self._current_trend = detect_trend(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
        else:
            self._current_trend = 'SIDEWAYS'

        # ========== HIGH VOLATILITY: EXIT ==========
        if predicted_regime == 'HIGH':
            if self.has_position():
                return 'CLOSE'
            return 'HOLD'

        # ========== LOW VOLATILITY ==========
        if predicted_regime == 'LOW':
            # Handle existing position
            if self.has_position():
                direction = self.position.direction

                # Exit LONG if trend turns down
                if direction == 'LONG' and self._current_trend == 'DOWNTREND':
                    return 'CLOSE'

                # Exit SHORT if trend turns up
                if direction == 'SHORT' and self._current_trend == 'UPTREND':
                    return 'CLOSE'

                return 'HOLD'

            # Consider new entry
            if self._bars_since_low < self.reentry_delay:
                return 'HOLD'

            # UPTREND: Go long
            if self._current_trend == 'UPTREND':
                return 'BUY'

            # DOWNTREND: Stay flat or short
            if self._current_trend == 'DOWNTREND':
                if self.trade_downtrend:
                    return 'SELL'  # Short in downtrend
                return 'HOLD'  # Stay flat (protect capital)

            # SIDEWAYS: Cautious long or flat
            if self._current_trend == 'SIDEWAYS':
                if self.trade_sideways:
                    return 'BUY'
                return 'HOLD'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Trend-Adaptive Defensive V3',
            'trend_filter': f'Fast: {self.fast_ma_period}h, Slow: {self.slow_ma_period}h',
            'reentry_delay': self.reentry_delay,
            'trade_downtrend': self.trade_downtrend,
            'trade_sideways': self.trade_sideways
        }

    def reset(self):
        super().reset()
        self._bars_since_low = 0
        self._prev_regime = None
        self._current_trend = 'SIDEWAYS'
