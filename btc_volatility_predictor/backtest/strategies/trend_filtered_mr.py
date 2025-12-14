"""
Trend-Filtered Mean Reversion Strategy V3.

Key improvements over V2:
1. Only trades LONG in UPTREND + LOW vol
2. Only trades SHORT in DOWNTREND + LOW vol
3. Stays flat in HIGH vol regardless of trend
4. Uses crossover signals with fixed TP/SL
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, get_trend_strength, TrendType


class TrendFilteredMeanReversion(BaseStrategy):
    """
    Mean reversion strategy that only trades WITH the trend.

    Logic:
    - UPTREND + LOW vol → LONG on RSI oversold
    - DOWNTREND + LOW vol → SHORT on RSI overbought
    - SIDEWAYS + LOW vol → Trade both directions (cautious)
    - HIGH vol → No trades / Exit
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,    # 2%
        stop_loss_pct: float = 0.015,     # 1.5%
        max_holding_bars: int = 24,
        fast_ma_period: int = 168,        # 7-day MA
        slow_ma_period: int = 720,        # 30-day MA
        trade_sideways: bool = False,     # Whether to trade in sideways market
        min_trend_strength: float = 0.1,  # Minimum strength to consider trend
    ):
        super().__init__(name="TrendFilteredMR")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trade_sideways = trade_sideways
        self.min_trend_strength = min_trend_strength

        # State tracking
        self._prev_rsi: Optional[float] = None
        self._bars_held: int = 0
        self._current_trend: TrendType = 'SIDEWAYS'

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Generate trend-filtered mean reversion signal."""

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)

        # Update bars held
        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        # Detect trend (need enough history)
        if len(history) >= self.slow_ma_period:
            self._current_trend = detect_trend(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
            trend_strength = get_trend_strength(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
        else:
            self._current_trend = 'SIDEWAYS'
            trend_strength = 0.0

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction

            if direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price

            # 1. Take profit
            if pnl_pct >= self.take_profit_pct:
                self._prev_rsi = rsi
                return 'CLOSE'

            # 2. Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self._prev_rsi = rsi
                return 'CLOSE'

            # 3. Exit on HIGH volatility
            if predicted_regime == 'HIGH':
                self._prev_rsi = rsi
                return 'CLOSE'

            # 4. Exit if trend reverses against position
            if direction == 'LONG' and self._current_trend == 'DOWNTREND':
                self._prev_rsi = rsi
                return 'CLOSE'
            if direction == 'SHORT' and self._current_trend == 'UPTREND':
                self._prev_rsi = rsi
                return 'CLOSE'

            # 5. Max holding period
            if self._bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                return 'CLOSE'

            self._prev_rsi = rsi
            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only trade in LOW volatility
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'

        # Need enough history for trend detection
        if len(history) < self.slow_ma_period:
            self._prev_rsi = rsi
            return 'HOLD'

        signal: Signal = 'HOLD'

        # Detect crossovers
        if self._prev_rsi is not None:
            rsi_cross_down = (rsi < self.rsi_oversold and
                             self._prev_rsi >= self.rsi_oversold)
            rsi_cross_up = (rsi > self.rsi_overbought and
                          self._prev_rsi <= self.rsi_overbought)

            # LONG: Only in UPTREND or SIDEWAYS (if enabled)
            if rsi_cross_down:
                if self._current_trend == 'UPTREND':
                    signal = 'BUY'
                elif self._current_trend == 'SIDEWAYS' and self.trade_sideways:
                    signal = 'BUY'
                # No LONG in DOWNTREND

            # SHORT: Only in DOWNTREND or SIDEWAYS (if enabled)
            if signal == 'HOLD' and rsi_cross_up:
                if self._current_trend == 'DOWNTREND':
                    signal = 'SELL'
                elif self._current_trend == 'SIDEWAYS' and self.trade_sideways:
                    signal = 'SELL'
                # No SHORT in UPTREND

        self._prev_rsi = rsi
        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'Trend-Filtered Mean Reversion V3',
            'trend_filter': f'Fast MA: {self.fast_ma_period}h, Slow MA: {self.slow_ma_period}h',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'max_holding_bars': self.max_holding_bars,
            'trade_sideways': self.trade_sideways
        }

    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._bars_held = 0
        self._current_trend = 'SIDEWAYS'
