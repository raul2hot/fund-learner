"""
Mean Reversion Strategy V2 - Based on proven approach.

Key changes from V1:
1. Use signal CROSSOVER (RSI crosses below 30), not just level
2. Fixed take profit / stop loss percentages
3. Mean reversion exit to Bollinger middle band
4. Maximum holding period (24 bars)
5. Only trade in LOW volatility regime

This strategy is based on the proven vol_filtered_mean_reversion_v2.py approach
which showed +39.90% return with vol filtering vs ~0% unfiltered.
"""

from typing import Optional
from .base import BaseStrategy, Signal, Position


class MeanReversionV2Strategy(BaseStrategy):
    """
    Proven mean reversion approach with fixed TP/SL.

    Entry Signals (LONG only in LOW vol):
    - RSI crosses below oversold threshold (30)
    - OR price crosses below lower Bollinger Band

    Exit:
    - Take profit: +2% from entry
    - Stop loss: -1.5% from entry
    - Mean reversion: Price returns to BB middle
    - Max holding: 24 bars
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,  # 2%
        stop_loss_pct: float = 0.015,   # 1.5%
        max_holding_bars: int = 24,
        min_bars_between_signals: int = 4,
        trade_short: bool = True,  # Also trade shorts on overbought
    ):
        super().__init__(name="MeanReversionV2")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.min_bars_between = min_bars_between_signals
        self.trade_short = trade_short

        # Track state
        self._prev_rsi: Optional[float] = None
        self._prev_bb_position: Optional[float] = None
        self._bars_since_signal: int = 999
        self._entry_bar: Optional[int] = None

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Generate mean reversion signal with crossover detection."""

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        bb_position = row.get('bb_position', 0.5)

        self._bars_since_signal += 1

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction
            bars_held = self._bars_since_signal

            # Calculate current P&L
            if direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price

            # Exit conditions (in priority order)

            # 1. Take profit
            if pnl_pct >= self.take_profit_pct:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'

            # 2. Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'

            # 3. Mean reversion (price returned to middle BB)
            if direction == 'LONG' and bb_position >= 0.45:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            if direction == 'SHORT' and bb_position <= 0.55:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'

            # 4. Max holding period
            if bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'

            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only enter in LOW volatility regime
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'

        # Respect minimum bars between signals
        if self._bars_since_signal < self.min_bars_between:
            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'

        signal: Signal = 'HOLD'

        # Detect CROSSOVERS (not just levels)
        if self._prev_rsi is not None and self._prev_bb_position is not None:

            # LONG: RSI crosses below oversold
            rsi_cross_down = (rsi < self.rsi_oversold and
                             self._prev_rsi >= self.rsi_oversold)

            # LONG: BB position crosses below lower band (0.0)
            bb_cross_down = (bb_position < 0.0 and
                            self._prev_bb_position >= 0.0)

            if rsi_cross_down or bb_cross_down:
                signal = 'BUY'
                self._bars_since_signal = 0

            # SHORT signals (if enabled)
            if self.trade_short and signal == 'HOLD':
                rsi_cross_up = (rsi > self.rsi_overbought and
                               self._prev_rsi <= self.rsi_overbought)

                bb_cross_up = (bb_position > 1.0 and
                              self._prev_bb_position <= 1.0)

                if rsi_cross_up or bb_cross_up:
                    signal = 'SELL'
                    self._bars_since_signal = 0

        self._prev_rsi = rsi
        self._prev_bb_position = bb_position
        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'Mean Reversion V2',
            'regime': 'LOW only',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit_pct': f"{self.take_profit_pct*100}%",
            'stop_loss_pct': f"{self.stop_loss_pct*100}%",
            'max_holding_bars': self.max_holding_bars,
            'trade_short': self.trade_short
        }

    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._prev_bb_position = None
        self._bars_since_signal = 999
        self._entry_bar = None
