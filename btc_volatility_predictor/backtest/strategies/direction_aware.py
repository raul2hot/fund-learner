"""
Direction-Aware Mean Reversion Strategy.

Uses BOTH volatility AND direction predictions:
- LOW vol + UP direction: LONG mean reversion
- LOW vol + DOWN direction: SHORT mean reversion
- HIGH vol: No trades (or exit)

This strategy requires the combined predictor output or
predicted_direction in the row data.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class DirectionAwareMeanReversion(BaseStrategy):
    """
    Only trade when BOTH conditions are met:
    1. LOW volatility predicted
    2. Direction prediction matches trade direction

    This strategy uses combined predictions (e.g., "LOW_UP")
    or separate regime + direction predictions.
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,  # 2%
        stop_loss_pct: float = 0.015,   # 1.5%
        max_holding_bars: int = 24,
        require_direction_confirmation: bool = True,
        min_direction_confidence: float = 0.0,  # Minimum confidence for direction
    ):
        super().__init__(name="DirectionAwareMR")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.require_direction = require_direction_confirmation
        self.min_direction_confidence = min_direction_confidence
        self._prev_rsi: Optional[float] = None
        self._bars_held: int = 0

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate signal using both vol regime and direction.

        Note: predicted_regime can be:
        - Combined like "LOW_UP" or "HIGH_DOWN"
        - Simple like "LOW" or "HIGH" with direction in row['predicted_direction']
        """
        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)

        # Parse combined regime or get direction separately
        if '_' in predicted_regime:
            vol_regime, direction = predicted_regime.split('_')
        else:
            vol_regime = predicted_regime
            direction = row.get('predicted_direction', 'UNKNOWN')

        # Get direction confidence if available
        dir_confidence = row.get('dir_confidence', 1.0)

        # Track bars held
        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            pos_direction = self.position.direction

            if pos_direction == 'LONG':
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

            # 3. Exit if regime turns HIGH
            if vol_regime == 'HIGH':
                self._prev_rsi = rsi
                return 'CLOSE'

            # 4. Max holding period
            if self._bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                return 'CLOSE'

            self._prev_rsi = rsi
            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only in LOW vol
        if vol_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'

        signal: Signal = 'HOLD'

        if self._prev_rsi is not None:
            # LONG signal: RSI crosses below oversold
            rsi_cross_down = (rsi < self.rsi_oversold and
                             self._prev_rsi >= self.rsi_oversold)

            if rsi_cross_down:
                # Check direction confirmation
                if not self.require_direction:
                    signal = 'BUY'
                elif direction == 'UP':
                    if dir_confidence >= self.min_direction_confidence:
                        signal = 'BUY'
                # If direction is DOWN or UNKNOWN, don't take long

            # SHORT signal: RSI crosses above overbought
            if signal == 'HOLD':
                rsi_cross_up = (rsi > self.rsi_overbought and
                               self._prev_rsi <= self.rsi_overbought)

                if rsi_cross_up:
                    if not self.require_direction:
                        signal = 'SELL'
                    elif direction == 'DOWN':
                        if dir_confidence >= self.min_direction_confidence:
                            signal = 'SELL'
                    # If direction is UP or UNKNOWN, don't take short

        self._prev_rsi = rsi
        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'Direction-Aware Mean Reversion',
            'require_direction_confirmation': self.require_direction,
            'min_direction_confidence': self.min_direction_confidence,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'max_holding_bars': self.max_holding_bars
        }

    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._bars_held = 0


class DirectionOnlyStrategy(BaseStrategy):
    """
    Trade purely based on direction prediction.

    - UP direction: LONG
    - DOWN direction: SHORT or flat
    - HIGH vol: Exit

    Simpler version without mean reversion signals.
    """

    def __init__(
        self,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.015,
        min_confidence: float = 0.6,
        reentry_delay: int = 2,
    ):
        super().__init__(name="DirectionOnly")
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.reentry_delay = reentry_delay
        self._bars_since_trade: int = 999

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)

        # Parse combined regime
        if '_' in predicted_regime:
            vol_regime, direction = predicted_regime.split('_')
        else:
            vol_regime = predicted_regime
            direction = row.get('predicted_direction', 'UNKNOWN')

        dir_confidence = row.get('dir_confidence', 0.5)

        self._bars_since_trade += 1

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            pos_direction = self.position.direction

            if pos_direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price

            # Take profit
            if pnl_pct >= self.take_profit_pct:
                return 'CLOSE'

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return 'CLOSE'

            # Exit on HIGH vol
            if vol_regime == 'HIGH':
                return 'CLOSE'

            # Exit if direction reversed
            if pos_direction == 'LONG' and direction == 'DOWN' and dir_confidence >= self.min_confidence:
                return 'CLOSE'
            if pos_direction == 'SHORT' and direction == 'UP' and dir_confidence >= self.min_confidence:
                return 'CLOSE'

            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only trade in LOW vol
        if vol_regime == 'HIGH':
            return 'HOLD'

        # Respect reentry delay
        if self._bars_since_trade < self.reentry_delay:
            return 'HOLD'

        # Check confidence
        if dir_confidence < self.min_confidence:
            return 'HOLD'

        if direction == 'UP':
            self._bars_since_trade = 0
            return 'BUY'
        elif direction == 'DOWN':
            self._bars_since_trade = 0
            return 'SELL'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Direction Only',
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'min_confidence': self.min_confidence,
            'reentry_delay': self.reentry_delay
        }

    def reset(self):
        super().reset()
        self._bars_since_trade = 999
