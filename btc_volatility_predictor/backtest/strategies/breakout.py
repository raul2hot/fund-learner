"""
Breakout Strategy (HIGH volatility regime).

Logic:
- ONLY trade when predicted regime = HIGH
- Entry LONG: Close breaks above lookback high WITH volume > volume_multiplier x average
- Entry SHORT: Close breaks below lookback low WITH volume > volume_multiplier x average
- Exit: Trailing stop at atr_multiplier x ATR OR opposite signal
- Confirm with "momentum candle" (current candle range > 2x average range)
"""

from .base import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for HIGH volatility regimes."""

    def __init__(
        self,
        lookback: int = 24,  # Hours for high/low
        volume_multiplier: float = 1.5,
        atr_trailing_stop: float = 2.0,
        range_multiplier: float = 2.0  # For momentum candle confirmation
    ):
        super().__init__(name="Breakout")
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier
        self.atr_trailing_stop = atr_trailing_stop
        self.range_multiplier = range_multiplier

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate breakout signal.

        Only trades in HIGH volatility regime.
        """
        close = row.get('close', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)
        volume_ma_ratio = row.get('volume_ma_ratio', 1.0)
        atr = row.get('atr_14', 0) * close  # Convert from normalized ATR

        # Check if we have enough history
        if len(history) < self.lookback:
            return 'HOLD'

        # Calculate lookback high/low
        lookback_data = history[-self.lookback:]
        lookback_high = max(h.get('high', 0) for h in lookback_data)
        lookback_low = min(h.get('low', float('inf')) for h in lookback_data)

        # Calculate average range for momentum candle check
        ranges = [(h.get('high', 0) - h.get('low', 0)) for h in lookback_data]
        avg_range = sum(ranges) / len(ranges) if ranges else 0
        current_range = high - low
        is_momentum_candle = current_range > (avg_range * self.range_multiplier)

        # Volume confirmation
        high_volume = volume_ma_ratio > self.volume_multiplier

        # Check stop loss and update trailing stop
        if self.has_position():
            # Update trailing stop
            self.update_trailing_stop(close, atr, self.atr_trailing_stop)

            if self.check_stop_loss(close):
                return 'CLOSE'

            direction = self.get_position_direction()

            # Check for opposite breakout signal (exit and reverse)
            if direction == 'LONG':
                if close < lookback_low and high_volume and is_momentum_candle:
                    return 'CLOSE'  # Will be followed by SELL on next iteration
            elif direction == 'SHORT':
                if close > lookback_high and high_volume and is_momentum_candle:
                    return 'CLOSE'  # Will be followed by BUY on next iteration

            return 'HOLD'

        # Only enter in HIGH volatility regime
        if predicted_regime != 'HIGH':
            return 'HOLD'

        # Entry conditions with volume and momentum confirmation
        # LONG: Breakout above lookback high
        if close > lookback_high and high_volume and is_momentum_candle:
            return 'BUY'

        # SHORT: Breakout below lookback low
        if close < lookback_low and high_volume and is_momentum_candle:
            return 'SELL'

        return 'HOLD'

    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate initial stop loss price (trailing stop will update)."""
        stop_distance = atr * self.atr_trailing_stop
        if direction == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            'strategy': 'Breakout',
            'regime': 'HIGH',
            'lookback': self.lookback,
            'volume_multiplier': self.volume_multiplier,
            'atr_trailing_stop': self.atr_trailing_stop,
            'range_multiplier': self.range_multiplier
        }
