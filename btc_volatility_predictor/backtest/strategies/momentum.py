"""
Momentum Strategy (HIGH volatility regime).

Logic:
- ONLY trade when predicted regime = HIGH
- Entry LONG: EMA crossover (fast > slow) AND MACD histogram > 0
- Entry SHORT: EMA crossover (fast < slow) AND MACD histogram < 0
- Exit: Opposite crossover OR trailing stop at atr_multiplier x ATR
"""

from .base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Momentum/trend following strategy for HIGH volatility regimes."""

    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        use_macd_confirmation: bool = True,
        atr_trailing_stop: float = 3.0
    ):
        super().__init__(name="Momentum")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.use_macd_confirmation = use_macd_confirmation
        self.atr_trailing_stop = atr_trailing_stop

        # Track EMA crossover state
        self._prev_fast_above_slow = None

    def _calculate_ema(self, history: list[dict], period: int) -> float:
        """Calculate EMA from history."""
        if len(history) < period:
            # Not enough data, use simple average
            closes = [h.get('close', 0) for h in history]
            return sum(closes) / len(closes) if closes else 0

        # Calculate EMA
        multiplier = 2 / (period + 1)
        closes = [h.get('close', 0) for h in history]

        # Initialize with SMA
        ema = sum(closes[:period]) / period

        # Apply EMA formula
        for close in closes[period:]:
            ema = (close - ema) * multiplier + ema

        return ema

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate momentum signal based on EMA crossover.

        Only trades in HIGH volatility regime.
        """
        close = row.get('close', 0)
        macd_hist = row.get('macd_hist', 0)
        atr = row.get('atr_14', 0) * close  # Convert from normalized ATR

        # Need enough history for EMA calculation
        if len(history) < self.slow_ema:
            return 'HOLD'

        # Include current bar in history for EMA calculation
        full_history = history + [row]

        # Calculate EMAs
        fast_ema = self._calculate_ema(full_history, self.fast_ema)
        slow_ema = self._calculate_ema(full_history, self.slow_ema)

        # Determine crossover state
        fast_above_slow = fast_ema > slow_ema

        # Detect crossover
        bullish_crossover = False
        bearish_crossover = False

        if self._prev_fast_above_slow is not None:
            bullish_crossover = fast_above_slow and not self._prev_fast_above_slow
            bearish_crossover = not fast_above_slow and self._prev_fast_above_slow

        self._prev_fast_above_slow = fast_above_slow

        # MACD confirmation
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0

        # Check stop loss and update trailing stop
        if self.has_position():
            self.update_trailing_stop(close, atr, self.atr_trailing_stop)

            if self.check_stop_loss(close):
                return 'CLOSE'

            direction = self.get_position_direction()

            # Exit on opposite crossover
            if direction == 'LONG' and bearish_crossover:
                return 'CLOSE'
            if direction == 'SHORT' and bullish_crossover:
                return 'CLOSE'

            return 'HOLD'

        # Only enter in HIGH volatility regime
        if predicted_regime != 'HIGH':
            return 'HOLD'

        # Entry conditions
        # LONG: Bullish crossover (optionally with MACD confirmation)
        if bullish_crossover:
            if not self.use_macd_confirmation or macd_bullish:
                return 'BUY'

        # SHORT: Bearish crossover (optionally with MACD confirmation)
        if bearish_crossover:
            if not self.use_macd_confirmation or macd_bearish:
                return 'SELL'

        return 'HOLD'

    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate initial stop loss price."""
        stop_distance = atr * self.atr_trailing_stop
        if direction == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            'strategy': 'Momentum',
            'regime': 'HIGH',
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'use_macd_confirmation': self.use_macd_confirmation,
            'atr_trailing_stop': self.atr_trailing_stop
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._prev_fast_above_slow = None
