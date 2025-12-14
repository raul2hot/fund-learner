"""
Mean Reversion Strategy (LOW volatility regime).

Logic:
- ONLY trade when predicted regime = LOW
- Entry LONG: RSI < oversold AND price below lower Bollinger Band position
- Entry SHORT: RSI > overbought AND price above upper Bollinger Band position
- Exit: Price returns to middle BB (bb_position ~ 0.5) OR RSI normalizes (40-60)
- Stop Loss: 1.5x ATR from entry
"""

from .base import BaseStrategy, Position, Signal


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for LOW volatility regimes."""

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_lower_threshold: float = 0.2,  # bb_position threshold for lower band
        bb_upper_threshold: float = 0.8,  # bb_position threshold for upper band
        rsi_exit_low: float = 40,
        rsi_exit_high: float = 60,
        bb_exit_low: float = 0.4,
        bb_exit_high: float = 0.6,
        atr_stop_multiplier: float = 1.5
    ):
        super().__init__(name="MeanReversion")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_lower_threshold = bb_lower_threshold
        self.bb_upper_threshold = bb_upper_threshold
        self.rsi_exit_low = rsi_exit_low
        self.rsi_exit_high = rsi_exit_high
        self.bb_exit_low = bb_exit_low
        self.bb_exit_high = bb_exit_high
        self.atr_stop_multiplier = atr_stop_multiplier

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate mean reversion signal.

        Only trades in LOW volatility regime.
        """
        # Get indicators
        rsi = row.get('rsi_14', 50)
        bb_position = row.get('bb_position', 0.5)
        atr = row.get('atr_14', 0)
        close = row.get('close', 0)

        # Check stop loss first
        if self.has_position() and self.check_stop_loss(close):
            return 'CLOSE'

        # If we have a position, check exit conditions
        if self.has_position():
            direction = self.get_position_direction()

            # Exit LONG: RSI normalizes or price returns to middle BB
            if direction == 'LONG':
                rsi_normalized = self.rsi_exit_low <= rsi <= self.rsi_exit_high
                bb_normalized = bb_position >= self.bb_exit_low
                if rsi_normalized or bb_normalized:
                    return 'CLOSE'

            # Exit SHORT: RSI normalizes or price returns to middle BB
            elif direction == 'SHORT':
                rsi_normalized = self.rsi_exit_low <= rsi <= self.rsi_exit_high
                bb_normalized = bb_position <= self.bb_exit_high
                if rsi_normalized or bb_normalized:
                    return 'CLOSE'

            return 'HOLD'

        # Only enter in LOW volatility regime
        if predicted_regime != 'LOW':
            return 'HOLD'

        # Entry conditions
        # LONG: RSI oversold + price near lower BB
        if rsi < self.rsi_oversold and bb_position < self.bb_lower_threshold:
            return 'BUY'

        # SHORT: RSI overbought + price near upper BB
        if rsi > self.rsi_overbought and bb_position > self.bb_upper_threshold:
            return 'SELL'

        return 'HOLD'

    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate stop loss price."""
        stop_distance = atr * self.atr_stop_multiplier
        if direction == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            'strategy': 'Mean Reversion',
            'regime': 'LOW',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_lower_threshold': self.bb_lower_threshold,
            'bb_upper_threshold': self.bb_upper_threshold,
            'rsi_exit_low': self.rsi_exit_low,
            'rsi_exit_high': self.rsi_exit_high,
            'atr_stop_multiplier': self.atr_stop_multiplier
        }
