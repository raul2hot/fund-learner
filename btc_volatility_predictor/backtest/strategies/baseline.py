"""
Baseline: Buy and Hold Strategy.

Logic:
- Buy at start, hold until end
- Used as benchmark comparison
"""

from .base import BaseStrategy, Signal


class BuyAndHoldStrategy(BaseStrategy):
    """Buy at the start, hold until end. Benchmark strategy."""

    def __init__(self):
        super().__init__(name="BuyAndHold")
        self._entered = False

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Buy once at the start, then hold.

        Args:
            row: Current bar data
            predicted_regime: Ignored for this strategy
            history: Previous bars

        Returns:
            'BUY' on first bar, 'HOLD' thereafter
        """
        if not self._entered and not self.has_position():
            self._entered = True
            return 'BUY'
        return 'HOLD'

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            'strategy': 'Buy and Hold',
            'description': 'Benchmark: buy at start, hold until end'
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._entered = False
