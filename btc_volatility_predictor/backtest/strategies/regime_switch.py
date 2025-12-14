"""
Regime-Switching Strategy (Adaptive).

Logic:
- When predicted regime = LOW: Use Mean Reversion rules
- When predicted regime = HIGH: Use Momentum rules
- Optionally close positions when regime switches
"""

from .base import BaseStrategy, Signal
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy


class RegimeSwitchStrategy(BaseStrategy):
    """Adaptive strategy that switches between mean reversion and momentum."""

    def __init__(
        self,
        close_on_regime_switch: bool = True,
        # Mean reversion params
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # Momentum params
        fast_ema: int = 9,
        slow_ema: int = 21,
        use_macd_confirmation: bool = True
    ):
        super().__init__(name="RegimeSwitch")
        self.close_on_regime_switch = close_on_regime_switch

        # Initialize sub-strategies
        self.mean_reversion = MeanReversionStrategy(
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought
        )
        self.momentum = MomentumStrategy(
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            use_macd_confirmation=use_macd_confirmation
        )

        self._prev_regime = None
        self._active_strategy = None

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate signal using appropriate strategy for current regime.

        Switches between mean reversion (LOW) and momentum (HIGH).
        """
        # Check for regime switch
        regime_switched = (self._prev_regime is not None and
                          self._prev_regime != predicted_regime)

        # Close position on regime switch if configured
        if regime_switched and self.close_on_regime_switch and self.has_position():
            self._prev_regime = predicted_regime
            return 'CLOSE'

        self._prev_regime = predicted_regime

        # Select strategy based on regime
        if predicted_regime == 'LOW':
            strategy = self.mean_reversion
            self._active_strategy = 'mean_reversion'
        else:  # HIGH
            strategy = self.momentum
            self._active_strategy = 'momentum'

        # Sync position state with active strategy
        strategy.set_position(self.position)

        # Generate signal from active strategy
        # Note: We pass the ACTUAL regime to sub-strategy (they will accept it)
        signal = strategy.generate_signal(row, predicted_regime, history)

        return signal

    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate stop loss using active strategy."""
        if self._active_strategy == 'mean_reversion':
            return self.mean_reversion.calculate_stop_loss(entry_price, atr, direction)
        else:
            return self.momentum.calculate_stop_loss(entry_price, atr, direction)

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {
            'strategy': 'Regime Switch',
            'close_on_regime_switch': self.close_on_regime_switch,
            'low_regime_strategy': 'Mean Reversion',
            'high_regime_strategy': 'Momentum',
            'mean_reversion_params': self.mean_reversion.get_params(),
            'momentum_params': self.momentum.get_params()
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._prev_regime = None
        self._active_strategy = None
        self.mean_reversion.reset()
        self.momentum.reset()
