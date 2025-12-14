"""
Defensive Strategy - Use HIGH vol prediction to EXIT, not ENTER.

Logic:
- Hold a base LONG position during LOW volatility
- Exit (go flat) when HIGH volatility is predicted
- Re-enter when LOW volatility returns

This exploits the fact that HIGH vol is dangerous (we lose money there)
while LOW vol is safer for holding.

This is essentially a "risk-off" toggle using volatility predictions.
"""

from typing import Optional
from .base import BaseStrategy, Signal


class DefensiveStrategy(BaseStrategy):
    """
    Use volatility prediction defensively.

    - LOW vol: Hold long position
    - HIGH vol: Exit to cash (flat)

    This is essentially a "risk-off" toggle.
    """

    def __init__(
        self,
        reentry_delay: int = 2,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
    ):
        """
        Args:
            reentry_delay: Wait N bars after HIGH->LOW before re-entering
            take_profit_pct: Optional take profit (None = hold indefinitely)
            stop_loss_pct: Optional stop loss (None = only exit on HIGH vol)
        """
        super().__init__(name="Defensive")
        self.reentry_delay = reentry_delay
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self._bars_since_low: int = 0
        self._prev_regime: Optional[str] = None

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)

        # Track regime transitions
        regime_just_changed = (self._prev_regime != predicted_regime)

        if predicted_regime == 'LOW':
            self._bars_since_low += 1
        else:
            self._bars_since_low = 0

        self._prev_regime = predicted_regime

        # ========== HIGH VOLATILITY: EXIT ==========
        if predicted_regime == 'HIGH':
            if self.has_position():
                return 'CLOSE'
            return 'HOLD'

        # ========== LOW VOLATILITY: ENTER/HOLD ==========
        if predicted_regime == 'LOW':
            # Check TP/SL if in position
            if self.has_position():
                entry_price = self.position.entry_price
                direction = self.position.direction

                if direction == 'LONG':
                    pnl_pct = (close - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - close) / entry_price

                # Take profit
                if self.take_profit_pct and pnl_pct >= self.take_profit_pct:
                    return 'CLOSE'

                # Stop loss
                if self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct:
                    return 'CLOSE'

                return 'HOLD'

            # Not in position - consider entry
            # Wait for regime to stabilize
            if self._bars_since_low < self.reentry_delay:
                return 'HOLD'

            # Enter long position
            return 'BUY'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Defensive (Risk-Off Toggle)',
            'logic': 'Long in LOW vol, Flat in HIGH vol',
            'reentry_delay': self.reentry_delay,
            'take_profit_pct': f"{self.take_profit_pct*100}%" if self.take_profit_pct else 'None',
            'stop_loss_pct': f"{self.stop_loss_pct*100}%" if self.stop_loss_pct else 'None'
        }

    def reset(self):
        super().reset()
        self._bars_since_low = 0
        self._prev_regime = None


class DefensiveWithTPSL(DefensiveStrategy):
    """
    Defensive strategy with default take profit and stop loss.

    Same as Defensive but with sensible TP/SL defaults.
    """

    def __init__(
        self,
        reentry_delay: int = 2,
        take_profit_pct: float = 0.03,  # 3%
        stop_loss_pct: float = 0.02,    # 2%
    ):
        super().__init__(
            reentry_delay=reentry_delay,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct
        )
        self.name = "DefensiveTPSL"
