"""
V4 Strategy: ADX-Enhanced Trend Following

Combines:
1. Volatility regime (75.7% accurate)
2. Trend direction (7d/30d MA)
3. Trend strength (ADX)

Only trades when ALL three align.
"""

from typing import Optional, Literal
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, TrendType


class ADXTrendStrategy(BaseStrategy):
    """
    Triple-filter strategy: Vol + Trend + ADX

    Entry conditions:
    - Vol = LOW
    - Trend = UPTREND (for long) or DOWNTREND (for short)
    - ADX > threshold (confirms trend strength)

    Position sizing based on ADX:
    - ADX > 30: 100% position
    - ADX 25-30: 75% position
    - ADX 20-25: 50% position
    - ADX < 20: Skip or very small
    """

    def __init__(
        self,
        # ADX parameters
        adx_strong: float = 25,      # Strong trend threshold
        adx_weak: float = 20,        # Weak trend threshold
        use_dynamic_sizing: bool = True,

        # Entry parameters
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,

        # Exit parameters
        take_profit_pct: float = 0.025,  # 2.5% (slightly higher for strong trends)
        stop_loss_pct: float = 0.015,    # 1.5%
        max_holding_bars: int = 48,      # 2 days (longer for trends)

        # Trend parameters
        fast_ma_period: int = 168,
        slow_ma_period: int = 720,

        # Risk parameters
        allow_short: bool = False,
        require_adx_confirm: bool = True,
    ):
        super().__init__(name="ADXTrend")
        self.adx_strong = adx_strong
        self.adx_weak = adx_weak
        self.use_dynamic_sizing = use_dynamic_sizing
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.allow_short = allow_short
        self.require_adx_confirm = require_adx_confirm

        # State
        self._prev_rsi: Optional[float] = None
        self._bars_held: int = 0
        self._position_size: float = 1.0
        self._entry_adx: float = 0

    def _get_position_size(self, adx: float) -> float:
        """Calculate position size based on ADX."""
        if not self.use_dynamic_sizing:
            return 1.0

        if adx >= 30:
            return 1.0      # Full size for strong trends
        elif adx >= self.adx_strong:
            return 0.75     # 75% for moderate-strong
        elif adx >= self.adx_weak:
            return 0.5      # 50% for moderate
        else:
            return 0.25     # 25% for weak trends

    def _should_enter_long(
        self,
        trend: TrendType,
        adx: float,
        rsi: float,
        rsi_crossed: bool
    ) -> bool:
        """Check if long entry conditions are met."""
        # Must be in uptrend
        if trend != 'UPTREND':
            return False

        # ADX confirmation (if required)
        if self.require_adx_confirm and adx < self.adx_weak:
            return False

        # RSI oversold crossover
        if rsi_crossed and rsi < self.rsi_oversold:
            return True

        # Strong trend: enter on any pullback to RSI < 40
        if adx > self.adx_strong and rsi < 40:
            return True

        return False

    def _should_enter_short(
        self,
        trend: TrendType,
        adx: float,
        rsi: float,
        rsi_crossed: bool
    ) -> bool:
        """Check if short entry conditions are met."""
        if not self.allow_short:
            return False

        # Must be in downtrend
        if trend != 'DOWNTREND':
            return False

        # ADX confirmation - REQUIRE strong trend for shorts
        if adx < self.adx_strong:
            return False

        # RSI overbought crossover
        if rsi_crossed and rsi > self.rsi_overbought:
            return True

        # Strong downtrend: enter on any bounce to RSI > 60
        if adx > 30 and rsi > 60:
            return True

        return False

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Generate signal using Vol + Trend + ADX triple filter."""

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        adx = row.get('adx_14', 20)  # Default to moderate if missing

        # Track bars held
        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        # Detect trend
        if len(history) >= self.slow_ma_period:
            trend = detect_trend(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
        else:
            trend = 'SIDEWAYS'

        # RSI crossover detection
        rsi_cross_down = False
        rsi_cross_up = False
        if self._prev_rsi is not None:
            rsi_cross_down = (rsi < self.rsi_oversold and
                            self._prev_rsi >= self.rsi_oversold)
            rsi_cross_up = (rsi > self.rsi_overbought and
                          self._prev_rsi <= self.rsi_overbought)

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction

            if direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price

            # Dynamic TP based on entry ADX
            tp = self.take_profit_pct
            if self._entry_adx > 30:
                tp = self.take_profit_pct * 1.5  # 3.75% for strong trends

            # 1. Take profit (dynamic)
            if pnl_pct >= tp:
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

            # 4. Exit if trend reverses
            if direction == 'LONG' and trend == 'DOWNTREND':
                self._prev_rsi = rsi
                return 'CLOSE'
            if direction == 'SHORT' and trend == 'UPTREND':
                self._prev_rsi = rsi
                return 'CLOSE'

            # 5. Exit if ADX collapses (trend dying)
            if adx < 15 and self._entry_adx > 25:
                self._prev_rsi = rsi
                return 'CLOSE'

            # 6. Max holding period
            if self._bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                return 'CLOSE'

            self._prev_rsi = rsi
            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Filter 1: Only trade in LOW volatility
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'

        # Filter 2: Need enough history
        if len(history) < self.slow_ma_period:
            self._prev_rsi = rsi
            return 'HOLD'

        signal: Signal = 'HOLD'

        # Check long entry
        if self._should_enter_long(trend, adx, rsi, rsi_cross_down):
            self._position_size = self._get_position_size(adx)
            self._entry_adx = adx
            signal = 'BUY'

        # Check short entry
        elif self._should_enter_short(trend, adx, rsi, rsi_cross_up):
            self._position_size = self._get_position_size(adx)
            self._entry_adx = adx
            signal = 'SELL'

        # SIDEWAYS market with low ADX: Mean reversion
        elif trend == 'SIDEWAYS' and adx < self.adx_weak:
            if rsi_cross_down:
                self._position_size = 0.5  # Half size for ranging
                self._entry_adx = adx
                signal = 'BUY'
            elif rsi_cross_up and self.allow_short:
                self._position_size = 0.5
                self._entry_adx = adx
                signal = 'SELL'

        self._prev_rsi = rsi
        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'ADX Trend Strategy V4',
            'filters': 'Vol + Trend + ADX (Triple Filter)',
            'adx_strong': self.adx_strong,
            'adx_weak': self.adx_weak,
            'dynamic_sizing': self.use_dynamic_sizing,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'allow_short': self.allow_short,
            'require_adx_confirm': self.require_adx_confirm
        }

    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._bars_held = 0
        self._position_size = 1.0
        self._entry_adx = 0


class ADXMeanReversion(BaseStrategy):
    """
    ADX-filtered mean reversion for SIDEWAYS markets.

    Only trades when:
    - Vol = LOW
    - ADX < 20 (ranging/sideways market)
    - RSI at extremes

    This is the opposite of ADXTrendStrategy.
    """

    def __init__(
        self,
        adx_max: float = 20,         # Only trade when ADX below this
        rsi_oversold: float = 25,    # More extreme for ranging
        rsi_overbought: float = 75,
        take_profit_pct: float = 0.015,  # Smaller targets in range
        stop_loss_pct: float = 0.01,
        max_holding_bars: int = 12,  # Shorter holds in range
    ):
        super().__init__(name="ADXMeanRevert")
        self.adx_max = adx_max
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars

        self._prev_rsi: Optional[float] = None
        self._bars_held: int = 0

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Mean reversion in low-ADX environments."""

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        adx = row.get('adx_14', 25)
        bb_position = row.get('bb_position', 0.5)

        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction

            if direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price

            # Take profit
            if pnl_pct >= self.take_profit_pct:
                self._prev_rsi = rsi
                return 'CLOSE'

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self._prev_rsi = rsi
                return 'CLOSE'

            # Exit on HIGH vol OR if ADX rises (trend starting)
            if predicted_regime == 'HIGH' or adx > 25:
                self._prev_rsi = rsi
                return 'CLOSE'

            # Mean reversion target: BB middle
            if direction == 'LONG' and bb_position > 0.45:
                self._prev_rsi = rsi
                return 'CLOSE'
            if direction == 'SHORT' and bb_position < 0.55:
                self._prev_rsi = rsi
                return 'CLOSE'

            # Max holding
            if self._bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                return 'CLOSE'

            self._prev_rsi = rsi
            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only trade in LOW vol AND low ADX (ranging)
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'

        if adx > self.adx_max:
            self._prev_rsi = rsi
            return 'HOLD'

        signal: Signal = 'HOLD'

        # RSI crossover detection
        if self._prev_rsi is not None:
            rsi_cross_down = (rsi < self.rsi_oversold and
                            self._prev_rsi >= self.rsi_oversold)
            rsi_cross_up = (rsi > self.rsi_overbought and
                          self._prev_rsi <= self.rsi_overbought)

            # Additional BB confirmation for ranging markets
            if rsi_cross_down and bb_position < 0.1:
                signal = 'BUY'
            elif rsi_cross_up and bb_position > 0.9:
                signal = 'SELL'

        self._prev_rsi = rsi
        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'ADX Mean Reversion V4',
            'adx_max': self.adx_max,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%"
        }

    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._bars_held = 0


class CombinedADXStrategy(BaseStrategy):
    """
    Combines both approaches:
    - High ADX (>25): Trend following
    - Low ADX (<20): Mean reversion

    Automatically switches based on market regime.
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25,
        adx_range_threshold: float = 20,
    ):
        super().__init__(name="CombinedADX")

        self.trend_strategy = ADXTrendStrategy(
            adx_strong=adx_trend_threshold,
            require_adx_confirm=True,
            allow_short=False
        )

        self.range_strategy = ADXMeanReversion(
            adx_max=adx_range_threshold
        )

        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self._active_strategy: str = 'none'

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Route to appropriate strategy based on ADX."""

        adx = row.get('adx_14', 22)

        # Sync position with sub-strategies
        self.trend_strategy.set_position(self.position)
        self.range_strategy.set_position(self.position)

        # If in position, use the strategy that opened it
        if self.has_position():
            if self._active_strategy == 'trend':
                return self.trend_strategy.generate_signal(row, predicted_regime, history)
            elif self._active_strategy == 'range':
                return self.range_strategy.generate_signal(row, predicted_regime, history)

        # Not in position: decide which strategy to use
        if adx >= self.adx_trend_threshold:
            self._active_strategy = 'trend'
            signal = self.trend_strategy.generate_signal(row, predicted_regime, history)
        elif adx <= self.adx_range_threshold:
            self._active_strategy = 'range'
            signal = self.range_strategy.generate_signal(row, predicted_regime, history)
        else:
            # ADX in "no-man's land" (20-25): wait
            self._active_strategy = 'none'
            signal = 'HOLD'

        return signal

    def get_params(self) -> dict:
        return {
            'strategy': 'Combined ADX V4 (Auto-Switch)',
            'adx_trend_threshold': self.adx_trend_threshold,
            'adx_range_threshold': self.adx_range_threshold,
            'mode': 'Trend when ADX>25, Range when ADX<20'
        }

    def reset(self):
        super().reset()
        self.trend_strategy.reset()
        self.range_strategy.reset()
        self._active_strategy = 'none'
