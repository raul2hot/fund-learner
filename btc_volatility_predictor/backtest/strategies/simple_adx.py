"""
Simplified ADX Strategy V4.1 - Relaxed Conditions

Problems with V4:
1. ADX might not be in predictions (need to check)
2. Requiring RSI CROSSOVER + ADX > 25 + UPTREND + LOW vol = too rare
3. 720-bar warmup for trend = lose first 30 days

Fixes:
1. Use RSI LEVEL instead of crossover
2. Lower ADX thresholds
3. Shorter MA periods for trend
4. Fallback if ADX not available
"""

import os
import sys

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from backtest.strategies.base import BaseStrategy, Signal
else:
    from .base import BaseStrategy, Signal

from typing import Optional


def simple_trend(close: float, history: list[dict], fast: int = 72, slow: int = 168) -> str:
    """
    Simplified trend detection with shorter periods.

    Args:
        close: Current close price
        history: Price history
        fast: Fast MA period (3 days = 72 hours)
        slow: Slow MA period (7 days = 168 hours)

    Returns:
        'UP', 'DOWN', or 'SIDE'
    """
    if len(history) < slow:
        return 'SIDE'

    # Calculate SMAs
    fast_prices = [h.get('close', 0) for h in history[-fast:]]
    slow_prices = [h.get('close', 0) for h in history[-slow:]]

    sma_fast = sum(fast_prices) / len(fast_prices)
    sma_slow = sum(slow_prices) / len(slow_prices)

    # Trend conditions
    if close > sma_fast > sma_slow:
        return 'UP'
    elif close < sma_fast < sma_slow:
        return 'DOWN'
    else:
        return 'SIDE'


class SimpleADXTrend(BaseStrategy):
    """
    Simplified ADX strategy with relaxed conditions.

    Entry (LONG):
    - Vol = LOW
    - Trend = UP (3d/7d MA)
    - ADX > 15 (if available) OR skip ADX check
    - RSI < 45 (not extreme, just below middle)

    Exit:
    - TP: 2%
    - SL: 1.5%
    - Vol = HIGH
    - Trend reverses
    """

    def __init__(
        self,
        adx_min: float = 15,           # Lower than before
        rsi_entry_long: float = 45,    # Below middle, not extreme
        rsi_entry_short: float = 55,   # Above middle
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.015,
        max_holding_bars: int = 48,
        require_adx: bool = False,     # Can work without ADX
    ):
        super().__init__(name="SimpleADXTrend")
        self.adx_min = adx_min
        self.rsi_entry_long = rsi_entry_long
        self.rsi_entry_short = rsi_entry_short
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.require_adx = require_adx

        self._bars_held: int = 0

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        adx = row.get('adx_14', None)  # Might not exist

        # Track holding period
        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        # Get trend (shorter periods)
        trend = simple_trend(close, history, fast=72, slow=168)

        # ADX check (flexible)
        adx_ok = True
        if self.require_adx:
            if adx is None:
                adx_ok = False
            elif adx < self.adx_min:
                adx_ok = False
        elif adx is not None:
            # ADX available but not required - still use it as soft filter
            adx_ok = adx >= self.adx_min

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
                return 'CLOSE'

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return 'CLOSE'

            # Exit on HIGH vol
            if predicted_regime == 'HIGH':
                return 'CLOSE'

            # Exit on trend reversal
            if direction == 'LONG' and trend == 'DOWN':
                return 'CLOSE'

            # Max holding
            if self._bars_held >= self.max_holding_bars:
                return 'CLOSE'

            return 'HOLD'

        # ========== ENTRY LOGIC ==========

        # Only trade in LOW vol
        if predicted_regime != 'LOW':
            return 'HOLD'

        # Need some history for trend
        if len(history) < 168:
            return 'HOLD'

        # LONG: Uptrend + RSI not overbought
        if trend == 'UP' and rsi < self.rsi_entry_long:
            if adx_ok:
                return 'BUY'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Simple ADX Trend V4.1',
            'adx_min': self.adx_min,
            'rsi_entry_long': self.rsi_entry_long,
            'require_adx': self.require_adx,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%"
        }

    def reset(self):
        super().reset()
        self._bars_held = 0


class TrendStrengthStrategy(BaseStrategy):
    """
    Uses ADX purely for position SIZING, not entry filtering.

    - Always trade with trend in LOW vol
    - Size position based on ADX:
        - ADX > 25: Full size (100%)
        - ADX 15-25: Medium size (simulated via tighter stops)
        - ADX < 15: Small size (very tight stops)
    """

    def __init__(
        self,
        base_tp: float = 0.02,
        base_sl: float = 0.015,
    ):
        super().__init__(name="TrendStrength")
        self.base_tp = base_tp
        self.base_sl = base_sl

        self._bars_held: int = 0
        self._current_tp: float = base_tp
        self._current_sl: float = base_sl

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        adx = row.get('adx_14', 20)  # Default to moderate

        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        trend = simple_trend(close, history, 72, 168)

        # Adjust TP/SL based on ADX
        if adx > 25:
            self._current_tp = self.base_tp * 1.5  # 3% for strong trends
            self._current_sl = self.base_sl
        elif adx > 15:
            self._current_tp = self.base_tp
            self._current_sl = self.base_sl
        else:
            self._current_tp = self.base_tp * 0.75  # Smaller target in range
            self._current_sl = self.base_sl * 0.75  # Tighter stop too

        # ========== EXIT ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction

            pnl_pct = (close - entry_price) / entry_price if direction == 'LONG' else (entry_price - close) / entry_price

            if pnl_pct >= self._current_tp:
                return 'CLOSE'
            if pnl_pct <= -self._current_sl:
                return 'CLOSE'
            if predicted_regime == 'HIGH':
                return 'CLOSE'
            if direction == 'LONG' and trend == 'DOWN':
                return 'CLOSE'
            if self._bars_held >= 48:
                return 'CLOSE'

            return 'HOLD'

        # ========== ENTRY ==========
        if predicted_regime != 'LOW':
            return 'HOLD'

        if len(history) < 168:
            return 'HOLD'

        # Entry on trend + RSI
        if trend == 'UP' and rsi < 45:
            return 'BUY'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Trend Strength (ADX for sizing)',
            'base_tp': f"{self.base_tp*100}%",
            'base_sl': f"{self.base_sl*100}%",
        }

    def reset(self):
        super().reset()
        self._bars_held = 0


class VolTrendCombo(BaseStrategy):
    """
    The simplest combo that should definitely trade:

    - Vol = LOW -> OK to trade
    - Trend = UP -> LONG
    - Trend = DOWN -> FLAT (no short to keep it simple)

    No ADX, no RSI entry filter. Just Vol + Trend.
    """

    def __init__(
        self,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.015,
        max_holding_bars: int = 72,
    ):
        super().__init__(name="VolTrendCombo")
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self._bars_held: int = 0

    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:

        close = row.get('close', 0)

        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0

        trend = simple_trend(close, history, 72, 168)

        # ========== EXIT ==========
        if self.has_position():
            entry_price = self.position.entry_price
            pnl_pct = (close - entry_price) / entry_price

            # TP
            if pnl_pct >= self.take_profit_pct:
                return 'CLOSE'
            # SL
            if pnl_pct <= -self.stop_loss_pct:
                return 'CLOSE'
            # High vol
            if predicted_regime == 'HIGH':
                return 'CLOSE'
            # Trend reversal
            if trend == 'DOWN':
                return 'CLOSE'
            # Max hold
            if self._bars_held >= self.max_holding_bars:
                return 'CLOSE'

            return 'HOLD'

        # ========== ENTRY ==========
        # Simple: LOW vol + UPTREND = LONG
        if predicted_regime == 'LOW' and len(history) >= 168:
            if trend == 'UP':
                return 'BUY'

        return 'HOLD'

    def get_params(self) -> dict:
        return {
            'strategy': 'Vol + Trend Combo (Simplest)',
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'logic': 'LOW vol + UPTREND = LONG'
        }

    def reset(self):
        super().reset()
        self._bars_held = 0


# ============================================================
# BACKTEST RUNNER
# ============================================================

def run_v4_simplified():
    """Run simplified V4 strategies."""
    import pandas as pd

    from backtest.engine import BacktestEngine
    from backtest.strategies import BuyAndHoldStrategy, DefensiveStrategy
    from backtest.strategies.trend_defensive import TrendAdaptiveDefensive

    print("="*60)
    print("V4.1 SIMPLIFIED ADX STRATEGIES")
    print("="*60)

    # Load data
    predictions_path = "backtest/results_v3/test_predictions_90d.csv"
    if not os.path.exists(predictions_path):
        predictions_path = "backtest/results_v2/test_predictions_90d.csv"

    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} samples")

    # Check for ADX
    has_adx = 'adx_14' in df.columns
    print(f"ADX available: {has_adx}")

    if has_adx:
        print(f"ADX mean: {df['adx_14'].mean():.2f}")

    # Strategies
    strategies = [
        BuyAndHoldStrategy(),
        DefensiveStrategy(reentry_delay=2),
        TrendAdaptiveDefensive(trade_downtrend=False, trade_sideways=True),

        # V4.1 Simplified
        VolTrendCombo(),  # Simplest - should definitely trade
        TrendStrengthStrategy(),  # Uses ADX for sizing only
        SimpleADXTrend(require_adx=False),  # Works without ADX
        SimpleADXTrend(require_adx=True, adx_min=15),  # Requires ADX but lower threshold
    ]

    engine = BacktestEngine(initial_capital=10000)

    results = []
    for s in strategies:
        print(f"\nTesting {s.name}...")
        result = engine.run(s, df)
        results.append(result)

        print(f"  Return: {result.total_return*100:+.2f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Trades: {result.num_trades}")
        if result.num_trades > 0:
            print(f"  WinRate: {result.win_rate*100:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Trades':>8}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x.total_return, reverse=True):
        print(f"{r.strategy_name:<25} {r.total_return*100:>+9.2f}% {r.sharpe_ratio:>8.2f} {r.num_trades:>8}")


if __name__ == "__main__":
    run_v4_simplified()
