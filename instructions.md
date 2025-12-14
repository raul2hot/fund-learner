# BTC Volatility Strategy V3: Trend-Filtered Mean Reversion

## Executive Summary

V2 backtesting revealed critical insights:
- **Volatility prediction works**: 75.7% accuracy ✅
- **Direction prediction fails**: 49.8% (coin flip) ❌
- **Bear market problem**: All LONG strategies lost money (-12% to -26%)
- **Defensive strategy wins**: Beat market by 20% (-2.32% vs -22.68%)

**Key Insight**: We need a TREND FILTER, not a direction predictor.

---

## V3 Strategy: Trend-Filtered Mean Reversion

### Core Concept

Instead of predicting direction hour-by-hour (which doesn't work), we:
1. **Detect the macro trend** using longer-term moving averages (7-day, 30-day)
2. **Only trade WITH the trend** in LOW volatility
3. **Exit/stay flat** in HIGH volatility OR against-trend conditions

### Decision Matrix

| Trend | Vol Regime | Action |
|-------|------------|--------|
| UPTREND | LOW | LONG mean reversion signals |
| UPTREND | HIGH | EXIT / Reduce exposure |
| DOWNTREND | LOW | SHORT mean reversion OR stay flat |
| DOWNTREND | HIGH | EXIT / Stay flat |
| SIDEWAYS | LOW | Trade both directions cautiously |
| SIDEWAYS | HIGH | EXIT / Stay flat |

---

## Implementation Plan

### Phase 1: Setup Project Structure

```bash
# Create project structure (if not exists)
mkdir -p btc_volatility_predictor/{data/{raw,processed},backtest/{strategies,results_v3,trades},checkpoints,models,figures}

# Copy existing V2 files as base
cp backtest/strategies/mean_reversion_v2.py backtest/strategies/trend_filtered_mr.py
cp backtest/strategies/defensive.py backtest/strategies/trend_defensive.py
cp backtest/run_backtest_v2.py backtest/run_backtest_v3.py
```

### Phase 2: Add Trend Detection

Create `/backtest/strategies/trend_utils.py`:

```python
"""
Trend detection utilities for V3 strategies.

Trend Classification:
- UPTREND: Price > SMA_168 (7-day) AND SMA_168 > SMA_720 (30-day)
- DOWNTREND: Price < SMA_168 AND SMA_168 < SMA_720
- SIDEWAYS: Mixed conditions
"""

import numpy as np
from typing import Literal

TrendType = Literal['UPTREND', 'DOWNTREND', 'SIDEWAYS']


def calculate_sma(history: list[dict], period: int, price_key: str = 'close') -> float:
    """Calculate Simple Moving Average from history."""
    if len(history) < period:
        return None
    
    prices = [h.get(price_key, 0) for h in history[-period:]]
    return sum(prices) / len(prices)


def calculate_ema(history: list[dict], period: int, price_key: str = 'close') -> float:
    """Calculate Exponential Moving Average from history."""
    if len(history) < period:
        return None
    
    prices = [h.get(price_key, 0) for h in history]
    multiplier = 2 / (period + 1)
    
    # Initialize with SMA
    ema = sum(prices[:period]) / period
    
    # Apply EMA formula
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def detect_trend(
    current_price: float,
    history: list[dict],
    fast_period: int = 168,   # 7 days (168 hours)
    slow_period: int = 720,   # 30 days (720 hours)
    threshold: float = 0.005  # 0.5% buffer for sideways
) -> TrendType:
    """
    Detect market trend based on price vs moving averages.
    
    Args:
        current_price: Current close price
        history: List of historical bars
        fast_period: Short-term MA period (default 7 days)
        slow_period: Long-term MA period (default 30 days)
        threshold: Percentage buffer for sideways detection
    
    Returns:
        'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
    """
    if len(history) < slow_period:
        return 'SIDEWAYS'  # Not enough data
    
    sma_fast = calculate_sma(history, fast_period)
    sma_slow = calculate_sma(history, slow_period)
    
    if sma_fast is None or sma_slow is None:
        return 'SIDEWAYS'
    
    # Conditions for uptrend
    price_above_fast = current_price > sma_fast * (1 + threshold)
    fast_above_slow = sma_fast > sma_slow * (1 + threshold)
    
    # Conditions for downtrend
    price_below_fast = current_price < sma_fast * (1 - threshold)
    fast_below_slow = sma_fast < sma_slow * (1 - threshold)
    
    if price_above_fast and fast_above_slow:
        return 'UPTREND'
    elif price_below_fast and fast_below_slow:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'


def get_trend_strength(
    current_price: float,
    history: list[dict],
    fast_period: int = 168,
    slow_period: int = 720
) -> float:
    """
    Calculate trend strength (-1 to +1).
    
    Returns:
        -1.0 = Strong downtrend
         0.0 = Sideways
        +1.0 = Strong uptrend
    """
    if len(history) < slow_period:
        return 0.0
    
    sma_fast = calculate_sma(history, fast_period)
    sma_slow = calculate_sma(history, slow_period)
    
    if sma_fast is None or sma_slow is None:
        return 0.0
    
    # Price deviation from fast MA (normalized)
    price_dev = (current_price - sma_fast) / sma_fast
    
    # Fast MA deviation from slow MA (normalized)
    ma_dev = (sma_fast - sma_slow) / sma_slow
    
    # Combined strength (clamped to -1, +1)
    strength = (price_dev + ma_dev) / 2
    return max(-1.0, min(1.0, strength * 10))  # Scale up for sensitivity
```

### Phase 3: Create Trend-Filtered Strategies

#### Strategy 1: Trend-Filtered Mean Reversion

Create `/backtest/strategies/trend_filtered_mr.py`:

```python
"""
Trend-Filtered Mean Reversion Strategy V3.

Key improvements over V2:
1. Only trades LONG in UPTREND + LOW vol
2. Only trades SHORT in DOWNTREND + LOW vol
3. Stays flat in HIGH vol regardless of trend
4. Uses crossover signals with fixed TP/SL
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, get_trend_strength, TrendType


class TrendFilteredMeanReversion(BaseStrategy):
    """
    Mean reversion strategy that only trades WITH the trend.
    
    Logic:
    - UPTREND + LOW vol → LONG on RSI oversold
    - DOWNTREND + LOW vol → SHORT on RSI overbought  
    - SIDEWAYS + LOW vol → Trade both directions (cautious)
    - HIGH vol → No trades / Exit
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,    # 2%
        stop_loss_pct: float = 0.015,     # 1.5%
        max_holding_bars: int = 24,
        fast_ma_period: int = 168,        # 7-day MA
        slow_ma_period: int = 720,        # 30-day MA
        trade_sideways: bool = False,     # Whether to trade in sideways market
        min_trend_strength: float = 0.1,  # Minimum strength to consider trend
    ):
        super().__init__(name="TrendFilteredMR")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trade_sideways = trade_sideways
        self.min_trend_strength = min_trend_strength
        
        # State tracking
        self._prev_rsi: Optional[float] = None
        self._bars_held: int = 0
        self._current_trend: TrendType = 'SIDEWAYS'
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Generate trend-filtered mean reversion signal."""
        
        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        
        # Update bars held
        if self.has_position():
            self._bars_held += 1
        else:
            self._bars_held = 0
        
        # Detect trend (need enough history)
        if len(history) >= self.slow_ma_period:
            self._current_trend = detect_trend(
                close, history, 
                self.fast_ma_period, 
                self.slow_ma_period
            )
            trend_strength = get_trend_strength(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
        else:
            self._current_trend = 'SIDEWAYS'
            trend_strength = 0.0
        
        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction
            
            if direction == 'LONG':
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
            
            # 3. Exit on HIGH volatility
            if predicted_regime == 'HIGH':
                self._prev_rsi = rsi
                return 'CLOSE'
            
            # 4. Exit if trend reverses against position
            if direction == 'LONG' and self._current_trend == 'DOWNTREND':
                self._prev_rsi = rsi
                return 'CLOSE'
            if direction == 'SHORT' and self._current_trend == 'UPTREND':
                self._prev_rsi = rsi
                return 'CLOSE'
            
            # 5. Max holding period
            if self._bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                return 'CLOSE'
            
            self._prev_rsi = rsi
            return 'HOLD'
        
        # ========== ENTRY LOGIC ==========
        
        # Only trade in LOW volatility
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'
        
        # Need enough history for trend detection
        if len(history) < self.slow_ma_period:
            self._prev_rsi = rsi
            return 'HOLD'
        
        signal: Signal = 'HOLD'
        
        # Detect crossovers
        if self._prev_rsi is not None:
            rsi_cross_down = (rsi < self.rsi_oversold and 
                             self._prev_rsi >= self.rsi_oversold)
            rsi_cross_up = (rsi > self.rsi_overbought and 
                          self._prev_rsi <= self.rsi_overbought)
            
            # LONG: Only in UPTREND or SIDEWAYS (if enabled)
            if rsi_cross_down:
                if self._current_trend == 'UPTREND':
                    signal = 'BUY'
                elif self._current_trend == 'SIDEWAYS' and self.trade_sideways:
                    signal = 'BUY'
                # No LONG in DOWNTREND
            
            # SHORT: Only in DOWNTREND or SIDEWAYS (if enabled)
            if signal == 'HOLD' and rsi_cross_up:
                if self._current_trend == 'DOWNTREND':
                    signal = 'SELL'
                elif self._current_trend == 'SIDEWAYS' and self.trade_sideways:
                    signal = 'SELL'
                # No SHORT in UPTREND
        
        self._prev_rsi = rsi
        return signal
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Trend-Filtered Mean Reversion V3',
            'trend_filter': f'Fast MA: {self.fast_ma_period}h, Slow MA: {self.slow_ma_period}h',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%",
            'max_holding_bars': self.max_holding_bars,
            'trade_sideways': self.trade_sideways
        }
    
    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._bars_held = 0
        self._current_trend = 'SIDEWAYS'
```

#### Strategy 2: Trend-Adaptive Defensive

Create `/backtest/strategies/trend_defensive.py`:

```python
"""
Trend-Adaptive Defensive Strategy.

Improvement over V2 Defensive:
1. In UPTREND: Hold long during LOW vol, exit during HIGH vol (original defensive)
2. In DOWNTREND: Stay flat entirely OR go short during LOW vol
3. In SIDEWAYS: Reduced position size or flat
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, TrendType


class TrendAdaptiveDefensive(BaseStrategy):
    """
    Defensive strategy that adapts to the macro trend.
    
    - UPTREND + LOW: Hold long
    - UPTREND + HIGH: Exit to flat
    - DOWNTREND: Stay flat (protect capital in bear market)
    - SIDEWAYS + LOW: Optional cautious long
    - SIDEWAYS + HIGH: Exit
    """
    
    def __init__(
        self,
        reentry_delay: int = 2,
        fast_ma_period: int = 168,
        slow_ma_period: int = 720,
        trade_downtrend: bool = False,  # If True, short in downtrend
        trade_sideways: bool = True,
    ):
        super().__init__(name="TrendDefensive")
        self.reentry_delay = reentry_delay
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trade_downtrend = trade_downtrend
        self.trade_sideways = trade_sideways
        
        self._bars_since_low: int = 0
        self._prev_regime: Optional[str] = None
        self._current_trend: TrendType = 'SIDEWAYS'
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        
        close = row.get('close', 0)
        
        # Track regime
        if predicted_regime == 'LOW':
            self._bars_since_low += 1
        else:
            self._bars_since_low = 0
        
        self._prev_regime = predicted_regime
        
        # Detect trend
        if len(history) >= self.slow_ma_period:
            self._current_trend = detect_trend(
                close, history,
                self.fast_ma_period,
                self.slow_ma_period
            )
        else:
            self._current_trend = 'SIDEWAYS'
        
        # ========== HIGH VOLATILITY: EXIT ==========
        if predicted_regime == 'HIGH':
            if self.has_position():
                return 'CLOSE'
            return 'HOLD'
        
        # ========== LOW VOLATILITY ==========
        if predicted_regime == 'LOW':
            # Handle existing position
            if self.has_position():
                direction = self.position.direction
                
                # Exit LONG if trend turns down
                if direction == 'LONG' and self._current_trend == 'DOWNTREND':
                    return 'CLOSE'
                
                # Exit SHORT if trend turns up
                if direction == 'SHORT' and self._current_trend == 'UPTREND':
                    return 'CLOSE'
                
                return 'HOLD'
            
            # Consider new entry
            if self._bars_since_low < self.reentry_delay:
                return 'HOLD'
            
            # UPTREND: Go long
            if self._current_trend == 'UPTREND':
                return 'BUY'
            
            # DOWNTREND: Stay flat or short
            if self._current_trend == 'DOWNTREND':
                if self.trade_downtrend:
                    return 'SELL'  # Short in downtrend
                return 'HOLD'  # Stay flat (protect capital)
            
            # SIDEWAYS: Cautious long or flat
            if self._current_trend == 'SIDEWAYS':
                if self.trade_sideways:
                    return 'BUY'
                return 'HOLD'
        
        return 'HOLD'
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Trend-Adaptive Defensive V3',
            'trend_filter': f'Fast: {self.fast_ma_period}h, Slow: {self.slow_ma_period}h',
            'reentry_delay': self.reentry_delay,
            'trade_downtrend': self.trade_downtrend,
            'trade_sideways': self.trade_sideways
        }
    
    def reset(self):
        super().reset()
        self._bars_since_low = 0
        self._prev_regime = None
        self._current_trend = 'SIDEWAYS'
```

#### Strategy 3: Pure Trend Following (No Vol Filter)

Create `/backtest/strategies/trend_follower.py`:

```python
"""
Pure Trend Following Strategy (for comparison).

This strategy ignores volatility prediction and uses only trend.
Useful for comparing: Is vol filtering adding value on top of trend?
"""

from typing import Optional
from .base import BaseStrategy, Signal
from .trend_utils import detect_trend, TrendType


class TrendFollowerStrategy(BaseStrategy):
    """
    Pure trend following without volatility filter.
    
    - UPTREND: Hold long
    - DOWNTREND: Hold short or flat
    - SIDEWAYS: Flat
    
    Comparison baseline for trend + vol strategies.
    """
    
    def __init__(
        self,
        fast_ma_period: int = 168,
        slow_ma_period: int = 720,
        allow_short: bool = False,
    ):
        super().__init__(name="TrendFollower")
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.allow_short = allow_short
        self._current_trend: TrendType = 'SIDEWAYS'
        self._prev_trend: Optional[TrendType] = None
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,  # Ignored
        history: list[dict]
    ) -> Signal:
        
        close = row.get('close', 0)
        
        if len(history) < self.slow_ma_period:
            return 'HOLD'
        
        self._prev_trend = self._current_trend
        self._current_trend = detect_trend(
            close, history,
            self.fast_ma_period,
            self.slow_ma_period
        )
        
        # Exit on trend change
        if self.has_position():
            direction = self.position.direction
            
            if direction == 'LONG' and self._current_trend != 'UPTREND':
                return 'CLOSE'
            if direction == 'SHORT' and self._current_trend != 'DOWNTREND':
                return 'CLOSE'
            
            return 'HOLD'
        
        # Entry
        if self._current_trend == 'UPTREND':
            return 'BUY'
        
        if self._current_trend == 'DOWNTREND' and self.allow_short:
            return 'SELL'
        
        return 'HOLD'
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Pure Trend Follower',
            'fast_ma': self.fast_ma_period,
            'slow_ma': self.slow_ma_period,
            'allow_short': self.allow_short,
            'vol_filter': 'NONE'
        }
    
    def reset(self):
        super().reset()
        self._current_trend = 'SIDEWAYS'
        self._prev_trend = None
```

### Phase 4: Update Strategy Registry

Add to `/backtest/strategies/__init__.py`:

```python
# V3 Strategies - Trend Filtered
from .trend_utils import detect_trend, get_trend_strength, TrendType
from .trend_filtered_mr import TrendFilteredMeanReversion
from .trend_defensive import TrendAdaptiveDefensive
from .trend_follower import TrendFollowerStrategy

__all__ = [
    # ... existing exports ...
    # V3 Strategies
    'detect_trend',
    'get_trend_strength', 
    'TrendType',
    'TrendFilteredMeanReversion',
    'TrendAdaptiveDefensive',
    'TrendFollowerStrategy',
]
```

### Phase 5: Create V3 Backtest Runner

Create `/backtest/run_backtest_v3.py`:

```python
"""
V3 Backtest Runner - Trend-Filtered Strategies

Tests the hypothesis: Trend filter + Vol filter > Vol filter alone
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.analyze import generate_report
from backtest.strategies import (
    # Baselines
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    # V2 (Vol filter only)
    MeanReversionV2Strategy,
    DefensiveStrategy,
    # V3 (Trend + Vol filter)
    TrendFilteredMeanReversion,
    TrendAdaptiveDefensive,
    TrendFollowerStrategy,
)

# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

DATA_PATH = "data/processed/features_365d.csv"
CHECKPOINT_PATH = "checkpoints/best_regime_model_90d.pt"
PREDICTIONS_PATH = "backtest/results_v3/test_predictions_90d.csv"
RESULTS_DIR = "backtest/results_v3"


def get_v3_strategies():
    """All strategies for V3 comparison."""
    return [
        # === BASELINES ===
        BuyAndHoldStrategy(),
        
        # === V2: Vol Filter Only ===
        DefensiveStrategy(reentry_delay=2),
        MeanReversionV2Strategy(
            rsi_oversold=30,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            trade_short=True
        ),
        
        # === V3: Trend Only (no vol filter) ===
        TrendFollowerStrategy(
            fast_ma_period=168,
            slow_ma_period=720,
            allow_short=False
        ),
        TrendFollowerStrategy(
            fast_ma_period=168,
            slow_ma_period=720,
            allow_short=True
        ),
        
        # === V3: Trend + Vol Filter ===
        TrendFilteredMeanReversion(
            rsi_oversold=30,
            rsi_overbought=70,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_sideways=False
        ),
        TrendFilteredMeanReversion(
            rsi_oversold=30,
            rsi_overbought=70,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_sideways=True  # More aggressive
        ),
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=False,  # Stay flat in downtrend
            trade_sideways=True
        ),
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=True,   # Short in downtrend
            trade_sideways=False
        ),
    ]


def run_v3_backtest():
    """Run V3 backtest with trend-filtered strategies."""
    
    print("="*60)
    print("BTC VOLATILITY STRATEGY V3 BACKTESTER")
    print("Trend-Filtered Strategies")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/trades", exist_ok=True)
    
    # Load predictions (reuse from V2)
    if os.path.exists(PREDICTIONS_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
    else:
        print(f"ERROR: Predictions not found at {PREDICTIONS_PATH}")
        print("Run V2 backtest first to generate predictions.")
        return
    
    print(f"Loaded {len(predictions_df)} samples ({len(predictions_df)/24:.0f} days)")
    
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )
    
    strategies = get_v3_strategies()
    results = []
    
    print("\n" + "="*60)
    print("RUNNING V3 BACKTESTS")
    print("="*60)
    
    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] {strategy.name}")
        print(f"   Params: {strategy.get_params()}")
        
        result = engine.run(strategy, predictions_df)
        results.append(result)
        
        if result.num_trades > 0:
            engine.save_trades(result, f"{RESULTS_DIR}/trades")
        
        print(f"   Return: {result.total_return*100:+.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   MaxDD:  {result.max_drawdown*100:.1f}%")
        print(f"   Trades: {result.num_trades}")
        if result.num_trades > 0:
            print(f"   WinRate: {result.win_rate*100:.1f}%")
    
    # Compare V2 vs V3
    print("\n" + "="*60)
    print("V2 vs V3 COMPARISON")
    print("="*60)
    
    # Find key strategies
    buy_hold = next((r for r in results if r.strategy_name == "BuyAndHold"), None)
    defensive_v2 = next((r for r in results if r.strategy_name == "Defensive"), None)
    trend_defensive = [r for r in results if "TrendDefensive" in r.strategy_name]
    trend_mr = [r for r in results if "TrendFilteredMR" in r.strategy_name]
    
    if buy_hold:
        print(f"\nBuy & Hold: {buy_hold.total_return*100:+.2f}%")
    
    if defensive_v2:
        print(f"Defensive V2 (Vol Only): {defensive_v2.total_return*100:+.2f}%")
    
    print("\nTrend-Filtered Strategies:")
    for r in trend_defensive + trend_mr:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% (Sharpe: {r.sharpe_ratio:.2f})")
    
    # Generate report
    try:
        generate_report(results, predictions_df, RESULTS_DIR)
    except Exception as e:
        print(f"Warning: Could not generate full report: {e}")
    
    # Save summary
    summary = []
    for r in results:
        summary.append({
            'strategy': r.strategy_name,
            'return_pct': r.total_return * 100,
            'sharpe': r.sharpe_ratio,
            'max_dd_pct': r.max_drawdown * 100,
            'win_rate': r.win_rate * 100 if r.num_trades > 0 else 0,
            'num_trades': r.num_trades,
            'profit_factor': r.profit_factor
        })
    
    pd.DataFrame(summary).to_csv(f"{RESULTS_DIR}/summary_v3.csv", index=False)
    
    # Best strategy
    best = max(results, key=lambda r: r.sharpe_ratio)
    print(f"\nBest Strategy: {best.strategy_name}")
    print(f"  Return: {best.total_return*100:+.2f}%")
    print(f"  Sharpe: {best.sharpe_ratio:.2f}")
    
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_v3_backtest()
```

---

## Execution Steps

### Step 1: Ensure V2 Prerequisites

```bash
cd btc_volatility_predictor

# Check if V2 data and models exist
ls -la data/processed/features_365d.csv
ls -la checkpoints/best_regime_model_90d.pt
ls -la backtest/results_v2/test_predictions_90d.csv
```

If missing, run V2 first:
```bash
python fetch_extended_data.py
python -c "from data.features import prepare_dataset; prepare_dataset('data/raw/btcusdt_1h_365d.csv', 'data/processed/features_365d.csv')"
python train_regime_extended.py
python backtest/run_backtest_v2.py
```

### Step 2: Create V3 Files

```bash
# Create trend utilities
cat > backtest/strategies/trend_utils.py << 'EOF'
# ... paste trend_utils.py content ...
EOF

# Create trend-filtered strategies
cat > backtest/strategies/trend_filtered_mr.py << 'EOF'
# ... paste trend_filtered_mr.py content ...
EOF

cat > backtest/strategies/trend_defensive.py << 'EOF'
# ... paste trend_defensive.py content ...
EOF

cat > backtest/strategies/trend_follower.py << 'EOF'
# ... paste trend_follower.py content ...
EOF

# Update __init__.py
# Add V3 imports to backtest/strategies/__init__.py

# Create V3 runner
cat > backtest/run_backtest_v3.py << 'EOF'
# ... paste run_backtest_v3.py content ...
EOF
```

### Step 3: Run V3 Backtest

```bash
python backtest/run_backtest_v3.py
```

---

## Expected Results

### Hypothesis

| Strategy Type | Expected in Bear Market | Expected in Bull Market |
|--------------|-------------------------|-------------------------|
| Buy & Hold | -20% to -30% | +20% to +50% |
| Vol Filter Only (V2) | -2% to -5% | +5% to +15% |
| Trend Only (No Vol) | -5% to +5% | +15% to +30% |
| **Trend + Vol (V3)** | **-2% to +5%** | **+20% to +40%** |

### Success Criteria

1. **Trend-filtered strategies outperform Buy & Hold** in bear market
2. **V3 outperforms V2** in terms of Sharpe ratio
3. **TrendAdaptiveDefensive** with `trade_downtrend=False` should have lowest drawdown
4. **TrendFilteredMR** should have better risk-adjusted returns than MeanReversionV2

---

## Key Insights from V2 → V3

| V2 Finding | V3 Response |
|------------|-------------|
| Direction prediction = 49.8% | Use trend (168/720h MA) instead |
| All LONG strategies lost in bear market | Only LONG in uptrend, flat/short in downtrend |
| Defensive beat market by 20% | Add trend awareness to defensive |
| Vol filtering improved returns by 14% | Keep vol filter, add trend layer |

---

## Parameter Tuning Guidelines

### Trend Detection Parameters

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| fast_ma_period | 336 (14d) | 168 (7d) | 72 (3d) |
| slow_ma_period | 1440 (60d) | 720 (30d) | 336 (14d) |
| threshold | 0.01 | 0.005 | 0.002 |

### Strategy Parameters

| Parameter | Safe | Balanced | Aggressive |
|-----------|------|----------|------------|
| trade_sideways | False | True | True |
| trade_downtrend | False | False | True (short) |
| take_profit_pct | 0.015 | 0.02 | 0.03 |
| stop_loss_pct | 0.01 | 0.015 | 0.02 |

---

## Summary

V3 addresses the core problem discovered in V2:
- **Problem**: Can't predict hour-by-hour direction
- **Solution**: Use macro trend (7-day / 30-day MA) instead
- **Result**: Trade WITH the trend, not against it

The combination of:
1. **Volatility filter** (75.7% accurate) → Avoid HIGH vol
2. **Trend filter** (not prediction, just detection) → Trade WITH trend

Should produce the most robust strategy for varying market conditions.
