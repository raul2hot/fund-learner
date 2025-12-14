# BTC Volatility Strategy V4: ADX Trend Strength Filter

## Evolution Summary

| Version | Components | Best Result |
|---------|------------|-------------|
| V2 | Vol prediction only | -2.32% (beat market by 20%) |
| V3 | Vol + Trend | -0.39% (beat market by 22%) |
| **V4** | **Vol + Trend + ADX** | **Target: +5% to +15%** |

## The Missing Piece: Trend STRENGTH

V3 detected trend DIRECTION but not STRENGTH:
- Weak uptrend → Should be cautious
- Strong uptrend → Should be aggressive
- Ranging market → Mean reversion works better

**ADX (Average Directional Index)** solves this:
- ADX > 25 → Strong trend (follow it!)
- ADX 20-25 → Moderate trend
- ADX < 20 → Weak/ranging (mean revert)

---

## V4 Strategy Matrix

```
┌─────────────┬────────────┬───────────┬──────────────────────────────┐
│   TREND     │ VOL REGIME │    ADX    │          ACTION              │
├─────────────┼────────────┼───────────┼──────────────────────────────┤
│  UPTREND    │    LOW     │   > 25    │ AGGRESSIVE LONG (100% size)  │
│  UPTREND    │    LOW     │   20-25   │ MODERATE LONG (75% size)     │
│  UPTREND    │    LOW     │   < 20    │ CAUTIOUS LONG (50% size)     │
│  UPTREND    │    HIGH    │   any     │ EXIT / Reduce to 25%         │
├─────────────┼────────────┼───────────┼──────────────────────────────┤
│  DOWNTREND  │    LOW     │   > 25    │ AGGRESSIVE SHORT (if enabled)│
│  DOWNTREND  │    LOW     │   < 25    │ FLAT (protect capital)       │
│  DOWNTREND  │    HIGH    │   any     │ FLAT                         │
├─────────────┼────────────┼───────────┼──────────────────────────────┤
│  SIDEWAYS   │    LOW     │   < 20    │ MEAN REVERSION (both dirs)   │
│  SIDEWAYS   │    LOW     │   > 20    │ WAIT (conflicting signals)   │
│  SIDEWAYS   │    HIGH    │   any     │ FLAT                         │
└─────────────┴────────────┴───────────┴──────────────────────────────┘
```

---

## Implementation

### Step 1: Add ADX Calculation

Add to `data/features.py`:

```python
def calc_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index).
    
    Returns:
        adx: Average Directional Index (trend strength 0-100)
        plus_di: +DI (bullish directional indicator)
        minus_di: -DI (bearish directional indicator)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Smoothed DM
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di
```

Update `engineer_features()` to include:

```python
# ADX - Trend Strength
features['adx_14'], features['plus_di_14'], features['minus_di_14'] = calc_adx(df, 14)
features['adx_21'], _, _ = calc_adx(df, 21)
```

### Step 2: Create ADX-Enhanced Strategy

Create `backtest/strategies/adx_trend_strategy.py`:

```python
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
```

### Step 3: Update Feature Engineering

Ensure ADX is in features. Add to `engineer_features()` in `data/features.py`:

```python
# ADX - Average Directional Index (Trend Strength)
adx_14, plus_di_14, minus_di_14 = calc_adx(df, 14)
features['adx_14'] = adx_14
features['plus_di_14'] = plus_di_14
features['minus_di_14'] = minus_di_14

# Also calculate longer period for confirmation
adx_21, _, _ = calc_adx(df, 21)
features['adx_21'] = adx_21
```

### Step 4: V4 Backtest Runner

Create `backtest/run_backtest_v4.py`:

```python
"""
V4 Backtest: ADX-Enhanced Strategies
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.analyze import generate_report
from backtest.strategies import (
    BuyAndHoldStrategy,
    DefensiveStrategy,
    TrendDefensive,
    ADXTrendStrategy,
    ADXMeanReversion,
    CombinedADXStrategy,
)

INITIAL_CAPITAL = 10000
RESULTS_DIR = "backtest/results_v4"


def get_v4_strategies():
    return [
        # Baselines
        BuyAndHoldStrategy(),
        DefensiveStrategy(reentry_delay=2),
        TrendDefensive(trade_downtrend=False, trade_sideways=True),
        
        # V4: ADX Strategies
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=True,
            allow_short=False,
            require_adx_confirm=True
        ),
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=True,
            allow_short=True,  # Allow shorting
            require_adx_confirm=True
        ),
        ADXMeanReversion(
            adx_max=20,
            rsi_oversold=25,
            rsi_overbought=75
        ),
        CombinedADXStrategy(
            adx_trend_threshold=25,
            adx_range_threshold=20
        ),
    ]


def main():
    print("="*60)
    print("V4 BACKTEST: ADX-Enhanced Strategies")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load predictions (reuse)
    predictions_df = pd.read_csv("backtest/results_v3/test_predictions_90d.csv")
    
    # Check if ADX exists
    if 'adx_14' not in predictions_df.columns:
        print("WARNING: ADX not in predictions. Need to regenerate features.")
        print("Run: python data/features.py with ADX calculation")
        return
    
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    
    results = []
    for strategy in get_v4_strategies():
        print(f"\nTesting {strategy.name}...")
        result = engine.run(strategy, predictions_df)
        results.append(result)
        print(f"  Return: {result.total_return*100:+.2f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    
    generate_report(results, predictions_df, RESULTS_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("V4 RESULTS SUMMARY")
    print("="*60)
    for r in sorted(results, key=lambda x: x.total_return, reverse=True):
        print(f"{r.strategy_name:25s} {r.total_return*100:+7.2f}%  Sharpe: {r.sharpe_ratio:+.2f}")


if __name__ == "__main__":
    main()
```

---

## V4 Expected Benefits

| Enhancement | Expected Impact |
|-------------|-----------------|
| ADX filter | +2-5% by avoiding weak trends |
| Dynamic position sizing | Lower drawdown, better risk-adjusted |
| Dual-mode (trend/range) | Captures more market conditions |
| Extended TP for strong trends | Lets winners run |
| ADX collapse exit | Early exit when trend dies |

---

## Comparison: V2 vs V3 vs V4

| Metric | V2 (Vol Only) | V3 (Vol+Trend) | V4 (Vol+Trend+ADX) |
|--------|---------------|----------------|---------------------|
| Bear Return | -2.32% | -0.39% | **Target: +2%** |
| Win Rate | 45% | 44% | **Target: 55%+** |
| Max DD | 7.9% | 7.9% | **Target: 5%** |
| Trades | 51 | 39 | **More selective** |

---

## Quick Setup

```bash
# 1. Add ADX to features
# Edit data/features.py to include calc_adx function

# 2. Regenerate features with ADX
python -c "from data.features import prepare_dataset; prepare_dataset('data/raw/btcusdt_1h_365d.csv', 'data/processed/features_365d.csv')"

# 3. Regenerate predictions (or add ADX to existing)
python backtest/run_backtest_v2.py  # This regenerates predictions

# 4. Create V4 strategy files
# Copy ADXTrendStrategy, ADXMeanReversion, CombinedADXStrategy to:
# backtest/strategies/adx_trend_strategy.py

# 5. Run V4 backtest
python backtest/run_backtest_v4.py
```

---

## Other Indicators to Consider (Future V5)

| Indicator | Use Case | Priority |
|-----------|----------|----------|
| **Volume Profile** | Support/resistance levels | High |
| **VWAP** | Intraday fair value | Medium |
| **Ichimoku Cloud** | Multi-signal confirmation | Medium |
| **Fibonacci** | Retracement levels | Low |
| **Order Flow** | Institutional activity | Advanced |

The key is **not adding too many indicators** - each additional filter reduces trade frequency. ADX is the highest-value addition because it directly measures what we need: **trend strength**.

# ADDITION OF ADX

```
"""
ADX Calculation - Add to data/features.py

Copy this function and add the feature calculations to engineer_features()
"""

import numpy as np
import pandas as pd


def calc_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """
    Calculate ADX (Average Directional Index) - Trend Strength Indicator.
    
    ADX values interpretation:
    - 0-20: Weak or absent trend (ranging market)
    - 20-25: Emerging trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)
    
    Returns:
        tuple: (adx, plus_di, minus_di)
            - adx: Average Directional Index (0-100)
            - plus_di: Positive Directional Indicator
            - minus_di: Negative Directional Indicator
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed True Range (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators (+DI and -DI)
    plus_di = 100 * (plus_dm_smooth / (atr + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (atr + 1e-10))
    
    # DX (Directional Movement Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx, plus_di, minus_di


# ============================================================
# ADD TO engineer_features() function in features.py:
# ============================================================
"""
# After the existing ATR calculations, add:

# --- ADX (Trend Strength) ---
features['adx_14'], features['plus_di_14'], features['minus_di_14'] = calc_adx(df, 14)
features['adx_21'], _, _ = calc_adx(df, 21)

# ADX-derived features
features['adx_trend_strength'] = features['adx_14'] / 50  # Normalized to ~0-2 range
features['di_diff'] = features['plus_di_14'] - features['minus_di_14']  # Directional bias
"""


# ============================================================
# UPDATE ENGINEERED_COLS in dataset.py to include:
# ============================================================
"""
# Add to ENGINEERED_COLS list:
'adx_14', 'plus_di_14', 'minus_di_14', 'adx_21', 'adx_trend_strength', 'di_diff',
"""


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    adx, plus_di, minus_di = calc_adx(df, 14)
    
    print("ADX Statistics:")
    print(f"  Mean: {adx.mean():.2f}")
    print(f"  Min:  {adx.min():.2f}")
    print(f"  Max:  {adx.max():.2f}")
    print(f"  Last 5 values: {adx.tail().values}")
    
    # Interpretation
    print("\nADX Interpretation:")
    print(f"  Weak trend (ADX < 20):   {(adx < 20).sum()} periods ({(adx < 20).mean()*100:.1f}%)")
    print(f"  Strong trend (ADX > 25): {(adx > 25).sum()} periods ({(adx > 25).mean()*100:.1f}%)")
    ```
