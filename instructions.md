# BTC Volatility Strategy V2 - Instructions for Claude Code

## Executive Summary

**Problem**: Our initial backtest (10 days, 80% vol accuracy) showed ALL strategies lost money:
- Breakout/Momentum in HIGH vol: -6% to -10%
- Mean Reversion in LOW vol: ~0% (no signals triggered)
- Buy & Hold: -3.7%

**Solution**: Learn from the proven `vol_filtered_mean_reversion_v2.py` approach:
- Use 180-365 days of data (not 173)
- Test on 90 days (not 10) for statistical significance  
- **Only trade LOW volatility** (avoid HIGH vol entirely)
- Use proper take profit/stop loss (not just indicator exits)
- Add direction prediction to know UP vs DOWN

---

## Phase 1: Data & Model Retraining

### Step 1.1: Fetch More Data (365 Days)

Modify `data/fetch_binance.py` or create new script:

```python
# fetch_extended_data.py
"""Fetch 365 days of BTC/USDT data for extended backtesting."""

from data.fetch_binance import fetch_binance_klines

if __name__ == "__main__":
    # Fetch 365 days instead of 180
    df = fetch_binance_klines(
        symbol="BTCUSDT",
        interval="1h",
        days=365,
        save_path="data/raw/btcusdt_1h_365d.csv"
    )
    print(f"Fetched {len(df)} candles ({len(df)/24:.0f} days)")
```

### Step 1.2: Re-engineer Features

```bash
# Run feature engineering on new data
python data/features.py --input data/raw/btcusdt_1h_365d.csv --output data/processed/features_365d.csv
```

Or modify `data/features.py` to accept command line args.

### Step 1.3: Retrain Model with 90-Day Test Split

**Critical Change**: Modify `train_regime.py` to use 90-day test period:

```python
# In create_regime_dataloaders():
test_days = 90  # Changed from 10 to 90
```

Create `train_regime_extended.py`:

```python
"""
Train regime classifier with extended data and 90-day test period.
"""
# Key changes from train_regime.py:
# 1. test_days = 90 (2160 hours)
# 2. data_path = "data/processed/features_365d.csv"
# 3. Save to checkpoints/best_regime_model_90d.pt
```

---

## Phase 2: Add Direction Prediction

### Step 2.1: Modify Target Labels

The current model predicts: `HIGH` or `LOW` volatility

We need to predict: `HIGH_UP`, `HIGH_DOWN`, `LOW_UP`, `LOW_DOWN`

**Option A**: Single 4-class classifier
**Option B**: Two separate classifiers (volatility + direction) ← Recommended

Create `train_direction.py`:

```python
"""
Train a separate direction classifier.
Predicts: Will next hour close HIGHER or LOWER than current close?
"""

# Target calculation:
# direction_up = (df['close'].shift(-1) > df['close']).astype(int)

# Model: Can reuse SPHNet architecture with classification head
# Or simpler: Use the MLP approach from vol_filtered_mean_reversion_v2.py
```

### Step 2.2: Combined Prediction for Trading

```python
class CombinedPredictor:
    """Combines volatility regime and direction predictions."""
    
    def __init__(self, vol_model, dir_model):
        self.vol_model = vol_model
        self.dir_model = dir_model
    
    def predict(self, prices, features):
        vol_regime = self.vol_model.predict(prices, features)  # HIGH/LOW
        direction = self.dir_model.predict(prices, features)   # UP/DOWN
        
        return {
            'vol_regime': vol_regime,
            'direction': direction,
            'combined': f"{vol_regime}_{direction}"  # e.g., "LOW_UP"
        }
```

---

## Phase 3: Implement Proven Mean Reversion Strategy

### Key Insights from `vol_filtered_mean_reversion_v2.py`

The old script showed **+39.90% return** with vol filtering vs unfiltered. Key differences:

| Our Failed Approach | Proven Approach |
|---------------------|-----------------|
| RSI < 30 AND bb_position < 0.2 | RSI crosses below 30 (momentum) |
| Exit on bb_position > 0.4 | Exit on take_profit (2%) or stop_loss (1%) |
| No fixed TP/SL | Fixed TP=2%, SL=1.5% |
| Complex exit conditions | Simple: TP, SL, mean_reversion, or max_holding |

### Step 3.1: Create Improved Mean Reversion Strategy

Create `backtest/strategies/mean_reversion_v2.py`:

```python
"""
Mean Reversion Strategy V2 - Based on proven approach.

Key changes from V1:
1. Use signal CROSSOVER (RSI crosses below 30), not just level
2. Fixed take profit / stop loss percentages
3. Mean reversion exit to Bollinger middle band
4. Maximum holding period (24 bars)
5. Only trade in LOW volatility regime
"""

from dataclasses import dataclass
from typing import Optional
from .base import BaseStrategy, Signal, Position


class MeanReversionV2Strategy(BaseStrategy):
    """
    Proven mean reversion approach with fixed TP/SL.
    
    Entry Signals (LONG only in LOW vol):
    - RSI crosses below oversold threshold (30)
    - OR price crosses below lower Bollinger Band
    - OR Z-score crosses below -2.0
    
    Exit:
    - Take profit: +2% from entry
    - Stop loss: -1.5% from entry  
    - Mean reversion: Price returns to BB middle
    - Max holding: 24 bars
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,  # 2%
        stop_loss_pct: float = 0.015,   # 1.5%
        max_holding_bars: int = 24,
        min_bars_between_signals: int = 4,
        trade_short: bool = True,  # Also trade shorts on overbought
    ):
        super().__init__(name="MeanReversionV2")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars
        self.min_bars_between = min_bars_between_signals
        self.trade_short = trade_short
        
        # Track state
        self._prev_rsi = None
        self._prev_bb_position = None
        self._bars_since_signal = 999
        self._entry_bar = None
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """Generate mean reversion signal with crossover detection."""
        
        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        bb_position = row.get('bb_position', 0.5)
        
        self._bars_since_signal += 1
        
        # ========== EXIT LOGIC ==========
        if self.has_position():
            entry_price = self.position.entry_price
            direction = self.position.direction
            bars_held = self._bars_since_signal
            
            # Calculate current P&L
            if direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price
            
            # Exit conditions (in priority order)
            
            # 1. Take profit
            if pnl_pct >= self.take_profit_pct:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            
            # 2. Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            
            # 3. Mean reversion (price returned to middle BB)
            if direction == 'LONG' and bb_position >= 0.45:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            if direction == 'SHORT' and bb_position <= 0.55:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            
            # 4. Max holding period
            if bars_held >= self.max_holding_bars:
                self._prev_rsi = rsi
                self._prev_bb_position = bb_position
                return 'CLOSE'
            
            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'
        
        # ========== ENTRY LOGIC ==========
        
        # Only enter in LOW volatility regime
        if predicted_regime != 'LOW':
            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'
        
        # Respect minimum bars between signals
        if self._bars_since_signal < self.min_bars_between:
            self._prev_rsi = rsi
            self._prev_bb_position = bb_position
            return 'HOLD'
        
        signal = 'HOLD'
        
        # Detect CROSSOVERS (not just levels)
        if self._prev_rsi is not None:
            
            # LONG: RSI crosses below oversold
            rsi_cross_down = (rsi < self.rsi_oversold and 
                             self._prev_rsi >= self.rsi_oversold)
            
            # LONG: BB position crosses below lower band (0.0)
            bb_cross_down = (bb_position < 0.0 and 
                            self._prev_bb_position >= 0.0)
            
            if rsi_cross_down or bb_cross_down:
                signal = 'BUY'
                self._bars_since_signal = 0
            
            # SHORT signals (if enabled)
            if self.trade_short:
                rsi_cross_up = (rsi > self.rsi_overbought and 
                               self._prev_rsi <= self.rsi_overbought)
                
                bb_cross_up = (bb_position > 1.0 and 
                              self._prev_bb_position <= 1.0)
                
                if rsi_cross_up or bb_cross_up:
                    signal = 'SELL'
                    self._bars_since_signal = 0
        
        self._prev_rsi = rsi
        self._prev_bb_position = bb_position
        return signal
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Mean Reversion V2',
            'regime': 'LOW only',
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'take_profit_pct': f"{self.take_profit_pct*100}%",
            'stop_loss_pct': f"{self.stop_loss_pct*100}%",
            'max_holding_bars': self.max_holding_bars,
            'trade_short': self.trade_short
        }
    
    def reset(self):
        super().reset()
        self._prev_rsi = None
        self._prev_bb_position = None
        self._bars_since_signal = 999
        self._entry_bar = None
```

### Step 3.2: Create Defensive Strategy

Create `backtest/strategies/defensive.py`:

```python
"""
Defensive Strategy - Use HIGH vol prediction to EXIT, not ENTER.

Logic:
- Hold a base LONG position during LOW volatility
- Exit (go flat) when HIGH volatility is predicted
- Re-enter when LOW volatility returns

This exploits the fact that HIGH vol is dangerous (we lose money there)
while LOW vol is safer for holding.
"""

class DefensiveStrategy(BaseStrategy):
    """
    Use volatility prediction defensively.
    
    - LOW vol: Hold long position
    - HIGH vol: Exit to cash (flat)
    
    This is essentially a "risk-off" toggle.
    """
    
    def __init__(self, reentry_delay: int = 2):
        super().__init__(name="Defensive")
        self.reentry_delay = reentry_delay  # Wait N bars after HIGH->LOW
        self._bars_since_low = 0
        self._prev_regime = None
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        
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
            # Wait for regime to stabilize
            if self._bars_since_low < self.reentry_delay:
                return 'HOLD'
            
            # Enter if not in position
            if not self.has_position():
                return 'BUY'
            
            return 'HOLD'
        
        return 'HOLD'
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Defensive (Risk-Off Toggle)',
            'logic': 'Long in LOW vol, Flat in HIGH vol',
            'reentry_delay': self.reentry_delay
        }
    
    def reset(self):
        super().reset()
        self._bars_since_low = 0
        self._prev_regime = None
```

### Step 3.3: Create Direction-Aware Strategy

Create `backtest/strategies/direction_aware.py`:

```python
"""
Direction-Aware Mean Reversion Strategy.

Uses BOTH volatility AND direction predictions:
- LOW vol + UP direction → LONG mean reversion
- LOW vol + DOWN direction → SHORT mean reversion  
- HIGH vol → No trades (or exit)
"""

class DirectionAwareMeanReversion(BaseStrategy):
    """
    Only trade when BOTH conditions are met:
    1. LOW volatility predicted
    2. Direction prediction matches trade direction
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.015,
        require_direction_confirmation: bool = True,
    ):
        super().__init__(name="DirectionAwareMR")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.require_direction = require_direction_confirmation
        self._prev_rsi = None
    
    def generate_signal(
        self,
        row: dict,
        predicted_regime: str,
        history: list[dict]
    ) -> Signal:
        """
        Generate signal using both vol regime and direction.
        
        Note: predicted_regime should now be like "LOW_UP" or "HIGH_DOWN"
              Or pass direction separately in row['predicted_direction']
        """
        close = row.get('close', 0)
        rsi = row.get('rsi_14', 50)
        
        # Parse combined regime or get direction separately
        if '_' in predicted_regime:
            vol_regime, direction = predicted_regime.split('_')
        else:
            vol_regime = predicted_regime
            direction = row.get('predicted_direction', 'UNKNOWN')
        
        # Exit logic (same as MeanReversionV2)
        if self.has_position():
            entry_price = self.position.entry_price
            pos_direction = self.position.direction
            
            if pos_direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - close) / entry_price
            
            if pnl_pct >= self.take_profit_pct:
                return 'CLOSE'
            if pnl_pct <= -self.stop_loss_pct:
                return 'CLOSE'
            
            # Also exit if regime turns HIGH
            if vol_regime == 'HIGH':
                return 'CLOSE'
            
            self._prev_rsi = rsi
            return 'HOLD'
        
        # Entry logic - only in LOW vol
        if vol_regime != 'LOW':
            self._prev_rsi = rsi
            return 'HOLD'
        
        signal = 'HOLD'
        
        if self._prev_rsi is not None:
            # LONG signal
            rsi_cross_down = (rsi < self.rsi_oversold and 
                             self._prev_rsi >= self.rsi_oversold)
            
            if rsi_cross_down:
                # Check direction confirmation
                if not self.require_direction or direction == 'UP':
                    signal = 'BUY'
            
            # SHORT signal
            rsi_cross_up = (rsi > self.rsi_overbought and 
                           self._prev_rsi <= self.rsi_overbought)
            
            if rsi_cross_up:
                if not self.require_direction or direction == 'DOWN':
                    signal = 'SELL'
        
        self._prev_rsi = rsi
        return signal
    
    def get_params(self) -> dict:
        return {
            'strategy': 'Direction-Aware Mean Reversion',
            'require_direction_confirmation': self.require_direction,
            'take_profit': f"{self.take_profit_pct*100}%",
            'stop_loss': f"{self.stop_loss_pct*100}%"
        }
    
    def reset(self):
        super().reset()
        self._prev_rsi = None
```

---

## Phase 4: Update Backtest Engine

### Step 4.1: Fix Engine to Handle TP/SL Properly

The current engine doesn't properly handle percentage-based TP/SL. Update `backtest/engine.py`:

```python
# Add to BacktestEngine.run():

# Inside the main loop, after getting signal:
if signal == 'BUY' or signal == 'SELL':
    # Record entry for TP/SL calculation
    position.entry_price = close
    position.entry_bar = i

# When checking position status:
if position is not None:
    # Let strategy handle TP/SL internally via generate_signal
    # The strategy returns 'CLOSE' when TP/SL is hit
    pass
```

### Step 4.2: Add Trade Analysis by Signal Type

```python
# In BacktestResult, add:
@dataclass
class BacktestResult:
    # ... existing fields ...
    trades_by_signal_type: dict = field(default_factory=dict)
    trades_by_exit_reason: dict = field(default_factory=dict)
```

---

## Phase 5: Execution Plan

### 5.1: File Structure to Create/Modify

```
btc_volatility_predictor/
├── INSTRUCTIONS_V2.md              # This file
├── fetch_extended_data.py          # NEW: Fetch 365 days
├── data/
│   ├── raw/
│   │   └── btcusdt_1h_365d.csv    # NEW: Extended data
│   └── processed/
│       └── features_365d.csv       # NEW: Extended features
├── train_regime_extended.py        # NEW: Train with 90-day test
├── train_direction.py              # NEW: Direction classifier
├── checkpoints/
│   ├── best_regime_model_90d.pt   # NEW: Extended model
│   └── best_direction_model.pt    # NEW: Direction model
├── backtest/
│   ├── strategies/
│   │   ├── mean_reversion_v2.py   # NEW: Proven approach
│   │   ├── defensive.py           # NEW: Risk-off toggle
│   │   └── direction_aware.py     # NEW: Combined predictor
│   ├── engine.py                   # MODIFY: Better TP/SL handling
│   └── run_backtest_v2.py         # NEW: Extended backtest runner
└── combined_predictor.py           # NEW: Vol + Direction
```

### 5.2: Execution Commands (In Order)

```bash
# ============================================
# PHASE 1: DATA & MODEL
# ============================================

# Step 1: Fetch extended data (365 days)
cd btc_volatility_predictor
python fetch_extended_data.py

# Step 2: Engineer features
python -c "
from data.features import prepare_dataset
prepare_dataset('data/raw/btcusdt_1h_365d.csv', 'data/processed/features_365d.csv')
"

# Step 3: Train regime model with 90-day test
python train_regime_extended.py

# ============================================
# PHASE 2: DIRECTION MODEL (Optional but recommended)
# ============================================

# Step 4: Train direction classifier
python train_direction.py

# ============================================
# PHASE 3: BACKTEST
# ============================================

# Step 5: Run extended backtest
python backtest/run_backtest_v2.py

# ============================================
# PHASE 4: ANALYZE
# ============================================

# Results will be in backtest/results_v2/
```

---

## Phase 6: Expected Outcomes

### Based on `vol_filtered_mean_reversion_v2.py` Results

The proven approach achieved:
- **Unfiltered**: ~50-60% win rate
- **Vol-Filtered (LOW only)**: ~69% win rate
- **Return improvement**: From ~0% to +39.90%
- **Sharpe improvement**: From ~0 to 9.73

### What We Expect to See

| Strategy | Expected Outcome |
|----------|-----------------|
| MeanReversionV2 (LOW only) | +10-30% return, 60-70% win rate |
| Defensive (risk-off toggle) | Beat buy-and-hold with lower DD |
| DirectionAware | Highest Sharpe if direction model works |
| OLD strategies (HIGH vol) | Still negative (confirms our learning) |

### Success Criteria

1. **MeanReversionV2 is profitable** (>0% return over 90 days)
2. **Sharpe ratio > 1.0** for best strategy
3. **Max drawdown < 15%**
4. **At least 50+ trades** for statistical significance
5. **Vol filtering shows clear improvement** over unfiltered

---

## Phase 7: Key Code Snippets

### 7.1: Modified Data Split (90 Days Test)

```python
# train_regime_extended.py

def create_regime_dataloaders_extended(
    data_path="data/processed/features_365d.csv",
    window_size=48,
    batch_size=32,
    test_days=90,        # CHANGED: 90 days = 2160 hours
    val_ratio=0.15,
    percentile=50
):
    """Create dataloaders with 90-day test period."""
    df = pd.read_csv(data_path)
    
    n_total = len(df)
    n_test = test_days * 24  # 2160 hours
    n_trainval = n_total - n_test
    n_val = int(n_trainval * 0.15)
    n_train = n_trainval - n_val
    
    print(f"Total: {n_total} samples ({n_total/24:.0f} days)")
    print(f"Train: {n_train} ({n_train/24:.0f} days)")
    print(f"Val: {n_val} ({n_val/24:.0f} days)")
    print(f"Test: {n_test} ({n_test/24:.0f} days)")  # 90 days
    
    # ... rest same as original
```

### 7.2: Direction Classifier Training

```python
# train_direction.py

"""
Train direction classifier: Will next hour be UP or DOWN?
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from config import Config
from models import SPHNet

def create_direction_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create binary direction labels.
    1 = next close > current close (UP)
    0 = next close <= current close (DOWN)
    """
    next_close = df['close'].shift(-1)
    current_close = df['close']
    direction = (next_close > current_close).astype(int)
    return direction.values

def main():
    config = Config()
    config.data_path = "data/processed/features_365d.csv"
    
    # Load data
    df = pd.read_csv(config.data_path)
    
    # Create direction labels
    df['target_direction_binary'] = create_direction_labels(df)
    
    # ... rest similar to train_regime.py but predicting direction
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, "checkpoints/best_direction_model.pt")

if __name__ == "__main__":
    main()
```

### 7.3: Combined Predictor

```python
# combined_predictor.py

"""
Combined volatility regime + direction predictor.
"""

import torch
from config import Config
from models import SPHNet

class CombinedPredictor:
    """
    Predicts both volatility regime and price direction.
    
    Returns: {
        'vol_regime': 'HIGH' or 'LOW',
        'direction': 'UP' or 'DOWN',
        'combined': 'LOW_UP', 'LOW_DOWN', 'HIGH_UP', or 'HIGH_DOWN',
        'vol_prob': float,
        'dir_prob': float
    }
    """
    
    def __init__(
        self,
        vol_checkpoint="checkpoints/best_regime_model_90d.pt",
        dir_checkpoint="checkpoints/best_direction_model.pt",
        device='cpu'
    ):
        self.device = torch.device(device)
        
        # Load volatility model
        self.vol_model = self._load_model(vol_checkpoint)
        
        # Load direction model
        self.dir_model = self._load_model(dir_checkpoint)
    
    def _load_model(self, checkpoint_path):
        config = Config()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            for k, v in checkpoint['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)
        
        model = SPHNet(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    @torch.no_grad()
    def predict(self, prices: torch.Tensor, features: torch.Tensor) -> dict:
        """
        Predict both regime and direction.
        
        Args:
            prices: [1, window_size, n_price_features]
            features: [1, window_size, n_eng_features]
        """
        # Volatility prediction
        vol_out = self.vol_model(prices, features)
        vol_prob = torch.sigmoid(vol_out['direction_pred']).item()
        vol_regime = 'HIGH' if vol_prob > 0.5 else 'LOW'
        
        # Direction prediction
        dir_out = self.dir_model(prices, features)
        dir_prob = torch.sigmoid(dir_out['direction_pred']).item()
        direction = 'UP' if dir_prob > 0.5 else 'DOWN'
        
        return {
            'vol_regime': vol_regime,
            'direction': direction,
            'combined': f"{vol_regime}_{direction}",
            'vol_prob': vol_prob,
            'dir_prob': dir_prob,
            'vol_confidence': abs(vol_prob - 0.5) * 2,
            'dir_confidence': abs(dir_prob - 0.5) * 2
        }
```

---

## Appendix: Quick Reference

### A. Data Requirements

| Item | Current | Required |
|------|---------|----------|
| Data days | 173 | 365 |
| Test days | 10 | 90 |
| Test samples | 240 | 2160 |
| Min trades for significance | ~10 | ~100 |

### B. Strategy Decision Matrix

| Vol Regime | Direction | Action |
|------------|-----------|--------|
| LOW | UP | LONG mean reversion |
| LOW | DOWN | SHORT mean reversion |
| HIGH | UP | NO TRADE (or cautious trend follow) |
| HIGH | DOWN | NO TRADE (or exit existing) |

### C. Key Parameters to Test

```python
# Mean Reversion V2 parameters to grid search:
rsi_oversold = [25, 30, 35]
take_profit_pct = [0.015, 0.02, 0.025]
stop_loss_pct = [0.01, 0.015, 0.02]
max_holding_bars = [12, 24, 48]
```

### D. Risk Management Rules

1. **Max position size**: 100% (single trade at a time)
2. **Max daily loss**: -2% of capital → stop trading for day
3. **Max drawdown**: -15% → reduce position size to 50%
4. **Minimum bars between trades**: 4 hours

---

## Summary

**Do This**:
1. Fetch 365 days of data
2. Retrain with 90-day test split
3. Train separate direction model
4. Implement MeanReversionV2 (LOW vol only, fixed TP/SL)
5. Implement Defensive strategy (exit on HIGH vol)
6. Run extended backtest
7. Compare filtered vs unfiltered results

**Don't Do This**:
- Trade breakout/momentum in HIGH volatility
- Use indicator-based exits without TP/SL
- Test on < 30 days (not statistically significant)
- Ignore the proven approach from vol_filtered_mean_reversion_v2.py

**Expected Result**: 
- MeanReversionV2 should show +10-40% return over 90 days
- Defensive strategy should beat buy-and-hold with lower drawdown
- Clear evidence that vol filtering adds value

---

*Last Updated: Created for Claude Code Opus Implementation*
