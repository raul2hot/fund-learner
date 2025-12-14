# BTC Volatility Regime Strategy Backtester - Instructions for Claude Code

## Project Overview

We have trained a **HIGH/LOW volatility regime classifier** for BTC/USDT 1-hour candles with **80% test accuracy**. The goal is to backtest trading strategies that exploit these volatility predictions.

### Current State
- ✅ Model trained: `checkpoints/best_regime_model.pt`
- ✅ Test accuracy: 80.1% on 10 days (240 hours) of unseen data
- ✅ Features engineered: Technical indicators, volatility estimators
- ❌ Strategy backtesting: **NOT YET IMPLEMENTED**

### Key Clarification
The existing `inference.py` is for **LIVE prediction** only (fetches current data, predicts next hour). For backtesting, we need to generate predictions for the **entire test period** and simulate trades.

---

## Task: Create Strategy Backtesting Framework

### Step 1: Create Test Set Predictions Generator

Create `backtest/generate_predictions.py`:

```python
"""
Generate predictions for the entire test period.
This gives us a DataFrame with:
- timestamp
- OHLCV data
- predicted regime (HIGH/LOW)
- actual regime (HIGH/LOW)
- prediction probability
"""
```

**Requirements:**
1. Load the trained model from `checkpoints/best_regime_model.pt`
2. Load the processed features from `data/processed/features.csv`
3. Use the same train/val/test split as training (last 10 days = test)
4. Generate predictions for each hour in the test set
5. Save to `backtest/test_predictions.csv` with columns:
   - `timestamp` (if available, else index)
   - `open`, `high`, `low`, `close`, `volume`
   - `predicted_regime` (1=HIGH, 0=LOW)
   - `actual_regime` (1=HIGH, 0=LOW)
   - `prediction_prob` (probability of HIGH)
   - `correct` (boolean)

### Step 2: Create Strategy Base Class

Create `backtest/strategies/base.py`:

```python
"""
Base class for all trading strategies.
Each strategy should implement:
- generate_signals(row, prediction) -> 'BUY', 'SELL', or 'HOLD'
- get_params() -> dict of strategy parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Signal = Literal['BUY', 'SELL', 'HOLD', 'CLOSE']

@dataclass
class Position:
    entry_price: float
    entry_time: int
    direction: Literal['LONG', 'SHORT']
    size: float = 1.0

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.position: Position | None = None
    
    @abstractmethod
    def generate_signal(self, row: dict, predicted_regime: str, 
                        history: list[dict]) -> Signal:
        """Generate trading signal based on current data and prediction."""
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Return strategy parameters for logging."""
        pass
```

### Step 3: Implement Trading Strategies

Create these strategy files in `backtest/strategies/`:

#### 3.1 Mean Reversion Strategy (LOW volatility)
File: `backtest/strategies/mean_reversion.py`

**Logic:**
- ONLY trade when predicted regime = LOW
- Entry LONG: RSI < 30 AND price touches lower Bollinger Band
- Entry SHORT: RSI > 70 AND price touches upper Bollinger Band
- Exit: Price returns to middle BB (20-period SMA) OR RSI normalizes (40-60)
- Stop Loss: 1.5x ATR from entry

**Parameters to test:**
- RSI oversold threshold: [25, 30, 35]
- RSI overbought threshold: [65, 70, 75]
- BB period: [20]
- BB std: [2.0]

#### 3.2 Breakout Strategy (HIGH volatility)
File: `backtest/strategies/breakout.py`

**Logic:**
- ONLY trade when predicted regime = HIGH
- Entry LONG: Close breaks above 24h high WITH volume > 1.5x average
- Entry SHORT: Close breaks below 24h low WITH volume > 1.5x average
- Exit: Trailing stop at 2x ATR OR opposite signal
- Confirm with "momentum candle" (current candle range > 2x average)

**Parameters to test:**
- Lookback for high/low: [12, 24, 48]
- Volume multiplier: [1.5, 2.0]
- ATR trailing stop: [1.5, 2.0, 2.5]

#### 3.3 Momentum Strategy (HIGH volatility)
File: `backtest/strategies/momentum.py`

**Logic:**
- ONLY trade when predicted regime = HIGH
- Entry LONG: EMA(9) crosses above EMA(21) AND MACD histogram > 0
- Entry SHORT: EMA(9) crosses below EMA(21) AND MACD histogram < 0
- Exit: Opposite crossover OR 3x ATR trailing stop

**Parameters to test:**
- Fast EMA: [9, 12]
- Slow EMA: [21, 26]
- MACD confirmation: [True, False]

#### 3.4 Regime-Switching Strategy (Adaptive)
File: `backtest/strategies/regime_switch.py`

**Logic:**
- When predicted regime = LOW: Use Mean Reversion rules
- When predicted regime = HIGH: Use Momentum rules
- Close positions when regime switches (optional parameter)

#### 3.5 Baseline: Buy and Hold
File: `backtest/strategies/baseline.py`

**Logic:**
- Buy at start, hold until end
- Used as benchmark comparison

### Step 4: Create Backtesting Engine

Create `backtest/engine.py`:

```python
"""
Backtesting engine that:
1. Loads test predictions
2. Runs each strategy
3. Tracks positions, P&L, drawdown
4. Generates performance metrics
"""

@dataclass
class Trade:
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: str
    pnl: float
    pnl_pct: float
    regime_at_entry: str

@dataclass 
class BacktestResult:
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float
    profit_factor: float
    trades: list[Trade]
    equity_curve: list[float]
```

**Key features:**
- Transaction costs: 0.1% per trade (Binance taker fee)
- Slippage: 0.05% 
- No leverage (or configurable)
- Track equity curve for visualization

### Step 5: Create Analysis & Visualization

Create `backtest/analyze.py`:

```python
"""
Analyze and compare strategy performance:
1. Performance metrics table
2. Equity curves plot
3. Drawdown analysis
4. Trade distribution by regime
5. Statistical significance tests
"""
```

**Generate these outputs:**

1. **Performance Summary Table** (`backtest/results/summary.csv`):
   | Strategy | Return | Sharpe | MaxDD | WinRate | Trades | ProfitFactor |
   |----------|--------|--------|-------|---------|--------|--------------|
   | Buy&Hold | X% | X | X% | N/A | 1 | N/A |
   | MeanReversion | X% | X | X% | X% | X | X |
   | Breakout | X% | X | X% | X% | X | X |
   | Momentum | X% | X | X% | X% | X | X |
   | RegimeSwitch | X% | X | X% | X% | X | X |

2. **Equity Curves Plot** (`backtest/results/equity_curves.png`):
   - All strategies on same chart
   - Shaded regions for HIGH/LOW volatility periods

3. **Regime Analysis** (`backtest/results/regime_analysis.png`):
   - Performance breakdown by predicted regime
   - Bar chart: Returns in HIGH vs LOW periods per strategy

4. **Trade Analysis** (`backtest/results/trades_analysis.png`):
   - Distribution of trade P&L
   - Win/Loss by regime
   - Holding period distribution

### Step 6: Create Main Runner

Create `backtest/run_backtest.py`:

```python
"""
Main entry point:
1. Generate predictions if not exists
2. Run all strategies
3. Generate analysis
4. Save results
"""

if __name__ == "__main__":
    # Config
    INITIAL_CAPITAL = 10000
    TRANSACTION_COST = 0.001  # 0.1%
    
    # Run
    results = run_all_strategies()
    analyze_results(results)
    generate_report()
```

---

## File Structure to Create

```
btc_volatility_predictor/
├── backtest/
│   ├── __init__.py
│   ├── generate_predictions.py    # Step 1
│   ├── engine.py                  # Step 4
│   ├── analyze.py                 # Step 5
│   ├── run_backtest.py           # Step 6 (main entry)
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py               # Step 2
│   │   ├── baseline.py           # Step 3.5
│   │   ├── mean_reversion.py     # Step 3.1
│   │   ├── breakout.py           # Step 3.2
│   │   ├── momentum.py           # Step 3.3
│   │   └── regime_switch.py      # Step 3.4
│   └── results/                  # Output directory
│       ├── summary.csv
│       ├── test_predictions.csv
│       ├── equity_curves.png
│       └── trades/               # Individual trade logs
└── INSTRUCTIONS.md               # This file
```

---

## Technical Details from Existing Codebase

### Data Split (from train_regime.py)
```python
n_test = 10 * 24  # 240 hours = 10 days
n_trainval = n_total - n_test
n_val = int(n_trainval * 0.15)
n_train = n_trainval - n_val
```

### Feature Columns Available (from dataset.py)
```python
PRICE_COLS = ['open', 'high', 'low', 'close', 'volume', 'log_return',
              'hl_range', 'oc_range', 'log_return_abs']

ENGINEERED_COLS = [
    # Volatility features
    'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h', 'vol_rs_1h', 'vol_rs_24h',
    'vol_yz_24h', 'vol_realized_24h', 'vol_of_vol',
    # Momentum
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h', 'momentum_48h',
    # Technical indicators (THESE ARE KEY FOR STRATEGIES)
    'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24', 'macd_hist',
    # Volume features
    'volume_ma_ratio', 'volume_change', 'obv_momentum', 'vwap_deviation',
    # Time features
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
```

### Loading the Trained Model
```python
checkpoint = torch.load("checkpoints/best_regime_model.pt", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
vol_threshold = checkpoint.get('vol_threshold')  # Median volatility threshold
```

### Scalers Location
After training, scalers should be saved. If not, recreate from training data:
```python
# From train_regime.py
price_scaler = train_dataset.price_scaler
feature_scaler = train_dataset.feature_scaler
```

---

## Important Considerations

### 1. Look-Ahead Bias
- Strategies should ONLY use data available at decision time
- RSI, BB, etc. should be calculated up to current bar (not including it)
- Predictions are for NEXT hour, so use prediction[t] to trade at close[t]

### 2. Realistic Assumptions
- Entry at CLOSE of signal bar (conservative)
- Or entry at OPEN of next bar (more realistic)
- Include transaction costs (0.1% per trade)
- Include slippage (0.05%)

### 3. Position Sizing
- Start simple: 100% capital per trade
- Advanced: Kelly criterion or volatility-adjusted sizing

### 4. Statistical Validation
- 240 hours is limited sample size
- Report confidence intervals
- Consider: Would results hold with 10% worse prediction accuracy?

---

## Success Criteria

The backtesting framework is successful if:

1. **Regime-aware strategies outperform buy-and-hold** (or at least have better risk-adjusted returns)

2. **Mean Reversion works better in LOW volatility periods** than HIGH

3. **Breakout/Momentum works better in HIGH volatility periods** than LOW

4. **Regime-switching strategy captures best of both worlds**

5. **80% prediction accuracy translates to positive edge** (this is the key question!)

---

## Execution Order

Run these commands in sequence:

```bash
# 1. Ensure data and model exist
python data/fetch_binance.py         # If raw data missing
python data/features.py              # If features missing
python train_regime.py               # If model missing

# 2. Run backtest
cd btc_volatility_predictor
python backtest/run_backtest.py

# 3. Results will be in backtest/results/
```

---

## Questions to Answer After Backtesting

1. Does the 80% accuracy translate to profitable strategies?
2. Which strategy has the best Sharpe ratio?
3. How much of the edge comes from regime prediction vs the base strategy?
4. What's the maximum drawdown traders should expect?
5. How many trades does each strategy generate?
6. Is there enough statistical significance with 240 hours of data?

---

## Extension Ideas (Future Work)

1. **Walk-forward optimization**: Retrain model periodically
2. **Multi-timeframe**: Combine 1h predictions with 4h trend
3. **Position sizing**: Kelly criterion based on prediction confidence
4. **Ensemble strategies**: Weight strategies by recent performance
5. **Live paper trading**: Connect to Binance testnet

---

*Last updated: Generated for Claude Code implementation*
