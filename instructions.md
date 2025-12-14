# TrendStrengthML_50 Strategy Robustness Testing Instructions

## Executive Summary

After XGBoost label training, the **TrendStrengthML_50** strategy achieved:
- **Win Rate: 83.3%** (+33.3pp improvement over baseline)
- **Trades: 6** 
- **Return: +4.30%**

**CRITICAL**: This was tested on a **bear market period only**. To validate robustness, we need comprehensive testing across multiple market cycles with non-overlapping training/validation data and random date sampling.

---

## Objective

Validate TrendStrengthML_50 robustness by testing across:
1. **1 additional bear market period** (2018 Crypto Winter)
2. **2 bull market periods** (2019-2020 Recovery, 2023-2024 Bull Run)
3. **Non-overlapping training/validation periods**
4. **Random starting/ending dates** to ensure strategy isn't overfit to specific timeframes

---

## Bitcoin Historical Market Cycles

### Data Availability
- **Binance BTC/USDT trade data**: Available from **August 17, 2017**
- **Binance BTC/USDT hourly klines**: Available from **December 18, 2017**
- **Source**: Binance API (free, no API key required) or https://data.binance.vision

### Key Market Cycles

#### Bear Markets (Drawdown > 70%)
| Period | Start Date | End Date | Peak Price | Bottom Price | Drawdown |
|--------|------------|----------|------------|--------------|----------|
| 2018 Crypto Winter | Jan 8, 2018 | Dec 15, 2018 | $19,783 | $3,122 | -84% |
| 2022 Bear Market | Nov 10, 2021 | Nov 21, 2022 | $69,000 | $15,476 | -78% |
| COVID Crash | Feb 13, 2020 | Mar 13, 2020 | $10,500 | $3,850 | -63% |

#### Bull Markets (Major Rallies)
| Period | Start Date | End Date | Bottom Price | Peak Price | Gain |
|--------|------------|----------|--------------|------------|------|
| 2019 Recovery | Dec 15, 2018 | Jun 26, 2019 | $3,122 | $13,880 | +345% |
| 2020-2021 Bull | Mar 13, 2020 | Nov 10, 2021 | $3,850 | $69,000 | +1,692% |
| 2023-2024 Bull | Nov 21, 2022 | Mar 14, 2024 | $15,476 | $73,750 | +376% |
| 2024+ Bull | Aug 5, 2024 | Present | $49,000 | $103,000+ | +110%+ |

#### Halving Events (Key Cycle Markers)
- July 9, 2016: Block reward 25 → 12.5 BTC
- May 11, 2020: Block reward 12.5 → 6.25 BTC
- April 19, 2024: Block reward 6.25 → 3.125 BTC

---

## Recommended Test Periods

### Period 1: 2018 Bear Market (REQUIRED)
- **Type**: Bear Market
- **Date Range**: January 1, 2018 → December 31, 2018
- **Training/Val Split**: Jan-Aug 2018 (train), Sep-Dec 2018 (test)
- **Characteristics**: -84% drawdown, post-ICO crash, high volatility

### Period 2: 2019-2020 Recovery (REQUIRED - Bull Market #1)
- **Type**: Bull Market / Recovery
- **Date Range**: January 1, 2019 → December 31, 2020
- **Training/Val Split**: Jan 2019 - Aug 2020 (train), Sep-Dec 2020 (test)
- **Characteristics**: Recovery from bottom, includes COVID crash and bounce

### Period 3: 2023-2024 Bull Market (REQUIRED - Bull Market #2)
- **Type**: Bull Market
- **Date Range**: January 1, 2023 → October 31, 2024
- **Training/Val Split**: Jan 2023 - Apr 2024 (train), May-Oct 2024 (test)
- **Characteristics**: ETF approval rally, new ATH, halving event

### Period 4: Current Period (CONTROL - Already Tested)
- **Date Range**: November 2024 onwards (90-day test)
- **Status**: Already tested - baseline results

---

## Implementation Plan

### Phase 1: Data Infrastructure

#### Task 1.1: Extended Data Fetcher
Create `fetch_historical_periods.py`:

```python
"""
Fetch historical BTC/USDT data for multiple market cycles.

Periods to fetch:
1. 2018 Bear: Jan 2018 - Dec 2018 (366 days)
2. 2019-2020 Recovery: Jan 2019 - Dec 2020 (731 days)
3. 2023-2024 Bull: Jan 2023 - Nov 2024 (700 days)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

PERIODS = {
    '2018_bear': {
        'start': '2018-01-01',
        'end': '2018-12-31',
        'type': 'bear'
    },
    '2019_2020_recovery': {
        'start': '2019-01-01',
        'end': '2020-12-31',
        'type': 'bull'
    },
    '2023_2024_bull': {
        'start': '2023-01-01',
        'end': '2024-10-31',
        'type': 'bull'
    }
}

def fetch_binance_klines(start_date: str, end_date: str, 
                         symbol: str = "BTCUSDT", interval: str = "1h"):
    """Fetch klines from Binance API with pagination."""
    base_url = "https://api.binance.com/api/v3/klines"
    
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
            
        data = response.json()
        if not data:
            break
            
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.1)  # Rate limiting
        
    # Convert to DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
               'taker_buy_quote', 'ignore']
    
    df = pd.DataFrame(all_klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)
        
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'trades', 'quote_volume']]

def fetch_all_periods():
    """Fetch all required historical periods."""
    os.makedirs("data/raw/historical", exist_ok=True)
    
    for period_name, config in PERIODS.items():
        print(f"Fetching {period_name}...")
        df = fetch_binance_klines(config['start'], config['end'])
        
        output_path = f"data/raw/historical/btcusdt_1h_{period_name}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  Saved {len(df)} candles to {output_path}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    fetch_all_periods()
```

#### Task 1.2: Feature Engineering for Historical Data
Extend `data/features.py` to process all historical periods:

```python
def prepare_all_historical_datasets():
    """Prepare feature datasets for all historical periods."""
    from data.features import prepare_dataset
    
    periods = ['2018_bear', '2019_2020_recovery', '2023_2024_bull']
    
    for period in periods:
        input_path = f"data/raw/historical/btcusdt_1h_{period}.csv"
        output_path = f"data/processed/historical/features_{period}.csv"
        
        if os.path.exists(input_path):
            prepare_dataset(input_path, output_path)
```

---

### Phase 2: Multi-Period Backtesting Framework

#### Task 2.1: Create `backtest/run_backtest_multiperiod.py`

```python
#!/usr/bin/env python3
"""
Multi-Period Robustness Testing for TrendStrengthML_50

Tests the strategy across multiple market cycles:
1. 2018 Bear Market
2. 2019-2020 Recovery/Bull
3. 2023-2024 Bull Market
4. Random date sampling within each period

Success Criteria:
- Consistent positive returns across ALL market types
- Win rate > 60% in each period
- Sharpe ratio > 0.5 in each period
- Max drawdown < 20% in each period
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.strategies import TrendStrengthWithML, TrendStrengthStrategy

# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

PERIODS = {
    '2018_bear': {
        'path': 'data/processed/historical/features_2018_bear.csv',
        'type': 'bear',
        'description': '2018 Crypto Winter (-84% drawdown)'
    },
    '2019_2020_recovery': {
        'path': 'data/processed/historical/features_2019_2020_recovery.csv',
        'type': 'bull',
        'description': '2019-2020 Recovery (+345% then +1692%)'
    },
    '2023_2024_bull': {
        'path': 'data/processed/historical/features_2023_2024_bull.csv',
        'type': 'bull',
        'description': '2023-2024 Bull Market (ETF Rally, +376%)'
    }
}

def generate_random_windows(df: pd.DataFrame, n_windows: int = 10, 
                            min_days: int = 60, max_days: int = 120) -> list:
    """
    Generate random non-overlapping date windows for testing.
    
    Args:
        df: DataFrame with data
        n_windows: Number of random windows to generate
        min_days: Minimum window size in days
        max_days: Maximum window size in days
        
    Returns:
        List of (start_idx, end_idx) tuples
    """
    n_samples = len(df)
    min_samples = min_days * 24
    max_samples = max_days * 24
    
    windows = []
    used_indices = set()
    
    attempts = 0
    while len(windows) < n_windows and attempts < n_windows * 10:
        # Random window size
        window_size = random.randint(min_samples, max_samples)
        
        # Random start point (leave room for window)
        max_start = n_samples - window_size - 168  # Leave warmup
        if max_start < 168:
            break
            
        start_idx = random.randint(168, max_start)
        end_idx = start_idx + window_size
        
        # Check for overlap
        window_set = set(range(start_idx, end_idx))
        if not window_set.intersection(used_indices):
            windows.append((start_idx, end_idx))
            used_indices.update(window_set)
            
        attempts += 1
    
    return windows

def run_period_backtest(period_name: str, config: dict) -> dict:
    """
    Run comprehensive backtest for a single period.
    
    Returns:
        Dict with results for full period and random windows
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {period_name.upper()}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    if not os.path.exists(config['path']):
        print(f"ERROR: Data not found at {config['path']}")
        return None
    
    df = pd.read_csv(config['path'])
    print(f"Loaded {len(df)} samples ({len(df)/24:.0f} days)")
    
    # Generate volatility predictions if not present
    if 'predicted_regime' not in df.columns:
        # Use median volatility as threshold
        vol_col = 'target_volatility' if 'target_volatility' in df.columns else 'vol_realized_24h'
        if vol_col in df.columns:
            threshold = df[vol_col].median()
            df['predicted_regime'] = (df[vol_col] > threshold).astype(int)
        else:
            df['predicted_regime'] = 0
    
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )
    
    results = {
        'period': period_name,
        'type': config['type'],
        'full_period': None,
        'random_windows': []
    }
    
    # === Test 1: Full Period ===
    print("\n--- Full Period Test ---")
    
    strategies = [
        TrendStrengthStrategy(),  # Baseline
        TrendStrengthWithML(ml_threshold=0.5),  # Our target strategy
    ]
    
    full_results = []
    for strategy in strategies:
        result = engine.run(strategy, df)
        full_results.append({
            'strategy': strategy.name,
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'trades': result.num_trades,
            'win_rate': result.win_rate if result.num_trades > 0 else 0
        })
        
        print(f"  {strategy.name}: Return={result.total_return*100:+.2f}%, "
              f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.num_trades}, "
              f"WinRate={result.win_rate*100:.1f}%")
    
    results['full_period'] = full_results
    
    # === Test 2: Random Windows ===
    print("\n--- Random Window Tests (10 windows) ---")
    
    windows = generate_random_windows(df, n_windows=10, min_days=60, max_days=90)
    
    for i, (start_idx, end_idx) in enumerate(windows):
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Run TrendStrengthML_50 on this window
        strategy = TrendStrengthWithML(ml_threshold=0.5)
        result = engine.run(strategy, window_df)
        
        window_result = {
            'window': i + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'days': (end_idx - start_idx) / 24,
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'trades': result.num_trades,
            'win_rate': result.win_rate if result.num_trades > 0 else 0
        }
        results['random_windows'].append(window_result)
        
        print(f"  Window {i+1}: {(end_idx-start_idx)/24:.0f}d, "
              f"Return={result.total_return*100:+.2f}%, "
              f"Trades={result.num_trades}, WinRate={result.win_rate*100:.1f}%")
    
    return results

def run_cross_validation(n_folds: int = 5):
    """
    Run k-fold cross-validation across all data.
    
    Combines all historical data and splits into folds.
    Each fold tests on different time periods.
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION TEST")
    print("="*60)
    
    # TODO: Implement time-series cross-validation
    # - Split each period into train/test
    # - Train XGBoost on train portion
    # - Test on test portion
    # - Aggregate results
    
    pass

def main():
    """Run multi-period robustness testing."""
    print("="*60)
    print("TRENDSTRENGTHML_50 ROBUSTNESS TESTING")
    print("Multi-Period Analysis")
    print("="*60)
    
    all_results = {}
    
    # Run tests for each period
    for period_name, config in PERIODS.items():
        results = run_period_backtest(period_name, config)
        if results:
            all_results[period_name] = results
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL PERIODS")
    print("="*60)
    
    summary_data = []
    for period_name, results in all_results.items():
        if results and results['full_period']:
            ml_result = next((r for r in results['full_period'] 
                            if 'ML' in r['strategy']), None)
            if ml_result:
                summary_data.append({
                    'Period': period_name,
                    'Type': results['type'],
                    'Return': ml_result['return'],
                    'Sharpe': ml_result['sharpe'],
                    'MaxDD': ml_result['max_dd'],
                    'Trades': ml_result['trades'],
                    'WinRate': ml_result['win_rate']
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Aggregate stats
        print(f"\n--- Aggregate Statistics ---")
        print(f"Average Return: {summary_df['Return'].mean()*100:+.2f}%")
        print(f"Average Sharpe: {summary_df['Sharpe'].mean():.2f}")
        print(f"Average Win Rate: {summary_df['WinRate'].mean()*100:.1f}%")
        print(f"Worst Drawdown: {summary_df['MaxDD'].max()*100:.1f}%")
        
        # Success criteria check
        print("\n--- Success Criteria Check ---")
        all_positive = all(r['Return'] > 0 for r in summary_data)
        all_high_wr = all(r['WinRate'] > 0.6 for r in summary_data)
        all_good_sharpe = all(r['Sharpe'] > 0.5 for r in summary_data)
        all_low_dd = all(r['MaxDD'] < 0.20 for r in summary_data)
        
        print(f"  All periods positive: {'PASS' if all_positive else 'FAIL'}")
        print(f"  All periods WR > 60%: {'PASS' if all_high_wr else 'FAIL'}")
        print(f"  All periods Sharpe > 0.5: {'PASS' if all_good_sharpe else 'FAIL'}")
        print(f"  All periods MaxDD < 20%: {'PASS' if all_low_dd else 'FAIL'}")
    
    # Save results
    os.makedirs("backtest/results_multiperiod", exist_ok=True)
    with open("backtest/results_multiperiod/robustness_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to backtest/results_multiperiod/robustness_results.json")

if __name__ == "__main__":
    main()
```

---

### Phase 3: Walk-Forward Optimization

#### Task 3.1: Implement Rolling Window Training

For each test period, implement:
1. **Train XGBoost on first 70% of period**
2. **Validate on middle 15%**
3. **Test on final 15%**

This ensures truly non-overlapping training/test data.

```python
def walk_forward_test(df: pd.DataFrame, period_name: str):
    """
    Walk-forward analysis with rolling XGBoost training.
    
    Split:
    - Train: First 70% of data
    - Validate: Next 15% of data
    - Test: Final 15% of data
    """
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    print(f"Walk-Forward Split for {period_name}:")
    print(f"  Train: {n_train} samples ({n_train/24:.0f} days)")
    print(f"  Val: {n_val} samples ({n_val/24:.0f} days)")
    print(f"  Test: {n_test} samples ({n_test/24:.0f} days)")
    
    # Step 1: Generate trade labels from training data
    # Step 2: Train XGBoost on training labels
    # Step 3: Validate threshold selection on val data
    # Step 4: Final test on test data
    
    return {
        'train_samples': n_train,
        'val_samples': n_val,
        'test_samples': n_test
    }
```

---

### Phase 4: Statistical Significance Testing

#### Task 4.1: Bootstrap Confidence Intervals

```python
def bootstrap_confidence_interval(returns: list, n_bootstrap: int = 1000, 
                                   confidence: float = 0.95) -> tuple:
    """
    Calculate bootstrap confidence interval for returns.
    """
    import numpy as np
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return lower, upper
```

#### Task 4.2: Monte Carlo Permutation Test

```python
def permutation_test(strategy_returns: list, baseline_returns: list, 
                     n_permutations: int = 10000) -> float:
    """
    Test if strategy significantly outperforms baseline.
    
    Returns:
        p-value for null hypothesis that strategy = baseline
    """
    observed_diff = np.mean(strategy_returns) - np.mean(baseline_returns)
    
    combined = strategy_returns + baseline_returns
    n_strategy = len(strategy_returns)
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_strategy = combined[:n_strategy]
        perm_baseline = combined[n_strategy:]
        perm_diff = np.mean(perm_strategy) - np.mean(perm_baseline)
        
        if perm_diff >= observed_diff:
            count_extreme += 1
    
    return count_extreme / n_permutations
```

---

## Success Criteria

### Primary Metrics (ALL must pass):
| Metric | Threshold | Description |
|--------|-----------|-------------|
| Positive Returns | ALL periods | Strategy must be profitable in bear AND bull markets |
| Win Rate | > 60% | Consistent edge across market conditions |
| Sharpe Ratio | > 0.5 | Risk-adjusted returns acceptable |
| Max Drawdown | < 20% | Manageable risk profile |

### Secondary Metrics (Desirable):
| Metric | Target | Description |
|--------|--------|-------------|
| Trade Count | > 5 per 90 days | Sufficient sample size |
| Consistency | σ(returns) < 5% | Low variance across periods |
| Beat Baseline | > 0 pp | Improvement over TrendStrength |

---

## Execution Order

### Step 1: Data Collection (30 min)
```bash
cd btc_volatility_predictor
python fetch_historical_periods.py
```

### Step 2: Feature Engineering (15 min)
```bash
python -c "from data.features import prepare_dataset; \
  prepare_dataset('data/raw/historical/btcusdt_1h_2018_bear.csv', 'data/processed/historical/features_2018_bear.csv'); \
  prepare_dataset('data/raw/historical/btcusdt_1h_2019_2020_recovery.csv', 'data/processed/historical/features_2019_2020_recovery.csv'); \
  prepare_dataset('data/raw/historical/btcusdt_1h_2023_2024_bull.csv', 'data/processed/historical/features_2023_2024_bull.csv')"
```

### Step 3: Run Multi-Period Backtest (20 min)
```bash
python backtest/run_backtest_multiperiod.py
```

### Step 4: Review Results
- Check `backtest/results_multiperiod/robustness_results.json`
- Verify all success criteria are met
- If any period fails, investigate and potentially retrain

---

## Expected Outcomes

### If Strategy is Robust:
- Consistent positive returns across all market types
- Win rate remains above 60% in bear and bull markets
- XGBoost filter effectively removes losing trades in all conditions

### If Strategy is Overfit:
- Strong performance in one market type, weak in others
- Win rate drops significantly in different market conditions
- Returns correlation with specific market regime

### Action Items on Failure:
1. **Retrain XGBoost** with data from multiple market cycles
2. **Adjust ML threshold** (try 0.55, 0.6, 0.65)
3. **Review features** - some may be period-specific
4. **Ensemble approach** - train separate models for bull/bear detection

---

## File Structure

```
btc_volatility_predictor/
├── data/
│   ├── raw/
│   │   └── historical/
│   │       ├── btcusdt_1h_2018_bear.csv
│   │       ├── btcusdt_1h_2019_2020_recovery.csv
│   │       └── btcusdt_1h_2023_2024_bull.csv
│   └── processed/
│       └── historical/
│           ├── features_2018_bear.csv
│           ├── features_2019_2020_recovery.csv
│           └── features_2023_2024_bull.csv
├── backtest/
│   ├── run_backtest_v5.py (existing - baseline)
│   ├── run_backtest_multiperiod.py (NEW)
│   └── results_multiperiod/
│       └── robustness_results.json
└── ROBUSTNESS_TESTING_INSTRUCTIONS.md (this file)
```

---

## Notes for Implementation

1. **Rate Limiting**: Binance API has rate limits. Use `time.sleep(0.1)` between requests.

2. **Memory Management**: Large datasets (~700 days = 16,800 hourly candles). Process one period at a time.

3. **Random Seed**: Set `random.seed(42)` for reproducibility when generating random windows.

4. **Model Caching**: The XGBoost model at `checkpoints/trade_classifier.json` was trained on recent data. For true robustness, consider retraining on combined historical data.

5. **Feature Availability**: Ensure ADX and other indicators are calculated for historical periods. The existing `data/features.py` should handle this.

---

## References

- Current best model: `python backtest/run_backtest_v5.py`
- TrendStrengthML_50 implementation: `backtest/strategies/simple_adx.py`
- XGBoost training: `ml/train_xgboost.py`
- Feature engineering: `data/features.py`