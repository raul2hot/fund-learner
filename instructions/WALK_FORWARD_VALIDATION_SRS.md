# Walk-Forward Validation - Software Requirements Specification

**Document Version:** 1.0  
**Created:** 2025-12-16  
**For:** Claude Code Opus  
**Project:** fund-learner  
**Priority:** CRITICAL - Make or Break Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Data Gap Analysis](#2-critical-data-gap-analysis)
3. [Phase 1: Data Acquisition](#3-phase-1-data-acquisition)
4. [Phase 2: Walk-Forward Implementation](#4-phase-2-walk-forward-implementation)
5. [Phase 3: Execution & Reporting](#5-phase-3-execution--reporting)
6. [Technical Specifications](#6-technical-specifications)
7. [Validation Checklist](#7-validation-checklist)
8. [Appendix: File Specifications](#appendix-file-specifications)

---

## 1. Executive Summary

### 1.1 Objective

Implement rigorous walk-forward validation to definitively answer:

> **Does the two-stage trading classifier generalize across different market regimes, or did we overfit to specific patterns?**

### 1.2 Current Model Performance (2025 Test Period)

| Metric | Value |
|--------|-------|
| Total Return | +32.69% |
| Trade Frequency | 3.92% (320 trades / 8,162 candles) |
| Win Rate | 52.81% |
| Sharpe Ratio | 1.89 (trade-adjusted) |

### 1.3 Success Criteria

| Grade | Requirement |
|-------|-------------|
| **PASS** | ≥3 of 4 primary periods profitable, no period < -20%, avg Sharpe > 0.8 |
| **FAIL** | <2 periods profitable OR any period < -30% |

### 1.4 Constraints

- **NO hyperparameter tuning** on test periods
- **NO cherry-picking** results
- **NO future data leakage** in normalization or features
- **IDENTICAL model configuration** across all periods

---

## 2. Data Availability - CONFIRMED ✅

### 2.1 Data File Status

**BTCUSDT_ml_data.parquet:**
```
Shape:      54,984 rows × 19 columns
Period:     2019-09-08 to 2025-12-15 (~6.3 years)
Status:     ✅ READY FOR WALK-FORWARD VALIDATION
```

### 2.2 Data Quality Notes

```
NaN Count: 36 rows (at dataset boundaries)
- First ~17 rows: 2019-09-08 (before Binance Futures data)
- Last ~19 rows: 2025-12-15 (future timestamps)
- Action: Drop rows with NaN in OHLCV columns (handled in pipeline)
```

### 2.3 Data Requirements by Test Period

| Period | Test Range | Min Train Candles | Data Available? |
|--------|------------|-------------------|-----------------|
| COVID Crash | 2020-03 to 2020-05 | ~4,300 | ✅ YES |
| May 2021 Crash | 2021-05 to 2021-07 | ~14,500 | ✅ YES |
| Luna/3AC | 2022-05 to 2022-07 | ~23,000 | ✅ YES |
| FTX Crash | 2022-11 to 2023-01 | ~27,500 | ✅ YES |
| ETF Rally | 2024-01 to 2024-03 | ~38,000 | ✅ YES |

### 2.4 Available Columns

```python
['timestamp', 'open', 'high', 'low', 'close', 'volume', 
 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
 'funding_rate', 'fear_greed_value', 'taker_buy_ratio', 'taker_sell_volume',
 'order_flow_imbalance', 'spread_proxy', 'hour', 'day_of_week', 'session']
```

---

## 3. Phase 1: Data Validation (Quick Check)

### 3.1 Pre-Validation Script

**Script:** `scripts/validate_wfv_data.py`

```python
"""
Quick validation that data is ready for walk-forward validation.
Run this ONCE before starting validation.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

def main():
    print("=" * 60)
    print("WALK-FORWARD DATA VALIDATION")
    print("=" * 60)
    
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Drop NaN rows
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    df_clean = df.dropna(subset=critical_cols)
    
    print(f"\nTotal rows:     {len(df):,}")
    print(f"After NaN drop: {len(df_clean):,}")
    print(f"Date range:     {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    
    # Check coverage for each period
    periods = [
        ('COVID', '2020-03-01', 4000),
        ('May 2021', '2021-05-01', 14000),
        ('Luna/3AC', '2022-05-01', 22000),
        ('FTX', '2022-11-01', 27000),
        ('ETF Rally', '2024-01-01', 37000),
    ]
    
    print("\nPeriod Coverage Check:")
    print("-" * 50)
    all_ok = True
    for name, start_date, min_train in periods:
        train_data = df_clean[df_clean['timestamp'] < start_date]
        if len(train_data) >= min_train:
            print(f"  {name:<12} ✅ {len(train_data):,} candles (need {min_train:,})")
        else:
            print(f"  {name:<12} ❌ {len(train_data):,} candles (need {min_train:,})")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ DATA READY - Proceed to walk_forward_validation.py")
    else:
        print("❌ DATA INSUFFICIENT - Check data pipeline")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 4. Phase 2: Walk-Forward Implementation

### 4.1 File Structure

```
fund-learner/
├── scripts/
│   └── walk_forward_validation.py    # Main script (CREATE)
│
├── experiments/
│   └── walk_forward/                  # Output directory
│       ├── period_0_covid/
│       │   ├── model.pt
│       │   ├── train_metrics.json
│       │   ├── test_results.json
│       │   └── predictions.csv
│       ├── period_1_may2021/
│       ├── period_2_luna/
│       ├── period_3_ftx/
│       ├── period_4_etf/
│       └── summary_report.json
│
├── data_pipleine/ml_data/
│   └── BTCUSDT_ml_data.parquet       # Full dataset (REQUIRED)
│
└── prepared_data/                     # Will be created per-period
```

### 4.2 Walk-Forward Validation Script

**File:** `scripts/walk_forward_validation.py`

**Requirements:**
1. Load full dataset from `data_pipleine/ml_data/BTCUSDT_ml_data.parquet`
2. For each test period:
   - Split data chronologically (train uses all data before test period)
   - Apply labeling (using `labeling/candle_classifier.py`)
   - Apply feature engineering (using `features/feature_pipeline.py`)
   - Normalize features using ONLY training data statistics
   - Train fresh two-stage model
   - Evaluate on test period
   - Record all metrics without modification
3. Generate summary report
4. NO hyperparameter tuning between periods

### 4.3 Test Period Definitions

```python
TEST_PERIODS = {
    'period_0_covid': {
        'name': 'COVID Crash',
        'train_end': '2020-02-29 23:59:59',
        'test_start': '2020-03-01 00:00:00',
        'test_end': '2020-05-31 23:59:59',
        'is_primary': False,  # Limited training data
        'description': 'COVID-19 market crash, -50% in days'
    },
    'period_1_may2021': {
        'name': 'May 2021 Crash',
        'train_end': '2021-04-30 23:59:59',
        'test_start': '2021-05-01 00:00:00',
        'test_end': '2021-07-31 23:59:59',
        'is_primary': True,
        'description': 'China mining ban, Elon tweets, -53% correction'
    },
    'period_2_luna': {
        'name': 'Luna/3AC Collapse',
        'train_end': '2022-04-30 23:59:59',
        'test_start': '2022-05-01 00:00:00',
        'test_end': '2022-07-31 23:59:59',
        'is_primary': True,
        'description': 'UST depeg, Luna death spiral, 3AC liquidation'
    },
    'period_3_ftx': {
        'name': 'FTX Crash',
        'train_end': '2022-10-31 23:59:59',
        'test_start': '2022-11-01 00:00:00',
        'test_end': '2023-01-31 23:59:59',
        'is_primary': True,
        'description': 'FTX insolvency, exchange collapse, -25% drop'
    },
    'period_4_etf': {
        'name': 'ETF Rally',
        'train_end': '2023-12-31 23:59:59',
        'test_start': '2024-01-01 00:00:00',
        'test_end': '2024-03-31 23:59:59',
        'is_primary': True,
        'description': 'Bitcoin ETF approval, institutional inflow'
    }
}
```

### 4.4 Model Configuration (FROZEN)

```python
MODEL_CONFIG = {
    # Architecture (DO NOT CHANGE)
    'model_type': 'two_stage',
    'd_model': 64,
    'n_heads': 4,
    'n_encoder_layers': 2,
    'dropout': 0.2,
    'window_size': 64,
    
    # Training (DO NOT CHANGE)
    'batch_size': 32,
    'learning_rate': 5e-5,
    'epochs': 100,
    'patience': 25,
    
    # Inference (DO NOT CHANGE)
    'trade_threshold': 0.55,
    'filter_high_volatility': True,
    'stop_loss': None,  # Evaluate without stop-loss first
}

LABELING_CONFIG = {
    # Thresholds (DO NOT CHANGE)
    'strong_move_threshold': 0.010,   # 1.0%
    'weak_move_threshold': 0.004,     # 0.4%
    'clean_path_mae_threshold': 0.010 # 1.0%
}
```

### 4.5 Critical Implementation Rules

#### Rule 1: Normalization Must Use Training Data Only

```python
def prepare_period_data(full_df, train_end, test_start, test_end):
    """
    CRITICAL: Test data normalization uses TRAINING statistics only.
    """
    # Split chronologically
    train_df = full_df[full_df['timestamp'] <= train_end].copy()
    test_df = full_df[
        (full_df['timestamp'] >= test_start) & 
        (full_df['timestamp'] <= test_end)
    ].copy()
    
    # Feature engineering
    pipeline = FeaturePipeline()
    train_features = pipeline.compute_all_features(train_df)
    test_features = pipeline.compute_all_features(test_df)
    
    # CRITICAL: Normalize using TRAINING statistics
    train_normalized, norm_stats = pipeline.normalize_features(
        train_features, 
        fit_data=train_features  # Fit on training
    )
    test_normalized = pipeline.apply_normalization(
        test_features, 
        norm_stats  # Apply training stats to test
    )
    
    return train_normalized, test_normalized, norm_stats
```

#### Rule 2: No Information Leakage in Labeling

```python
def label_data(df):
    """
    Labels use NEXT candle's data (already implemented correctly).
    Last row will have NaN label.
    """
    labeler = CandleLabeler(LabelingConfig(
        strong_move_threshold=0.010,
        weak_move_threshold=0.004,
        clean_path_mae_threshold=0.010
    ))
    return labeler.label_dataset(df)
```

#### Rule 3: Train/Validation Split Within Training Period

```python
def split_train_val(train_df):
    """
    Split training data 85/15 for early stopping.
    Validation is the LAST 15% of training period.
    """
    val_split_idx = int(len(train_df) * 0.85)
    return train_df.iloc[:val_split_idx], train_df.iloc[val_split_idx:]
```

#### Rule 4: Fresh Model for Each Period

```python
def train_period_model(train_loader, val_loader, config):
    """
    Train a NEW model from scratch for each period.
    No transfer learning from previous periods.
    """
    model = TwoStageModel(config)  # Fresh initialization
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    return model
```

---

## 5. Phase 3: Execution & Reporting

### 5.1 Metrics to Record Per Period

```python
REQUIRED_METRICS = {
    # Dataset info
    'train_candles': int,
    'test_candles': int,
    'train_date_range': str,
    'test_date_range': str,
    
    # Trading performance
    'total_return': float,          # % return
    'n_trades': int,
    'trade_frequency': float,       # %
    'win_rate': float,              # %
    'avg_return_per_trade': float,  # %
    
    # Long trades
    'long_trades': int,
    'long_total_return': float,
    'long_win_rate': float,
    
    # Short trades
    'short_trades': int,
    'short_total_return': float,
    'short_win_rate': float,
    
    # Risk metrics
    'sharpe_ratio': float,          # Trade-frequency adjusted
    'max_drawdown': float,          # %
    'worst_trade': float,           # %
    'best_trade': float,            # %
    
    # Classification metrics
    'accuracy': float,
    'tradeable_precision': float,
    'tradeable_recall': float,
}
```

### 5.2 Summary Report Format

```python
SUMMARY_REPORT = {
    'generated_at': str,
    'model_config': dict,           # Frozen config used
    'labeling_config': dict,        # Frozen thresholds
    
    'periods': {
        'period_0_covid': REQUIRED_METRICS,
        'period_1_may2021': REQUIRED_METRICS,
        'period_2_luna': REQUIRED_METRICS,
        'period_3_ftx': REQUIRED_METRICS,
        'period_4_etf': REQUIRED_METRICS,
    },
    
    'aggregated': {
        'primary_periods_profitable': int,  # Out of 4
        'average_return': float,
        'average_sharpe': float,
        'worst_period_return': float,
        'total_trades_all_periods': int,
    },
    
    'verdict': {
        'grade': str,       # A/B/C/D/F
        'passed': bool,
        'reasoning': str,
    }
}
```

### 5.3 Grading Criteria

```python
def compute_verdict(results: dict) -> dict:
    """
    Compute final verdict based on primary periods (1-4).
    Period 0 (COVID) is bonus only.
    """
    primary_returns = [
        results['period_1_may2021']['total_return'],
        results['period_2_luna']['total_return'],
        results['period_3_ftx']['total_return'],
        results['period_4_etf']['total_return'],
    ]
    
    primary_sharpes = [
        results['period_1_may2021']['sharpe_ratio'],
        results['period_2_luna']['sharpe_ratio'],
        results['period_3_ftx']['sharpe_ratio'],
        results['period_4_etf']['sharpe_ratio'],
    ]
    
    n_profitable = sum(1 for r in primary_returns if r > 0)
    avg_return = np.mean(primary_returns)
    avg_sharpe = np.mean(primary_sharpes)
    worst_return = min(primary_returns)
    max_dd = max(results[f'period_{i}']['max_drawdown'] for i in [1,2,3,4])
    
    # Grading
    if n_profitable >= 4 and avg_sharpe > 1.5 and max_dd < 15:
        grade = 'A'
        reasoning = 'Production Ready: All periods profitable with excellent risk-adjusted returns'
    elif n_profitable >= 3 and avg_sharpe > 1.0 and max_dd < 20:
        grade = 'B'
        reasoning = 'Promising: Most periods profitable with good risk metrics'
    elif n_profitable >= 2 and avg_sharpe > 0.5:
        grade = 'C'
        reasoning = 'Needs Work: Mixed results, consider regime-specific models'
    elif n_profitable >= 1:
        grade = 'D'
        reasoning = 'Significant Issues: Only one period profitable'
    else:
        grade = 'F'
        reasoning = 'Failed: No profitable periods or catastrophic loss'
    
    # Automatic fail conditions
    if worst_return < -30:
        grade = 'F'
        reasoning = f'Failed: Catastrophic loss in one period ({worst_return:.1f}%)'
    
    passed = grade in ['A', 'B']
    
    return {
        'grade': grade,
        'passed': passed,
        'reasoning': reasoning,
        'n_profitable': n_profitable,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'worst_return': worst_return,
    }
```

---

## 6. Technical Specifications

### 6.1 Dependencies

All dependencies already exist in the codebase:
- `labeling/candle_classifier.py` - CandleLabeler, LabelingConfig
- `features/feature_pipeline.py` - FeaturePipeline
- `data/data_splitter.py` - TemporalSplitter
- `data/dataset.py` - TradingDataset
- `sph_net/models/two_stage.py` - TwoStageModel, CalibratedTwoStageModel
- `training/trainer.py` - Trainer

### 6.2 Memory Considerations

With ~55,000 candles of hourly data:
- Raw data: ~50MB
- Features: ~200MB
- Model: ~5MB
- Training batch: ~100MB

Total peak memory: <2GB (should fit in standard environments)

### 6.3 Runtime Estimates

Per period (estimated):
- Data preparation: ~30 seconds
- Training (100 epochs max, early stopping): ~5-15 minutes
- Evaluation: ~30 seconds

Total for 5 periods: ~30-90 minutes

---

## 7. Validation Checklist

### 7.1 Pre-Execution Checklist

```
□ Data Coverage Verified
  □ BTCUSDT_ml_data.parquet contains data from 2019-09-08
  □ At least 50,000 rows present
  □ No gaps in hourly data

□ Code Review Complete
  □ No future data leakage in normalization
  □ No future data leakage in labeling
  □ Fresh model initialized for each period
  □ Train/val split is temporal (not random)

□ Configuration Frozen
  □ MODEL_CONFIG matches specification
  □ LABELING_CONFIG matches specification
  □ No hyperparameter tuning code present
```

### 7.2 Post-Execution Checklist

```
□ All Periods Completed
  □ period_0_covid: model.pt exists, test_results.json valid
  □ period_1_may2021: model.pt exists, test_results.json valid
  □ period_2_luna: model.pt exists, test_results.json valid
  □ period_3_ftx: model.pt exists, test_results.json valid
  □ period_4_etf: model.pt exists, test_results.json valid

□ Sanity Checks Passed
  □ No period has 0 trades (model is trading)
  □ Both long and short trades in each period
  □ Train candle counts increase across periods
  □ No impossibly high returns (sanity check for bugs)

□ Summary Report Generated
  □ summary_report.json contains all required fields
  □ Verdict computed correctly
  □ All metrics are reasonable values
```

---

## Appendix: File Specifications

### A.1 test_results.json Schema

```json
{
  "period_id": "period_1_may2021",
  "period_name": "May 2021 Crash",
  "is_primary": true,
  
  "data": {
    "train_candles": 14500,
    "test_candles": 2208,
    "train_start": "2019-09-08T00:00:00Z",
    "train_end": "2021-04-30T23:00:00Z",
    "test_start": "2021-05-01T00:00:00Z",
    "test_end": "2021-07-31T23:00:00Z"
  },
  
  "performance": {
    "total_return": 12.45,
    "n_trades": 89,
    "trade_frequency": 4.03,
    "win_rate": 53.93,
    "avg_return_per_trade": 0.14
  },
  
  "long_trades": {
    "count": 45,
    "total_return": 7.23,
    "win_rate": 55.56
  },
  
  "short_trades": {
    "count": 44,
    "total_return": 5.22,
    "win_rate": 52.27
  },
  
  "risk": {
    "sharpe_ratio": 1.67,
    "max_drawdown": 8.34,
    "worst_trade": -2.15,
    "best_trade": 3.42
  },
  
  "classification": {
    "accuracy": 0.4521,
    "tradeable_precision": 0.5823,
    "tradeable_recall": 0.4912
  }
}
```

### A.2 summary_report.json Schema

```json
{
  "generated_at": "2025-12-16T15:30:00Z",
  "execution_time_minutes": 45,
  
  "config": {
    "model": {
      "model_type": "two_stage",
      "d_model": 64,
      "trade_threshold": 0.55
    },
    "labeling": {
      "strong_move_threshold": 0.010,
      "weak_move_threshold": 0.004,
      "clean_path_mae_threshold": 0.010
    }
  },
  
  "periods": {
    "period_0_covid": { "...": "..." },
    "period_1_may2021": { "...": "..." },
    "period_2_luna": { "...": "..." },
    "period_3_ftx": { "...": "..." },
    "period_4_etf": { "...": "..." }
  },
  
  "aggregated": {
    "primary_periods_profitable": 3,
    "average_return": 8.34,
    "average_sharpe": 1.23,
    "worst_period_return": -5.67,
    "best_period_return": 18.92,
    "total_trades_all_periods": 412
  },
  
  "verdict": {
    "grade": "B",
    "passed": true,
    "reasoning": "Promising: 3/4 periods profitable with good risk metrics",
    "recommendation": "Proceed to paper trading with caution"
  }
}
```

---

## Execution Order

1. **Run** `scripts/verify_data_coverage.py`
   - If data insufficient → Run `scripts/download_historical_data.py`
   - Then run `scripts/regenerate_ml_data.py`

2. **Run** `scripts/walk_forward_validation.py`
   - Creates all period directories
   - Trains and evaluates each period
   - Generates summary report

3. **Review** `experiments/walk_forward/summary_report.json`
   - Check verdict
   - Analyze per-period results
   - Make go/no-go decision

---

**END OF DOCUMENT**
