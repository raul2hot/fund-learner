# XGBoost Trade Quality Classifier - Implementation Guide

## Objective

Train an XGBoost classifier to predict whether a potential trade will be a WIN or LOSS, then use it as an additional entry filter to improve win rate by +15-25%.

## Background

Current best strategy: **TrendStrength/SimpleADXTrend** achieves +4.30% return with Sharpe 2.57 in a -22.68% bear market. The goal is to add a machine learning filter that only allows trades when the classifier predicts WIN.

### Current Entry Logic (from simple_adx.py)
```
Entry = LOW vol + UPTREND + RSI < 45
```

### New Entry Logic (after this implementation)
```
Entry = LOW vol + UPTREND + RSI < 45 + XGBoost_predicts_WIN
```

---

## Implementation Plan

### Step 1: Generate Trade Labels from Backtests

**File to create:** `btc_volatility_predictor/ml/generate_trade_labels.py`

```python
"""
Generate WIN/LOSS labels from historical backtest trades.

Labels are derived automatically from backtest results:
- WIN: Trade P&L > 0
- LOSS: Trade P&L <= 0

Output: CSV with features at trade entry + WIN/LOSS label
"""
```

**Key Logic:**
1. Load predictions CSV (`backtest/results_v2/test_predictions_90d.csv`)
2. Run the winning strategy (TrendStrength or SimpleADXTrend) on FULL 365-day data
3. For each trade entry point, extract:
   - All features at that moment (from features_365d.csv)
   - The trade outcome (WIN=1, LOSS=0)
4. Save as labeled dataset

**Feature Set for Each Trade Entry:**
```python
TRADE_FEATURES = [
    # Existing features (from features_365d.csv)
    'rsi_14', 'rsi_6',
    'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24',
    'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h',
    'vol_realized_24h', 'vol_of_vol',
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h',
    'macd_hist',
    'volume_ma_ratio', 'volume_change',
    'adx_14', 'plus_di_14', 'minus_di_14',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    
    # NEW features to compute at trade entry
    'trend_strength',           # SMA_fast / SMA_slow ratio
    'price_vs_sma_fast',        # close / SMA_72
    'price_vs_sma_slow',        # close / SMA_168
    'rsi_slope_3h',             # RSI change over last 3 hours
    'vol_regime_prob',          # Model's confidence in LOW vol
    'bars_since_last_trade',    # Time since last closed trade
    'recent_trade_pnl',         # P&L of last closed trade
    'consecutive_wins',         # Count of consecutive wins
    'consecutive_losses',       # Count of consecutive losses
]
```

---

### Step 2: Create Training Pipeline

**File to create:** `btc_volatility_predictor/ml/train_xgboost.py`

```python
"""
Train XGBoost classifier to predict WIN/LOSS trades.

Training approach:
- Train on ~275 days of data (365 - 90 test days)
- Validate on held-out portion
- Test on last 90 days (same period as backtest)

Target: Binary classification (WIN=1, LOSS=0)
"""

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import optuna  # For hyperparameter tuning
```

**XGBoost Configuration:**
```python
DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'max_depth': 4,              # Shallow to prevent overfitting
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'scale_pos_weight': 1.0,     # Adjust if class imbalance
    'random_state': 42,
}
```

**Training Process:**
1. Load labeled trade data from Step 1
2. Time-based train/val/test split (no shuffling - preserve temporal order)
3. Train with early stopping on validation AUC
4. Evaluate on test set
5. Save model to `checkpoints/trade_classifier.json`

---

### Step 3: Create Prediction Filter

**File to create:** `btc_volatility_predictor/ml/trade_filter.py`

```python
"""
Trade quality filter using trained XGBoost model.

Usage in strategy:
    filter = TradeQualityFilter('checkpoints/trade_classifier.json')
    
    if filter.should_trade(features_dict, threshold=0.6):
        # Execute trade
"""

class TradeQualityFilter:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.threshold = threshold
        self.feature_names = [...]  # Same order as training
    
    def predict_win_probability(self, features: dict) -> float:
        """Return probability that trade will be a WIN."""
        pass
    
    def should_trade(self, features: dict, threshold: float = None) -> bool:
        """Return True if classifier predicts WIN with confidence >= threshold."""
        pass
```

---

### Step 4: Integrate into Strategy

**File to modify:** `btc_volatility_predictor/backtest/strategies/simple_adx.py`

Add new strategy class:

```python
class TrendStrengthWithML(BaseStrategy):
    """
    TrendStrength strategy + XGBoost WIN/LOSS filter.
    
    Entry conditions (ALL must be true):
    1. Vol = LOW (from SPHNet model)
    2. Trend = UP (72h/168h SMA)
    3. RSI < 45
    4. XGBoost predicts WIN with probability >= threshold
    """
    
    def __init__(
        self,
        ml_model_path: str = "checkpoints/trade_classifier.json",
        ml_threshold: float = 0.6,  # Only trade if P(WIN) >= 60%
        **kwargs
    ):
        super().__init__(name="TrendStrengthML")
        self.ml_filter = TradeQualityFilter(ml_model_path, ml_threshold)
        # ... rest of init
```

---

### Step 5: Create Backtest Runner

**File to create:** `btc_volatility_predictor/backtest/run_backtest_v5.py`

```python
"""
V5 Backtest Runner - ML-Enhanced Strategies

Tests the hypothesis: XGBoost filter improves win rate by 15-25%
"""

# Compare:
# 1. TrendStrength (baseline)
# 2. TrendStrengthML with threshold=0.5
# 3. TrendStrengthML with threshold=0.6
# 4. TrendStrengthML with threshold=0.7
```

---

## File Structure After Implementation

```
btc_volatility_predictor/
├── ml/                              # NEW DIRECTORY
│   ├── __init__.py
│   ├── generate_trade_labels.py     # Step 1
│   ├── train_xgboost.py             # Step 2
│   ├── trade_filter.py              # Step 3
│   └── feature_engineering.py       # Helper for new features
├── backtest/
│   ├── strategies/
│   │   ├── simple_adx.py            # MODIFY: Add TrendStrengthWithML
│   │   └── ...
│   ├── run_backtest_v5.py           # NEW: V5 runner
│   └── ...
├── checkpoints/
│   ├── trade_classifier.json        # Trained XGBoost model
│   └── ...
└── data/
    └── ml/
        └── trade_labels.csv         # Generated labels
```

---

## Expected Results

| Metric | TrendStrength (Baseline) | TrendStrengthML (Expected) |
|--------|-------------------------|---------------------------|
| Win Rate | ~58% (7/12 trades) | 73-83% (+15-25%) |
| Trades | 12 | ~8-10 (fewer, higher quality) |
| Return | +4.30% | +4-6% (similar or better) |
| Sharpe | 2.57 | 2.5-3.5 |

---

## Implementation Commands

Run these in order:

```bash
cd btc_volatility_predictor

# Step 1: Generate trade labels
python ml/generate_trade_labels.py

# Step 2: Train XGBoost classifier
python ml/train_xgboost.py

# Step 3: Run V5 backtest with ML filter
python backtest/run_backtest_v5.py
```

---

## Key Considerations

### 1. Avoid Look-Ahead Bias
- Only use features available AT THE TIME of trade entry
- Train on historical data BEFORE test period
- Use walk-forward validation if possible

### 2. Handle Class Imbalance
- If WIN/LOSS ratio is skewed, use `scale_pos_weight` in XGBoost
- Consider oversampling minority class (SMOTE)
- Use stratified sampling in cross-validation

### 3. Feature Importance Analysis
After training, analyze which features matter most:
```python
xgb.plot_importance(model, max_num_features=15)
```

### 4. Threshold Tuning
- Higher threshold = fewer trades, higher win rate
- Lower threshold = more trades, lower win rate
- Find optimal balance via backtesting

### 5. Overfitting Prevention
- Use early stopping
- Keep model shallow (max_depth=3-5)
- Strong regularization (alpha, lambda)
- Cross-validate on time-series splits

---

## Detailed Implementation: generate_trade_labels.py

```python
#!/usr/bin/env python3
"""
Generate WIN/LOSS labels from historical trades.

This script:
1. Loads full 365-day feature data
2. Simulates TrendStrength strategy on entire dataset
3. Records features at each trade entry
4. Labels each trade as WIN (P&L > 0) or LOSS (P&L <= 0)
5. Saves labeled dataset for XGBoost training
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.strategies.simple_adx import TrendStrengthStrategy


# Features to extract at trade entry
CORE_FEATURES = [
    'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24', 'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h', 'vol_realized_24h', 'vol_of_vol',
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h',
    'macd_hist', 'volume_ma_ratio', 'volume_change',
    'adx_14', 'plus_di_14', 'minus_di_14',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]


def compute_additional_features(row: dict, history: List[dict]) -> dict:
    """Compute additional features for trade entry."""
    features = {}
    
    close = row.get('close', 0)
    
    # Trend strength features
    if len(history) >= 168:
        sma_72 = np.mean([h['close'] for h in history[-72:]])
        sma_168 = np.mean([h['close'] for h in history[-168:]])
        
        features['trend_strength'] = sma_72 / sma_168 if sma_168 > 0 else 1.0
        features['price_vs_sma_fast'] = close / sma_72 if sma_72 > 0 else 1.0
        features['price_vs_sma_slow'] = close / sma_168 if sma_168 > 0 else 1.0
    else:
        features['trend_strength'] = 1.0
        features['price_vs_sma_fast'] = 1.0
        features['price_vs_sma_slow'] = 1.0
    
    # RSI slope (change over last 3 hours)
    if len(history) >= 3:
        rsi_now = row.get('rsi_14', 50)
        rsi_3h_ago = history[-3].get('rsi_14', 50)
        features['rsi_slope_3h'] = rsi_now - rsi_3h_ago
    else:
        features['rsi_slope_3h'] = 0.0
    
    # Volatility regime probability (from predictions if available)
    features['vol_regime_prob'] = row.get('prediction_prob', 0.5)
    
    return features


def generate_labels(
    data_path: str = "data/processed/features_365d.csv",
    predictions_path: str = None,  # Optional: include model predictions
    output_path: str = "data/ml/trade_labels.csv",
    strategy_class = TrendStrengthStrategy,
) -> pd.DataFrame:
    """
    Generate trade labels by running strategy on historical data.
    
    Returns DataFrame with:
    - Features at trade entry
    - Label: 1 = WIN, 0 = LOSS
    - Trade metadata (entry_price, exit_price, pnl, etc.)
    """
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # If predictions available, merge them
    if predictions_path and os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        # Align by index or timestamp
        if 'predicted_regime' in pred_df.columns:
            # Use predictions for the overlapping period
            pass
    
    # For full 365 days, we need to generate predictions
    # For now, use a simple volatility threshold
    vol_threshold = df['target_volatility'].median()
    df['predicted_regime'] = (df['target_volatility'] > vol_threshold).astype(int)
    
    print(f"Running strategy on {len(df)} samples...")
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=10000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    strategy = strategy_class()
    result = engine.run(strategy, df)
    
    print(f"Generated {result.num_trades} trades")
    print(f"Win rate: {result.win_rate:.1%}")
    
    # Extract features for each trade
    labeled_data = []
    data_records = df.to_dict('records')
    
    for trade in result.trades:
        entry_idx = trade.entry_time
        
        if entry_idx < 0 or entry_idx >= len(data_records):
            continue
        
        row = data_records[entry_idx]
        history = data_records[max(0, entry_idx-168):entry_idx]
        
        # Extract core features
        features = {}
        for feat in CORE_FEATURES:
            features[feat] = row.get(feat, 0)
        
        # Add computed features
        features.update(compute_additional_features(row, history))
        
        # Add label
        features['label'] = 1 if trade.pnl > 0 else 0
        
        # Add metadata (for analysis, not training)
        features['_entry_time'] = entry_idx
        features['_exit_time'] = trade.exit_time
        features['_entry_price'] = trade.entry_price
        features['_exit_price'] = trade.exit_price
        features['_pnl'] = trade.pnl
        features['_pnl_pct'] = trade.pnl_pct
        features['_holding_period'] = trade.holding_period
        
        labeled_data.append(features)
    
    # Create DataFrame
    labels_df = pd.DataFrame(labeled_data)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*50}")
    print("LABEL GENERATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total trades: {len(labels_df)}")
    print(f"Wins: {labels_df['label'].sum()} ({labels_df['label'].mean():.1%})")
    print(f"Losses: {len(labels_df) - labels_df['label'].sum()}")
    print(f"Features: {len(CORE_FEATURES) + 5}")  # +5 for computed features
    print(f"Saved to: {output_path}")
    
    return labels_df


if __name__ == "__main__":
    generate_labels()
```

---

## Detailed Implementation: train_xgboost.py

```python
#!/usr/bin/env python3
"""
Train XGBoost classifier for trade quality prediction.

Predicts: Will this trade be a WIN (1) or LOSS (0)?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Features to use (exclude metadata columns starting with _)
FEATURE_COLS = [
    'rsi_14', 'rsi_6', 'bb_bandwidth_20', 'bb_position',
    'atr_14', 'atr_24', 'vol_gk_1h', 'vol_gk_6h', 'vol_gk_24h',
    'vol_park_1h', 'vol_park_24h', 'vol_realized_24h', 'vol_of_vol',
    'momentum_1h', 'momentum_6h', 'momentum_12h', 'momentum_24h',
    'macd_hist', 'volume_ma_ratio', 'volume_change',
    'adx_14', 'plus_di_14', 'minus_di_14',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'trend_strength', 'price_vs_sma_fast', 'price_vs_sma_slow',
    'rsi_slope_3h', 'vol_regime_prob',
]


def load_and_split_data(
    data_path: str = "data/ml/trade_labels.csv",
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Load labeled data and split by time (no shuffling).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = pd.read_csv(data_path)
    
    # Sort by entry time to ensure temporal order
    df = df.sort_values('_entry_time').reset_index(drop=True)
    
    # Get features and labels
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Time-based split
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train WIN rate: {y_train.mean():.1%}")
    print(f"Test WIN rate: {y_test.mean():.1%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_xgboost(
    X_train, X_val, y_train, y_val,
    feature_names: list,
    params: dict = None,
) -> xgb.Booster:
    """
    Train XGBoost classifier with early stopping.
    """
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
        }
    
    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = pos_weight
    print(f"Class weight: {pos_weight:.2f}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=20,
    )
    
    return model


def evaluate_model(
    model: xgb.Booster,
    X_test, y_test,
    feature_names: list,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on test set."""
    
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_prob = model.predict(dtest)
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
    }
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS (threshold={threshold})")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
    
    return metrics


def plot_feature_importance(model: xgb.Booster, output_path: str = "figures/xgb_importance.png"):
    """Plot and save feature importance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain')
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature importance to {output_path}")


def main():
    # Load data
    print("Loading labeled trade data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_data()
    
    # Train
    print("\nTraining XGBoost...")
    model = train_xgboost(X_train, X_val, y_train, y_val, feature_names)
    
    # Evaluate at different thresholds
    for threshold in [0.5, 0.6, 0.7]:
        evaluate_model(model, X_test, y_test, feature_names, threshold)
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model.save_model("checkpoints/trade_classifier.json")
    print("\nModel saved to checkpoints/trade_classifier.json")
    
    # Save feature names for inference
    with open("checkpoints/trade_classifier_features.json", 'w') as f:
        json.dump(feature_names, f)
    
    # Plot importance
    plot_feature_importance(model)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## Detailed Implementation: trade_filter.py

```python
#!/usr/bin/env python3
"""
Trade quality filter using trained XGBoost model.

Provides a simple interface for strategies to check
if a potential trade is predicted to be a WIN.
"""

import os
import json
import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional


class TradeQualityFilter:
    """
    Filter trades based on XGBoost WIN/LOSS prediction.
    
    Usage:
        filter = TradeQualityFilter()
        
        # In strategy:
        if filter.should_trade(features, threshold=0.6):
            return 'BUY'
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/trade_classifier.json",
        features_path: str = "checkpoints/trade_classifier_features.json",
        default_threshold: float = 0.5,
    ):
        """
        Initialize filter with trained model.
        
        Args:
            model_path: Path to XGBoost model file
            features_path: Path to feature names JSON
            default_threshold: Default probability threshold for WIN prediction
        """
        self.default_threshold = default_threshold
        self.model = None
        self.feature_names = None
        
        # Load model
        if os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            print(f"Loaded trade classifier from {model_path}")
        else:
            print(f"WARNING: Trade classifier not found at {model_path}")
        
        # Load feature names
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
            print(f"WARNING: Feature names not found at {features_path}")
    
    def is_ready(self) -> bool:
        """Check if filter is properly initialized."""
        return self.model is not None and self.feature_names is not None
    
    def extract_features(self, row: dict, history: List[dict]) -> np.ndarray:
        """
        Extract features from current row and history.
        
        Args:
            row: Current bar data (dict with feature values)
            history: List of previous bars
            
        Returns:
            Feature array in correct order
        """
        features = {}
        
        # Core features from row
        for feat in self.feature_names:
            if feat in row:
                features[feat] = row[feat]
            elif feat == 'trend_strength':
                # Compute trend strength
                if len(history) >= 168:
                    sma_72 = np.mean([h.get('close', 0) for h in history[-72:]])
                    sma_168 = np.mean([h.get('close', 0) for h in history[-168:]])
                    features[feat] = sma_72 / sma_168 if sma_168 > 0 else 1.0
                else:
                    features[feat] = 1.0
            elif feat == 'price_vs_sma_fast':
                if len(history) >= 72:
                    sma_72 = np.mean([h.get('close', 0) for h in history[-72:]])
                    features[feat] = row.get('close', 0) / sma_72 if sma_72 > 0 else 1.0
                else:
                    features[feat] = 1.0
            elif feat == 'price_vs_sma_slow':
                if len(history) >= 168:
                    sma_168 = np.mean([h.get('close', 0) for h in history[-168:]])
                    features[feat] = row.get('close', 0) / sma_168 if sma_168 > 0 else 1.0
                else:
                    features[feat] = 1.0
            elif feat == 'rsi_slope_3h':
                if len(history) >= 3:
                    rsi_now = row.get('rsi_14', 50)
                    rsi_3h_ago = history[-3].get('rsi_14', 50)
                    features[feat] = rsi_now - rsi_3h_ago
                else:
                    features[feat] = 0.0
            elif feat == 'vol_regime_prob':
                features[feat] = row.get('prediction_prob', 0.5)
            else:
                features[feat] = 0.0  # Default
        
        # Convert to array in correct order
        return np.array([features.get(f, 0.0) for f in self.feature_names])
    
    def predict_win_probability(
        self,
        row: dict,
        history: List[dict],
    ) -> float:
        """
        Predict probability that trade will be a WIN.
        
        Args:
            row: Current bar data
            history: Previous bars
            
        Returns:
            Probability [0, 1] that trade will be WIN
        """
        if not self.is_ready():
            return 0.5  # Neutral if model not loaded
        
        features = self.extract_features(row, history)
        dmatrix = xgb.DMatrix(features.reshape(1, -1), feature_names=self.feature_names)
        
        prob = self.model.predict(dmatrix)[0]
        return float(prob)
    
    def should_trade(
        self,
        row: dict,
        history: List[dict],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if trade should be taken.
        
        Args:
            row: Current bar data
            history: Previous bars
            threshold: Probability threshold (default: self.default_threshold)
            
        Returns:
            True if model predicts WIN with probability >= threshold
        """
        if not self.is_ready():
            return True  # Allow trade if model not loaded
        
        threshold = threshold or self.default_threshold
        prob = self.predict_win_probability(row, history)
        
        return prob >= threshold


# Singleton instance for easy import
_default_filter = None

def get_trade_filter(threshold: float = 0.5) -> TradeQualityFilter:
    """Get or create default trade filter instance."""
    global _default_filter
    if _default_filter is None:
        _default_filter = TradeQualityFilter(default_threshold=threshold)
    return _default_filter
```

---

## Success Criteria

After implementation, run backtests and verify:

1. **Win Rate Improvement**: TrendStrengthML achieves 73-83% win rate (up from ~58%)
2. **Maintained Returns**: Return >= 3% (allowing slight decrease for higher quality trades)
3. **Reduced Trades**: Fewer trades (8-10 vs 12), but higher quality
4. **Sharpe Ratio**: Maintained or improved (>= 2.5)
5. **Feature Importance**: Top features make intuitive sense

---

## Troubleshooting

### Issue: Too few labeled samples
**Solution**: Run strategy on full 365 days, not just test period. May need to lower entry thresholds temporarily to generate more trades for training.

### Issue: Class imbalance
**Solution**: Use `scale_pos_weight` in XGBoost, or undersample majority class.

### Issue: Overfitting
**Solution**: 
- Reduce `max_depth` to 3
- Increase `min_child_weight` to 10
- Increase regularization (`reg_alpha`, `reg_lambda`)
- Use cross-validation

### Issue: Model not improving win rate
**Solution**: 
- Add more features (price patterns, volume analysis)
- Try different thresholds
- Ensemble with other models

---

## Next Steps After V5

1. **Walk-Forward Optimization**: Retrain model periodically on rolling windows
2. **Multi-Model Ensemble**: Combine XGBoost with other classifiers (LightGBM, RandomForest)
3. **Dynamic Threshold**: Adjust threshold based on market conditions
4. **Position Sizing**: Use win probability to size positions (higher prob = larger position)