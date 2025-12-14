#!/usr/bin/env python3
"""
Multi-Period Robustness Testing for TrendStrengthML_50

Tests the strategy across multiple market cycles:
1. 2018 Bear Market (-84% drawdown)
2. 2019-2020 Recovery/Bull (+345% then +1692%)
3. 2023-2024 Bull Market (ETF Rally, +376%)
4. Random date sampling within each period

Includes:
- Full period backtesting
- Random window testing to prevent overfitting
- Walk-forward optimization with proper train/val/test splits
- Bootstrap confidence intervals
- Monte Carlo permutation tests for statistical significance

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
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.strategies import TrendStrengthWithML, TrendStrengthStrategy

# Import ML modules for walk-forward training
try:
    from ml.generate_trade_labels import generate_labels
    from ml.train_xgboost import load_and_split_data, train_xgboost, save_model, DEFAULT_PARAMS
    from ml.feature_engineering import TRADE_FEATURES, CORE_FEATURES, compute_additional_features
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available ({e}). Walk-forward training disabled.")
    ML_AVAILABLE = False

# Import SPHNet for volatility regime training
try:
    import torch
    from train_regime import VolatilityRegimeDataset, create_regime_dataloaders
    from models import SPHNet
    from config import Config
    SPHNET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SPHNet modules not available ({e}). Volatility model training disabled.")
    SPHNET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
RANDOM_SEED = 42

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

# Success criteria thresholds
SUCCESS_CRITERIA = {
    'positive_return': True,  # Return > 0
    'min_win_rate': 0.60,     # Win rate > 60%
    'min_sharpe': 0.5,        # Sharpe ratio > 0.5
    'max_drawdown': 0.20,     # Max drawdown < 20%
    'min_trades': 5           # At least 5 trades per 90 days
}


# =============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

def bootstrap_confidence_interval(
    returns: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for returns.

    Args:
        returns: List of trade returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if len(returns) < 2:
        return (np.nan, np.nan)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return lower, upper


def permutation_test(
    strategy_returns: List[float],
    baseline_returns: List[float],
    n_permutations: int = 10000
) -> float:
    """
    Test if strategy significantly outperforms baseline using permutation test.

    Args:
        strategy_returns: List of strategy trade returns
        baseline_returns: List of baseline trade returns
        n_permutations: Number of permutations

    Returns:
        p-value for null hypothesis that strategy = baseline
    """
    if len(strategy_returns) < 2 or len(baseline_returns) < 2:
        return np.nan

    observed_diff = np.mean(strategy_returns) - np.mean(baseline_returns)

    combined = list(strategy_returns) + list(baseline_returns)
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


def calculate_statistical_tests(
    strategy_results: Dict,
    baseline_results: Dict
) -> Dict:
    """
    Calculate comprehensive statistical tests.

    Args:
        strategy_results: Dict with strategy backtest results
        baseline_results: Dict with baseline backtest results

    Returns:
        Dict with statistical test results
    """
    stats = {}

    # Get trade returns from results (mock implementation if needed)
    strategy_returns = strategy_results.get('trade_returns', [])
    baseline_returns = baseline_results.get('trade_returns', [])

    # Bootstrap CI for strategy
    ci_lower, ci_upper = bootstrap_confidence_interval(strategy_returns)
    stats['bootstrap_ci'] = {
        'lower': ci_lower,
        'upper': ci_upper,
        'confidence': 0.95
    }

    # Permutation test
    p_value = permutation_test(strategy_returns, baseline_returns)
    stats['permutation_test'] = {
        'p_value': p_value,
        'significant_at_05': p_value < 0.05 if not np.isnan(p_value) else False,
        'significant_at_01': p_value < 0.01 if not np.isnan(p_value) else False
    }

    return stats


# =============================================================================
# RANDOM WINDOW GENERATION
# =============================================================================

def generate_random_windows(
    df: pd.DataFrame,
    n_windows: int = 10,
    min_days: int = 60,
    max_days: int = 120
) -> List[Tuple[int, int]]:
    """
    Generate random non-overlapping date windows for testing.

    This prevents overfitting by testing on randomly selected time periods
    rather than fixed intervals.

    Args:
        df: DataFrame with data
        n_windows: Number of random windows to generate
        min_days: Minimum window size in days
        max_days: Maximum window size in days

    Returns:
        List of (start_idx, end_idx) tuples
    """
    random.seed(RANDOM_SEED)

    n_samples = len(df)
    min_samples = min_days * 24  # Hourly data
    max_samples = max_days * 24

    windows = []
    used_indices = set()

    # Need warmup period for indicators
    warmup = 168  # 1 week

    attempts = 0
    max_attempts = n_windows * 20

    while len(windows) < n_windows and attempts < max_attempts:
        # Random window size
        window_size = random.randint(min_samples, max_samples)

        # Random start point (leave room for window and warmup)
        max_start = n_samples - window_size - warmup
        if max_start < warmup:
            break

        start_idx = random.randint(warmup, max_start)
        end_idx = start_idx + window_size

        # Check for overlap with existing windows
        window_set = set(range(start_idx, end_idx))
        if not window_set.intersection(used_indices):
            windows.append((start_idx, end_idx))
            used_indices.update(window_set)

        attempts += 1

    return windows


# =============================================================================
# WALK-FORWARD VOLATILITY MODEL TRAINING
# =============================================================================

def train_volatility_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    period_name: str,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Train SPHNet volatility regime model for a specific period.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        period_name: Name of the period
        output_dir: Directory to save the model
        verbose: Whether to print progress

    Returns:
        Path to saved model, or None if training failed
    """
    if not SPHNET_AVAILABLE:
        if verbose:
            print("    SPHNet not available, using threshold-based regime")
        return None

    # Use absolute path for output directory
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "checkpoints", "multiperiod")
    elif not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    try:
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create datasets
        train_dataset = VolatilityRegimeDataset(
            train_df,
            window_size=config.window_size,
            percentile=50,
            fit_scalers=True
        )

        val_dataset = VolatilityRegimeDataset(
            val_df,
            window_size=config.window_size,
            price_scaler=train_dataset.price_scaler,
            feature_scaler=train_dataset.feature_scaler,
            vol_threshold=train_dataset.vol_threshold
        )

        if len(train_dataset) < 100:
            if verbose:
                print(f"    WARNING: Only {len(train_dataset)} training samples, skipping SPHNet")
            return None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False
        )

        # Create model
        model = SPHNet(
            n_price_features=len(train_dataset.price_cols),
            n_eng_features=len(train_dataset.eng_cols),
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            dropout=config.dropout,
            output_dim=1  # Binary classification
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Training loop (simplified)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        if verbose:
            print(f"    Training SPHNet: {len(train_dataset)} train, {len(val_dataset)} val samples")

        for epoch in range(50):  # Max epochs
            model.train()
            train_loss = 0
            for batch in train_loader:
                prices = batch['prices'].to(device)
                features = batch['features'].to(device)
                targets = batch['target'].to(device)

                optimizer.zero_grad()
                outputs = model(prices, features).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    prices = batch['prices'].to(device)
                    features = batch['features'].to(device)
                    targets = batch['target'].to(device)
                    outputs = model(prices, features).squeeze()
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = os.path.join(output_dir, f"sphnet_{period_name}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vol_threshold': train_dataset.vol_threshold,
                    'price_scaler': train_dataset.price_scaler,
                    'feature_scaler': train_dataset.feature_scaler,
                    'price_cols': train_dataset.price_cols,
                    'eng_cols': train_dataset.eng_cols,
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if verbose:
            print(f"    SPHNet trained, best val loss: {best_val_loss:.4f}")
            print(f"    Model saved to {model_path}")

        return model_path

    except Exception as e:
        if verbose:
            print(f"    ERROR training SPHNet: {e}")
        return None


def generate_regime_predictions_with_model(
    df: pd.DataFrame,
    model_path: str,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate regime predictions using trained SPHNet model.

    Args:
        df: DataFrame with features
        model_path: Path to trained SPHNet model
        verbose: Whether to print progress

    Returns:
        DataFrame with predicted_regime column
    """
    df = df.copy()

    if not SPHNET_AVAILABLE or not os.path.exists(model_path):
        # Fall back to threshold-based
        return prepare_regime_predictions(df)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=device)

        config = Config()
        model = SPHNet(
            n_price_features=len(checkpoint['price_cols']),
            n_eng_features=len(checkpoint['eng_cols']),
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            dropout=0,  # No dropout at inference
            output_dim=1
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create dataset for prediction
        dataset = VolatilityRegimeDataset(
            df,
            window_size=config.window_size,
            price_scaler=checkpoint['price_scaler'],
            feature_scaler=checkpoint['feature_scaler'],
            vol_threshold=checkpoint['vol_threshold']
        )

        # Generate predictions
        predictions = []
        with torch.no_grad():
            for i in range(len(dataset)):
                batch = dataset[i]
                prices = batch['prices'].unsqueeze(0).to(device)
                features = batch['features'].unsqueeze(0).to(device)
                output = torch.sigmoid(model(prices, features)).item()
                predictions.append(1 if output > 0.5 else 0)

        # Pad predictions to match df length
        pad_length = len(df) - len(predictions)
        full_predictions = [0] * pad_length + predictions

        df['predicted_regime'] = full_predictions

        if verbose:
            high_pct = sum(full_predictions) / len(full_predictions) * 100
            print(f"    SPHNet predictions: {high_pct:.1f}% HIGH volatility")

        return df

    except Exception as e:
        if verbose:
            print(f"    ERROR generating predictions: {e}")
        return prepare_regime_predictions(df)


# =============================================================================
# WALK-FORWARD XGBOOST TRAINING
# =============================================================================

def generate_labels_from_df(
    df: pd.DataFrame,
    output_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate trade labels from a DataFrame by running TrendStrength strategy.

    Args:
        df: DataFrame with features and predicted_regime
        output_path: Path to save labeled data
        verbose: Whether to print progress

    Returns:
        DataFrame with labeled trades
    """
    if not ML_AVAILABLE:
        raise RuntimeError("ML modules not available")

    # Run backtest to get trades
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategy = TrendStrengthStrategy()
    result = engine.run(strategy, df)

    if verbose:
        print(f"    Generated {result.num_trades} trades (WR: {result.win_rate:.1%})")

    if result.num_trades == 0:
        return pd.DataFrame()

    # Extract features for each trade
    labeled_data = []
    data_records = df.to_dict('records')

    for trade in result.trades:
        entry_idx = trade.entry_time

        if entry_idx < 0 or entry_idx >= len(data_records):
            continue

        row = data_records[entry_idx]
        history = data_records[max(0, entry_idx - 168):entry_idx]

        # Extract core features
        features = {}
        for feat in CORE_FEATURES:
            features[feat] = row.get(feat, 0)

        # Add computed features
        features.update(compute_additional_features(row, history))

        # Add label: WIN (1) if P&L > 0, else LOSS (0)
        features['label'] = 1 if trade.pnl > 0 else 0

        # Add metadata
        features['_entry_time'] = entry_idx
        features['_pnl'] = trade.pnl
        features['_pnl_pct'] = trade.pnl_pct

        labeled_data.append(features)

    labels_df = pd.DataFrame(labeled_data)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels_df.to_csv(output_path, index=False)

    return labels_df


def train_period_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    period_name: str,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Train XGBoost model for a specific period using walk-forward approach.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        period_name: Name of the period
        output_dir: Directory to save the model
        verbose: Whether to print progress

    Returns:
        Path to saved model, or None if training failed
    """
    if not ML_AVAILABLE:
        print("    ML modules not available, skipping training")
        return None

    # Use absolute path for output directory
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "checkpoints", "multiperiod")
    elif not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Generate labels from training data
    labels_path = os.path.join(output_dir, f"labels_{period_name}_train.csv")

    if verbose:
        print(f"    Generating trade labels from training data...")

    train_df_copy = prepare_regime_predictions(train_df.copy())
    labels_df = generate_labels_from_df(train_df_copy, labels_path, verbose=verbose)

    if len(labels_df) < 10:
        print(f"    WARNING: Only {len(labels_df)} trades in training data, insufficient for training")
        return None

    # Also generate validation labels for early stopping
    val_labels_path = os.path.join(output_dir, f"labels_{period_name}_val.csv")
    val_df_copy = prepare_regime_predictions(val_df.copy())
    val_labels_df = generate_labels_from_df(val_df_copy, val_labels_path, verbose=False)

    if len(val_labels_df) < 3:
        # Use part of training for validation
        n_val = max(3, len(labels_df) // 5)
        val_labels_df = labels_df.tail(n_val)
        labels_df = labels_df.head(len(labels_df) - n_val)

    # Prepare features
    feature_cols = [c for c in TRADE_FEATURES if c in labels_df.columns]
    missing = [c for c in TRADE_FEATURES if c not in labels_df.columns]
    for feat in missing:
        labels_df[feat] = 0
        val_labels_df[feat] = 0
    feature_cols = TRADE_FEATURES

    X_train = labels_df[feature_cols].values
    y_train = labels_df['label'].values
    X_val = val_labels_df[feature_cols].values
    y_val = val_labels_df['label'].values

    if verbose:
        print(f"    Training XGBoost: {len(X_train)} train, {len(X_val)} val samples")
        print(f"    Train WIN rate: {y_train.mean():.1%}")

    # Train model
    try:
        model = train_xgboost(
            X_train, X_val, y_train, y_val,
            feature_names=feature_cols,
            params=DEFAULT_PARAMS.copy(),
            num_boost_round=200,
            early_stopping_rounds=20,
            verbose=False
        )

        # Save model
        model_path = os.path.join(output_dir, f"trade_classifier_{period_name}.json")
        features_path = os.path.join(output_dir, f"trade_classifier_{period_name}_features.json")

        model.save_model(model_path)
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)

        if verbose:
            print(f"    Model saved to {model_path}")

        return model_path

    except Exception as e:
        print(f"    ERROR training model: {e}")
        return None


# =============================================================================
# WALK-FORWARD OPTIMIZATION
# =============================================================================

def walk_forward_test(df: pd.DataFrame, period_name: str, train_model: bool = True) -> Dict:
    """
    Walk-forward analysis with proper model training on each period.

    Split:
    - Train: First 70% of data (used for model training)
    - Validate: Next 15% of data (used for early stopping)
    - Test: Final 15% of data (out-of-sample evaluation)

    For each period:
    1. Train SPHNet volatility model on training data
    2. Generate regime predictions using trained SPHNet
    3. Generate trade labels from training data
    4. Train XGBoost classifier on those labels
    5. Test on val/test splits with both freshly trained models

    Args:
        df: DataFrame with features
        period_name: Name of the period for logging
        train_model: Whether to train new models (True) or use existing

    Returns:
        Dict with walk-forward results
    """
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"\n  Walk-Forward Split for {period_name}:")
    print(f"    Train: {n_train} samples ({n_train/24:.0f} days)")
    print(f"    Val:   {n_val} samples ({n_val/24:.0f} days)")
    print(f"    Test:  {n_test} samples ({n_test/24:.0f} days)")

    results = {
        'train_samples': n_train,
        'val_samples': n_val,
        'test_samples': n_test,
        'sphnet_path': None,
        'xgb_model_path': None,
        'splits': {}
    }

    # Step 1: Train SPHNet volatility model on training data
    sphnet_path = None
    if train_model and SPHNET_AVAILABLE:
        print(f"\n  Step 1: Training SPHNet volatility model...")
        sphnet_path = train_volatility_model(train_df, val_df, period_name)
        results['sphnet_path'] = sphnet_path

    # Step 2: Generate regime predictions for training data using trained SPHNet
    if sphnet_path:
        print(f"\n  Step 2: Generating regime predictions with trained SPHNet...")
        train_df_with_regime = generate_regime_predictions_with_model(train_df, sphnet_path)
        val_df_with_regime = generate_regime_predictions_with_model(val_df, sphnet_path)
    else:
        train_df_with_regime = prepare_regime_predictions(train_df)
        val_df_with_regime = prepare_regime_predictions(val_df)

    # Step 3: Train XGBoost on training data with regime predictions
    xgb_model_path = None
    if train_model and ML_AVAILABLE:
        print(f"\n  Step 3: Training XGBoost trade classifier...")
        xgb_model_path = train_period_model(train_df_with_regime, val_df_with_regime, period_name)
        results['xgb_model_path'] = xgb_model_path

        if xgb_model_path:
            print(f"    Using trained XGBoost for evaluation")

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    # Step 4: Evaluate on each split
    print(f"\n  Step 4: Evaluating on train/val/test splits...")
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(split_df) < 168:  # Need warmup
            results['splits'][split_name] = {'error': 'Insufficient data'}
            continue

        # Generate regime predictions using trained SPHNet (or threshold)
        if sphnet_path:
            split_df = generate_regime_predictions_with_model(split_df, sphnet_path)
        else:
            split_df = prepare_regime_predictions(split_df)

        # Create strategy with period-specific XGBoost model if available
        if xgb_model_path and os.path.exists(xgb_model_path):
            features_path = xgb_model_path.replace('.json', '_features.json')
            strategy = TrendStrengthWithML(
                ml_model_path=xgb_model_path,
                ml_features_path=features_path,
                ml_threshold=0.5
            )
        else:
            strategy = TrendStrengthWithML(ml_threshold=0.5)

        result = engine.run(strategy, split_df)

        results['splits'][split_name] = {
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'trades': result.num_trades,
            'win_rate': result.win_rate if result.num_trades > 0 else 0
        }

        print(f"    {split_name.capitalize():6s}: Return={result.total_return*100:+.2f}%, "
              f"Trades={result.num_trades}, WinRate={result.win_rate*100:.1f}%")

    return results


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_regime_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare volatility regime predictions if not present.

    Uses median volatility as threshold for HIGH/LOW classification.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with predicted_regime column
    """
    df = df.copy()

    if 'predicted_regime' not in df.columns:
        # Try different volatility columns
        vol_cols = ['target_volatility', 'vol_realized_24h', 'vol_gk_24h']
        vol_col = None

        for col in vol_cols:
            if col in df.columns:
                vol_col = col
                break

        if vol_col:
            threshold = df[vol_col].median()
            df['predicted_regime'] = (df[vol_col] > threshold).astype(int)
        else:
            # Fallback: all LOW volatility
            df['predicted_regime'] = 0

    return df


# =============================================================================
# SINGLE PERIOD TESTING
# =============================================================================

def run_period_backtest(period_name: str, config: Dict) -> Optional[Dict]:
    """
    Run comprehensive backtest for a single period.

    Includes:
    - Full period test with baseline and ML strategies
    - Random window tests
    - Walk-forward analysis

    Args:
        period_name: Name of the period
        config: Period configuration dict

    Returns:
        Dict with all results, or None if data not found
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {period_name.upper()}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Handle path relative to script or absolute
    data_path = config['path']
    if not os.path.isabs(data_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, data_path)

    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        print(f"  Run 'python fetch_historical_periods.py' first to download data")
        print(f"  Then run 'python data/features.py --historical' to prepare features")
        return None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples ({len(df)/24:.0f} days)")

    # Prepare regime predictions
    df = prepare_regime_predictions(df)

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    results = {
        'period': period_name,
        'type': config['type'],
        'samples': len(df),
        'days': len(df) / 24,
        'full_period': None,
        'random_windows': [],
        'walk_forward': None,
        'statistics': {}
    }

    # === Test 1: Full Period ===
    print("\n--- Full Period Test ---")

    strategies = [
        ('TrendStrength', TrendStrengthStrategy()),
        ('TrendStrengthML_50', TrendStrengthWithML(ml_threshold=0.5)),
    ]

    full_results = []
    baseline_result = None
    ml_result = None

    for strategy_name, strategy in strategies:
        result = engine.run(strategy, df)

        result_dict = {
            'strategy': strategy_name,
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'trades': result.num_trades,
            'win_rate': result.win_rate if result.num_trades > 0 else 0,
            'profit_factor': result.profit_factor if hasattr(result, 'profit_factor') else 0
        }

        # Extract trade returns if available
        if hasattr(result, 'trades') and result.trades:
            result_dict['trade_returns'] = [t.pnl_pct if hasattr(t, 'pnl_pct') else t.pnl / INITIAL_CAPITAL
                                           for t in result.trades]
        else:
            result_dict['trade_returns'] = []

        full_results.append(result_dict)

        if 'ML' in strategy_name:
            ml_result = result_dict
        else:
            baseline_result = result_dict

        print(f"  {strategy_name}: Return={result.total_return*100:+.2f}%, "
              f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.num_trades}, "
              f"WinRate={result.win_rate*100:.1f}%")

    results['full_period'] = full_results

    # Calculate improvement
    if baseline_result and ml_result:
        results['improvement'] = {
            'return_diff': ml_result['return'] - baseline_result['return'],
            'win_rate_diff': ml_result['win_rate'] - baseline_result['win_rate'],
            'trade_reduction': baseline_result['trades'] - ml_result['trades']
        }

    # === Test 2: Random Windows ===
    print("\n--- Random Window Tests (10 windows) ---")

    windows = generate_random_windows(df, n_windows=10, min_days=60, max_days=90)

    for i, (start_idx, end_idx) in enumerate(windows):
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        window_df = prepare_regime_predictions(window_df)

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

    # === Test 3: Walk-Forward Analysis ===
    print("\n--- Walk-Forward Analysis ---")
    results['walk_forward'] = walk_forward_test(df, period_name)

    # === Statistical Tests ===
    if baseline_result and ml_result:
        results['statistics'] = calculate_statistical_tests(ml_result, baseline_result)

    return results


# =============================================================================
# SUCCESS CRITERIA EVALUATION
# =============================================================================

def evaluate_success_criteria(results: Dict) -> Dict:
    """
    Evaluate whether results meet success criteria.

    Args:
        results: Dict with backtest results

    Returns:
        Dict with pass/fail status for each criterion
    """
    if not results.get('full_period'):
        return {'overall': False, 'error': 'No results'}

    ml_result = next((r for r in results['full_period'] if 'ML' in r['strategy']), None)
    if not ml_result:
        return {'overall': False, 'error': 'No ML strategy result'}

    criteria = {}

    # Positive return
    criteria['positive_return'] = {
        'passed': ml_result['return'] > 0,
        'value': ml_result['return'],
        'threshold': '>0'
    }

    # Win rate
    criteria['win_rate'] = {
        'passed': ml_result['win_rate'] >= SUCCESS_CRITERIA['min_win_rate'],
        'value': ml_result['win_rate'],
        'threshold': f">={SUCCESS_CRITERIA['min_win_rate']*100:.0f}%"
    }

    # Sharpe ratio
    criteria['sharpe'] = {
        'passed': ml_result['sharpe'] >= SUCCESS_CRITERIA['min_sharpe'],
        'value': ml_result['sharpe'],
        'threshold': f">={SUCCESS_CRITERIA['min_sharpe']}"
    }

    # Max drawdown
    criteria['max_drawdown'] = {
        'passed': ml_result['max_dd'] <= SUCCESS_CRITERIA['max_drawdown'],
        'value': ml_result['max_dd'],
        'threshold': f"<={SUCCESS_CRITERIA['max_drawdown']*100:.0f}%"
    }

    # Trade count (normalized to 90 days)
    days = results.get('days', 90)
    trades_per_90d = ml_result['trades'] * (90 / days) if days > 0 else 0
    criteria['trade_count'] = {
        'passed': trades_per_90d >= SUCCESS_CRITERIA['min_trades'],
        'value': trades_per_90d,
        'threshold': f">={SUCCESS_CRITERIA['min_trades']} per 90d"
    }

    # Overall pass
    criteria['overall'] = all(c['passed'] for c in criteria.values() if isinstance(c, dict))

    return criteria


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_period_tests() -> Dict:
    """
    Run comprehensive robustness testing across all periods.

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("TRENDSTRENGTHML_50 ROBUSTNESS TESTING")
    print("Multi-Period Analysis")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_results = {}

    # Run tests for each period
    for period_name, config in PERIODS.items():
        results = run_period_backtest(period_name, config)
        if results:
            # Evaluate success criteria
            results['criteria'] = evaluate_success_criteria(results)
            all_results[period_name] = results

    return all_results


def print_summary(all_results: Dict):
    """Print comprehensive summary of all results."""

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL PERIODS")
    print("=" * 60)

    if not all_results:
        print("No results to summarize.")
        return

    # Build summary table
    summary_data = []
    for period_name, results in all_results.items():
        if results and results.get('full_period'):
            ml_result = next((r for r in results['full_period']
                            if 'ML' in r['strategy']), None)
            if ml_result:
                summary_data.append({
                    'Period': period_name,
                    'Type': results['type'].upper(),
                    'Days': results.get('days', 0),
                    'Return': ml_result['return'],
                    'Sharpe': ml_result['sharpe'],
                    'MaxDD': ml_result['max_dd'],
                    'Trades': ml_result['trades'],
                    'WinRate': ml_result['win_rate']
                })

    if summary_data:
        # Print table
        print(f"\n{'Period':<25} {'Type':<6} {'Days':<6} {'Return':>10} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'Trades':>7} {'WinRate':>8}")
        print("-" * 88)

        for row in summary_data:
            print(f"{row['Period']:<25} {row['Type']:<6} {row['Days']:<6.0f} "
                  f"{row['Return']*100:>+9.2f}% {row['Sharpe']:>8.2f} "
                  f"{row['MaxDD']*100:>7.1f}% {row['Trades']:>7} "
                  f"{row['WinRate']*100:>7.1f}%")

        # Aggregate stats
        print("\n--- Aggregate Statistics ---")
        avg_return = np.mean([r['Return'] for r in summary_data])
        avg_sharpe = np.mean([r['Sharpe'] for r in summary_data])
        avg_win_rate = np.mean([r['WinRate'] for r in summary_data])
        worst_dd = max(r['MaxDD'] for r in summary_data)

        print(f"Average Return:   {avg_return*100:+.2f}%")
        print(f"Average Sharpe:   {avg_sharpe:.2f}")
        print(f"Average Win Rate: {avg_win_rate*100:.1f}%")
        print(f"Worst Drawdown:   {worst_dd*100:.1f}%")

        # Success criteria check
        print("\n--- Success Criteria Check ---")
        all_positive = all(r['Return'] > 0 for r in summary_data)
        all_high_wr = all(r['WinRate'] >= SUCCESS_CRITERIA['min_win_rate'] for r in summary_data)
        all_good_sharpe = all(r['Sharpe'] >= SUCCESS_CRITERIA['min_sharpe'] for r in summary_data)
        all_low_dd = all(r['MaxDD'] <= SUCCESS_CRITERIA['max_drawdown'] for r in summary_data)

        print(f"  All periods positive return: {'PASS' if all_positive else 'FAIL'}")
        print(f"  All periods WR >= 60%:       {'PASS' if all_high_wr else 'FAIL'}")
        print(f"  All periods Sharpe >= 0.5:   {'PASS' if all_good_sharpe else 'FAIL'}")
        print(f"  All periods MaxDD <= 20%:    {'PASS' if all_low_dd else 'FAIL'}")

        overall_pass = all_positive and all_high_wr and all_good_sharpe and all_low_dd
        print(f"\n  OVERALL: {'PASS - Strategy is ROBUST' if overall_pass else 'FAIL - Strategy needs improvement'}")

    # Random window consistency
    print("\n--- Random Window Consistency ---")
    for period_name, results in all_results.items():
        if results and results.get('random_windows'):
            windows = results['random_windows']
            returns = [w['return'] for w in windows]
            win_rates = [w['win_rate'] for w in windows if w['trades'] > 0]

            print(f"\n  {period_name}:")
            print(f"    Return range: {min(returns)*100:+.2f}% to {max(returns)*100:+.2f}%")
            print(f"    Return std:   {np.std(returns)*100:.2f}%")
            if win_rates:
                print(f"    WinRate range: {min(win_rates)*100:.1f}% to {max(win_rates)*100:.1f}%")
            positive_windows = sum(1 for r in returns if r > 0)
            print(f"    Positive windows: {positive_windows}/{len(windows)}")


def save_results(all_results: Dict, output_dir: str = "backtest/results_multiperiod"):
    """Save results to JSON file."""

    # Handle path
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    cleaned_results = clean_for_json(all_results)

    output_path = os.path.join(output_dir, "robustness_results.json")
    with open(output_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Also save a summary CSV
    summary_path = os.path.join(output_dir, "robustness_summary.csv")
    summary_rows = []

    for period_name, results in all_results.items():
        if results and results.get('full_period'):
            for strategy_result in results['full_period']:
                summary_rows.append({
                    'period': period_name,
                    'market_type': results['type'],
                    'strategy': strategy_result['strategy'],
                    'return_pct': strategy_result['return'] * 100,
                    'sharpe': strategy_result['sharpe'],
                    'max_drawdown_pct': strategy_result['max_dd'] * 100,
                    'trades': strategy_result['trades'],
                    'win_rate_pct': strategy_result['win_rate'] * 100
                })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Summary CSV saved to {summary_path}")


def run_walk_forward_all_periods(train_models: bool = True) -> Dict:
    """
    Run walk-forward testing with XGBoost training for all periods.

    This is the recommended way to test robustness - each period gets its own
    model trained on the training portion of that period's data.

    Args:
        train_models: Whether to train fresh models for each period

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("TRENDSTRENGTHML_50 WALK-FORWARD ROBUSTNESS TESTING")
    print("With Period-Specific XGBoost Training")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 60)

    if not ML_AVAILABLE:
        print("\nWARNING: ML modules not available. Install xgboost to enable walk-forward training.")
        print("Falling back to using pre-trained model (if available).\n")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_results = {}

    for period_name, config in PERIODS.items():
        print(f"\n{'='*60}")
        print(f"PERIOD: {period_name.upper()}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Handle path
        data_path = config['path']
        if not os.path.isabs(data_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, data_path)

        if not os.path.exists(data_path):
            print(f"ERROR: Data not found at {data_path}")
            continue

        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples ({len(df)/24:.0f} days)")

        # Run walk-forward test with training
        results = walk_forward_test(df, period_name, train_model=train_models)

        # Add metadata
        results['period'] = period_name
        results['type'] = config['type']
        results['description'] = config['description']

        all_results[period_name] = results

    return all_results


def print_walk_forward_summary(all_results: Dict):
    """Print summary of walk-forward test results."""

    print("\n" + "=" * 60)
    print("WALK-FORWARD TEST SUMMARY")
    print("=" * 60)

    if not all_results:
        print("No results to summarize.")
        return

    # Focus on TEST split results (out-of-sample)
    print(f"\n{'Period':<25} {'Type':<6} {'Test Return':>12} {'Test WR':>10} {'Trades':>8}")
    print("-" * 70)

    for period_name, results in all_results.items():
        if results and results.get('splits', {}).get('test'):
            test = results['splits']['test']
            print(f"{period_name:<25} {results['type'].upper():<6} "
                  f"{test['return']*100:>+11.2f}% {test['win_rate']*100:>9.1f}% "
                  f"{test['trades']:>8}")

    # Check success on TEST splits
    print("\n--- Out-of-Sample (Test Split) Success Check ---")
    test_results = []
    for period_name, results in all_results.items():
        if results and results.get('splits', {}).get('test'):
            test_results.append(results['splits']['test'])

    if test_results:
        all_positive = all(r['return'] > 0 for r in test_results)
        all_high_wr = all(r['win_rate'] >= 0.5 for r in test_results)  # Lower bar for OOS

        print(f"  All periods positive return (OOS): {'PASS' if all_positive else 'FAIL'}")
        print(f"  All periods WR >= 50% (OOS):       {'PASS' if all_high_wr else 'FAIL'}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Period Robustness Testing")
    parser.add_argument('--period', type=str, choices=list(PERIODS.keys()),
                        help='Test specific period only')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    parser.add_argument('--walk-forward', '-wf', action='store_true',
                        help='Run walk-forward testing with period-specific model training')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip model training in walk-forward mode (use existing models)')

    args = parser.parse_args()

    if args.walk_forward:
        # Walk-forward testing with period-specific training
        all_results = run_walk_forward_all_periods(train_models=not args.no_train)
        print_walk_forward_summary(all_results)
        if not args.no_save:
            save_results(all_results, output_dir="backtest/results_multiperiod/walk_forward")
    elif args.period:
        # Test single period
        results = run_period_backtest(args.period, PERIODS[args.period])
        if results:
            results['criteria'] = evaluate_success_criteria(results)
            all_results = {args.period: results}
            print_summary(all_results)
            if not args.no_save:
                save_results(all_results)
    else:
        # Test all periods with existing model
        all_results = run_all_period_tests()
        print_summary(all_results)
        if not args.no_save:
            save_results(all_results)


if __name__ == "__main__":
    main()
