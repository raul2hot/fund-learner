#!/usr/bin/env python3
"""
Train XGBoost classifier for trade quality prediction.

Predicts: Will this trade be a WIN (1) or LOSS (0)?

Training approach:
- Train on ~275 days of data (365 - 90 test days)
- Validate on held-out portion
- Test on last 90 days (same period as backtest)
"""

import os
import sys
import json

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from typing import Tuple, List, Optional

from ml.feature_engineering import TRADE_FEATURES


# Default XGBoost parameters (conservative to prevent overfitting)
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
    'random_state': 42,
}


def load_and_split_data(
    data_path: str = "data/ml/trade_labels.csv",
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load labeled data and split by time (no shuffling).

    Args:
        data_path: Path to labeled trade data CSV
        test_ratio: Fraction of data for testing
        val_ratio: Fraction of data for validation

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    df = pd.read_csv(data_path)

    # Sort by entry time to ensure temporal order
    if '_entry_time' in df.columns:
        df = df.sort_values('_entry_time').reset_index(drop=True)

    # Get features and labels (exclude metadata columns starting with _)
    feature_cols = [c for c in TRADE_FEATURES if c in df.columns]

    # Handle missing features
    missing_features = [c for c in TRADE_FEATURES if c not in df.columns]
    if missing_features:
        print(f"Warning: Missing features (will use 0): {missing_features}")
        for feat in missing_features:
            df[feat] = 0
        feature_cols = TRADE_FEATURES

    X = df[feature_cols].values
    y = df['label'].values

    # Time-based split (no shuffling to avoid look-ahead bias)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    # Ensure at least 1 sample in each split
    if n_train < 1 or n_val < 1 or n_test < 1:
        print(f"Warning: Very small dataset ({n} samples). Using minimal splits.")
        n_train = max(1, n - 2)
        n_val = max(1, min(n - n_train - 1, 1))
        n_test = n - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Train WIN rate: {y_train.mean():.1%}")
    if len(y_test) > 0:
        print(f"Test WIN rate: {y_test.mean():.1%}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_xgboost(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    params: Optional[dict] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    verbose: bool = True,
) -> xgb.Booster:
    """
    Train XGBoost classifier with early stopping.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        feature_names: List of feature names
        params: XGBoost parameters (uses DEFAULT_PARAMS if None)
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Stop if no improvement after this many rounds
        verbose: Whether to print training progress

    Returns:
        Trained XGBoost Booster model
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Handle class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos > 0 and n_neg > 0:
        pos_weight = n_neg / n_pos
        params['scale_pos_weight'] = pos_weight
        if verbose:
            print(f"Class balance: WIN={n_pos}, LOSS={n_neg}, weight={pos_weight:.2f}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=20 if verbose else False,
    )

    return model


def evaluate_model(
    model: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        threshold: Probability threshold for WIN prediction
        verbose: Whether to print results

    Returns:
        Dictionary of evaluation metrics
    """
    if len(X_test) == 0:
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}

    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_prob = model.predict(dtest)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    # AUC requires at least 2 classes
    if len(np.unique(y_test)) > 1:
        metrics['auc'] = roc_auc_score(y_test, y_prob)
    else:
        metrics['auc'] = 0.5

    if verbose:
        print(f"\n{'='*50}")
        print(f"TEST RESULTS (threshold={threshold})")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.1%}")
        print(f"Precision: {metrics['precision']:.1%}")
        print(f"Recall:    {metrics['recall']:.1%}")
        print(f"F1 Score:  {metrics['f1']:.3f}")
        print(f"AUC:       {metrics['auc']:.3f}")

        if len(y_test) > 1:
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN'], zero_division=0))

    return metrics


def plot_feature_importance(
    model: xgb.Booster,
    output_path: str = "figures/xgb_importance.png",
    max_features: int = 20,
) -> None:
    """
    Plot and save feature importance.

    Args:
        model: Trained XGBoost model
        output_path: Path to save the plot
        max_features: Maximum number of features to show
    """
    try:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax, max_num_features=max_features, importance_type='gain')
        plt.title("XGBoost Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved feature importance to {output_path}")
    except ImportError:
        print("Warning: matplotlib not available, skipping feature importance plot")
    except Exception as e:
        print(f"Warning: Could not save feature importance plot: {e}")


def save_model(
    model: xgb.Booster,
    feature_names: List[str],
    model_path: str = "checkpoints/trade_classifier.json",
    features_path: str = "checkpoints/trade_classifier_features.json",
) -> None:
    """
    Save model and feature names for inference.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names used
        model_path: Path to save the model
        features_path: Path to save feature names
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to {features_path}")


def main(
    data_path: str = "data/ml/trade_labels.csv",
    model_output: str = "checkpoints/trade_classifier.json",
    plot_importance: bool = True,
) -> Tuple[xgb.Booster, dict]:
    """
    Main training pipeline.

    Args:
        data_path: Path to labeled trade data
        model_output: Path to save trained model
        plot_importance: Whether to plot feature importance

    Returns:
        Trained model and evaluation metrics
    """
    print("="*60)
    print("XGBOOST TRADE CLASSIFIER TRAINING")
    print("="*60)

    # Load data
    print("\nLoading labeled trade data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_data(data_path)

    # Check if we have enough data
    if len(X_train) < 3:
        print("\nERROR: Not enough training data!")
        print("Please run generate_trade_labels.py first to create labeled data.")
        return None, {}

    # Train
    print("\nTraining XGBoost...")
    model = train_xgboost(X_train, X_val, y_train, y_val, feature_names)

    # Evaluate at different thresholds
    print("\nEvaluating model at different thresholds...")
    metrics = {}
    for threshold in [0.5, 0.6, 0.7]:
        m = evaluate_model(model, X_test, y_test, feature_names, threshold)
        metrics[f'threshold_{threshold}'] = m

    # Save model
    save_model(
        model,
        feature_names,
        model_path=model_output,
        features_path=model_output.replace('.json', '_features.json')
    )

    # Plot importance
    if plot_importance:
        plot_feature_importance(model)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost trade classifier")
    parser.add_argument(
        "--data", "-d",
        default="data/ml/trade_labels.csv",
        help="Path to labeled trade data CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default="checkpoints/trade_classifier.json",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip feature importance plot"
    )

    args = parser.parse_args()

    main(
        data_path=args.data,
        model_output=args.output,
        plot_importance=not args.no_plot,
    )
