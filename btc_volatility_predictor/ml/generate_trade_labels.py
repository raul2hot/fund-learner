#!/usr/bin/env python3
"""
Generate WIN/LOSS labels from historical trades.

This script:
1. Loads full 365-day feature data (or available data)
2. Simulates TrendStrength strategy on entire dataset
3. Records features at each trade entry
4. Labels each trade as WIN (P&L > 0) or LOSS (P&L <= 0)
5. Saves labeled dataset for XGBoost training
"""

import os
import sys

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Optional

from backtest.engine import BacktestEngine
from backtest.strategies.simple_adx import TrendStrengthStrategy
from ml.feature_engineering import CORE_FEATURES, compute_additional_features


def generate_labels(
    data_path: str = "data/processed/features_365d.csv",
    predictions_path: Optional[str] = None,
    output_path: str = "data/ml/trade_labels.csv",
    strategy_class=TrendStrengthStrategy,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate trade labels by running strategy on historical data.

    Args:
        data_path: Path to feature data CSV
        predictions_path: Optional path to predictions CSV (for regime labels)
        output_path: Path to save labeled trade data
        strategy_class: Strategy class to use for generating trades
        verbose: Whether to print progress

    Returns:
        DataFrame with:
        - Features at trade entry
        - Label: 1 = WIN, 0 = LOSS
        - Trade metadata (entry_price, exit_price, pnl, etc.)
    """

    # Try multiple data paths
    data_paths_to_try = [
        data_path,
        "backtest/results_v3/test_predictions_90d.csv",
        "backtest/results_v2/test_predictions_90d.csv",
        "../data/processed/features_365d.csv",
    ]

    df = None
    used_path = None
    for path in data_paths_to_try:
        if os.path.exists(path):
            if verbose:
                print(f"Loading data from {path}...")
            df = pd.read_csv(path)
            used_path = path
            break

    if df is None:
        raise FileNotFoundError(
            f"Could not find data file. Tried: {data_paths_to_try}"
        )

    if verbose:
        print(f"Loaded {len(df)} samples from {used_path}")

    # If predictions available, merge them
    if predictions_path and os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        if 'predicted_regime' in pred_df.columns:
            # Use predictions for the overlapping period
            if verbose:
                print(f"Loaded predictions from {predictions_path}")

    # For data without predictions, generate using simple volatility threshold
    if 'predicted_regime' not in df.columns:
        if 'target_volatility' in df.columns:
            vol_threshold = df['target_volatility'].median()
            df['predicted_regime'] = (df['target_volatility'] > vol_threshold).astype(int)
            if verbose:
                print(f"Generated regime predictions using volatility threshold: {vol_threshold:.4f}")
        else:
            # Use volatility proxy if available
            vol_col = None
            for col in ['vol_realized_24h', 'vol_gk_24h', 'atr_14']:
                if col in df.columns:
                    vol_col = col
                    break

            if vol_col:
                vol_threshold = df[vol_col].median()
                df['predicted_regime'] = (df[vol_col] > vol_threshold).astype(int)
                if verbose:
                    print(f"Generated regime predictions using {vol_col} threshold: {vol_threshold:.4f}")
            else:
                # Default to LOW vol regime (0)
                df['predicted_regime'] = 0
                if verbose:
                    print("Warning: No volatility column found, defaulting to LOW regime")

    if verbose:
        print(f"\nRunning {strategy_class.__name__} strategy on {len(df)} samples...")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=10000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    strategy = strategy_class()
    result = engine.run(strategy, df)

    if verbose:
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
        history = data_records[max(0, entry_idx - 168):entry_idx]

        # Extract core features
        features = {}
        for feat in CORE_FEATURES:
            features[feat] = row.get(feat, 0)

        # Add computed features
        features.update(compute_additional_features(row, history))

        # Add label: WIN (1) if P&L > 0, else LOSS (0)
        features['label'] = 1 if trade.pnl > 0 else 0

        # Add metadata (for analysis, not training)
        features['_entry_time'] = entry_idx
        features['_exit_time'] = trade.exit_time
        features['_entry_price'] = trade.entry_price
        features['_exit_price'] = trade.exit_price
        features['_pnl'] = trade.pnl
        features['_pnl_pct'] = trade.pnl_pct
        features['_holding_period'] = trade.holding_period
        features['_regime_at_entry'] = trade.regime_at_entry

        labeled_data.append(features)

    # Create DataFrame
    labels_df = pd.DataFrame(labeled_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    labels_df.to_csv(output_path, index=False)

    # Summary
    if verbose:
        print(f"\n{'='*50}")
        print("LABEL GENERATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total trades: {len(labels_df)}")
        if len(labels_df) > 0:
            wins = labels_df['label'].sum()
            losses = len(labels_df) - wins
            print(f"Wins:   {wins} ({labels_df['label'].mean():.1%})")
            print(f"Losses: {losses}")
            print(f"Features: {len(CORE_FEATURES) + 5}")  # +5 for computed features
        print(f"Saved to: {output_path}")

    return labels_df


def generate_more_trades(
    data_path: str = "data/processed/features_365d.csv",
    output_path: str = "data/ml/trade_labels_extended.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate more trades by running multiple strategy variants.

    This is useful when the base strategy produces too few trades
    for effective ML training.
    """
    from backtest.strategies.simple_adx import (
        TrendStrengthStrategy,
        SimpleADXTrend,
        VolTrendCombo
    )

    all_labels = []

    strategies = [
        (TrendStrengthStrategy, {}),
        (SimpleADXTrend, {'require_adx': False}),
        (VolTrendCombo, {}),
    ]

    for strategy_class, kwargs in strategies:
        try:
            labels = generate_labels(
                data_path=data_path,
                output_path=f"/tmp/{strategy_class.__name__}_labels.csv",
                strategy_class=lambda: strategy_class(**kwargs),
                verbose=verbose
            )
            labels['_strategy'] = strategy_class.__name__
            all_labels.append(labels)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate labels for {strategy_class.__name__}: {e}")

    if all_labels:
        combined = pd.concat(all_labels, ignore_index=True)
        # Remove duplicates based on entry time
        combined = combined.drop_duplicates(subset=['_entry_time'])
        combined.to_csv(output_path, index=False)

        if verbose:
            print(f"\nCombined {len(combined)} unique trades from {len(all_labels)} strategies")
            print(f"Saved to: {output_path}")

        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate trade labels for XGBoost training")
    parser.add_argument(
        "--data", "-d",
        default="data/processed/features_365d.csv",
        help="Path to feature data CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/ml/trade_labels.csv",
        help="Output path for labeled data"
    )
    parser.add_argument(
        "--extended", "-e",
        action="store_true",
        help="Generate extended labels using multiple strategies"
    )

    args = parser.parse_args()

    if args.extended:
        generate_more_trades(
            data_path=args.data,
            output_path=args.output,
        )
    else:
        generate_labels(
            data_path=args.data,
            output_path=args.output,
        )
