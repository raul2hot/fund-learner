#!/usr/bin/env python3
"""
V5 Backtest Runner - ML-Enhanced Strategies

Tests the hypothesis: XGBoost filter improves win rate by 15-25%

This script compares:
1. TrendStrength (baseline) - the best performing V4.1 strategy
2. TrendStrengthML with threshold=0.5 (50% confidence)
3. TrendStrengthML with threshold=0.6 (60% confidence)
4. TrendStrengthML with threshold=0.7 (70% confidence)

Expected results:
- Higher win rate (from ~58% to 73-83%)
- Fewer trades (8-10 instead of 12)
- Similar or better returns
- Improved Sharpe ratio

Usage:
    cd btc_volatility_predictor

    # First, generate trade labels and train the model:
    python ml/generate_trade_labels.py
    python ml/train_xgboost.py

    # Then run the backtest:
    python backtest/run_backtest_v5.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from backtest.engine import BacktestEngine, BacktestResult
from backtest.strategies import (
    BuyAndHoldStrategy,
    TrendStrengthStrategy,
    TrendStrengthWithML,
)


def find_data_file() -> str:
    """Find the best available data file for backtesting."""
    paths_to_try = [
        "backtest/results_v3/test_predictions_90d.csv",
        "backtest/results_v2/test_predictions_90d.csv",
        "data/processed/features_365d.csv",
        "../backtest/results_v3/test_predictions_90d.csv",
        "../backtest/results_v2/test_predictions_90d.csv",
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find data file. Tried: {paths_to_try}\n"
        "Please run generate_predictions.py first."
    )


def print_separator(char: str = "=", length: int = 70) -> None:
    """Print a separator line."""
    print(char * length)


def print_header(title: str) -> None:
    """Print a section header."""
    print_separator()
    print(title)
    print_separator()


def format_percent(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:+.2f}%"


def run_ml_comparison():
    """
    Run backtest comparing baseline TrendStrength with ML-enhanced versions.
    """
    print_header("V5 BACKTEST: ML-ENHANCED STRATEGIES")

    # Load data
    data_path = find_data_file()
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples ({len(df) / 24:.1f} days)")

    # Check for predicted_regime column
    if 'predicted_regime' not in df.columns:
        print("\nGenerating volatility regime predictions...")
        if 'target_volatility' in df.columns:
            vol_threshold = df['target_volatility'].median()
            df['predicted_regime'] = (df['target_volatility'] > vol_threshold).astype(int)
        else:
            # Use a volatility proxy
            for col in ['vol_realized_24h', 'vol_gk_24h', 'atr_14']:
                if col in df.columns:
                    vol_threshold = df[col].median()
                    df['predicted_regime'] = (df[col] > vol_threshold).astype(int)
                    break
            else:
                df['predicted_regime'] = 0

    # Calculate market stats
    if 'close' in df.columns:
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        print(f"Market return: {format_percent(market_return)}")

    # Check if ML model exists
    model_path = "checkpoints/trade_classifier.json"
    ml_model_exists = os.path.exists(model_path)

    if not ml_model_exists:
        print(f"\nWARNING: ML model not found at {model_path}")
        print("ML strategies will run without filtering (baseline behavior)")
        print("\nTo train the model, run:")
        print("  python ml/generate_trade_labels.py")
        print("  python ml/train_xgboost.py")
    else:
        print(f"\nML model loaded from: {model_path}")

    # Define strategies to test
    strategies = [
        BuyAndHoldStrategy(),
        TrendStrengthStrategy(),  # Baseline
        TrendStrengthWithML(ml_threshold=0.5),  # 50% confidence
        TrendStrengthWithML(ml_threshold=0.6),  # 60% confidence
        TrendStrengthWithML(ml_threshold=0.7),  # 70% confidence
    ]

    # Run backtests
    print("\n" + "-" * 70)
    print("RUNNING BACKTESTS")
    print("-" * 70)

    engine = BacktestEngine(
        initial_capital=10000,
        transaction_cost=0.001,  # 0.1% per trade
        slippage=0.0005,         # 0.05%
    )

    results: List[BacktestResult] = []
    ml_stats: Dict[str, Dict] = {}

    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")
        result = engine.run(strategy, df)
        results.append(result)

        # Print immediate results
        print(f"  Return: {format_percent(result.total_return)}")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Trades: {result.num_trades}")
        if result.num_trades > 0:
            print(f"  Win Rate: {result.win_rate * 100:.1f}%")
            print(f"  Profit Factor: {result.profit_factor:.2f}")

        # Collect ML stats if available
        if hasattr(strategy, 'get_ml_stats'):
            ml_stats[strategy.name] = strategy.get_ml_stats()

    # Print summary comparison
    print_header("RESULTS SUMMARY")

    # Table header
    header = f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'Win Rate':>10} {'PF':>8}"
    print(header)
    print("-" * 70)

    # Sort by return (descending)
    sorted_results = sorted(results, key=lambda r: r.total_return, reverse=True)

    for r in sorted_results:
        win_rate_str = f"{r.win_rate * 100:.1f}%" if r.num_trades > 0 else "N/A"
        pf_str = f"{r.profit_factor:.2f}" if r.num_trades > 0 and r.profit_factor != float('inf') else "N/A"

        print(
            f"{r.strategy_name:<25} "
            f"{format_percent(r.total_return):>10} "
            f"{r.sharpe_ratio:>8.2f} "
            f"{r.num_trades:>8} "
            f"{win_rate_str:>10} "
            f"{pf_str:>8}"
        )

    # Print ML filter statistics
    if ml_stats:
        print("\n" + "-" * 70)
        print("ML FILTER STATISTICS")
        print("-" * 70)

        for name, stats in ml_stats.items():
            if stats['ml_allows'] + stats['ml_blocks'] > 0:
                print(f"\n{name}:")
                print(f"  Trades allowed:  {stats['ml_allows']}")
                print(f"  Trades blocked:  {stats['ml_blocks']}")
                print(f"  Filter rate:     {stats['ml_filter_rate'] * 100:.1f}%")

    # Improvement analysis
    print_header("IMPROVEMENT ANALYSIS")

    # Find baseline (TrendStrength)
    baseline = next((r for r in results if r.strategy_name == "TrendStrength"), None)

    if baseline and baseline.num_trades > 0:
        print(f"\nBaseline: {baseline.strategy_name}")
        print(f"  Win Rate: {baseline.win_rate * 100:.1f}%")
        print(f"  Trades: {baseline.num_trades}")
        print(f"  Return: {format_percent(baseline.total_return)}")

        print("\nML Strategy Improvements:")
        for r in results:
            if "ML" in r.strategy_name and r.num_trades > 0:
                win_rate_change = (r.win_rate - baseline.win_rate) * 100
                trade_change = r.num_trades - baseline.num_trades
                return_change = r.total_return - baseline.total_return

                print(f"\n  {r.strategy_name}:")
                print(f"    Win Rate: {r.win_rate * 100:.1f}% ({win_rate_change:+.1f}pp)")
                print(f"    Trades:   {r.num_trades} ({trade_change:+d})")
                print(f"    Return:   {format_percent(r.total_return)} ({format_percent(return_change)})")

    # Expected vs actual
    print("\n" + "-" * 70)
    print("EXPECTED vs ACTUAL (Target: +15-25% win rate)")
    print("-" * 70)

    for r in results:
        if "ML" in r.strategy_name and baseline and baseline.num_trades > 0 and r.num_trades > 0:
            improvement = (r.win_rate - baseline.win_rate) * 100
            target_met = "YES" if 15 <= improvement <= 25 else "NO"
            if improvement > 25:
                target_met = "EXCEEDED"
            elif improvement < 0:
                target_met = "WORSE"
            print(f"  {r.strategy_name}: {improvement:+.1f}pp improvement - {target_met}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return results


def run_full_pipeline():
    """
    Run the complete ML pipeline: generate labels, train, and backtest.
    """
    print_header("FULL ML PIPELINE")

    # Step 1: Generate trade labels
    print("\nStep 1: Generating trade labels...")
    try:
        from ml.generate_trade_labels import generate_labels
        labels_df = generate_labels(verbose=True)
        print(f"Generated {len(labels_df)} trade labels")
    except Exception as e:
        print(f"Error generating labels: {e}")
        print("Skipping to backtest...")
        labels_df = None

    # Step 2: Train XGBoost (if we have labels)
    if labels_df is not None and len(labels_df) >= 5:
        print("\nStep 2: Training XGBoost classifier...")
        try:
            from ml.train_xgboost import main as train_model
            model, metrics = train_model(
                data_path="data/ml/trade_labels.csv",
                plot_importance=True
            )
            print("Model trained successfully")
        except Exception as e:
            print(f"Error training model: {e}")
    else:
        print("\nStep 2: Skipping training (not enough labeled data)")

    # Step 3: Run backtest
    print("\nStep 3: Running backtest comparison...")
    results = run_ml_comparison()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V5 Backtest: ML-Enhanced Strategies")
    parser.add_argument(
        "--full-pipeline", "-f",
        action="store_true",
        help="Run full pipeline: generate labels, train model, then backtest"
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        help="Path to data file (auto-detected if not specified)"
    )

    args = parser.parse_args()

    if args.full_pipeline:
        run_full_pipeline()
    else:
        run_ml_comparison()
