#!/usr/bin/env python
"""
Compare static vs adaptive threshold strategies across all walk-forward periods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import logging

from sph_net.models.two_stage import TwoStageModel, CalibratedTwoStageModel, apply_stop_loss_to_returns
from sph_net.config import SPHNetConfig
from data.dataset import TradingDataset
from features.feature_pipeline import FeaturePipeline
from labeling.candle_classifier import CandleLabeler, LabelingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")
WALK_FORWARD_DIR = Path("experiments/walk_forward")
OUTPUT_DIR = Path("experiments/walk_forward/comparison")


def load_period_model(period_id: str) -> tuple:
    """Load a trained model for a specific period."""
    model_path = WALK_FORWARD_DIR / period_id / "best_model.pt"

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    model = TwoStageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, config


def load_predictions(period_id: str) -> pd.DataFrame:
    """Load saved predictions for a period."""
    pred_path = WALK_FORWARD_DIR / period_id / "predictions.csv"

    if not pred_path.exists():
        logger.warning(f"Predictions not found: {pred_path}")
        return None

    return pd.read_csv(pred_path)


def simulate_strategy(
    predictions_df: pd.DataFrame,
    base_threshold: float,
    use_adaptive: bool,
    trend_efficiency_values: np.ndarray = None,
    vol_ratio_values: np.ndarray = None,
    stop_loss_pct: float = -0.02,
) -> dict:
    """
    Simulate a trading strategy on predictions.

    Args:
        predictions_df: DataFrame with trade_prob, trade_return columns
        base_threshold: Base threshold (0.55, 0.60, etc.)
        use_adaptive: Whether to use adaptive threshold
        trend_efficiency_values: Optional array of trend efficiency values
        vol_ratio_values: Optional array of vol ratio values
        stop_loss_pct: Stop-loss percentage

    Returns:
        dict with performance metrics
    """
    df = predictions_df.copy()
    n_samples = len(df)

    # Compute threshold for each sample
    thresholds = np.full(n_samples, base_threshold)

    if use_adaptive:
        # Adaptive threshold logic (mirrors CalibratedTwoStageModel)
        TREND_EFFICIENCY_LOW = 0.3
        TREND_EFFICIENCY_MED = 0.5
        VOL_RATIO_HIGH = 1.3
        CHOPPY_ADJUSTMENT = 0.05
        VERY_CHOPPY_ADJUSTMENT = 0.10
        HIGH_VOL_ADJUSTMENT = 0.05
        MAX_THRESHOLD = 0.70

        if trend_efficiency_values is not None:
            very_choppy = trend_efficiency_values < TREND_EFFICIENCY_LOW
            somewhat_choppy = (trend_efficiency_values >= TREND_EFFICIENCY_LOW) & \
                              (trend_efficiency_values < TREND_EFFICIENCY_MED)

            thresholds = np.where(very_choppy, thresholds + VERY_CHOPPY_ADJUSTMENT, thresholds)
            thresholds = np.where(somewhat_choppy, thresholds + CHOPPY_ADJUSTMENT, thresholds)

        if vol_ratio_values is not None:
            high_vol = vol_ratio_values > VOL_RATIO_HIGH
            thresholds = np.where(high_vol, thresholds + HIGH_VOL_ADJUSTMENT, thresholds)

        thresholds = np.clip(thresholds, None, MAX_THRESHOLD)

    # Determine which trades to take
    should_trade = df['trade_prob'].values >= thresholds
    trades = df[should_trade].copy()

    n_trades = len(trades)

    if n_trades == 0:
        return {
            'n_samples': n_samples,
            'n_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'avg_threshold': float(thresholds.mean()),
        }

    # Apply stop-loss if configured
    if stop_loss_pct is not None:
        mae_values = trades['trade_mae'].values if 'trade_mae' in trades.columns else None
        sl_results = apply_stop_loss_to_returns(
            trades['trade_return'].values,
            stop_loss_pct=stop_loss_pct,
            mae_values=mae_values,
        )
        returns = sl_results['adjusted_returns']
        n_stopped = int(sl_results['n_stopped'])
    else:
        returns = trades['trade_return'].values
        n_stopped = 0

    # Compute metrics
    total_return = returns.sum() * 100
    win_rate = (returns > 0).mean() * 100

    # Sharpe ratio (annualized)
    if returns.std() > 0:
        trades_per_year = n_trades * 4  # Assuming 3-month test periods
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        'n_samples': n_samples,
        'n_trades': n_trades,
        'total_return': float(total_return),
        'win_rate': float(win_rate),
        'sharpe_ratio': float(sharpe),
        'n_stopped': n_stopped,
        'avg_threshold': float(thresholds[should_trade].mean()) if should_trade.any() else base_threshold,
    }


def run_comparison():
    """Run comparison across all periods and strategies."""

    strategies = {
        'static_055': {'use_adaptive': False, 'base_threshold': 0.55},
        'static_060': {'use_adaptive': False, 'base_threshold': 0.60},
        'static_065': {'use_adaptive': False, 'base_threshold': 0.65},
        'adaptive': {'use_adaptive': True, 'base_threshold': 0.55},
    }

    periods = ['period_1_may2021', 'period_2_luna', 'period_3_ftx', 'period_4_etf']

    results = []

    for period_id in periods:
        logger.info(f"\nProcessing {period_id}...")

        # Load predictions
        predictions_df = load_predictions(period_id)
        if predictions_df is None:
            logger.warning(f"Skipping {period_id} - no predictions found")
            continue

        # For adaptive threshold, we need regime features
        # These would need to be saved in predictions or re-computed
        # For now, simulate without them (adaptive threshold will use None values)
        trend_efficiency = None
        vol_ratio = None

        for strategy_name, strategy_config in strategies.items():
            metrics = simulate_strategy(
                predictions_df,
                base_threshold=strategy_config['base_threshold'],
                use_adaptive=strategy_config['use_adaptive'],
                trend_efficiency_values=trend_efficiency,
                vol_ratio_values=vol_ratio,
            )

            results.append({
                'period': period_id,
                'strategy': strategy_name,
                **metrics
            })

            logger.info(f"  {strategy_name}: Return={metrics['total_return']:+.2f}%, "
                        f"Trades={metrics['n_trades']}, WinRate={metrics['win_rate']:.1f}%")

    results_df = pd.DataFrame(results)

    # Output results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False)

    # Pivot for easy comparison
    print("\n" + "="*80)
    print("TOTAL RETURN BY PERIOD AND STRATEGY (%)")
    print("="*80)
    pivot = results_df.pivot(index='period', columns='strategy', values='total_return')
    print(pivot.round(2).to_string())

    # Summary
    print("\n" + "="*80)
    print("STRATEGY SUMMARY (Primary Periods Only)")
    print("="*80)

    print(f"\n{'Strategy':<15} {'Avg Return':>12} {'Profitable':>12} {'Avg Sharpe':>12} {'Avg Trades':>12}")
    print("-"*65)

    for strategy in strategies.keys():
        strategy_results = results_df[results_df['strategy'] == strategy]
        if len(strategy_results) == 0:
            continue

        avg_return = strategy_results['total_return'].mean()
        n_profitable = (strategy_results['total_return'] > 0).sum()
        avg_sharpe = strategy_results['sharpe_ratio'].mean()
        avg_trades = strategy_results['n_trades'].mean()

        print(f"{strategy:<15} {avg_return:>+11.2f}% {n_profitable:>8}/4     {avg_sharpe:>11.2f} {avg_trades:>12.0f}")

    # Detailed period breakdown
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for period_id in periods:
        period_results = results_df[results_df['period'] == period_id]
        if len(period_results) == 0:
            continue

        print(f"\n{period_id}:")
        print(f"  {'Strategy':<15} {'Return':>10} {'Trades':>8} {'WinRate':>10} {'Sharpe':>8}")
        print(f"  {'-'*55}")

        for _, row in period_results.iterrows():
            print(f"  {row['strategy']:<15} {row['total_return']:>+9.2f}% {row['n_trades']:>8} "
                  f"{row['win_rate']:>9.1f}% {row['sharpe_ratio']:>8.2f}")

    # Find best strategy
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    best_by_avg_return = results_df.groupby('strategy')['total_return'].mean().idxmax()
    best_by_profitable = results_df.groupby('strategy').apply(
        lambda x: (x['total_return'] > 0).sum()
    ).idxmax()

    print(f"\nBest by Average Return: {best_by_avg_return}")
    print(f"Best by # Profitable Periods: {best_by_profitable}")

    return results_df


def main():
    print("\n" + "="*80)
    print("THRESHOLD STRATEGY COMPARISON")
    print("="*80)

    # Check for walk-forward results
    if not WALK_FORWARD_DIR.exists():
        logger.error(f"Walk-forward directory not found: {WALK_FORWARD_DIR}")
        logger.error("Run walk_forward_validation.py first!")
        return

    results_df = run_comparison()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
