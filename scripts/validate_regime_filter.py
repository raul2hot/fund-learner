#!/usr/bin/env python
"""
Validate Regime Filter on Existing Walk-Forward Results

This script loads existing predictions from the multi-seed walk-forward validation
and applies the regime filter retroactively to analyze potential improvement.

Usage:
    python scripts/validate_regime_filter.py

    # Compare different regime presets
    python scripts/validate_regime_filter.py --compare-presets

    # Use specific preset
    python scripts/validate_regime_filter.py --preset conservative

Output:
    Prints comparison tables showing:
    - Original returns vs filtered returns per period
    - Trade reduction statistics
    - Regime distribution analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from regime_filter import (
    RegimeFilter,
    RegimeConfig,
    RegimePresets,
    MarketRegime,
    apply_regime_filter_vectorized
)

# Paths
RESULTS_DIR = Path("experiments/walk_forward")
DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

SEEDS = [42, 123, 456, 789, 1337]
PERIODS = [
    ('period_0_covid', 'COVID Crash', False),
    ('period_1_may2021', 'May 2021 Crash', True),
    ('period_2_luna', 'Luna/3AC Collapse', True),
    ('period_3_ftx', 'FTX Crash', True),
    ('period_4_etf', 'ETF Rally', True),
    ('period_5_full', 'Full Data Holdout', True),
]


def load_price_data() -> pd.DataFrame:
    """Load full price dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_predictions(seed: int, period_id: str) -> Optional[pd.DataFrame]:
    """Load predictions from a specific seed/period."""
    pred_path = RESULTS_DIR / f"seed_{seed}" / period_id / 'predictions.csv'

    if not pred_path.exists():
        return None

    df = pd.read_csv(pred_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def get_period_data(full_df: pd.DataFrame, period_id: str) -> pd.DataFrame:
    """Get price data for a specific period."""
    # Define period boundaries
    period_bounds = {
        'period_0_covid': ('2020-03-01', '2020-05-31'),
        'period_1_may2021': ('2021-05-01', '2021-07-31'),
        'period_2_luna': ('2022-05-01', '2022-07-31'),
        'period_3_ftx': ('2022-11-01', '2023-01-31'),
        'period_4_etf': ('2024-01-01', '2024-03-31'),
        'period_5_full': ('2024-10-01', '2025-12-15'),
    }

    if period_id not in period_bounds:
        raise ValueError(f"Unknown period: {period_id}")

    start, end = period_bounds[period_id]
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts = pd.Timestamp(end, tz='UTC')

    return full_df[
        (full_df['timestamp'] >= start_ts) &
        (full_df['timestamp'] <= end_ts)
    ].copy()


def calculate_returns(predictions: pd.DataFrame, with_stop_loss: float = -0.02) -> Dict:
    """
    Calculate trading returns from predictions.

    Args:
        predictions: DataFrame with 'should_trade', 'is_long', 'trade_return', 'trade_mae' columns
        with_stop_loss: Stop-loss level to apply

    Returns:
        Dictionary with metrics
    """
    trades = predictions[predictions['should_trade']].copy()
    n_total = len(predictions)
    n_trades = len(trades)

    if n_trades == 0:
        return {
            'total_return': 0.0,
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
        }

    # Apply stop-loss to returns
    returns = trades['trade_return'].copy()
    if with_stop_loss is not None and 'trade_mae' in trades.columns:
        mae = trades['trade_mae'].values
        stop_triggered = mae > abs(with_stop_loss)
        returns = np.where(stop_triggered, with_stop_loss, returns.values)
        returns = pd.Series(returns)

    return {
        'total_return': float(returns.sum() * 100),
        'n_trades': n_trades,
        'win_rate': float((returns > 0).mean() * 100),
        'avg_return': float(returns.mean() * 100),
        'trade_frequency': n_trades / n_total * 100,
    }


def apply_filter_and_calculate(
    predictions: pd.DataFrame,
    prices: pd.Series,
    funding_rates: Optional[pd.Series],
    config: RegimeConfig,
    stop_loss: float = -0.02
) -> Tuple[Dict, Dict]:
    """
    Apply regime filter and calculate metrics.

    Returns:
        Tuple of (filtered_metrics, regime_stats)
    """
    # Apply regime filter
    filtered = apply_regime_filter_vectorized(
        predictions.copy(),
        prices,
        funding_rates,
        config=config
    )

    # Calculate metrics
    metrics = calculate_returns(filtered, with_stop_loss=stop_loss)

    # Regime statistics
    regime_counts = filtered['regime'].value_counts()
    regime_stats = {
        'normal_pct': float(regime_counts.get('normal', 0) / len(filtered) * 100),
        'elevated_pct': float(regime_counts.get('elevated', 0) / len(filtered) * 100),
        'extreme_pct': float(regime_counts.get('extreme', 0) / len(filtered) * 100),
        'trades_blocked': int(filtered['regime_blocked'].sum()),
    }

    return metrics, regime_stats


def validate_single_preset(
    preset_name: str,
    config: RegimeConfig,
    full_df: pd.DataFrame,
    stop_loss: float = -0.02
) -> pd.DataFrame:
    """
    Validate a single preset across all seeds and periods.

    Returns:
        DataFrame with results for each seed/period combination
    """
    results = []

    for period_id, period_name, is_primary in PERIODS:
        try:
            # Get price data for this period
            period_df = get_period_data(full_df, period_id)
            prices = period_df.set_index('timestamp')['close']
            funding = period_df.set_index('timestamp')['funding_rate'] if 'funding_rate' in period_df.columns else None

            for seed in SEEDS:
                predictions = load_predictions(seed, period_id)
                if predictions is None:
                    continue

                # Ensure timestamp column exists
                if 'timestamp' not in predictions.columns:
                    # Try to align with period data based on index
                    predictions['timestamp'] = period_df['timestamp'].iloc[:len(predictions)].values

                # Calculate original metrics
                original_metrics = calculate_returns(predictions, with_stop_loss=stop_loss)

                # Apply filter and calculate filtered metrics
                filtered_metrics, regime_stats = apply_filter_and_calculate(
                    predictions, prices, funding, config, stop_loss
                )

                results.append({
                    'preset': preset_name,
                    'seed': seed,
                    'period_id': period_id,
                    'period_name': period_name,
                    'is_primary': is_primary,
                    'original_return': original_metrics['total_return'],
                    'filtered_return': filtered_metrics['total_return'],
                    'improvement': filtered_metrics['total_return'] - original_metrics['total_return'],
                    'original_trades': original_metrics['n_trades'],
                    'filtered_trades': filtered_metrics['n_trades'],
                    'trades_blocked': regime_stats['trades_blocked'],
                    'trade_reduction_pct': (1 - filtered_metrics['n_trades'] / max(original_metrics['n_trades'], 1)) * 100,
                    'regime_normal_pct': regime_stats['normal_pct'],
                    'regime_elevated_pct': regime_stats['elevated_pct'],
                    'regime_extreme_pct': regime_stats['extreme_pct'],
                })

        except Exception as e:
            print(f"  Error processing {period_name}: {e}")

    return pd.DataFrame(results)


def print_summary_table(results_df: pd.DataFrame, preset_name: str):
    """Print formatted summary table for a preset."""
    print(f"\n{'='*100}")
    print(f"REGIME FILTER VALIDATION: {preset_name.upper()}")
    print(f"{'='*100}")

    # Aggregate by period (mean ± std across seeds)
    period_agg = results_df.groupby(['period_id', 'period_name', 'is_primary']).agg({
        'original_return': ['mean', 'std'],
        'filtered_return': ['mean', 'std'],
        'improvement': ['mean', 'std'],
        'original_trades': 'mean',
        'filtered_trades': 'mean',
        'trades_blocked': 'mean',
        'regime_extreme_pct': 'mean',
    }).reset_index()

    # Flatten column names
    period_agg.columns = [
        'period_id', 'period_name', 'is_primary',
        'orig_mean', 'orig_std',
        'filt_mean', 'filt_std',
        'imp_mean', 'imp_std',
        'orig_trades', 'filt_trades', 'blocked',
        'extreme_pct'
    ]

    print(f"\n{'Period':<20} {'Original Return':>18} {'Filtered Return':>18} {'Improvement':>15} {'Trades':>15} {'Extreme%':>10}")
    print("-" * 100)

    for _, row in period_agg.iterrows():
        status = "PRIMARY" if row['is_primary'] else "BONUS"
        improved = "+" if row['imp_mean'] > 0 else ""
        print(
            f"{row['period_name']:<20} "
            f"{row['orig_mean']:>+7.2f}%±{row['orig_std']:>5.2f}% "
            f"{row['filt_mean']:>+7.2f}%±{row['filt_std']:>5.2f}% "
            f"{improved}{row['imp_mean']:>+6.2f}% "
            f"{int(row['filt_trades']):>5}/{int(row['orig_trades']):<5} "
            f"{row['extreme_pct']:>8.1f}%"
        )

    # Overall statistics for primary periods
    primary = results_df[results_df['is_primary']]

    print("\n" + "-" * 100)
    print("PRIMARY PERIODS SUMMARY:")
    print(f"  Average original return: {primary['original_return'].mean():+.2f}% ± {primary['original_return'].std():.2f}%")
    print(f"  Average filtered return: {primary['filtered_return'].mean():+.2f}% ± {primary['filtered_return'].std():.2f}%")
    print(f"  Average improvement:     {primary['improvement'].mean():+.2f}%")
    print(f"  Trade reduction:         {primary['trade_reduction_pct'].mean():.1f}%")


def compare_presets(full_df: pd.DataFrame, stop_loss: float = -0.02):
    """Compare all regime filter presets."""
    presets = {
        'conservative': RegimePresets.conservative(),
        'moderate': RegimePresets.moderate(),
        'aggressive': RegimePresets.aggressive(),
    }

    all_results = []

    for name, config in presets.items():
        print(f"\nValidating {name} preset...")
        results = validate_single_preset(name, config, full_df, stop_loss)
        all_results.append(results)
        print_summary_table(results, name)

    # Combined comparison for May 2021 specifically
    combined = pd.concat(all_results)
    may2021 = combined[combined['period_id'] == 'period_1_may2021']

    print("\n" + "=" * 100)
    print("MAY 2021 COMPARISON (PROBLEM PERIOD)")
    print("=" * 100)
    print(f"\n{'Preset':<15} {'Original':>12} {'Filtered':>12} {'Improvement':>12} {'Trades Blocked':>15}")
    print("-" * 70)

    for preset in presets.keys():
        preset_data = may2021[may2021['preset'] == preset]
        print(
            f"{preset:<15} "
            f"{preset_data['original_return'].mean():>+10.2f}% "
            f"{preset_data['filtered_return'].mean():>+10.2f}% "
            f"{preset_data['improvement'].mean():>+10.2f}% "
            f"{int(preset_data['trades_blocked'].mean()):>10}/{int(preset_data['original_trades'].mean())}"
        )

    # Save detailed results
    output_path = RESULTS_DIR / 'regime_filter_validation.csv'
    combined.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate regime filter on existing walk-forward results'
    )
    parser.add_argument(
        '--compare-presets',
        action='store_true',
        help='Compare all regime filter presets'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='moderate',
        choices=['conservative', 'moderate', 'aggressive'],
        help='Regime preset to validate (default: moderate)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=-0.02,
        help='Stop-loss level (default: -0.02)'
    )

    args = parser.parse_args()

    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Run walk_forward_validation.py first to generate results.")
        sys.exit(1)

    # Load price data
    print("Loading price data...")
    try:
        full_df = load_price_data()
        print(f"  Loaded {len(full_df):,} candles")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.compare_presets:
        compare_presets(full_df, args.stop_loss)
    else:
        presets = {
            'conservative': RegimePresets.conservative(),
            'moderate': RegimePresets.moderate(),
            'aggressive': RegimePresets.aggressive(),
        }
        config = presets[args.preset]
        results = validate_single_preset(args.preset, config, full_df, args.stop_loss)
        print_summary_table(results, args.preset)

        # Save results
        output_path = RESULTS_DIR / f'regime_filter_validation_{args.preset}.csv'
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
