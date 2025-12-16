#!/usr/bin/env python
"""
Analyze predictions across walk-forward periods to understand
why May 2021 failed and what threshold would have helped.

Key Finding: The model is "confidently wrong" in May 2021:
- HIGH confidence trades LOSE money
- LOW confidence trades are PROFITABLE

This suggests we need to detect when model confidence is unreliable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json

# Load all period predictions
PERIODS = {
    'covid': 'experiments/walk_forward/period_0_covid/predictions.csv',
    'may2021': 'experiments/walk_forward/period_1_may2021/predictions.csv',
    'luna': 'experiments/walk_forward/period_2_luna/predictions.csv',
    'ftx': 'experiments/walk_forward/period_3_ftx/predictions.csv',
    'etf': 'experiments/walk_forward/period_4_etf/predictions.csv',
}


def load_predictions():
    """Load all prediction files."""
    data = {}
    for name, path in PERIODS.items():
        if Path(path).exists():
            data[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(data[name])} samples")
        else:
            print(f"Missing: {path}")
    return data


def analyze_trade_prob_distribution(data):
    """Compare trade_prob distributions across periods."""
    print("\n" + "="*70)
    print("TRADE PROBABILITY DISTRIBUTION BY PERIOD")
    print("="*70)

    for name, df in data.items():
        trades = df[df['should_trade'] == True]
        print(f"\n{name.upper()}:")
        print(f"  Total samples: {len(df)}")
        print(f"  Trades taken: {len(trades)} ({len(trades)/len(df)*100:.1f}%)")
        if len(trades) > 0:
            print(f"  trade_prob stats (for taken trades):")
            print(f"    mean: {trades['trade_prob'].mean():.3f}")
            print(f"    std:  {trades['trade_prob'].std():.3f}")
            print(f"    min:  {trades['trade_prob'].min():.3f}")
            print(f"    25%:  {trades['trade_prob'].quantile(0.25):.3f}")
            print(f"    50%:  {trades['trade_prob'].quantile(0.50):.3f}")
            print(f"    75%:  {trades['trade_prob'].quantile(0.75):.3f}")
            print(f"    max:  {trades['trade_prob'].max():.3f}")


def threshold_sweep(df, period_name):
    """Sweep through thresholds and find optimal for a period."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP: {period_name.upper()}")
    print(f"{'='*70}")

    results = []
    for thresh in np.arange(0.50, 0.85, 0.05):
        trades = df[df['trade_prob'] >= thresh].copy()
        n_trades = len(trades)

        if n_trades == 0:
            results.append({
                'threshold': thresh,
                'n_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_return': 0,
            })
            continue

        # Calculate returns
        total_return = trades['trade_return'].sum() * 100
        win_rate = (trades['trade_return'] > 0).mean() * 100
        avg_return = trades['trade_return'].mean() * 100

        # Sharpe approximation
        if trades['trade_return'].std() > 0:
            sharpe = (trades['trade_return'].mean() / trades['trade_return'].std()) * np.sqrt(len(trades) * 4)
        else:
            sharpe = 0

        results.append({
            'threshold': thresh,
            'n_trades': n_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
        })

    results_df = pd.DataFrame(results)
    print(f"\n{'Thresh':>7} {'Trades':>8} {'TotRet%':>10} {'WinRate%':>10} {'AvgRet%':>10} {'Sharpe':>8}")
    print("-"*60)
    for _, row in results_df.iterrows():
        print(f"{row['threshold']:>7.2f} {row['n_trades']:>8} {row['total_return']:>10.2f} "
              f"{row['win_rate']:>10.1f} {row['avg_return']:>10.4f} {row.get('sharpe', 0):>8.2f}")

    # Find break-even threshold
    profitable = results_df[results_df['total_return'] > 0]
    if len(profitable) > 0:
        min_profitable_thresh = profitable['threshold'].min()
        print(f"\nMinimum threshold for profitability: {min_profitable_thresh:.2f}")
    else:
        print(f"\nNo threshold makes this period profitable")

    return results_df


def analyze_confidence_buckets(df, period_name, stop_loss_pct=-0.02):
    """Analyze trades by confidence bucket - KEY ANALYSIS for inverse confidence."""
    print(f"\n{'='*70}")
    print(f"CONFIDENCE BUCKET ANALYSIS: {period_name.upper()}")
    print(f"{'='*70}")

    trades = df[df['should_trade'] == True].copy()

    if len(trades) == 0:
        print("No trades to analyze")
        return

    # Apply stop-loss to trade returns
    trades['adjusted_return'] = trades['trade_return'].copy()
    if stop_loss_pct is not None and 'trade_mae' in trades.columns:
        stop_triggered = trades['trade_mae'] > abs(stop_loss_pct)
        trades.loc[stop_triggered, 'adjusted_return'] = stop_loss_pct

    # Split into confidence buckets
    buckets = [
        (0.55, 0.60, 'Marginal (0.55-0.60)'),
        (0.60, 0.65, 'Medium (0.60-0.65)'),
        (0.65, 0.70, 'High (0.65-0.70)'),
        (0.70, 0.75, 'Very High (0.70-0.75)'),
        (0.75, 1.00, 'Extreme (0.75+)'),
    ]

    print(f"\n{'Bucket':<25} {'Count':>8} {'TotRet%':>10} {'WinRate%':>10} {'AvgRet%':>10}")
    print("-"*70)

    bucket_results = []
    for low, high, name in buckets:
        subset = trades[(trades['trade_prob'] >= low) & (trades['trade_prob'] < high)]
        if len(subset) > 0:
            total_ret = subset['adjusted_return'].sum() * 100
            win_rate = (subset['adjusted_return'] > 0).mean() * 100
            avg_ret = subset['adjusted_return'].mean() * 100
            print(f"{name:<25} {len(subset):>8} {total_ret:>10.2f} {win_rate:>10.1f} {avg_ret:>10.4f}")
            bucket_results.append({
                'bucket': name,
                'low': low,
                'high': high,
                'count': len(subset),
                'total_return': total_ret,
                'win_rate': win_rate,
                'avg_return': avg_ret,
            })
        else:
            print(f"{name:<25} {0:>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    # Detect inverse confidence pattern
    if len(bucket_results) >= 2:
        low_conf_return = bucket_results[0]['total_return'] if bucket_results else 0
        high_conf_return = sum(b['total_return'] for b in bucket_results if b['low'] >= 0.65)

        if low_conf_return > 0 and high_conf_return < 0:
            print(f"\n*** INVERSE CONFIDENCE DETECTED ***")
            print(f"  Low confidence (0.55-0.60): {low_conf_return:+.2f}% - PROFITABLE")
            print(f"  High confidence (0.65+): {high_conf_return:+.2f}% - LOSING")
            print(f"  Recommendation: Cap maximum confidence or filter high confidence trades")

    return bucket_results


def analyze_confidence_cap_strategy(df, period_name, stop_loss_pct=-0.02):
    """Test strategy of capping maximum confidence (filtering out very high confidence)."""
    print(f"\n{'='*70}")
    print(f"CONFIDENCE CAP STRATEGY: {period_name.upper()}")
    print(f"{'='*70}")

    trades = df[df['should_trade'] == True].copy()

    if len(trades) == 0:
        print("No trades to analyze")
        return

    # Apply stop-loss
    trades['adjusted_return'] = trades['trade_return'].copy()
    if stop_loss_pct is not None and 'trade_mae' in trades.columns:
        stop_triggered = trades['trade_mae'] > abs(stop_loss_pct)
        trades.loc[stop_triggered, 'adjusted_return'] = stop_loss_pct

    print(f"\n{'Max Cap':>8} {'Trades':>8} {'TotRet%':>10} {'WinRate%':>10} {'AvgRet%':>10}")
    print("-"*55)

    # Test different max confidence caps
    for max_cap in [1.00, 0.75, 0.70, 0.65, 0.60]:
        capped = trades[(trades['trade_prob'] >= 0.55) & (trades['trade_prob'] <= max_cap)]
        if len(capped) > 0:
            total_ret = capped['adjusted_return'].sum() * 100
            win_rate = (capped['adjusted_return'] > 0).mean() * 100
            avg_ret = capped['adjusted_return'].mean() * 100
            print(f"{max_cap:>8.2f} {len(capped):>8} {total_ret:>10.2f} {win_rate:>10.1f} {avg_ret:>10.4f}")


def find_optimal_window_strategy(df, period_name):
    """Find optimal trade_prob window (min, max) for each period."""
    print(f"\n{'='*70}")
    print(f"OPTIMAL CONFIDENCE WINDOW: {period_name.upper()}")
    print(f"{'='*70}")

    trades = df[df['should_trade'] == True].copy()

    if len(trades) == 0:
        print("No trades to analyze")
        return None

    best_return = float('-inf')
    best_window = (0.55, 1.00)

    # Grid search over windows
    for min_thresh in np.arange(0.55, 0.75, 0.05):
        for max_thresh in np.arange(min_thresh + 0.05, 1.05, 0.05):
            subset = trades[(trades['trade_prob'] >= min_thresh) & (trades['trade_prob'] <= max_thresh)]
            if len(subset) > 10:  # Need minimum trades
                total_ret = subset['trade_return'].sum() * 100
                if total_ret > best_return:
                    best_return = total_ret
                    best_window = (min_thresh, max_thresh)

    print(f"Best window: [{best_window[0]:.2f}, {best_window[1]:.2f}] -> {best_return:+.2f}% return")
    return best_window


def main():
    data = load_predictions()

    if not data:
        print("No prediction files found. Run walk_forward_validation.py first.")
        return

    # 1. Compare distributions
    analyze_trade_prob_distribution(data)

    # 2. Threshold sweep for each period
    all_sweeps = {}
    for name, df in data.items():
        all_sweeps[name] = threshold_sweep(df, name)

    # 3. Confidence bucket analysis (KEY for May 2021)
    print("\n" + "="*80)
    print("CONFIDENCE BUCKET ANALYSIS - KEY DIAGNOSTIC")
    print("="*80)
    for name, df in data.items():
        analyze_confidence_buckets(df, name)

    # 4. Test confidence cap strategy
    print("\n" + "="*80)
    print("CONFIDENCE CAP STRATEGY - POTENTIAL FIX")
    print("="*80)
    for name, df in data.items():
        analyze_confidence_cap_strategy(df, name)

    # 5. Find optimal confidence windows
    print("\n" + "="*80)
    print("OPTIMAL CONFIDENCE WINDOWS BY PERIOD")
    print("="*80)
    windows = {}
    for name, df in data.items():
        windows[name] = find_optimal_window_strategy(df, name)

    # 6. Find optimal global threshold
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLD ACROSS ALL PERIODS")
    print("="*70)

    for thresh in np.arange(0.55, 0.75, 0.05):
        total_return = 0
        n_periods_profitable = 0

        for name, df in data.items():
            if name == 'covid':  # Skip non-primary
                continue
            trades = df[df['trade_prob'] >= thresh]
            if len(trades) > 0:
                period_return = trades['trade_return'].sum() * 100
                total_return += period_return
                if period_return > 0:
                    n_periods_profitable += 1

        print(f"Threshold {thresh:.2f}: {n_periods_profitable}/4 profitable, Total: {total_return:+.2f}%")

    # Save analysis results
    output_dir = Path('experiments/walk_forward/analysis')
    output_dir.mkdir(exist_ok=True, parents=True)

    for name, sweep_df in all_sweeps.items():
        sweep_df.to_csv(output_dir / f'{name}_threshold_sweep.csv', index=False)

    print(f"\nSaved sweep results to {output_dir}")


if __name__ == "__main__":
    main()
