#!/usr/bin/env python
"""
Threshold Analysis Script

Analyzes your data to find optimal labeling thresholds.
Run this BEFORE prepare_data.py to determine the best thresholds.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def analyze_thresholds(df: pd.DataFrame):
    """
    Analyze what threshold combinations give reasonable distributions.
    df needs: open, high, low, close columns
    """
    # Compute metrics
    df = df.copy()
    df['return'] = (df['close'] - df['open']) / df['open']
    df['mae_long'] = (df['open'] - df['low']) / df['open']
    df['mae_short'] = (df['high'] - df['open']) / df['open']

    # Shift for "next candle"
    df['next_return'] = df['return'].shift(-1)
    df['next_mae_long'] = df['mae_long'].shift(-1)
    df['next_mae_short'] = df['mae_short'].shift(-1)
    df = df.dropna()

    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"\nTotal candles: {len(df)}")

    # Test different threshold combinations
    configs = [
        (0.015, 0.005, "Strict"),
        (0.012, 0.008, "Moderate"),
        (0.010, 0.010, "Relaxed A"),
        (0.008, 0.012, "Relaxed B"),
        (0.006, 0.012, "Aggressive A"),
        (0.006, 0.015, "Aggressive B"),
        (0.005, 0.015, "Very Aggressive A"),
        (0.005, 0.020, "Very Aggressive B"),
        (0.004, 0.020, "Ultra Aggressive"),
    ]

    print(f"\n{'Config':<20} {'Move%':<8} {'MAE%':<8} {'HIGH_BULL':<12} {'LOW_BEAR':<12} {'Total%':<10}")
    print("-" * 70)

    for move_thresh, mae_thresh, name in configs:
        # HIGH_BULL: return > move_thresh AND mae_long < mae_thresh
        high_bull = (
            (df['next_return'] > move_thresh) &
            (df['next_mae_long'] < mae_thresh)
        ).sum()

        # LOW_BEAR: return < -move_thresh AND mae_short < mae_thresh
        low_bear = (
            (df['next_return'] < -move_thresh) &
            (df['next_mae_short'] < mae_thresh)
        ).sum()

        total_pct = (high_bull + low_bear) / len(df) * 100

        print(f"{name:<20} {move_thresh*100:<8.1f} {mae_thresh*100:<8.1f} "
              f"{high_bull:<12} {low_bear:<12} {total_pct:<10.1f}%")

    # Distribution of returns
    print("\n" + "=" * 70)
    print("RETURN DISTRIBUTION (Next Candle)")
    print("=" * 70)
    print(f"Mean:   {df['next_return'].mean()*100:.3f}%")
    print(f"Std:    {df['next_return'].std()*100:.3f}%")
    print(f"Min:    {df['next_return'].min()*100:.3f}%")
    print(f"Max:    {df['next_return'].max()*100:.3f}%")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = df['next_return'].quantile(p/100) * 100
        print(f"  {p:>3}th: {val:>+7.3f}%")

    # Distribution of MAE
    print("\n" + "=" * 70)
    print("MAE DISTRIBUTION (Next Candle)")
    print("=" * 70)
    print(f"\nMAE Long (drawdown if long):")
    for p in [50, 75, 90, 95, 99]:
        val = df['next_mae_long'].quantile(p/100) * 100
        print(f"  {p:>3}th: {val:>7.3f}%")

    print(f"\nMAE Short (drawdown if short):")
    for p in [50, 75, 90, 95, 99]:
        val = df['next_mae_short'].quantile(p/100) * 100
        print(f"  {p:>3}th: {val:>7.3f}%")

    # What % of strong moves have clean paths?
    print("\n" + "=" * 70)
    print("PATH QUALITY ANALYSIS")
    print("=" * 70)

    for thresh in [0.015, 0.012, 0.010, 0.008]:
        strong_bull = df['next_return'] > thresh
        strong_bear = df['next_return'] < -thresh

        if strong_bull.sum() > 0:
            print(f"\nStrong bullish candles (>{thresh*100:.1f}%): {strong_bull.sum()}")
            for mae_thresh in [0.005, 0.008, 0.010, 0.012]:
                clean_pct = (df.loc[strong_bull, 'next_mae_long'] < mae_thresh).mean() * 100
                print(f"  With MAE < {mae_thresh*100:.1f}%: {clean_pct:.1f}%")

        if strong_bear.sum() > 0:
            print(f"\nStrong bearish candles (<-{thresh*100:.1f}%): {strong_bear.sum()}")
            for mae_thresh in [0.005, 0.008, 0.010, 0.012]:
                clean_pct = (df.loc[strong_bear, 'next_mae_short'] < mae_thresh).mean() * 100
                print(f"  With MAE < {mae_thresh*100:.1f}%: {clean_pct:.1f}%")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find config that gives 8-12% tradeable
    best_config = None
    for move_thresh, mae_thresh, name in configs:
        high_bull = (
            (df['next_return'] > move_thresh) &
            (df['next_mae_long'] < mae_thresh)
        ).sum()
        low_bear = (
            (df['next_return'] < -move_thresh) &
            (df['next_mae_short'] < mae_thresh)
        ).sum()
        total_pct = (high_bull + low_bear) / len(df) * 100

        if 8 <= total_pct <= 15:
            best_config = (move_thresh, mae_thresh, name, total_pct)
            break

    if best_config:
        print(f"\nRecommended config: {best_config[2]}")
        print(f"  strong_move_threshold = {best_config[0]}")
        print(f"  clean_path_mae_threshold = {best_config[1]}")
        print(f"  Expected tradeable %: {best_config[3]:.1f}%")
    else:
        print("\nNo config in 8-15% range. Consider more aggressive thresholds.")
        print("Try: strong_move_threshold=0.008, clean_path_mae_threshold=0.012")


def main():
    DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Please run the data pipeline first.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Drop NaN in critical columns
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    analyze_thresholds(df)


if __name__ == "__main__":
    main()
