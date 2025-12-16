#!/usr/bin/env python
"""
Analyze which regime features could predict May 2021's choppiness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Load full dataset with features
DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

def load_featured_data():
    """Load data and compute regime features."""
    from features.feature_pipeline import FeaturePipeline

    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    pipeline = FeaturePipeline()
    df = pipeline.compute_all_features(df)

    return df

def analyze_regime_by_period(df):
    """Compare regime features across different periods."""

    periods = {
        'may2021': ('2021-05-01', '2021-07-31'),
        'luna': ('2022-05-01', '2022-07-31'),
        'ftx': ('2022-11-01', '2023-01-31'),
        'etf': ('2024-01-01', '2024-03-31'),
    }

    regime_cols = [
        'vol_percentile', 'vol_ratio', 'trend_efficiency',
        'trend_efficiency_72h', 'bb_width_percentile', 'atr_percentile'
    ]

    print("\n" + "="*80)
    print("REGIME FEATURES BY PERIOD")
    print("="*80)

    results = {}
    for period_name, (start, end) in periods.items():
        mask = (df['timestamp'] >= start) & (df['timestamp'] < end)
        period_df = df[mask]

        if len(period_df) == 0:
            print(f"\n{period_name}: No data")
            continue

        print(f"\n{period_name.upper()} ({start} to {end}):")
        print(f"  Samples: {len(period_df)}")

        results[period_name] = {}
        for col in regime_cols:
            if col in period_df.columns:
                mean_val = period_df[col].mean()
                std_val = period_df[col].std()
                results[period_name][col] = mean_val
                print(f"  {col:25s}: mean={mean_val:.3f}, std={std_val:.3f}")

    # Compute difference from May 2021 to others
    print("\n" + "="*80)
    print("MAY 2021 vs OTHER PERIODS (difference in means)")
    print("="*80)

    if 'may2021' in results:
        may21 = results['may2021']
        for period_name, metrics in results.items():
            if period_name == 'may2021':
                continue
            print(f"\n{period_name}:")
            for col in regime_cols:
                if col in may21 and col in metrics:
                    diff = may21[col] - metrics[col]
                    print(f"  {col:25s}: {diff:+.3f}")

    return results

def find_discriminating_features(df):
    """Find features that separate profitable from unprofitable periods."""

    # Define periods and their outcomes
    periods = {
        'may2021': {'start': '2021-05-01', 'end': '2021-07-31', 'profitable': False},
        'luna': {'start': '2022-05-01', 'end': '2022-07-31', 'profitable': True},
        'ftx': {'start': '2022-11-01', 'end': '2023-01-31', 'profitable': True},
        'etf': {'start': '2024-01-01', 'end': '2024-03-31', 'profitable': True},
    }

    regime_cols = [
        'vol_percentile', 'vol_ratio', 'trend_efficiency',
        'trend_efficiency_72h', 'bb_width_percentile'
    ]

    profitable_data = {col: [] for col in regime_cols}
    unprofitable_data = {col: [] for col in regime_cols}

    for period_name, info in periods.items():
        mask = (df['timestamp'] >= info['start']) & (df['timestamp'] < info['end'])
        period_df = df[mask]

        for col in regime_cols:
            if col in period_df.columns:
                if info['profitable']:
                    profitable_data[col].extend(period_df[col].dropna().tolist())
                else:
                    unprofitable_data[col].extend(period_df[col].dropna().tolist())

    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION: PROFITABLE vs UNPROFITABLE PERIODS")
    print("="*80)

    print(f"\n{'Feature':<25} {'Profitable':>15} {'Unprofitable':>15} {'Diff':>10}")
    print("-"*70)

    for col in regime_cols:
        if profitable_data[col] and unprofitable_data[col]:
            prof_mean = np.mean(profitable_data[col])
            unprof_mean = np.mean(unprofitable_data[col])
            diff = unprof_mean - prof_mean
            print(f"{col:<25} {prof_mean:>15.3f} {unprof_mean:>15.3f} {diff:>+10.3f}")

def suggest_threshold_adjustment(df):
    """
    Suggest how to adjust threshold based on regime features.

    Key hypothesis: When trend_efficiency is low, the market is choppy
    and we should require higher confidence to trade.
    """

    print("\n" + "="*80)
    print("THRESHOLD ADJUSTMENT RULES")
    print("="*80)

    print("""
    Based on analysis, consider these rules:

    1. TREND EFFICIENCY BASED:
       - trend_efficiency > 0.5: Normal threshold (0.55)
       - trend_efficiency 0.3-0.5: Elevated threshold (0.60)
       - trend_efficiency < 0.3: High threshold (0.65-0.70)

    2. VOLATILITY RATIO BASED:
       - vol_ratio < 1.2: Normal threshold (0.55)
       - vol_ratio 1.2-1.5: Elevated threshold (0.60)
       - vol_ratio > 1.5: High threshold (0.65)

    3. ROLLING TRADE FREQUENCY BASED:
       - If >20 trades in last 24h: Increase threshold by 0.05
       - This is self-adaptive to model confidence patterns

    4. COMBINED (Recommended):
       base_threshold = 0.55
       if trend_efficiency_24h < 0.3:
           threshold += 0.05
       if vol_ratio > 1.3:
           threshold += 0.05
       threshold = min(threshold, 0.70)  # Cap at 0.70
    """)

def main():
    print("Loading and processing data...")
    df = load_featured_data()

    analyze_regime_by_period(df)
    find_discriminating_features(df)
    suggest_threshold_adjustment(df)

    print("\n" + "="*80)
    print("NEXT STEP: Implement adaptive threshold in CalibratedTwoStageModel")
    print("="*80)

if __name__ == "__main__":
    main()
