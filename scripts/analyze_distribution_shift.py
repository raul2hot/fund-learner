#!/usr/bin/env python
"""
Distribution Shift Analysis Script

Diagnoses distribution shift between training and test periods.
This helps understand why the model might be failing on test data.

Usage:
    python scripts/analyze_distribution_shift.py

    # Test on volatile 2022 period
    python scripts/analyze_distribution_shift.py --preset volatile_2022

    # Custom date split
    python scripts/analyze_distribution_shift.py --train-end 2022-06-01 --val-end 2022-11-01
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import logging

from data.data_splitter import TemporalSplitter, SplitConfig, VOLATILE_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_volatility_by_period(df: pd.DataFrame, timestamp_col: str = 'timestamp'):
    """Analyze volatility by year/month to identify regime changes."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['year_month'] = df[timestamp_col].dt.to_period('M')

    # Compute returns if not present
    if 'return_1h' not in df.columns:
        df['return_1h'] = df['close'].pct_change()

    # Group by year
    yearly = df.groupby('year').agg({
        'return_1h': ['std', 'mean', 'count'],
        'label': lambda x: ((x == 0) | (x == 4)).mean() * 100  # tradeable %
    }).round(4)

    print("\n" + "="*70)
    print("VOLATILITY BY YEAR")
    print("="*70)
    print(yearly.to_string())

    # Group by quarter
    df['quarter'] = df[timestamp_col].dt.to_period('Q')
    quarterly = df.groupby('quarter').agg({
        'return_1h': 'std',
        'label': lambda x: ((x == 0) | (x == 4)).mean() * 100
    }).round(4)
    quarterly.columns = ['volatility', 'tradeable_pct']

    print("\n" + "="*70)
    print("VOLATILITY BY QUARTER")
    print("="*70)

    # Highlight high/low volatility quarters
    vol_mean = quarterly['volatility'].mean()
    vol_std = quarterly['volatility'].std()

    for idx, row in quarterly.iterrows():
        vol = row['volatility']
        trade_pct = row['tradeable_pct']
        if vol > vol_mean + vol_std:
            status = "HIGH VOL"
        elif vol < vol_mean - vol_std:
            status = "LOW VOL"
        else:
            status = ""
        print(f"{idx}: vol={vol:.4f}, tradeable={trade_pct:.1f}%  {status}")

    return yearly, quarterly


def compare_presets(df: pd.DataFrame):
    """Compare different split presets."""
    print("\n" + "="*70)
    print("COMPARING SPLIT PRESETS")
    print("="*70)

    results = {}

    for preset_name, preset_info in VOLATILE_PRESETS.items():
        print(f"\n--- {preset_name}: {preset_info['description']} ---")

        config = SplitConfig(preset=preset_name)
        splitter = TemporalSplitter(config)

        try:
            train_df, val_df, test_df = splitter.split(df.copy())

            # Compute metrics
            if 'return_1h' not in train_df.columns:
                train_df['return_1h'] = train_df['close'].pct_change()
                val_df['return_1h'] = val_df['close'].pct_change()
                test_df['return_1h'] = test_df['close'].pct_change()

            analysis = splitter.analyze_distribution_shift(train_df, val_df, test_df)

            results[preset_name] = {
                'train_vol': analysis['train']['volatility'],
                'test_vol': analysis['test']['volatility'],
                'vol_ratio': analysis['shift_metrics']['volatility_ratio'],
                'train_tradeable': analysis['train'].get('tradeable_pct', 0),
                'test_tradeable': analysis['test'].get('tradeable_pct', 0),
                'trade_ratio': analysis['shift_metrics']['tradeable_ratio'],
                'train_samples': analysis['train']['n_samples'],
                'test_samples': analysis['test']['n_samples'],
            }

            # Print summary
            print(f"  Train vol: {results[preset_name]['train_vol']:.4f}, Test vol: {results[preset_name]['test_vol']:.4f}")
            print(f"  Vol ratio: {results[preset_name]['vol_ratio']:.2f}")
            print(f"  Train tradeable: {results[preset_name]['train_tradeable']:.1f}%, Test tradeable: {results[preset_name]['test_tradeable']:.1f}%")
            print(f"  Trade ratio: {results[preset_name]['trade_ratio']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[preset_name] = None

    # Summary table
    print("\n" + "="*70)
    print("PRESET COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Preset':<20} {'Vol Ratio':>12} {'Trade Ratio':>12} {'Recommendation':>20}")
    print("-"*70)

    for preset_name, metrics in results.items():
        if metrics is None:
            print(f"{preset_name:<20} {'ERROR':>12}")
            continue

        vol_ratio = metrics['vol_ratio']
        trade_ratio = metrics['trade_ratio']

        # Recommendation based on ratios
        if 0.8 <= vol_ratio <= 1.2 and 0.8 <= trade_ratio <= 1.2:
            rec = "RECOMMENDED"
        elif 0.7 <= vol_ratio <= 1.5 and 0.7 <= trade_ratio <= 1.5:
            rec = "OK"
        else:
            rec = "HIGH SHIFT"

        print(f"{preset_name:<20} {vol_ratio:>12.2f} {trade_ratio:>12.2f} {rec:>20}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze distribution shift')
    parser.add_argument('--data', type=str, default='prepared_data/full_data.parquet',
                       help='Path to data file')
    parser.add_argument('--preset', type=str, default=None,
                       choices=list(VOLATILE_PRESETS.keys()),
                       help='Split preset to use')
    parser.add_argument('--train-end', type=str, default=None,
                       help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--val-end', type=str, default=None,
                       help='Validation end date (YYYY-MM-DD)')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all presets')

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        # Try alternative paths
        alternatives = [
            Path('prepared_data/train.parquet'),
            Path('data/processed/btc_hourly.parquet'),
        ]
        for alt in alternatives:
            if alt.exists():
                data_path = alt
                break

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'python scripts/prepare_data.py' first to generate data")
        return

    logger.info(f"Loading data from {data_path}")

    # Load all data if we have split files
    if 'train' in str(data_path):
        train_df = pd.read_parquet(data_path)
        val_path = data_path.parent / 'val.parquet'
        test_path = data_path.parent / 'test.parquet'
        if val_path.exists() and test_path.exists():
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        else:
            df = train_df
    else:
        df = pd.read_parquet(data_path)

    logger.info(f"Loaded {len(df):,} rows")

    # Analyze by period
    yearly, quarterly = analyze_volatility_by_period(df)

    # Compare all presets if requested
    if args.compare_all:
        compare_presets(df)
        return

    # Create split config
    if args.preset:
        config = SplitConfig(preset=args.preset)
    elif args.train_end and args.val_end:
        config = SplitConfig(
            train_end_date=args.train_end,
            val_end_date=args.val_end
        )
    else:
        config = SplitConfig()  # Default 70/15/15

    # Split and analyze
    splitter = TemporalSplitter(config)
    train_df, val_df, test_df = splitter.split(df)

    # Compute returns if needed
    for split_df in [train_df, val_df, test_df]:
        if 'return_1h' not in split_df.columns:
            split_df['return_1h'] = split_df['close'].pct_change()

    # Print analysis
    splitter.print_distribution_analysis(train_df, val_df, test_df)

    # Quick volatility comparison
    print("\n" + "="*70)
    print("QUICK VOLATILITY CHECK")
    print("="*70)
    train_vol = train_df['return_1h'].std() * 100
    test_vol = test_df['return_1h'].std() * 100
    ratio = test_vol / train_vol

    print(f"Train period volatility: {train_vol:.4f}%")
    print(f"Test period volatility:  {test_vol:.4f}%")
    print(f"Ratio (test/train):      {ratio:.2f}x")

    if ratio < 0.7:
        print("\n⚠️  TEST IS MUCH CALMER than training!")
        print("   The model learned from volatile markets but is being tested on calm markets.")
        print("   Consider using --preset volatile_2022 or volatile_2021 for fair comparison.")
    elif ratio > 1.5:
        print("\n⚠️  TEST IS MUCH MORE VOLATILE than training!")
        print("   The model may struggle with higher volatility than it was trained on.")


if __name__ == "__main__":
    main()
