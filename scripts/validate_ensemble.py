#!/usr/bin/env python
"""
Validate Ensemble Model on Walk-Forward Periods

Compares ensemble predictions against individual seed predictions
to measure improvement in crash resistance.

The key insight: Seed 1337 was profitable in May 2021 (+15.91%)
while other seeds lost -47% to -78%. By averaging predictions,
seed 1337's crash-resistant signals dampen the bad signals.

Usage:
    python scripts/validate_ensemble.py
    python scripts/validate_ensemble.py --method weighted
    python scripts/validate_ensemble.py --method voting
    python scripts/validate_ensemble.py --method mean --period period_1_may2021

Output:
    experiments/walk_forward/ensemble_validation_{method}.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from sph_net.config import SPHNetConfig
from sph_net.models.two_stage import TwoStageModel
from sph_net.ensemble import (
    EnsemblePredictor,
    EnsembleConfig,
    EnsembleMethod,
    ModelLoader,
    PerformanceWeightedEnsemble,
    load_validation_results,
    create_ensemble_from_walk_forward,
    ensemble_predict_dataframe,
    calculate_ensemble_trading_returns,
)
from features.feature_pipeline import FeaturePipeline
from labeling.candle_classifier import CandleLabeler, LabelingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("experiments/walk_forward")
DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")

SEEDS = [42, 123, 456, 789, 1337]
WINDOW_SIZE = 64

# Labeling configuration - must match training
LABELING_CONFIG = LabelingConfig(
    strong_move_threshold=0.010,    # 1.0%
    weak_move_threshold=0.004,      # 0.4%
    clean_path_mae_threshold=0.010  # 1.0%
)

# Test periods (aligned with walk_forward_validation.py)
PERIODS = [
    ('period_0_covid', 'COVID Crash', False, '2020-03-01', '2020-05-31', '2020-02-29'),
    ('period_1_may2021', 'May 2021 Crash', True, '2021-05-01', '2021-07-31', '2021-04-30'),
    ('period_2_luna', 'Luna/3AC Collapse', True, '2022-05-01', '2022-07-31', '2022-04-30'),
    ('period_3_ftx', 'FTX Crash', True, '2022-11-01', '2023-01-31', '2022-10-31'),
    ('period_4_etf', 'ETF Rally', True, '2024-01-01', '2024-03-31', '2023-12-31'),
    ('period_5_full', 'Full Data Holdout', True, '2024-10-01', '2025-12-15', '2024-09-30'),
]


def load_raw_data() -> pd.DataFrame:
    """Load raw price dataset."""
    logger.info(f"Loading data from {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Drop rows with NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna(subset=critical_cols)

    logger.info(f"Loaded {len(df):,} rows, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    return df


def prepare_period_data(
    full_df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
    feature_pipeline: FeaturePipeline,
    labeler: CandleLabeler
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare and normalize data for a test period.

    Uses training data (up to train_end) to compute normalization statistics,
    then applies them to test data.

    Returns:
        (test_df_normalized, price_columns, feature_columns)
    """
    train_end_ts = pd.Timestamp(train_end, tz='UTC')
    test_start_ts = pd.Timestamp(test_start, tz='UTC')
    test_end_ts = pd.Timestamp(test_end, tz='UTC')

    # Split chronologically
    train_raw = full_df[full_df['timestamp'] <= train_end_ts].copy()
    test_raw = full_df[
        (full_df['timestamp'] >= test_start_ts) &
        (full_df['timestamp'] <= test_end_ts)
    ].copy()

    logger.info(f"  Train: {len(train_raw):,} candles (for normalization)")
    logger.info(f"  Test:  {len(test_raw):,} candles")

    if len(test_raw) == 0:
        raise ValueError("No test data for this period")

    # Apply labeling (needed for some features that use labels)
    train_labeled = labeler.label_dataset(train_raw)
    test_labeled = labeler.label_dataset(test_raw)

    # Compute features
    logger.info("  Computing features...")
    train_featured = feature_pipeline.compute_all_features(train_labeled)
    test_featured = feature_pipeline.compute_all_features(test_labeled)

    # Drop warmup period
    warmup = feature_pipeline.get_warmup_periods()
    train_clean = train_featured.iloc[warmup:].copy()
    test_clean = test_featured.iloc[warmup:].copy()

    logger.info(f"  After warmup: train={len(train_clean):,}, test={len(test_clean):,}")

    # Get feature columns
    price_columns = FeaturePipeline.PRICE_FEATURES
    feature_columns = []
    for group_features in FeaturePipeline.ENGINEERED_FEATURE_GROUPS.values():
        feature_columns.extend(group_features)

    # Filter to columns that exist
    feature_columns = [col for col in feature_columns if col in test_clean.columns]
    logger.info(f"  Using {len(price_columns)} price + {len(feature_columns)} engineered features")

    # Normalize using training data statistics
    logger.info("  Normalizing features (using training stats)...")
    train_normalized, norm_stats = feature_pipeline.normalize_features(
        train_clean,
        fit_data=train_clean
    )
    test_normalized = feature_pipeline.apply_normalization(
        test_clean,
        norm_stats
    )

    return test_normalized, price_columns, feature_columns


def evaluate_ensemble_on_period(
    ensemble: EnsemblePredictor,
    period_data: pd.DataFrame,
    price_columns: List[str],
    feature_columns: List[str],
    stop_loss: float = -0.02
) -> Dict:
    """Evaluate ensemble on a single period."""
    # Generate predictions
    predictions = ensemble_predict_dataframe(
        ensemble,
        period_data,
        price_columns,
        feature_columns,
        window_size=WINDOW_SIZE,
        batch_size=256,
        return_individual=True
    )

    if len(predictions) == 0:
        return {
            'total_return': 0,
            'n_trades': 0,
            'sharpe': 0,
            'win_rate': 0,
            'avg_agreement': 0,
            'avg_confidence': 0,
            'trade_rate': 0,
        }

    # Calculate trading returns
    metrics = calculate_ensemble_trading_returns(
        predictions,
        period_data,
        stop_loss_pct=stop_loss
    )

    # Add ensemble-specific metrics
    metrics['avg_agreement'] = predictions['agreement'].mean()
    metrics['avg_confidence'] = predictions['confidence'].mean()
    metrics['trade_rate'] = predictions['should_trade'].mean() * 100

    # Add volatility filtering stats
    if 'volatility_filtered' in predictions.columns:
        metrics['n_volatility_filtered'] = int(predictions['volatility_filtered'].sum())
        metrics['pct_volatility_filtered'] = predictions['volatility_filtered'].mean() * 100

    return metrics


def load_individual_seed_results(period_id: str) -> Dict[int, float]:
    """Load individual seed returns for a period from saved results."""
    results = {}

    for seed in SEEDS:
        results[seed] = 0  # Default

        # Try seed_summary.json first (structure: {"seed": X, "results": {period: {metrics: {...}}}})
        summary_path = RESULTS_DIR / f"seed_{seed}" / "seed_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                    if 'results' in data and period_id in data['results']:
                        period_data = data['results'][period_id]
                        if isinstance(period_data, dict) and 'metrics' in period_data:
                            results[seed] = period_data['metrics'].get('total_return', 0)
                        elif isinstance(period_data, dict):
                            results[seed] = period_data.get('total_return', 0)
                        continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: try test_results.json in period folder
        results_path = RESULTS_DIR / f"seed_{seed}" / period_id / "test_results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    data = json.load(f)
                    results[seed] = data.get('total_return', 0)
            except (json.JSONDecodeError, KeyError):
                pass

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate ensemble model')
    parser.add_argument('--method', type=str, default='weighted',
                        choices=['mean', 'weighted', 'median', 'voting'],
                        help='Ensemble combination method')
    parser.add_argument('--stop-loss', type=float, default=-0.02,
                        help='Stop-loss percentage (e.g., -0.02 for -2%%)')
    parser.add_argument('--period', type=str, default=None,
                        help='Specific period to evaluate (default: all)')
    parser.add_argument('--weight-by', type=str, default='period_1_may2021',
                        help='Period to use for performance weighting')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for weighted ensemble softmax (higher = more uniform weights)')
    args = parser.parse_args()

    # Map method string to enum
    method_map = {
        'mean': EnsembleMethod.MEAN,
        'weighted': EnsembleMethod.WEIGHTED,
        'median': EnsembleMethod.MEDIAN,
        'voting': EnsembleMethod.VOTING,
    }
    method = method_map[args.method]

    print("=" * 100)
    print(f"ENSEMBLE VALIDATION - {args.method.upper()} METHOD")
    print("=" * 100)
    print(f"Stop-loss: {args.stop_loss * 100:.1f}%")
    if args.method == 'weighted':
        print(f"Weight by: {args.weight_by} (temperature={args.temperature})")
    print()

    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        print("Run walk_forward_validation.py first to generate model checkpoints.")
        return 1

    # Check for model files
    available_models = list(RESULTS_DIR.glob("seed_*/period_*/best_model.pt"))
    if not available_models:
        print(f"ERROR: No model checkpoints found in {RESULTS_DIR}")
        print("Run walk_forward_validation.py first to train models.")
        return 1

    print(f"Found {len(available_models)} model checkpoints")

    # Load raw data
    print("\nLoading raw data...")
    try:
        full_df = load_raw_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize feature pipeline and labeler
    feature_pipeline = FeaturePipeline(window_size=WINDOW_SIZE)
    labeler = CandleLabeler(LABELING_CONFIG)

    # Filter periods if specific one requested
    if args.period:
        periods = [p for p in PERIODS if p[0] == args.period]
        if not periods:
            print(f"ERROR: Unknown period '{args.period}'")
            print(f"Available: {[p[0] for p in PERIODS]}")
            return 1
    else:
        periods = PERIODS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_results = []

    for period_id, period_name, is_primary, start_date, end_date, train_end in periods:
        print(f"\n{'='*80}")
        print(f"Period: {period_name} ({period_id})")
        print(f"Date range: {start_date} to {end_date}")
        print("-" * 80)

        try:
            # Create ensemble for this period
            print(f"  Creating {args.method} ensemble...")
            ensemble = create_ensemble_from_walk_forward(
                RESULTS_DIR,
                method=method,
                period=period_id,
                weight_by_period=args.weight_by,
                seeds=SEEDS,
                device=device,
                temperature=args.temperature
            )

            # Prepare period data with proper feature computation
            print("  Preparing data...")
            test_data, price_columns, feature_columns = prepare_period_data(
                full_df,
                train_end,
                start_date,
                end_date,
                feature_pipeline,
                labeler
            )

            print(f"  Test data: {len(test_data):,} rows")

            if len(test_data) < WINDOW_SIZE + 10:
                print(f"  SKIP: Insufficient data")
                continue

            # Evaluate ensemble
            print("  Evaluating ensemble...")
            ensemble_metrics = evaluate_ensemble_on_period(
                ensemble,
                test_data,
                price_columns,
                feature_columns,
                args.stop_loss
            )

            # Load individual seed results
            individual_returns = load_individual_seed_results(period_id)
            valid_returns = [r for r in individual_returns.values() if r != 0]

            if valid_returns:
                avg_individual = np.mean(valid_returns)
                best_individual = max(valid_returns)
                worst_individual = min(valid_returns)
            else:
                avg_individual = 0
                best_individual = 0
                worst_individual = 0

            improvement = ensemble_metrics['total_return'] - avg_individual

            # Print results
            print(f"\nResults:")
            print(f"  Ensemble return:      {ensemble_metrics['total_return']:>+10.2f}%")
            print(f"  Avg individual:       {avg_individual:>+10.2f}%")
            print(f"  Best seed:            {best_individual:>+10.2f}%")
            print(f"  Worst seed:           {worst_individual:>+10.2f}%")
            print(f"  Improvement:          {improvement:>+10.2f}%")
            print(f"  Trades: {ensemble_metrics['n_trades']}, "
                  f"Sharpe: {ensemble_metrics['sharpe']:.2f}, "
                  f"Win rate: {ensemble_metrics['win_rate']:.1f}%")
            print(f"  Model agreement: {ensemble_metrics['avg_agreement']:.1%}")
            if 'n_volatility_filtered' in ensemble_metrics:
                print(f"  Volatility filtered: {ensemble_metrics['n_volatility_filtered']} "
                      f"({ensemble_metrics['pct_volatility_filtered']:.1f}%)")

            # Individual seed returns
            if any(individual_returns.values()):
                print(f"\n  Individual seed returns:")
                for seed in sorted(individual_returns.keys()):
                    ret = individual_returns[seed]
                    weight = ensemble.config.weights.get(seed, 0)
                    print(f"    Seed {seed}: {ret:>+8.2f}% (weight={weight:.3f})")

            all_results.append({
                'period_id': period_id,
                'period_name': period_name,
                'is_primary': is_primary,
                'start_date': start_date,
                'end_date': end_date,
                'ensemble_return': ensemble_metrics['total_return'],
                'ensemble_sharpe': ensemble_metrics['sharpe'],
                'ensemble_win_rate': ensemble_metrics['win_rate'],
                'ensemble_n_trades': ensemble_metrics['n_trades'],
                'avg_individual': avg_individual,
                'best_individual': best_individual,
                'worst_individual': worst_individual,
                'improvement': improvement,
                'avg_agreement': ensemble_metrics['avg_agreement'],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)

        # Overall
        print("\nAll periods:")
        print(f"  Ensemble average:    {results_df['ensemble_return'].mean():+.2f}%")
        print(f"  Individual average:  {results_df['avg_individual'].mean():+.2f}%")
        print(f"  Average improvement: {results_df['improvement'].mean():+.2f}%")

        # Primary periods only
        primary = results_df[results_df['is_primary']]
        if len(primary) > 0:
            print("\nPrimary periods only:")
            print(f"  Ensemble average:    {primary['ensemble_return'].mean():+.2f}%")
            print(f"  Individual average:  {primary['avg_individual'].mean():+.2f}%")
            print(f"  Average improvement: {primary['improvement'].mean():+.2f}%")

        # May 2021 (the problem period)
        may2021 = results_df[results_df['period_id'] == 'period_1_may2021']
        if len(may2021) > 0:
            row = may2021.iloc[0]
            print("\nMay 2021 (PROBLEM PERIOD):")
            print(f"  Ensemble return:   {row['ensemble_return']:+.2f}%")
            print(f"  Avg individual:    {row['avg_individual']:+.2f}%")
            print(f"  Improvement:       {row['improvement']:+.2f}%")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f'ensemble_validation_{args.method}.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Also save as JSON
        json_path = RESULTS_DIR / f'ensemble_validation_{args.method}.json'
        results_df.to_json(json_path, orient='records', indent=2)
        print(f"JSON saved to: {json_path}")
    else:
        print("\nNo results generated. Check errors above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
