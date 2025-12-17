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
    calculate_individual_seed_returns,
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


def get_weight_period(test_period: str) -> Optional[str]:
    """
    Get appropriate weighting period (must be BEFORE test period).

    For true walk-forward validation, we can only use performance data
    from periods that occurred BEFORE the test period.
    """
    # Can't weight by future data - use equal weights for early periods
    if test_period in ['period_0_covid', 'period_1_may2021']:
        return None  # Use equal weights
    # For all other periods, weight by May 2021 crash performance
    # This is the first major crash with enough data to assess model robustness
    return 'period_1_may2021'


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
    stop_loss: float = -0.02,
    use_regime_filter: bool = True
) -> Dict:
    """
    Evaluate ensemble on a single period.

    Uses the same return calculation methodology as individual seeds:
    - Open-to-close returns (next_return column)
    - MAE-aware stop-loss detection
    - Regime filtering for high volatility periods
    """
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

    # Calculate trading returns using same methodology as individual seeds
    # This ensures apples-to-apples comparison
    metrics = calculate_ensemble_trading_returns(
        predictions,
        period_data,
        stop_loss_pct=stop_loss,
        use_regime_filter=use_regime_filter
    )

    # Add ensemble-specific metrics
    metrics['avg_agreement'] = predictions['agreement'].mean()
    metrics['avg_confidence'] = predictions['confidence'].mean()
    metrics['trade_rate'] = predictions['should_trade'].mean() * 100

    # Add volatility filtering stats
    if 'volatility_filtered' in predictions.columns:
        metrics['n_volatility_filtered'] = int(predictions['volatility_filtered'].sum())
        metrics['pct_volatility_filtered'] = predictions['volatility_filtered'].mean() * 100

    # Add agreement filtering stats
    if 'agreement_filtered' in predictions.columns:
        metrics['n_agreement_filtered'] = int(predictions['agreement_filtered'].sum())
        metrics['pct_agreement_filtered'] = predictions['agreement_filtered'].mean() * 100

    # Compute individual seed returns using SAME methodology (apples-to-apples)
    individual_same_method = calculate_individual_seed_returns(
        predictions,
        period_data,
        stop_loss_pct=stop_loss
    )
    metrics['individual_same_method'] = individual_same_method

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
    parser.add_argument('--regime-filter', action='store_true', default=True,
                        dest='regime_filter',
                        help='Apply regime filter during validation (default: enabled)')
    parser.add_argument('--no-regime-filter', action='store_false',
                        dest='regime_filter',
                        help='Disable regime filter')
    parser.add_argument('--agreement-threshold', type=float, default=0.0,
                        help='Min model agreement to trade (0.0=disabled, 0.7=require 70%% agreement)')
    parser.add_argument('--dynamic-agreement', action='store_true', default=False,
                        help='Enable dynamic agreement threshold based on volatility')
    parser.add_argument('--threshold-normal', type=float, default=0.70,
                        help='Agreement threshold for normal volatility (default: 0.70)')
    parser.add_argument('--threshold-crisis', type=float, default=0.95,
                        help='Agreement threshold for high volatility/crisis (default: 0.95)')
    parser.add_argument('--crisis-percentile', type=float, default=0.90,
                        help='Volatility percentile to trigger crisis mode (default: 0.90)')
    parser.add_argument('--single-seed', type=int, default=None,
                        help='DIAGNOSTIC: Run with only one seed to verify calculation parity')
    args = parser.parse_args()

    # Handle single-seed diagnostic mode
    if args.single_seed is not None:
        print("=" * 100)
        print(f"DIAGNOSTIC MODE: Single-seed ensemble (seed={args.single_seed})")
        print("=" * 100)
        print("Purpose: Verify that ensemble calculation matches individual seed methodology")
        print("Expected: Single-seed ensemble return should equal saved individual return")
        print()
        # Override to use only the specified seed
        global SEEDS
        SEEDS = [args.single_seed]

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
    print(f"Strategy: Walk-forward validation using period-appropriate models (no look-ahead bias)")
    print(f"Methodology: Open-to-close returns with MAE-aware stop-loss (same as individual seeds)")
    print(f"Stop-loss: {args.stop_loss * 100:.1f}% (MAE-aware)")
    print(f"Regime filter: {'ENABLED (moderate)' if args.regime_filter else 'DISABLED'}")
    if args.dynamic_agreement:
        print(f"Agreement threshold: DYNAMIC (normal={args.threshold_normal:.0%}, crisis={args.threshold_crisis:.0%})")
        print(f"  Crisis mode triggers when volatility > {args.crisis_percentile:.0%} percentile")
    elif args.agreement_threshold > 0:
        print(f"Agreement threshold: {args.agreement_threshold:.0%} (only trade when models agree)")
    else:
        print(f"Agreement threshold: DISABLED (trade even when models disagree)")
    if args.method == 'weighted':
        print(f"Default weight by: {args.weight_by} (temperature={args.temperature})")
        print(f"Note: Early periods use equal weights (no prior crash data available)")
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
            # Create ensemble using period-appropriate models (walk-forward validation)
            # Uses models trained BEFORE each test period to avoid look-ahead bias
            weight_period = get_weight_period(period_id)
            effective_method = method if weight_period else EnsembleMethod.MEAN

            if weight_period:
                print(f"  Creating {args.method} ensemble (models={period_id}, weights={weight_period})...")
            else:
                print(f"  Creating MEAN ensemble (models={period_id}, no prior crash data for weighting)...")

            ensemble = create_ensemble_from_walk_forward(
                RESULTS_DIR,
                method=effective_method,
                period=period_id,  # Use period-appropriate models (no look-ahead bias)
                weight_by_period=weight_period,  # Weight by prior crash performance (or None)
                seeds=SEEDS,
                device=device,
                temperature=args.temperature,
                agreement_threshold=args.agreement_threshold,
                dynamic_agreement_threshold=args.dynamic_agreement,
                agreement_threshold_normal=args.threshold_normal,
                agreement_threshold_crisis=args.threshold_crisis,
                crisis_volatility_percentile=args.crisis_percentile
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
                args.stop_loss,
                use_regime_filter=args.regime_filter
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
            # Show filtering stats
            n_stopped = ensemble_metrics.get('n_stopped_out', 0)
            n_regime = ensemble_metrics.get('n_regime_blocked', 0)
            n_agreement = ensemble_metrics.get('n_agreement_filtered', 0)
            filter_stats = []
            if n_stopped > 0:
                filter_stats.append(f"Stop-loss: {n_stopped}")
            if n_regime > 0:
                filter_stats.append(f"Regime: {n_regime}")
            if n_agreement > 0:
                filter_stats.append(f"Low agreement: {n_agreement}")
            if filter_stats:
                print(f"  Trades filtered: {', '.join(filter_stats)}")
            if 'n_volatility_filtered' in ensemble_metrics:
                print(f"  Volatility filtered: {ensemble_metrics['n_volatility_filtered']} "
                      f"({ensemble_metrics['pct_volatility_filtered']:.1f}%)")

            # Individual seed returns (saved results - may use different methodology)
            if any(individual_returns.values()):
                print(f"\n  Individual seed returns (saved - may differ):")
                for seed in sorted(individual_returns.keys()):
                    ret = individual_returns[seed]
                    weight = ensemble.config.weights.get(seed, 0)
                    print(f"    Seed {seed}: {ret:>+8.2f}% (weight={weight:.3f})")

            # APPLES-TO-APPLES comparison using same methodology
            same_method = ensemble_metrics.get('individual_same_method', {})
            if same_method:
                same_method_returns = list(same_method.values())
                avg_same_method = np.mean(same_method_returns) if same_method_returns else 0
                improvement_same_method = ensemble_metrics['total_return'] - avg_same_method

                print(f"\n  APPLES-TO-APPLES comparison (same methodology):")
                for seed in sorted(same_method.keys()):
                    ret = same_method[seed]
                    weight = ensemble.config.weights.get(seed, 0)
                    print(f"    Seed {seed}: {ret:>+8.2f}% (weight={weight:.3f})")
                print(f"    ----------------------------------------")
                print(f"    Avg individual:    {avg_same_method:>+8.2f}%")
                print(f"    Ensemble:          {ensemble_metrics['total_return']:>+8.2f}%")
                print(f"    Improvement:       {improvement_same_method:>+8.2f}%")

                if improvement_same_method > 0:
                    print(f"    ✓ Ensemble BEATS average by {improvement_same_method:.2f}%")
                elif abs(improvement_same_method) < 1.0:
                    print(f"    ~ Ensemble MATCHES average (diff < 1%)")
                else:
                    print(f"    ✗ Ensemble UNDERPERFORMS by {-improvement_same_method:.2f}%")

            # DIAGNOSTIC: Single-seed comparison
            if args.single_seed is not None:
                saved_return = individual_returns.get(args.single_seed, 0)
                ensemble_return = ensemble_metrics['total_return']
                diff = abs(ensemble_return - saved_return)

                print(f"\n  {'='*60}")
                print(f"  DIAGNOSTIC: Single-seed calculation parity check")
                print(f"  {'='*60}")
                print(f"  Saved individual result:    {saved_return:>+10.2f}%")
                print(f"  Single-seed ensemble:       {ensemble_return:>+10.2f}%")
                print(f"  Difference:                 {diff:>10.2f}%")

                if diff < 0.1:
                    print(f"  Status: ✓ MATCH - Calculations are equivalent")
                elif diff < 1.0:
                    print(f"  Status: ~ CLOSE - Minor difference (likely rounding)")
                else:
                    print(f"  Status: ✗ MISMATCH - Calculation methodologies differ!")
                    print(f"  ")
                    print(f"  Possible causes:")
                    print(f"    1. Different return formula (open-to-close vs close-to-close)")
                    print(f"    2. Different stop-loss application (MAE vs simple clip)")
                    print(f"    3. Different regime filtering")
                    print(f"    4. Different trade threshold or timing")

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
