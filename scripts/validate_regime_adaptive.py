#!/usr/bin/env python
"""
Regime-Adaptive Ensemble Validation

Two-regime strategy:
- HIGH volatility → Defensive weighted ensemble (all 5 seeds) with high agreement
- LOW volatility → Aggressive single seed (seed 1337 - best performer)

The key insight: use the ensemble's crash protection during danger,
but let the best seed run during calm markets.

Usage:
    python scripts/validate_regime_adaptive.py
    python scripts/validate_regime_adaptive.py --vol-threshold 0.70
    python scripts/validate_regime_adaptive.py --aggressive-seed 1337

Output:
    experiments/walk_forward/regime_adaptive_validation.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import torch.nn.functional as F
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


def get_weight_period(test_period: str) -> Optional[str]:
    """Get appropriate weighting period (must be BEFORE test period)."""
    if test_period in ['period_0_covid', 'period_1_may2021']:
        return None  # Use equal weights
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
    """Prepare and normalize data for a test period."""
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

    # Apply labeling
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


def get_regime(price_data: pd.DataFrame, lookback: int = 168, threshold: float = 0.70) -> pd.Series:
    """
    Classify regime based on rolling volatility.

    Args:
        price_data: DataFrame with 'close' column
        lookback: Rolling window size (default 168 = 7 days of hourly candles)
        threshold: Percentile threshold for high volatility (default 0.70)

    Returns:
        Series of 'high_vol' or 'low_vol' per row
    """
    returns = price_data['close'].pct_change()
    rolling_vol = returns.rolling(lookback, min_periods=lookback // 2).std()
    vol_percentile = rolling_vol.rank(pct=True)

    # High vol = top (1-threshold)%, Low vol = bottom threshold%
    regime = pd.Series('low_vol', index=price_data.index)
    regime[vol_percentile > threshold] = 'high_vol'

    # Fill NaN at the start with 'low_vol' (conservative default)
    regime = regime.fillna('low_vol')

    return regime


def load_single_seed_model(
    results_dir: Path,
    seed: int,
    period: str,
    device: str
) -> torch.nn.Module:
    """Load a single model for the aggressive regime."""
    model_path = results_dir / f"seed_{seed}" / period / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = SPHNetConfig()

    # Instantiate model
    model = TwoStageModel(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


class RegimeAdaptivePredictor:
    """
    Regime-adaptive predictor that switches between ensemble and single seed.

    - HIGH volatility: Defensive weighted ensemble with high agreement threshold
    - LOW volatility: Aggressive single seed (best performer)
    """

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        aggressive_model: torch.nn.Module,
        high_vol_agreement: float = 0.80,
        trade_threshold: float = 0.55,
        device: str = "cpu"
    ):
        self.ensemble = ensemble
        self.aggressive_model = aggressive_model
        self.high_vol_agreement = high_vol_agreement
        self.trade_threshold = trade_threshold
        self.device = device

    @torch.no_grad()
    def predict_single(
        self,
        prices: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions using single aggressive seed."""
        self.aggressive_model.eval()
        output = self.aggressive_model(prices, features)

        tradeable_logits = output['tradeable_logits']
        direction_logits = output['direction_logits']

        tradeable_prob = F.softmax(tradeable_logits, dim=-1)[:, 1]
        direction_prob = F.softmax(direction_logits, dim=-1)[:, 0]  # P(long)

        should_trade = tradeable_prob >= self.trade_threshold
        is_long = direction_prob > 0.5

        return {
            'should_trade': should_trade,
            'is_long': is_long,
            'tradeable_prob': tradeable_prob,
            'long_prob': direction_prob,
            'agreement': torch.ones_like(tradeable_prob),  # N/A for single seed
            'confidence': tradeable_prob,
            'regime': 'low_vol',
        }

    @torch.no_grad()
    def predict_ensemble(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        volatility: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions using defensive ensemble."""
        result = self.ensemble.predict(prices, features, volatility=volatility)

        # Additional filter: require high agreement during high volatility
        if self.high_vol_agreement > 0:
            agreement_mask = result['direction_agreement'] >= self.high_vol_agreement
            result['should_trade'] = result['should_trade'] & agreement_mask

        result['regime'] = 'high_vol'
        return result

    def predict_adaptive(
        self,
        prices: torch.Tensor,
        features: torch.Tensor,
        regime_mask: torch.Tensor,  # True = high_vol, False = low_vol
        volatility: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions adaptively based on regime.

        Args:
            prices: [batch, window_size, n_price_features]
            features: [batch, window_size, n_engineered_features]
            regime_mask: [batch] boolean tensor (True = high_vol)
            volatility: [batch] optional volatility values

        Returns:
            Combined predictions with regime info
        """
        batch_size = prices.shape[0]
        device = prices.device

        # Get predictions from both strategies
        ensemble_result = self.predict_ensemble(prices, features, volatility)
        single_result = self.predict_single(prices, features)

        # Combine based on regime
        result = {}
        for key in ['should_trade', 'is_long', 'tradeable_prob', 'long_prob', 'agreement', 'confidence']:
            if key in ensemble_result and key in single_result:
                result[key] = torch.where(
                    regime_mask,
                    ensemble_result[key],
                    single_result[key]
                )

        # Track regime per sample
        result['is_high_vol'] = regime_mask

        return result


def regime_adaptive_predict_dataframe(
    adaptive_predictor: RegimeAdaptivePredictor,
    data: pd.DataFrame,
    price_columns: List[str],
    feature_columns: List[str],
    regime_series: pd.Series,
    window_size: int = 64,
    batch_size: int = 256
) -> pd.DataFrame:
    """
    Generate regime-adaptive predictions for a DataFrame.

    Args:
        adaptive_predictor: RegimeAdaptivePredictor instance
        data: DataFrame with price and feature columns
        price_columns: List of price column names
        feature_columns: List of feature column names
        regime_series: Series with 'high_vol' or 'low_vol' per row
        window_size: Lookback window size
        batch_size: Batch size for prediction

    Returns:
        DataFrame with predictions and regime info
    """
    device = adaptive_predictor.device
    predictions = []

    # Prepare arrays
    prices_arr = data[price_columns].values.astype(np.float32)
    features_arr = data[feature_columns].values.astype(np.float32)

    # Handle NaN
    features_arr = np.nan_to_num(features_arr, nan=0.0)

    timestamps = data['timestamp'].values
    n_samples = len(prices_arr) - window_size + 1

    if n_samples <= 0:
        return pd.DataFrame()

    # Align regime series with predictions (offset by window_size - 1)
    regime_values = regime_series.values[window_size - 1:]

    logger.info(f"Generating adaptive predictions for {n_samples} samples...")

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)

        # Create batch
        batch_prices = []
        batch_features = []
        batch_regime = []

        for i in range(start_idx, end_idx):
            batch_prices.append(prices_arr[i:i + window_size])
            batch_features.append(features_arr[i:i + window_size])
            batch_regime.append(regime_values[i] == 'high_vol')

        # Convert to tensors
        prices_tensor = torch.tensor(np.array(batch_prices), dtype=torch.float32, device=device)
        features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32, device=device)
        regime_mask = torch.tensor(batch_regime, dtype=torch.bool, device=device)

        # Compute volatility from features
        volatility = features_tensor[:, -1, :].std(dim=-1)

        # Get adaptive predictions
        result = adaptive_predictor.predict_adaptive(
            prices_tensor, features_tensor, regime_mask, volatility
        )

        # Store results
        for j in range(end_idx - start_idx):
            idx = start_idx + j
            pred = {
                'timestamp': timestamps[idx + window_size - 1],
                'should_trade': result['should_trade'][j].item(),
                'is_long': result['is_long'][j].item(),
                'tradeable_prob': result['tradeable_prob'][j].item(),
                'long_prob': result['long_prob'][j].item(),
                'confidence': result['confidence'][j].item(),
                'agreement': result['agreement'][j].item(),
                'is_high_vol': result['is_high_vol'][j].item(),
                'regime': 'high_vol' if result['is_high_vol'][j].item() else 'low_vol',
            }
            predictions.append(pred)

    logger.info(f"Generated {len(predictions)} predictions")
    return pd.DataFrame(predictions)


def calculate_adaptive_trading_returns(
    predictions: pd.DataFrame,
    price_data: pd.DataFrame,
    stop_loss_pct: float = -0.02
) -> Dict[str, float]:
    """
    Calculate trading returns from adaptive predictions.

    Returns metrics split by regime.
    """
    pred_df = predictions.copy()

    # Get required columns from price_data
    available_cols = [c for c in price_data.columns if c in
                      ['timestamp', 'open', 'close', 'next_return',
                       'next_mae_long', 'next_mae_short']]
    price_df = price_data[available_cols].copy()

    # Ensure timestamps match
    if pred_df['timestamp'].dt.tz is None:
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.tz_localize('UTC')
    if price_df['timestamp'].dt.tz is None:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp']).dt.tz_localize('UTC')

    # Filter price_df to prediction timestamps
    pred_timestamps = set(pred_df['timestamp'])
    price_df_filtered = price_df[price_df['timestamp'].isin(pred_timestamps)].copy()

    # Merge
    merged = pred_df.merge(price_df_filtered, on='timestamp', how='left')

    # Use next_return (open-to-close)
    if 'next_return' in merged.columns:
        merged['fwd_return'] = merged['next_return']
    else:
        logger.warning("next_return not found, using close-to-close")
        if 'close' in merged.columns:
            merged['fwd_return'] = merged['close'].pct_change().shift(-1)
        else:
            return {'total_return': 0, 'n_trades': 0}

    # Calculate position
    merged['position'] = 0.0
    trade_mask = merged['should_trade']
    merged.loc[trade_mask & merged['is_long'], 'position'] = 1.0
    merged.loc[trade_mask & ~merged['is_long'], 'position'] = -1.0

    # Trade returns
    merged['trade_return'] = merged['position'] * merged['fwd_return']

    # Apply MAE-aware stop-loss
    n_stopped = 0
    if stop_loss_pct is not None and 'next_mae_long' in merged.columns:
        long_mask = merged['position'] == 1.0
        short_mask = merged['position'] == -1.0

        long_stopped = long_mask & (merged['next_mae_long'] > abs(stop_loss_pct))
        short_stopped = short_mask & (merged['next_mae_short'] > abs(stop_loss_pct))

        stopped_mask = long_stopped | short_stopped
        n_stopped = int(stopped_mask.sum())
        merged.loc[stopped_mask, 'trade_return'] = stop_loss_pct

    # Calculate metrics
    valid = merged.dropna(subset=['trade_return'])
    trade_returns = valid[valid['position'] != 0]['trade_return']

    total_return = trade_returns.sum() * 100
    n_trades = int((valid['position'] != 0).sum())

    # Split by regime
    high_vol_trades = valid[(valid['position'] != 0) & (valid['regime'] == 'high_vol')]
    low_vol_trades = valid[(valid['position'] != 0) & (valid['regime'] == 'low_vol')]

    high_vol_return = high_vol_trades['trade_return'].sum() * 100 if len(high_vol_trades) > 0 else 0
    low_vol_return = low_vol_trades['trade_return'].sum() * 100 if len(low_vol_trades) > 0 else 0

    # Win rates
    if len(trade_returns) > 0:
        win_rate = (trade_returns > 0).mean() * 100
    else:
        win_rate = 0

    # Sharpe
    if len(trade_returns) > 1 and trade_returns.std() > 0:
        days = (valid['timestamp'].max() - valid['timestamp'].min()).days
        if days > 0:
            trades_per_year = n_trades / days * 365
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Regime breakdown
    n_high_vol = int((valid['regime'] == 'high_vol').sum())
    n_low_vol = int((valid['regime'] == 'low_vol').sum())
    pct_high_vol = n_high_vol / len(valid) * 100 if len(valid) > 0 else 0

    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_stopped_out': n_stopped,
        # High volatility regime
        'high_vol_trades': len(high_vol_trades),
        'high_vol_return': high_vol_return,
        # Low volatility regime
        'low_vol_trades': len(low_vol_trades),
        'low_vol_return': low_vol_return,
        # Regime breakdown
        'n_high_vol_candles': n_high_vol,
        'n_low_vol_candles': n_low_vol,
        'pct_high_vol': pct_high_vol,
    }


def main():
    parser = argparse.ArgumentParser(description='Regime-Adaptive Ensemble Validation')
    parser.add_argument('--vol-threshold', type=float, default=0.70,
                        help='Volatility percentile threshold for high/low regime (default: 0.70)')
    parser.add_argument('--vol-lookback', type=int, default=168,
                        help='Rolling window for volatility calculation (default: 168 = 7 days)')
    parser.add_argument('--aggressive-seed', type=int, default=1337,
                        help='Seed to use in low volatility regime (default: 1337)')
    parser.add_argument('--high-vol-agreement', type=float, default=0.80,
                        help='Min agreement for high volatility trades (default: 0.80)')
    parser.add_argument('--stop-loss', type=float, default=-0.02,
                        help='Stop-loss percentage (default: -0.02)')
    parser.add_argument('--period', type=str, default=None,
                        help='Specific period to evaluate (default: all)')
    parser.add_argument('--weight-by', type=str, default='period_1_may2021',
                        help='Period to use for ensemble weighting')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for weighted ensemble softmax')
    args = parser.parse_args()

    print("=" * 100)
    print("REGIME-ADAPTIVE ENSEMBLE VALIDATION")
    print("=" * 100)
    print(f"Strategy: Two-regime adaptive switching")
    print(f"  HIGH volatility (>{args.vol_threshold:.0%}): Defensive ensemble (all 5 seeds, {args.high_vol_agreement:.0%} agreement)")
    print(f"  LOW volatility  (<{args.vol_threshold:.0%}): Aggressive single seed ({args.aggressive_seed})")
    print(f"Volatility lookback: {args.vol_lookback} candles ({args.vol_lookback // 24:.1f} days)")
    print(f"Stop-loss: {args.stop_loss * 100:.1f}%")
    print()

    # Check results directory
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return 1

    # Load raw data
    print("Loading raw data...")
    try:
        full_df = load_raw_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize feature pipeline and labeler
    feature_pipeline = FeaturePipeline(window_size=WINDOW_SIZE)
    labeler = CandleLabeler(LABELING_CONFIG)

    # Filter periods if requested
    if args.period:
        periods = [p for p in PERIODS if p[0] == args.period]
        if not periods:
            print(f"ERROR: Unknown period '{args.period}'")
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
            # Determine weight period
            weight_period = get_weight_period(period_id)

            # Create defensive ensemble (all 5 seeds)
            print(f"  Loading defensive ensemble (all seeds)...")
            ensemble = create_ensemble_from_walk_forward(
                RESULTS_DIR,
                method=EnsembleMethod.WEIGHTED if weight_period else EnsembleMethod.MEAN,
                period=period_id,
                weight_by_period=weight_period,
                seeds=SEEDS,
                device=device,
                temperature=args.temperature
            )

            # Load aggressive single seed model
            print(f"  Loading aggressive model (seed {args.aggressive_seed})...")
            aggressive_model = load_single_seed_model(
                RESULTS_DIR,
                args.aggressive_seed,
                period_id,
                device
            )

            # Create adaptive predictor
            adaptive = RegimeAdaptivePredictor(
                ensemble=ensemble,
                aggressive_model=aggressive_model,
                high_vol_agreement=args.high_vol_agreement,
                device=device
            )

            # Prepare period data
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

            # Calculate regime
            print(f"  Calculating volatility regime...")
            regime_series = get_regime(
                test_data,
                lookback=args.vol_lookback,
                threshold=args.vol_threshold
            )

            # Regime breakdown
            n_high = (regime_series == 'high_vol').sum()
            n_low = (regime_series == 'low_vol').sum()
            pct_high = n_high / len(regime_series) * 100
            print(f"  Regime breakdown: {pct_high:.0f}% high_vol, {100-pct_high:.0f}% low_vol")

            # Generate adaptive predictions
            print("  Generating adaptive predictions...")
            predictions = regime_adaptive_predict_dataframe(
                adaptive,
                test_data,
                price_columns,
                feature_columns,
                regime_series,
                window_size=WINDOW_SIZE,
                batch_size=256
            )

            if len(predictions) == 0:
                print("  SKIP: No predictions generated")
                continue

            # Calculate returns
            adaptive_metrics = calculate_adaptive_trading_returns(
                predictions,
                test_data,
                stop_loss_pct=args.stop_loss
            )

            # Also calculate pure ensemble for comparison
            print("  Calculating pure ensemble baseline...")
            from sph_net.ensemble import ensemble_predict_dataframe, calculate_ensemble_trading_returns

            pure_ensemble_preds = ensemble_predict_dataframe(
                ensemble,
                test_data,
                price_columns,
                feature_columns,
                window_size=WINDOW_SIZE,
                batch_size=256,
                return_individual=False
            )

            pure_ensemble_metrics = calculate_ensemble_trading_returns(
                pure_ensemble_preds,
                test_data,
                stop_loss_pct=args.stop_loss,
                use_regime_filter=True
            )

            improvement = adaptive_metrics['total_return'] - pure_ensemble_metrics['total_return']

            # Print results
            print(f"\nResults:")
            print(f"  Regime breakdown: {adaptive_metrics['pct_high_vol']:.0f}% high_vol, "
                  f"{100-adaptive_metrics['pct_high_vol']:.0f}% low_vol")
            print(f"  High vol trades: {adaptive_metrics['high_vol_trades']} (ensemble) → "
                  f"{adaptive_metrics['high_vol_return']:+.2f}%")
            print(f"  Low vol trades:  {adaptive_metrics['low_vol_trades']} (seed {args.aggressive_seed}) → "
                  f"{adaptive_metrics['low_vol_return']:+.2f}%")
            print()
            print(f"  Adaptive return:    {adaptive_metrics['total_return']:>+10.2f}%")
            print(f"  Pure ensemble:      {pure_ensemble_metrics['total_return']:>+10.2f}%")
            print(f"  Improvement:        {improvement:>+10.2f}%")
            print(f"  Trades: {adaptive_metrics['n_trades']}, "
                  f"Sharpe: {adaptive_metrics['sharpe']:.2f}, "
                  f"Win rate: {adaptive_metrics['win_rate']:.1f}%")

            if improvement > 0:
                print(f"  ✓ Adaptive strategy BEATS pure ensemble")
            elif improvement > -1.0:
                print(f"  ~ Adaptive strategy MATCHES pure ensemble")
            else:
                print(f"  ✗ Adaptive strategy UNDERPERFORMS")

            all_results.append({
                'period_id': period_id,
                'period_name': period_name,
                'is_primary': is_primary,
                'start_date': start_date,
                'end_date': end_date,
                'adaptive_return': adaptive_metrics['total_return'],
                'pure_ensemble_return': pure_ensemble_metrics['total_return'],
                'improvement': improvement,
                'n_trades': adaptive_metrics['n_trades'],
                'sharpe': adaptive_metrics['sharpe'],
                'win_rate': adaptive_metrics['win_rate'],
                'pct_high_vol': adaptive_metrics['pct_high_vol'],
                'high_vol_trades': adaptive_metrics['high_vol_trades'],
                'high_vol_return': adaptive_metrics['high_vol_return'],
                'low_vol_trades': adaptive_metrics['low_vol_trades'],
                'low_vol_return': adaptive_metrics['low_vol_return'],
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

        print("\nAll periods:")
        print(f"  Adaptive average:     {results_df['adaptive_return'].mean():+.2f}%")
        print(f"  Pure ensemble avg:    {results_df['pure_ensemble_return'].mean():+.2f}%")
        print(f"  Average improvement:  {results_df['improvement'].mean():+.2f}%")

        # Primary periods
        primary = results_df[results_df['is_primary']]
        if len(primary) > 0:
            print("\nPrimary periods only:")
            print(f"  Adaptive average:     {primary['adaptive_return'].mean():+.2f}%")
            print(f"  Pure ensemble avg:    {primary['pure_ensemble_return'].mean():+.2f}%")
            print(f"  Average improvement:  {primary['improvement'].mean():+.2f}%")

        # Per-period summary
        print("\nPer-period results:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            status = "✓" if row['improvement'] > 0 else ("~" if row['improvement'] > -1 else "✗")
            print(f"  {status} {row['period_name']:25s} "
                  f"Adaptive: {row['adaptive_return']:+7.2f}%  "
                  f"Ensemble: {row['pure_ensemble_return']:+7.2f}%  "
                  f"Δ: {row['improvement']:+6.2f}%")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / 'regime_adaptive_validation.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        json_path = RESULTS_DIR / 'regime_adaptive_validation.json'
        results_df.to_json(json_path, orient='records', indent=2)
        print(f"JSON saved to: {json_path}")
    else:
        print("\nNo results generated. Check errors above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
