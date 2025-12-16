#!/usr/bin/env python
"""
Walk-Forward Validation Script

CRITICAL: This is the make-or-break test for model generalization.
NO HACKS, NO WORKAROUNDS, NO CHERRY-PICKING.

Usage:
    python scripts/walk_forward_validation.py

Output:
    experiments/walk_forward/
    ├── period_X_name/
    │   ├── model.pt
    │   ├── train_metrics.json
    │   ├── test_results.json
    │   └── predictions.csv
    └── summary_report.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
import torch
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from labeling.candle_classifier import CandleLabeler, LabelingConfig
from features.feature_pipeline import FeaturePipeline
from data.dataset import TradingDataset
from torch.utils.data import DataLoader
from sph_net.config import SPHNetConfig
from sph_net.models.two_stage import TwoStageModel, TwoStageLoss, CalibratedTwoStageModel, apply_stop_loss_to_returns
from training.trainer import Trainer
from training.metrics import MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - FROZEN (DO NOT MODIFY BETWEEN PERIODS)
# =============================================================================

DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")
OUTPUT_DIR = Path("experiments/walk_forward")

# Labeling configuration - FROZEN
LABELING_CONFIG = LabelingConfig(
    strong_move_threshold=0.010,    # 1.0%
    weak_move_threshold=0.004,      # 0.4%
    clean_path_mae_threshold=0.010  # 1.0%
)

# Model configuration - FROZEN
def get_model_config(n_price_features: int, n_engineered_features: int, tradeable_pos_weight: float) -> SPHNetConfig:
    """Create frozen model configuration."""
    return SPHNetConfig(
        n_price_features=n_price_features,
        n_engineered_features=n_engineered_features,
        n_classes=5,
        window_size=64,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        dropout=0.2,
        batch_size=32,
        learning_rate=5e-5,
        epochs=100,
        patience=25,
        focal_gamma=2.0,
        model_type='two_stage',
        tradeable_pos_weight=tradeable_pos_weight,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

# Inference configuration - FROZEN
INFERENCE_CONFIG = {
    'trade_threshold': 0.55,
    'filter_high_volatility': True,
    'stop_loss_pct': -0.02,  # -2.0% stop-loss (critical for limiting tail risk)
}

# Test periods
@dataclass
class TestPeriod:
    """Definition of a walk-forward test period."""
    period_id: str
    name: str
    train_end: str
    test_start: str
    test_end: str
    is_primary: bool
    description: str

TEST_PERIODS = [
    TestPeriod(
        period_id='period_0_covid',
        name='COVID Crash',
        train_end='2020-02-29 23:59:59',
        test_start='2020-03-01 00:00:00',
        test_end='2020-05-31 23:59:59',
        is_primary=False,
        description='COVID-19 market crash - LIMITED TRAINING DATA'
    ),
    TestPeriod(
        period_id='period_1_may2021',
        name='May 2021 Crash',
        train_end='2021-04-30 23:59:59',
        test_start='2021-05-01 00:00:00',
        test_end='2021-07-31 23:59:59',
        is_primary=True,
        description='China mining ban, Elon tweets, -53% correction'
    ),
    TestPeriod(
        period_id='period_2_luna',
        name='Luna/3AC Collapse',
        train_end='2022-04-30 23:59:59',
        test_start='2022-05-01 00:00:00',
        test_end='2022-07-31 23:59:59',
        is_primary=True,
        description='UST depeg, Luna death spiral, 3AC liquidation'
    ),
    TestPeriod(
        period_id='period_3_ftx',
        name='FTX Crash',
        train_end='2022-10-31 23:59:59',
        test_start='2022-11-01 00:00:00',
        test_end='2023-01-31 23:59:59',
        is_primary=True,
        description='FTX insolvency, exchange collapse'
    ),
    TestPeriod(
        period_id='period_4_etf',
        name='ETF Rally',
        train_end='2023-12-31 23:59:59',
        test_start='2024-01-01 00:00:00',
        test_end='2024-03-31 23:59:59',
        is_primary=True,
        description='Bitcoin ETF approval, institutional inflow'
    ),
    TestPeriod(
        period_id='period_5_full',
        name='Full Data Holdout',
        train_end='2024-09-30 23:59:59',
        test_start='2024-10-01 00:00:00',
        test_end='2025-12-15 23:59:59',
        is_primary=True,
        description='Full data holdout test (~15% most recent data, aligns with prepare_data.py split)'
    ),
]


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_full_dataset() -> pd.DataFrame:
    """Load and validate full dataset."""
    logger.info(f"Loading data from {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            "Ensure data_pipleine/ml_data/BTCUSDT_ml_data.parquet exists."
        )

    df = pd.read_parquet(DATA_PATH)

    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        raise ValueError("Dataset must have 'timestamp' column")

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Dataset loaded: {len(df):,} rows")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Drop rows with NaN in critical columns (expected at dataset boundaries)
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    nan_count = df[critical_cols].isna().any(axis=1).sum()
    df = df.dropna(subset=critical_cols)

    logger.info(f"Dropped {nan_count} rows with NaN in OHLCV")
    logger.info(f"Clean rows: {len(df):,}")

    # Validate date range
    data_start = df['timestamp'].min()
    data_end = df['timestamp'].max()

    # Check if we have enough historical data
    required_start = pd.Timestamp('2019-09-10', tz='UTC')  # Allow few days buffer
    if data_start > required_start:
        logger.error(f"INSUFFICIENT DATA: Dataset starts at {data_start.date()}")
        logger.error(f"Walk-forward validation requires data from ~{required_start.date()}")
        raise ValueError("Insufficient historical data for walk-forward validation")

    logger.info(f"Data validated: {len(df):,} clean candles from {data_start.date()} to {data_end.date()}")

    return df


def prepare_period_data(
    full_df: pd.DataFrame,
    period: TestPeriod,
    feature_pipeline: FeaturePipeline,
    labeler: CandleLabeler
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prepare data for a single period.

    CRITICAL: Normalization uses TRAINING data statistics only.

    Returns:
        (train_df, test_df, normalization_stats)
    """
    train_end = pd.Timestamp(period.train_end, tz='UTC')
    test_start = pd.Timestamp(period.test_start, tz='UTC')
    test_end = pd.Timestamp(period.test_end, tz='UTC')

    # Split chronologically
    train_raw = full_df[full_df['timestamp'] <= train_end].copy()
    test_raw = full_df[
        (full_df['timestamp'] >= test_start) &
        (full_df['timestamp'] <= test_end)
    ].copy()

    logger.info(f"  Train: {len(train_raw):,} candles ({train_raw['timestamp'].min()} to {train_raw['timestamp'].max()})")
    logger.info(f"  Test:  {len(test_raw):,} candles ({test_raw['timestamp'].min()} to {test_raw['timestamp'].max()})")

    if len(train_raw) < 1000:
        logger.warning(f"  LIMITED TRAINING DATA: Only {len(train_raw)} candles")

    if len(test_raw) == 0:
        raise ValueError(f"No test data for period {period.name}")

    # Apply labeling
    logger.info("  Labeling data...")
    train_labeled = labeler.label_dataset(train_raw)
    test_labeled = labeler.label_dataset(test_raw)

    # Apply feature engineering
    logger.info("  Computing features...")
    train_featured = feature_pipeline.compute_all_features(train_labeled)
    test_featured = feature_pipeline.compute_all_features(test_labeled)

    # Drop warmup period
    warmup = feature_pipeline.get_warmup_periods()
    train_clean = train_featured.iloc[warmup:].copy()
    test_clean = test_featured.iloc[warmup:].copy()

    # Drop rows with NaN labels
    train_clean = train_clean[train_clean['label'].notna()]
    test_clean = test_clean[test_clean['label'].notna()]

    logger.info(f"  After warmup/cleanup: train={len(train_clean):,}, test={len(test_clean):,}")

    # CRITICAL: Normalize using TRAINING data statistics
    logger.info("  Normalizing features (using training stats only)...")
    train_normalized, norm_stats = feature_pipeline.normalize_features(
        train_clean,
        fit_data=train_clean  # Fit on training data
    )
    test_normalized = feature_pipeline.apply_normalization(
        test_clean,
        norm_stats  # Apply training stats to test
    )

    return train_normalized, test_normalized, norm_stats


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    price_cols: list,
    feature_cols: list,
    config: SPHNetConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test."""

    train_dataset = TradingDataset(
        train_df, price_cols, feature_cols,
        window_size=config.window_size
    )
    val_dataset = TradingDataset(
        val_df, price_cols, feature_cols,
        window_size=config.window_size
    )
    test_dataset = TradingDataset(
        test_df, price_cols, feature_cols,
        window_size=config.window_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_period_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SPHNetConfig,
    output_dir: Path
) -> TwoStageModel:
    """
    Train a fresh model for a single period.

    CRITICAL: New model from scratch (no transfer learning).
    CRITICAL: Returns BEST model from early stopping, not final model.
    """
    # Create fresh model
    model = TwoStageModel(config)

    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir)
    )

    final_metrics = trainer.train()

    # CRITICAL: Load the BEST model (saved during early stopping), not the final model
    # This matches what run_calibrated.py does
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        logger.info(f"  Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(f"  Best model not found at {best_model_path}, using final model")

    return model


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_period(
    model: TwoStageModel,
    test_loader: DataLoader,
    config: SPHNetConfig,
    trade_threshold: float = 0.55,
    filter_high_vol: bool = True
) -> Dict:
    """
    Evaluate model on test period.

    Returns comprehensive metrics without any modification.
    """
    device = torch.device(config.device)

    # Wrap with calibration
    calibrated_model = CalibratedTwoStageModel(
        model,
        trade_threshold=trade_threshold,
        filter_high_volatility=filter_high_vol,
        use_position_sizing=False
    )
    calibrated_model.to(device)
    calibrated_model.eval()

    all_results = []

    for batch in test_loader:
        prices = batch['prices'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        next_return = batch['next_return'].to(device)
        next_mae_long = batch['next_mae_long'].to(device)
        next_mae_short = batch['next_mae_short'].to(device)

        # Get volatility proxy
        volatility = features[:, -1, :].std(dim=-1)

        # Get predictions
        results = calibrated_model.predict_with_sizing(prices, features, volatility)

        for i in range(len(results['should_trade'])):
            trade_return = 0.0
            trade_mae = 0.0
            is_long = results['is_long'][i].item()

            if results['should_trade'][i]:
                if is_long:
                    trade_return = next_return[i].item()
                    trade_mae = next_mae_long[i].item()
                else:
                    trade_return = -next_return[i].item()
                    trade_mae = next_mae_short[i].item()

            all_results.append({
                'true_label': labels[i].item(),
                'should_trade': results['should_trade'][i].item(),
                'is_long': is_long,
                'trade_prob': results['trade_prob'][i].item(),
                'direction_confidence': results['direction_confidence'][i].item(),
                'next_return': next_return[i].item(),
                'trade_return': trade_return,
                'trade_mae': trade_mae,
            })

    results_df = pd.DataFrame(all_results)

    # Compute metrics with stop-loss
    metrics = compute_trading_metrics(results_df, stop_loss_pct=INFERENCE_CONFIG['stop_loss_pct'])

    return metrics, results_df


def compute_trading_metrics(results_df: pd.DataFrame, stop_loss_pct: float = None) -> Dict:
    """Compute comprehensive trading metrics with optional stop-loss."""
    trades = results_df[results_df['should_trade']].copy()
    n_total = len(results_df)
    n_trades = len(trades)

    metrics = {
        'total_samples': n_total,
        'n_trades': n_trades,
        'trade_frequency': n_trades / n_total * 100 if n_total > 0 else 0,
        'stop_loss_pct': stop_loss_pct,
    }

    if n_trades == 0:
        metrics.update({
            'total_return': 0.0,
            'total_return_no_sl': 0.0,
            'win_rate': 0.0,
            'avg_return_per_trade': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'worst_trade': 0.0,
            'best_trade': 0.0,
            'long_trades': 0,
            'long_total_return': 0.0,
            'long_win_rate': 0.0,
            'short_trades': 0,
            'short_total_return': 0.0,
            'short_win_rate': 0.0,
            'n_stopped_out': 0,
        })
        return metrics

    # Store original returns before stop-loss
    original_returns = trades['trade_return'].copy()
    metrics['total_return_no_sl'] = float(original_returns.sum() * 100)

    # Apply stop-loss if configured
    if stop_loss_pct is not None:
        mae_values = trades['trade_mae'].values if 'trade_mae' in trades.columns else None
        sl_results = apply_stop_loss_to_returns(
            trades['trade_return'].values,
            stop_loss_pct=stop_loss_pct,
            mae_values=mae_values,
        )
        trades['trade_return'] = sl_results['adjusted_returns']
        metrics['n_stopped_out'] = int(sl_results['n_stopped'])
        metrics['pct_stopped'] = float(sl_results['pct_stopped'])
    else:
        metrics['n_stopped_out'] = 0
        metrics['pct_stopped'] = 0.0

    returns = trades['trade_return']

    # Overall metrics
    metrics['total_return'] = float(returns.sum() * 100)
    metrics['win_rate'] = float((returns > 0).mean() * 100)
    metrics['avg_return_per_trade'] = float(returns.mean() * 100)

    # Sharpe ratio (trade-frequency adjusted)
    if returns.std() > 0:
        # Estimate trades per year based on test period
        # ~2200 candles = 3 months, so ~4 periods per year
        trades_per_year = n_trades * 4
        metrics['sharpe_ratio'] = float(
            (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
        )
    else:
        metrics['sharpe_ratio'] = 0.0

    # Drawdown
    cumulative = returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    metrics['max_drawdown'] = float(drawdown.min() * 100)

    # Best/worst
    metrics['worst_trade'] = float(returns.min() * 100)
    metrics['best_trade'] = float(returns.max() * 100)

    # Long trades
    long_trades = trades[trades['is_long']]
    metrics['long_trades'] = len(long_trades)
    if len(long_trades) > 0:
        metrics['long_total_return'] = float(long_trades['trade_return'].sum() * 100)
        metrics['long_win_rate'] = float((long_trades['trade_return'] > 0).mean() * 100)
    else:
        metrics['long_total_return'] = 0.0
        metrics['long_win_rate'] = 0.0

    # Short trades
    short_trades = trades[~trades['is_long']]
    metrics['short_trades'] = len(short_trades)
    if len(short_trades) > 0:
        metrics['short_total_return'] = float(short_trades['trade_return'].sum() * 100)
        metrics['short_win_rate'] = float((short_trades['trade_return'] > 0).mean() * 100)
    else:
        metrics['short_total_return'] = 0.0
        metrics['short_win_rate'] = 0.0

    return metrics


# =============================================================================
# REPORTING
# =============================================================================

def generate_summary_report(period_results: Dict, output_dir: Path) -> Dict:
    """Generate final summary report."""

    # Aggregate primary periods (1-5)
    primary_ids = ['period_1_may2021', 'period_2_luna', 'period_3_ftx', 'period_4_etf', 'period_5_full']

    primary_returns = []
    primary_returns_no_sl = []
    primary_sharpes = []
    total_trades = 0

    for pid in primary_ids:
        if pid in period_results:
            r = period_results[pid]['metrics']
            primary_returns.append(r['total_return'])
            primary_returns_no_sl.append(r.get('total_return_no_sl', r['total_return']))
            primary_sharpes.append(r['sharpe_ratio'])
            total_trades += r['n_trades']

    n_profitable = sum(1 for r in primary_returns if r > 0)
    avg_return = np.mean(primary_returns) if primary_returns else 0
    avg_return_no_sl = np.mean(primary_returns_no_sl) if primary_returns_no_sl else 0
    avg_sharpe = np.mean(primary_sharpes) if primary_sharpes else 0
    worst_return = min(primary_returns) if primary_returns else 0
    best_return = max(primary_returns) if primary_returns else 0

    # Compute verdict
    verdict = compute_verdict(n_profitable, avg_return, avg_sharpe, worst_return)

    summary = {
        'generated_at': datetime.now().isoformat(),
        'config': {
            'model': {
                'model_type': 'two_stage',
                'd_model': 64,
                'n_heads': 4,
                'trade_threshold': INFERENCE_CONFIG['trade_threshold'],
            },
            'labeling': {
                'strong_move_threshold': LABELING_CONFIG.strong_move_threshold,
                'weak_move_threshold': LABELING_CONFIG.weak_move_threshold,
                'clean_path_mae_threshold': LABELING_CONFIG.clean_path_mae_threshold,
            },
            'risk_management': {
                'stop_loss_pct': INFERENCE_CONFIG['stop_loss_pct'],
                'filter_high_volatility': INFERENCE_CONFIG['filter_high_volatility'],
            }
        },
        'periods': {
            pid: {
                'name': period_results[pid]['name'],
                'is_primary': period_results[pid]['is_primary'],
                'train_candles': period_results[pid]['train_candles'],
                'test_candles': period_results[pid]['test_candles'],
                'metrics': period_results[pid]['metrics']
            }
            for pid in period_results
        },
        'aggregated': {
            'primary_periods_profitable': n_profitable,
            'average_return': float(avg_return),
            'average_return_no_sl': float(avg_return_no_sl),
            'average_sharpe': float(avg_sharpe),
            'worst_period_return': float(worst_return),
            'best_period_return': float(best_return),
            'total_trades_all_periods': total_trades,
        },
        'verdict': verdict
    }

    # Save
    with open(output_dir / 'summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def compute_verdict(n_profitable: int, avg_return: float, avg_sharpe: float, worst_return: float) -> Dict:
    """Compute final verdict based on primary periods (5 periods total)."""

    # Grading logic (updated for 5 primary periods)
    if worst_return < -30:
        grade = 'F'
        reasoning = f'Failed: Catastrophic loss in one period ({worst_return:.1f}%)'
    elif n_profitable >= 5 and avg_sharpe > 1.5:
        grade = 'A'
        reasoning = 'Production Ready: All 5 periods profitable with excellent risk-adjusted returns'
    elif n_profitable >= 4 and avg_sharpe > 1.0:
        grade = 'B'
        reasoning = 'Promising: 4/5 periods profitable with good risk metrics'
    elif n_profitable >= 4 and avg_sharpe > 0.8:
        grade = 'B'
        reasoning = 'Promising: 4/5 periods profitable, meets minimum Sharpe requirement'
    elif n_profitable >= 3 and avg_sharpe > 0.5:
        grade = 'C'
        reasoning = 'Needs Work: Mixed results, consider regime-specific models'
    elif n_profitable >= 2:
        grade = 'D'
        reasoning = 'Significant Issues: Only 2/5 periods profitable'
    elif n_profitable >= 1:
        grade = 'D'
        reasoning = 'Significant Issues: Only 1/5 period profitable'
    else:
        grade = 'F'
        reasoning = 'Failed: No profitable periods'

    passed = grade in ['A', 'B']

    return {
        'grade': grade,
        'passed': passed,
        'reasoning': reasoning,
        'recommendation': 'Proceed to paper trading' if passed else 'Model needs improvement'
    }


def print_summary(summary: Dict):
    """Print formatted summary to console."""

    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION SUMMARY")
    sl_pct = INFERENCE_CONFIG['stop_loss_pct']
    if sl_pct:
        print(f"(With {sl_pct*100:.1f}% Stop-Loss Applied)")
    print("=" * 100)

    print("\nPER-PERIOD RESULTS:")
    print("-" * 100)
    print(f"{'Period':<20} {'Return':>10} {'NoSL':>10} {'Sharpe':>8} {'Trades':>7} {'Stop':>6} {'Win%':>7} {'Status':>10}")
    print("-" * 100)

    for pid, data in summary['periods'].items():
        m = data['metrics']
        status = "PRIMARY" if data['is_primary'] else "BONUS"
        profitable = "PASS" if m['total_return'] > 0 else "FAIL"
        no_sl_return = m.get('total_return_no_sl', m['total_return'])
        n_stopped = m.get('n_stopped_out', 0)
        print(f"{data['name']:<20} {m['total_return']:>+9.2f}% {no_sl_return:>+9.2f}% {m['sharpe_ratio']:>8.2f} "
              f"{m['n_trades']:>7} {n_stopped:>6} {m['win_rate']:>6.1f}% {status:>8} {profitable}")

    print("\n" + "=" * 100)
    print("AGGREGATED (PRIMARY PERIODS ONLY):")
    print("-" * 100)
    agg = summary['aggregated']
    print(f"Profitable Periods:    {agg['primary_periods_profitable']}/5")
    print(f"Average Return:        {agg['average_return']:+.2f}% (with stop-loss)")
    print(f"Avg Return (no SL):    {agg.get('average_return_no_sl', agg['average_return']):+.2f}%")
    print(f"Average Sharpe:        {agg['average_sharpe']:.2f}")
    print(f"Worst Period:          {agg['worst_period_return']:+.2f}%")
    print(f"Best Period:           {agg['best_period_return']:+.2f}%")
    print(f"Total Trades:          {agg['total_trades_all_periods']}")

    print("\n" + "=" * 100)
    print("VERDICT:")
    print("-" * 100)
    v = summary['verdict']
    print(f"Grade:          {v['grade']}")
    print(f"Passed:         {'YES' if v['passed'] else 'NO'}")
    print(f"Reasoning:      {v['reasoning']}")
    print(f"Recommendation: {v['recommendation']}")
    print("=" * 100 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_walk_forward_validation():
    """Execute full walk-forward validation."""

    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Started: {start_time}")
    print(f"Output:  {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    full_df = load_full_dataset()

    # Initialize components
    feature_pipeline = FeaturePipeline()
    labeler = CandleLabeler(LABELING_CONFIG)
    price_cols, feature_cols = feature_pipeline.get_feature_columns()

    # Store results
    period_results = {}

    # Process each period
    for period in TEST_PERIODS:
        print("\n" + "=" * 70)
        print(f"PERIOD: {period.name}")
        print(f"Description: {period.description}")
        print("=" * 70)

        period_dir = OUTPUT_DIR / period.period_id
        period_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare data
            logger.info("Preparing data...")
            train_df, test_df, norm_stats = prepare_period_data(
                full_df, period, feature_pipeline, labeler
            )

            # Split train/val (85/15 temporal split)
            val_split_idx = int(len(train_df) * 0.85)
            train_split = train_df.iloc[:val_split_idx]
            val_split = train_df.iloc[val_split_idx:]

            logger.info(f"  Train/Val split: {len(train_split):,} / {len(val_split):,}")

            # Compute class weights for this period's training data
            label_counts = train_split['label'].value_counts()
            tradeable_count = label_counts.get(0, 0) + label_counts.get(4, 0)
            total = len(train_split)
            tradeable_ratio = tradeable_count / total if total > 0 else 0.1
            tradeable_pos_weight = (1 - tradeable_ratio) / max(tradeable_ratio, 0.01)

            logger.info(f"  Tradeable ratio: {tradeable_ratio:.2%}, pos_weight: {tradeable_pos_weight:.2f}")

            # Create config for this period
            available_feature_cols = [c for c in feature_cols if c in train_split.columns]
            config = get_model_config(
                n_price_features=len(price_cols),
                n_engineered_features=len(available_feature_cols),
                tradeable_pos_weight=tradeable_pos_weight
            )

            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_split, val_split, test_df,
                price_cols, available_feature_cols, config
            )

            logger.info(f"  Dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

            # Train model
            logger.info("Training model...")
            model = train_period_model(train_loader, val_loader, config, period_dir)

            # Evaluate
            logger.info("Evaluating on test period...")
            metrics, predictions_df = evaluate_period(
                model, test_loader, config,
                trade_threshold=INFERENCE_CONFIG['trade_threshold'],
                filter_high_vol=INFERENCE_CONFIG['filter_high_volatility']
            )

            # Store results
            period_results[period.period_id] = {
                'name': period.name,
                'is_primary': period.is_primary,
                'train_candles': len(train_df),
                'test_candles': len(test_df),
                'metrics': metrics
            }

            # Save predictions
            predictions_df.to_csv(period_dir / 'predictions.csv', index=False)

            # Save test results
            with open(period_dir / 'test_results.json', 'w') as f:
                json.dump({
                    'period_id': period.period_id,
                    'period_name': period.name,
                    'is_primary': period.is_primary,
                    'train_candles': len(train_df),
                    'test_candles': len(test_df),
                    'train_range': f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
                    'test_range': f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}",
                    'metrics': metrics
                }, f, indent=2)

            # Print period summary
            print(f"\n  Results:")
            print(f"    Total Return:   {metrics['total_return']:+.2f}%")
            print(f"    Trades:         {metrics['n_trades']}")
            print(f"    Win Rate:       {metrics['win_rate']:.1f}%")
            print(f"    Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.error(f"Error processing period {period.name}: {e}")
            import traceback
            traceback.print_exc()

            period_results[period.period_id] = {
                'name': period.name,
                'is_primary': period.is_primary,
                'train_candles': 0,
                'test_candles': 0,
                'metrics': {
                    'total_return': 0.0,
                    'n_trades': 0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'error': str(e)
                }
            }

    # Generate summary
    logger.info("\nGenerating summary report...")
    summary = generate_summary_report(period_results, OUTPUT_DIR)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    # Print summary
    print_summary(summary)

    print(f"Execution time: {duration:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR}")

    return summary


if __name__ == "__main__":
    try:
        summary = run_walk_forward_validation()
        sys.exit(0 if summary['verdict']['passed'] else 1)
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
