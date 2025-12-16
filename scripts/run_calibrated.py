#!/usr/bin/env python
"""
Run Calibrated Model for Inference

Uses the CalibratedTwoStageModel wrapper with optimal settings:
- Threshold: 0.55 (best Sharpe ratio)
- No position sizing (equal sizing works better)
- High volatility filtering enabled
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
import torch
from torch.utils.data import DataLoader

from sph_net.config import SPHNetConfig
from sph_net.models.two_stage import TwoStageModel, CalibratedTwoStageModel
from data.dataset import TradingDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibrated_model(
    model_path: Path,
    trade_threshold: float = 0.55,
    filter_high_volatility: bool = True
) -> CalibratedTwoStageModel:
    """Load trained model and wrap with calibration."""

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # Load base model
    model = TwoStageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with calibration
    calibrated = CalibratedTwoStageModel(
        model,
        trade_threshold=trade_threshold,
        filter_high_volatility=filter_high_volatility,
        use_position_sizing=False  # Equal sizing works better
    )

    return calibrated, config


def run_inference(
    calibrated_model: CalibratedTwoStageModel,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> pd.DataFrame:
    """Run inference with calibrated model."""

    calibrated_model.to(device)
    calibrated_model.eval()

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            prices = batch['prices'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            next_return = batch['next_return'].to(device)
            next_mae_long = batch['next_mae_long'].to(device)
            next_mae_short = batch['next_mae_short'].to(device)

            # Extract volatility from features (last timestep, use a volatility-related feature)
            # Assuming volatility info is in features - adjust index as needed
            volatility = features[:, -1, :].std(dim=-1)  # Simple proxy

            # Get calibrated predictions
            results = calibrated_model.predict_with_sizing(prices, features, volatility)

            # Store results
            for i in range(len(results['should_trade'])):
                trade_return = 0.0
                if results['should_trade'][i]:
                    if results['is_long'][i]:
                        trade_return = next_return[i].item()
                    else:
                        trade_return = -next_return[i].item()

                all_results.append({
                    'batch': batch_idx,
                    'sample': i,
                    'true_label': labels[i].item(),
                    'should_trade': results['should_trade'][i].item(),
                    'is_long': results['is_long'][i].item(),
                    'trade_prob': results['trade_prob'][i].item(),
                    'direction_confidence': results['direction_confidence'][i].item(),
                    'position_size': results['position_size'][i].item(),
                    'regime_filtered': results['regime_filtered'][i].item(),
                    'next_return': next_return[i].item(),
                    'trade_return': trade_return,
                    'next_mae_long': next_mae_long[i].item(),
                    'next_mae_short': next_mae_short[i].item(),
                })

    return pd.DataFrame(all_results)


def compute_performance(results_df: pd.DataFrame) -> dict:
    """Compute performance metrics from results."""

    trades = results_df[results_df['should_trade']]
    filtered = results_df[results_df['regime_filtered']]

    metrics = {
        'total_samples': len(results_df),
        'total_trades': len(trades),
        'trades_filtered_by_regime': len(filtered),
        'trade_frequency': len(trades) / len(results_df) * 100,
    }

    if len(trades) > 0:
        # Long trades
        long_trades = trades[trades['is_long']]
        if len(long_trades) > 0:
            metrics['long_trades'] = len(long_trades)
            metrics['long_total_return'] = long_trades['trade_return'].sum() * 100
            metrics['long_avg_return'] = long_trades['trade_return'].mean() * 100
            metrics['long_win_rate'] = (long_trades['trade_return'] > 0).mean() * 100
            metrics['long_correct_class'] = (long_trades['true_label'] == 0).mean() * 100

        # Short trades
        short_trades = trades[~trades['is_long']]
        if len(short_trades) > 0:
            metrics['short_trades'] = len(short_trades)
            metrics['short_total_return'] = short_trades['trade_return'].sum() * 100
            metrics['short_avg_return'] = short_trades['trade_return'].mean() * 100
            metrics['short_win_rate'] = (short_trades['trade_return'] > 0).mean() * 100
            metrics['short_correct_class'] = (short_trades['true_label'] == 4).mean() * 100

        # Combined
        metrics['total_return'] = trades['trade_return'].sum() * 100
        metrics['avg_return_per_trade'] = trades['trade_return'].mean() * 100
        metrics['overall_win_rate'] = (trades['trade_return'] > 0).mean() * 100

        # Sharpe ratio (annualized based on actual trading frequency)
        # Using sqrt(35000) assumes trading every 15-min candle, which overstates Sharpe
        # Instead, use actual trades per year for proper annualization
        if trades['trade_return'].std() > 0:
            # Estimate trading days from total samples (assuming ~96 candles/day)
            trading_days = len(results_df) / 96
            trades_per_year = len(trades) * (365 / max(trading_days, 1))
            metrics['sharpe_ratio'] = (
                trades['trade_return'].mean() / trades['trade_return'].std()
            ) * np.sqrt(trades_per_year)
            metrics['trades_per_year'] = trades_per_year

            # Also compute the original (overstated) for comparison
            metrics['sharpe_ratio_original'] = (
                trades['trade_return'].mean() / trades['trade_return'].std()
            ) * np.sqrt(35000)
        else:
            metrics['sharpe_ratio'] = 0.0
            metrics['sharpe_ratio_original'] = 0.0

    return metrics


def print_report(metrics: dict):
    """Print formatted performance report."""

    print("\n" + "=" * 70)
    print("CALIBRATED MODEL PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\nSAMPLE SUMMARY")
    print("-" * 40)
    print(f"Total Samples:              {metrics['total_samples']:,}")
    print(f"Total Trades:               {metrics['total_trades']:,}")
    print(f"Filtered by High Vol:       {metrics['trades_filtered_by_regime']:,}")
    print(f"Trade Frequency:            {metrics['trade_frequency']:.2f}%")

    if metrics['total_trades'] > 0:
        print(f"\nLONG TRADES")
        print("-" * 40)
        if 'long_trades' in metrics:
            print(f"Count:                      {metrics['long_trades']:,}")
            print(f"Total Return:               {metrics['long_total_return']:+.2f}%")
            print(f"Avg Return/Trade:           {metrics['long_avg_return']:+.4f}%")
            print(f"Win Rate:                   {metrics['long_win_rate']:.2f}%")
            print(f"Correct Class (HIGH_BULL):  {metrics['long_correct_class']:.2f}%")
        else:
            print("No long trades")

        print(f"\nSHORT TRADES")
        print("-" * 40)
        if 'short_trades' in metrics:
            print(f"Count:                      {metrics['short_trades']:,}")
            print(f"Total Return:               {metrics['short_total_return']:+.2f}%")
            print(f"Avg Return/Trade:           {metrics['short_avg_return']:+.4f}%")
            print(f"Win Rate:                   {metrics['short_win_rate']:.2f}%")
            print(f"Correct Class (LOW_BEAR):   {metrics['short_correct_class']:.2f}%")
        else:
            print("No short trades")

        print(f"\nOVERALL PERFORMANCE")
        print("-" * 40)
        print(f"Total Return:               {metrics['total_return']:+.2f}%")
        print(f"Avg Return/Trade:           {metrics['avg_return_per_trade']:+.4f}%")
        print(f"Overall Win Rate:           {metrics['overall_win_rate']:.2f}%")
        print(f"Trades per Year (est.):     {metrics.get('trades_per_year', 'N/A'):.1f}")
        print(f"Sharpe Ratio (corrected):   {metrics['sharpe_ratio']:.2f}")
        print(f"Sharpe Ratio (original):    {metrics.get('sharpe_ratio_original', 'N/A'):.2f}  ← overstated")

    print("\n" + "=" * 70)


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    MODEL_DIR = Path("experiments/run_001")
    OUTPUT_DIR = MODEL_DIR / "calibrated_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calibration settings (from analysis)
    TRADE_THRESHOLD = 0.55
    FILTER_HIGH_VOL = True

    # Check paths
    if not (MODEL_DIR / "best_model.pt").exists():
        logger.error(f"Model not found at {MODEL_DIR / 'best_model.pt'}")
        return

    if not (DATA_DIR / "test.parquet").exists():
        logger.error(f"Test data not found at {DATA_DIR / 'test.parquet'}")
        return

    # === Load Model ===
    logger.info(f"Loading calibrated model (threshold={TRADE_THRESHOLD})...")
    calibrated_model, config = load_calibrated_model(
        MODEL_DIR / "best_model.pt",
        trade_threshold=TRADE_THRESHOLD,
        filter_high_volatility=FILTER_HIGH_VOL
    )

    # === Load Test Data ===
    logger.info("Loading test data...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)

    price_cols = feature_info['price_columns']
    eng_cols = [c for c in feature_info['engineered_columns'] if c in test_df.columns]

    test_dataset = TradingDataset(
        test_df, price_cols, eng_cols,
        window_size=config.window_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # === Run Inference ===
    logger.info("Running calibrated inference...")
    results_df = run_inference(calibrated_model, test_loader, device=config.device)

    # === Compute Metrics ===
    metrics = compute_performance(results_df)

    # === Print Report ===
    print_report(metrics)

    # === Save Results ===
    results_df.to_csv(OUTPUT_DIR / "calibrated_predictions.csv", index=False)

    with open(OUTPUT_DIR / "calibrated_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")

    # === Show Sample Signals ===
    print("\nSAMPLE TRADE SIGNALS (first 10 trades):")
    print("-" * 60)

    trades = results_df[results_df['should_trade']].head(10)
    for _, row in trades.iterrows():
        direction = "LONG" if row['is_long'] else "SHORT"
        result = "WIN" if row['trade_return'] > 0 else "LOSS"
        print(f"  {direction:5} | Conf: {row['trade_prob']:.3f} | "
              f"Return: {row['trade_return']*100:+.3f}% | {result}")


if __name__ == "__main__":
    main()
