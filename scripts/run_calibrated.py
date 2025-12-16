#!/usr/bin/env python
"""
Run Calibrated Model for Inference

Uses the CalibratedTwoStageModel wrapper with optimal settings:
- Threshold: 0.55 (best Sharpe ratio)
- No position sizing (equal sizing works better)
- High volatility filtering enabled
- Stop-loss: -1.78% (moderate, recommended)

Risk Management:
- Conservative stop-loss (-1.32%): +65.05% return, stops 5% of trades
- Moderate stop-loss (-1.78%): +52.45% return, stops 2.5% of trades (DEFAULT)
- Aggressive stop-loss (-2.27%): +44.78% return, stops 1% of trades
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
from sph_net.models.two_stage import TwoStageModel, CalibratedTwoStageModel, apply_stop_loss_to_returns
from data.dataset import TradingDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibrated_model(
    model_path: Path,
    trade_threshold: float = 0.55,
    filter_high_volatility: bool = True,
    stop_loss_pct: float = -0.0178,  # -1.78% moderate stop-loss (recommended)
    take_profit_pct: float = None,   # None = let winners run
) -> CalibratedTwoStageModel:
    """Load trained model and wrap with calibration and risk management."""

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # Load base model
    model = TwoStageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with calibration and risk management
    calibrated = CalibratedTwoStageModel(
        model,
        trade_threshold=trade_threshold,
        filter_high_volatility=filter_high_volatility,
        use_position_sizing=False,  # Equal sizing works better
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
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
                trade_mae = 0.0  # Maximum adverse excursion for this trade
                is_long = results['is_long'][i].item()

                if results['should_trade'][i]:
                    if is_long:
                        trade_return = next_return[i].item()
                        trade_mae = next_mae_long[i].item()  # MAE for long position
                    else:
                        trade_return = -next_return[i].item()
                        trade_mae = next_mae_short[i].item()  # MAE for short position

                all_results.append({
                    'batch': batch_idx,
                    'sample': i,
                    'true_label': labels[i].item(),
                    'should_trade': results['should_trade'][i].item(),
                    'is_long': is_long,
                    'trade_prob': results['trade_prob'][i].item(),
                    'direction_confidence': results['direction_confidence'][i].item(),
                    'position_size': results['position_size'][i].item(),
                    'regime_filtered': results['regime_filtered'][i].item(),
                    'next_return': next_return[i].item(),
                    'trade_return': trade_return,
                    'trade_mae': trade_mae,  # MAE for stop-loss simulation
                    'next_mae_long': next_mae_long[i].item(),
                    'next_mae_short': next_mae_short[i].item(),
                })

    return pd.DataFrame(all_results)


def compute_performance(
    results_df: pd.DataFrame,
    stop_loss_pct: float = None,
    take_profit_pct: float = None,
) -> dict:
    """Compute performance metrics from results, with optional stop-loss simulation."""

    trades = results_df[results_df['should_trade']].copy()
    filtered = results_df[results_df['regime_filtered']]

    metrics = {
        'total_samples': len(results_df),
        'total_trades': len(trades),
        'trades_filtered_by_regime': len(filtered),
        'trade_frequency': len(trades) / len(results_df) * 100,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
    }

    if len(trades) == 0:
        return metrics

    # Apply stop-loss if configured
    if stop_loss_pct is not None or take_profit_pct is not None:
        # Use MAE-based stop-loss simulation if available
        mae_values = trades['trade_mae'].values if 'trade_mae' in trades.columns else None

        sl_results = apply_stop_loss_to_returns(
            trades['trade_return'].values,
            stop_loss_pct=stop_loss_pct,
            mae_values=mae_values,
            take_profit_pct=take_profit_pct,
        )

        # Store both original and adjusted returns
        trades['original_return'] = trades['trade_return']
        trades['trade_return'] = sl_results['adjusted_returns']
        trades['stopped_out'] = sl_results['stopped_out']
        trades['took_profit'] = sl_results['took_profit']

        # Stop-loss specific metrics (ensure Python native types for JSON serialization)
        metrics['n_stopped_out'] = int(sl_results['n_stopped'])
        metrics['n_took_profit'] = int(sl_results['n_took_profit'])
        metrics['pct_stopped'] = float(sl_results['pct_stopped'])
        metrics['original_total_return'] = float(sl_results['original_total_return'])
        metrics['sl_improvement'] = float(sl_results['improvement'])

    # Long trades (ensure Python native types for JSON serialization)
    long_trades = trades[trades['is_long']]
    if len(long_trades) > 0:
        metrics['long_trades'] = int(len(long_trades))
        metrics['long_total_return'] = float(long_trades['trade_return'].sum() * 100)
        metrics['long_avg_return'] = float(long_trades['trade_return'].mean() * 100)
        metrics['long_win_rate'] = float((long_trades['trade_return'] > 0).mean() * 100)
        metrics['long_correct_class'] = float((long_trades['true_label'] == 0).mean() * 100)
        if 'stopped_out' in trades.columns:
            metrics['long_stopped'] = int(long_trades['stopped_out'].sum())

    # Short trades
    short_trades = trades[~trades['is_long']]
    if len(short_trades) > 0:
        metrics['short_trades'] = int(len(short_trades))
        metrics['short_total_return'] = float(short_trades['trade_return'].sum() * 100)
        metrics['short_avg_return'] = float(short_trades['trade_return'].mean() * 100)
        metrics['short_win_rate'] = float((short_trades['trade_return'] > 0).mean() * 100)
        metrics['short_correct_class'] = float((short_trades['true_label'] == 4).mean() * 100)
        if 'stopped_out' in trades.columns:
            metrics['short_stopped'] = int(short_trades['stopped_out'].sum())

    # Combined (ensure Python native types for JSON serialization)
    metrics['total_return'] = float(trades['trade_return'].sum() * 100)
    metrics['avg_return_per_trade'] = float(trades['trade_return'].mean() * 100)
    metrics['overall_win_rate'] = float((trades['trade_return'] > 0).mean() * 100)

    # Sharpe ratio (annualized based on actual trading frequency)
    if trades['trade_return'].std() > 0:
        # Estimate trading days from total samples (assuming ~96 candles/day)
        trading_days = len(results_df) / 96
        trades_per_year = len(trades) * (365 / max(trading_days, 1))
        metrics['sharpe_ratio'] = float(
            (trades['trade_return'].mean() / trades['trade_return'].std())
            * np.sqrt(trades_per_year)
        )
        metrics['trades_per_year'] = float(trades_per_year)

        # Also compute the original (overstated) for comparison
        metrics['sharpe_ratio_original'] = float(
            (trades['trade_return'].mean() / trades['trade_return'].std())
            * np.sqrt(35000)
        )
    else:
        metrics['sharpe_ratio'] = 0.0
        metrics['sharpe_ratio_original'] = 0.0

    # Tail risk metrics
    metrics['worst_trade'] = float(trades['trade_return'].min() * 100)
    metrics['best_trade'] = float(trades['trade_return'].max() * 100)
    metrics['max_drawdown'] = float(
        (trades['trade_return'].cumsum().cummax() -
         trades['trade_return'].cumsum()).max() * 100
    )

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

    # Risk Management Settings
    print(f"\nRISK MANAGEMENT")
    print("-" * 40)
    sl_pct = metrics.get('stop_loss_pct')
    tp_pct = metrics.get('take_profit_pct')
    print(f"Stop-Loss:                  {sl_pct*100:.2f}%" if sl_pct else "Stop-Loss:                  None")
    print(f"Take-Profit:                {tp_pct*100:.2f}%" if tp_pct else "Take-Profit:                None (let winners run)")

    if 'n_stopped_out' in metrics:
        print(f"Trades Stopped Out:         {metrics['n_stopped_out']} ({metrics['pct_stopped']:.1f}%)")
        if metrics.get('n_took_profit', 0) > 0:
            print(f"Trades Hit Take-Profit:     {metrics['n_took_profit']}")
        print(f"Original Total Return:      {metrics['original_total_return']:+.2f}%")
        print(f"Stop-Loss Improvement:      {metrics['sl_improvement']:+.2f}%")

    if metrics['total_trades'] > 0:
        print(f"\nLONG TRADES")
        print("-" * 40)
        if 'long_trades' in metrics:
            print(f"Count:                      {metrics['long_trades']:,}")
            print(f"Total Return:               {metrics['long_total_return']:+.2f}%")
            print(f"Avg Return/Trade:           {metrics['long_avg_return']:+.4f}%")
            print(f"Win Rate:                   {metrics['long_win_rate']:.2f}%")
            print(f"Correct Class (HIGH_BULL):  {metrics['long_correct_class']:.2f}%")
            if 'long_stopped' in metrics:
                print(f"Stopped Out:                {metrics['long_stopped']}")
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
            if 'short_stopped' in metrics:
                print(f"Stopped Out:                {metrics['short_stopped']}")
        else:
            print("No short trades")

        print(f"\nOVERALL PERFORMANCE")
        print("-" * 40)
        print(f"Total Return:               {metrics['total_return']:+.2f}%")
        print(f"Avg Return/Trade:           {metrics['avg_return_per_trade']:+.4f}%")
        print(f"Overall Win Rate:           {metrics['overall_win_rate']:.2f}%")
        print(f"Trades per Year (est.):     {metrics.get('trades_per_year', 'N/A'):.1f}")
        print(f"Sharpe Ratio (corrected):   {metrics['sharpe_ratio']:.2f}")

        print(f"\nTAIL RISK")
        print("-" * 40)
        print(f"Worst Trade:                {metrics.get('worst_trade', 'N/A'):+.4f}%")
        print(f"Best Trade:                 {metrics.get('best_trade', 'N/A'):+.4f}%")
        print(f"Max Drawdown:               {metrics.get('max_drawdown', 'N/A'):.2f}%")

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

    # Risk Management Settings (from tail risk analysis)
    # Stop-loss options:
    #   Conservative: -0.0132 (-1.32%) - stops 5% of trades
    #   Moderate:     -0.0178 (-1.78%) - stops 2.5% of trades (RECOMMENDED)
    #   Aggressive:   -0.0227 (-2.27%) - stops 1% of trades
    STOP_LOSS_PCT = -0.0178  # Moderate stop-loss (recommended)
    TAKE_PROFIT_PCT = None   # Let winners run (positive skew in returns)

    # Check paths
    if not (MODEL_DIR / "best_model.pt").exists():
        logger.error(f"Model not found at {MODEL_DIR / 'best_model.pt'}")
        return

    if not (DATA_DIR / "test.parquet").exists():
        logger.error(f"Test data not found at {DATA_DIR / 'test.parquet'}")
        return

    # === Load Model ===
    logger.info(f"Loading calibrated model (threshold={TRADE_THRESHOLD}, stop_loss={STOP_LOSS_PCT*100:.2f}%)...")
    calibrated_model, config = load_calibrated_model(
        MODEL_DIR / "best_model.pt",
        trade_threshold=TRADE_THRESHOLD,
        filter_high_volatility=FILTER_HIGH_VOL,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
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

    # === Compute Metrics (with stop-loss) ===
    metrics = compute_performance(
        results_df,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
    )

    # === Print Report ===
    print_report(metrics)

    # === Compare with and without stop-loss ===
    print("\n" + "=" * 70)
    print("STOP-LOSS COMPARISON")
    print("=" * 70)

    # Without stop-loss
    metrics_no_sl = compute_performance(results_df, stop_loss_pct=None)
    print(f"\nWithout Stop-Loss:")
    print(f"  Total Return:    {metrics_no_sl['total_return']:+.2f}%")
    print(f"  Sharpe Ratio:    {metrics_no_sl['sharpe_ratio']:.2f}")
    print(f"  Worst Trade:     {metrics_no_sl.get('worst_trade', 'N/A'):+.4f}%")

    print(f"\nWith {STOP_LOSS_PCT*100:.2f}% Stop-Loss:")
    print(f"  Total Return:    {metrics['total_return']:+.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Worst Trade:     {metrics.get('worst_trade', 'N/A'):+.4f}%")
    print(f"  Improvement:     {metrics.get('sl_improvement', 0):+.2f}%")
    print("=" * 70)

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
