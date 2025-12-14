"""
Extended Backtest Runner V2

Key differences from V1:
1. Uses 90-day test period (2160 hours)
2. Tests V2 strategies (MeanReversionV2, Defensive, DirectionAware)
3. Compares filtered vs unfiltered results
4. Validates statistical significance with more data

Based on proven approaches from vol_filtered_mean_reversion_v2.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine, BacktestResult
from backtest.analyze import generate_report
from backtest.strategies import (
    # V1 Strategies (for comparison)
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    # V2 Strategies (improved)
    MeanReversionV2Strategy,
    DefensiveStrategy,
    DefensiveWithTPSL,
    DirectionAwareMeanReversion,
    DirectionOnlyStrategy,
)


# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1%
SLIPPAGE = 0.0005  # 0.05%

# Paths - V2 uses extended data
DATA_PATH = "data/processed/features_365d.csv"
CHECKPOINT_PATH = "checkpoints/best_regime_model_90d.pt"
DIRECTION_CHECKPOINT = "checkpoints/best_direction_model.pt"
PREDICTIONS_PATH = "backtest/results_v2/test_predictions_90d.csv"
RESULTS_DIR = "backtest/results_v2"


def generate_predictions_v2(
    data_path: str,
    checkpoint_path: str,
    direction_checkpoint: str,
    output_path: str,
    test_days: int = 90
) -> pd.DataFrame:
    """
    Generate predictions using the 90-day test model.
    Also adds direction predictions if model available.
    """
    import torch
    from sklearn.preprocessing import RobustScaler

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    from models import SPHNet
    from train_regime import VolatilityRegimeDataset

    print("Loading data and model...")

    # Load data
    df = pd.read_csv(data_path)
    n_total = len(df)
    n_test = test_days * 24
    test_df = df.iloc[-n_test:].copy()

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Setup config
    config = Config()
    if 'config' in checkpoint:
        for k, v in checkpoint['config'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    vol_threshold = checkpoint.get('vol_threshold', 0.01)

    # Create dataset (need to fit scalers on train data first)
    train_df = df.iloc[:-n_test]
    train_dataset = VolatilityRegimeDataset(
        train_df, window_size=config.window_size, fit_scalers=True, vol_threshold=vol_threshold
    )

    test_dataset = VolatilityRegimeDataset(
        test_df, window_size=config.window_size,
        price_scaler=train_dataset.price_scaler,
        feature_scaler=train_dataset.feature_scaler,
        vol_threshold=vol_threshold
    )

    # Load volatility model
    vol_model = SPHNet(config).to(device)
    vol_model.load_state_dict(checkpoint['model_state_dict'])
    vol_model.eval()

    # Try to load direction model
    dir_model = None
    if os.path.exists(direction_checkpoint):
        try:
            dir_checkpoint = torch.load(direction_checkpoint, map_location=device, weights_only=False)
            dir_config = Config()
            if 'config' in dir_checkpoint:
                for k, v in dir_checkpoint['config'].items():
                    if hasattr(dir_config, k):
                        setattr(dir_config, k, v)
            dir_model = SPHNet(dir_config).to(device)
            dir_model.load_state_dict(dir_checkpoint['model_state_dict'])
            dir_model.eval()
            print("Direction model loaded successfully!")
        except Exception as e:
            print(f"Could not load direction model: {e}")
            dir_model = None

    # Generate predictions
    print("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            prices = sample['prices'].unsqueeze(0).to(device)
            features = sample['features'].unsqueeze(0).to(device)
            target = sample['target'].item()

            # Volatility prediction
            vol_out = vol_model(prices, features)
            vol_prob = torch.sigmoid(vol_out['direction_pred']).item()
            vol_pred = 1 if vol_prob > 0.5 else 0

            # Direction prediction
            if dir_model:
                dir_out = dir_model(prices, features)
                dir_prob = torch.sigmoid(dir_out['direction_pred']).item()
                dir_pred = 'UP' if dir_prob > 0.5 else 'DOWN'
                dir_confidence = abs(dir_prob - 0.5) * 2
            else:
                dir_prob = 0.5
                dir_pred = 'UNKNOWN'
                dir_confidence = 0.0

            predictions.append({
                'idx': i,
                'predicted_regime': vol_pred,
                'actual_regime': target,
                'correct': vol_pred == target,
                'vol_prob': vol_prob,
                'predicted_direction': dir_pred,
                'dir_prob': dir_prob,
                'dir_confidence': dir_confidence
            })

    predictions_df = pd.DataFrame(predictions)

    # Merge with actual price data
    test_start = n_total - n_test + config.window_size
    price_data = df.iloc[test_start:test_start + len(predictions_df)].reset_index(drop=True)

    for col in price_data.columns:
        predictions_df[col] = price_data[col].values

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved {len(predictions_df)} predictions to {output_path}")

    return predictions_df


def get_v2_strategies():
    """Initialize V2 strategies to test."""
    return [
        # Baseline
        BuyAndHoldStrategy(),

        # V1 Mean Reversion (for comparison)
        MeanReversionStrategy(rsi_oversold=30, rsi_overbought=70),

        # V2 Mean Reversion with fixed TP/SL
        MeanReversionV2Strategy(
            rsi_oversold=30,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            max_holding_bars=24,
            trade_short=True
        ),

        # V2 Mean Reversion (more conservative)
        MeanReversionV2Strategy(
            rsi_oversold=25,
            take_profit_pct=0.025,
            stop_loss_pct=0.01,
            max_holding_bars=48,
            trade_short=False
        ),

        # Defensive strategies
        DefensiveStrategy(reentry_delay=2),
        DefensiveWithTPSL(reentry_delay=2, take_profit_pct=0.03, stop_loss_pct=0.02),

        # Direction-aware (if direction model available)
        DirectionAwareMeanReversion(
            require_direction_confirmation=True,
            min_direction_confidence=0.0
        ),

        DirectionAwareMeanReversion(
            require_direction_confirmation=True,
            min_direction_confidence=0.6
        ),

        DirectionOnlyStrategy(
            min_confidence=0.6,
            take_profit_pct=0.02,
            stop_loss_pct=0.015
        ),
    ]


def run_filtered_comparison(predictions_df: pd.DataFrame, engine: BacktestEngine) -> dict:
    """
    Compare vol-filtered vs unfiltered mean reversion.
    This validates the key insight from vol_filtered_mean_reversion_v2.py
    """
    print("\n" + "="*60)
    print("VOL-FILTERED VS UNFILTERED COMPARISON")
    print("="*60)

    # Unfiltered: Use dummy predictions that always say LOW
    unfiltered_df = predictions_df.copy()
    unfiltered_df['predicted_regime'] = 0  # Always LOW

    # Filtered: Use actual predictions
    filtered_df = predictions_df.copy()

    # Run MeanReversionV2 on both
    strategy_unfiltered = MeanReversionV2Strategy(
        rsi_oversold=30,
        take_profit_pct=0.02,
        stop_loss_pct=0.015,
        trade_short=True
    )
    strategy_unfiltered.name = "MR_V2_Unfiltered"

    strategy_filtered = MeanReversionV2Strategy(
        rsi_oversold=30,
        take_profit_pct=0.02,
        stop_loss_pct=0.015,
        trade_short=True
    )
    strategy_filtered.name = "MR_V2_Filtered"

    result_unfiltered = engine.run(strategy_unfiltered, unfiltered_df)
    result_filtered = engine.run(strategy_filtered, filtered_df)

    print(f"\nUnfiltered (trade all conditions):")
    print(f"  Return: {result_unfiltered.total_return*100:.2f}%")
    print(f"  Sharpe: {result_unfiltered.sharpe_ratio:.2f}")
    print(f"  Trades: {result_unfiltered.num_trades}")
    print(f"  Win Rate: {result_unfiltered.win_rate*100:.1f}%")

    print(f"\nFiltered (LOW vol only):")
    print(f"  Return: {result_filtered.total_return*100:.2f}%")
    print(f"  Sharpe: {result_filtered.sharpe_ratio:.2f}")
    print(f"  Trades: {result_filtered.num_trades}")
    print(f"  Win Rate: {result_filtered.win_rate*100:.1f}%")

    improvement = result_filtered.total_return - result_unfiltered.total_return
    print(f"\nImprovement from filtering: {improvement*100:.2f}%")

    return {
        'unfiltered': result_unfiltered.to_dict(),
        'filtered': result_filtered.to_dict(),
        'improvement': improvement
    }


def run_all_v2_strategies(predictions_df: pd.DataFrame) -> list:
    """Run backtests for all V2 strategies."""
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategies = get_v2_strategies()
    results = []

    print("\n" + "="*60)
    print("RUNNING V2 BACKTESTS (90-day test period)")
    print("="*60)

    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] Testing {strategy.name}...")

        # Print strategy parameters
        params = strategy.get_params()
        print(f"   Parameters: {params}")

        # Run backtest
        result = engine.run(strategy, predictions_df)
        results.append(result)

        # Save trade log
        if result.num_trades > 0:
            trades_file = engine.save_trades(result, f"{RESULTS_DIR}/trades")
            print(f"   Trades saved to: {trades_file}")

        # Print quick summary
        print(f"   Return: {result.total_return*100:.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   Max DD: {result.max_drawdown*100:.2f}%")
        print(f"   Trades: {result.num_trades}")

        if result.num_trades > 0:
            print(f"   Win Rate: {result.win_rate*100:.1f}%")

    return results


def validate_success_criteria(results: list, predictions_df: pd.DataFrame) -> dict:
    """
    Validate against success criteria from instructions:
    1. MeanReversionV2 is profitable (>0% return)
    2. Sharpe ratio > 1.0 for best strategy
    3. Max drawdown < 15%
    4. At least 50+ trades for statistical significance
    5. Vol filtering shows clear improvement
    """
    print("\n" + "="*60)
    print("SUCCESS CRITERIA VALIDATION")
    print("="*60)

    criteria = {}

    # 1. MeanReversionV2 profitability
    mr_v2_results = [r for r in results if 'MeanReversionV2' in r.strategy_name]
    mr_v2_profitable = any(r.total_return > 0 for r in mr_v2_results)
    criteria['mr_v2_profitable'] = bool(mr_v2_profitable)
    print(f"\n1. MeanReversionV2 profitable: {'PASS' if mr_v2_profitable else 'FAIL'}")
    for r in mr_v2_results:
        print(f"   {r.strategy_name}: {r.total_return*100:.2f}%")

    # 2. Best Sharpe > 1.0
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    sharpe_pass = best_sharpe.sharpe_ratio > 1.0
    criteria['best_sharpe_gt_1'] = bool(sharpe_pass)
    print(f"\n2. Best Sharpe > 1.0: {'PASS' if sharpe_pass else 'FAIL'}")
    print(f"   {best_sharpe.strategy_name}: Sharpe = {best_sharpe.sharpe_ratio:.2f}")

    # 3. Max drawdown < 15%
    all_dd_ok = all(r.max_drawdown < 0.15 for r in results)
    criteria['max_dd_lt_15'] = bool(all_dd_ok)
    worst_dd = max(results, key=lambda r: r.max_drawdown)
    print(f"\n3. All max drawdown < 15%: {'PASS' if all_dd_ok else 'FAIL'}")
    print(f"   Worst: {worst_dd.strategy_name} at {worst_dd.max_drawdown*100:.1f}%")

    # 4. 50+ trades for significance
    active_results = [r for r in results if r.strategy_name != 'BuyAndHold']
    trades_sufficient = any(r.num_trades >= 50 for r in active_results)
    criteria['sufficient_trades'] = bool(trades_sufficient)
    total_trades = sum(r.num_trades for r in active_results)
    print(f"\n4. At least one strategy with 50+ trades: {'PASS' if trades_sufficient else 'FAIL'}")
    print(f"   Total trades across strategies: {total_trades}")

    # 5. Statistical significance note
    n_samples = len(predictions_df)
    n_days = n_samples / 24
    print(f"\n5. Sample size: {n_samples} hours ({n_days:.0f} days)")

    # Overall
    all_pass = all([mr_v2_profitable, sharpe_pass, all_dd_ok, trades_sufficient])
    criteria['all_passed'] = bool(all_pass)
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED'}")
    print(f"{'='*60}")

    return criteria


def main():
    """Main entry point for V2 backtest."""
    print("="*60)
    print("BTC VOLATILITY STRATEGY V2 BACKTESTER")
    print("Extended 90-day Test Period")
    print("="*60)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/trades", exist_ok=True)

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Extended data not found at {DATA_PATH}")
        print("Please run the following first:")
        print("  python fetch_extended_data.py")
        print("  python -c \"from data.features import prepare_dataset; prepare_dataset('data/raw/btcusdt_1h_365d.csv', 'data/processed/features_365d.csv')\"")
        return

    # Check if model exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Extended model not found at {CHECKPOINT_PATH}")
        print("Please run train_regime_extended.py first.")
        return

    # Step 1: Generate predictions
    print("\n[Step 1] Generating predictions...")
    if os.path.exists(PREDICTIONS_PATH):
        print(f"   Loading existing predictions from {PREDICTIONS_PATH}")
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
    else:
        predictions_df = generate_predictions_v2(
            data_path=DATA_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            direction_checkpoint=DIRECTION_CHECKPOINT,
            output_path=PREDICTIONS_PATH,
            test_days=90
        )

    print(f"   Loaded {len(predictions_df)} test samples ({len(predictions_df)/24:.0f} days)")

    # Print prediction summary
    accuracy = predictions_df['correct'].mean()
    high_pct = predictions_df['predicted_regime'].mean()
    print(f"   Volatility prediction accuracy: {accuracy:.1%}")
    print(f"   HIGH regime predictions: {high_pct:.1%}")

    if 'predicted_direction' in predictions_df.columns:
        dir_available = predictions_df['predicted_direction'].iloc[0] != 'UNKNOWN'
        print(f"   Direction predictions available: {dir_available}")

    # Step 2: Run vol-filtered comparison
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )
    comparison = run_filtered_comparison(predictions_df, engine)

    # Step 3: Run all V2 strategies
    print("\n[Step 2] Running V2 strategy backtests...")
    results = run_all_v2_strategies(predictions_df)

    # Step 4: Validate success criteria
    criteria = validate_success_criteria(results, predictions_df)

    # Step 5: Generate analysis report
    print("\n[Step 3] Generating analysis report...")
    try:
        generate_report(results, predictions_df, RESULTS_DIR)
    except Exception as e:
        print(f"   Warning: Could not generate full report: {e}")

    # Save summary
    summary_data = []
    for r in results:
        summary_data.append({
            'strategy': r.strategy_name,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'win_rate': r.win_rate,
            'num_trades': r.num_trades,
            'avg_trade_pnl': r.avg_trade_pnl,
            'profit_factor': r.profit_factor,
            'regime_return_high': r.regime_returns.get('HIGH', 0),
            'regime_return_low': r.regime_returns.get('LOW', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)

    # Save detailed results
    with open(f"{RESULTS_DIR}/detailed_results.json", 'w') as f:
        json.dump({
            'comparison': comparison,
            'criteria': criteria,
            'prediction_accuracy': float(accuracy),
            'test_days': 90,
            'n_samples': int(len(predictions_df))
        }, f, indent=2)

    # Final summary
    print("\n" + "="*60)
    print("V2 BACKTEST COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nFiles generated:")
    print("  - summary.csv")
    print("  - detailed_results.json")
    print("  - trades/*.csv")

    # Print best strategy
    best_return = max(results, key=lambda r: r.total_return)
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)

    print(f"\nBest by Return: {best_return.strategy_name} ({best_return.total_return*100:.2f}%)")
    print(f"Best by Sharpe: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")

    # Compare to buy and hold
    buy_hold = next((r for r in results if r.strategy_name == "BuyAndHold"), None)
    if buy_hold:
        outperform = [r for r in results if r.total_return > buy_hold.total_return]
        print(f"\nStrategies beating Buy&Hold: {len(outperform)}/{len(results)-1}")


if __name__ == "__main__":
    main()
