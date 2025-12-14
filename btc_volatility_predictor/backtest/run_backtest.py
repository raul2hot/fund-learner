"""
Main entry point for backtesting:
1. Generate predictions if not exists
2. Run all strategies
3. Generate analysis
4. Save results
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.generate_predictions import generate_test_predictions
from backtest.engine import BacktestEngine
from backtest.analyze import generate_report
from backtest.strategies import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    MomentumStrategy,
    RegimeSwitchStrategy
)


# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1%
SLIPPAGE = 0.0005  # 0.05%

# Paths
DATA_PATH = "data/processed/features.csv"
CHECKPOINT_PATH = "checkpoints/best_regime_model.pt"
PREDICTIONS_PATH = "backtest/results/test_predictions.csv"
RESULTS_DIR = "backtest/results"


def get_strategies():
    """Initialize all strategies to test."""
    return [
        # Baseline
        BuyAndHoldStrategy(),

        # Mean Reversion variations
        MeanReversionStrategy(rsi_oversold=30, rsi_overbought=70),
        MeanReversionStrategy(rsi_oversold=25, rsi_overbought=75),

        # Breakout variations
        BreakoutStrategy(lookback=24, volume_multiplier=1.5),
        BreakoutStrategy(lookback=12, volume_multiplier=2.0),

        # Momentum variations
        MomentumStrategy(fast_ema=9, slow_ema=21, use_macd_confirmation=True),
        MomentumStrategy(fast_ema=12, slow_ema=26, use_macd_confirmation=False),

        # Regime switching
        RegimeSwitchStrategy(close_on_regime_switch=True),
        RegimeSwitchStrategy(close_on_regime_switch=False),
    ]


def run_all_strategies(predictions_df: pd.DataFrame) -> list:
    """Run backtests for all strategies."""
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategies = get_strategies()
    results = []

    print("\n" + "="*60)
    print("RUNNING BACKTESTS")
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


def main():
    """Main entry point."""
    print("="*60)
    print("BTC VOLATILITY REGIME STRATEGY BACKTESTER")
    print("="*60)

    # Check if model exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Model not found at {CHECKPOINT_PATH}")
        print("Please run train_regime.py first to train the model.")
        return

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data not found at {DATA_PATH}")
        print("Please run data/fetch_binance.py and data/features.py first.")
        return

    # Step 1: Generate predictions if needed
    print("\n[Step 1] Checking for predictions...")
    if os.path.exists(PREDICTIONS_PATH):
        print(f"   Loading existing predictions from {PREDICTIONS_PATH}")
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
    else:
        print(f"   Generating predictions...")
        predictions_df = generate_test_predictions(
            data_path=DATA_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            output_path=PREDICTIONS_PATH
        )

    print(f"   Loaded {len(predictions_df)} test samples")

    # Print prediction summary
    accuracy = predictions_df['correct'].mean()
    high_pct = predictions_df['predicted_regime'].mean()
    print(f"   Prediction accuracy: {accuracy:.1%}")
    print(f"   HIGH regime predictions: {high_pct:.1%}")

    # Step 2: Run all strategies
    print("\n[Step 2] Running backtests...")
    results = run_all_strategies(predictions_df)

    # Step 3: Generate analysis report
    print("\n[Step 3] Generating analysis report...")
    generate_report(results, predictions_df, RESULTS_DIR)

    # Final summary
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nFiles generated:")
    print("  - summary.csv")
    print("  - equity_curves.png")
    print("  - regime_analysis.png")
    print("  - trades_analysis.png")
    print("  - drawdown.png")
    print("  - trades/*.csv")

    # Answer key questions from instructions
    print("\n" + "="*60)
    print("ANSWERING KEY QUESTIONS")
    print("="*60)

    buy_hold = next((r for r in results if r.strategy_name == "BuyAndHold"), None)

    # 1. Does accuracy translate to profitable strategies?
    profitable = [r for r in results if r.total_return > 0]
    print(f"\n1. Profitable strategies: {len(profitable)}/{len(results)}")

    if buy_hold:
        outperform = [r for r in results if r.total_return > buy_hold.total_return]
        print(f"   Strategies beating Buy&Hold: {len(outperform)}/{len(results)-1}")

    # 2. Best Sharpe ratio
    best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
    print(f"\n2. Best Sharpe Ratio: {best_sharpe.strategy_name} ({best_sharpe.sharpe_ratio:.2f})")

    # 3. Regime contribution
    print("\n3. Regime contribution analysis:")
    for r in results:
        if r.num_trades > 0:
            total = r.total_return
            high_contrib = r.regime_returns.get('HIGH', 0)
            low_contrib = r.regime_returns.get('LOW', 0)
            print(f"   {r.strategy_name}: Total={total*100:.1f}%, "
                  f"HIGH={high_contrib*100:.1f}%, LOW={low_contrib*100:.1f}%")

    # 4. Max drawdown
    worst_dd = max(results, key=lambda r: r.max_drawdown)
    best_dd = min(results, key=lambda r: r.max_drawdown)
    print(f"\n4. Drawdown range: {best_dd.max_drawdown*100:.1f}% to {worst_dd.max_drawdown*100:.1f}%")

    # 5. Trading activity
    total_trades = sum(r.num_trades for r in results if r.strategy_name != "BuyAndHold")
    print(f"\n5. Total trades (excl. Buy&Hold): {total_trades}")

    # 6. Statistical note
    print(f"\n6. Sample size: {len(predictions_df)} hours ({len(predictions_df)/24:.0f} days)")
    print("   Note: Limited sample size. Results should be validated on more data.")


if __name__ == "__main__":
    main()
