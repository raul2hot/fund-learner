"""
V3 Backtest Runner - Trend-Filtered Strategies

Tests the hypothesis: Trend filter + Vol filter > Vol filter alone
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.analyze import generate_report
from backtest.strategies import (
    # Baselines
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    # V2 (Vol filter only)
    MeanReversionV2Strategy,
    DefensiveStrategy,
    # V3 (Trend + Vol filter)
    TrendFilteredMeanReversion,
    TrendAdaptiveDefensive,
    TrendFollowerStrategy,
)

# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

DATA_PATH = "data/processed/features_365d.csv"
CHECKPOINT_PATH = "checkpoints/best_regime_model_90d.pt"
PREDICTIONS_PATH = "backtest/results_v2/test_predictions_90d.csv"
RESULTS_DIR = "backtest/results_v3"


def get_v3_strategies():
    """All strategies for V3 comparison."""
    return [
        # === BASELINES ===
        BuyAndHoldStrategy(),

        # === V2: Vol Filter Only ===
        DefensiveStrategy(reentry_delay=2),
        MeanReversionV2Strategy(
            rsi_oversold=30,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            trade_short=True
        ),

        # === V3: Trend Only (no vol filter) ===
        TrendFollowerStrategy(
            fast_ma_period=168,
            slow_ma_period=720,
            allow_short=False
        ),
        TrendFollowerStrategy(
            fast_ma_period=168,
            slow_ma_period=720,
            allow_short=True
        ),

        # === V3: Trend + Vol Filter ===
        TrendFilteredMeanReversion(
            rsi_oversold=30,
            rsi_overbought=70,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_sideways=False
        ),
        TrendFilteredMeanReversion(
            rsi_oversold=30,
            rsi_overbought=70,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_sideways=True  # More aggressive
        ),
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=False,  # Stay flat in downtrend
            trade_sideways=True
        ),
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=True,   # Short in downtrend
            trade_sideways=False
        ),
    ]


def run_v3_backtest():
    """Run V3 backtest with trend-filtered strategies."""

    print("="*60)
    print("BTC VOLATILITY STRATEGY V3 BACKTESTER")
    print("Trend-Filtered Strategies")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/trades", exist_ok=True)

    # Load predictions (reuse from V2)
    if os.path.exists(PREDICTIONS_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
    else:
        print(f"ERROR: Predictions not found at {PREDICTIONS_PATH}")
        print("Run V2 backtest first to generate predictions.")
        return

    print(f"Loaded {len(predictions_df)} samples ({len(predictions_df)/24:.0f} days)")

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategies = get_v3_strategies()
    results = []

    print("\n" + "="*60)
    print("RUNNING V3 BACKTESTS")
    print("="*60)

    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] {strategy.name}")
        print(f"   Params: {strategy.get_params()}")

        result = engine.run(strategy, predictions_df)
        results.append(result)

        if result.num_trades > 0:
            engine.save_trades(result, f"{RESULTS_DIR}/trades")

        print(f"   Return: {result.total_return*100:+.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   MaxDD:  {result.max_drawdown*100:.1f}%")
        print(f"   Trades: {result.num_trades}")
        if result.num_trades > 0:
            print(f"   WinRate: {result.win_rate*100:.1f}%")

    # Compare V2 vs V3
    print("\n" + "="*60)
    print("V2 vs V3 COMPARISON")
    print("="*60)

    # Find key strategies
    buy_hold = next((r for r in results if r.strategy_name == "BuyAndHold"), None)
    defensive_v2 = next((r for r in results if r.strategy_name == "Defensive"), None)
    trend_defensive = [r for r in results if "TrendDefensive" in r.strategy_name]
    trend_mr = [r for r in results if "TrendFilteredMR" in r.strategy_name]

    if buy_hold:
        print(f"\nBuy & Hold: {buy_hold.total_return*100:+.2f}%")

    if defensive_v2:
        print(f"Defensive V2 (Vol Only): {defensive_v2.total_return*100:+.2f}%")

    print("\nTrend-Filtered Strategies:")
    for r in trend_defensive + trend_mr:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% (Sharpe: {r.sharpe_ratio:.2f})")

    # Generate report
    try:
        generate_report(results, predictions_df, RESULTS_DIR)
    except Exception as e:
        print(f"Warning: Could not generate full report: {e}")

    # Save summary
    summary = []
    for r in results:
        summary.append({
            'strategy': r.strategy_name,
            'return_pct': r.total_return * 100,
            'sharpe': r.sharpe_ratio,
            'max_dd_pct': r.max_drawdown * 100,
            'win_rate': r.win_rate * 100 if r.num_trades > 0 else 0,
            'num_trades': r.num_trades,
            'profit_factor': r.profit_factor
        })

    pd.DataFrame(summary).to_csv(f"{RESULTS_DIR}/summary_v3.csv", index=False)

    # Best strategy
    best = max(results, key=lambda r: r.sharpe_ratio)
    print(f"\nBest Strategy: {best.strategy_name}")
    print(f"  Return: {best.total_return*100:+.2f}%")
    print(f"  Sharpe: {best.sharpe_ratio:.2f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_v3_backtest()
