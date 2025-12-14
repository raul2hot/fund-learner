"""
V4.1 Backtest Runner - Simplified ADX Strategies

Fixes V4's zero-trade problem with relaxed conditions:
- Shorter MA periods (72/168 vs 168/720)
- RSI level instead of crossover
- Lower ADX thresholds (15 vs 25)
- ADX optional (works without it)
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.analyze import generate_report
from backtest.strategies import (
    # Baselines
    BuyAndHoldStrategy,
    DefensiveStrategy,
    # V3 (Trend + Vol filter)
    TrendAdaptiveDefensive,
    TrendFilteredMeanReversion,
    # V4 (ADX Enhanced - original)
    ADXTrendStrategy,
    ADXMeanReversion,
    CombinedADXStrategy,
    # V4.1 (Simplified ADX)
    SimpleADXTrend,
    TrendStrengthStrategy,
    VolTrendCombo,
)

# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

PREDICTIONS_PATH_V3 = "backtest/results_v3/test_predictions_90d.csv"
PREDICTIONS_PATH_V2 = "backtest/results_v2/test_predictions_90d.csv"
RESULTS_DIR = "backtest/results_v4_1"


def get_v4_1_strategies():
    """All strategies for V4.1 comparison."""
    return [
        # === BASELINES ===
        BuyAndHoldStrategy(),
        DefensiveStrategy(reentry_delay=2),

        # === V3: Trend + Vol Filter ===
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=False,
            trade_sideways=True
        ),

        # === V4: Original ADX (for comparison) ===
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=True,
            allow_short=False,
            require_adx_confirm=True
        ),

        # === V4.1: Simplified Strategies ===
        # Simplest - just Vol + Trend
        VolTrendCombo(
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            max_holding_bars=72
        ),

        # ADX for sizing only
        TrendStrengthStrategy(
            base_tp=0.02,
            base_sl=0.015
        ),

        # Relaxed ADX (no ADX required)
        SimpleADXTrend(
            adx_min=15,
            rsi_entry_long=45,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            require_adx=False
        ),

        # Relaxed ADX (ADX required but lower threshold)
        SimpleADXTrend(
            adx_min=15,
            rsi_entry_long=45,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            require_adx=True
        ),

        # More aggressive RSI entry
        SimpleADXTrend(
            adx_min=10,
            rsi_entry_long=50,  # Enter even at RSI 50
            take_profit_pct=0.015,
            stop_loss_pct=0.01,
            require_adx=False
        ),
    ]


def run_v4_1_backtest():
    """Run V4.1 backtest with simplified ADX strategies."""

    print("="*60)
    print("BTC VOLATILITY STRATEGY V4.1 BACKTESTER")
    print("Simplified ADX Strategies (Relaxed Conditions)")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/trades", exist_ok=True)

    # Load predictions
    predictions_df = None
    for path in [PREDICTIONS_PATH_V3, PREDICTIONS_PATH_V2]:
        if os.path.exists(path):
            predictions_df = pd.read_csv(path)
            print(f"Loaded predictions from {path}")
            break

    if predictions_df is None:
        print("ERROR: Predictions not found")
        print("Run V2 or V3 backtest first to generate predictions.")
        return

    # Check for ADX columns
    has_adx = 'adx_14' in predictions_df.columns
    print(f"ADX available: {has_adx}")
    if has_adx:
        print(f"ADX stats: mean={predictions_df['adx_14'].mean():.1f}, "
              f"min={predictions_df['adx_14'].min():.1f}, "
              f"max={predictions_df['adx_14'].max():.1f}")
    else:
        print("WARNING: ADX not in data. Strategies will use defaults.")

    print(f"Loaded {len(predictions_df)} samples ({len(predictions_df)/24:.0f} days)")

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategies = get_v4_1_strategies()
    results = []

    print("\n" + "="*60)
    print("RUNNING V4.1 BACKTESTS")
    print("="*60)

    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] {strategy.name}")

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

    # Summary
    print("\n" + "="*60)
    print("V4.1 RESULTS SUMMARY")
    print("="*60)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'WinRate':>10}")
    print("-"*65)

    for r in sorted(results, key=lambda x: x.total_return, reverse=True):
        wr = f"{r.win_rate*100:.1f}%" if r.num_trades > 0 else "N/A"
        print(f"{r.strategy_name:<25} {r.total_return*100:>+9.2f}% {r.sharpe_ratio:>8.2f} {r.num_trades:>8} {wr:>10}")

    # V4 vs V4.1 Comparison
    print("\n" + "="*60)
    print("V4 vs V4.1 COMPARISON")
    print("="*60)

    v4_strats = [r for r in results if r.strategy_name == "ADXTrend"]
    v41_strats = [r for r in results if r.strategy_name in ["VolTrendCombo", "TrendStrength", "SimpleADXTrend"]]

    print("\nV4 (Original - strict conditions):")
    for r in v4_strats:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% ({r.num_trades} trades)")

    print("\nV4.1 (Simplified - relaxed conditions):")
    for r in v41_strats:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% ({r.num_trades} trades)")

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

    pd.DataFrame(summary).to_csv(f"{RESULTS_DIR}/summary_v4_1.csv", index=False)

    # Best strategies
    with_trades = [r for r in results if r.num_trades > 0]
    if with_trades:
        best = max(with_trades, key=lambda r: r.sharpe_ratio)
        print(f"\nBest Strategy (by Sharpe): {best.strategy_name}")
        print(f"  Return: {best.total_return*100:+.2f}%")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")
        print(f"  Trades: {best.num_trades}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_v4_1_backtest()
