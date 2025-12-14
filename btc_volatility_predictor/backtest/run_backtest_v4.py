"""
V4 Backtest Runner - ADX-Enhanced Strategies

Tests the hypothesis: Vol + Trend + ADX (Triple Filter) > Vol + Trend
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
    # V4 (ADX Enhanced)
    ADXTrendStrategy,
    ADXMeanReversion,
    CombinedADXStrategy,
)

# Configuration
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

DATA_PATH = "data/processed/features_365d.csv"
PREDICTIONS_PATH = "backtest/results_v3/test_predictions_90d.csv"
PREDICTIONS_PATH_V2 = "backtest/results_v2/test_predictions_90d.csv"
RESULTS_DIR = "backtest/results_v4"


def get_v4_strategies():
    """All strategies for V4 comparison."""
    return [
        # === BASELINES ===
        BuyAndHoldStrategy(),

        # === V2: Vol Filter Only ===
        DefensiveStrategy(reentry_delay=2),

        # === V3: Trend + Vol Filter ===
        TrendAdaptiveDefensive(
            reentry_delay=2,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_downtrend=False,
            trade_sideways=True
        ),
        TrendFilteredMeanReversion(
            rsi_oversold=30,
            rsi_overbought=70,
            take_profit_pct=0.02,
            stop_loss_pct=0.015,
            fast_ma_period=168,
            slow_ma_period=720,
            trade_sideways=True
        ),

        # === V4: ADX Strategies (Long Only) ===
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=True,
            allow_short=False,
            require_adx_confirm=True
        ),
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=False,  # Fixed sizing
            allow_short=False,
            require_adx_confirm=True
        ),

        # === V4: ADX Strategies (Long + Short) ===
        ADXTrendStrategy(
            adx_strong=25,
            adx_weak=20,
            use_dynamic_sizing=True,
            allow_short=True,
            require_adx_confirm=True
        ),

        # === V4: Mean Reversion for Ranging Markets ===
        ADXMeanReversion(
            adx_max=20,
            rsi_oversold=25,
            rsi_overbought=75
        ),
        ADXMeanReversion(
            adx_max=18,  # More strict
            rsi_oversold=20,
            rsi_overbought=80
        ),

        # === V4: Combined Strategy (Auto-Switch) ===
        CombinedADXStrategy(
            adx_trend_threshold=25,
            adx_range_threshold=20
        ),
    ]


def run_v4_backtest():
    """Run V4 backtest with ADX-enhanced strategies."""

    print("="*60)
    print("BTC VOLATILITY STRATEGY V4 BACKTESTER")
    print("ADX-Enhanced Strategies (Triple Filter)")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/trades", exist_ok=True)

    # Load predictions (try V3 first, then V2)
    predictions_df = None
    for path in [PREDICTIONS_PATH, PREDICTIONS_PATH_V2]:
        if os.path.exists(path):
            predictions_df = pd.read_csv(path)
            print(f"Loaded predictions from {path}")
            break

    if predictions_df is None:
        print(f"ERROR: Predictions not found")
        print("Run V2 or V3 backtest first to generate predictions.")
        return

    # Check for ADX columns
    if 'adx_14' not in predictions_df.columns:
        print("\nWARNING: ADX features not found in predictions!")
        print("Strategies will use default ADX=20 (moderate)")
        print("For accurate results, regenerate features with ADX:")
        print("  python -c \"from data.features import prepare_dataset; prepare_dataset('data/raw/btcusdt_1h_365d.csv', 'data/processed/features_365d.csv')\"")
        print()

    print(f"Loaded {len(predictions_df)} samples ({len(predictions_df)/24:.0f} days)")

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE
    )

    strategies = get_v4_strategies()
    results = []

    print("\n" + "="*60)
    print("RUNNING V4 BACKTESTS")
    print("="*60)

    for i, strategy in enumerate(strategies):
        print(f"\n[{i+1}/{len(strategies)}] {strategy.name}")
        params = strategy.get_params()
        # Print key params
        for k, v in list(params.items())[:4]:
            print(f"   {k}: {v}")

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

    # V3 vs V4 Comparison
    print("\n" + "="*60)
    print("V3 vs V4 COMPARISON")
    print("="*60)

    # Group results
    buy_hold = next((r for r in results if r.strategy_name == "BuyAndHold"), None)
    v2_strategies = [r for r in results if r.strategy_name == "Defensive"]
    v3_strategies = [r for r in results if "Trend" in r.strategy_name and "ADX" not in r.strategy_name]
    v4_strategies = [r for r in results if "ADX" in r.strategy_name or "Combined" in r.strategy_name]

    if buy_hold:
        print(f"\nBuy & Hold (Benchmark): {buy_hold.total_return*100:+.2f}%")

    print("\nV2 Strategies (Vol Filter Only):")
    for r in v2_strategies:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% (Sharpe: {r.sharpe_ratio:.2f})")

    print("\nV3 Strategies (Vol + Trend):")
    for r in v3_strategies:
        print(f"  {r.strategy_name}: {r.total_return*100:+.2f}% (Sharpe: {r.sharpe_ratio:.2f})")

    print("\nV4 Strategies (Vol + Trend + ADX):")
    for r in v4_strategies:
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

    pd.DataFrame(summary).to_csv(f"{RESULTS_DIR}/summary_v4.csv", index=False)

    # Best strategy by different metrics
    print("\n" + "="*60)
    print("BEST STRATEGIES BY METRIC")
    print("="*60)

    # Filter to strategies with trades
    with_trades = [r for r in results if r.num_trades > 0]

    if with_trades:
        best_return = max(with_trades, key=lambda r: r.total_return)
        best_sharpe = max(with_trades, key=lambda r: r.sharpe_ratio)
        lowest_dd = min(with_trades, key=lambda r: r.max_drawdown)

        print(f"\nBest Return: {best_return.strategy_name}")
        print(f"  Return: {best_return.total_return*100:+.2f}%, Sharpe: {best_return.sharpe_ratio:.2f}")

        print(f"\nBest Sharpe: {best_sharpe.strategy_name}")
        print(f"  Sharpe: {best_sharpe.sharpe_ratio:.2f}, Return: {best_sharpe.total_return*100:+.2f}%")

        print(f"\nLowest Drawdown: {lowest_dd.strategy_name}")
        print(f"  MaxDD: {lowest_dd.max_drawdown*100:.1f}%, Return: {lowest_dd.total_return*100:+.2f}%")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_v4_backtest()
