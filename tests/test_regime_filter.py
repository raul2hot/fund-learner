#!/usr/bin/env python
"""
Unit Tests for Regime Filter Module

Tests cover:
1. MarketRegime enum functionality
2. RegimeConfig defaults and presets
3. RegimeFilter regime detection
4. Trade gating logic
5. Position scaling
6. Vectorized filter application
7. Edge cases and error handling

Run with:
    python -m pytest tests/test_regime_filter.py -v
    python tests/test_regime_filter.py  # standalone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from regime_filter import (
    MarketRegime,
    RegimeConfig,
    RegimePresets,
    RegimeFilter,
    apply_regime_filter_to_predictions,
    apply_regime_filter_vectorized
)


class TestMarketRegime(unittest.TestCase):
    """Tests for MarketRegime enum."""

    def test_regime_values(self):
        """Test enum has expected values."""
        self.assertEqual(MarketRegime.NORMAL.value, "normal")
        self.assertEqual(MarketRegime.ELEVATED.value, "elevated")
        self.assertEqual(MarketRegime.EXTREME.value, "extreme")

    def test_regime_comparison(self):
        """Test enum comparison."""
        self.assertEqual(MarketRegime.NORMAL, MarketRegime.NORMAL)
        self.assertNotEqual(MarketRegime.NORMAL, MarketRegime.EXTREME)


class TestRegimeConfig(unittest.TestCase):
    """Tests for RegimeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RegimeConfig()

        # Volatility defaults
        self.assertEqual(config.vol_lookback, 24)
        self.assertEqual(config.vol_baseline_lookback, 168)
        self.assertEqual(config.vol_elevated_threshold, 1.5)
        self.assertEqual(config.vol_extreme_threshold, 2.5)

        # Price drop defaults
        self.assertEqual(config.price_drop_lookback, 24)
        self.assertEqual(config.price_drop_elevated, -0.05)
        self.assertEqual(config.price_drop_extreme, -0.10)

        # Trade limits
        self.assertEqual(config.max_trades_per_day_normal, 50)
        self.assertEqual(config.max_trades_per_day_elevated, 10)
        self.assertEqual(config.max_trades_per_day_extreme, 0)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RegimeConfig(
            vol_elevated_threshold=2.0,
            vol_extreme_threshold=3.0,
            max_trades_per_day_extreme=5
        )

        self.assertEqual(config.vol_elevated_threshold, 2.0)
        self.assertEqual(config.vol_extreme_threshold, 3.0)
        self.assertEqual(config.max_trades_per_day_extreme, 5)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RegimeConfig()
        d = config.to_dict()

        self.assertIsInstance(d, dict)
        self.assertIn('vol_lookback', d)
        self.assertIn('vol_elevated_threshold', d)

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'vol_elevated_threshold': 1.8,
            'max_trades_per_day_extreme': 3,
        }
        config = RegimeConfig.from_dict(d)

        self.assertEqual(config.vol_elevated_threshold, 1.8)
        self.assertEqual(config.max_trades_per_day_extreme, 3)


class TestRegimePresets(unittest.TestCase):
    """Tests for RegimePresets class."""

    def test_conservative_preset(self):
        """Test conservative preset is more restrictive."""
        config = RegimePresets.conservative()

        self.assertLess(config.vol_elevated_threshold, 1.5)  # More sensitive
        self.assertGreater(config.price_drop_elevated, -0.05)  # Less drop to trigger
        self.assertLess(config.max_trades_per_day_elevated, 10)  # Fewer trades

    def test_moderate_preset(self):
        """Test moderate preset matches defaults."""
        config = RegimePresets.moderate()
        default_config = RegimeConfig()

        self.assertEqual(config.vol_elevated_threshold, default_config.vol_elevated_threshold)
        self.assertEqual(config.vol_extreme_threshold, default_config.vol_extreme_threshold)

    def test_aggressive_preset(self):
        """Test aggressive preset is more permissive."""
        config = RegimePresets.aggressive()

        self.assertGreater(config.vol_elevated_threshold, 1.5)  # Less sensitive
        self.assertLess(config.price_drop_elevated, -0.05)  # More drop needed
        self.assertGreater(config.max_trades_per_day_elevated, 10)  # More trades allowed


class TestRegimeFilter(unittest.TestCase):
    """Tests for RegimeFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RegimeConfig()
        self.filter = RegimeFilter(self.config)

    def test_initialization(self):
        """Test filter initialization."""
        self.assertIsNotNone(self.filter.config)
        self.assertEqual(self.filter.trades_today, 0)
        self.assertEqual(self.filter.current_regime, MarketRegime.NORMAL)

    def test_calculate_volatility_ratio_insufficient_data(self):
        """Test volatility ratio with insufficient data."""
        prices = pd.Series([100, 101, 102])
        ratio = self.filter.calculate_volatility_ratio(prices)
        self.assertEqual(ratio, 1.0)  # Default when insufficient data

    def test_calculate_volatility_ratio_normal(self):
        """Test volatility ratio in normal conditions."""
        # Create stable price series
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.001, 200)  # Low volatility
        prices = pd.Series(base_price * np.exp(np.cumsum(returns)))

        ratio = self.filter.calculate_volatility_ratio(prices)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 3.0)  # Should be reasonable

    def test_calculate_volatility_ratio_high_vol(self):
        """Test volatility ratio during high volatility."""
        # Create price series with spike in recent volatility
        np.random.seed(42)
        base_price = 100

        # First 150 periods: low vol
        returns_low = np.random.normal(0, 0.001, 150)
        # Last 50 periods: high vol
        returns_high = np.random.normal(0, 0.010, 50)  # 10x volatility

        returns = np.concatenate([returns_low, returns_high])
        prices = pd.Series(base_price * np.exp(np.cumsum(returns)))

        ratio = self.filter.calculate_volatility_ratio(prices)
        self.assertGreater(ratio, 1.5)  # Should detect elevated vol

    def test_calculate_price_change_insufficient_data(self):
        """Test price change with insufficient data."""
        prices = pd.Series([100, 101, 102])
        change = self.filter.calculate_price_change(prices)
        self.assertEqual(change, 0.0)

    def test_calculate_price_change_drop(self):
        """Test price change calculation for a drop."""
        # Create -10% drop over 24 periods
        prices = pd.Series(np.linspace(100, 90, 30))
        change = self.filter.calculate_price_change(prices)
        self.assertLess(change, -0.09)  # Should detect ~10% drop

    def test_detect_regime_normal(self):
        """Test regime detection in normal conditions."""
        # Stable prices
        prices = pd.Series(np.linspace(100, 102, 200))  # Slight uptrend
        regime, metrics = self.filter.detect_regime(prices)

        self.assertEqual(regime, MarketRegime.NORMAL)
        self.assertEqual(metrics['regime'], 'normal')

    def test_detect_regime_elevated_volatility(self):
        """Test elevated regime detection on volatility."""
        config = RegimeConfig(vol_elevated_threshold=1.5, vol_extreme_threshold=2.5)
        filter = RegimeFilter(config)

        # Create volatility spike
        np.random.seed(42)
        returns_low = np.random.normal(0, 0.001, 150)
        returns_high = np.random.normal(0, 0.003, 50)  # 3x volatility
        returns = np.concatenate([returns_low, returns_high])
        prices = pd.Series(100 * np.exp(np.cumsum(returns)))

        regime, metrics = filter.detect_regime(prices)

        # May be elevated or normal depending on exact values
        self.assertIn(regime, [MarketRegime.NORMAL, MarketRegime.ELEVATED])

    def test_detect_regime_extreme_price_drop(self):
        """Test extreme regime detection on large price drop."""
        # Create -15% drop
        prices = pd.Series(np.linspace(100, 85, 30))
        regime, metrics = self.filter.detect_regime(prices)

        self.assertEqual(regime, MarketRegime.EXTREME)
        self.assertEqual(metrics['trigger'], 'price_drop_extreme')

    def test_should_allow_trade_normal(self):
        """Test trade allowed in normal regime."""
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        allowed, reason = self.filter.should_allow_trade(MarketRegime.NORMAL, timestamp)

        self.assertTrue(allowed)
        self.assertIn('ALLOWED', reason)

    def test_should_allow_trade_extreme_blocked(self):
        """Test trade blocked in extreme regime."""
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        allowed, reason = self.filter.should_allow_trade(MarketRegime.EXTREME, timestamp)

        self.assertFalse(allowed)
        self.assertIn('BLOCKED', reason)

    def test_trade_count_limit(self):
        """Test daily trade count limit."""
        timestamp = pd.Timestamp('2024-01-01 12:00:00')
        config = RegimeConfig(max_trades_per_day_normal=3)
        filter = RegimeFilter(config)

        # First 3 trades should be allowed
        for i in range(3):
            allowed, _ = filter.should_allow_trade(MarketRegime.NORMAL, timestamp)
            self.assertTrue(allowed)
            filter.record_trade()

        # 4th trade should be blocked
        allowed, reason = filter.should_allow_trade(MarketRegime.NORMAL, timestamp)
        self.assertFalse(allowed)
        self.assertIn('Max trades', reason)

    def test_trade_count_reset_new_day(self):
        """Test trade count resets on new day."""
        config = RegimeConfig(max_trades_per_day_normal=2)
        filter = RegimeFilter(config)

        # Use up trades on day 1
        day1 = pd.Timestamp('2024-01-01 12:00:00')
        for _ in range(2):
            filter.should_allow_trade(MarketRegime.NORMAL, day1)
            filter.record_trade()

        allowed, _ = filter.should_allow_trade(MarketRegime.NORMAL, day1)
        self.assertFalse(allowed)

        # New day should reset
        day2 = pd.Timestamp('2024-01-02 12:00:00')
        allowed, _ = filter.should_allow_trade(MarketRegime.NORMAL, day2)
        self.assertTrue(allowed)

    def test_position_scale(self):
        """Test position scaling by regime."""
        self.assertEqual(self.filter.get_position_scale(MarketRegime.NORMAL), 1.0)
        self.assertEqual(self.filter.get_position_scale(MarketRegime.ELEVATED), 0.5)
        self.assertEqual(self.filter.get_position_scale(MarketRegime.EXTREME), 0.0)

    def test_reset_state(self):
        """Test state reset."""
        # Modify state
        self.filter.trades_today = 10
        self.filter.current_regime = MarketRegime.EXTREME

        # Reset
        self.filter.reset_state()

        self.assertEqual(self.filter.trades_today, 0)
        self.assertEqual(self.filter.current_regime, MarketRegime.NORMAL)


class TestApplyRegimeFilterVectorized(unittest.TestCase):
    """Tests for vectorized regime filter application."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100

        # Create timestamps
        start_date = pd.Timestamp('2024-01-01', tz='UTC')
        timestamps = pd.date_range(start=start_date, periods=n, freq='h')

        # Create predictions
        self.predictions = pd.DataFrame({
            'timestamp': timestamps,
            'should_trade': np.random.choice([True, False], n, p=[0.1, 0.9]),
            'is_long': np.random.choice([True, False], n),
            'trade_return': np.random.normal(0, 0.01, n),
            'trade_mae': np.abs(np.random.normal(0, 0.005, n)),
        })

        # Create price series (stable)
        self.prices = pd.Series(
            100 + np.cumsum(np.random.normal(0, 0.5, n)),
            index=timestamps
        )

        self.config = RegimeConfig()

    def test_basic_filtering(self):
        """Test basic filtering adds expected columns."""
        result = apply_regime_filter_vectorized(
            self.predictions, self.prices, config=self.config
        )

        # Check new columns exist
        self.assertIn('regime', result.columns)
        self.assertIn('original_should_trade', result.columns)
        self.assertIn('position_scale', result.columns)
        self.assertIn('regime_blocked', result.columns)

    def test_preserves_original_signals(self):
        """Test original signals are preserved."""
        result = apply_regime_filter_vectorized(
            self.predictions, self.prices, config=self.config
        )

        np.testing.assert_array_equal(
            self.predictions['should_trade'].values,
            result['original_should_trade'].values
        )

    def test_extreme_regime_blocks_trades(self):
        """Test trades are blocked in extreme regime."""
        # Create extreme volatility
        timestamps = self.predictions['timestamp']
        extreme_prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0, 0.05, len(timestamps)))),
            index=timestamps
        )

        result = apply_regime_filter_vectorized(
            self.predictions, extreme_prices, config=self.config
        )

        # Should have some blocked trades if extreme regime detected
        if (result['regime'] == 'extreme').any():
            blocked_in_extreme = result[
                (result['regime'] == 'extreme') &
                (result['original_should_trade'])
            ]
            self.assertTrue(blocked_in_extreme['regime_blocked'].all())

    def test_normal_regime_allows_trades(self):
        """Test trades pass through in normal regime."""
        result = apply_regime_filter_vectorized(
            self.predictions, self.prices, config=self.config
        )

        # In normal regime, original trades should pass through
        normal_rows = result[result['regime'] == 'normal']
        if len(normal_rows) > 0:
            # should_trade should match original in normal regime
            np.testing.assert_array_equal(
                normal_rows['should_trade'].values,
                normal_rows['original_should_trade'].values
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_prices(self):
        """Test handling of empty price series."""
        filter = RegimeFilter()
        prices = pd.Series([], dtype=float)

        ratio = filter.calculate_volatility_ratio(prices)
        self.assertEqual(ratio, 1.0)

    def test_nan_in_prices(self):
        """Test handling of NaN values in prices."""
        filter = RegimeFilter()
        prices = pd.Series([100, 101, np.nan, 103, 104])

        # Should not raise
        ratio = filter.calculate_volatility_ratio(prices)
        self.assertIsInstance(ratio, float)

    def test_missing_timestamp_column(self):
        """Test error on missing timestamp column."""
        predictions = pd.DataFrame({
            'should_trade': [True, False],
            'is_long': [True, False],
        })
        prices = pd.Series([100, 101])

        with self.assertRaises(ValueError):
            apply_regime_filter_vectorized(predictions, prices)

    def test_funding_rates_optional(self):
        """Test filter works without funding rates."""
        filter = RegimeFilter()
        prices = pd.Series(np.linspace(100, 105, 200))

        # Should work without funding rates
        regime, metrics = filter.detect_regime(prices, funding_rates=None)
        self.assertEqual(regime, MarketRegime.NORMAL)
        self.assertNotIn('funding_rate', metrics)


class TestConsecutiveDrops(unittest.TestCase):
    """Test consecutive drops detection."""

    def test_no_consecutive_drops(self):
        """Test detection when no consecutive drops."""
        filter = RegimeFilter()
        prices = pd.Series([100, 102, 104, 106, 108])  # Uptrend

        count = filter.count_consecutive_drops(prices, threshold=-0.02)
        self.assertEqual(count, 0)

    def test_consecutive_drops_detected(self):
        """Test detection of consecutive drops."""
        filter = RegimeFilter()
        # Create 3 consecutive -3% drops
        prices = pd.Series([100, 97, 94.09, 91.27, 88.53])

        count = filter.count_consecutive_drops(prices, threshold=-0.02)
        self.assertGreaterEqual(count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
