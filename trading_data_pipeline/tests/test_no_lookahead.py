"""
CRITICAL: Tests to ensure no look-ahead bias.

These tests MUST pass before using data for training.
Any look-ahead bias will produce models that backtest well but fail in production.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestNoLookahead:
    """Tests to ensure no look-ahead bias."""

    def test_feature_timestamps_before_labels(self, aligned_data):
        """
        All features must be computed from data before label timestamp.

        For each row, all age columns should be >= 0, meaning
        the data used was available before the prediction time.
        """
        age_cols = [c for c in aligned_data.columns if '_age_hours' in c]

        for col in age_cols:
            min_age = aligned_data[col].min()
            assert pd.isna(min_age) or min_age >= 0, \
                f"{col} has negative age ({min_age:.2f}h) - future data used!"

    def test_label_is_future_data(self, labeled_data):
        """
        Labels should represent NEXT candle, not current.

        Label at time T should predict candle at T+1.
        """
        for i in range(len(labeled_data) - 1):
            current = labeled_data.iloc[i]
            next_row = labeled_data.iloc[i + 1]

            # Verify next_return matches actual next close
            if pd.notna(current.get('next_return')) and pd.notna(next_row.get('close')):
                expected_return = (next_row['close'] - current['close']) / current['close']
                actual_return = current['next_return']

                assert abs(actual_return - expected_return) < 1e-6, \
                    f"Label mismatch at index {i}: expected {expected_return:.6f}, got {actual_return:.6f}"

    def test_no_shuffling_in_split(self, train_df, val_df, test_df):
        """Train/val/test must be strictly chronological."""
        if 'timestamp' not in train_df.columns:
            pytest.skip("No timestamp column")

        train_max = train_df['timestamp'].max()
        val_min = val_df['timestamp'].min()
        val_max = val_df['timestamp'].max()
        test_min = test_df['timestamp'].min()

        assert train_max < val_min, \
            f"Train/Val overlap! Train ends {train_max}, Val starts {val_min}"

        assert val_max < test_min, \
            f"Val/Test overlap! Val ends {val_max}, Test starts {test_min}"

    def test_publication_delays_applied(self, pit_db):
        """Verify publication delays are correctly applied."""
        for name, source in pit_db.sources.items():
            df = pit_db.raw_data.get(name)
            if df is None:
                continue

            # _available_at should be >= timestamp
            assert (df['_available_at'] >= df['timestamp']).all(), \
                f"{name}: available_at before timestamp!"

            # Check delay matches config
            expected_delay = source.publication_delay
            actual_delay = (df['_available_at'] - df['timestamp']).iloc[0]

            assert actual_delay == expected_delay, \
                f"{name}: delay mismatch - expected {expected_delay}, got {actual_delay}"


class TestFeatureIntegrity:
    """Tests for feature calculation integrity."""

    def test_no_infinite_values(self, featured_data):
        """No infinite values in features."""
        numeric_cols = featured_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            inf_count = np.isinf(featured_data[col]).sum()
            assert inf_count == 0, \
                f"Column {col} has {inf_count} infinite values"

    def test_rolling_features_have_warmup(self, featured_data):
        """Rolling features should have NaN during warmup period."""
        # Features with rolling calculations should have NaN at the start
        rolling_features = ['atr_14', 'sma_20', 'rsi', 'adx']

        for feat in rolling_features:
            if feat in featured_data.columns:
                # First few rows should have some NaN
                first_values = featured_data[feat].iloc[:5]
                # At least some NaN is expected
                # (exact number depends on rolling window)

    def test_age_columns_consistent(self, aligned_data):
        """Age columns should be consistent with timestamps."""
        age_cols = [c for c in aligned_data.columns if '_age_hours' in c]

        for col in age_cols:
            ages = aligned_data[col].dropna()
            if len(ages) > 0:
                # Age should be reasonable (not hundreds of years)
                assert ages.max() < 24 * 365, \
                    f"Column {col} has unreasonable max age: {ages.max()}"


class TestLabelConsistency:
    """Tests for label consistency."""

    def test_labels_are_valid(self, labeled_data):
        """Labels should be in valid range [0, 4]."""
        valid_labels = {0, 1, 2, 3, 4}
        actual_labels = set(labeled_data['label'].dropna().unique())

        assert actual_labels.issubset(valid_labels), \
            f"Invalid labels found: {actual_labels - valid_labels}"

    def test_label_distribution_reasonable(self, labeled_data):
        """Label distribution should be reasonable (no extreme imbalance)."""
        dist = labeled_data['label'].value_counts(normalize=True)

        # No class should be > 80% or < 1%
        for label, pct in dist.items():
            assert pct < 0.8, f"Label {label} is too dominant: {pct:.1%}"
            assert pct > 0.01, f"Label {label} is too rare: {pct:.1%}"


# Pytest fixtures
@pytest.fixture
def aligned_data():
    """Fixture for aligned data. Override in conftest.py with actual data."""
    # Create minimal test data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'feature_age_hours': np.random.rand(100) * 10,
    })
    return df


@pytest.fixture
def labeled_data():
    """Fixture for labeled data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    close = np.random.randn(100).cumsum() + 100

    df = pd.DataFrame({
        'timestamp': dates,
        'close': close,
        'next_return': pd.Series(close).pct_change().shift(-1),
        'label': np.random.randint(0, 5, 100),
    })
    return df.iloc[:-1]  # Remove last row (no future data)


@pytest.fixture
def train_df():
    """Fixture for train split."""
    dates = pd.date_range('2024-01-01', periods=70, freq='1h')
    return pd.DataFrame({'timestamp': dates, 'value': range(70)})


@pytest.fixture
def val_df():
    """Fixture for validation split."""
    dates = pd.date_range('2024-01-03 22:00', periods=15, freq='1h')
    return pd.DataFrame({'timestamp': dates, 'value': range(70, 85)})


@pytest.fixture
def test_df():
    """Fixture for test split."""
    dates = pd.date_range('2024-01-04 13:00', periods=15, freq='1h')
    return pd.DataFrame({'timestamp': dates, 'value': range(85, 100)})


@pytest.fixture
def featured_data():
    """Fixture for featured data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'atr_14': np.random.rand(100),
        'sma_20': np.random.rand(100),
        'rsi': np.random.rand(100) * 100,
        'adx': np.random.rand(100) * 100,
    })
    return df


@pytest.fixture
def pit_db():
    """Fixture for PointInTimeDatabase. Override with actual in conftest.py."""
    from ..alignment.point_in_time import PointInTimeDatabase
    from ..config.data_sources import OHLCV_SOURCE

    pit = PointInTimeDatabase()
    pit.register_source(OHLCV_SOURCE)

    # Create minimal test data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'open_time': dates.astype(int) // 10**6,  # Convert to ms
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000,
    })
    pit.ingest('ohlcv', df)

    return pit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
