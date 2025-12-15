"""
CRITICAL: Point-In-Time Database for preventing look-ahead bias.

Key principle: For any timestamp T, we can only use data that was
ACTUALLY AVAILABLE at time T, not data timestamped at T.

Example:
- Candle closes at 14:00:00
- We receive candle data at 14:00:05
- Therefore, at 14:00:00, we could NOT have known the 14:00 candle!
- For 14:00 prediction, we can only use data available_at <= 14:00
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings

from ..config.data_sources import DataSource, ResampleMethod


class PointInTimeDatabase:
    """
    Stores all data with availability timestamps.
    Ensures no look-ahead bias in feature construction.

    This is the MOST CRITICAL component of the data pipeline.
    Any bug here will result in models that backtest well but fail in production.
    """

    def __init__(self, base_frequency: str = '1h'):
        """
        Initialize PointInTimeDatabase.

        Args:
            base_frequency: Base frequency for alignment (default: 1h)
        """
        self.base_frequency = base_frequency
        self.sources: Dict[str, DataSource] = {}
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.aligned_data: Optional[pd.DataFrame] = None

    def register_source(self, config: DataSource):
        """
        Register a data source configuration.

        Args:
            config: DataSource configuration object
        """
        self.sources[config.name] = config
        print(f"Registered source: {config.name} "
              f"(freq={config.frequency}, delay={config.publication_delay})")

    def ingest(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ingest raw data with proper timestamp handling.
        Adds _available_at column for point-in-time queries.

        Args:
            name: Name of the data source (must be registered first)
            df: Raw DataFrame from fetcher

        Returns:
            Processed DataFrame with _available_at column
        """
        if name not in self.sources:
            raise ValueError(f"Unknown source: {name}. Register first with register_source().")

        config = self.sources[name]
        df = df.copy()

        # 1. Standardize timestamp
        df = self._standardize_timestamp(df, config)

        # 2. Add availability timestamp (CRITICAL!)
        # This is when the data actually becomes available for use
        df['_available_at'] = df['timestamp'] + config.publication_delay

        # 3. Sort and validate
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        self._validate_timestamps(df, name)

        # 4. Store
        self.raw_data[name] = df
        print(f"Ingested {name}: {len(df)} rows, "
              f"{df['timestamp'].min()} -> {df['timestamp'].max()}")

        return df

    def _standardize_timestamp(
        self,
        df: pd.DataFrame,
        config: DataSource
    ) -> pd.DataFrame:
        """Convert all timestamps to UTC pandas Timestamp."""
        ts_col = config.timestamp_col

        # Check if timestamp column exists or if 'timestamp' is already present
        if 'timestamp' in df.columns:
            # Already standardized
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            elif df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            return df

        if ts_col not in df.columns:
            raise ValueError(f"Timestamp column '{ts_col}' not found in DataFrame. "
                           f"Available columns: {list(df.columns)}")

        if config.timestamp_format == 'unix_ms':
            df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
        elif config.timestamp_format == 'unix_s':
            df['timestamp'] = pd.to_datetime(df[ts_col], unit='s', utc=True)
        elif config.timestamp_format == 'iso':
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
        elif config.timestamp_format == 'date':
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
        else:
            raise ValueError(f"Unknown timestamp format: {config.timestamp_format}")

        return df

    def _validate_timestamps(self, df: pd.DataFrame, name: str):
        """Validate timestamp quality."""
        nat_count = df['timestamp'].isna().sum()
        if nat_count > 0:
            raise ValueError(f"{name}: {nat_count} NaT timestamps found!")

        if not df['timestamp'].is_monotonic_increasing:
            dup_count = df['timestamp'].duplicated().sum()
            if dup_count > 0:
                warnings.warn(f"{name}: {dup_count} duplicate timestamps were removed")

    def get_point_in_time(
        self,
        as_of: pd.Timestamp,
        source: str
    ) -> Optional[pd.Series]:
        """
        Get the most recent data available at a specific time.

        THIS IS THE KEY FUNCTION FOR PREVENTING LOOK-AHEAD BIAS.

        Args:
            as_of: The timestamp we're querying from
            source: Name of the data source

        Returns:
            Most recent available data row, or None if no data available
        """
        if source not in self.raw_data:
            return None

        df = self.raw_data[source]

        # Only data that was AVAILABLE at as_of time
        available = df[df['_available_at'] <= as_of]

        if len(available) == 0:
            return None

        return available.iloc[-1]

    def build_aligned_dataset(
        self,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Build aligned dataset at base frequency.
        All features are point-in-time correct.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame aligned to base frequency with all sources merged
        """
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')

        # Create base timeline from OHLCV
        if 'ohlcv' not in self.raw_data:
            raise ValueError("OHLCV data required as base timeline. Ingest OHLCV first.")

        base_df = self.raw_data['ohlcv'].copy()
        base_df = base_df[
            (base_df['timestamp'] >= start_ts) &
            (base_df['timestamp'] <= end_ts)
        ]

        if len(base_df) == 0:
            raise ValueError(f"No OHLCV data found between {start} and {end}")

        # Initialize aligned dataframe
        aligned = pd.DataFrame(index=base_df['timestamp'])
        aligned.index.name = 'timestamp'

        # Add OHLCV columns directly (same frequency)
        ohlcv_config = self.sources.get('ohlcv')
        if ohlcv_config and ohlcv_config.value_columns:
            for col in ohlcv_config.value_columns:
                if col in base_df.columns:
                    aligned[col] = base_df.set_index('timestamp')[col]

        print(f"\nBuilding aligned dataset from {start} to {end}")
        print(f"Base timeline: {len(aligned)} rows")

        # Align other sources
        for name, config in self.sources.items():
            if name == 'ohlcv':
                continue
            print(f"Aligning {name}...")
            aligned = self._align_source(aligned, name, config)

        # Final validation
        self._validate_no_lookahead(aligned)

        self.aligned_data = aligned.reset_index()
        return self.aligned_data

    def _align_source(
        self,
        aligned: pd.DataFrame,
        name: str,
        config: DataSource
    ) -> pd.DataFrame:
        """
        Align a source to base timeline using point-in-time logic.

        Uses merge_asof for efficient point-in-time merging.
        """
        if name not in self.raw_data:
            warnings.warn(f"Source {name} not ingested, marking as missing")
            for col in config.value_columns or []:
                aligned[f"{name}_{col}"] = np.nan
                aligned[f"{name}_{col}_missing"] = 1
                aligned[f"{name}_{col}_age_hours"] = np.nan
            return aligned

        source_df = self.raw_data[name].copy()
        value_cols = config.value_columns or []

        # Sort by availability time for merge_asof
        source_df = source_df.sort_values('_available_at')

        # Prepare aligned dataframe for merge
        aligned_reset = aligned.reset_index()

        for col in value_cols:
            if col not in source_df.columns:
                warnings.warn(f"Column {col} not found in {name} data")
                aligned[f"{name}_{col}"] = np.nan
                aligned[f"{name}_{col}_missing"] = 1
                aligned[f"{name}_{col}_age_hours"] = np.nan
                continue

            # Prepare source data for merge
            source_for_merge = source_df[['_available_at', 'timestamp', col]].copy()
            source_for_merge = source_for_merge.rename(columns={
                'timestamp': '_source_ts',
                '_available_at': '_avail_ts'
            })

            # Merge based on availability time (not data timestamp!)
            merged = pd.merge_asof(
                aligned_reset[['timestamp']].sort_values('timestamp'),
                source_for_merge.sort_values('_avail_ts'),
                left_on='timestamp',
                right_on='_avail_ts',
                direction='backward'
            )

            # Calculate age (hours since source timestamp)
            age_hours = (merged['timestamp'] - merged['_source_ts']).dt.total_seconds() / 3600

            # Apply fill limit if specified
            if config.fill_limit_hours:
                mask = age_hours > config.fill_limit_hours
                merged.loc[mask, col] = np.nan
                age_hours.loc[mask] = np.nan

            # Store in aligned dataframe
            aligned[f"{name}_{col}"] = merged[col].values
            aligned[f"{name}_{col}_age_hours"] = age_hours.values

        # Add missingness indicators
        for col in value_cols:
            aligned[f"{name}_{col}_missing"] = aligned[f"{name}_{col}"].isna().astype(int)

        return aligned

    def _validate_no_lookahead(self, aligned: pd.DataFrame):
        """
        CRITICAL: Validate no look-ahead bias exists.

        Checks that all age columns are non-negative (data comes from past).
        """
        print("\n" + "=" * 50)
        print("LOOK-AHEAD BIAS VALIDATION")
        print("=" * 50)

        issues = []

        # Check age columns are all non-negative
        age_cols = [c for c in aligned.columns if '_age_hours' in c]
        for col in age_cols:
            if col in aligned.columns:
                min_age = aligned[col].min()
                if pd.notna(min_age) and min_age < 0:
                    issues.append(f"FAIL: {col}: negative age ({min_age:.2f}h) = FUTURE DATA USED!")

        if issues:
            for issue in issues:
                print(issue)
            raise ValueError("LOOK-AHEAD BIAS DETECTED! Fix before proceeding.")

        print("PASS: No look-ahead bias detected")
        print("=" * 50 + "\n")

    def get_coverage_report(self) -> Dict[str, Dict]:
        """
        Generate a coverage report for all ingested data sources.

        Returns:
            Dict with coverage statistics for each source
        """
        report = {}

        for name, df in self.raw_data.items():
            config = self.sources.get(name)
            report[name] = {
                'rows': len(df),
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max()),
                'frequency': config.frequency if config else 'unknown',
                'missing_pct': df.isna().mean().mean() * 100 if len(df) > 0 else 100,
            }

        return report
