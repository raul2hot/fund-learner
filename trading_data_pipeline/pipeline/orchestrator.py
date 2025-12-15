"""
Main pipeline orchestrator.

Coordinates all data pipeline components:
1. Data fetching
2. Point-in-time alignment
3. Feature engineering
4. Labeling
5. Train/val/test splitting
6. Export to Parquet
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from ..config.settings import (
    TimeConfig,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SYMBOL,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from ..config.data_sources import (
    OHLCV_SOURCE,
    FUNDING_RATE_SOURCE,
    FEAR_GREED_SOURCE,
    OPEN_INTEREST_SOURCE,
    LONG_SHORT_RATIO_SOURCE,
    TAKER_VOLUME_SOURCE,
)
from ..fetchers import (
    BinanceOHLCVFetcher,
    BinanceFundingFetcher,
    FearGreedFetcher,
    HybridDataFetcher,
)
from ..alignment import PointInTimeDatabase, AlignmentValidator
from ..features import FeatureEngineer
from ..labeling import CandleLabeler
from .splitter import DataSplitter


class PipelineOrchestrator:
    """
    Main pipeline orchestrator.

    Runs the complete data pipeline from fetching to final dataset export.
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        output_dir: str = "data",
        kaggle_data_dir: str = "data/kaggle",
    ):
        """
        Initialize PipelineOrchestrator.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            start_date: Start date (default: 2019-09-15)
            end_date: End date (default: today)
            output_dir: Directory for output files
            kaggle_data_dir: Directory containing Kaggle CSVs
        """
        # Validate dates
        self.start_date, self.end_date = TimeConfig.validate_dates(start_date, end_date)
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.kaggle_data_dir = kaggle_data_dir

        # Create output directories
        for subdir in ['raw', 'aligned', 'featured', 'final', 'reports']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pit_db = PointInTimeDatabase(base_frequency='1h')

        # Metadata to track
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.utcnow().isoformat(),
            'symbol': self.symbol,
            'date_range': {
                'start': self.start_date,
                'end': self.end_date,
            },
            'sources': {},
        }

    def run(self, skip_fetch: bool = False) -> pd.DataFrame:
        """
        Run the complete pipeline.

        Args:
            skip_fetch: If True, skip fetching and load from cache

        Returns:
            Final featured and labeled DataFrame
        """
        print("\n" + "=" * 70)
        print("TRADING DATA PIPELINE")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print("=" * 70 + "\n")

        # Step 1: Fetch data
        if not skip_fetch:
            self._fetch_all_data()
        else:
            self._load_cached_data()

        # Step 2: Build aligned dataset
        aligned_df = self._build_aligned_dataset()

        # Step 3: Engineer features
        featured_df = self._engineer_features(aligned_df)

        # Step 4: Generate labels
        labeled_df = self._generate_labels(featured_df)

        # Step 5: Validate
        self._validate_dataset(labeled_df)

        # Step 6: Split data
        train_df, val_df, test_df = self._split_data(labeled_df)

        # Step 7: Export
        self._export_datasets(train_df, val_df, test_df, labeled_df)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70 + "\n")

        return labeled_df

    def _fetch_all_data(self):
        """Fetch all data sources."""
        print("\n--- STEP 1: FETCHING DATA ---\n")

        # 1. OHLCV
        print("Fetching OHLCV data...")
        ohlcv_fetcher = BinanceOHLCVFetcher()
        ohlcv_df = ohlcv_fetcher.fetch(self.start_date, self.end_date, self.symbol)
        self._save_raw(ohlcv_df, 'ohlcv')
        self.pit_db.register_source(OHLCV_SOURCE)
        self.pit_db.ingest('ohlcv', ohlcv_df)
        self.metadata['sources']['ohlcv'] = {'rows': len(ohlcv_df)}

        # 2. Funding Rate
        print("\nFetching Funding Rate data...")
        funding_fetcher = BinanceFundingFetcher()
        funding_df = funding_fetcher.fetch(self.start_date, self.end_date, self.symbol)
        self._save_raw(funding_df, 'funding_rate')
        self.pit_db.register_source(FUNDING_RATE_SOURCE)
        self.pit_db.ingest('funding_rate', funding_df)
        self.metadata['sources']['funding_rate'] = {'rows': len(funding_df)}

        # 3. Fear & Greed
        print("\nFetching Fear & Greed data...")
        fg_fetcher = FearGreedFetcher()
        fg_df = fg_fetcher.fetch(self.start_date, self.end_date)
        self._save_raw(fg_df, 'fear_greed')
        self.pit_db.register_source(FEAR_GREED_SOURCE)
        self.pit_db.ingest('fear_greed', fg_df)
        self.metadata['sources']['fear_greed'] = {'rows': len(fg_df)}

        # 4. Hybrid data (Open Interest, L/S Ratio, Taker Volume)
        print("\nFetching hybrid data sources...")
        hybrid_fetcher = HybridDataFetcher(kaggle_data_dir=self.kaggle_data_dir)

        # Open Interest
        try:
            oi_df = hybrid_fetcher.fetch('open_interest', self.start_date, self.end_date, self.symbol)
            self._save_raw(oi_df, 'open_interest')
            self.pit_db.register_source(OPEN_INTEREST_SOURCE)
            self.pit_db.ingest('open_interest', oi_df)
            self.metadata['sources']['open_interest'] = {'rows': len(oi_df)}
        except Exception as e:
            print(f"WARNING: Could not fetch open interest: {e}")

        # Long/Short Ratio
        try:
            ls_df = hybrid_fetcher.fetch('long_short_ratio', self.start_date, self.end_date, self.symbol)
            self._save_raw(ls_df, 'long_short_ratio')
            self.pit_db.register_source(LONG_SHORT_RATIO_SOURCE)
            self.pit_db.ingest('long_short_ratio', ls_df)
            self.metadata['sources']['long_short_ratio'] = {'rows': len(ls_df)}
        except Exception as e:
            print(f"WARNING: Could not fetch long/short ratio: {e}")

        # Taker Volume
        try:
            tv_df = hybrid_fetcher.fetch('taker_volume', self.start_date, self.end_date, self.symbol)
            self._save_raw(tv_df, 'taker_volume')
            self.pit_db.register_source(TAKER_VOLUME_SOURCE)
            self.pit_db.ingest('taker_volume', tv_df)
            self.metadata['sources']['taker_volume'] = {'rows': len(tv_df)}
        except Exception as e:
            print(f"WARNING: Could not fetch taker volume: {e}")

    def _load_cached_data(self):
        """Load data from cached Parquet files."""
        print("\n--- LOADING CACHED DATA ---\n")

        raw_dir = self.output_dir / 'raw'

        # Load OHLCV
        ohlcv_path = raw_dir / f'ohlcv_{self.symbol}_1h.parquet'
        if ohlcv_path.exists():
            ohlcv_df = pd.read_parquet(ohlcv_path)
            self.pit_db.register_source(OHLCV_SOURCE)
            self.pit_db.ingest('ohlcv', ohlcv_df)

        # Load other sources similarly...
        for name, source in [
            ('funding_rate', FUNDING_RATE_SOURCE),
            ('fear_greed', FEAR_GREED_SOURCE),
            ('open_interest', OPEN_INTEREST_SOURCE),
            ('long_short_ratio', LONG_SHORT_RATIO_SOURCE),
            ('taker_volume', TAKER_VOLUME_SOURCE),
        ]:
            path = raw_dir / f'{name}_{self.symbol}_1h.parquet'
            if not path.exists():
                path = raw_dir / f'{name}.parquet'
            if path.exists():
                df = pd.read_parquet(path)
                self.pit_db.register_source(source)
                self.pit_db.ingest(name, df)

    def _save_raw(self, df: pd.DataFrame, name: str):
        """Save raw data to Parquet."""
        filename = f'{name}_{self.symbol}_1h.parquet'
        if name in ['fear_greed']:
            filename = f'{name}.parquet'

        path = self.output_dir / 'raw' / filename
        df.to_parquet(path, index=False)
        print(f"  Saved: {path}")

    def _build_aligned_dataset(self) -> pd.DataFrame:
        """Build point-in-time aligned dataset."""
        print("\n--- STEP 2: BUILDING ALIGNED DATASET ---\n")

        aligned_df = self.pit_db.build_aligned_dataset(self.start_date, self.end_date)

        # Save aligned data
        path = self.output_dir / 'aligned' / f'aligned_{self.symbol}_1h.parquet'
        aligned_df.to_parquet(path, index=False)
        print(f"Saved aligned data: {path}")

        return aligned_df

    def _engineer_features(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features."""
        print("\n--- STEP 3: ENGINEERING FEATURES ---\n")

        engineer = FeatureEngineer(aligned_df)
        featured_df = engineer.compute_all_features()

        # Save featured data
        path = self.output_dir / 'featured' / f'featured_{self.symbol}_1h.parquet'
        featured_df.to_parquet(path, index=False)
        print(f"Saved featured data: {path}")

        self.metadata['features'] = {
            'total_columns': len(featured_df.columns),
        }

        return featured_df

    def _generate_labels(self, featured_df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels for classification."""
        print("\n--- STEP 4: GENERATING LABELS ---\n")

        labeler = CandleLabeler()
        labeled_df = labeler.generate_labels(featured_df)

        # Print distribution
        labeler.print_distribution(labeled_df)

        # Store in metadata
        dist = labeler.get_label_distribution(labeled_df)
        self.metadata['labels'] = {
            'distribution': dist.to_dict('records')
        }

        return labeled_df

    def _validate_dataset(self, df: pd.DataFrame):
        """Validate the final dataset."""
        print("\n--- STEP 5: VALIDATING DATASET ---\n")

        validator = AlignmentValidator(df)
        passed = validator.print_report()

        if not passed:
            raise ValueError("Dataset validation FAILED. Fix issues before proceeding.")

    def _split_data(self, df: pd.DataFrame):
        """Split data into train/val/test."""
        print("\n--- STEP 6: SPLITTING DATA ---\n")

        splitter = DataSplitter(
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO
        )

        train_df, val_df, test_df = splitter.split(df)

        return train_df, val_df, test_df

    def _export_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame
    ):
        """Export final datasets."""
        print("\n--- STEP 7: EXPORTING DATASETS ---\n")

        final_dir = self.output_dir / 'final'

        # Export splits
        train_df.to_parquet(final_dir / f'train_{self.symbol}_1h.parquet', index=False)
        val_df.to_parquet(final_dir / f'val_{self.symbol}_1h.parquet', index=False)
        test_df.to_parquet(final_dir / f'test_{self.symbol}_1h.parquet', index=False)

        print(f"Exported train: {len(train_df)} rows")
        print(f"Exported val: {len(val_df)} rows")
        print(f"Exported test: {len(test_df)} rows")

        # Update metadata
        def get_date_range(df):
            if 'timestamp' in df.columns:
                return str(df['timestamp'].min()), str(df['timestamp'].max())
            return None, None

        self.metadata['splits'] = {
            'train': {
                'rows': len(train_df),
                'start': get_date_range(train_df)[0],
                'end': get_date_range(train_df)[1],
            },
            'val': {
                'rows': len(val_df),
                'start': get_date_range(val_df)[0],
                'end': get_date_range(val_df)[1],
            },
            'test': {
                'rows': len(test_df),
                'start': get_date_range(test_df)[0],
                'end': get_date_range(test_df)[1],
            },
        }

        # Save metadata
        metadata_path = final_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"\nSaved metadata: {metadata_path}")
