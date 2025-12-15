"""
Kaggle CSV data loader for historical data.

Used for data that has limited API history (Open Interest, L/S Ratio, Taker Volume).
Kaggle datasets are updated quarterly by jesusgraterol.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


class KaggleLoader:
    """
    Load historical data from Kaggle CSV files.

    Datasets by jesusgraterol (updated quarterly):
    - bitcoin-open-interest-binance-futures
    - bitcoin-longshort-ratio-binance-futures
    - bitcoin-taker-buysell-volume-binance-futures
    - bitcoin-funding-rate-binance-futures (optional, API has full history)
    """

    # Dataset configurations
    KAGGLE_DATASETS: Dict[str, Dict[str, Any]] = {
        'open_interest': {
            'kaggle_slug': 'jesusgraterol/bitcoin-open-interest-binance-futures',
            'filename': 'open_interest.csv',
            'timestamp_col': 'ot',  # Open time
            'value_cols': ['soi', 'soiv'],  # Sum OI, Sum OI Value
            'rename_map': {
                'soi': 'sumOpenInterest',
                'soiv': 'sumOpenInterestValue',
            },
        },
        'long_short_ratio': {
            'kaggle_slug': 'jesusgraterol/bitcoin-longshort-ratio-binance-futures',
            'filename': 'long_short_ratio.csv',
            'timestamp_col': 'ot',
            'value_cols': ['lsr', 'la', 'sa'],  # LS Ratio, Long Acct, Short Acct
            'rename_map': {
                'lsr': 'longShortRatio',
                'la': 'longAccount',
                'sa': 'shortAccount',
            },
        },
        'taker_volume': {
            'kaggle_slug': 'jesusgraterol/bitcoin-taker-buysell-volume-binance-futures',
            'filename': 'taker_buy_sell_volume.csv',
            'timestamp_col': 'ot',
            'value_cols': ['bsr', 'bv', 'sv'],  # Buy/Sell Ratio, Buy Vol, Sell Vol
            'rename_map': {
                'bsr': 'buySellRatio',
                'bv': 'buyVol',
                'sv': 'sellVol',
            },
        },
        'funding_rate': {
            'kaggle_slug': 'jesusgraterol/bitcoin-funding-rate-binance-futures',
            'filename': 'funding_rate.csv',
            'timestamp_col': 'ct',  # Close time
            'value_cols': ['fr'],  # Funding Rate
            'rename_map': {
                'fr': 'fundingRate',
            },
        },
    }

    def __init__(self, kaggle_data_dir: str = "data/kaggle"):
        """
        Initialize KaggleLoader.

        Args:
            kaggle_data_dir: Directory containing downloaded Kaggle CSVs
        """
        self.kaggle_dir = Path(kaggle_data_dir)

    def load(self, data_type: str) -> pd.DataFrame:
        """
        Load and standardize Kaggle CSV data.

        Args:
            data_type: One of 'open_interest', 'long_short_ratio', 'taker_volume', 'funding_rate'

        Returns:
            DataFrame with standardized columns and UTC timestamps
        """
        if data_type not in self.KAGGLE_DATASETS:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available: {list(self.KAGGLE_DATASETS.keys())}"
            )

        config = self.KAGGLE_DATASETS[data_type]
        filepath = self.kaggle_dir / config['filename']

        if not filepath.exists():
            raise FileNotFoundError(
                f"Kaggle data not found: {filepath}\n"
                f"Download from: kaggle.com/datasets/{config['kaggle_slug']}\n"
                f"Or run: kaggle datasets download -d {config['kaggle_slug']} -p {self.kaggle_dir}/ --unzip"
            )

        print(f"Loading Kaggle data from {filepath}...")

        df = pd.read_csv(filepath)

        # Standardize timestamp
        ts_col = config['timestamp_col']
        df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)

        # Rename columns
        df = df.rename(columns=config['rename_map'])

        # Convert value columns to numeric
        for col in config['rename_map'].values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add data source marker
        df['data_source'] = 'kaggle'

        # Keep only relevant columns
        keep_cols = ['timestamp', 'data_source'] + list(config['rename_map'].values())
        df = df[[c for c in keep_cols if c in df.columns]]

        # Sort and deduplicate
        df = df.sort_values('timestamp').drop_duplicates('timestamp')

        print(f"Loaded {len(df)} records from Kaggle ({df['timestamp'].min()} to {df['timestamp'].max()})")

        return df.reset_index(drop=True)

    def get_data_end_date(self, data_type: str) -> Optional[pd.Timestamp]:
        """
        Get the end date of Kaggle data for gap detection.

        Args:
            data_type: Type of data

        Returns:
            Last timestamp in the Kaggle data, or None if not loaded
        """
        try:
            df = self.load(data_type)
            return df['timestamp'].max()
        except FileNotFoundError:
            return None

    def check_availability(self) -> Dict[str, Dict[str, Any]]:
        """
        Check which Kaggle datasets are available.

        Returns:
            Dict with availability status for each dataset
        """
        status = {}

        for data_type, config in self.KAGGLE_DATASETS.items():
            filepath = self.kaggle_dir / config['filename']
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, nrows=1)
                    # Get full file to check date range
                    df_full = self.load(data_type)
                    status[data_type] = {
                        'available': True,
                        'path': str(filepath),
                        'rows': len(df_full),
                        'start': str(df_full['timestamp'].min()),
                        'end': str(df_full['timestamp'].max()),
                    }
                except Exception as e:
                    status[data_type] = {
                        'available': False,
                        'error': str(e),
                    }
            else:
                status[data_type] = {
                    'available': False,
                    'path': str(filepath),
                    'download': f"kaggle datasets download -d {config['kaggle_slug']} -p {self.kaggle_dir}/ --unzip",
                }

        return status

    @staticmethod
    def print_download_instructions():
        """Print instructions for downloading Kaggle datasets."""
        print("""
=== Kaggle Dataset Download Instructions ===

1. Install Kaggle CLI:
   pip install kaggle

2. Configure Kaggle API:
   - Go to kaggle.com/account
   - Click "Create New Token"
   - Save kaggle.json to ~/.kaggle/kaggle.json
   - Run: chmod 600 ~/.kaggle/kaggle.json

3. Download datasets:
   mkdir -p data/kaggle
   kaggle datasets download -d jesusgraterol/bitcoin-open-interest-binance-futures -p data/kaggle/ --unzip
   kaggle datasets download -d jesusgraterol/bitcoin-longshort-ratio-binance-futures -p data/kaggle/ --unzip
   kaggle datasets download -d jesusgraterol/bitcoin-taker-buysell-volume-binance-futures -p data/kaggle/ --unzip

Note: These datasets are updated quarterly.
For gap filling, see: github.com/jesusgraterol/binance-futures-dataset-builder
""")
