"""
main_pipeline.py - Complete ML Data Pipeline Execution
Version: 2.0
Date: December 2025

Complete data pipeline for crypto ML trading strategy.
Uses Binance API + Alternative.me (FREE tier).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import os

from data_fetcher import BinanceFetcher, FearGreedFetcher, CoinGeckoFetcher, DemoDataGenerator
from data_processor import DataProcessor
from edge_cases import EdgeCaseHandler, run_all_validations, print_validation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLDataPipeline:
    """
    Complete ML data pipeline for crypto trading strategy.
    
    Orchestrates:
    1. Data fetching from all sources
    2. Alignment and merging
    3. Edge case handling
    4. Feature engineering
    5. Validation and export
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        binance_api_key: str = None,
        output_dir: str = "./data",
        use_spot: bool = False,
        use_coingecko: bool = False,
        demo_mode: bool = False
    ):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_coingecko = use_coingecko
        self.demo_mode = demo_mode

        # Initialize components
        api_key = binance_api_key or os.environ.get('BINANCE_API_KEY')
        self.binance = BinanceFetcher(api_key=api_key, use_spot=use_spot)
        self.coingecko = CoinGeckoFetcher()
        self.demo = DemoDataGenerator()
        self.fng = FearGreedFetcher()
        self.processor = DataProcessor()
        self.edge_handler = EdgeCaseHandler()
        
        # Pipeline metadata
        self.metadata = {
            'symbol': symbol,
            'pipeline_version': '2.0',
            'created_at': None,
            'data_sources': {
                'ohlcv': 'Binance Futures API',
                'funding_rate': 'Binance Futures API',
                'fear_greed': 'Alternative.me API'
            }
        }
    
    def run(
        self,
        start_date: str,
        end_date: str = None,
        save_intermediate: bool = False
    ) -> pd.DataFrame:
        """
        Execute the complete data pipeline.
        
        Parameters:
        -----------
        start_date : str
            Start date in "YYYY-MM-DD" format
        end_date : str, optional
            End date in "YYYY-MM-DD" format (default: today)
        save_intermediate : bool
            Whether to save intermediate data files
        
        Returns:
        --------
        pd.DataFrame with all features ready for ML
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info("=" * 60)
        logger.info("Starting ML Data Pipeline")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # Step 1: Fetch all data
        logger.info("\n[1/5] Fetching data from sources...")

        if self.demo_mode:
            logger.info("Using DEMO MODE with synthetic data")
            ohlcv_df = self.demo.fetch_klines(
                symbol=self.symbol,
                interval="1h",
                start_date=start_date,
                end_date=end_date
            )
            funding_df = self.demo.fetch_funding_rate(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date
            )
            # Try to get real Fear & Greed data, fall back to empty if unavailable
            try:
                fng_df = self.fng.fetch_all()
            except Exception:
                fng_df = pd.DataFrame(columns=['timestamp', 'fear_greed_value', 'fear_greed_class'])
        elif self.use_coingecko:
            logger.info("Using CoinGecko API (Binance unavailable)")
            ohlcv_df = self.coingecko.fetch_klines(
                symbol=self.symbol,
                interval="1h",
                start_date=start_date,
                end_date=end_date
            )
            funding_df = pd.DataFrame(columns=['timestamp', 'funding_rate', 'mark_price'])
            fng_df = self.fng.fetch_all()
        else:
            try:
                ohlcv_df = self.binance.fetch_klines(
                    symbol=self.symbol,
                    interval="1h",
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                logger.warning(f"Binance API failed: {e}. Falling back to demo mode...")
                self.demo_mode = True
                ohlcv_df = self.demo.fetch_klines(
                    symbol=self.symbol,
                    interval="1h",
                    start_date=start_date,
                    end_date=end_date
                )

            if self.demo_mode:
                funding_df = self.demo.fetch_funding_rate(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                funding_df = self.binance.fetch_funding_rate(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=end_date
                )

            fng_df = self.fng.fetch_all()
        
        if save_intermediate:
            self._save_intermediate(ohlcv_df, 'ohlcv_raw.parquet')
            self._save_intermediate(funding_df, 'funding_raw.parquet')
            self._save_intermediate(fng_df, 'fng_raw.parquet')
        
        # Step 2: Handle edge cases in raw data
        logger.info("\n[2/5] Handling edge cases...")
        
        ohlcv_gaps = self.edge_handler.detect_gaps(ohlcv_df, '1h')
        funding_gaps = self.edge_handler.detect_gaps(funding_df, '8h')
        
        ohlcv_df = self.edge_handler.fill_ohlcv_gaps(ohlcv_df)
        ohlcv_df = self.edge_handler.handle_extreme_values(ohlcv_df)
        
        # Step 3: Merge all data
        logger.info("\n[3/5] Merging and aligning data...")
        
        merged_df = self.processor.merge_all_data(
            ohlcv_df=ohlcv_df,
            funding_df=funding_df,
            fng_df=fng_df,
            start_date=start_date,
            end_date=end_date
        )
        
        if save_intermediate:
            self._save_intermediate(merged_df, 'merged_raw.parquet')
        
        # Step 4: Compute derived features
        logger.info("\n[4/5] Computing derived features...")
        
        final_df = self.processor.compute_derived_features(merged_df)
        
        # Final interpolation for remaining NaNs
        final_df = self.edge_handler.interpolate_missing(final_df, method='ffill', limit=4)
        
        # Step 5: Validate and export
        logger.info("\n[5/5] Validating and exporting...")
        
        is_valid, issues = self.edge_handler.validate_data_integrity(final_df)
        
        if not is_valid:
            logger.error("Data validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("Data validation passed!")
        
        # Update metadata
        self.metadata['created_at'] = datetime.now().isoformat()
        self.metadata['date_range'] = {'start': start_date, 'end': end_date}
        self.metadata['total_rows'] = len(final_df)
        self.metadata['validation'] = {'passed': is_valid, 'issues': issues}
        
        # Save final output
        self._save_final(final_df)
        
        # Log summary
        self._log_summary(final_df)
        
        return final_df
    
    def _save_intermediate(self, df: pd.DataFrame, filename: str):
        """Save intermediate data file."""
        path = self.output_dir / 'intermediate' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug(f"Saved intermediate: {path}")
    
    def _save_final(self, df: pd.DataFrame):
        """Save final dataset and metadata."""
        # Save as parquet
        parquet_path = self.output_dir / f"{self.symbol}_ml_data.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved: {parquet_path}")
        
        # Save as CSV
        csv_path = self.output_dir / f"{self.symbol}_ml_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
        
        # Save metadata
        meta_path = self.output_dir / f"{self.symbol}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved: {meta_path}")
    
    def _log_summary(self, df: pd.DataFrame):
        """Log summary statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"\nColumn statistics:")
        
        for col in df.columns:
            if col in ['timestamp', 'session']:
                continue
            if df[col].dtype in [np.float64, np.int64, float, int]:
                nan_count = df[col].isna().sum()
                nan_pct = nan_count / len(df) * 100
                logger.info(f"  {col}: mean={df[col].mean():.4f}, "
                          f"std={df[col].std():.4f}, "
                          f"NaN={nan_count} ({nan_pct:.2f}%)")


def quick_test(use_spot: bool = False, use_coingecko: bool = False, demo_mode: bool = False):
    """Quick test with recent data."""
    print("\n" + "=" * 60)
    print("QUICK TEST - Last 7 days of data")
    print("=" * 60)

    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    pipeline = MLDataPipeline(
        symbol="BTCUSDT",
        output_dir="./test_data",
        use_spot=use_spot,
        use_coingecko=use_coingecko,
        demo_mode=demo_mode
    )

    df = pipeline.run(start_date=start_date, end_date=end_date)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")
    
    # Run validations
    results = run_all_validations(df)
    print_validation_report(results)
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ML Data Pipeline for Crypto Trading')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--start', required=False, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=False, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='./ml_data', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Run quick test with last 7 days')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate files')
    parser.add_argument('--use-spot', action='store_true',
                       help='Use Spot API instead of Futures (no funding rate data)')
    parser.add_argument('--use-coingecko', action='store_true',
                       help='Use CoinGecko API instead of Binance (fallback for geo-blocked regions)')
    parser.add_argument('--demo', action='store_true',
                       help='Use synthetic demo data (for testing when APIs are unavailable)')

    args = parser.parse_args()

    if args.test:
        quick_test(use_spot=args.use_spot, use_coingecko=args.use_coingecko, demo_mode=args.demo)
    elif args.start:
        pipeline = MLDataPipeline(
            symbol=args.symbol,
            output_dir=args.output,
            use_spot=args.use_spot,
            use_coingecko=args.use_coingecko,
            demo_mode=args.demo
        )

        df = pipeline.run(
            start_date=args.start,
            end_date=args.end,
            save_intermediate=args.save_intermediate
        )

        print(f"\nPipeline completed. Output saved to {args.output}")
        print(f"Final dataset shape: {df.shape}")
    else:
        print("Usage:")
        print("  Quick test:      python main_pipeline.py --test")
        print("  Demo mode:       python main_pipeline.py --test --demo")
        print("  Full run:        python main_pipeline.py --start 2023-01-01 --end 2024-12-31")
        print("  With options:    python main_pipeline.py --start 2023-01-01 --symbol ETHUSDT --output ./eth_data")
