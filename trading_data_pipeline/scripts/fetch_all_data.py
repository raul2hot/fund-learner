#!/usr/bin/env python3
"""
CLI entry point for the trading data pipeline.

Usage:
    # Full history (recommended for initial training)
    python fetch_all_data.py

    # Custom absolute dates
    python fetch_all_data.py --start 2021-01-01 --end 2023-12-31

    # Rolling window (last N years from today)
    python fetch_all_data.py --rolling-years 3

    # Different symbol
    python fetch_all_data.py --symbol ETHUSDT

    # Skip fetching, use cached data
    python fetch_all_data.py --skip-fetch
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_data_pipeline.config.settings import (
    TimeConfig,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SYMBOL,
)
from trading_data_pipeline.pipeline.orchestrator import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Trading Data Pipeline - Fetch, align, and engineer features"
    )

    # Date arguments
    parser.add_argument(
        '--start',
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START_DATE}"
    )
    parser.add_argument(
        '--end',
        type=str,
        default=DEFAULT_END_DATE,
        help=f"End date (YYYY-MM-DD). Default: {DEFAULT_END_DATE}"
    )
    parser.add_argument(
        '--rolling-years',
        type=int,
        default=None,
        help="Use rolling window of N years from today instead of fixed dates"
    )

    # Symbol
    parser.add_argument(
        '--symbol',
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Trading symbol. Default: {DEFAULT_SYMBOL}"
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default="data",
        help="Output directory. Default: data"
    )

    # Kaggle data
    parser.add_argument(
        '--kaggle-dir',
        type=str,
        default="data/kaggle",
        help="Directory containing Kaggle CSVs. Default: data/kaggle"
    )

    # Cache control
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help="Skip fetching, use cached data"
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )

    args = parser.parse_args()

    # Handle rolling window
    start_date = args.start
    end_date = args.end

    if args.rolling_years:
        start_date = TimeConfig.get_relative_start(args.rolling_years)
        end_date = TimeConfig.get_default_end()
        print(f"Using rolling window: last {args.rolling_years} years")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")

    # Run pipeline
    try:
        orchestrator = PipelineOrchestrator(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output_dir,
            kaggle_data_dir=args.kaggle_dir,
        )

        df = orchestrator.run(skip_fetch=args.skip_fetch)

        print("\nPipeline completed successfully!")
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"\nOutput files in: {args.output_dir}/final/")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
