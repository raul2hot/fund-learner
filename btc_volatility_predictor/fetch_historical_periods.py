#!/usr/bin/env python3
"""
Fetch historical BTC/USDT data for multiple market cycles.

Periods to fetch:
1. 2018 Bear: Jan 2018 - Dec 2018 (366 days)
2. 2019-2020 Recovery: Jan 2019 - Dec 2020 (731 days)
3. 2023-2024 Bull: Jan 2023 - Nov 2024 (700 days)

This data is used for robustness testing of TrendStrengthML_50 strategy
across different market conditions (bear vs bull markets).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

PERIODS = {
    '2018_bear': {
        'start': '2018-01-01',
        'end': '2018-12-31',
        'type': 'bear',
        'description': '2018 Crypto Winter (-84% drawdown)'
    },
    '2019_2020_recovery': {
        'start': '2019-01-01',
        'end': '2020-12-31',
        'type': 'bull',
        'description': '2019-2020 Recovery (includes COVID crash and bounce)'
    },
    '2023_2024_bull': {
        'start': '2023-01-01',
        'end': '2024-10-31',
        'type': 'bull',
        'description': '2023-2024 Bull Market (ETF Rally, +376%)'
    }
}


def fetch_binance_klines(start_date: str, end_date: str,
                         symbol: str = "BTCUSDT", interval: str = "1h",
                         show_progress: bool = True) -> pd.DataFrame:
    """
    Fetch klines from Binance API with pagination.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbol: Trading pair symbol (default: BTCUSDT)
        interval: Kline interval (default: 1h)
        show_progress: Whether to print progress

    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts += 24 * 60 * 60 * 1000 - 1  # Include the end date fully

    all_klines = []
    current_start = start_ts
    request_count = 0

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }

        # Retry logic with exponential backoff
        max_retries = 4
        retry_delays = [2, 4, 8, 16]

        for attempt in range(max_retries + 1):
            try:
                response = requests.get(base_url, params=params, timeout=30)

                if response.status_code == 200:
                    break
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries:
                        delay = retry_delays[attempt]
                        if show_progress:
                            print(f"    Rate limited, waiting {delay}s...")
                        time.sleep(delay)
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries")
                else:
                    raise Exception(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    if show_progress:
                        print(f"    Network error, retrying in {delay}s... ({e})")
                    time.sleep(delay)
                else:
                    raise Exception(f"Network error after {max_retries} retries: {e}")

        data = response.json()
        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1  # Move past last candle
        request_count += 1

        if show_progress and request_count % 10 == 0:
            progress_date = datetime.fromtimestamp(data[-1][0] / 1000)
            print(f"    Fetched up to {progress_date.strftime('%Y-%m-%d')} ({len(all_klines)} candles)")

        time.sleep(0.1)  # Rate limiting - be nice to the API

    # Convert to DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
               'taker_buy_quote', 'ignore']

    df = pd.DataFrame(all_klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)

    df['trades'] = df['trades'].astype(int)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'trades', 'quote_volume']]


def fetch_period(period_name: str, config: dict, output_dir: str = "data/raw/historical") -> str:
    """
    Fetch data for a single period.

    Args:
        period_name: Name of the period
        config: Period configuration dict with start, end, type
        output_dir: Directory to save output

    Returns:
        Path to the saved file
    """
    print(f"\nFetching {period_name}...")
    print(f"  Period: {config['start']} to {config['end']}")
    print(f"  Type: {config['type'].upper()} market")
    if 'description' in config:
        print(f"  Description: {config['description']}")

    df = fetch_binance_klines(config['start'], config['end'])

    output_path = os.path.join(output_dir, f"btcusdt_1h_{period_name}.csv")
    df.to_csv(output_path, index=False)

    # Print summary statistics
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"  Saved {len(df)} candles ({days} days) to {output_path}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Price statistics
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return output_path


def fetch_all_periods(output_dir: str = "data/raw/historical",
                      periods: dict = None) -> dict:
    """
    Fetch all required historical periods.

    Args:
        output_dir: Directory to save output files
        periods: Optional dict of periods to fetch (uses PERIODS if not specified)

    Returns:
        Dict mapping period names to output paths
    """
    if periods is None:
        periods = PERIODS

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HISTORICAL DATA FETCHER")
    print("=" * 60)
    print(f"Periods to fetch: {len(periods)}")
    print(f"Output directory: {output_dir}")

    results = {}

    for period_name, config in periods.items():
        try:
            output_path = fetch_period(period_name, config, output_dir)
            results[period_name] = {
                'path': output_path,
                'status': 'success'
            }
        except Exception as e:
            print(f"  ERROR: Failed to fetch {period_name}: {e}")
            results[period_name] = {
                'status': 'error',
                'error': str(e)
            }

    # Summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"Successfully fetched: {successful}/{len(periods)} periods")

    for period_name, result in results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status} {period_name}")

    return results


def validate_data(filepath: str) -> dict:
    """
    Validate fetched data for quality issues.

    Args:
        filepath: Path to CSV file

    Returns:
        Dict with validation results
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    results = {
        'filepath': filepath,
        'rows': len(df),
        'issues': []
    }

    # Check for missing timestamps (gaps in hourly data)
    df = df.sort_values('timestamp')
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow small tolerance

    if len(gaps) > 0:
        results['issues'].append(f"Found {len(gaps)} gaps in data")
        results['gaps'] = len(gaps)

    # Check for zero/null values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        null_count = df[col].isnull().sum()
        zero_count = (df[col] == 0).sum()

        if null_count > 0:
            results['issues'].append(f"{col} has {null_count} null values")
        if zero_count > 0 and col != 'volume':  # Volume can be zero
            results['issues'].append(f"{col} has {zero_count} zero values")

    # Check OHLC validity
    invalid_ohlc = df[(df['high'] < df['low']) |
                      (df['high'] < df['open']) |
                      (df['high'] < df['close']) |
                      (df['low'] > df['open']) |
                      (df['low'] > df['close'])]

    if len(invalid_ohlc) > 0:
        results['issues'].append(f"Found {len(invalid_ohlc)} invalid OHLC rows")

    results['valid'] = len(results['issues']) == 0

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical BTC/USDT data")
    parser.add_argument('--period', type=str, choices=list(PERIODS.keys()),
                        help='Specific period to fetch (default: all)')
    parser.add_argument('--output-dir', type=str, default='data/raw/historical',
                        help='Output directory')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing data instead of fetching')

    args = parser.parse_args()

    if args.validate:
        # Validate existing files
        print("Validating existing data files...")
        for period_name in PERIODS.keys():
            filepath = os.path.join(args.output_dir, f"btcusdt_1h_{period_name}.csv")
            if os.path.exists(filepath):
                result = validate_data(filepath)
                status = "VALID" if result['valid'] else "ISSUES"
                print(f"  {period_name}: {status} ({result['rows']} rows)")
                for issue in result['issues']:
                    print(f"    - {issue}")
            else:
                print(f"  {period_name}: NOT FOUND")
    elif args.period:
        # Fetch specific period
        os.makedirs(args.output_dir, exist_ok=True)
        fetch_period(args.period, PERIODS[args.period], args.output_dir)
    else:
        # Fetch all periods
        fetch_all_periods(args.output_dir)
