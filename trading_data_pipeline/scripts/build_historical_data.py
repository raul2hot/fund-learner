#!/usr/bin/env python3
"""
Build historical data from Binance API with proper rate limiting.
Fetches Open Interest, Long/Short Ratio, and Taker Volume.

Usage:
    python build_historical_data.py --start 2020-01-01 --end 2025-12-15
"""

import argparse
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

# Rate limiting: Binance allows ~1200 requests/min for these endpoints
# We'll be conservative with 1 request per second
REQUEST_DELAY = 1.0
MAX_RETRIES = 3

def fetch_with_retry(url: str, params: dict, retries: int = MAX_RETRIES) -> list:
    """Fetch data with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP {response.status_code}, retrying...")
                time.sleep(5 * (attempt + 1))

        except Exception as e:
            print(f"  Error: {e}, retrying...")
            time.sleep(5 * (attempt + 1))

    return []

def fetch_open_interest(start_date: str, end_date: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch Open Interest history."""
    print(f"\n=== Fetching Open Interest ===")
    url = "https://fapi.binance.com/futures/data/openInterestHist"

    all_data = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=30), end_dt)

        params = {
            "symbol": symbol,
            "period": "1h",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000),
            "limit": 500
        }

        print(f"  Fetching {current_start.date()} to {current_end.date()}...", end=" ")
        data = fetch_with_retry(url, params)

        if data:
            all_data.extend(data)
            print(f"got {len(data)} records")
        else:
            print("no data")

        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        'timestamp': 'timestamp',
        'sumOpenInterest': 'sum_open_interest',
        'sumOpenInterestValue': 'sum_open_interest_value'
    })
    df = df[['timestamp', 'sum_open_interest', 'sum_open_interest_value']]
    df = df.drop_duplicates('timestamp').sort_values('timestamp')

    print(f"  Total: {len(df)} records")
    return df

def fetch_long_short_ratio(start_date: str, end_date: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch Long/Short Ratio history."""
    print(f"\n=== Fetching Long/Short Ratio ===")
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"

    all_data = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=30), end_dt)

        params = {
            "symbol": symbol,
            "period": "1h",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000),
            "limit": 500
        }

        print(f"  Fetching {current_start.date()} to {current_end.date()}...", end=" ")
        data = fetch_with_retry(url, params)

        if data:
            all_data.extend(data)
            print(f"got {len(data)} records")
        else:
            print("no data")

        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        'timestamp': 'timestamp',
        'longShortRatio': 'long_short_ratio',
        'longAccount': 'long_account',
        'shortAccount': 'short_account'
    })
    df = df[['timestamp', 'long_account', 'short_account', 'long_short_ratio']]
    df = df.drop_duplicates('timestamp').sort_values('timestamp')

    print(f"  Total: {len(df)} records")
    return df

def fetch_taker_volume(start_date: str, end_date: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch Taker Buy/Sell Volume history."""
    print(f"\n=== Fetching Taker Volume ===")
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"

    all_data = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=30), end_dt)

        params = {
            "symbol": symbol,
            "period": "1h",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(current_end.timestamp() * 1000),
            "limit": 500
        }

        print(f"  Fetching {current_start.date()} to {current_end.date()}...", end=" ")
        data = fetch_with_retry(url, params)

        if data:
            all_data.extend(data)
            print(f"got {len(data)} records")
        else:
            print("no data")

        current_start = current_end

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        'timestamp': 'timestamp',
        'buySellRatio': 'buy_sell_ratio',
        'buyVol': 'buy_vol',
        'sellVol': 'sell_vol'
    })
    df = df[['timestamp', 'buy_vol', 'sell_vol', 'buy_sell_ratio']]
    df = df.drop_duplicates('timestamp').sort_values('timestamp')

    print(f"  Total: {len(df)} records")
    return df

def main():
    parser = argparse.ArgumentParser(description="Build historical Binance futures data")
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.utcnow().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/kaggle', help='Output directory')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building historical data from {args.start} to {args.end}")
    print(f"Output directory: {output_dir}")
    print(f"Symbol: {args.symbol}")
    print("This may take a while due to rate limiting...")

    # Fetch Open Interest
    oi_df = fetch_open_interest(args.start, args.end, args.symbol)
    if not oi_df.empty:
        oi_path = output_dir / "open_interest.csv"
        oi_df.to_csv(oi_path, index=False)
        print(f"Saved: {oi_path}")

    # Fetch Long/Short Ratio
    ls_df = fetch_long_short_ratio(args.start, args.end, args.symbol)
    if not ls_df.empty:
        ls_path = output_dir / "long_short_ratio.csv"
        ls_df.to_csv(ls_path, index=False)
        print(f"Saved: {ls_path}")

    # Fetch Taker Volume
    tv_df = fetch_taker_volume(args.start, args.end, args.symbol)
    if not tv_df.empty:
        tv_path = output_dir / "taker_buy_sell_volume.csv"
        tv_df.to_csv(tv_path, index=False)
        print(f"Saved: {tv_path}")

    print("\n=== Done! ===")
    print("You can now run the main pipeline:")
    print("  python trading_data_pipeline/scripts/fetch_all_data.py")

if __name__ == '__main__':
    main()
