"""
Fetch BTC/USDT 1h OHLCV data from Binance public API.
No API key required for historical klines.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180,
    save_path: str = "data/raw/btcusdt_1h.csv"
):
    """
    Fetch historical klines from Binance API.

    Args:
        symbol: Trading pair (default BTCUSDT)
        interval: Candle interval (1h for hourly)
        days: Number of days of history to fetch
        save_path: Where to save the CSV

    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"

    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_time

    print(f"Fetching {days} days of {interval} data for {symbol}...")

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000  # Max per request
        }

        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()

        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1  # Next millisecond after last candle

        print(f"  Fetched {len(all_klines)} candles...")
        time.sleep(0.1)  # Rate limiting

    # Convert to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]

    df = pd.DataFrame(all_klines, columns=columns)

    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = df[col].astype(float)

    df['trades'] = df['trades'].astype(int)

    # Keep essential columns
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'trades', 'quote_volume']]
    df = df.rename(columns={'open_time': 'timestamp'})

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} candles to {save_path}")

    return df


if __name__ == "__main__":
    df = fetch_binance_klines(days=180)
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
