"""
Fetch 365 days of BTC/USDT data for extended backtesting.

This script fetches extended historical data to enable:
- 90-day test period (2160 hours) for statistical significance
- More training data for better model generalization
"""

from data.fetch_binance import fetch_binance_klines

if __name__ == "__main__":
    # Fetch 365 days instead of 180
    df = fetch_binance_klines(
        symbol="BTCUSDT",
        interval="1h",
        days=365,
        save_path="data/raw/btcusdt_1h_365d.csv"
    )
    print(f"Fetched {len(df)} candles ({len(df)/24:.0f} days)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
