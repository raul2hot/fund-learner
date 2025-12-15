# Kaggle Data Downloads

This directory should contain Kaggle CSV files for historical data that has limited API availability.

## Required Datasets

Download the following datasets from Kaggle:

1. **Open Interest**
   - URL: https://kaggle.com/datasets/jesusgraterol/bitcoin-open-interest-binance-futures
   - File: `open_interest.csv`

2. **Long/Short Ratio**
   - URL: https://kaggle.com/datasets/jesusgraterol/bitcoin-longshort-ratio-binance-futures
   - File: `long_short_ratio.csv`

3. **Taker Buy/Sell Volume**
   - URL: https://kaggle.com/datasets/jesusgraterol/bitcoin-taker-buysell-volume-binance-futures
   - File: `taker_buy_sell_volume.csv`

## Download Instructions

### Option 1: Using Kaggle CLI (Recommended)

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Configure Kaggle API:
   - Go to https://kaggle.com/account
   - Click "Create New Token"
   - Save `kaggle.json` to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Download datasets:
   ```bash
   kaggle datasets download -d jesusgraterol/bitcoin-open-interest-binance-futures -p data/kaggle/ --unzip
   kaggle datasets download -d jesusgraterol/bitcoin-longshort-ratio-binance-futures -p data/kaggle/ --unzip
   kaggle datasets download -d jesusgraterol/bitcoin-taker-buysell-volume-binance-futures -p data/kaggle/ --unzip
   ```

### Option 2: Manual Download

1. Visit each URL above
2. Click "Download" button
3. Extract the ZIP files to this directory

## Why Kaggle Data?

Binance API only provides **30 days** of historical data for:
- Open Interest
- Long/Short Ratio
- Taker Buy/Sell Volume

Kaggle datasets (by jesusgraterol) provide **full historical data** going back to 2020.

The pipeline uses a **hybrid strategy**:
1. Load historical data from Kaggle CSV
2. Fetch recent data from Binance API (last 30 days)
3. Merge and deduplicate

## Update Frequency

These Kaggle datasets are updated **quarterly**. If you need more recent historical data:
1. Check for Kaggle dataset updates
2. Or use the builder script: https://github.com/jesusgraterol/binance-futures-dataset-builder

## Expected Files After Download

```
data/kaggle/
├── README.md (this file)
├── open_interest.csv
├── long_short_ratio.csv
└── taker_buy_sell_volume.csv
```
