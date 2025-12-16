#!/usr/bin/env python
"""
Data Preparation Script

Loads raw data, computes labels and features, saves prepared datasets.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import json

from labeling.candle_classifier import CandleLabeler, LabelingConfig


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


from labeling.label_analyzer import LabelAnalyzer
from features.feature_pipeline import FeaturePipeline
from data.data_splitter import TemporalSplitter, SplitConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_PATH = Path("data_pipleine/ml_data/BTCUSDT_ml_data.parquet")
    OUTPUT_DIR = Path("prepared_data")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # === Load Data ===
    logger.info(f"Loading data from {DATA_PATH}")

    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.info("Please run the data pipeline first to generate the data.")
        logger.info("Alternative: Creating synthetic data for testing...")

        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 10000

        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        base_price = 30000

        # Generate synthetic OHLCV data
        returns = np.random.randn(n_samples) * 0.01  # 1% std
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.005)),
            'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.005)),
            'close': prices * (1 + np.random.randn(n_samples) * 0.003),
            'volume': np.random.randint(100, 10000, n_samples).astype(float),
        })

        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        logger.info(f"Created synthetic data with {len(df)} rows")
    else:
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Loaded {len(df)} rows")

    # Drop rows with NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna(subset=critical_cols)
    logger.info(f"After dropping NaN: {len(df)} rows")

    # === Labeling ===
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Labeling")
    logger.info("="*50)

    # Labeling thresholds - adjust these based on your data:
    # - Lower strong_move_threshold = more HIGH_BULL/LOW_BEAR samples
    # - Higher clean_path_mae_threshold = higher survival rate but noisier signals
    label_config = LabelingConfig(
        strong_move_threshold=0.012,    # 1.2% (was 1.5%) - more tradeable signals
        weak_move_threshold=0.004,      # 0.4% (was 0.5%) - tighter range filter
        clean_path_mae_threshold=0.006  # 0.6% (was 0.5%) - better survival rate
    )
    labeler = CandleLabeler(label_config)

    df_labeled = labeler.label_dataset(df)

    # Analyze labels
    analyzer = LabelAnalyzer(df_labeled)
    stats = analyzer.compute_statistics()

    logger.info("\nLabel Distribution:")
    for i in range(5):
        logger.info(f"  Class {i}: {stats[f'class_{i}_count']:,} ({stats[f'class_{i}_pct']:.2f}%)")

    balanced, msg = analyzer.check_class_balance()
    logger.info(f"\nBalance check: {msg}")

    validation = analyzer.validate_label_correctness()
    logger.info(f"Label validation: {'PASSED' if validation['all_valid'] else 'FAILED'}")

    # === Feature Engineering ===
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*50)

    feature_pipeline = FeaturePipeline()
    df_features = feature_pipeline.compute_all_features(df_labeled)

    # Validate features
    validation = feature_pipeline.validate_features(df_features)
    logger.info(f"Feature validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['issues']:
        for col, issue in list(validation['issues'].items())[:10]:
            logger.warning(f"  {col}: {issue}")

    # === Drop Warmup Period ===
    warmup = feature_pipeline.get_warmup_periods()
    df_clean = df_features.iloc[warmup:].copy()
    logger.info(f"Dropped {warmup} warmup rows, remaining: {len(df_clean)}")

    # Drop rows with NaN labels (last row)
    df_clean = df_clean[df_clean['label'].notna()]
    logger.info(f"After dropping NaN labels: {len(df_clean)}")

    # === Split Data ===
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Data Splitting")
    logger.info("="*50)

    splitter = TemporalSplitter(SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    ))

    train_df, val_df, test_df = splitter.split(df_clean, 'timestamp')

    # Verify no leakage
    no_leakage = splitter.verify_no_leakage(train_df, val_df, test_df, 'timestamp')
    assert no_leakage, "TEMPORAL LEAKAGE DETECTED!"

    # Get split summary
    summary = splitter.get_split_summary(train_df, val_df, test_df)

    logger.info("\nSplit Summary:")
    for split_name, split_stats in summary.items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Samples: {split_stats['n_samples']}")
        logger.info(f"  Label Distribution: {split_stats['label_pct']}")

    # === Normalize Features ===
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Feature Normalization")
    logger.info("="*50)

    # Normalize using training data statistics
    price_cols, eng_cols = feature_pipeline.get_feature_columns()

    train_normalized, norm_stats = feature_pipeline.normalize_features(
        train_df, fit_data=train_df
    )
    val_normalized = feature_pipeline.apply_normalization(val_df, norm_stats)
    test_normalized = feature_pipeline.apply_normalization(test_df, norm_stats)

    logger.info(f"Normalized {len(norm_stats)} features")

    # === Save Prepared Data ===
    logger.info("\n" + "="*50)
    logger.info("STEP 5: Saving Prepared Data")
    logger.info("="*50)

    train_normalized.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_normalized.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test_normalized.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    # Save normalization stats
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(convert_to_serializable(norm_stats), f, indent=2)

    # Save feature columns
    feature_info = {
        'price_columns': price_cols,
        'engineered_columns': eng_cols,
        'all_columns': list(train_normalized.columns)
    }
    with open(OUTPUT_DIR / "feature_info.json", 'w') as f:
        json.dump(convert_to_serializable(feature_info), f, indent=2)

    # Save label statistics
    with open(OUTPUT_DIR / "label_stats.json", 'w') as f:
        json.dump(convert_to_serializable(stats), f, indent=2)

    logger.info(f"\nSaved prepared data to {OUTPUT_DIR}/")
    logger.info("Files created:")
    logger.info("  - train.parquet")
    logger.info("  - val.parquet")
    logger.info("  - test.parquet")
    logger.info("  - normalization_stats.json")
    logger.info("  - feature_info.json")
    logger.info("  - label_stats.json")

    logger.info("\n" + "="*50)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*50)


if __name__ == "__main__":
    main()
