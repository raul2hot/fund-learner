#!/usr/bin/env python
"""
Training Script for SPH-Net

Trains the model on prepared data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import json
import logging
import torch

from sph_net.config import SPHNetConfig
from sph_net.models.sph_net import SPHNet
from data.dataset import create_dataloaders
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    OUTPUT_DIR = Path("experiments/run_001")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not (DATA_DIR / "train.parquet").exists():
        logger.error(f"Prepared data not found in {DATA_DIR}")
        logger.error("Please run 'python scripts/prepare_data.py' first")
        return

    # === Load Prepared Data ===
    logger.info("Loading prepared data...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")

    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)

    price_cols = feature_info['price_columns']
    eng_cols = feature_info['engineered_columns']

    # Filter to available columns
    eng_cols = [c for c in eng_cols if c in train_df.columns]

    logger.info(f"Price features: {len(price_cols)}")
    logger.info(f"Engineered features: {len(eng_cols)}")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")

    # === Compute Class Weights from Data ===
    # Higher weights for rare classes to address imbalance
    label_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)

    # Inverse frequency weighting with smoothing
    class_weights = []
    for i in range(5):
        count = label_counts.get(i, 1)
        # Weight = total / (n_classes * count), capped at 20
        weight = min(total / (5 * count), 20.0)
        class_weights.append(weight)

    logger.info(f"Label distribution: {label_counts.to_dict()}")
    logger.info(f"Computed class weights: {[f'{w:.2f}' for w in class_weights]}")

    # === Create Model Config ===
    model_config = SPHNetConfig(
        n_price_features=len(price_cols),
        n_engineered_features=len(eng_cols),
        n_classes=5,
        window_size=64,
        d_model=128,
        n_heads=8,
        n_encoder_layers=3,
        dropout=0.1,
        batch_size=64,
        learning_rate=1e-4,
        epochs=100,
        patience=20,  # More patience for imbalanced data
        class_weights=class_weights,
        focal_gamma=2.0,
        device='cuda'
    )

    # === Create DataLoaders ===
    logger.info("Creating DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, val_df,  # Use val as placeholder for test
        price_columns=price_cols,
        feature_columns=eng_cols,
        window_size=model_config.window_size,
        batch_size=model_config.batch_size
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # === Create Model ===
    logger.info("Creating model...")
    model = SPHNet(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # === Train ===
    logger.info("\nStarting training...")
    trainer = Trainer(
        model=model,
        config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(OUTPUT_DIR)
    )

    final_metrics = trainer.train()

    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Final macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
