#!/usr/bin/env python
"""
Training Script for SPH-Net

Trains the model on prepared data.
Supports both standard 5-class and two-stage models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
import torch

from sph_net.config import SPHNetConfig
from sph_net.models.sph_net import SPHNet
from sph_net.models.two_stage import TwoStageModel
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

    # === MODEL TYPE SELECTION ===
    # Options: "standard" (5-class) or "two_stage" (recommended for distribution shift)
    MODEL_TYPE = "two_stage"

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
    label_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)

    class_weights = []
    for i in range(5):
        count = label_counts.get(i, 1)
        weight = min(np.sqrt(total / (5 * count)), 5.0)
        class_weights.append(weight)

    logger.info(f"Label distribution: {label_counts.to_dict()}")
    logger.info(f"Computed class weights: {[f'{w:.2f}' for w in class_weights]}")

    # Compute tradeable ratio for two-stage model
    tradeable_count = ((train_df['label'] == 0) | (train_df['label'] == 4)).sum()
    tradeable_ratio = tradeable_count / total
    tradeable_pos_weight = (1 - tradeable_ratio) / tradeable_ratio
    logger.info(f"Tradeable ratio: {tradeable_ratio:.2%}, pos_weight: {tradeable_pos_weight:.2f}")

    # === Create Model Config ===
    model_config = SPHNetConfig(
        n_price_features=len(price_cols),
        n_engineered_features=len(eng_cols),
        n_classes=5,
        window_size=64,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        dropout=0.2,
        batch_size=32,
        learning_rate=5e-5,
        epochs=100,
        patience=25,
        class_weights=class_weights,
        focal_gamma=2.0,
        device='cuda',
        # Model type
        model_type=MODEL_TYPE,
        # Two-stage parameters
        tradeable_pos_weight=tradeable_pos_weight,
    )

    # === Create DataLoaders ===
    logger.info("Creating DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_df, val_df, val_df,
        price_columns=price_cols,
        feature_columns=eng_cols,
        window_size=model_config.window_size,
        batch_size=model_config.batch_size
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # === Create Model ===
    logger.info(f"Creating {MODEL_TYPE} model...")
    if MODEL_TYPE == "two_stage":
        model = TwoStageModel(model_config)
        logger.info("Using Two-Stage Model: Stage1=Tradeable?, Stage2=Long/Short")
    else:
        model = SPHNet(model_config)
        logger.info("Using Standard 5-class Model")

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
