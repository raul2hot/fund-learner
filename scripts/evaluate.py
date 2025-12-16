#!/usr/bin/env python
"""
Evaluation Script for SPH-Net

Evaluates trained model on test set.
Supports both standard 5-class and two-stage models.
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
from sph_net.models.two_stage import TwoStageModel
from data.dataset import TradingDataset
from torch.utils.data import DataLoader
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # === Configuration ===
    DATA_DIR = Path("prepared_data")
    MODEL_DIR = Path("experiments/run_001")
    OUTPUT_DIR = MODEL_DIR / "evaluation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not (MODEL_DIR / "best_model.pt").exists():
        logger.error(f"Model not found in {MODEL_DIR}")
        logger.error("Please run 'python scripts/train.py' first")
        return

    # === Load Test Data ===
    logger.info("Loading test data...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "feature_info.json") as f:
        feature_info = json.load(f)

    price_cols = feature_info['price_columns']
    eng_cols = [c for c in feature_info['engineered_columns'] if c in test_df.columns]

    logger.info(f"Test samples: {len(test_df)}")

    # === Load Model ===
    logger.info("Loading model...")
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # Detect model type from config
    model_type = getattr(config, 'model_type', 'standard')
    logger.info(f"Model type: {model_type}")

    if model_type == "two_stage":
        model = TwoStageModel(config)
        logger.info("Loaded Two-Stage Model")
    else:
        model = SPHNet(config)
        logger.info("Loaded Standard 5-class Model")

    model.load_state_dict(checkpoint['model_state_dict'])

    # === Create Test DataLoader ===
    test_dataset = TradingDataset(
        test_df, price_cols, eng_cols,
        window_size=config.window_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    logger.info(f"Test batches: {len(test_loader)}")

    # === Evaluate ===
    logger.info("\nEvaluating model...")
    evaluator = Evaluator(model, test_loader, device=config.device)
    metrics, predictions_df = evaluator.evaluate()

    # === Generate Report ===
    logger.info("\nGenerating report...")
    report = evaluator.generate_report(metrics, predictions_df, OUTPUT_DIR)

    print("\n" + report)

    logger.info(f"\nEvaluation results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
