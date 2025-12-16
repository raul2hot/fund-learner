"""
Training Loop for SPH-Net

Handles training, validation, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional

from .losses import TradingLoss
from .metrics import MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for SPH-Net trading classifier.

    Features:
    - Training and validation loops
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    - Gradient clipping
    - Detailed logging
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "experiments"
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Loss function
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32)
        self.criterion = TradingLoss(
            class_weights=class_weights,
            focal_gamma=config.focal_gamma
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_tradeable_accuracy': [],
            'learning_rate': []
        }

    def train_epoch(self) -> Dict:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(prices, features)

            # Compute loss
            loss_dict = self.criterion(outputs, labels, next_return)
            loss = loss_dict['total']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            preds = torch.argmax(outputs['logits'], dim=-1)
            tracker.update(
                preds=preds,
                targets=labels,
                loss=loss.item()
            )

            pbar.set_postfix({'loss': loss.item()})

        return tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Dict:
        """Run validation."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        tracker = MetricTracker()

        for batch in loader:
            prices = batch['prices'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            next_return = batch['next_return'].to(self.device)
            next_mae_long = batch['next_mae_long'].to(self.device)
            next_mae_short = batch['next_mae_short'].to(self.device)

            outputs = self.model(prices, features)
            loss_dict = self.criterion(outputs, labels, next_return)

            preds = torch.argmax(outputs['logits'], dim=-1)
            probs = torch.softmax(outputs['logits'], dim=-1)

            tracker.update(
                preds=preds,
                targets=labels,
                probs=probs,
                loss=loss_dict['total'].item(),
                next_return=next_return,
                next_mae_long=next_mae_long,
                next_mae_short=next_mae_short
            )

        metrics = tracker.compute()
        trading_metrics = tracker.compute_trading_metrics()
        metrics.update(trading_metrics)

        return metrics

    def train(self) -> Dict:
        """
        Full training loop with early stopping.

        Returns final metrics.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
            logger.info(f"Val Tradeable Accuracy: {val_metrics['tradeable_accuracy']:.4f}")

            # Trading metrics
            if 'trade_frequency' in val_metrics:
                logger.info(f"Trade Frequency: {val_metrics['trade_frequency']:.2f}%")
                if 'long_survival_rate' in val_metrics:
                    logger.info(f"Long Survival Rate: {val_metrics['long_survival_rate']:.2f}%")
                if 'short_survival_rate' in val_metrics:
                    logger.info(f"Short Survival Rate: {val_metrics['short_survival_rate']:.2f}%")

            # Update history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.history['val_tradeable_accuracy'].append(val_metrics['tradeable_accuracy'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Learning rate scheduling
            self.scheduler.step(val_metrics['avg_loss'])

            # Early stopping check
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt', val_metrics)
                logger.info("New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)

        # Save final model and history
        self._save_checkpoint('final_model.pt', val_metrics)
        self._save_history()

        return val_metrics

    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        # Convert metrics to serializable format
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': serializable_metrics,
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)

    def _save_history(self):
        """Save training history."""
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_best_model(self):
        """Load best model checkpoint."""
        checkpoint = torch.load(self.output_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['metrics']
