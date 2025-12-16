"""SPH-Net Configuration for 5-Class Trading Classification"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SPHNetConfig:
    """Configuration for SPH-Net model."""

    # Data
    window_size: int = 64
    n_price_features: int = 5          # OHLCV
    n_engineered_features: int = 25    # Technical indicators
    n_classes: int = 5                  # 5-class classification

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_feedforward: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15

    # Loss weights
    class_weights: List[float] = field(default_factory=lambda: [
        2.0,   # HIGH_BULL - upweight rare class
        1.0,   # BULL
        0.5,   # RANGE_BOUND - downweight majority
        1.0,   # BEAR
        2.0    # LOW_BEAR - upweight rare class
    ])

    # Focal loss parameters (handles class imbalance)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Device
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Paths
    data_path: str = "data_pipleine/ml_data/BTCUSDT_ml_data.parquet"
    output_dir: str = "experiments"

    # Data split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Labeling
    strong_move_threshold: float = 0.015
    weak_move_threshold: float = 0.005
    clean_path_mae_threshold: float = 0.005

    # Model
    model: SPHNetConfig = field(default_factory=SPHNetConfig)

    # Random seed
    seed: int = 42


# Backwards compatibility alias
Config = SPHNetConfig
