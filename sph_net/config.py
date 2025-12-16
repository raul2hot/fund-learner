"""SPH-Net Configuration for 5-Class Trading Classification"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SPHNetConfig:
    """Configuration for SPH-Net model."""

    # Data
    window_size: int = 64
    n_price_features: int = 5          # OHLCV
    n_engineered_features: int = 34    # Technical indicators (increased for regime features)
    n_classes: int = 5                  # 5-class classification

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_feedforward: int = 512
    dropout: float = 0.1

    # Model type: "standard" (5-class) or "two_stage" (binary tradeable + direction)
    model_type: str = "standard"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15

    # Loss weights for 5-class model
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

    # Trading-aware loss parameters
    use_trading_aware_loss: bool = False
    fn_penalty: float = 3.0    # Missing tradeable opportunity
    fp_penalty: float = 1.5    # False positive for tradeable
    direction_penalty: float = 2.0  # Wrong direction on tradeable

    # Two-stage model parameters
    tradeable_pos_weight: float = 7.0  # Handles ~12%/88% imbalance
    tradeable_threshold: float = 0.5   # Threshold for predicting tradeable
    direction_threshold: float = 0.5   # Threshold for LONG vs SHORT

    # Device
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Paths
    data_path: str = "data_pipleine/ml_data/BTCUSDT_ml_data.parquet"
    output_dir: str = "experiments"

    # Data split - standard ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Data split - date-based (overrides ratios if set)
    train_end_date: Optional[str] = None
    val_end_date: Optional[str] = None

    # Data split - preset (overrides both ratios and dates)
    # Options: "standard", "volatile_2022", "volatile_2021", "volatile_2020"
    split_preset: Optional[str] = None

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


# Preset configurations for common use cases
def get_volatile_2022_config() -> ExperimentConfig:
    """Config for testing on volatile 2022 bear market."""
    config = ExperimentConfig()
    config.split_preset = "volatile_2022"
    return config


def get_two_stage_config() -> ExperimentConfig:
    """Config for two-stage model approach."""
    config = ExperimentConfig()
    config.model.model_type = "two_stage"
    config.model.n_engineered_features = 34  # With regime features
    return config


def get_trading_aware_config() -> ExperimentConfig:
    """Config with trading-aware loss function."""
    config = ExperimentConfig()
    config.model.use_trading_aware_loss = True
    config.model.fn_penalty = 3.0
    config.model.fp_penalty = 1.5
    return config
