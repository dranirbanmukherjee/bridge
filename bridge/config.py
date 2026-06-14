"""
Configuration dataclass for BRIDGE pipeline.

Provides centralized configuration for all pipeline components including
model architecture, training, and I/O settings.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BRIDGEConfig:
    """Configuration for BRIDGE pipeline hyperparameters.

    This dataclass holds hyperparameters for the BRIDGE pipeline, including
    model architecture, training settings, and tuning configuration.

    Domain-specific settings (like attribute names) are passed directly to
    BRIDGEPipeline, not stored here. This keeps BRIDGEConfig generic and
    reusable across different domains (wine, coffee, etc.).

    WARNING: Do not modify these defaults directly. Instead, override via:
        - BRIDGEConfig(epochs_final=50, ...) in your script
        - Complexity presets: BRIDGEConfig.quick(), .standard(), .thorough()
        - External JSON config: BRIDGEConfig.load("my_config.json")

    Attributes:
        embedding_backend: Embedding backend to use ("openai" or "gemma").
        embedding_model: Model identifier for the selected backend.
        embedding_dim: Full dimensionality of embeddings (3072 for OpenAI, 768 for gemma).
            Dimension reduction is handled by mask_size, which the tuner optimizes.
        mini_batch_size: Number of items in each contrastive mini-batch
            (1 anchor + N-1 negatives).
        batch_size: Training batch size.
        projection_units: Units in the first projection layer.
        embedding_units_per_attribute: Embedding dimensions per attribute.
        mask_size: Number of embedding dimensions to use (for dimensionality reduction).
        learning_rate: Initial learning rate for Adam optimizer.
        contrastive_temp: Temperature for contrastive loss.
        contrastive_weight: Weight for contrastive loss term.
        epochs_tuning: Max epochs per hyperparameter tuning trial.
        epochs_final: Max epochs for final training.
        early_stopping_patience: Epochs without improvement before stopping.
        tuner_trials: Number of Optuna hyperparameter search trials.
        validation_split: Fraction of data for validation.
        num_nuisance_dims: Number of nuisance control dimensions. If None, uses
            default (5) with a warning and elbow plot to guide selection.
        nuisance_method: Method for nuisance extraction ("svd" or "umap").
        nuisance_show_elbow_plot: For SVD, show elbow plot when using default dims.
        nuisance_umap_*: UMAP-specific parameters (n_neighbors, min_dist, metric).
        seed: Random seed for reproducibility.
    """

    # Embedding settings
    embedding_backend: str = "openai"  # "openai" or "gemma"
    embedding_model: str = "text-embedding-3-large"  # Model ID for the backend
    embedding_dim: int = 3072  # Full embedding dimension (3072 for openai, 768 for gemma)

    # Model architecture
    mini_batch_size: int = 10
    batch_size: int = 8
    projection_units: int = 128
    embedding_units_per_attribute: int = 8
    mask_size: int = 2048
    dropout_rate: float = 0.125

    # Training settings
    learning_rate: float = 0.00423
    contrastive_temp: float = 0.1
    contrastive_weight: float = 0.1
    epochs_tuning: int = 50
    epochs_final: int = 1000
    early_stopping_patience: int = 10
    lr_reduce_patience: int = 2
    lr_reduce_factor: float = 0.9
    min_lr: float = 1e-6

    # Tuner settings
    tuner_trials: int = 75
    tuner_startup_trials: int = 16

    # Data settings
    validation_split: float = 0.1

    # Nuisance control settings
    num_nuisance_dims: int | None = None  # None = use default (5) with warning/elbow plot
    nuisance_method: str = "svd"  # "svd" or "umap"
    nuisance_show_elbow_plot: bool = True  # For SVD, show elbow plot when using default dims
    nuisance_umap_n_neighbors: int = 15  # UMAP n_neighbors parameter
    nuisance_umap_min_dist: float = 0.1  # UMAP min_dist parameter
    nuisance_umap_metric: str = "euclidean"  # UMAP distance metric
    nuisance_random_state: int = 42  # Random seed for UMAP

    # Embedding truncation
    fixed_mask_size: int | None = None  # If set, locks mask_size during tuning.
    # Use for non-Matryoshka embeddings (e.g., BERT) where dimension order is arbitrary.
    # For Matryoshka embeddings (OpenAI), leave as None to let the tuner search.

    # Reproducibility
    seed: int = 88  # General seed (matches original R global seed)
    array_seed: int = 42  # Seed for 3D array building (matches original R Block 7)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return asdict(self)

    def save(self, path: str) -> None:
        """Save config to JSON file.

        Args:
            path: Path to the output JSON file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BRIDGEConfig":
        """Load config from JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            BRIDGEConfig instance populated with values from the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def quick(cls) -> "BRIDGEConfig":
        """Create config for fast iteration during development.

        Uses reduced epochs and trials for quick feedback loops.
        Suitable for debugging, testing new ideas, or small datasets.

        Settings: epochs_final=50, epochs_tuning=25, tuner_trials=25

        Returns:
            BRIDGEConfig with quick iteration settings.
        """
        return cls(
            epochs_final=50,
            epochs_tuning=25,
            tuner_trials=25,
            tuner_startup_trials=10,
        )

    @classmethod
    def standard(cls) -> "BRIDGEConfig":
        """Create config for standard production use.

        Balanced settings suitable for most experiments. Provides
        reasonable training time with good model quality.

        Settings: epochs_final=200, epochs_tuning=35, tuner_trials=50

        Returns:
            BRIDGEConfig with standard production settings.
        """
        return cls(
            epochs_final=200,
            epochs_tuning=35,
            tuner_trials=50,
            tuner_startup_trials=12,
        )

    @classmethod
    def thorough(cls) -> "BRIDGEConfig":
        """Create config for thorough/publication-quality training.

        Maximum epochs and trials for best possible model quality.
        Use when model performance is critical and compute time is not.

        Settings: epochs_final=1000, epochs_tuning=50, tuner_trials=75

        Returns:
            BRIDGEConfig with thorough training settings.
        """
        return cls(
            epochs_final=1000,
            epochs_tuning=50,
            tuner_trials=75,
            tuner_startup_trials=16,
        )

    @classmethod
    def for_gemma(cls) -> "BRIDGEConfig":
        """Create config for local Gemma embedding backend.

        Uses the local embeddinggemma-300m model instead of OpenAI API.
        Embeddings are 768 dimensions; mask_size is adjusted accordingly.

        Returns:
            BRIDGEConfig configured for Gemma embeddings.
        """
        return cls(
            embedding_backend="gemma",
            embedding_model="google/embeddinggemma-300m",
            embedding_dim=768,
            mask_size=512,
        )


# Default configuration instance (production settings)
DEFAULT_CONFIG = BRIDGEConfig()
