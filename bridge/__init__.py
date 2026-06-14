"""
BRIDGE: Behavioral Research Through Interpretable, Dimensionality-reduced Generative AI Embeddings

A Python package for creating interpretable, attribute-specific embeddings from text descriptions
using a partitioned neural network with contrastive learning.

This package implements the BRIDGE methodology for consumer research, enabling experiments
with real-world product descriptions as stimuli while extracting structured, low-dimensional,
interpretable attribute-specific representations.

Example usage:
    from bridge import BRIDGEPipeline

    # Initialize pipeline
    pipeline = BRIDGEPipeline(
        attributes=["region", "varietal"],  # Your attribute names
        output_dir="./output"
    )

    # Fit on data
    pipeline.fit(
        descriptions=df["description"],
        labels={"region": df["region"], "varietal": df["variety"]}
    )

    # Extract representations
    embeddings = pipeline.transform(df["description"])

    # Access attribute-specific embeddings
    region_emb = embeddings["region"]      # (n_samples, embedding_dim)
    varietal_emb = embeddings["varietal"]  # (n_samples, embedding_dim)
    nuisance = embeddings["nuisance"]      # (n_samples, nuisance_dim)

For R integration, use export():
    pipeline.export("./output")  # Creates .npy files readable by RcppCNPy
"""

__version__ = "0.1.0"
__author__ = "Anirban Mukherjee, Hannah H. Chang, and Sachin Gupta"

# Public API
from bridge.augmentation import (
    CONCISE,
    CREATIVE,
    DESCRIPTIVE,
    TECHNICAL,
    AugmentationResult,
    VariationStrategy,
    augment_descriptions,
    augment_descriptions_sync,
)
from bridge.baselines import WordEmbeddingBaseline, generate_word_embedding_baseline
from bridge.config import BRIDGEConfig
from bridge.empath_controls import generate_empath_controls, save_empath_controls
from bridge.encoder import AttributeEncoder
from bridge.extraction import compute_nuisance_controls, extract_representations
from bridge.model import BRIDGEModel
from bridge.pipeline import BRIDGEPipeline
from bridge.training import train_model, tune_hyperparameters
from bridge.validation import (
    compute_classification_metrics,
    load_classification_metrics,
    print_classification_metrics,
    save_classification_metrics,
)

__all__ = [
    # Main classes
    "BRIDGEPipeline",
    "BRIDGEModel",
    "BRIDGEConfig",
    "AttributeEncoder",
    # Functions
    "train_model",
    "tune_hyperparameters",
    "extract_representations",
    "compute_nuisance_controls",
    # Validation metrics
    "compute_classification_metrics",
    "print_classification_metrics",
    "save_classification_metrics",
    "load_classification_metrics",
    # Empath controls
    "generate_empath_controls",
    "save_empath_controls",
    # Baselines
    "WordEmbeddingBaseline",
    "generate_word_embedding_baseline",
    # Augmentation
    "VariationStrategy",
    "AugmentationResult",
    "augment_descriptions",
    "augment_descriptions_sync",
    "CONCISE",
    "DESCRIPTIVE",
    "TECHNICAL",
    "CREATIVE",
]
