"""
BRIDGE neural network model implementation in PyTorch.

This module provides the core neural network architecture for BRIDGE:
- Custom layers (EmbeddingTrimmingLayer, SplitLayer, TakeAnchorLayer, ContrastiveLayer)
- BRIDGEModel class with multi-task learning (classification + contrastive)
- Support for variable number of attributes
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device for the best available hardware accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Custom layers
# ---------------------------------------------------------------------------

class EmbeddingTrimmingLayer(nn.Module):
    """Selects the first mask_size dimensions from the last axis.

    Used for dimensionality reduction by taking only the most informative
    dimensions from the input embeddings.

    Attributes:
        mask_size: Number of dimensions to keep.

    Input shape:
        (batch, mini_batch, features)

    Output shape:
        (batch, mini_batch, mask_size)
    """

    def __init__(self, mask_size: int):
        """Initialize the trimming layer.

        Args:
            mask_size: Number of dimensions to keep from the input.
        """
        super().__init__()
        self.mask_size = mask_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trim input to first mask_size dimensions.

        Args:
            x: Input tensor of shape (batch, mini_batch, features).

        Returns:
            Tensor of shape (batch, mini_batch, mask_size).
        """
        return x[:, :, :self.mask_size]

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"mask_size={self.mask_size}"


class SplitLayer(nn.Module):
    """Splits the last dimension into N equal parts.

    Used to partition the shared representation into attribute-specific
    embeddings.

    Attributes:
        num_splits: Number of parts to split into.

    Input shape:
        (..., D) where D is divisible by num_splits

    Output:
        Tuple of num_splits tensors each with shape (..., D // num_splits)
    """

    def __init__(self, num_splits: int = 2):
        """Initialize the split layer.

        Args:
            num_splits: Number of equal parts to split the last dimension into.
        """
        super().__init__()
        self.num_splits = num_splits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split input tensor along the last dimension.

        Args:
            x: Input tensor of shape (..., D) where D is divisible by num_splits.

        Returns:
            Tuple of num_splits tensors, each of shape (..., D // num_splits).

        Raises:
            ValueError: If input's last dimension is not divisible by num_splits.
        """
        last_dim = x.shape[-1]
        if last_dim % self.num_splits != 0:
            raise ValueError(
                f"SplitLayer input's last dimension ({last_dim}) must be "
                f"divisible by num_splits ({self.num_splits})."
            )
        return torch.chunk(x, self.num_splits, dim=-1)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"num_splits={self.num_splits}"


class TakeAnchorLayer(nn.Module):
    """Extracts slice at index 0 along the mini-batch dimension.

    Used to extract only the anchor embedding from the contrastive mini-batch
    for classification tasks.

    Input shape:
        (batch, mini_batch, features)

    Output shape:
        (batch, features)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the anchor (first item) from each mini-batch.

        Args:
            x: Input tensor of shape (batch, mini_batch, features).

        Returns:
            Tensor of shape (batch, features) containing only anchor embeddings.
        """
        return x[:, 0, :]


class ContrastiveLayer(nn.Module):
    """Computes InfoNCE-style contrastive loss from attribute embeddings.

    For each attribute, computes cosine similarity between anchor (position 0)
    and negatives (positions 1..N-1), then combines via log-sum-exp to produce
    a differentiable contrastive loss.

    Attributes:
        mini_batch_size: Number of items in mini-batch (1 anchor + N-1 negatives).
        temperature: Temperature scaling for cosine similarities. Lower values
            make the model more confident in distinguishing samples.
    """

    def __init__(self, mini_batch_size: int = 10, temperature: float = 0.1):
        """Initialize the contrastive layer.

        Args:
            mini_batch_size: Number of items in mini-batch (1 anchor + N-1 negatives).
            temperature: Temperature scaling for cosine similarities.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.temperature = temperature

    def forward(self, *attribute_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for any number of attribute embeddings.

        Args:
            *attribute_embeddings: Variable number of tensors, each of shape
                (batch, mini_batch, dim).

        Returns:
            Tensor of shape (batch,) containing per-sample contrastive loss.
        """
        num_negatives = self.mini_batch_size - 1

        # Stack and normalize: (num_attrs, batch, mini_batch, dim)
        stacked = torch.stack(attribute_embeddings, dim=0)
        normalized = F.normalize(stacked, p=2, dim=-1)

        # Anchor vs negatives (vectorized over attributes)
        anchor = normalized[:, :, 0:1, :].expand(-1, -1, num_negatives, -1)
        negatives = normalized[:, :, 1:self.mini_batch_size, :]
        similarities = (anchor * negatives).sum(dim=-1)

        # InfoNCE denominator, sum over attributes
        denom = 1.0 + torch.exp(similarities / self.temperature).sum(dim=2)
        return torch.log(denom).sum(dim=0)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"mini_batch_size={self.mini_batch_size}, temperature={self.temperature}"


# ---------------------------------------------------------------------------
# BRIDGE Model
# ---------------------------------------------------------------------------

class BRIDGEModel(nn.Module):
    """BRIDGE neural network for learning attribute-specific embeddings.

    Multi-task learning model that jointly optimizes classification and
    contrastive objectives to learn interpretable, attribute-specific
    representations from text embeddings.

    Architecture:
        Input (batch, mini_batch, orig_size)
        -> EmbeddingTrimmingLayer (mask_size)
        -> Dense + GELU + Dropout (projection_units)
        -> Dense + GELU + Dropout (embedding_units)
        -> SplitLayer -> attribute embeddings (one per attribute)
        -> TakeAnchorLayer (for classification path)
        -> Dense (softmax) for each attribute classification
        -> ContrastiveLayer (for contrastive loss)

    Attributes:
        attribute_names: Names of attributes to learn embeddings for.
        attribute_sizes: Number of classes for each attribute.
        orig_size: Original embedding dimensionality (e.g., 3072 for OpenAI).
        mini_batch_size: Size of contrastive mini-batches.
        projection_units: Units in projection layer.
        embedding_units_per_attribute: Embedding dimensions per attribute.
        mask_size: Number of input dimensions to use.
        dropout_rate: Dropout probability.
        contrastive_temperature: Temperature for contrastive loss.
        contrastive_weight: Weight for contrastive loss term.
    """

    def __init__(
        self,
        attribute_names: list[str],
        attribute_sizes: dict[str, int],
        orig_size: int = 3072,
        mini_batch_size: int = 10,
        projection_units: int = 128,
        embedding_units_per_attribute: int = 8,
        mask_size: int = 2048,
        dropout_rate: float = 0.125,
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 0.1,
    ):
        """Initialize the BRIDGE model.

        Args:
            attribute_names: List of attribute names (e.g., ['region', 'varietal']).
            attribute_sizes: Dict mapping attribute names to number of classes.
            orig_size: Input embedding dimensionality (768 for Gemma, 3072 for OpenAI).
            mini_batch_size: Items per contrastive mini-batch (1 anchor + N-1 negatives).
            projection_units: Hidden units in the projection layer.
            embedding_units_per_attribute: Output embedding dimensions per attribute.
            mask_size: Number of input dimensions to use (for MRL truncation).
            dropout_rate: Dropout probability for regularization.
            contrastive_temperature: Temperature for InfoNCE contrastive loss.
            contrastive_weight: Weight for contrastive loss term in total loss.
        """
        super().__init__()

        self.attribute_names = list(attribute_names)
        self.attribute_sizes = dict(attribute_sizes)
        self.orig_size = orig_size
        self.mini_batch_size = mini_batch_size
        self.projection_units = projection_units
        self.embedding_units_per_attribute = embedding_units_per_attribute
        self.mask_size = mask_size
        self.dropout_rate = dropout_rate
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

        # Total embedding units (sum across attributes)
        num_attributes = len(attribute_names)
        total_embedding_units = embedding_units_per_attribute * num_attributes

        # Layers
        self.trim_layer = EmbeddingTrimmingLayer(mask_size)
        self.projection_dense = nn.Linear(mask_size, projection_units)
        self.dropout_projection = nn.Dropout(dropout_rate)
        self.reduction_dense = nn.Linear(projection_units, total_embedding_units)
        self.dropout_reduction = nn.Dropout(dropout_rate)
        self.split_layer = SplitLayer(num_splits=num_attributes)
        self.take_anchor_layer = TakeAnchorLayer()

        # Classification heads (one per attribute)
        self.classifiers = nn.ModuleDict({
            name: nn.Linear(embedding_units_per_attribute, attribute_sizes[name])
            for name in attribute_names
        })

        # Contrastive layer
        self.contrastive_layer = ContrastiveLayer(
            mini_batch_size=mini_batch_size,
            temperature=contrastive_temperature,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, mini_batch, orig_size).

        Returns:
            Dictionary with keys:
            - One key per attribute: classification logits (batch, num_classes)
            - 'contrastive': contrastive loss values (batch,)
        """
        # Shared encoder path
        x = self.trim_layer(x)
        x = self.projection_dense(x)
        x = F.gelu(x)  # pylint: disable=not-callable  # torch C-extension; pylint can't infer it is callable
        x = self.dropout_projection(x)
        x = self.reduction_dense(x)
        x = F.gelu(x)  # pylint: disable=not-callable  # torch C-extension; pylint can't infer it is callable
        x = self.dropout_reduction(x)

        # Split into attribute embeddings
        attribute_embeddings = self.split_layer(x)  # tuple of (batch, mini_batch, emb_dim)

        # Store embeddings by name
        embeddings_dict = dict(zip(self.attribute_names, attribute_embeddings))

        # Classification path (anchor only)
        outputs = {}
        for name, emb in embeddings_dict.items():
            anchor = self.take_anchor_layer(emb)  # (batch, emb_dim)
            logits = self.classifiers[name](anchor)  # (batch, num_classes)
            outputs[name] = logits

        # Contrastive path (full mini-batch)
        outputs['contrastive'] = self.contrastive_layer(*attribute_embeddings)

        return outputs

    def get_attribute_embeddings(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract intermediate attribute embeddings (3D, full mini-batch).

        Args:
            x: Input tensor of shape (batch, mini_batch, orig_size).

        Returns:
            Dictionary mapping attribute names to embeddings of shape
            (batch, mini_batch, emb_dim).
        """
        # Shared encoder path
        x = self.trim_layer(x)
        x = self.projection_dense(x)
        x = F.gelu(x)  # pylint: disable=not-callable  # torch C-extension; pylint can't infer it is callable
        x = self.dropout_projection(x)
        x = self.reduction_dense(x)
        x = F.gelu(x)  # pylint: disable=not-callable  # torch C-extension; pylint can't infer it is callable
        x = self.dropout_reduction(x)

        # Split into attribute embeddings
        attribute_embeddings = self.split_layer(x)

        return dict(zip(self.attribute_names, attribute_embeddings))

    def get_anchor_embeddings(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract anchor-only attribute embeddings (2D).

        Extracts embeddings for only the anchor samples (position 0 in mini-batch),
        discarding the negative samples.

        Args:
            x: Input tensor of shape (batch, mini_batch, orig_size).

        Returns:
            Dictionary mapping attribute names to embeddings of shape (batch, emb_dim).
        """
        embeddings_3d = self.get_attribute_embeddings(x)
        return {
            name: self.take_anchor_layer(emb)
            for name, emb in embeddings_3d.items()
        }

    def get_hp_dict(self) -> dict[str, Any]:
        """Get hyperparameters as dictionary (for saving/loading).

        Returns:
            Dictionary containing all model hyperparameters needed to
            reconstruct the model architecture.
        """
        return {
            "attribute_names": self.attribute_names,
            "attribute_sizes": self.attribute_sizes,
            "orig_size": self.orig_size,
            "mini_batch_size": self.mini_batch_size,
            "projection_units": self.projection_units,
            "embedding_units_per_attribute": self.embedding_units_per_attribute,
            "mask_size": self.mask_size,
            "dropout_rate": self.dropout_rate,
            "contrastive_temperature": self.contrastive_temperature,
            "contrastive_weight": self.contrastive_weight,
        }

    @classmethod
    def from_hp_dict(cls, hp_dict: dict[str, Any]) -> "BRIDGEModel":
        """Create model from hyperparameter dictionary.

        Args:
            hp_dict: Dictionary of hyperparameters as returned by get_hp_dict().

        Returns:
            New BRIDGEModel instance with the specified hyperparameters.
        """
        return cls(**hp_dict)


# ---------------------------------------------------------------------------
# Model save/load utilities
# ---------------------------------------------------------------------------

def save_model(model: BRIDGEModel, path: str) -> None:
    """Save model state and hyperparameters.

    Saves both the model weights and architecture hyperparameters so the
    model can be fully reconstructed from the checkpoint.

    Args:
        model: BRIDGEModel to save.
        path: Path to save the .pt checkpoint file.
    """
    torch.save({
        "state_dict": model.state_dict(),
        "hp_dict": model.get_hp_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, hp_dict: dict[str, Any] | None = None) -> BRIDGEModel:
    """Load model from checkpoint.

    Args:
        path: Path to .pt checkpoint file.
        hp_dict: Hyperparameters dictionary. If None, uses hyperparameters
            stored in the checkpoint.

    Returns:
        BRIDGEModel loaded with weights and set to eval mode.

    Raises:
        ValueError: If hp_dict is not found in checkpoint and not provided.
    """
    checkpoint = torch.load(path, map_location=DEVICE)

    if hp_dict is None:
        hp_dict = checkpoint.get("hp_dict")
        if hp_dict is None:
            raise ValueError("hp_dict not found in checkpoint and not provided")

    model = BRIDGEModel.from_hp_dict(hp_dict)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded from {path}")
    return model
