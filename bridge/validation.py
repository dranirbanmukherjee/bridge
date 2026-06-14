"""
Classification metrics for validating BRIDGE model performance.

Provides utilities for computing and reporting classification metrics
(accuracy, F1 scores, top-k accuracy) for each focal attribute after
model training.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bridge.model import DEVICE, BRIDGEModel


def compute_classification_metrics(
    model: BRIDGEModel,
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    batch_size: int = 64,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute classification metrics for focal attributes.

    Runs inference on the input embeddings and computes classification
    metrics comparing predicted classes to true labels for each attribute.

    Args:
        model: Trained BRIDGEModel.
        embeddings: 3D input array of shape (n_samples, mini_batch, embedding_dim).
        labels: Dictionary mapping attribute names to label arrays of shape (n_samples,).
        batch_size: Batch size for inference.
        device: Device to use. If None, auto-detects best device.
        show_progress: Whether to show a progress bar.

    Returns:
        Dictionary mapping attribute names to metric dictionaries:
        {
            "region": {
                "accuracy": 0.85,
                "macro_f1": 0.72,
                "weighted_f1": 0.83,
                "top_5_accuracy": 0.95,
                "num_classes": 426,
                "num_samples": 119955
            },
            "varietal": {...}
        }
    """
    if device is None:
        device = DEVICE

    model = model.to(device)
    model.eval()

    n_samples = embeddings.shape[0]
    attribute_names = model.attribute_names

    # Initialize storage for predictions and true labels
    all_preds = {name: [] for name in attribute_names}
    all_logits = {name: [] for name in attribute_names}

    # Process in batches
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing predictions")

    with torch.no_grad():
        for i in iterator:
            batch = embeddings[i:i + batch_size]
            x = torch.tensor(batch, dtype=torch.float32, device=device)

            # Get model outputs (classification logits)
            outputs = model(x)

            for name in attribute_names:
                logits = outputs[name]
                preds = logits.argmax(dim=1)
                all_preds[name].append(preds.cpu().numpy())
                all_logits[name].append(logits.cpu().numpy())

    # Concatenate all predictions
    all_preds = {name: np.concatenate(preds) for name, preds in all_preds.items()}
    all_logits = {name: np.concatenate(logits) for name, logits in all_logits.items()}

    # Compute metrics for each attribute
    metrics = {}
    for name in attribute_names:
        true_labels = labels[name]
        pred_labels = all_preds[name]
        logits = all_logits[name]
        num_classes = model.attribute_sizes[name]

        metrics[name] = _compute_attribute_metrics(
            true_labels=true_labels,
            pred_labels=pred_labels,
            logits=logits,
            num_classes=num_classes,
        )

    return metrics


def _compute_attribute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Compute classification metrics for a single attribute.

    Args:
        true_labels: Ground truth labels of shape (n_samples,).
        pred_labels: Predicted labels of shape (n_samples,).
        logits: Raw logits of shape (n_samples, num_classes).
        num_classes: Total number of classes.

    Returns:
        Dictionary of metrics.
    """
    n_samples = len(true_labels)

    # Accuracy
    correct = pred_labels == true_labels
    accuracy = np.mean(correct)

    # Per-class precision, recall, F1
    # Using vectorized operations for efficiency
    precisions = []
    recalls = []
    f1_scores = []
    class_counts = []

    for c in range(num_classes):
        true_c = true_labels == c
        pred_c = pred_labels == c

        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)

        # Handle edge cases
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        class_counts.append(np.sum(true_c))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    class_counts = np.array(class_counts)

    # Macro F1 (unweighted average across classes)
    # Only average over classes that appear in the data
    valid_classes = class_counts > 0
    macro_f1 = np.mean(f1_scores[valid_classes]) if np.any(valid_classes) else 0.0

    # Weighted F1 (weighted by class support)
    total_support = np.sum(class_counts)
    if total_support > 0:
        weighted_f1 = np.sum(f1_scores * class_counts) / total_support
    else:
        weighted_f1 = 0.0

    # Top-5 accuracy (for attributes with many classes)
    if num_classes > 5:
        # Get top 5 predictions per sample
        top_5_preds = np.argsort(logits, axis=1)[:, -5:]
        top_5_correct = np.any(top_5_preds == true_labels[:, np.newaxis], axis=1)
        top_5_accuracy = np.mean(top_5_correct)
    else:
        # For attributes with <= 5 classes, top-5 = top-1
        top_5_accuracy = accuracy

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top_5_accuracy": float(top_5_accuracy),
        "num_classes": int(num_classes),
        "num_classes_in_data": int(np.sum(valid_classes)),
        "num_samples": int(n_samples),
    }


def print_classification_metrics(
    metrics: dict[str, dict[str, float]],
    title: str = "Classification Metrics",
) -> None:
    """Print classification metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics as returned by compute_classification_metrics.
        title: Title for the metrics report.
    """
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

    # Header
    print(f"\n{'Attribute':<15} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Weighted F1':>12} {'Top-5 Acc':>10} {'Classes':>10}")
    print("-" * 70)

    # Per-attribute metrics
    for attr_name, attr_metrics in metrics.items():
        print(
            f"{attr_name:<15} "
            f"{attr_metrics['accuracy']:>10.4f} "
            f"{attr_metrics['macro_f1']:>10.4f} "
            f"{attr_metrics['weighted_f1']:>12.4f} "
            f"{attr_metrics['top_5_accuracy']:>10.4f} "
            f"{attr_metrics['num_classes_in_data']:>4}/{attr_metrics['num_classes']:<5}"
        )

    print("-" * 70)

    # Summary statistics
    avg_accuracy = np.mean([m['accuracy'] for m in metrics.values()])
    avg_macro_f1 = np.mean([m['macro_f1'] for m in metrics.values()])
    avg_weighted_f1 = np.mean([m['weighted_f1'] for m in metrics.values()])
    avg_top5 = np.mean([m['top_5_accuracy'] for m in metrics.values()])

    print(f"{'Average':<15} {avg_accuracy:>10.4f} {avg_macro_f1:>10.4f} "
          f"{avg_weighted_f1:>12.4f} {avg_top5:>10.4f}")
    print("=" * 70 + "\n")


def save_classification_metrics(
    metrics: dict[str, dict[str, float]],
    output_path: str,
) -> str:
    """Save classification metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics as returned by compute_classification_metrics.
        output_path: Path to save the JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return str(output_path)


def load_classification_metrics(input_path: str) -> dict[str, dict[str, float]]:
    """Load classification metrics from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Dictionary of metrics.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
