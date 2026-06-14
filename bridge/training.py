"""
Training module for BRIDGE pipeline.

Provides training loop, hyperparameter tuning (via Optuna),
early stopping, and learning rate scheduling.
"""

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

from bridge.model import DEVICE, BRIDGEModel, save_model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BRIDGEDataset(Dataset):
    """PyTorch Dataset for BRIDGE training.

    Wraps the 3D embedding array and label dictionaries for use with
    PyTorch DataLoader.

    Attributes:
        embeddings: 3D tensor of shape (n_samples, mini_batch, embedding_dim).
        labels: Dictionary mapping attribute names to label tensors.
        attribute_names: List of attribute names.
        device: Device where tensors are stored.
        _contrastive_dummy: Pre-allocated dummy tensor for contrastive loss target.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: dict[str, np.ndarray],
        device: torch.device | None = None,
    ):
        """Initialize the dataset.

        Args:
            embeddings: 3D array of shape (n_samples, mini_batch, embedding_dim).
            labels: Dict mapping attribute names to label arrays of shape (n_samples,).
            device: Device to store tensors on. If None, uses CPU. For systems with
                large unified memory (e.g., M2 Ultra with 128GB), pass the MPS device
                to eliminate per-batch CPU-GPU transfers.
        """
        self.device = device
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = {
            name: torch.tensor(arr, dtype=torch.long)
            for name, arr in labels.items()
        }
        self.attribute_names = list(labels.keys())

        # Move to device if specified (eliminates per-batch transfers)
        if device is not None:
            self.embeddings = self.embeddings.to(device)
            self.labels = {name: arr.to(device) for name, arr in self.labels.items()}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x, y) where x is the embedding tensor of shape
            (mini_batch, embedding_dim) and y is a dict of target labels.
        """
        x = self.embeddings[idx]
        y = {name: self.labels[name][idx] for name in self.attribute_names}
        return x, y

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get a batch of samples by indices (zero-copy for on-device data).

        This method uses batched indexing which is more efficient than
        stacking individual samples from __getitem__.

        Args:
            indices: 1D tensor of sample indices.

        Returns:
            Tuple of (x, y) where x has shape (batch_size, mini_batch, embedding_dim)
            and y is a dict of batched target labels.
        """
        x = self.embeddings[indices]
        y = {name: self.labels[name][indices] for name in self.attribute_names}
        return x, y


class BatchedDataLoader:
    """Data loader using batched indexing for on-device datasets.

    For datasets with data pre-loaded on device (e.g., MPS with unified memory),
    this loader:
    1. Uses batched indexing (dataset.embeddings[indices]) instead of per-sample access
    2. Skips the collate step
    3. Pre-computes shuffled indices for the entire epoch

    Example:
        dataset = BRIDGEDataset(embeddings, labels, device=torch.device('mps'))
        loader = BatchedDataLoader(dataset, batch_size=8, shuffle=True)
        for x, targets in loader:
            outputs = model(x)
    """

    def __init__(
        self,
        dataset: BRIDGEDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
    ):
        """Initialize the batched data loader.

        Args:
            dataset: BRIDGEDataset instance (should have data on device).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle indices each epoch.
            drop_last: Whether to drop the last incomplete batch.
            seed: Random seed for shuffling (for reproducibility).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(dataset)
        self.device = dataset.device

        # Use numpy RNG for shuffling
        self.rng = np.random.default_rng(seed)

        # Pre-allocate index tensor on device
        self._indices = torch.arange(self.n_samples, device=self.device)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches for one epoch."""
        # Shuffle indices
        if self.shuffle:
            perm = self.rng.permutation(self.n_samples)
            indices = self._indices[torch.from_numpy(perm).to(self.device)]
        else:
            indices = self._indices

        # Yield batches using batched indexing
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            yield self.dataset.get_batch(batch_indices)


def prepare_data_loaders(
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    batch_size: int = 8,
    validation_split: float = 0.1,
    seed: int = 42,
    filter_zero_anchors: bool = True,
    device: torch.device | None = None,
) -> tuple[DataLoader | "BatchedDataLoader", DataLoader | "BatchedDataLoader"]:
    """Create train/validation DataLoaders.

    Splits data into training and validation sets, optionally filtering out
    samples with invalid (zero) anchor embeddings.

    When device is specified, uses BatchedDataLoader for zero-copy batching
    on unified memory systems (e.g., M2 Ultra). Otherwise uses standard
    PyTorch DataLoader.

    Args:
        embeddings: 3D input array of shape (n_samples, mini_batch, embedding_dim).
        labels: Dictionary mapping attribute names to label arrays.
        batch_size: Batch size for training.
        validation_split: Fraction of data to use for validation.
        seed: Random seed for reproducible splitting.
        filter_zero_anchors: Whether to remove samples with zero anchor embeddings.
        device: Device to pre-load data onto. For systems with large unified memory
            (e.g., M2 Ultra with 128GB), pass the MPS device to eliminate per-batch
            CPU-GPU transfers and use BatchedDataLoader. If None, uses standard DataLoader.

    Returns:
        Tuple of (train_loader, val_loader). Type depends on device parameter:
        - device=None: Standard PyTorch DataLoader instances
        - device specified: BatchedDataLoader instances
    """
    # Filter zero anchors
    if filter_zero_anchors:
        valid_mask = ~np.all(embeddings[:, 0, :] == 0.0, axis=1)
        embeddings = embeddings[valid_mask]
        labels = {name: arr[valid_mask] for name, arr in labels.items()}

    # Split indices
    n_samples = len(embeddings)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split data before creating datasets
    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]
    train_labels = {name: arr[train_indices] for name, arr in labels.items()}
    val_labels = {name: arr[val_indices] for name, arr in labels.items()}

    # Create datasets
    train_dataset = BRIDGEDataset(train_embeddings, train_labels, device=device)
    val_dataset = BRIDGEDataset(val_embeddings, val_labels, device=device)

    # Use BatchedDataLoader for on-device data (zero-copy batching)
    if device is not None:
        train_loader = BatchedDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        val_loader = BatchedDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )
    else:
        # Standard DataLoader for CPU data
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    attribute_names: list[str],
    contrastive_weight: float = 0.1,
    return_tensors: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined classification and contrastive loss.

    Args:
        outputs: Model outputs containing logits per attribute and contrastive loss.
        targets: Target labels dictionary.
        attribute_names: List of attribute names to compute losses for.
        contrastive_weight: Weight for the contrastive loss term.
        return_tensors: If True, return loss tensors instead of floats in the dict.
            This avoids GPU-CPU synchronization per batch. Use True during training
            and accumulate losses, then call .item() once at epoch end.

    Returns:
        Tuple of (total_loss, losses_dict) where total_loss is a tensor for
        backpropagation and losses_dict contains individual loss values.
        If return_tensors=False (default), values are floats.
        If return_tensors=True, values are detached tensors (no sync penalty).
    """
    losses = {}
    total_loss = 0.0

    # Classification losses (one per attribute)
    for name in attribute_names:
        loss = F.cross_entropy(outputs[name], targets[name])
        losses[f"{name}_loss"] = loss.detach() if return_tensors else loss.item()
        total_loss = total_loss + loss

    # Contrastive loss
    contrastive_loss = outputs['contrastive'].mean()
    losses['contrastive_loss'] = (
        contrastive_loss.detach() if return_tensors else contrastive_loss.item()
    )
    total_loss = total_loss + contrastive_weight * contrastive_loss

    losses['total_loss'] = total_loss.detach() if return_tensors else total_loss.item()
    return total_loss, losses


def train_epoch(
    model: BRIDGEModel,
    train_loader: DataLoader | "BatchedDataLoader",
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Optimized for systems with unified memory (e.g., Apple Silicon with MPS).
    If the dataset is pre-loaded on device, skips per-batch transfers.
    Uses running sums to avoid storing tensors and minimize memory overhead.

    Args:
        model: BRIDGEModel to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Device to train on.

    Returns:
        Dictionary of mean loss values for the epoch.
    """
    model.train()
    loss_keys = (['total_loss', 'contrastive_loss']
                 + [f"{name}_loss" for name in model.attribute_names])
    running_sums = {key: torch.tensor(0.0, device=device) for key in loss_keys}
    n_batches = 0

    for x, targets in train_loader:
        # Skip .to() if already on device (pre-loaded dataset)
        if x.device != device:
            x = x.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss, losses = compute_loss(
            outputs, targets, model.attribute_names, model.contrastive_weight,
            return_tensors=True,  # Avoid per-batch GPU-CPU sync
        )
        loss.backward()
        optimizer.step()

        for key, val in losses.items():
            running_sums[key] = running_sums[key] + val
        n_batches += 1

    # Single GPU-CPU sync at epoch end
    return {k: (v / n_batches).item() for k, v in running_sums.items()}


@torch.no_grad()
def validate(
    model: BRIDGEModel,
    val_loader: DataLoader | "BatchedDataLoader",
    device: torch.device,
) -> dict[str, float]:
    """Validate model and compute metrics.

    Optimized for systems with unified memory (e.g., Apple Silicon with MPS).
    Uses running sums to minimize memory overhead and GPU-CPU synchronization.

    Args:
        model: BRIDGEModel to validate.
        val_loader: DataLoader for validation data.
        device: Device to run validation on.

    Returns:
        Dictionary containing validation losses and per-attribute accuracies.
    """
    model.eval()
    loss_keys = (['total_loss', 'contrastive_loss']
                 + [f"{name}_loss" for name in model.attribute_names])
    running_sums = {key: torch.tensor(0.0, device=device) for key in loss_keys}
    correct_sums = {name: torch.tensor(0, device=device) for name in model.attribute_names}
    total_samples = {name: 0 for name in model.attribute_names}
    n_batches = 0

    for x, targets in val_loader:
        # Skip .to() if already on device (pre-loaded dataset)
        if x.device != device:
            x = x.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        outputs = model(x)
        _, losses = compute_loss(
            outputs, targets, model.attribute_names, model.contrastive_weight,
            return_tensors=True,  # Avoid per-batch GPU-CPU sync
        )

        for key, val in losses.items():
            running_sums[key] = running_sums[key] + val
        n_batches += 1

        # Accuracies - running sum
        for name in model.attribute_names:
            preds = outputs[name].argmax(dim=1)
            correct_sums[name] = correct_sums[name] + (preds == targets[name]).sum()
            total_samples[name] += len(preds)

    # Single GPU-CPU sync at epoch end
    metrics = {f"val_{k}": (v / n_batches).item() for k, v in running_sums.items()}
    for name in model.attribute_names:
        metrics[f"val_{name}_acc"] = correct_sums[name].item() / max(total_samples[name], 1)

    return metrics


def train_model(
    model: BRIDGEModel,
    train_loader: DataLoader | "BatchedDataLoader",
    val_loader: DataLoader | "BatchedDataLoader",
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 10,
    lr_reduce_patience: int = 2,
    lr_reduce_factor: float = 0.9,
    min_lr: float = 1e-6,
    start_from_epoch: int = 3,
    checkpoint_path: str | None = None,
    verbose: int = 1,
    device: torch.device | None = None,
) -> tuple[BRIDGEModel, dict[str, list[float]]]:
    """Train BRIDGE model with early stopping and LR scheduling.

    Implements a complete training loop with AdamW optimizer (decoupled weight
    decay), learning rate reduction on plateau, and early stopping to prevent
    overfitting. AdamW is preferred over Adam for contrastive learning as it
    provides more consistent regularization of the embedding space.

    Args:
        model: BRIDGEModel to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum number of epochs.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: Decoupled weight decay coefficient. Unlike L2 regularization
            in Adam, AdamW applies decay directly to weights independent of
            gradient magnitude. Typical range for contrastive learning: 1e-4 to 1e-1.
        early_stopping_patience: Epochs without improvement before stopping.
        lr_reduce_patience: Epochs without improvement before reducing LR.
        lr_reduce_factor: Factor to multiply LR by when reducing.
        min_lr: Minimum learning rate floor.
        start_from_epoch: Epoch after which to start early stopping checks.
        checkpoint_path: Path to save best model checkpoint. If None, no
            checkpoint is saved.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        device: Device to train on. If None, auto-detects best device.

    Returns:
        Tuple of (model, history) where model has best weights restored and
        history is a dictionary of training metrics per epoch.
    """
    if device is None:
        device = DEVICE

    model = model.to(device)

    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception:  # pylint: disable=broad-exception-caught  # any compile failure falls back to eager mode
            pass  # Fall back to eager mode if compile fails

    # Use AdamW with decoupled weight decay (preferred for contrastive learning)
    try:
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
        )
    except (TypeError, RuntimeError):
        # Fall back if fused not supported (older PyTorch or CPU)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=min_lr,
    )

    history = {key: [] for key in [
        'train_loss', 'val_loss', 'learning_rate'
    ] + [f"val_{name}_acc" for name in model.attribute_names]}

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        # Record history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['val_total_loss'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        for name in model.attribute_names:
            history[f"val_{name}_acc"].append(val_metrics[f"val_{name}_acc"])

        # LR scheduling
        scheduler.step(val_metrics['val_total_loss'])

        # Early stopping
        if epoch >= start_from_epoch:
            if val_metrics['val_total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_total_loss']
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0

                if checkpoint_path:
                    save_model(model, checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            # Before start_from_epoch, always save best
            if val_metrics['val_total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_total_loss']
                best_state = copy.deepcopy(model.state_dict())

        if verbose >= 1:
            acc_str = ", ".join([
                f"{name}={val_metrics[f'val_{name}_acc']:.3f}"
                for name in model.attribute_names
            ])
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"loss: {train_metrics['total_loss']:.4f}, "
                  f"val_loss: {val_metrics['val_total_loss']:.4f}, "
                  f"acc: [{acc_str}]")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    embeddings: np.ndarray,
    labels: dict[str, np.ndarray],
    attribute_sizes: dict[str, int],
    n_trials: int = 75,
    epochs_per_trial: int = 50,
    batch_size: int = 8,
    startup_trials: int = 16,
    seed: int = 42,
    verbose: int = 1,
    device: torch.device | None = None,
    fixed_mask_size: int | None = None,
) -> dict[str, Any]:
    """Tune hyperparameters using Optuna.

    Performs Bayesian optimization over model architecture and learning rate
    hyperparameters using the Tree-structured Parzen Estimator (TPE) sampler.

    Args:
        embeddings: 3D input array of shape (n_samples, mini_batch, embedding_dim).
        labels: Dictionary mapping attribute names to label arrays.
        attribute_sizes: Dictionary mapping attribute names to number of classes.
        n_trials: Number of Optuna optimization trials.
        epochs_per_trial: Maximum epochs per trial (with early stopping).
        batch_size: Batch size for training.
        startup_trials: Number of random trials before starting TPE.
        seed: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress).
        device: Device to pre-load data onto. If specified, uses BatchedDataLoader.
        fixed_mask_size: If set, locks mask_size to this value instead of searching.
            Use for non-Matryoshka embeddings (e.g., BERT) where dimension truncation
            is not meaningful.

    Returns:
        Dictionary of best hyperparameters including architecture parameters
        and learning rate.

    Raises:
        ImportError: If optuna package is not installed.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna required for tuning. Install via: pip install optuna")

    attribute_names = list(labels.keys())
    orig_size = embeddings.shape[2]
    mini_batch_size = embeddings.shape[1]

    # Pre-load data on device if specified
    train_loader, val_loader = prepare_data_loaders(
        embeddings, labels, batch_size=batch_size, seed=seed, device=device
    )

    # Determine mask_size search space based on embedding dimension
    # For Matryoshka embeddings (OpenAI): search over truncation levels
    # For non-Matryoshka (BERT, etc.): use fixed_mask_size to lock at full dim
    if fixed_mask_size is not None:
        mask_size_choices = [fixed_mask_size]
    else:
        # For OpenAI (3072): [128, 256, 512, 1024, 2048, 3072]
        # For gemma (768): [128, 256, 512, 768]
        # General rule: powers of 2 up to orig_size, plus orig_size itself
        mask_size_choices = [s for s in [128, 256, 512, 1024, 2048, 3072] if s <= orig_size]
        if orig_size not in mask_size_choices:
            mask_size_choices.append(orig_size)
        mask_size_choices = sorted(mask_size_choices)

    if verbose >= 1:
        print(f"  Embedding dimension: {orig_size}")
        if fixed_mask_size is not None:
            print(f"  mask_size: {fixed_mask_size} (fixed, non-Matryoshka embedding)")
        else:
            print(f"  mask_size search space: {mask_size_choices}")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object for suggesting hyperparameters.

        Returns:
            Validation loss (lower is better).
        """
        # Sample hyperparameters
        projection_units = trial.suggest_int("projection_units", 64, 128, step=8)
        embedding_units = trial.suggest_int("embedding_units_per_attribute", 4, 16, step=2)
        mask_size = trial.suggest_categorical("mask_size", mask_size_choices)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

        # Create model
        model = BRIDGEModel(
            attribute_names=attribute_names,
            attribute_sizes=attribute_sizes,
            orig_size=orig_size,
            mini_batch_size=mini_batch_size,
            projection_units=projection_units,
            embedding_units_per_attribute=embedding_units,
            mask_size=mask_size,
        )

        # Train (compilation happens in train_model)
        # Use the device parameter passed to tune_hyperparameters
        train_device = device if device is not None else DEVICE
        model, _ = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs_per_trial,
            learning_rate=lr,
            weight_decay=wd,
            verbose=0,
            device=train_device,
        )

        # Evaluate
        val_metrics = validate(model, val_loader, train_device)
        return val_metrics['val_total_loss']

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Callback to log trial results clearly (avoids progress bar overwriting)
    def log_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Log trial results with hyperparameters."""
        params = trial.params
        best_marker = " *BEST*" if trial.value == study.best_value else ""
        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{n_trials} completed{best_marker}")
        print(f"  Value: {trial.value:.4f} (best so far: {study.best_value:.4f})")
        print("  Hyperparameters:")
        print(f"    projection_units: {params.get('projection_units')}")
        print(f"    embedding_units_per_attribute: {params.get('embedding_units_per_attribute')}")
        print(f"    mask_size: {params.get('mask_size')}")
        print(f"    learning_rate: {params.get('learning_rate'):.6f}")
        print(f"    weight_decay: {params.get('weight_decay'):.6f}")
        if study.best_trial.number != trial.number:
            print(f"  Best trial: {study.best_trial.number + 1}")
        print(f"{'='*70}\n", flush=True)

    if verbose >= 1:
        print(f"Starting hyperparameter tuning ({n_trials} trials)...")
        print("  Search space:")
        print("    projection_units: 64-128 (step 8)")
        print("    embedding_units_per_attribute: 4-16 (step 2)")
        print(f"    mask_size: {mask_size_choices}")
        print("    learning_rate: 1e-4 to 1e-2 (log scale)")
        print("    weight_decay: 1e-4 to 1e-1 (log scale)")
        print()

    # Disable progress bar to avoid overwriting logs; use callback instead
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[log_trial_callback] if verbose >= 1 else None,
    )

    best_params = study.best_params
    best_params.update({
        "attribute_names": attribute_names,
        "attribute_sizes": attribute_sizes,
        "orig_size": orig_size,
        "mini_batch_size": mini_batch_size,
        "contrastive_temperature": 0.1,
        "contrastive_weight": 0.1,
    })

    if verbose >= 1:
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("=" * 70)
        print(f"Best trial: {study.best_trial.number + 1}/{n_trials}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        print(f"  projection_units: {best_params.get('projection_units')}")
        emb_units = best_params.get('embedding_units_per_attribute')
        print(f"  embedding_units_per_attribute: {emb_units}")
        print(f"  mask_size: {best_params.get('mask_size')}")
        print(f"  learning_rate: {best_params.get('learning_rate'):.6f}")
        print(f"  weight_decay: {best_params.get('weight_decay'):.6f}")
        print("=" * 70 + "\n")

    return best_params
