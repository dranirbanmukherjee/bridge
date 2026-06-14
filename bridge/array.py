"""
3D input array builder for BRIDGE pipeline.

Builds the 3D input arrays required for contrastive learning:
- Position 0: anchor embedding
- Positions 1..N-1: negative samples (different attribute values)

Optimized implementation pre-computes negative pools per unique attribute
combination, reducing complexity from O(n²) to O(n × k) where k is the
number of unique attribute combinations.
"""


import numpy as np
from tqdm import tqdm

DEFAULT_MINI_BATCH_SIZE = 10
DEFAULT_SEED = 42


def identify_valid_embeddings(embedding_matrix: np.ndarray) -> np.ndarray:
    """Identify indices of rows with valid (non-NaN) embeddings.

    Args:
        embedding_matrix: Embedding matrix which may contain NaN values
            from failed API calls.

    Returns:
        1D array of row indices where all values are non-NaN.
    """
    valid_mask = ~np.any(np.isnan(embedding_matrix), axis=1)
    return np.where(valid_mask)[0]


def _compute_attribute_keys(
    labels: dict[str, np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """Compute a unique key tuple for each sample based on its attribute values.

    Args:
        labels: Dictionary mapping attribute names to label arrays.
        n_samples: Number of samples.

    Returns:
        Array of tuples, one per sample, containing attribute values.
    """
    attr_names = sorted(labels.keys())
    # Create array of tuples for efficient lookup
    keys = np.empty(n_samples, dtype=object)
    for i in range(n_samples):
        keys[i] = tuple(labels[attr][i] for attr in attr_names)
    return keys


def _precompute_negative_pools(
    labels: dict[str, np.ndarray],
    valid_indices: np.ndarray,
    attr_keys: np.ndarray,
    verbose: bool = True,
) -> dict[tuple, np.ndarray]:
    """Pre-compute valid negative indices for each unique attribute combination.

    This is the key optimization: instead of computing negatives per-sample O(n²),
    we compute per unique attribute combination O(k × n) where k << n.

    Args:
        labels: Dictionary mapping attribute names to label arrays.
        valid_indices: Indices of samples with valid embeddings.
        attr_keys: Array of attribute key tuples for each sample.
        verbose: Whether to print progress info.

    Returns:
        Dictionary mapping attribute key tuples to arrays of valid negative indices.
    """
    attr_names = sorted(labels.keys())
    n_samples = len(attr_keys)

    # Get unique keys from valid samples only
    unique_keys = set(attr_keys[valid_indices])

    if verbose:
        print(f"  Pre-computing negative pools for {len(unique_keys)} "
              "unique attribute combinations...")

    key_to_negatives = {}

    for key in tqdm(unique_keys, desc="Pre-computing negatives", disable=not verbose):
        # Negatives: samples where ALL attributes differ from this key
        neg_mask = np.ones(n_samples, dtype=bool)
        for attr_idx, attr_name in enumerate(attr_names):
            neg_mask &= (labels[attr_name] != key[attr_idx])

        # Intersect with valid indices
        candidate_indices = np.where(neg_mask)[0]
        key_to_negatives[key] = np.intersect1d(candidate_indices, valid_indices)

    return key_to_negatives


def build_3d_array(
    embedding_matrix: np.ndarray,
    labels: dict[str, np.ndarray],
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    show_progress: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Build 3D input array for BRIDGE training.

    Constructs the contrastive learning input array where each sample is
    paired with negative examples that differ on all attributes.

    This optimized implementation pre-computes negative pools per unique
    attribute combination, reducing complexity from O(n²) to O(n × k).

    Structure:
        - Dimension 0: observations (anchors)
        - Dimension 1: mini-batch (position 0 = anchor, 1..N-1 = negatives)
        - Dimension 2: embedding dimensions

    Args:
        embedding_matrix: Full embedding matrix of shape (n_samples, embedding_dim).
        labels: Dictionary mapping attribute names to label arrays of shape (n_samples,).
        mini_batch_size: Items per mini-batch (1 anchor + N-1 negatives).
        seed: Random seed for reproducible negative sampling.
        show_progress: Whether to show a progress bar.
        verbose: Whether to print statistics.

    Returns:
        3D array of shape (n_samples, mini_batch_size, embedding_dim).
    """
    if verbose:
        print(f"Building 3D array (seed={seed}, mini_batch={mini_batch_size})...")

    rng = np.random.default_rng(seed)

    n_samples, embedding_dim = embedding_matrix.shape
    num_negatives = mini_batch_size - 1

    if verbose:
        print(f"  Samples: {n_samples}, Embedding dim: {embedding_dim}")
        print(f"  Mini-batch: {mini_batch_size} (1 anchor + {num_negatives} negatives)")

    # Find valid embeddings
    valid_indices = identify_valid_embeddings(embedding_matrix)

    if verbose:
        print(f"  Valid embeddings: {len(valid_indices)}")

    # Compute attribute keys for each sample
    attr_keys = _compute_attribute_keys(labels, n_samples)

    # Pre-compute negative pools (the key optimization)
    key_to_negatives = _precompute_negative_pools(
        labels, valid_indices, attr_keys, verbose=verbose
    )

    # Initialize array
    input_array = np.zeros((n_samples, mini_batch_size, embedding_dim), dtype=np.float32)

    # Counters
    invalid_anchors = 0
    no_negatives = 0

    if verbose:
        print("  Building array with pre-computed pools...")

    iterator = valid_indices  # Only iterate over valid samples
    if show_progress:
        iterator = tqdm(iterator, desc="Building 3D array")

    for i in iterator:
        # Set anchor at position 0
        input_array[i, 0, :] = embedding_matrix[i]

        # Look up pre-computed negatives (O(1) lookup!)
        valid_negs = key_to_negatives[attr_keys[i]]

        if len(valid_negs) > 0:
            replace = len(valid_negs) < num_negatives
            neg_indices = rng.choice(valid_negs, size=num_negatives, replace=replace)
            input_array[i, 1:mini_batch_size, :] = embedding_matrix[neg_indices]
        else:
            # No valid negatives: fill with anchor (rare edge case)
            no_negatives += 1
            input_array[i, 1:mini_batch_size, :] = embedding_matrix[i]

    # Count invalid anchors (samples we skipped)
    invalid_anchors = n_samples - len(valid_indices)

    if verbose:
        if invalid_anchors > 0:
            print(f"  Warning: {invalid_anchors} invalid anchors (zeroed)")
        if no_negatives > 0:
            print(f"  Warning: {no_negatives} samples with no valid negatives")
        print(f"3D array built. Shape: {input_array.shape}")

    return input_array


def validate_3d_array(
    input_array: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Validate 3D input array and compute statistics.

    Checks for common issues like zero anchors and NaN values.

    Args:
        input_array: 3D array to validate.
        verbose: Whether to print validation results.

    Returns:
        Dictionary containing validation statistics including shape,
        number of valid/zero anchors, NaN count, and memory usage.
    """
    n_samples, mini_batch_size, embedding_dim = input_array.shape

    zero_anchors = np.sum(np.all(input_array[:, 0, :] == 0.0, axis=1))
    nan_count = np.sum(np.isnan(input_array))

    stats = {
        "shape": input_array.shape,
        "n_samples": n_samples,
        "mini_batch_size": mini_batch_size,
        "embedding_dim": embedding_dim,
        "zero_anchors": int(zero_anchors),
        "valid_anchors": n_samples - int(zero_anchors),
        "nan_count": int(nan_count),
        "memory_mb": input_array.nbytes / (1024 * 1024),
    }

    if verbose:
        print("\n--- 3D Array Validation ---")
        print(f"Shape: {stats['shape']}")
        print(f"Valid anchors: {stats['valid_anchors']} / {stats['n_samples']}")
        print(f"Memory: {stats['memory_mb']:.2f} MB")

    return stats


def save_3d_array(array: np.ndarray, path: str) -> None:
    """Save 3D array to numpy file.

    Args:
        array: 3D numpy array to save.
        path: Path for the output .npy file.
    """
    np.save(path, array)
    print(f"3D array saved to {path}")


def load_3d_array(path: str) -> np.ndarray:
    """Load 3D array from numpy file.

    Args:
        path: Path to the .npy file.

    Returns:
        3D numpy array.

    Raises:
        ValueError: If the loaded array is not 3-dimensional.
    """
    arr = np.load(path)
    if len(arr.shape) != 3:
        raise ValueError(f"Loaded array is not 3D: shape {arr.shape}")
    return arr


# Keep old function for reference/testing (not exported)
def _find_valid_negatives_slow(
    anchor_idx: int,
    labels: dict[str, np.ndarray],
    valid_indices: np.ndarray,
) -> np.ndarray:
    """Find valid negatives for an anchor using brute-force filtering.

    Original O(n) per-sample implementation kept for reference and testing.
    The optimized version pre-computes pools per unique attribute combination.

    Args:
        anchor_idx: Index of the anchor sample in the dataset.
        labels: Dictionary mapping attribute names to label arrays of shape (n,).
        valid_indices: Array of indices with valid (non-NaN) embeddings.

    Returns:
        Array of valid negative indices that differ from the anchor on ALL attributes.
    """
    different_mask = np.ones(len(list(labels.values())[0]), dtype=bool)
    for _attr_name, attr_labels in labels.items():
        anchor_value = attr_labels[anchor_idx]
        different_mask = different_mask & (attr_labels != anchor_value)
    candidate_indices = np.where(different_mask)[0]
    return np.intersect1d(candidate_indices, valid_indices)
