"""
Representation extraction and nuisance controls for BRIDGE pipeline.

Provides utilities for:
- Extracting attribute-specific embeddings from trained models
- Computing nuisance controls via orthogonal projection (SVD or UMAP)
"""

import warnings
from typing import Any, Literal

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from bridge.model import DEVICE, BRIDGEModel

# Default number of nuisance dimensions (determined by elbow plot analysis)
DEFAULT_NUISANCE_DIMS = 5


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def extract_representations(
    model: BRIDGEModel,
    input_array: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Extract attribute-specific embeddings from trained model.

    Runs inference on the input array to extract the learned attribute-specific
    representations for each sample.

    Args:
        model: Trained BRIDGEModel.
        input_array: 3D input array of shape (n_samples, mini_batch, embedding_dim).
        batch_size: Batch size for inference.
        device: Device to use. If None, auto-detects best device.
        show_progress: Whether to show a progress bar.

    Returns:
        Dictionary mapping attribute names to embedding arrays of shape
        (n_samples, emb_dim). Only anchor embeddings (position 0) are returned.
    """
    if device is None:
        device = DEVICE

    model = model.to(device)
    model.eval()

    n_samples = input_array.shape[0]
    attribute_names = model.attribute_names
    emb_dim = model.embedding_units_per_attribute

    # Initialize output arrays
    embeddings = {name: np.zeros((n_samples, emb_dim), dtype=np.float32)
                  for name in attribute_names}

    # Process in batches
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting representations")

    with torch.no_grad():
        for i in iterator:
            batch = input_array[i:i + batch_size]
            x = torch.tensor(batch, dtype=torch.float32, device=device)

            # Get anchor embeddings
            anchor_embs = model.get_anchor_embeddings(x)

            for name in attribute_names:
                embeddings[name][i:i + len(batch)] = anchor_embs[name].cpu().numpy()

    return embeddings


# ---------------------------------------------------------------------------
# Nuisance controls
# ---------------------------------------------------------------------------

def identify_valid_rows(
    embeddings: dict[str, np.ndarray],
    full_embedding: np.ndarray,
) -> np.ndarray:
    """Identify rows valid in both attribute embeddings and full embeddings.

    A row is valid if it contains no NaN values in either the attribute
    embeddings or the full embedding.

    Args:
        embeddings: Dictionary of attribute embeddings.
        full_embedding: Full dimensionality embedding matrix.

    Returns:
        Boolean mask array where True indicates valid rows.
    """
    # Check full embedding
    valid_full = ~np.any(np.isnan(full_embedding), axis=1)

    # Check all attribute embeddings
    valid_attrs = np.ones(len(full_embedding), dtype=bool)
    for emb in embeddings.values():
        valid_attrs = valid_attrs & ~np.any(np.isnan(emb), axis=1)

    return valid_full & valid_attrs


def compute_projection_residuals(
    interpretable_embedding: np.ndarray,
    full_embedding: np.ndarray,
    valid_mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute residuals by projecting full embeddings onto interpretable space.

    The orthogonal projection removes the component of the full embedding
    that can be explained by the interpretable representations, leaving
    the nuisance component (style, tone, etc.).

    Args:
        interpretable_embedding: Combined interpretable embedding of shape (n, d_inter).
        full_embedding: Full dimensionality embedding of shape (n, d_full).
        valid_mask: Boolean mask for valid rows.
        verbose: Whether to print progress messages.

    Returns:
        Residual matrix of shape (n, d_full) with NaN for invalid rows.

    Raises:
        ValueError: If too few valid rows for projection.
    """
    n_valid = np.sum(valid_mask)
    n_inter_dims = interpretable_embedding.shape[1]

    if n_valid < n_inter_dims:
        raise ValueError(f"Too few valid rows ({n_valid}) for projection")

    if verbose:
        print(f"Computing projection residuals ({n_valid} valid rows)...")

    # Extract valid rows
    X = interpretable_embedding[valid_mask]  # (n_valid, d_inter)
    Y = full_embedding[valid_mask]  # (n_valid, d_full)

    # Projection: Residuals = Y - X @ B where B = lstsq(X, Y)
    # Using lstsq instead of explicit inverse for numerical stability
    B, _residuals_lstsq, _rank, _singular_values = linalg.lstsq(X, Y, cond=None)
    hat_matrix = X @ B
    residuals_valid = Y - hat_matrix

    # Create full residual matrix
    n_total = interpretable_embedding.shape[0]
    residuals = np.full((n_total, full_embedding.shape[1]), np.nan, dtype=np.float32)
    residuals[valid_mask] = residuals_valid

    return residuals


def _print_elbow_plot(singular_values: np.ndarray, save_path: str | None = None) -> None:
    """Print SVD elbow plot information and optionally save plot.

    Args:
        singular_values: Array of singular values from SVD.
        save_path: Optional path to save the plot image.
    """
    print("\n" + "=" * 60)
    print("SVD ELBOW PLOT - Singular Values")
    print("=" * 60)
    print("Use this to select the number of nuisance dimensions.")
    print("Look for the 'elbow' where singular values start to level off.\n")

    # Print singular values table
    print(f"{'Dim':<6}{'Singular Value':<18}{'Diff from Prev':<18}{'Cumulative %':<15}")
    print("-" * 57)

    total_variance = np.sum(singular_values ** 2)
    cumulative = 0.0

    for i, sv in enumerate(singular_values):
        variance = sv ** 2
        cumulative += variance
        pct = 100.0 * cumulative / total_variance
        diff = singular_values[i - 1] - sv if i > 0 else 0.0
        print(f"{i + 1:<6}{sv:<18.4f}{diff:<18.4f}{pct:<15.2f}")

    print("-" * 57)
    print("\nTo use a different number of dimensions, re-run with:")
    print("  num_nuisance_dims=<your_choice>")
    print("=" * 60 + "\n")

    # Try to create matplotlib plot if available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Singular values
        axes[0].plot(range(1, len(singular_values) + 1), singular_values, 'bo-', markersize=8)
        axes[0].set_xlabel('Component')
        axes[0].set_ylabel('Singular Value')
        axes[0].set_title('SVD Singular Values (Scree Plot)')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Differences (elbow detection)
        if len(singular_values) > 1:
            diffs = -np.diff(singular_values)  # Negative differences
            axes[1].plot(range(1, len(diffs) + 1), diffs, 'ro-', markersize=8)
            axes[1].set_xlabel('Component Index')
            axes[1].set_ylabel('Decrease in Singular Value')
            axes[1].set_title('Elbow Plot (Diff of Singular Values)')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Elbow plot saved to: {save_path}")
        else:
            # Try to save to a default location
            default_path = "svd_elbow_plot.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"Elbow plot saved to: {default_path}")

        plt.close(fig)

    except ImportError:
        print("(matplotlib not available - install it to generate plot images)")
    except Exception as e:  # pylint: disable=broad-exception-caught  # plotting is optional; never fail extraction over it
        print(f"(Could not generate plot: {e})")


def compute_svd_nuisance(
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    num_nuisance_dims: int = 5,
    k_svd: int = 20,
    show_elbow_plot: bool = False,
    elbow_plot_path: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract nuisance dimensions via SVD on residuals.

    Applies Singular Value Decomposition to the residual matrix to extract
    the principal nuisance directions, which capture variation not explained
    by the interpretable attribute embeddings.

    Args:
        residuals: Residual matrix with NaN for invalid rows.
        valid_mask: Boolean mask for valid rows.
        num_nuisance_dims: Number of nuisance dimensions to keep in output.
        k_svd: Number of SVD components to compute (should be >= num_nuisance_dims).
        show_elbow_plot: Whether to display elbow plot information.
        elbow_plot_path: Optional path to save elbow plot image.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (nuisance_embedding, singular_values) where nuisance_embedding
        has shape (n, num_nuisance_dims) with NaN for invalid rows, and
        singular_values contains the SVD singular values.
    """
    residuals_valid = residuals[valid_mask]

    if verbose:
        print(f"Computing SVD on residuals {residuals_valid.shape}...")

    # Ensure k_svd is large enough to show elbow if requested
    if show_elbow_plot:
        k_svd = max(k_svd, 20)

    try:
        from scipy.sparse.linalg import svds
        k = min(k_svd, min(residuals_valid.shape) - 1)
        U, s, _Vt = svds(residuals_valid.astype(np.float64), k=k)
        # Sort descending
        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
    except Exception:  # pylint: disable=broad-exception-caught  # fall back to dense SVD on any sparse-solver failure
        if verbose:
            print("Sparse SVD failed, using full SVD...")
        U, s, _Vt = linalg.svd(residuals_valid, full_matrices=False)
        U = U[:, :k_svd]
        s = s[:k_svd]

    if verbose:
        print(f"Top singular values: {s[:min(8, len(s))]}")

    # Show elbow plot if requested
    if show_elbow_plot:
        _print_elbow_plot(s, save_path=elbow_plot_path)

    # Select dimensions and scale by singular values
    actual_dims = min(num_nuisance_dims, len(s))
    nuisance_valid = U[:, :actual_dims] * s[:actual_dims]

    # Create full matrix
    n_total = residuals.shape[0]
    nuisance = np.full((n_total, actual_dims), np.nan, dtype=np.float32)
    nuisance[valid_mask] = nuisance_valid

    if verbose:
        print(f"Nuisance embedding shape: {nuisance.shape}")

    return nuisance, s


def compute_umap_nuisance(
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    num_nuisance_dims: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract nuisance dimensions via UMAP on residuals.

    Applies UMAP (Uniform Manifold Approximation and Projection) to the residual
    matrix to extract nonlinear nuisance dimensions. Unlike SVD, UMAP can capture
    nonlinear structure but dimensions are not orthogonal.

    Args:
        residuals: Residual matrix with NaN for invalid rows.
        valid_mask: Boolean mask for valid rows.
        num_nuisance_dims: Number of UMAP dimensions (n_components).
        n_neighbors: UMAP n_neighbors parameter (local neighborhood size).
        min_dist: UMAP min_dist parameter (minimum distance in embedding).
        metric: Distance metric for UMAP.
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (nuisance_embedding, umap_info) where nuisance_embedding
        has shape (n, num_nuisance_dims) with NaN for invalid rows, and
        umap_info contains UMAP parameters used.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is required for method='umap'. Install with: pip install umap-learn"
        ) from exc

    residuals_valid = residuals[valid_mask]

    if verbose:
        print(f"Computing UMAP on residuals {residuals_valid.shape}...")
        print(f"  n_components={num_nuisance_dims}, n_neighbors={n_neighbors}, "
              f"min_dist={min_dist}, metric={metric}")

    # Fit UMAP
    reducer = umap.UMAP(
        n_components=num_nuisance_dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
    )

    nuisance_valid = reducer.fit_transform(residuals_valid.astype(np.float64))

    # Create full matrix
    n_total = residuals.shape[0]
    nuisance = np.full((n_total, num_nuisance_dims), np.nan, dtype=np.float32)
    nuisance[valid_mask] = nuisance_valid.astype(np.float32)

    if verbose:
        print(f"Nuisance embedding shape: {nuisance.shape}")

    umap_info = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }

    return nuisance, umap_info


def compute_nuisance_controls(
    attribute_embeddings: dict[str, np.ndarray],
    full_embedding: np.ndarray,
    num_nuisance_dims: int | None = None,
    method: Literal["svd", "umap"] = "svd",
    show_elbow_plot: bool = True,
    elbow_plot_path: str | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute nuisance controls from attribute and full embeddings.

    Extracts nuisance dimensions that capture variation in the full embeddings
    not explained by the interpretable attribute-specific embeddings. This
    allows controlling for confounding factors (style, tone, etc.) in
    downstream analyses.

    Two methods are available:
    - SVD (default): Linear dimensionality reduction. Produces orthogonal dimensions
      that capture maximum variance in residuals. Recommended for statistical controls.
    - UMAP: Nonlinear dimensionality reduction. Can capture nonlinear structure but
      dimensions are not orthogonal. Better for visualization or nonlinear patterns.

    Args:
        attribute_embeddings: Attribute-specific embeddings from extract_representations().
        full_embedding: Full dimensionality embeddings (e.g., OpenAI 3072-dim).
        num_nuisance_dims: Number of nuisance dimensions to extract. If None, uses
            default (5) and prints a warning with guidance on selecting dimensions.
        method: Dimensionality reduction method - "svd" (default) or "umap".
        show_elbow_plot: For SVD, whether to display elbow plot to help select
            dimensions. Only shown when num_nuisance_dims is None (using default).
        elbow_plot_path: Optional path to save the elbow plot image.
        umap_n_neighbors: UMAP n_neighbors parameter (only used if method="umap").
        umap_min_dist: UMAP min_dist parameter (only used if method="umap").
        umap_metric: UMAP distance metric (only used if method="umap").
        random_state: Random seed for UMAP reproducibility.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (nuisance_embedding, interpretable_embedding, diagnostics):
        - nuisance_embedding: Nuisance controls of shape (n, num_nuisance_dims)
        - interpretable_embedding: Combined attribute embedding of shape (n, sum(attribute_dims))
        - diagnostics: Dict with method-specific info:
            - For SVD: {"method": "svd", "singular_values": array, "num_dims": int}
            - For UMAP: {"method": "umap", "umap_params": dict, "num_dims": int}
    """
    if verbose:
        print(f"Computing nuisance controls (method={method})...")

    # Handle default dimensions with warning
    using_default = num_nuisance_dims is None
    if using_default:
        num_nuisance_dims = DEFAULT_NUISANCE_DIMS

        if method == "svd":
            warnings.warn(
                f"\nUsing default of {num_nuisance_dims} nuisance dimensions. "
                f"Review the elbow plot below to verify this is appropriate for your data, "
                f"or specify num_nuisance_dims explicitly.",
                UserWarning,
            )
        else:  # umap
            warnings.warn(
                f"\nUsing default of {num_nuisance_dims} nuisance dimensions for UMAP. "
                f"Consider specifying num_nuisance_dims explicitly based on your analysis needs.",
                UserWarning,
            )

    # Combine attribute embeddings
    interpretable = np.hstack(list(attribute_embeddings.values()))

    if verbose:
        print(f"Combined interpretable embedding: {interpretable.shape}")

    # Identify valid rows
    valid_mask = identify_valid_rows(attribute_embeddings, full_embedding)
    n_valid = np.sum(valid_mask)

    if verbose:
        print(f"Valid rows: {n_valid}")

    # Compute residuals
    residuals = compute_projection_residuals(
        interpretable, full_embedding, valid_mask, verbose=verbose
    )

    # Apply dimensionality reduction
    if method == "svd":
        nuisance, singular_values = compute_svd_nuisance(
            residuals,
            valid_mask,
            num_nuisance_dims=num_nuisance_dims,
            show_elbow_plot=show_elbow_plot and using_default,
            elbow_plot_path=elbow_plot_path,
            verbose=verbose,
        )
        diagnostics = {
            "method": "svd",
            "singular_values": singular_values,
            "num_dims": num_nuisance_dims,
            "using_default": using_default,
        }

    elif method == "umap":
        nuisance, umap_info = compute_umap_nuisance(
            residuals,
            valid_mask,
            num_nuisance_dims=num_nuisance_dims,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
            verbose=verbose,
        )
        diagnostics = {
            "method": "umap",
            "umap_params": umap_info,
            "num_dims": num_nuisance_dims,
            "using_default": using_default,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'svd' or 'umap'.")

    return nuisance, interpretable, diagnostics
