"""
Empath psycholinguistic controls for BRIDGE.

This module generates Empath-based nuisance controls from product descriptions,
applying SVD dimensionality reduction to produce low-dimensional representations
suitable for use as control variables in regression analysis.

Empath is a text analysis tool that scores text across ~200 psycholinguistic
categories (emotions, topics, behaviors, etc.). These scores capture stylistic
and tonal aspects of text that may confound attribute-based analyses.

Example usage:
    from bridge.empath_controls import generate_empath_controls

    # Generate controls from descriptions
    empath_controls, diagnostics = generate_empath_controls(
        descriptions=df['description'].tolist(),
        n_components=10,
    )

    # diagnostics contains category names, explained variance, etc.
    print(f"Explained variance: {diagnostics['explained_variance_ratio'].sum():.1%}")
"""

from typing import Any

import numpy as np
from tqdm import tqdm


def generate_empath_controls(
    descriptions: list[str],
    n_components: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Generate Empath-based nuisance controls for product descriptions.

    This function:
    1. Scores each description across all Empath categories (~200)
    2. Applies TruncatedSVD to reduce dimensionality to n_components
    3. Returns the transformed scores and diagnostic information

    Args:
        descriptions: List of product descriptions to analyze.
        n_components: Number of SVD components to retain (default: 10, matches simulation).
        random_state: Random seed for SVD reproducibility.
        verbose: Whether to print progress information.

    Returns:
        Tuple of:
            empath_controls: (N, n_components) array of SVD-transformed Empath scores.
            diagnostics: Dict containing:
                - categories: List of Empath category names
                - n_categories: Number of categories used
                - explained_variance_ratio: Per-component explained variance
                - total_explained_variance: Sum of explained variance ratios
                - n_empty: Count of empty/failed descriptions
                - svd_components: The SVD components matrix (n_components x n_categories)
                - category_scores_mean: Mean score per category (for reference)

    Raises:
        ImportError: If empath package is not installed.
        ValueError: If n_components exceeds number of categories.

    Example:
        >>> from bridge.empath_controls import generate_empath_controls
        >>> descriptions = ["A rich, full-bodied wine with notes of cherry.",
        ...                 "Crisp and refreshing with citrus overtones."]
        >>> controls, diag = generate_empath_controls(descriptions, n_components=5)
        >>> print(controls.shape)
        (2, 5)
        >>> print(f"Explained variance: {diag['total_explained_variance']:.1%}")
    """
    # Import dependencies (check empath is installed)
    try:
        from empath import Empath
    except ImportError as e:
        raise ImportError(
            "The 'empath' package is required for Empath controls. "
            "Install it with: pip install empath"
        ) from e

    from sklearn.decomposition import TruncatedSVD

    if verbose:
        print("Generating Empath psycholinguistic controls...")

    # Initialize Empath lexicon
    lexicon = Empath()

    # Get all category names dynamically
    # Empath's analyze returns a dict with all categories, so we analyze an empty string
    # to get the full list of categories
    sample_analysis = lexicon.analyze("sample text", normalize=True)
    categories = sorted(sample_analysis.keys())
    n_categories = len(categories)

    if verbose:
        print(f"  Using {n_categories} Empath categories")

    # Validate n_components
    if n_components > n_categories:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed number of categories ({n_categories})"
        )

    # Analyze each description
    n_samples = len(descriptions)
    scores = np.zeros((n_samples, n_categories), dtype=np.float32)
    n_empty = 0

    iterator = tqdm(descriptions, desc="Analyzing text", disable=not verbose)
    for i, text in enumerate(iterator):
        # Handle empty/missing descriptions
        if not text or not isinstance(text, str) or text.strip() == "":
            n_empty += 1
            # Leave as zeros (neutral)
            continue

        # Analyze with normalization
        analysis = lexicon.analyze(text, normalize=True)

        # Convert to array in consistent order
        for j, cat in enumerate(categories):
            scores[i, j] = analysis.get(cat, 0.0)

    if verbose:
        print(f"  Analyzed {n_samples} descriptions ({n_empty} empty/invalid)")

    # Apply TruncatedSVD for dimensionality reduction
    if verbose:
        print(f"  Applying SVD ({n_categories} -> {n_components} dimensions)...")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    empath_controls = svd.fit_transform(scores)

    # Compute diagnostics
    explained_variance_ratio = svd.explained_variance_ratio_
    total_explained_variance = explained_variance_ratio.sum()

    if verbose:
        print("  Explained variance per component:")
        for i, var in enumerate(explained_variance_ratio):
            print(f"    Component {i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"  Total explained variance: {total_explained_variance:.3f} "
              f"({total_explained_variance*100:.1f}%)")

    diagnostics = {
        "categories": categories,
        "n_categories": n_categories,
        "explained_variance_ratio": explained_variance_ratio,
        "total_explained_variance": total_explained_variance,
        "n_empty": n_empty,
        "n_components": n_components,
        "svd_components": svd.components_,  # (n_components, n_categories)
        "category_scores_mean": scores.mean(axis=0),  # Mean score per category
        "category_scores_std": scores.std(axis=0),  # Std per category
    }

    return empath_controls, diagnostics


def save_empath_controls(
    controls: np.ndarray,
    diagnostics: dict[str, Any],
    output_dir: str,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Save Empath controls and diagnostics to files.

    Args:
        controls: (N, n_components) array of Empath controls.
        diagnostics: Diagnostics dict from generate_empath_controls.
        output_dir: Directory to save files.
        verbose: Whether to print file paths.

    Returns:
        Dict mapping output names to file paths.
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    representations_dir = output_path / "representations"
    metadata_dir = output_path / "metadata"

    representations_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Save controls array
    controls_path = representations_dir / "empath_controls.npy"
    np.save(controls_path, controls)
    outputs["empath_controls"] = str(controls_path)

    # Save diagnostics (convert numpy arrays to lists for JSON)
    diagnostics_json = {
        "categories": diagnostics["categories"],
        "n_categories": diagnostics["n_categories"],
        "n_components": diagnostics["n_components"],
        "explained_variance_ratio": diagnostics["explained_variance_ratio"].tolist(),
        "total_explained_variance": float(diagnostics["total_explained_variance"]),
        "n_empty": diagnostics["n_empty"],
        "category_scores_mean": diagnostics["category_scores_mean"].tolist(),
        "category_scores_std": diagnostics["category_scores_std"].tolist(),
    }

    diagnostics_path = metadata_dir / "empath_diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_json, f, indent=2)
    outputs["empath_diagnostics"] = str(diagnostics_path)

    if verbose:
        print(f"  Saved Empath controls to {controls_path}")
        print(f"  Saved Empath diagnostics to {diagnostics_path}")

    return outputs
