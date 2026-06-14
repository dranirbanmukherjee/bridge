"""
Linear Projection Baseline for BRIDGE

This module implements a linear projection baseline to compare against BRIDGE's
neural network approach. The baseline:

1. Constructs a design matrix D (one-hot encoded: region + varietal)
2. Performs Ridge regression of full embeddings E onto D
3. Extracts fitted values Ê as "linear attribute signal"
4. Computes residuals R = E - Ê
5. Applies SVD to R to get linear nuisance dimensions

This addresses AE Recommendation 3: justify the complexity of the deep neural
architecture by comparing against simpler linear alternatives.

Usage:
    python -m bridge.linear_baseline

Output:
    bridge_output/baselines/linear/
    ├── linear_attribute_embedding.npy   # Fitted values from projection
    ├── linear_nuisance.npy              # SVD of residuals
    ├── linear_config.json               # Configuration and metadata
    └── ridge_coefficients.npy           # Ridge regression coefficients
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def build_design_matrix(
    region_labels: np.ndarray,
    varietal_labels: np.ndarray,
) -> tuple[np.ndarray, OneHotEncoder, OneHotEncoder]:
    """
    Build one-hot encoded design matrix for region and varietal.

    Args:
        region_labels: Array of region labels (strings or encoded ints)
        varietal_labels: Array of varietal labels (strings or encoded ints)

    Returns:
        Tuple of (design_matrix, region_encoder, varietal_encoder)
    """
    # Reshape for sklearn
    region_reshaped = region_labels.reshape(-1, 1)
    varietal_reshaped = varietal_labels.reshape(-1, 1)

    # Fit one-hot encoders
    region_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    varietal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    region_onehot = region_encoder.fit_transform(region_reshaped)
    varietal_onehot = varietal_encoder.fit_transform(varietal_reshaped)

    # Concatenate to form design matrix
    design_matrix = np.hstack([region_onehot, varietal_onehot])

    return design_matrix, region_encoder, varietal_encoder


def linear_projection_baseline(
    embeddings: np.ndarray,
    region_labels: np.ndarray,
    varietal_labels: np.ndarray,
    alpha: float = 1.0,
    n_nuisance_dims: int = 5,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute linear projection baseline representations.

    This method:
    1. One-hot encodes region and varietal labels
    2. Regresses embeddings onto the design matrix using Ridge regression
    3. Extracts fitted values as "attribute signal"
    4. Computes residuals and applies SVD for nuisance dimensions

    Args:
        embeddings: Full embedding matrix (n_samples, embedding_dim)
        region_labels: Region labels for each sample
        varietal_labels: Varietal labels for each sample
        alpha: Ridge regression regularization strength
        n_nuisance_dims: Number of nuisance dimensions to extract via SVD
        verbose: Print progress

    Returns:
        Dictionary with:
        - 'attribute_embedding': Fitted values from Ridge regression
        - 'nuisance_embedding': Top-k SVD components of residuals
        - 'residuals': Full residual matrix
        - 'coefficients': Ridge regression coefficients
        - 'explained_variance_ratio': Variance explained by nuisance dims
    """
    n_samples, embedding_dim = embeddings.shape

    if verbose:
        print("Linear Projection Baseline")
        print(f"  Samples: {n_samples}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Ridge alpha: {alpha}")
        print(f"  Nuisance dims: {n_nuisance_dims}")

    # Step 1: Build design matrix
    if verbose:
        print("\n1. Building design matrix (one-hot encoding)...")

    design_matrix, region_enc, varietal_enc = build_design_matrix(
        region_labels, varietal_labels
    )

    n_region_cats = len(region_enc.categories_[0])
    n_varietal_cats = len(varietal_enc.categories_[0])

    if verbose:
        print(f"   Design matrix shape: {design_matrix.shape}")
        print(f"   Region categories: {n_region_cats}")
        print(f"   Varietal categories: {n_varietal_cats}")

    # Step 2: Ridge regression
    if verbose:
        print("\n2. Fitting Ridge regression...")

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(design_matrix, embeddings)

    # Fitted values = attribute signal
    attribute_embedding = ridge.predict(design_matrix)

    if verbose:
        print(f"   Coefficients shape: {ridge.coef_.shape}")
        print(f"   Attribute embedding shape: {attribute_embedding.shape}")

    # Step 3: Compute residuals
    if verbose:
        print("\n3. Computing residuals...")

    residuals = embeddings - attribute_embedding
    residual_norm = np.linalg.norm(residuals, 'fro')
    embedding_norm = np.linalg.norm(embeddings, 'fro')

    if verbose:
        print(f"   Residual Frobenius norm: {residual_norm:.2f}")
        print(f"   Residual / Original ratio: {residual_norm / embedding_norm:.4f}")

    # Step 4: SVD on residuals for nuisance dimensions
    if verbose:
        print(f"\n4. Extracting {n_nuisance_dims} nuisance dimensions via SVD...")

    svd = TruncatedSVD(n_components=n_nuisance_dims, random_state=42)
    nuisance_embedding = svd.fit_transform(residuals)

    if verbose:
        print(f"   Nuisance embedding shape: {nuisance_embedding.shape}")
        print(f"   Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        print(f"   Top singular values: {svd.singular_values_[:5]}")

    return {
        'attribute_embedding': attribute_embedding,
        'nuisance_embedding': nuisance_embedding,
        'residuals': residuals,
        'coefficients': ridge.coef_,
        'intercept': ridge.intercept_,
        'explained_variance_ratio': svd.explained_variance_ratio_,
        'singular_values': svd.singular_values_,
        'n_region_cats': n_region_cats,
        'n_varietal_cats': n_varietal_cats,
    }


def run_linear_baseline(
    output_dir: str = "bridge_output",
    alpha: float = 1.0,
    n_nuisance_dims: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full linear projection baseline pipeline on wine data.

    Loads wine data and full embeddings, computes linear projection baseline,
    and saves outputs for R analysis.

    Args:
        output_dir: Base output directory (creates baselines/linear/ subdirectory)
        alpha: Ridge regression regularization strength
        n_nuisance_dims: Number of nuisance dimensions
        verbose: Print progress

    Returns:
        Dictionary with paths to saved files and summary statistics
    """
    from sklearn.preprocessing import LabelEncoder

    from bridge.data import clean_data, create_composite_field, load_data

    output_path = Path(output_dir) / "baselines" / "linear"
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("LINEAR PROJECTION BASELINE")
        print("=" * 60)

    # Load wine data
    if verbose:
        print("\nLoading wine data...")

    wine_data_path = Path(output_dir).parent / "winemag-data-130k-v2.csv"
    if not wine_data_path.exists():
        # Try current directory
        wine_data_path = Path("winemag-data-130k-v2.csv")

    df = load_data(str(wine_data_path))
    df = clean_data(df)
    df = create_composite_field(df, ["country", "province"], "country_province", separator="#####")

    if verbose:
        print(f"  Loaded {len(df)} wines")

    # Load full embeddings
    embeddings_path = (
        Path(output_dir).parent / "_rData" / "full_dimensionality_embedding_matrix.npy"
    )
    if not embeddings_path.exists():
        # Try alternative path
        embeddings_path = (
            Path(output_dir) / "embeddings" / "full_embedding_openai_3072.npy"
        )

    if not embeddings_path.exists():
        path_a = (Path(output_dir).parent / "_rData"
                  / "full_dimensionality_embedding_matrix.npy")
        path_b = (Path(output_dir) / "embeddings"
                  / "full_embedding_openai_3072.npy")
        raise FileNotFoundError(
            f"Full embeddings not found. Checked:\n"
            f"  - {path_a}\n"
            f"  - {path_b}"
        )

    if verbose:
        print(f"  Loading embeddings from: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    if verbose:
        print(f"  Embeddings shape: {embeddings.shape}")

    # Encode labels using sklearn LabelEncoder
    if verbose:
        print("\nEncoding labels...")

    region_encoder = LabelEncoder()
    varietal_encoder = LabelEncoder()

    # Fill NaN values with empty string for encoding
    region_labels = region_encoder.fit_transform(df["country_province"].fillna("").astype(str))
    varietal_labels = varietal_encoder.fit_transform(df["variety"].fillna("").astype(str))

    if verbose:
        print(f"  Region classes: {len(region_encoder.classes_)}")
        print(f"  Varietal classes: {len(varietal_encoder.classes_)}")

    # Run linear projection baseline
    results = linear_projection_baseline(
        embeddings=embeddings,
        region_labels=region_labels,
        varietal_labels=varietal_labels,
        alpha=alpha,
        n_nuisance_dims=n_nuisance_dims,
        verbose=verbose,
    )

    # Save outputs
    if verbose:
        print("\n5. Saving outputs...")

    # Attribute embedding
    attr_path = output_path / "linear_attribute_embedding.npy"
    np.save(attr_path, results['attribute_embedding'])
    if verbose:
        print(f"   Saved: {attr_path}")

    # Nuisance embedding
    nuisance_path = output_path / "linear_nuisance.npy"
    np.save(nuisance_path, results['nuisance_embedding'])
    if verbose:
        print(f"   Saved: {nuisance_path}")

    # Ridge coefficients
    coef_path = output_path / "ridge_coefficients.npy"
    np.save(coef_path, results['coefficients'])
    if verbose:
        print(f"   Saved: {coef_path}")

    # Configuration and metadata
    config = {
        'alpha': alpha,
        'n_nuisance_dims': n_nuisance_dims,
        'n_samples': len(df),
        'embedding_dim': embeddings.shape[1],
        'n_region_cats': results['n_region_cats'],
        'n_varietal_cats': results['n_varietal_cats'],
        'design_matrix_cols': results['n_region_cats'] + results['n_varietal_cats'],
        'explained_variance_ratio': results['explained_variance_ratio'].tolist(),
        'singular_values': results['singular_values'].tolist(),
        'residual_variance_ratio': float(
            np.var(results['residuals']) / np.var(embeddings)
        ),
        'timestamp': datetime.now().isoformat(),
    }

    config_path = output_path / "linear_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"   Saved: {config_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("LINEAR PROJECTION BASELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutput directory: {output_path}")
        print(f"  linear_attribute_embedding.npy: {results['attribute_embedding'].shape}")
        print(f"  linear_nuisance.npy: {results['nuisance_embedding'].shape}")
        print(f"  ridge_coefficients.npy: {results['coefficients'].shape}")
        print("  linear_config.json")

    return {
        'output_dir': str(output_path),
        'attribute_embedding_path': str(attr_path),
        'nuisance_path': str(nuisance_path),
        'config': config,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run linear projection baseline for BRIDGE comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bridge_output",
        help="Base output directory (default: bridge_output)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--n-nuisance-dims",
        type=int,
        default=5,
        help="Number of nuisance dimensions to extract (default: 5)"
    )

    args = parser.parse_args()

    run_linear_baseline(
        output_dir=args.output_dir,
        alpha=args.alpha,
        n_nuisance_dims=args.n_nuisance_dims,
        verbose=True,
    )
