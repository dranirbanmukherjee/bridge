"""
High-level BRIDGE pipeline class.

Provides a unified interface for the complete BRIDGE workflow:
- Data loading and preprocessing
- Embedding generation
- Model training
- Representation extraction
- Export for R analysis
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bridge.array import build_3d_array
from bridge.baselines import generate_word_embedding_baseline
from bridge.config import BRIDGEConfig
from bridge.embeddings import generate_embeddings
from bridge.empath_controls import generate_empath_controls, save_empath_controls
from bridge.encoder import AttributeEncoder
from bridge.extraction import compute_nuisance_controls, extract_representations
from bridge.model import DEVICE, BRIDGEModel, load_model, save_model
from bridge.training import prepare_data_loaders, train_model, tune_hyperparameters
from bridge.validation import (
    compute_classification_metrics,
    print_classification_metrics,
    save_classification_metrics,
)


class BRIDGEPipeline:
    """High-level pipeline for BRIDGE embedding/ML process.

    This class provides a unified interface for the complete workflow:
    fitting on training data, extracting representations, and exporting
    results for R analysis.

    Attributes:
        attributes: List of attribute names to learn.
        config: BRIDGEConfig instance with hyperparameters.
        output_dir: Path to output directory.
        verbose: Whether to print progress messages.
        encoder: Fitted AttributeEncoder (set after fit()).
        model: Trained BRIDGEModel (set after fit()).
        embedding_matrix: OpenAI embeddings (set after fit()).
        input_array: 3D contrastive learning array (set after fit()).
        attribute_embeddings: Extracted attribute embeddings (set after fit()).
        nuisance_embedding: Extracted nuisance controls (set after fit()).
        hp_dict: Model hyperparameters (set after fit()).
        is_fitted: Whether the pipeline has been fitted.

    Example:
        >>> from bridge import BRIDGEPipeline
        >>>
        >>> # Initialize for wine domain
        >>> pipeline = BRIDGEPipeline(
        ...     attributes=["region", "varietal"],
        ...     output_dir="./output"
        ... )
        >>>
        >>> # Fit on data
        >>> pipeline.fit(
        ...     descriptions=df["description"],
        ...     labels={"region": df["province"], "varietal": df["variety"]}
        ... )
        >>>
        >>> # Get embeddings
        >>> embeddings = pipeline.transform(df["description"])
        >>> print(embeddings["region"].shape)  # (n_samples, embedding_dim)
    """

    def __init__(
        self,
        attributes: list[str],
        config: BRIDGEConfig | None = None,
        output_dir: str | None = None,
        verbose: bool = True,
    ):
        """Initialize pipeline.

        Args:
            attributes: Names of attributes to learn (e.g., ["region", "varietal"]).
            config: Configuration object. If None, uses defaults with specified attributes.
            output_dir: Directory for output files. Defaults to "./bridge_output".
            verbose: Whether to print progress messages.
        """
        self.attributes = list(attributes)
        self.config = config or BRIDGEConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("./bridge_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # State (set during fitting)
        self.encoder: AttributeEncoder | None = None
        self.model: BRIDGEModel | None = None
        self.embedding_matrix: np.ndarray | None = None
        self.input_array: np.ndarray | None = None
        self.attribute_embeddings: dict[str, np.ndarray] | None = None
        self.nuisance_embedding: np.ndarray | None = None
        self.nuisance_diagnostics: dict[str, Any] | None = None
        self.hp_dict: dict[str, Any] | None = None
        self.classification_metrics: dict[str, dict[str, float]] | None = None
        self.encoded_labels: dict[str, np.ndarray] | None = None
        self.is_fitted: bool = False

        if self.verbose:
            print("BRIDGEPipeline initialized")
            print(f"  Attributes: {self.attributes}")
            print(f"  Output: {self.output_dir}")
            print(f"  Device: {DEVICE}")

    def fit(
        self,
        descriptions: list[str] | pd.Series | np.ndarray,
        labels: dict[str, list | pd.Series | np.ndarray],
        embeddings: np.ndarray | None = None,
        tune: bool = True,
        n_trials: int = 75,
        use_cached_model: bool = True,
        cached_model_path: str | None = None,
    ) -> "BRIDGEPipeline":
        """Fit the pipeline on training data.

        Runs the complete BRIDGE workflow: label encoding, embedding generation,
        3D array construction, hyperparameter tuning, model training, and
        representation extraction.

        If a cached model exists (at cached_model_path or the default location),
        it will be loaded instead of training a new model.

        Args:
            descriptions: Text descriptions for embedding generation.
            labels: Dictionary mapping attribute names to label arrays.
            embeddings: Pre-computed embeddings. If None, generates via OpenAI API.
            tune: Whether to run Optuna hyperparameter tuning.
            n_trials: Number of Optuna tuning trials if tune=True.
            use_cached_model: Whether to use a cached model if available.
            cached_model_path: Path to cached model. If None, uses default location
                (output_dir/model/bridge_model.pt).

        Returns:
            Self for method chaining.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("BRIDGE Pipeline - Fitting")
            print("=" * 60)

        # Convert to numpy
        if isinstance(descriptions, pd.Series):
            descriptions = descriptions.tolist()
        descriptions = list(descriptions)

        labels_np = {}
        for name in self.attributes:
            arr = labels[name]
            if isinstance(arr, pd.Series):
                arr = arr.values
            labels_np[name] = np.array(arr)

        n_samples = len(descriptions)
        if self.verbose:
            print(f"Samples: {n_samples}")

        # Step 1: Encode labels
        if self.verbose:
            print("\n--- Step 1: Encoding labels ---")
        self.encoder = AttributeEncoder(self.attributes)
        self.encoded_labels = self.encoder.fit_transform(
            {name: labels_np[name] for name in self.attributes},
            verbose=self.verbose,
        )
        encoded_labels = self.encoded_labels

        # Step 2: Generate or use provided embeddings
        if self.verbose:
            print("\n--- Step 2: Embeddings ---")
        if embeddings is not None:
            self.embedding_matrix = embeddings
            if self.verbose:
                print(f"Using provided embeddings: {embeddings.shape}")
        else:
            self.embedding_matrix = generate_embeddings(
                descriptions,
                backend=self.config.embedding_backend,
                model=self.config.embedding_model,
                show_progress=self.verbose,
                verbose=self.verbose,
            )

        # Step 3: Build 3D array
        if self.verbose:
            print("\n--- Step 3: Building 3D array ---")
        self.input_array = build_3d_array(
            self.embedding_matrix,
            encoded_labels,
            mini_batch_size=self.config.mini_batch_size,
            seed=self.config.array_seed,  # Use array-specific seed (42) to match original
            verbose=self.verbose,
        )

        # Get attribute sizes
        attribute_sizes = {
            name: self.encoder.num_classes(name)
            for name in self.attributes
        }

        # Step 4: Load cached model or train new one
        if self.verbose:
            print("\n--- Step 4: Model ---")

        # Check for cached model
        model_dir = self.output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        default_model_path = model_dir / "bridge_model.pt"
        model_path = Path(cached_model_path) if cached_model_path else default_model_path

        if use_cached_model and model_path.exists():
            if self.verbose:
                print(f"Loading cached model from {model_path}")
            self.model = load_model(str(model_path))
            self.hp_dict = self.model.get_hp_dict()
            if self.verbose:
                print(f"  Loaded model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        else:
            # Train new model
            if self.verbose:
                if use_cached_model:
                    print(f"No cached model found at {model_path}")
                print("Training new model...")

            if tune:
                if self.verbose:
                    print(f"Running hyperparameter tuning ({n_trials} trials)...")
                self.hp_dict = tune_hyperparameters(
                    self.input_array,
                    encoded_labels,
                    attribute_sizes,
                    n_trials=n_trials,
                    epochs_per_trial=self.config.epochs_tuning,
                    batch_size=self.config.batch_size,
                    seed=self.config.seed,
                    verbose=1 if self.verbose else 0,
                    device=DEVICE,  # Pre-load data on GPU for unified memory systems
                    fixed_mask_size=self.config.fixed_mask_size,
                )
            else:
                self.hp_dict = {
                    "attribute_names": self.attributes,
                    "attribute_sizes": attribute_sizes,
                    "orig_size": self.embedding_matrix.shape[1],
                    "mini_batch_size": self.config.mini_batch_size,
                    "projection_units": self.config.projection_units,
                    "embedding_units_per_attribute": self.config.embedding_units_per_attribute,
                    "mask_size": self.config.mask_size,
                    "contrastive_temperature": self.config.contrastive_temp,
                    "contrastive_weight": self.config.contrastive_weight,
                }

            # Final training
            # Filter out training-only params (learning_rate, weight_decay) from hp_dict
            model_hp = {k: v for k, v in self.hp_dict.items()
                        if k not in ("learning_rate", "weight_decay")}
            self.model = BRIDGEModel.from_hp_dict(model_hp)
            train_loader, val_loader = prepare_data_loaders(
                self.input_array,
                encoded_labels,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                seed=self.config.seed,
                device=DEVICE,  # Pre-load data on GPU for unified memory systems
            )

            checkpoint_path = str(model_dir / "bridge_model.pt")
            self.model, _history = train_model(
                self.model,
                train_loader,
                val_loader,
                epochs=self.config.epochs_final,
                learning_rate=self.hp_dict.get("learning_rate", self.config.learning_rate),
                weight_decay=self.hp_dict.get("weight_decay", 0.01),
                early_stopping_patience=self.config.early_stopping_patience,
                checkpoint_path=checkpoint_path,
                verbose=1 if self.verbose else 0,
            )

        # Step 5: Extract representations
        if self.verbose:
            print("\n--- Step 5: Extracting representations ---")
        self.attribute_embeddings = extract_representations(
            self.model,
            self.input_array,
            show_progress=self.verbose,
        )

        # Step 6: Compute nuisance controls
        if self.verbose:
            print("\n--- Step 6: Computing nuisance controls ---")
        self.nuisance_embedding, _, self.nuisance_diagnostics = compute_nuisance_controls(
            self.attribute_embeddings,
            self.embedding_matrix,
            num_nuisance_dims=self.config.num_nuisance_dims,
            method=self.config.nuisance_method,
            show_elbow_plot=self.config.nuisance_show_elbow_plot,
            elbow_plot_path=str(Path(self.output_dir) / "svd_elbow_plot.png") if self.config.nuisance_show_elbow_plot else None,
            umap_n_neighbors=self.config.nuisance_umap_n_neighbors,
            umap_min_dist=self.config.nuisance_umap_min_dist,
            umap_metric=self.config.nuisance_umap_metric,
            random_state=self.config.nuisance_random_state,
            verbose=self.verbose,
        )

        # Step 7: Compute classification metrics
        if self.verbose:
            print("\n--- Step 7: Computing classification metrics ---")
        self.classification_metrics = compute_classification_metrics(
            model=self.model,
            embeddings=self.input_array,
            labels=self.encoded_labels,
            show_progress=self.verbose,
        )

        # Print and save metrics
        if self.verbose:
            print_classification_metrics(
                self.classification_metrics,
                title="BRIDGE Model Classification Performance",
            )

        # Save metrics to metadata directory
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metadata_dir / "classification_metrics.json"
        save_classification_metrics(self.classification_metrics, str(metrics_path))
        if self.verbose:
            print(f"Classification metrics saved to: {metrics_path}")

        self.is_fitted = True

        if self.verbose:
            print("\n" + "=" * 60)
            print("Pipeline fitting complete!")
            print("=" * 60)

        return self

    def transform(
        self,
        descriptions: list[str] | pd.Series | None = None,
        embeddings: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Get embeddings for data.

        If called without arguments after fit(), returns training embeddings.
        Otherwise, extracts embeddings for new descriptions/embeddings.

        Args:
            descriptions: New text descriptions (requires OpenAI API call).
            embeddings: Pre-computed embeddings for new data.

        Returns:
            Dictionary with keys for each attribute name and 'nuisance',
            containing embedding arrays of shape (n_samples, dim).

        Raises:
            RuntimeError: If called before fit().
            ValueError: If neither descriptions nor embeddings provided for new data.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Return training embeddings if no new data
        if descriptions is None and embeddings is None:
            result = dict(self.attribute_embeddings)
            result['nuisance'] = self.nuisance_embedding
            return result

        # Transform new data
        if embeddings is None:
            if descriptions is None:
                raise ValueError("Must provide descriptions or embeddings")
            if isinstance(descriptions, pd.Series):
                descriptions = descriptions.tolist()
            embeddings = generate_embeddings(
                list(descriptions),
                backend=self.config.embedding_backend,
                model=self.config.embedding_model,
                show_progress=self.verbose,
                verbose=self.verbose,
            )

        # Build simple 3D array (anchor only, no negatives needed for inference)
        # For inference, we just need the anchor - pad with zeros
        n_samples = len(embeddings)
        input_array = np.zeros(
            (n_samples, self.config.mini_batch_size, embeddings.shape[1]),
            dtype=np.float32
        )
        input_array[:, 0, :] = embeddings

        # Extract representations
        attr_embs = extract_representations(
            self.model,
            input_array,
            show_progress=self.verbose,
        )

        # Compute nuisance (for transform, don't show elbow plot again)
        nuisance, _, _ = compute_nuisance_controls(
            attr_embs,
            embeddings,
            num_nuisance_dims=self.config.num_nuisance_dims,
            method=self.config.nuisance_method,
            show_elbow_plot=False,  # Don't show again during transform
            umap_n_neighbors=self.config.nuisance_umap_n_neighbors,
            umap_min_dist=self.config.nuisance_umap_min_dist,
            umap_metric=self.config.nuisance_umap_metric,
            random_state=self.config.nuisance_random_state,
            verbose=self.verbose,
        )

        result = dict(attr_embs)
        result['nuisance'] = nuisance
        return result

    def export(
        self,
        output_dir: str | None = None,
        full_embeddings: np.ndarray | None = None,
        embedding_backend: str | None = None,
        include_empath: bool = False,
        descriptions: list[str] | None = None,
        empath_n_components: int = 10,
        include_word_baseline: bool = False,
        region_labels: list[str] | None = None,
        varietal_labels: list[str] | None = None,
    ) -> dict[str, str]:
        """Export results for R analysis.

        Creates .npy files that can be loaded by RcppCNPy or reticulate,
        along with JSON metadata files for encoder mappings and configuration.

        Directory structure:
            output_dir/
            ├── embeddings/           # Full-dimensionality embeddings
            │   └── full_embedding_{backend}_{dim}.npy
            ├── representations/      # Extracted BRIDGE representations
            │   ├── region_embedding.npy
            │   ├── varietal_embedding.npy
            │   ├── nuisance_embedding.npy
            │   └── empath_controls.npy  # If include_empath=True
            ├── baselines/            # Baseline comparisons (if include_word_baseline=True)
            │   ├── word_embedding_region.npy
            │   ├── word_embedding_varietal.npy
            │   ├── word_embedding_combined.npy
            │   └── word_embedding_config.json
            ├── model/                # Trained model checkpoint
            │   └── bridge_model.pt
            └── metadata/             # Configuration and mappings
                ├── encoder.json
                ├── hyperparameters.json
                ├── config.json
                ├── manifest.json
                └── empath_diagnostics.json  # If include_empath=True

        Args:
            output_dir: Output directory. Uses self.output_dir if not specified.
            full_embeddings: Full-dimensionality embeddings to save (optional).
            embedding_backend: Backend name for embedding file naming (e.g., "gemma", "openai").
            include_empath: Whether to generate and save Empath psycholinguistic controls.
            descriptions: Text descriptions for Empath analysis. Required if include_empath=True.
            empath_n_components: Number of SVD components for Empath controls (default: 10).
            include_word_baseline: Whether to generate word embedding baseline for comparison.
            region_labels: Region labels for word baseline. Required if include_word_baseline=True.
            varietal_labels: Varietal labels for word baseline. Required if include_word_baseline=True.

        Returns:
            Dictionary mapping output names to file paths.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If include_empath=True but descriptions not provided.
            ValueError: If include_word_baseline=True but labels not provided.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before export()")

        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        embeddings_dir = output_dir / "embeddings"
        representations_dir = output_dir / "representations"
        model_dir = output_dir / "model"
        metadata_dir = output_dir / "metadata"

        for subdir in [embeddings_dir, representations_dir, model_dir, metadata_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Default the embedding-file backend tag to the backend used at fit() time.
        if embedding_backend is None:
            embedding_backend = self.config.embedding_backend

        # Save full embeddings if provided
        if full_embeddings is not None:
            embedding_dim = full_embeddings.shape[1]
            path = embeddings_dir / f"full_embedding_{embedding_backend}_{embedding_dim}.npy"
            np.save(path, full_embeddings)
            outputs["full_embedding"] = str(path)

        # Save attribute embeddings (representations)
        for name in self.attributes:
            path = representations_dir / f"{name}_embedding.npy"
            np.save(path, self.attribute_embeddings[name])
            outputs[f"{name}_embedding"] = str(path)

        # Save nuisance
        path = representations_dir / "nuisance_embedding.npy"
        np.save(path, self.nuisance_embedding)
        outputs["nuisance_embedding"] = str(path)

        # Save encoder
        path = metadata_dir / "encoder.json"
        self.encoder.save(str(path))
        outputs["encoder"] = str(path)

        # Save hyperparameters
        path = metadata_dir / "hyperparameters.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.hp_dict, f, indent=2)
        outputs["hyperparameters"] = str(path)

        # Save config
        path = metadata_dir / "config.json"
        self.config.save(str(path))
        outputs["config"] = str(path)

        # Save classification metrics (copy to output dir if different from self.output_dir)
        if self.classification_metrics is not None:
            metrics_path = metadata_dir / "classification_metrics.json"
            save_classification_metrics(self.classification_metrics, str(metrics_path))
            outputs["classification_metrics"] = str(metrics_path)

        # Generate and save Empath controls if requested
        empath_diagnostics = None
        if include_empath:
            if descriptions is None:
                raise ValueError(
                    "descriptions must be provided when include_empath=True"
                )
            if self.verbose:
                print("\n--- Generating Empath psycholinguistic controls ---")

            # Convert to list if needed
            if isinstance(descriptions, pd.Series):
                descriptions = descriptions.tolist()
            descriptions = list(descriptions)

            empath_controls, empath_diagnostics = generate_empath_controls(
                descriptions=descriptions,
                n_components=empath_n_components,
                random_state=self.config.seed,
                verbose=self.verbose,
            )

            empath_outputs = save_empath_controls(
                controls=empath_controls,
                diagnostics=empath_diagnostics,
                output_dir=str(output_dir),
                verbose=self.verbose,
            )
            outputs.update(empath_outputs)

        # Generate and save word embedding baseline if requested
        if include_word_baseline:
            if region_labels is None or varietal_labels is None:
                raise ValueError(
                    "region_labels and varietal_labels must be provided when include_word_baseline=True"
                )
            if self.verbose:
                print("\n--- Generating word embedding baseline ---")

            # Convert to list if needed
            if isinstance(region_labels, pd.Series):
                region_labels = region_labels.tolist()
            if isinstance(varietal_labels, pd.Series):
                varietal_labels = varietal_labels.tolist()
            region_labels = list(region_labels)
            varietal_labels = list(varietal_labels)

            word_baseline_results = generate_word_embedding_baseline(
                region_labels=region_labels,
                varietal_labels=varietal_labels,
                output_dir=str(output_dir),
                verbose=self.verbose,
            )
            # generate_word_embedding_baseline returns {model: {"embeddings": ...,
            # "outputs": {filename: path}}}; merge each model's saved file paths
            # into the manifest's file list.
            for model_results in word_baseline_results.values():
                outputs.update(model_results.get("outputs", {}))

        # Save manifest
        manifest = {
            "attributes": self.attributes,
            "n_samples": len(self.attribute_embeddings[self.attributes[0]]),
            "embedding_dim_per_attribute": self.config.embedding_units_per_attribute,
            "num_nuisance_dims": self.config.num_nuisance_dims,
            "embedding_backend": embedding_backend,
            "full_embedding_dim": full_embeddings.shape[1] if full_embeddings is not None else None,
            "classification_metrics": self.classification_metrics,
            "files": outputs,
        }
        # Add Empath metadata if generated
        if empath_diagnostics is not None:
            manifest["empath"] = {
                "n_components": empath_diagnostics["n_components"],
                "n_categories": empath_diagnostics["n_categories"],
                "total_explained_variance": float(empath_diagnostics["total_explained_variance"]),
                "explained_variance_ratio": empath_diagnostics["explained_variance_ratio"].tolist(),
            }
        path = metadata_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        outputs["manifest"] = str(path)

        if self.verbose:
            print(f"\nExported to {output_dir}:")
            for name, fpath in outputs.items():
                print(f"  {name}: {fpath}")

        return outputs

    def save(self, path: str | None = None) -> str:
        """Save pipeline state (model + encoder).

        Args:
            path: Path for the checkpoint file. If None, saves to
                output_dir/model/bridge_model.pt.

        Returns:
            Path to the saved checkpoint file.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before save()")

        if path is None:
            model_dir = self.output_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            path = str(model_dir / "bridge_model.pt")

        save_model(self.model, path)
        return path

    @classmethod
    def load(cls, path: str, config: BRIDGEConfig | None = None) -> "BRIDGEPipeline":
        """Load pipeline from saved state.

        Note: The encoder and embeddings need to be reloaded separately
        if needed for full functionality.

        Args:
            path: Path to the checkpoint file.
            config: Optional configuration override.

        Returns:
            BRIDGEPipeline instance with loaded model.
        """
        model = load_model(path)
        hp_dict = model.get_hp_dict()

        pipeline = cls(
            attributes=hp_dict["attribute_names"],
            config=config,
        )
        pipeline.model = model
        pipeline.hp_dict = hp_dict
        pipeline.is_fitted = True  # Mark as fitted so transform/export work
        # Note: encoder and embeddings need to be reloaded separately

        return pipeline
