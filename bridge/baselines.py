"""
Baseline embedding methods for comparison with BRIDGE.

This module provides BERT and RoBERTa baselines that embed attribute labels only
(not full descriptions) to demonstrate why full description embeddings are necessary.

The baseline embeds only the short attribute labels (e.g., "Pinot Noir", "France Bordeaux")
rather than the full wine descriptions. This addresses reviewer comments about using
"off-the-shelf, pre-trained embeddings (e.g., from BERT or RoBERTa)".

Example usage:
    from bridge.baselines import TransformerBaseline, generate_transformer_baselines

    # Generate both BERT and RoBERTa baselines
    results = generate_transformer_baselines(
        region_labels=df['country_province'].tolist(),
        varietal_labels=df['variety'].tolist(),
        output_dir="./bridge_output",
    )

    # Access embeddings
    print(results['bert']['region'].shape)     # (N, 768)
    print(results['bert']['varietal'].shape)   # (N, 768)
    print(results['roberta']['region'].shape)  # (N, 768)
    print(results['roberta']['varietal'].shape)  # (N, 768)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Model name mappings for BERT and RoBERTa
BASELINE_MODELS = {
    "bert": "bert-base-nli-mean-tokens",      # BERT fine-tuned for similarity (768-dim)
    "roberta": "roberta-base-nli-mean-tokens",  # RoBERTa fine-tuned for similarity (768-dim)
}


class TransformerBaseline:
    """
    BERT/RoBERTa baseline for embedding attribute labels.

    Embeds attribute labels (not descriptions) using pre-trained transformer models.
    This directly addresses the reviewer question: "Could off-the-shelf, pre-trained
    embeddings (e.g., from BERT or RoBERTa) be used for the first step?"

    The baseline embeds only the attribute labels:
    - Region: "France#####Bordeaux" -> "France Bordeaux" -> embed() -> 768-dim
    - Varietal: "Pinot Noir" -> embed() -> 768-dim

    This approach is expected to perform poorly compared to BRIDGE because:
    1. Short labels lose full description context (500+ words reduced to 2-3 words)
    2. "Pinot Noir" and "Pinot Grigio" will be similar (shared "Pinot" token)
    3. Region nuance lost: "Napa Valley" lacks wine-specific context
    4. Style/quality information completely absent

    Attributes:
        model: The loaded sentence-transformers model.
        model_type: Either "bert" or "roberta".
        model_name: Full model name used.
        dim: Embedding dimension (768 for both BERT and RoBERTa).

    Example:
        >>> from bridge.baselines import TransformerBaseline
        >>> baseline = TransformerBaseline("bert")
        >>> region_emb = baseline.embed_region_label("France#####Bordeaux")
        >>> print(region_emb.shape)
        (768,)
        >>> varietal_emb = baseline.embed_varietal_label("Pinot Noir")
        >>> print(varietal_emb.shape)
        (768,)
    """

    def __init__(self, model_type: str = "bert"):
        """
        Load BERT or RoBERTa model via sentence-transformers.

        Args:
            model_type: Either "bert" or "roberta".

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If model_type is not "bert" or "roberta".
        """
        if model_type not in BASELINE_MODELS:
            raise ValueError(
                f"model_type must be one of {list(BASELINE_MODELS.keys())}, got '{model_type}'"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for TransformerBaseline. "
                "Install it with: pip install sentence-transformers"
            ) from e

        self.model_type = model_type
        self.model_name = BASELINE_MODELS[model_type]

        print(f"Loading {model_type.upper()} model ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Loaded with {self.dim}-dimensional embeddings")

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string using the transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector of shape (dim,).
        """
        text = text.strip()

        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_region_label(self, label: str) -> np.ndarray:
        """
        Embed region label as a single phrase.

        Converts "France#####Bordeaux" to "France Bordeaux" and embeds as one phrase.

        Args:
            label: Region label in format "country#####province".

        Returns:
            Embedding of shape (dim,) = (768,).
        """
        if not label or label.strip() == "":
            return np.zeros(self.dim, dtype=np.float32)

        # Replace separator with space: "France#####Bordeaux" -> "France Bordeaux"
        clean_label = label.replace("#####", " ").strip()

        # Clean up any double spaces
        while "  " in clean_label:
            clean_label = clean_label.replace("  ", " ")

        return self._embed_text(clean_label)

    def embed_varietal_label(self, label: str) -> np.ndarray:
        """
        Embed varietal label directly.

        Args:
            label: Varietal label (e.g., "Pinot Noir", "Chardonnay").

        Returns:
            Embedding of shape (dim,) = (768,).
        """
        if not label or label.strip() == "":
            return np.zeros(self.dim, dtype=np.float32)

        return self._embed_text(label.strip())

    def embed_all_wines(
        self,
        region_labels: list[str],
        varietal_labels: list[str],
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Embed all wines using their attribute labels.

        Args:
            region_labels: List of region labels in format "country#####province".
            varietal_labels: List of varietal labels (e.g., "Pinot Noir").
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary containing:
                - "region": (N, 768) array of region embeddings
                - "varietal": (N, 768) array of varietal embeddings
                - "combined": (N, 1536) array of concatenated embeddings

        Raises:
            ValueError: If region_labels and varietal_labels have different lengths.
        """
        if len(region_labels) != len(varietal_labels):
            raise ValueError(
                f"region_labels and varietal_labels must have the same length. "
                f"Got {len(region_labels)} and {len(varietal_labels)}."
            )

        n_samples = len(region_labels)
        print(f"\nEmbedding {n_samples} wines using {self.model_type.upper()}...")
        print(f"  Model: {self.model_name}")
        print(f"  Region: 'France#####Bordeaux' -> 'France Bordeaux' -> {self.dim}-dim")
        print(f"  Varietal: 'Pinot Noir' -> {self.dim}-dim")

        # Pre-allocate arrays
        region_embeddings = np.zeros((n_samples, self.dim), dtype=np.float32)
        varietal_embeddings = np.zeros((n_samples, self.dim), dtype=np.float32)

        # Track empty label statistics
        n_region_empty = 0
        n_varietal_empty = 0

        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc=f"{self.model_type.upper()} embeddings")

        for i in iterator:
            # Embed region
            region_emb = self.embed_region_label(region_labels[i])
            region_embeddings[i] = region_emb
            if np.allclose(region_emb, 0):
                n_region_empty += 1

            # Embed varietal
            varietal_emb = self.embed_varietal_label(varietal_labels[i])
            varietal_embeddings[i] = varietal_emb
            if np.allclose(varietal_emb, 0):
                n_varietal_empty += 1

        # Combine embeddings
        combined_embeddings = np.concatenate(
            [region_embeddings, varietal_embeddings], axis=1
        )

        print(f"  Region empty (zeros): {n_region_empty} / {n_samples} "
              f"({100*n_region_empty/n_samples:.1f}%)")
        print(f"  Varietal empty (zeros): {n_varietal_empty} / {n_samples} "
              f"({100*n_varietal_empty/n_samples:.1f}%)")

        return {
            "region": region_embeddings,
            "varietal": varietal_embeddings,
            "combined": combined_embeddings,
        }

    def save(
        self,
        embeddings: dict[str, np.ndarray],
        output_dir: str,
        verbose: bool = True,
    ) -> dict[str, str]:
        """
        Save embeddings to bridge_output/baselines/{model_type}/

        Creates the following files:
            - {model_type}_region.npy: (N, 768) region embeddings
            - {model_type}_varietal.npy: (N, 768) varietal embeddings
            - {model_type}_combined.npy: (N, 1536) combined embeddings
            - {model_type}_config.json: Model info, dimensions, etc.

        Args:
            embeddings: Dictionary from embed_all_wines().
            output_dir: Base output directory
                (baselines/{model_type}/ subdirectory will be created).
            verbose: Whether to print file paths.

        Returns:
            Dictionary mapping output names to file paths.
        """
        output_path = Path(output_dir)
        baselines_dir = output_path / "baselines" / self.model_type
        baselines_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}
        prefix = self.model_type

        # Save region embeddings
        region_path = baselines_dir / f"{prefix}_region.npy"
        np.save(region_path, embeddings["region"])
        outputs[f"{prefix}_region"] = str(region_path)

        # Save varietal embeddings
        varietal_path = baselines_dir / f"{prefix}_varietal.npy"
        np.save(varietal_path, embeddings["varietal"])
        outputs[f"{prefix}_varietal"] = str(varietal_path)

        # Save combined embeddings
        combined_path = baselines_dir / f"{prefix}_combined.npy"
        np.save(combined_path, embeddings["combined"])
        outputs[f"{prefix}_combined"] = str(combined_path)

        # Save config
        config = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "embedding_dim": self.dim,
            "region_dim": self.dim,
            "varietal_dim": self.dim,
            "combined_dim": self.dim * 2,
            "n_samples": embeddings["region"].shape[0],
            "region_encoding": "embed('France Bordeaux') - separator replaced with space",
            "varietal_encoding": "embed('Pinot Noir') - direct embedding",
            "note": (
                "Baseline using BERT/RoBERTa on attribute labels only (not descriptions). "
                "Addresses reviewer question about using off-the-shelf pre-trained embeddings."
            ),
        }

        config_path = baselines_dir / f"{prefix}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        outputs[f"{prefix}_config"] = str(config_path)

        if verbose:
            print(f"\nSaved {self.model_type.upper()} baseline to {baselines_dir}/")
            print(f"  {prefix}_region.npy: {embeddings['region'].shape}")
            print(f"  {prefix}_varietal.npy: {embeddings['varietal'].shape}")
            print(f"  {prefix}_combined.npy: {embeddings['combined'].shape}")
            print(f"  {prefix}_config.json")

        return outputs


def generate_transformer_baselines(
    region_labels: list[str],
    varietal_labels: list[str],
    models: list[str] | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Generate BERT and RoBERTa baselines for comparison with BRIDGE.

    This function creates baseline embeddings using pre-trained BERT and RoBERTa
    models on attribute labels only (not full descriptions). This directly addresses
    the reviewer question: "Could off-the-shelf, pre-trained embeddings (e.g., from
    BERT or RoBERTa) be used for the first step?"

    Args:
        region_labels: List of region labels in format "country#####province".
        varietal_labels: List of varietal labels (e.g., "Pinot Noir").
        models: List of model types to use. Default: ["bert", "roberta"].
        output_dir: If provided, save embeddings to this directory.
        verbose: Whether to print progress information.

    Returns:
        Dictionary mapping model type to results:
            {
                "bert": {
                    "embeddings": {"region": array, "varietal": array, "combined": array},
                    "outputs": {"bert_region": path, ...}  # if output_dir provided
                },
                "roberta": {
                    "embeddings": {"region": array, "varietal": array, "combined": array},
                    "outputs": {"roberta_region": path, ...}  # if output_dir provided
                }
            }

    Example:
        >>> from bridge.baselines import generate_transformer_baselines
        >>> results = generate_transformer_baselines(
        ...     region_labels=df['country_province'].tolist(),
        ...     varietal_labels=df['variety'].tolist(),
        ...     output_dir="./bridge_output",
        ... )
        >>> bert_region = results['bert']['embeddings']['region']  # (N, 768)
        >>> roberta_varietal = results['roberta']['embeddings']['varietal']  # (N, 768)
    """
    if models is None:
        models = ["bert", "roberta"]
    results = {}

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Generating {model_type.upper()} baseline")
        print(f"{'='*60}")

        # Create baseline
        baseline = TransformerBaseline(model_type=model_type)

        # Generate embeddings
        embeddings = baseline.embed_all_wines(
            region_labels=region_labels,
            varietal_labels=varietal_labels,
            show_progress=verbose,
        )

        # Save if output_dir provided
        outputs = {}
        if output_dir is not None:
            outputs = baseline.save(
                embeddings=embeddings,
                output_dir=output_dir,
                verbose=verbose,
            )

        results[model_type] = {
            "embeddings": embeddings,
            "outputs": outputs,
        }

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Baseline Generation Complete")
        print(f"{'='*60}")
        for model_type in models:
            emb = results[model_type]["embeddings"]
            print(f"\n{model_type.upper()}:")
            print(f"  Region:   {emb['region'].shape}")
            print(f"  Varietal: {emb['varietal'].shape}")
            print(f"  Combined: {emb['combined'].shape}")

    return results


# Backwards compatibility alias
WordEmbeddingBaseline = TransformerBaseline
generate_word_embedding_baseline = generate_transformer_baselines
