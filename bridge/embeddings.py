"""
Embedding generation module for BRIDGE pipeline.

Supports two backends:
- OpenAI API (text-embedding-3-large, 3072 dims)
- Local sentence-transformers (embeddinggemma-300m, 768 dims with MRL support)

The embeddinggemma-300m model is recommended for:
- No API costs (runs locally)
- Open weights (Apache 2.0 compatible Gemma license)
- Matryoshka Representation Learning (can truncate to 512/256/128 dims)
- Top performance on summarization tasks (MTEB benchmark)
"""

import os
import time
from typing import Literal

import numpy as np
from tqdm import tqdm

# Optional imports for different backends
try:
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Model configurations
EMBEDDING_MODELS = {
    "openai": {
        "model_id": "text-embedding-3-large",
        "dim": 3072,
        "mrl_dims": None,  # No MRL support
        "backend": "openai",
    },
    "gemma": {
        "model_id": "google/embeddinggemma-300m",
        "dim": 768,
        "mrl_dims": [768, 512, 256, 128],  # Supported MRL dimensions
        "backend": "sentence_transformers",
    },
}

# Default settings
DEFAULT_MODEL = "openai"  # For backward compatibility
DEFAULT_BATCH_SIZE = 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0


def get_openai_client(api_key: str | None = None) -> "OpenAI":
    """Initialize and return an OpenAI client.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY
            environment variable.

    Returns:
        Initialized OpenAI client.

    Raises:
        ImportError: If openai package is not installed.
        ValueError: If API key is not provided and not in environment.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required. Install via: pip install openai"
        )

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment "
            "variable or pass api_key parameter."
        )

    return OpenAI(api_key=api_key)


def get_sentence_transformer_model(
    model_name: str = "google/embeddinggemma-300m",
    truncate_dim: int | None = None,
    device: str | None = None,
) -> "SentenceTransformer":
    """Initialize and return a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.
        truncate_dim: Optional MRL truncation dimension. If specified, embeddings
            will be truncated to this dimension. Must be a valid MRL dimension
            for the model (e.g., 768, 512, 256, 128 for embeddinggemma-300m).
        device: Device to run inference on ('cuda', 'mps', 'cpu'). If None,
            automatically selects best available device.

    Returns:
        Initialized SentenceTransformer model.

    Raises:
        ImportError: If sentence-transformers package is not installed.
        ValueError: If truncate_dim is not a valid MRL dimension for the model.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers package is required. "
            "Install via: pip install sentence-transformers"
        )

    # Validate truncate_dim for embeddinggemma-300m
    if "embeddinggemma" in model_name.lower() and truncate_dim is not None:
        valid_dims = EMBEDDING_MODELS["gemma"]["mrl_dims"]
        if truncate_dim not in valid_dims:
            raise ValueError(
                f"truncate_dim={truncate_dim} is not valid for {model_name}. "
                f"Valid MRL dimensions: {valid_dims}"
            )

    kwargs = {}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    if device is not None:
        kwargs["device"] = device

    return SentenceTransformer(model_name, **kwargs)


def embed_batch_with_retry(
    client: "OpenAI",
    texts: list[str],
    model: str = "text-embedding-3-large",
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
) -> list[list[float]] | None:
    """Get embeddings for a batch of texts with retry logic (OpenAI backend).

    Handles API rate limits and transient errors with exponential backoff.

    Args:
        client: Initialized OpenAI client.
        texts: List of texts to embed.
        model: OpenAI embedding model identifier.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds for exponential backoff.

    Returns:
        List of embedding vectors (each a list of floats), or None if the
        batch failed after all retries.
    """
    # Preprocess texts - replace empties with " " to maintain alignment
    texts_cleaned = []
    for text in texts:
        if isinstance(text, str):
            cleaned = text.replace("\n", " ").strip()
            texts_cleaned.append(cleaned if cleaned else " ")
        else:
            texts_cleaned.append(" ")

    if not texts_cleaned:
        return []

    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            response = client.embeddings.create(input=texts_cleaned, model=model)

            if len(response.data) != len(texts_cleaned):
                print("WARNING: Response length mismatch. Batch failed.")
                return None

            return [item.embedding for item in response.data]

        except (RateLimitError, APIError, APIConnectionError, TimeoutError) as e:
            if retries == max_retries:
                print(f"ERROR: Batch failed after {max_retries} retries: {e}")
                return None

            print(f"Warning: {type(e).__name__}, retrying in {delay}s...")
            time.sleep(delay)
            retries += 1
            delay *= 2

        except Exception as e:  # pylint: disable=broad-exception-caught  # any non-retryable embedding-API error aborts the call
            print(f"ERROR: Non-retryable error: {e}")
            return None

    return None


def generate_embeddings_openai(
    texts: list[str],
    api_key: str | None = None,
    model: str = "text-embedding-3-large",
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Generate embeddings using OpenAI API.

    Args:
        texts: List of texts to embed.
        api_key: OpenAI API key. If None, reads from environment.
        model: OpenAI embedding model identifier.
        batch_size: Number of texts per API call.
        show_progress: Whether to show a progress bar.
        verbose: Whether to print summary statistics.

    Returns:
        Embedding matrix of shape (n_texts, 3072). Rows with failed
        embeddings contain NaN values.
    """
    if verbose:
        print(f"Generating embeddings for {len(texts)} texts via OpenAI...")

    client = get_openai_client(api_key)

    # Process in batches
    all_embeddings = [None] * len(texts)
    num_failed = 0

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings (OpenAI)")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch_with_retry(client, batch_texts, model=model)

        if batch_embeddings and len(batch_embeddings) == len(batch_texts):
            for j, emb in enumerate(batch_embeddings):
                if emb:
                    all_embeddings[i + j] = emb
        else:
            num_failed += 1

    # Convert to matrix
    embedding_matrix = _embeddings_to_matrix(all_embeddings, verbose=verbose)

    if verbose and num_failed > 0:
        print(f"Warning: {num_failed} batches failed")

    return embedding_matrix


def generate_embeddings_local(
    texts: list[str],
    model_name: str = "google/embeddinggemma-300m",
    batch_size: int = 32,
    device: str | None = None,
    show_progress: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Generate embeddings using local sentence-transformers model.

    Always generates at full dimensionality. Dimension reduction is handled
    by the BRIDGE model's EmbeddingTrimmingLayer (mask_size parameter),
    which allows the hyperparameter tuner to optimize the number of
    dimensions to use.

    Uses encode_document() method for embeddinggemma models, which is
    optimized for document (not query) embeddings. This is appropriate
    for BRIDGE where all wine descriptions are compared to each other.

    Args:
        texts: List of texts to embed.
        model_name: HuggingFace model identifier.
        batch_size: Number of texts per forward pass. Lower values use less
            memory but are slower.
        device: Device to run inference on. If None, auto-selects.
        show_progress: Whether to show a progress bar.
        verbose: Whether to print summary statistics.

    Returns:
        Embedding matrix of shape (n_texts, embedding_dim) at full dimensionality.
        For embeddinggemma-300m, this is (n_texts, 768).
    """
    if verbose:
        print(f"Generating embeddings for {len(texts)} texts via {model_name} (full dim)...")

    model = get_sentence_transformer_model(
        model_name=model_name,
        truncate_dim=None,  # Always use full dimensionality
        device=device,
    )

    if verbose:
        print(f"  Model loaded on device: {model.device}")

    # Clean texts (replace newlines, handle empty strings)
    texts_cleaned = [
        str(text).replace("\n", " ") if isinstance(text, str) and text.strip() else ""
        for text in texts
    ]

    # Use encode_document for embeddinggemma models (asymmetric model)
    # For symmetric models, use encode() directly
    if hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            texts_cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    else:
        embeddings = model.encode(
            texts_cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    if verbose:
        print(f"  Embedding shape: {embeddings.shape}")

    return embeddings.astype(np.float32)


def generate_embeddings(
    texts: list[str],
    backend: Literal["openai", "gemma", "auto"] = "auto",
    api_key: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    show_progress: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Unified interface supporting both OpenAI API and local models.
    Always generates at full dimensionality - dimension reduction is
    handled by the BRIDGE model's mask_size parameter, allowing the
    hyperparameter tuner to optimize the number of dimensions.

    Args:
        texts: List of texts to embed.
        backend: Embedding backend to use:
            - "openai": OpenAI API (text-embedding-3-large, 3072 dims)
            - "gemma": Local embeddinggemma-300m (768 dims)
            - "auto": Use gemma if sentence-transformers available, else openai
        api_key: OpenAI API key (only for openai backend).
        model: Model identifier. If None, uses default for backend.
        batch_size: Batch size. Defaults: 1024 for openai, 32 for gemma.
        device: Device for local inference (only for gemma backend).
        show_progress: Whether to show a progress bar.
        verbose: Whether to print summary statistics.

    Returns:
        Embedding matrix of shape (n_texts, embedding_dim) at full dimensionality.
    """
    # Auto-select backend
    if backend == "auto":
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            backend = "gemma"
            if verbose:
                print("Auto-selected backend: gemma (sentence-transformers available)")
        elif OPENAI_AVAILABLE:
            backend = "openai"
            if verbose:
                print("Auto-selected backend: openai")
        else:
            raise ImportError(
                "No embedding backend available. Install either:\n"
                "  pip install openai\n"
                "  pip install sentence-transformers"
            )

    if backend == "openai":
        model = model or EMBEDDING_MODELS["openai"]["model_id"]
        batch_size = batch_size or DEFAULT_BATCH_SIZE
        return generate_embeddings_openai(
            texts=texts,
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            show_progress=show_progress,
            verbose=verbose,
        )

    if backend == "gemma":
        model = model or EMBEDDING_MODELS["gemma"]["model_id"]
        batch_size = batch_size or 32
        return generate_embeddings_local(
            texts=texts,
            model_name=model,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
            verbose=verbose,
        )

    raise ValueError(f"Unknown backend: {backend}. Use 'openai', 'gemma', or 'auto'.")


def _embeddings_to_matrix(
    embeddings_list: list[list[float] | None],
    verbose: bool = True,
) -> np.ndarray:
    """Convert list of embeddings to numpy matrix with NaN for failures.

    Args:
        embeddings_list: List of embedding vectors, where None indicates
            a failed embedding.
        verbose: Whether to print statistics.

    Returns:
        2D numpy array of shape (n_texts, embedding_dim) with NaN values
        for failed embeddings.

    Raises:
        ValueError: If no successful embeddings were generated.
    """
    successful = [i for i, emb in enumerate(embeddings_list) if emb is not None]

    if not successful:
        raise ValueError("No successful embeddings")

    embedding_dim = len(embeddings_list[successful[0]])

    if verbose:
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Successful: {len(successful)} / {len(embeddings_list)}")

    matrix = np.full((len(embeddings_list), embedding_dim), np.nan, dtype=np.float32)

    # Vectorized assignment using fancy indexing
    valid_embeddings = [
        embeddings_list[i] for i in successful
        if len(embeddings_list[i]) == embedding_dim
    ]
    valid_indices = [i for i in successful if len(embeddings_list[i]) == embedding_dim]
    if valid_indices:
        matrix[valid_indices] = np.array(valid_embeddings, dtype=np.float32)

    return matrix


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from numpy file.

    Args:
        path: Path to the .npy file.

    Returns:
        Embedding matrix as a numpy array.
    """
    return np.load(path)


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """Save embeddings to numpy file.

    Args:
        embeddings: Embedding matrix to save.
        path: Path for the output .npy file.
    """
    np.save(path, embeddings)
    print(f"Embeddings saved to {path}")


def get_embedding_dim(backend: str = "openai") -> int:
    """Get the full embedding dimension for a given backend.

    Note: Dimension reduction is handled by the BRIDGE model's mask_size
    parameter, not at embedding generation time. This function returns
    the full dimensionality of the embeddings.

    Args:
        backend: Embedding backend ("openai" or "gemma").

    Returns:
        Full embedding dimension as an integer.
    """
    if backend == "openai":
        return EMBEDDING_MODELS["openai"]["dim"]
    if backend == "gemma":
        return EMBEDDING_MODELS["gemma"]["dim"]
    raise ValueError(f"Unknown backend: {backend}")
