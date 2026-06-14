"""
LLM-based description augmentation for data-poor applications.

This module provides tools to generate variations of product descriptions using
large language models, enabling BRIDGE training on small datasets by augmenting
baseline descriptions.

Supports two backends:
- Ollama (local, default): Free, private, uses local GPU
- OpenAI API: Best quality, API cost

Example usage:
    from bridge.augmentation import augment_descriptions_sync, CONCISE, DESCRIPTIVE

    result = augment_descriptions_sync(
        base_descriptions=["A fruity coffee with sweet notes."],
        num_variations=100,
        strategies=[CONCISE, DESCRIPTIVE],
        backend="ollama",
        system_prompt="You are a specialty coffee expert.",
        cache_path="augmented_coffee.json",
    )

    print(len(result))  # 1 base x 100 variations x 2 strategies = 200
    augmented_df = result.to_dataframe()
"""

import asyncio
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

# Optional backend imports with availability flags
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from openai import (
        APIConnectionError,
        AsyncOpenAI,
        RateLimitError,
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from tqdm.asyncio import tqdm_asyncio
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# Constants and Defaults
# =============================================================================

# Recommended models for Mac Studio M2 Ultra (128GB RAM)
RECOMMENDED_MODELS = {
    "qwen2.5:32b-instruct-q8_0": {
        "ram": "~34 GB",
        "best_for": "High precision, low hallucination (DEFAULT)",
    },
    "llama3.3:70b-instruct-q3_K_M": {
        "ram": "~34 GB",
        "best_for": "Complex reasoning, creative writing",
    },
    "gemma2:27b-instruct-q8_0": {
        "ram": "~29 GB",
        "best_for": "Good reasoning, leaves RAM headroom",
    },
    "mixtral:8x7b-instruct-v0.1-q6_K": {
        "ram": "~38 GB",
        "best_for": "Speed, long-context summarization",
    },
}

DEFAULT_OLLAMA_MODEL = "qwen2.5:32b-instruct-q8_0"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates product descriptions. "
    "Generate natural, authentic-sounding descriptions that maintain "
    "the core attributes while varying the style and phrasing."
)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class VariationStrategy:
    """Defines how to generate a variation of a description.

    Generalizes beyond hardcoded "Factual/Engaging/Creative" styles to allow
    user-defined variation types.

    Attributes:
        name: Strategy identifier (e.g., "concise", "elaborate", "technical").
        instruction: What makes this variation distinct (used in the prompt).
        temperature: Sampling temperature (0-2). Higher = more creative.
        max_tokens: Maximum response length.

    Example:
        >>> strategy = VariationStrategy(
        ...     name="poetic",
        ...     instruction="metaphorical, evocative, with sensory imagery",
        ...     temperature=0.9,
        ...     max_tokens=200,
        ... )
    """
    name: str
    instruction: str
    temperature: float = 0.7
    max_tokens: int = 200

    def __post_init__(self):
        """Validate strategy parameters after initialization.

        Raises:
            ValueError: If temperature is not in [0, 2] or max_tokens < 1.
        """
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class AugmentationResult:
    """Container for augmentation outputs.

    Attributes:
        descriptions: All generated descriptions.
        base_indices: Index of the base description for each generated one.
        strategy_names: Strategy name used for each generated description.
        metadata: Generation metadata (backend, model, timestamp, etc.).
    """
    descriptions: list[str]
    base_indices: list[int]
    strategy_names: list[str]
    metadata: dict[str, Any]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to DataFrame for downstream processing.

        Returns:
            DataFrame with columns: description, base_index, strategy.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install via: pip install pandas"
            )
        return pd.DataFrame({
            "description": self.descriptions,
            "base_index": self.base_indices,
            "strategy": self.strategy_names,
        })

    def __len__(self) -> int:
        return len(self.descriptions)

    def save(self, path: str) -> None:
        """Save augmentation results to JSON file.

        Args:
            path: Path to save the JSON file.
        """
        data = {
            "descriptions": self.descriptions,
            "base_indices": self.base_indices,
            "strategy_names": self.strategy_names,
            "metadata": self.metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Augmentation saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AugmentationResult":
        """Load augmentation results from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            AugmentationResult loaded from file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Augmentation loaded from {path} ({len(data['descriptions'])} descriptions)")
        return cls(
            descriptions=data["descriptions"],
            base_indices=data["base_indices"],
            strategy_names=data["strategy_names"],
            metadata=data["metadata"],
        )


# =============================================================================
# Pre-built Strategies
# =============================================================================

CONCISE = VariationStrategy(
    name="concise",
    instruction="extremely brief, factual, minimal words, no embellishments",
    temperature=0.3,
    max_tokens=50,
)

DESCRIPTIVE = VariationStrategy(
    name="descriptive",
    instruction="detailed, rich sensory language, vivid imagery",
    temperature=0.7,
    max_tokens=200,
)

TECHNICAL = VariationStrategy(
    name="technical",
    instruction="precise, domain-specific terminology, expert tone",
    temperature=0.5,
    max_tokens=150,
)

CREATIVE = VariationStrategy(
    name="creative",
    instruction="imaginative, metaphorical, poetic, unexpected comparisons",
    temperature=0.9,
    max_tokens=250,
)

DEFAULT_STRATEGIES = [CONCISE, DESCRIPTIVE]


# =============================================================================
# Utility Functions
# =============================================================================

def _check_backend_available(backend: str) -> None:
    """Raise ImportError if backend dependencies are missing.

    Args:
        backend: Backend name ("ollama" or "openai").

    Raises:
        ImportError: If required dependencies are not installed.
    """
    if backend == "ollama" and not AIOHTTP_AVAILABLE:
        raise ImportError(
            "aiohttp is required for Ollama backend. "
            "Install via: pip install aiohttp"
        )
    if backend == "openai" and not OPENAI_AVAILABLE:
        raise ImportError(
            "openai is required for OpenAI backend. "
            "Install via: pip install openai"
        )


async def check_ollama_available(
    host: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> tuple[bool, str | None]:
    """Check if Ollama service is running and accessible.

    Args:
        host: Ollama API host URL.
        timeout: Connection timeout in seconds.

    Returns:
        Tuple of (is_available, error_message).
        If available, error_message is None.
    """
    if not AIOHTTP_AVAILABLE:
        return False, "aiohttp not installed"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    return True, None
                return False, f"Ollama returned status {resp.status}"
    except aiohttp.ClientConnectorError:
        return False, (
            f"Cannot connect to Ollama at {host}. "
            "Is Ollama running? Start with: ollama serve"
        )
    except asyncio.TimeoutError:
        return False, f"Ollama connection timed out after {timeout}s"
    except Exception as e:  # pylint: disable=broad-exception-caught  # report any backend failure as unavailable
        return False, f"Ollama check failed: {e}"


def get_openai_client(api_key: str | None = None) -> "AsyncOpenAI":
    """Initialize and return an async OpenAI client.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY
            environment variable.

    Returns:
        Initialized AsyncOpenAI client.

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

    return AsyncOpenAI(api_key=api_key)


def _build_prompt(
    base_description: str,
    strategy: VariationStrategy,
    attributes: dict[str, str] | None = None,
) -> str:
    """Build the user prompt for variation generation.

    Args:
        base_description: Original description to vary.
        strategy: VariationStrategy defining the style.
        attributes: Optional dict of attribute name -> value for context.

    Returns:
        Formatted prompt string.
    """
    attr_context = ""
    if attributes:
        attr_lines = [f"- {k}: {v}" for k, v in attributes.items()]
        attr_context = "\n\nProduct attributes:\n" + "\n".join(attr_lines)

    return f"""Create a new product description based on this original:

"{base_description}"{attr_context}

Generate a variation that is: {strategy.instruction}

Return ONLY the new description, with no additional commentary."""


def _validate_response(
    response: str | None,
    min_length: int = 10,
) -> bool:
    """Validate that LLM response is usable.

    Args:
        response: Generated text to validate.
        min_length: Minimum acceptable response length.

    Returns:
        True if response is valid, False otherwise.
    """
    if not response:
        return False

    response = response.strip()

    # Check minimum length
    if len(response) < min_length:
        return False

    # Check for common LLM refusal patterns
    refusal_markers = [
        "I cannot",
        "I'm sorry",
        "I am sorry",
        "As an AI",
        "I'm not able",
        "I am not able",
        "I apologize",
        "cannot assist",
        "can't assist",
    ]
    response_lower = response.lower()
    if any(marker.lower() in response_lower for marker in refusal_markers):
        return False

    return True


def _get_retryable_errors() -> tuple:
    """Get tuple of retryable exception types.

    Returns:
        Tuple of exception classes that should trigger retry.
    """
    errors = [asyncio.TimeoutError]
    if AIOHTTP_AVAILABLE:
        errors.extend([
            aiohttp.ClientError,
            aiohttp.ServerDisconnectedError,
        ])
    if OPENAI_AVAILABLE:
        errors.extend([RateLimitError, APIConnectionError])
    return tuple(errors)


# =============================================================================
# Backend Generation Functions
# =============================================================================

async def _generate_ollama(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    host: str = "http://localhost:11434",
    session: "aiohttp.ClientSession" | None = None,
) -> str:
    """Generate using Ollama API.

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        model: Ollama model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        semaphore: Concurrency limiter.
        host: Ollama API host URL.
        session: Optional shared aiohttp session (for connection pooling).

    Returns:
        Generated text response.

    Raises:
        aiohttp.ClientError: On connection failure.
        KeyError: If response format is unexpected.
    """
    async with semaphore:
        # Use provided session or create new one
        should_close = session is None
        if session is None:
            session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["response"].strip()
        finally:
            if should_close:
                await session.close()


async def _generate_openai(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    client: "AsyncOpenAI",
) -> str:
    """Generate using OpenAI API.

    Args:
        prompt: User prompt.
        system_prompt: System prompt.
        model: OpenAI model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        semaphore: Concurrency limiter.
        client: Initialized AsyncOpenAI client.

    Returns:
        Generated text response.
    """
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


async def _generate_with_retry(
    generate_fn: Callable,
    max_retries: int,
    retry_delay: float,
    verbose: bool = False,
    **kwargs,
) -> str | None:
    """Wrapper with exponential backoff retry for retryable errors.

    Args:
        generate_fn: Async generation function to call.
        max_retries: Maximum retry attempts.
        retry_delay: Initial delay between retries (doubles each retry).
        verbose: Print retry messages.
        **kwargs: Arguments to pass to generate_fn.

    Returns:
        Generated text, or None if all retries failed.
    """
    retryable_errors = _get_retryable_errors()
    delay = retry_delay

    for attempt in range(max_retries):
        try:
            result = await generate_fn(**kwargs)
            # Validate output
            if _validate_response(result):
                return result
            # Invalid response, retry
            if verbose:
                print(f"  Invalid response, retrying ({attempt + 1}/{max_retries})")

        except retryable_errors as e:
            if attempt == max_retries - 1:
                warnings.warn(f"Failed after {max_retries} attempts: {e}")
                return None
            if verbose:
                print(f"  {type(e).__name__}, retrying in {delay:.1f}s "
                      f"({attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

        except Exception as e:  # pylint: disable=broad-exception-caught  # any non-retryable API error aborts this item
            # Non-retryable error (e.g., AuthenticationError, NotFoundError)
            warnings.warn(f"Non-retryable error: {type(e).__name__}: {e}")
            return None

    return None


# =============================================================================
# Main API
# =============================================================================

async def augment_descriptions(
    base_descriptions: list[str],
    num_variations: int = 10,
    strategies: list[VariationStrategy] | None = None,
    backend: Literal["ollama", "openai"] = "ollama",
    model: str | None = None,
    system_prompt: str | None = None,
    base_attributes: dict[str, list[str]] | None = None,
    # Backend configuration
    ollama_host: str = "http://localhost:11434",
    api_key: str | None = None,
    # Concurrency and retry
    concurrent_limit: int = 10,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    # Caching
    cache_path: str | None = None,
    use_cache: bool = True,
    # Other options
    show_progress: bool = True,
    verbose: bool = True,
) -> AugmentationResult:
    """Generate variations of base descriptions using an LLM.

    Creates multiple variations of each base description using specified
    strategies. For data-poor applications, this enables training BRIDGE
    on small datasets by augmenting baseline descriptions.

    Args:
        base_descriptions: List of seed descriptions to augment.
        num_variations: Number of variations per base description per strategy.
        strategies: List of VariationStrategy objects. Default: [CONCISE, DESCRIPTIVE].
        backend: "ollama" (local, default) or "openai" (API).
        model: Model name. Default: qwen2.5:32b-instruct-q8_0 (ollama) or gpt-5.5 (openai).
        system_prompt: Domain-specific context for the LLM.
            Example: "You are a specialty coffee expert writing product descriptions."
        base_attributes: Optional dict mapping attribute names to values per base description.
            Used to provide context in the prompt.
            Example: {"aroma": ["fruity", "nutty", ...], "taste": ["sweet", "bitter", ...]}
        ollama_host: Ollama API host URL. Default: "http://localhost:11434".
            Allows using remote Ollama instances.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
        concurrent_limit: Max concurrent LLM calls. Default 10 for local Ollama.
            Set to 1 for fully sequential generation (useful for low-RAM systems).
            For OpenAI backend, can increase to 50+ (rate limits handled by retries).
        max_retries: Retry attempts on failure.
        retry_delay: Initial delay between retries (exponential backoff).
        cache_path: Path to save/load cached results. If None, no caching.
        use_cache: If True and cache_path exists, load from cache instead of generating.
        show_progress: Show tqdm progress bar.
        verbose: Print status messages.

    Returns:
        AugmentationResult with all generated descriptions and metadata.

    Raises:
        ImportError: If required backend dependencies are missing.
        ValueError: If OpenAI backend is used without API key.
        RuntimeError: If Ollama service is not running (for ollama backend).

    Example:
        >>> from bridge.augmentation import augment_descriptions, CONCISE, DESCRIPTIVE
        >>>
        >>> result = await augment_descriptions(
        ...     base_descriptions=["A fruity coffee with sweet notes."],
        ...     num_variations=100,
        ...     strategies=[CONCISE, DESCRIPTIVE],
        ...     backend="ollama",
        ...     system_prompt="You are a specialty coffee expert.",
        ...     cache_path="augmented_coffee.json",
        ... )
        >>> print(len(result))  # 1 base x 100 variations x 2 strategies = 200
        200
    """
    # 1. Check cache first
    if use_cache and cache_path and Path(cache_path).exists():
        if verbose:
            print(f"Loading cached augmentation from {cache_path}")
        return AugmentationResult.load(cache_path)

    # 2. Validate backend availability
    _check_backend_available(backend)

    # 3. For Ollama, check service is running
    if backend == "ollama":
        available, error = await check_ollama_available(ollama_host)
        if not available:
            raise RuntimeError(f"Ollama not available: {error}")

    # 4. Initialize client (OpenAI) or session (Ollama)
    client = None
    session = None
    if backend == "openai":
        client = get_openai_client(api_key)
    else:
        session = aiohttp.ClientSession()

    try:
        # 5. Set up defaults
        strategies = strategies or DEFAULT_STRATEGIES
        model = model or (DEFAULT_OLLAMA_MODEL if backend == "ollama" else DEFAULT_OPENAI_MODEL)
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if verbose:
            print(f"Augmenting {len(base_descriptions)} descriptions via {backend} ({model})")
            print(f"  Variations per description: {num_variations}")
            print(f"  Strategies: {[s.name for s in strategies]}")
            expected_total = len(base_descriptions) * num_variations * len(strategies)
            print(f"  Expected total: {expected_total}")

        # 6. Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)

        # 7. Build task list: (base_idx, strategy, variation_num)
        tasks = []
        for base_idx in range(len(base_descriptions)):
            for strategy in strategies:
                for var_num in range(num_variations):
                    tasks.append((base_idx, strategy, var_num))

        # 8. Prepare attribute lookup if provided
        def get_attributes(base_idx: int) -> dict[str, str] | None:
            if not base_attributes:
                return None
            return {
                attr_name: values[base_idx]
                for attr_name, values in base_attributes.items()
            }

        # 9. Define generation task
        async def generate_one(
            task: tuple[int, VariationStrategy, int],
        ) -> tuple[int, str, str] | None:
            base_idx, strategy, _var_num = task
            prompt = _build_prompt(
                base_descriptions[base_idx],
                strategy,
                get_attributes(base_idx),
            )

            if backend == "ollama":
                result = await _generate_with_retry(
                    _generate_ollama,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    verbose=verbose and not show_progress,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=strategy.temperature,
                    max_tokens=strategy.max_tokens,
                    semaphore=semaphore,
                    host=ollama_host,
                    session=session,
                )
            else:
                result = await _generate_with_retry(
                    _generate_openai,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    verbose=verbose and not show_progress,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=strategy.temperature,
                    max_tokens=strategy.max_tokens,
                    semaphore=semaphore,
                    client=client,
                )

            if result:
                return (base_idx, strategy.name, result)
            return None

        # 10. Execute with progress bar
        if show_progress and TQDM_AVAILABLE:
            coroutines = [generate_one(task) for task in tasks]
            results = await tqdm_asyncio.gather(*coroutines, desc="Augmenting")
        else:
            results = await asyncio.gather(*[generate_one(task) for task in tasks])

        # Filter out failures
        results = [r for r in results if r is not None]

        if verbose:
            print(f"Generated {len(results)} / {len(tasks)} descriptions")

        # 11. Build result object
        augmentation_result = AugmentationResult(
            descriptions=[r[2] for r in results],
            base_indices=[r[0] for r in results],
            strategy_names=[r[1] for r in results],
            metadata={
                "backend": backend,
                "model": model,
                "num_base": len(base_descriptions),
                "num_variations": num_variations,
                "strategies": [s.name for s in strategies],
                "concurrent_limit": concurrent_limit,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # 12. Save to cache if path provided
        if cache_path:
            augmentation_result.save(cache_path)

        return augmentation_result

    finally:
        # Clean up session
        if session is not None:
            await session.close()


def augment_descriptions_sync(
    base_descriptions: list[str],
    **kwargs,
) -> AugmentationResult:
    """Synchronous wrapper for augment_descriptions.

    Convenience function for non-async contexts. All arguments are passed
    through to augment_descriptions().

    Args:
        base_descriptions: List of seed descriptions to augment.
        **kwargs: Additional arguments for augment_descriptions().

    Returns:
        AugmentationResult with all generated descriptions and metadata.

    Example:
        >>> from bridge.augmentation import augment_descriptions_sync, CONCISE
        >>>
        >>> result = augment_descriptions_sync(
        ...     base_descriptions=["A fruity coffee with sweet notes."],
        ...     num_variations=10,
        ...     strategies=[CONCISE],
        ... )
    """
    return asyncio.run(augment_descriptions(base_descriptions, **kwargs))


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    "VariationStrategy",
    "AugmentationResult",
    # Functions
    "augment_descriptions",
    "augment_descriptions_sync",
    "check_ollama_available",
    "get_openai_client",
    # Pre-built strategies
    "CONCISE",
    "DESCRIPTIVE",
    "TECHNICAL",
    "CREATIVE",
    "DEFAULT_STRATEGIES",
    # Constants
    "RECOMMENDED_MODELS",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OPENAI_MODEL",
    # Availability flags
    "AIOHTTP_AVAILABLE",
    "OPENAI_AVAILABLE",
    "TQDM_AVAILABLE",
]
