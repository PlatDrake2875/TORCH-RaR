"""Persistent cache for generated rubrics using DiskCache.

Features:
- Deterministic hashing based on input text + model + domain
- Automatic TTL expiration
- Thread-safe disk storage
- Cache statistics tracking
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from diskcache import Cache

from torch_rar.logging_config import get_logger

logger = get_logger(__name__)


class RubricCache:
    """Persistent cache for generated rubrics.

    Features:
    - Deterministic hashing based on input text + model + domain
    - Automatic TTL expiration
    - Thread-safe disk storage
    - Cache statistics tracking
    """

    def __init__(
        self,
        directory: str = ".cache/rubrics",
        ttl_seconds: int = 2592000,  # 30 days
        size_limit_gb: float = 1.0,
        enabled: bool = True,
    ):
        """Initialize the rubric cache.

        Args:
            directory: Directory for cache files.
            ttl_seconds: Time-to-live for cache entries in seconds.
            size_limit_gb: Maximum cache size in GB.
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds

        if enabled:
            # Create cache directory
            cache_path = Path(directory)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Initialize disk cache
            size_limit_bytes = int(size_limit_gb * 1024 * 1024 * 1024)
            self._cache = Cache(str(cache_path), size_limit=size_limit_bytes)
        else:
            self._cache = None

        self._stats = {"hits": 0, "misses": 0}

    def _create_cache_key(self, text: str, model: str, domain: str) -> str:
        """Create deterministic hash for cache lookup.

        Args:
            text: Input text for rubric generation.
            model: Model name used for generation.
            domain: Domain for prompt templates.

        Returns:
            SHA256 hash of normalized inputs.
        """
        content = json.dumps(
            {
                "text": text.strip().lower(),
                "model": model,
                "domain": domain,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str, domain: str) -> Optional[list]:
        """Retrieve cached rubrics if available.

        Args:
            text: Input text for rubric generation.
            model: Model name used for generation.
            domain: Domain for prompt templates.

        Returns:
            Cached rubrics list if found, None otherwise.
        """
        if not self.enabled or self._cache is None:
            return None

        key = self._create_cache_key(text, model, domain)
        result = self._cache.get(key)

        if result is not None:
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for key {key[:8]}...")
        else:
            self._stats["misses"] += 1

        return result

    def set(self, text: str, model: str, domain: str, rubrics: list) -> None:
        """Store rubrics in cache.

        Args:
            text: Input text for rubric generation.
            model: Model name used for generation.
            domain: Domain for prompt templates.
            rubrics: List of rubric dictionaries to cache.
        """
        if not self.enabled or self._cache is None:
            return

        key = self._create_cache_key(text, model, domain)
        self._cache.set(key, rubrics, expire=self.ttl_seconds)
        logger.debug(f"Cached rubrics for key {key[:8]}...")

    async def get_or_generate(
        self,
        text: str,
        model: str,
        domain: str,
        generator_fn: Callable[[], Any],
    ) -> tuple[list, bool]:
        """Get from cache or generate and cache.

        Args:
            text: Input text for rubric generation.
            model: Model name used for generation.
            domain: Domain for prompt templates.
            generator_fn: Async function to generate rubrics if not cached.

        Returns:
            Tuple of (list of rubrics, was_cached).
        """
        cached = self.get(text, model, domain)
        if cached is not None:
            return cached, True

        # Generate fresh rubrics
        rubrics = await generator_fn()

        # Cache the result (as dicts for serialization)
        rubrics_dicts = [
            r.to_dict() if hasattr(r, "to_dict") else r for r in rubrics
        ]
        self.set(text, model, domain, rubrics_dicts)

        return rubrics, False

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, total requests, hit rate, and cache size.
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        stats = {
            **self._stats,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2%}",
        }

        if self._cache is not None:
            stats["cache_size_mb"] = self._cache.volume() / (1024 * 1024)

        return stats

    def clear(self) -> None:
        """Clear all cached rubrics."""
        if self._cache is not None:
            self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}
        logger.info("Cache cleared")

    def close(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            self._cache.close()
