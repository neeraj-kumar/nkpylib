"""A fully-functional cacher with all the bells-and-whistles.

Implementation consists of:
- Cache: Main class managing multiple backends and policies
- CacheBackend: Base class for storage+formatter combinations
- CachePolicy: Base class for cache policies (TTL, limits, etc)
- CacheFormatter: Base class for serialization formats

Design for new version:
- different key_funcs
- runnable in background/async/futures/threads
- batchable
- decorable
- ignore certain args
- cache a list or dict:
  - figure out which are already cached and which aren't
  - where underlying function takes a batch
- use tempfile + rename when writing files
- JSON encoder/decoder class
- something for imdb data dump updates -> either run function or read from db/cache?
- expiration criteria
  - time
  - count
  - memory
  - other?
- single-value cache with different keys
  - e.g. the embeddings cache which checks for current normed, scale_mean, scale_std
- TTL
- ignore cache for individual calls
- force invalidate, either single key or all
- archival
- expiration
- delay + variance
- different formats - json, pickle, ...
- different backing stores - mem, fs, lmdb, numpylmdb
- one file per key, or one file overall, or ...?
- stats/timing
- prefetch?
- caching binary files (e.g. web fetch request)
- per-host timers (like in make_request)?
- works on class methods (how to check for other instance var dependencies?)
- store revisions?
- named revisions?
- external dependencies:
    external_counter = 0
    @cache(depends_on=lambda:[external_counter])
    def things_with_external(a,b,c):
        global external_counter
        from time import sleep; sleep(1) # <- simulating a long-running process
        return external_counter + a + b + c
"""

from __future__ import annotations

import time

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic

class CacheFormatter(ABC):
    """Base class for serialization formats."""
    @abstractmethod
    def dumps(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass

    @abstractmethod
    def loads(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass

KeyT = TypeVar('KeyT')

class CacheBackend(ABC, Generic[KeyT]):
    """Base class for storage backends.

    Each backend is initialized with a formatter that handles serialization.
    """
    def __init__(self, formatter: CacheFormatter):
        self.formatter = formatter

    @abstractmethod
    def get(self, key: KeyT) -> Optional[Any]:
        """Get value for key, returns None if not found."""
        pass

    @abstractmethod
    def set(self, key: KeyT, value: Any) -> None:
        """Set value for key."""
        pass

    @abstractmethod
    def delete(self, key: KeyT) -> None:
        """Delete key."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass


class CachePolicy(ABC, Generic[KeyT]):
    """Base class for cache policies."""
    @abstractmethod
    def should_cache(self, key: KeyT, value: Any) -> bool:
        """Return True if value should be cached."""
        pass

    @abstractmethod
    def should_evict(self, key: KeyT, value: Any, metadata: dict) -> bool:
        """Return True if entry should be evicted."""
        pass

class Cacher(Generic[KeyT]):
    """Main cacher class supporting multiple backends and policies."""
    def __init__(self,
                 backends: list[CacheBackend],
                 policies: Optional[list[CachePolicy]] = None):
        self.backends = backends
        self.policies = policies or []
        self.stats: dict[str, int] = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: KeyT) -> Optional[Any]:
        """Get value from first backend that has it."""
        for backend in self.backends:
            value = backend.get(key)
            if value is not None:
                self.stats['hits'] += 1
                return value
        self.stats['misses'] += 1
        return None

    def set(self, key: KeyT, value: Any) -> None:
        """Set value in all backends if policies allow."""
        metadata = {'timestamp': time.time()}

        # Check if we should cache
        for policy in self.policies:
            if not policy.should_cache(key, value):
                return

        # Check if we should evict
        for policy in self.policies:
            if policy.should_evict(key, value, metadata):
                self.delete(key)
                self.stats['evictions'] += 1
                return

        # Store in all backends
        for backend in self.backends:
            backend.set(key, value)

    def delete(self, key: KeyT) -> None:
        """Delete from all backends."""
        for backend in self.backends:
            backend.delete(key)

    def clear(self) -> None:
        """Clear all backends."""
        for backend in self.backends:
            backend.clear()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self.stats.copy()

# Example concrete implementations would go here:
# - JsonFormatter, PickleFormatter
# - MemoryBackend, FileSystemBackend, LmdbBackend
# - TTLPolicy, CountPolicy, MemoryPolicy
