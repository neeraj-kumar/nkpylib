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
- something for imdb data dump updates -> either run function or read from db/cache?
- expiration criteria
  - time (either relative from now, or absolute time)
  - count
  - memory
  - other?
- single-value cache with different keys
  - e.g. the embeddings cache which checks for current normed, scale_mean, scale_std
- TTL
- ignore cache for individual calls
- archival
- delay + variance
- different formats:
  - pickle
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

from typing import Any, Generic, Optional

from .backends import CacheBackend
from .constants import KeyT


class Cacher(Generic[KeyT]):
    """Main cacher class supporting multiple backends."""
    def __init__(self, backends: list[CacheBackend]):
        self.backends = backends
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
        """Set value in all backends."""
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
