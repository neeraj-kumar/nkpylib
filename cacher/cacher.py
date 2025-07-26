"""A fully-functional cacher with all the bells-and-whistles.

"""

from __future__ import annotations

from typing import Any, Generic, Optional

from nkpylib.cacher.backends import CacheBackend
from nkpylib.cacher.constants import KeyT


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
