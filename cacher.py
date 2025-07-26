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

import json
import os
import tempfile
import time


from abc import ABC, abstractmethod
from pathlib import Path
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

class CacheNotFound(Exception):
    """Exception raised when a cache key is not found."""
    def __init__(self, key: KeyT):
        super().__init__(f"Cache key '{key}' not found")
        self.key = key


class CacheBackend(ABC, Generic[KeyT]):
    """Base class for storage backends.

    Each backend is initialized with a formatter that handles serialization.
    """
    def __init__(self,
                 formatter: CacheFormatter,
                 key_type: type[KeyT] = str,
                 error_on_missing: bool = True):
        self.formatter = formatter
        self.key_type = key_type
        self.error_on_missing = error_on_missing

    @abstractmethod
    def get(self, key: KeyT) -> Any:
        """Get value for `key`.

        If `error_on_missing` is True, raises CacheNotFound if key is not found.
        Else, returns None.
        """
        pass

    @abstractmethod
    def set(self, key: KeyT, value: Any) -> None:
        """Set `value` for `key`"""
        pass

    @abstractmethod
    def delete(self, key: KeyT) -> None:
        """Delete `key` from cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    def not_found(self, key: KeyT) -> None:
        """Return `None` or raise `CacheNotFound` based on `error_on_missing`."""
        if self.error_on_missing:
            raise CacheNotFound(key)
        else:
            return None


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


class JsonFormatter(CacheFormatter):
    """JSON serialization format."""
    def __init__(self, EncoderCls=json.JSONEncoder, DecoderCls=json.JSONDecoder):
        self.EncoderCls = EncoderCls
        self.DecoderCls = DecoderCls

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, cls=self.EncoderCls, ensure_ascii=False).encode('utf-8')

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'), cls=self.DecoderCls)


class FileBackend(CacheBackend[KeyT], ABC):
    """Abstract base class for file-based cache backends.

    Provides common utilities for atomic file operations.
    """
    def _write_atomic(self, path: Path, data: bytes) -> None:
        """Write data to a file atomically using a temporary file."""
        # Create temporary file in same directory to ensure atomic rename
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=path.parent,
            prefix=path.name + '.',
            suffix='.tmp',
            delete=False
        ) as tf:
            tf.write(data)
            # Ensure all data is written to disk
            tf.flush()
            os.fsync(tf.fileno())

        # Atomic rename to final path
        os.replace(tf.name, path)

    def _read_file(self, path: Path) -> bytes|None:
        """Read file contents, returning None if file doesn't exist or is invalid."""
        try:
            with open(path, 'rb') as f:
                return f.read()
        except (FileNotFoundError, IOError):
            return None


class SeparateFileBackend(FileBackend[KeyT]):
    """Backend that stores each key in a separate file.

    Good for large objects like embeddings or images where you want
    to manage each cached item independently.
    """
    def __init__(self, cache_dir: str|Path, formatter: CacheFormatter):
        super().__init__(formatter)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: KeyT) -> Path:
        """Convert cache key to filesystem path."""
        # Use key as filename, replacing invalid chars
        #FIXME
        safe_key = "".join(c if c.isalnum() else '_' for c in key)
        return self.cache_dir / safe_key

    def get(self, key: KeyT) -> Any:
        """Get value for `key`"""
        path = self._key_to_path(key)
        data = self._read_file(path)
        if data is None:
            return self.not_found(key)
        try:
            return self.formatter.loads(data)
        except Exception:
            return self.not_found(key)

    def set(self, key: KeyT, value: Any) -> None:
        path = self._key_to_path(key)
        self._write_atomic(path, self.formatter.dumps(value))

    def delete(self, key: KeyT) -> None:
        """Deletes the file corresponding to `key`."""
        path = self._key_to_path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def clear(self) -> None:
        """Deletes all files in our cache directory."""
        for path in self.cache_dir.iterdir():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class JointFileBackend(FileBackend[KeyT]):
    """Backend that stores all keys in a single file.

    Good for small objects like metadata or settings where you want
    to keep everything in one place.

    The formatter should be dict-like at the top-level, supporting:
    - `key in cache`: Check if key exists
    - `cache[key]`: Get value for key
    - `cache[key] = value`: Set value for key
    - `del cache[key]`: Delete key
    - `cache.clear()`: Clear all entries
    """
    def __init__(self, cache_path: str|Path, formatter: CacheFormatter):
        super().__init__(formatter)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[KeyT, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from file."""
        try:
            with open(self.cache_path, 'rb') as f:
                self._cache = self.formatter.loads(f.read())
        except Exception:
            self._cache = {}

    def _save(self) -> None:
        """Save cache to file."""
        self._write_atomic(self.cache_path, self.formatter.dumps(self._cache))

    def get(self, key: KeyT) -> Any:
        if key not in self._cache:
            return self.not_found(key)
        return self._cache[key]

    def set(self, key: KeyT, value: Any) -> None:
        self._cache[key] = value
        self._save()

    def delete(self, key: KeyT) -> None:
        if key not in self._cache:
            return
        del self._cache[key]
        self._save()

    def clear(self) -> None:
        self._cache.clear()
        self._save()


class MemoryBackend(CacheBackend[KeyT]):
    """Backend that stores everything in memory.
    
    Good for temporary caching and testing. Data is lost when process exits.
    """
    def __init__(self, formatter: CacheFormatter):
        super().__init__(formatter)
        self._cache: dict[KeyT, Any] = {}

    def get(self, key: KeyT) -> Any:
        if key not in self._cache:
            return self.not_found(key)
        return self._cache[key]

    def set(self, key: KeyT, value: Any) -> None:
        self._cache[key] = value

    def delete(self, key: KeyT) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
