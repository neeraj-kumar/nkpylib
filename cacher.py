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

import hashlib
import json
import os
import tempfile
import time


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar, Generic


# type for cache keys
KeyT = TypeVar('KeyT')

# type for hash function outputs
HashT = TypeVar('HashT', str, bytes, int)

class Keyer(ABC, Generic[KeyT]):
    """Base class for converting function arguments into cache keys."""
    @abstractmethod
    def make_key(self, args: tuple, kwargs: dict) -> KeyT:
        """Convert function arguments into a cache key.

        Args:
            args: Tuple of positional arguments
            kwargs: Dict of keyword arguments

        Returns:
            A hashable key suitable for the cache backend
        """
        pass


class TupleKeyer(Keyer[tuple]):
    """Converts function arguments into an immutable tuple-based key.

    Handles nested data structures by converting:
    - lists/tuples → tuples
    - sets → frozensets
    - dicts → frozenset of items
    - other objects → their hash

    The final key is a tuple of:
    (converted_args..., frozenset_of_converted_kwargs)
    """
    def make_key(self, args: tuple, kwargs: dict) -> tuple:
        # Convert kwargs to sorted, frozen items
        kw_items = self._make_hashable(kwargs)
        # Convert args and combine with kwargs
        return tuple(self._make_hashable(arg) for arg in args) + (kw_items,)

    def _make_hashable(self, obj: Any) -> Any:
        """Recursively convert an object into a hashable form."""
        # Already hashable types
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Convert sequences
        if isinstance(obj, (list, tuple)):
            return tuple(self._make_hashable(x) for x in obj)
        # Convert mappings
        if isinstance(obj, dict):
            return frozenset(
                (k, self._make_hashable(v))
                for k, v in sorted(obj.items())
            )
        # Convert sets
        if isinstance(obj, set):
            return frozenset(self._make_hashable(x) for x in obj)
        # Fall back to object's hash
        return hash(obj)


class StringKeyer(Keyer[str]):
    """Converts function arguments into a string key.

    Uses `TupleKeyer` internally to handle conversion to hashable types,
    then converts the resulting tuple to a string representation.
    """
    def __init__(self):
        self._tuple_maker = TupleKeyer()

    def make_key(self, args: tuple, kwargs: dict) -> str:
        tuple_key = self._tuple_maker.make_key(args, kwargs)
        return str(tuple_key)


class BaseHashKeyer(Keyer[Any]):
    """Base class for hash-based keyers.

    Uses `StringKeyer` internally to convert args to a string, then applies a hash function to get a
    fixed-length key.
    """
    def __init__(self, hash_func: str | Callable[[str], HashT] = 'sha256'):
        """The input `hash_func` should be either:

        - A string naming a `hashlib` algorithm (e.g. 'sha256', 'md5')
        - A callable that takes a string and returns a hash object

        Defaults to 'sha256'.
        """
        self._string_maker = StringKeyer()
        if isinstance(hash_func, str):
            if not hasattr(hashlib, hash_func):
                raise ValueError(f"Hash algorithm '{hash_func}' not found in hashlib")
            self._hash_func = lambda s: getattr(hashlib, hash_func)(s.encode('utf-8'))
            self._is_hashlib = True
        else:
            self._hash_func = hash_func
            self._is_hashlib = False

    def make_key(self, args: tuple, kwargs: dict) -> Any:
        string_key = self._string_maker.make_key(args, kwargs)
        return self._get_hash(string_key)

    def _get_raw_hash(self, s: str) -> Any:
        """Get hash value, either as hashlib object or direct value."""
        return self._hash_func(s)

    @abstractmethod
    def _get_hash(self, s: str) -> HashT:
        """Convert string to final hash form."""
        pass


class HashStringKeyer(BaseHashKeyer):
    """Hash keyer that returns string digests.

    If using a hashlib algorithm, returns hexdigest().
    If using a custom hash function, converts result to string.
    """
    def _get_hash(self, s: str) -> str:
        h = self._get_raw_hash(s)
        return h.hexdigest() if self._is_hashlib else str(h)


class HashBytesKeyer(BaseHashKeyer):
    """Hash keyer that returns raw byte digests.

    If using a hashlib algorithm, returns digest().
    If using a custom hash function, converts result to bytes.
    """
    def _get_hash(self, s: str) -> bytes:
        h = self._get_raw_hash(s)
        if self._is_hashlib:
            return h.digest()
        # Convert custom hash to bytes
        if isinstance(h, bytes):
            return h
        if isinstance(h, str):
            return h.encode('utf-8')
        return str(h).encode('utf-8')


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


class CacheNotFound(Exception):
    """Exception raised when a cache key is not found."""
    def __init__(self, key: KeyT):
        super().__init__(f"Cache key '{key}' not found")
        self.key = key


class CacheBackend(ABC, Generic[KeyT]):
    """Base class for storage backends.

    Each backend is initialized with a formatter that handles serialization.
    """
    def __init__(self, *,
                 formatter: CacheFormatter,
                 strategies: list[CacheStrategy]|None = None,
                 error_on_missing: bool = True,
                 **kwargs):
        self.formatter = formatter
        self.strategies = strategies or []
        self.error_on_missing = error_on_missing

    def get(self, key: KeyT) -> Any:
        """Get value for key, running it through all strategies."""
        # Run ALL pre-get hooks
        proceed = all(
            strategy.pre_get(key)
            for strategy in self.strategies
        )
        if not proceed:
            return self.not_found(key)

        # Get the value
        value = self._get_value(key)
        if value is None:
            return self.not_found(key)

        # Run post-get hooks
        for strategy in self.strategies:
            value = strategy.post_get(key, value)

        return value

    def set(self, key: KeyT, value: Any) -> None:
        """Set value after running it through all strategies."""
        # Run ALL pre-set hooks
        proceed = all(
            strategy.pre_set(key, value)
            for strategy in self.strategies
        )
        if not proceed:
            return

        # Store the value
        self._set_value(key, value)

        # Run post-set hooks
        for strategy in self.strategies:
            strategy.post_set(key, value)

    def delete(self, key: KeyT) -> None:
        """Delete key after checking with all strategies."""
        # Run pre-delete hooks
        for strategy in self.strategies:
            if not strategy.pre_delete(key):
                return

        # Delete the value
        self._delete_value(key)

        # Run post-delete hooks
        for strategy in self.strategies:
            strategy.post_delete(key)

    def clear(self) -> None:
        """Clear all entries after checking with strategies."""
        # Run ALL pre-clear hooks
        proceed = all(
            strategy.pre_clear()
            for strategy in self.strategies
        )
        if not proceed:
            return

        # Clear the storage
        self._clear()

        # Run post-clear hooks
        for strategy in self.strategies:
            strategy.post_clear()

    @abstractmethod
    def _get_value(self, key: KeyT) -> Any:
        """Actually get the value from storage."""
        pass

    @abstractmethod
    def _set_value(self, key: KeyT, value: Any) -> None:
        """Actually set the value in storage."""
        pass

    @abstractmethod
    def _delete_value(self, key: KeyT) -> None:
        """Actually delete the value from storage."""
        pass

    def _clear(self) -> None:
        """Clear all entries.

        This version does it by iterating through our keys and deleting each one.
        You can implement a more efficient version in your subclass.
        """
        for key in list(self.iter_keys()):
            self._delete_value(key)

    def iter_keys(self) -> Iterator[KeyT]:
        """Iterate over all keys in the cache."""
        raise NotImplementedError("iter_keys not implemented")

    def not_found(self, key: KeyT) -> None:
        """Return `None` or raise `CacheNotFound` based on `error_on_missing`."""
        if self.error_on_missing:
            raise CacheNotFound(key)
        else:
            return None


class CacheStrategy(ABC, Generic[KeyT]):
    """Base class for cache strategies.

    Strategies can hook into different stages of cache operations:
    - pre_get: Before retrieving a value
    - post_get: After retrieving a value
    - pre_set: Before setting a value
    - post_set: After setting a value
    - pre_delete: Before deleting a value
    - post_delete: After deleting a value
    """
    def pre_get(self, key: KeyT) -> bool:
        """Called before retrieving a value.

        Returns:
            False to skip cache lookup, True to proceed
        """
        return True

    def post_get(self, key: KeyT, value: Any) -> Any:
        """Called after retrieving a value.

        Args:
            key: The cache key
            value: The retrieved value

        Returns:
            Potentially modified value
        """
        return value

    def pre_set(self, key: KeyT, value: Any) -> bool:
        """Called before setting a value.

        Returns:
            False to skip caching, True to proceed
        """
        return True

    def post_set(self, key: KeyT, value: Any) -> None:
        """Called after setting a value."""
        pass

    def pre_delete(self, key: KeyT) -> bool:
        """Called before deleting a value.

        Returns:
            False to skip deletion, True to proceed
        """
        return True

    def post_delete(self, key: KeyT) -> None:
        """Called after deleting a value."""
        pass

    def pre_clear(self) -> bool:
        """Called before clearing all entries.
        
        Returns:
            False to skip clearing, True to proceed
        """
        return True

    def post_clear(self) -> None:
        """Called after clearing all entries."""
        pass


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


class JsonFormatter(CacheFormatter):
    """JSON serialization format."""
    def __init__(self, EncoderCls=json.JSONEncoder, DecoderCls=json.JSONDecoder):
        self.EncoderCls = EncoderCls
        self.DecoderCls = DecoderCls

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, cls=self.EncoderCls, ensure_ascii=False).encode('utf-8')

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'), cls=self.DecoderCls)


def _write_atomic(path: Path, data: bytes) -> None:
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

def _read_file(path: Path) -> bytes|None:
    """Read file contents, returning None if file doesn't exist or is invalid."""
    try:
        with open(path, 'rb') as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return None


class SeparateFileBackend(CacheBackend[KeyT]):
    """Backend that stores each key in a separate file.

    Good for large objects like embeddings or images where you want
    to manage each cached item independently.
    """
    def __init__(self, cache_dir: str|Path, *, formatter: CacheFormatter, **kwargs):
        super().__init__(formatter=formatter, **kwargs)
        self.cache_dir = Path(kwargs['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: KeyT) -> Path:
        """Convert cache key to filesystem path."""
        # Use key as filename, replacing invalid chars
        #FIXME
        safe_key = "".join(c if c.isalnum() else '_' for c in key)
        return self.cache_dir / safe_key

    def _get_value(self, key: KeyT) -> Any:
        """Get value from file storage."""
        path = self._key_to_path(key)
        data = _read_file(path)
        if data is None:
            return self.not_found(key)
        try:
            return self.formatter.loads(data)
        except Exception:
            return self.not_found(key)

    def _set_value(self, key: KeyT, value: Any) -> None:
        """Store value in file storage."""
        path = self._key_to_path(key)
        _write_atomic(path, self.formatter.dumps(value))

    def _delete_value(self, key: KeyT) -> None:
        """Delete value from file storage."""
        path = self._key_to_path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _clear(self) -> None:
        """Clear all entries. This deletes all files in our cache dir."""
        for path in self.cache_dir.iterdir():
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class JointFileBackend(CacheBackend[KeyT]):
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
    def __init__(self, cache_path: str|Path, *, formatter: CacheFormatter, **kwargs):
        super().__init__(formatter=formatter, **kwargs)
        self.cache_path = Path(kwargs['cache_path'])
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
        _write_atomic(self.cache_path, self.formatter.dumps(self._cache))

    def _get_value(self, key: KeyT) -> Any:
        """Get value from cache dict."""
        return self._cache.get(key)

    def _set_value(self, key: KeyT, value: Any) -> None:
        """Store value in cache dict and save to file."""
        self._cache[key] = value
        self._save()

    def _delete_value(self, key: KeyT) -> None:
        """Delete value from cache dict and save to file."""
        if key in self._cache:
            del self._cache[key]
            self._save()

    def _clear(self) -> None:
        """Clear all entries in cache dict and save to file."""
        self._cache.clear()
        self._save()

    def iter_keys(self) -> Iterator[KeyT]:
        """Iterate over all keys in the cache dict."""
        yield from self._cache


class MemoryBackend(CacheBackend[KeyT]):
    """Backend that stores everything in memory.

    Good for temporary caching and testing. Data is lost when process exits.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache: dict[KeyT, Any] = {}

    def iter_keys(self) -> Iterator[KeyT]:
        """Iterate over all keys in memory cache."""
        yield from self._cache.keys()

    def _get_value(self, key: KeyT) -> Any:
        """Get value from memory cache."""
        return self._cache.get(key)

    def _set_value(self, key: KeyT, value: Any) -> None:
        """Store value in memory cache."""
        self._cache[key] = value

    def _delete_value(self, key: KeyT) -> None:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]

    def _clear(self) -> None:
        """Clear all entries in memory cache."""
        self._cache.clear()
