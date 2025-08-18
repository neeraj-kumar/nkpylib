from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, Iterator

import sqlalchemy

from nkpylib.cacher.constants import KeyT, CacheNotFound
from nkpylib.cacher.formatters import CacheFormatter, JsonFormatter
from nkpylib.cacher.strategies import CacheStrategy
from nkpylib.cacher.file_utils import _read_file, _write_atomic
from nkpylib.cacher.keyers import Keyer, TupleKeyer, HashStringKeyer


class CacheBackend(ABC, Generic[KeyT]):
    """Base class for storage backends.

    Each backend is initialized with a formatter that handles serialization.
    """
    # Sentinel object for cache misses
    CACHE_MISS = object()

    def __init__(self,
                 fn: Callable|None=None,
                 *,
                 formatter: CacheFormatter,
                 keyer: Keyer|None = None,
                 strategies: list[CacheStrategy]|None = None,
                 error_on_missing: bool = True,
                 batch_extractor: Callable[[tuple, dict], list]|None = None,
                 batch_combiner: Callable[[list, tuple, dict], Any]|None = None,
                 **kwargs):
        self.fn = fn
        self.formatter = formatter
        self.keyer = keyer or HashStringKeyer()
        self.strategies = []
        for strategy in (strategies or []):
            strategy._backend = self
            self.strategies.append(strategy)
        self.error_on_missing = error_on_missing
        self.batch_extractor = batch_extractor
        self.batch_combiner = batch_combiner
        self.stats: dict[str, int] = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def _to_key(self, fn: Callable|None, args: tuple, kwargs: dict) -> KeyT:
        """Convert function arguments to a cache key using the keyer."""
        return self.keyer.make_key(fn, args, kwargs)

    def call_with_cache(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a function with caching.

        Note that we do not use the instance's `fn` attribute here, and we also don't check our
        cache for the function itself, and hence the cache might not be for the right thing if
        you're not careful.

        Args:
            fn: The function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The cached result or the result of the function call.
        """
        # Convert function arguments to cache key
        key = self._to_key(fn, args, kwargs)

        # Try to get from cache
        if self.error_on_missing:
            try:
                result = self.get(key)
                return result
            except CacheNotFound:
                pass
        else:
            result = self.get(key)
            if not self.is_cache_miss(result):
                return result

        # Not in cache, call function
        result = fn(*args, **kwargs)

        # Store in cache
        self.set(key, result)
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call our cached function with the given arguments.

        If batch_extractor and batch_combiner are set, treats this as a batch operation.
        Otherwise, treats it as a normal function call.
        """
        if self.batch_extractor is not None and self.batch_combiner is not None:
            # Extract batch items
            #FIXME does this do what we want?
            items = self.batch_extractor(args, kwargs)

            # Process batch with caching
            results = self.call_with_batch_cache(self.fn, items)

            # Combine results back into expected format
            return self.batch_combiner(results, args, kwargs)
        else:
            # Normal function call
            return self.call_with_cache(self.fn, *args, **kwargs)

    def as_decorator(self) -> Callable:
        """Returns this cacher as a decorator that will cache function results.

        Returns:
            A decorator function that will cache results of the decorated function.
        """
        def decorator(func: Callable) -> CacheBackend:
            # Create new instance with same config but with the function
            return self.__class__(
                fn=func,
                formatter=self.formatter,
                keyer=self.keyer,
                strategies=self.strategies,
                error_on_missing=self.error_on_missing,
                batch_extractor=self.batch_extractor,
                batch_combiner=self.batch_combiner,
            )
        return decorator

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self.stats.copy()

    def get(self, key: KeyT) -> Any:
        """Get value for key, running it through all strategies."""
        # Run ALL pre-get hooks
        proceed = all(
            strategy.pre_get(key)
            for strategy in self.strategies
        )
        if not proceed:
            return self.handle_cache_miss(key)

        # Get the value
        value = self._get_value(key)
        if self.is_cache_miss(value):
            return self.handle_cache_miss(key)

        # Run post-get hooks
        for strategy in self.strategies:
            value = strategy.post_get(key, value)

        return self.handle_cache_hit(key, value)

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

    def get_many(self, keys: list[KeyT]) -> dict[KeyT, Any]:
        """Get multiple values at once.

        Args:
            keys: List of cache keys to retrieve

        Returns:
            Dict mapping keys to their values, only including found items
        """
        results = {}
        for key in keys:
            try:
                value = self.get(key)
                if not self.is_cache_miss(value):
                    results[key] = value
            except CacheNotFound:
                continue
        return results

    def set_many(self, items: dict[KeyT, Any]) -> None:
        """Set multiple key/value pairs at once.

        Args:
            items: Dict mapping keys to values to cache
        """
        for key, value in items.items():
            self.set(key, value)

    def call_with_batch_cache(self, fn: Callable[[list], list], items: list) -> list:
        """Call function with batched input, using cache where possible.

        Args:
            fn: Function that takes a list of items and returns list of results
            items: List of items to process

        Returns:
            List of results in same order as input items
        """
        # Convert each item to a cache key
        keys = [self._to_key((item,), {}) for item in items]
        key_to_idx = {key: i for i, key in enumerate(keys)}

        # Get cached results
        cached = self.get_many(keys)

        # Find items that need computing
        uncached_indices = [i for i, key in enumerate(keys) if key not in cached]
        if not uncached_indices:
            # Everything was cached
            return [cached[key] for key in keys]

        # Compute uncached items
        uncached_items = [items[i] for i in uncached_indices]
        new_results = fn(uncached_items)

        if len(new_results) != len(uncached_items):
            raise ValueError(
                f"Batch function returned {len(new_results)} results "
                f"for {len(uncached_items)} inputs"
            )

        # Cache new results
        new_items = {
            keys[uncached_indices[i]]: result
            for i, result in enumerate(new_results)
        }
        self.set_many(new_items)

        # Combine cached and new results
        all_results = list(items)  # Make same size as input
        for key, value in cached.items():
            all_results[key_to_idx[key]] = value
        for i, value in zip(uncached_indices, new_results):
            all_results[i] = value

        return all_results

    def iter_keys(self) -> Iterator[KeyT]:
        """Iterate over all keys in the cache."""
        raise NotImplementedError("iter_keys not implemented")

    def is_cache_miss(self, value: Any) -> bool:
        """Check if a value represents a cache miss."""
        return value is self.CACHE_MISS

    def handle_cache_miss(self, key: KeyT) -> Any:
        """Handle a cache miss based on error_on_missing setting."""
        self.stats['misses'] += 1
        if self.error_on_missing:
            raise CacheNotFound(key)
        return self.CACHE_MISS

    def handle_cache_hit(self, key: KeyT, value: Any) -> Any:
        """Handle a cache hit."""
        #print(f'handling cache hit for value: {value}')
        self.stats['hits'] += 1
        return value


class MemoryBackend(CacheBackend[KeyT]):
    """Backend that stores everything in memory.

    Good for temporary caching and testing. Data is lost when process exits.
    """
    def __init__(self, fn: Callable|None=None, **kwargs):
        super().__init__(fn=fn, **kwargs)
        self._cache: dict[KeyT, Any] = {}

    def iter_keys(self) -> Iterator[KeyT]:
        """Iterate over all keys in memory cache."""
        yield from self._cache.keys()

    def _get_value(self, key: KeyT) -> Any:
        """Get value from memory cache."""
        return self._cache.get(key, self.CACHE_MISS)

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

class SeparateFileBackend(CacheBackend[KeyT]):
    """Backend that stores each key in a separate file.

    Good for large objects like embeddings or images where you want
    to manage each cached item independently.
    """
    def __init__(self,
                 cache_dir: str|Path,
                 filename_fn: Callable|None=None,
                 fn: Callable|None=None,
                 *,
                 formatter: CacheFormatter|None=None,
                 **kwargs):
        super().__init__(fn=fn, formatter=formatter or JsonFormatter(), **kwargs)
        self.filename_fn = filename_fn
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: KeyT) -> Path:
        """Convert cache key to filesystem path."""
        if self.filename_fn:
            # Use custom filename function if provided
            filename = self.filename_fn(key)
        else:
            filename = str(key)
        # replace invalid chars #FIXME
        filename = "".join(c if c.isalnum() else '_' for c in filename)
        # add extension based on formatter
        if not filename.endswith(self.formatter.EXT):
            filename += self.formatter.EXT
        return self.cache_dir / filename

    def _get_value(self, key: KeyT) -> Any:
        """Get value from file storage."""
        path = self._key_to_path(key)
        data = _read_file(path)
        if self.is_cache_miss(data):
            return self.handle_cache_miss(key)
        try:
            return self.formatter.loads(data)
        except Exception:
            return self.CACHE_MISS

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
    def __init__(self, cache_path: str|Path, fn: Callable|None=None, *, formatter: CacheFormatter|None=None, **kwargs):
        super().__init__(fn=fn, formatter=formatter or JsonFormatter(), **kwargs)
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


class MultiplexBackend(CacheBackend[KeyT]):
    """Backend that multiplexes operations across multiple other backends.

    Gets return the first hit from any backend.
    Sets/deletes/clears apply to all backends.

    Can be used directly:
        cache = MultiplexBackend([...backends...])
        value = cache.get(key)
        cache.set(key, value)

    Or as a decorator:
        @cache.as_decorator()
        def expensive_function(x, y):
            return x + y
    """
    def __init__(self, backends: list[CacheBackend], **kwargs):
        super().__init__(formatter=backends[0].formatter if backends else None, **kwargs)
        self.backends = backends

    def _get_value(self, key: KeyT) -> Any:
        """Get value from first backend that has it."""
        for backend in self.backends:
            value = backend.get(key)
            if value is not None:
                return value
        return None

    def _set_value(self, key: KeyT, value: Any) -> None:
        """Set value in all backends."""
        for backend in self.backends:
            backend.set(key, value)

    def _delete_value(self, key: KeyT) -> None:
        """Delete from all backends."""
        for backend in self.backends:
            backend.delete(key)

    def _clear(self) -> None:
        """Clear all backends."""
        for backend in self.backends:
            backend.clear()
