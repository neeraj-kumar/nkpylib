from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterator

from .constants import KeyT, CacheNotFound
from .formatters import CacheFormatter
from .strategies import CacheStrategy
from .file_utils import _read_file, _write_atomic


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


