"""A fully-functional cacher with all the bells-and-whistles.

"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Generic, Optional, TypeVar

from nkpylib.cacher.backends import CacheBackend
from nkpylib.cacher.constants import KeyT
from nkpylib.cacher.keyers import Keyer, TupleKeyer


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

    def as_decorator(self, keyer: Keyer|None = None) -> Callable:
        """Create a decorator that will cache function results.

        Args:
            keyer: Optional Keyer instance to convert function args to cache keys.
                  Defaults to TupleKeyer if not specified.

        Returns:
            A decorator function that will cache results of the decorated function.
        """
        if keyer is None:
            keyer = TupleKeyer()

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Convert function arguments to cache key
                key = keyer.make_key(args, kwargs)

                # Try to get from cache
                result = self.get(key)
                if result is not None:
                    return result

                # Not in cache, call function
                result = func(*args, **kwargs)

                # Store in cache
                self.set(key, result)

                return result
            return wrapper
        return decorator
