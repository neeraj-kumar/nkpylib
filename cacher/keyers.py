from __future__ import annotations

import hashlib

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Type, TypeVar

from .constants import KeyT, HashT

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


