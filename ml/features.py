"""Basic feature classes and utilities.

This module provides classes and functions for working with features (in the ML sense):

- Feature: Base class for all feature types, providing a common interface
- ConstantFeature: A feature that returns constant values
- PairwiseMax: Computes maximum similarity between two sets using a comparison function
- TimeContext: Extracts temporal features from timestamps
- Recency: Computes time differences between pairs of timestamps
- MappingFeature: Wraps dictionary-like objects as features
- FunctionFeature: Wraps arbitrary functions as features
- FeatureMap: A mapping interface to a collection of features

Features can be combined, transformed, and used as inputs to machine learning models.
The module handles proper typing, validation, and efficient computation of features.

For groups of features put together, as well as storage and retrieval, see feature_set.py
"""

from __future__ import annotations

import datetime
import logging
import os
import time

from abc import ABC
from collections.abc import Mapping, MutableMapping
from os.path import dirname
from typing import Any, Sequence, TypeVar, Generic, Callable, Iterator

import numpy as np

from matplotlib.pyplot import imsave

from nkpylib.time_utils import parse_ts

logger = logging.getLogger(__name__)

__all__ = [
    'Feature', 'ConstantFeature', 'PairwiseMax', 'TimeContext', 'Recency',
    'MappingFeature', 'FunctionFeature', 'FeatureMap'
]

class Feature(ABC):
    """Base class for all features.

    A Feature instance represents one logical feature, but can be many dimensions.

    This class provides a common interface for working with features that can be:
    - Single values or arrays
    - Computed on demand or cached
    - Combined hierarchically (via children features)

    Attributes:
    - name (str): Name of the feature, defaults to class name
    - description (str): Human-readable description of what this feature represents
    - children (list[Feature]): Child features that get concatenated with this one

    Methods implemented here:
    - get(): Returns the feature as a numpy array, and validates it.
      - If we have children, it concatenates their results, else it calls `_get()`.
    - len(): Returns the length of the feature (default is len(get()))

    Key methods that subclasses must implement:
    - _get(): Returns the feature as a numpy array
    - _len(): (optional) Returns the length of the feature, defaults to len(self.get())

    Optional methods subclasses may implement:
    - update(**kw): Update the feature's internal state
    - validate(arr, feat): Validate feature array output (default checks len > 0)

    Usage:
        Features can be used individually or composed into feature hierarchies:

        # Single feature
        feat = MyFeature(name='example', description='An example feature')
        arr = feat.get()  # Returns numpy array

        # Composite features
        parent = ParentFeature(children=[
            ChildFeature1(),
            ChildFeature2()
        ])
        arr = parent.get()  # Returns concatenated arrays from children
    """
    def __init__(self,
                 name: str='',
                 description: str='',
                 children: list[Feature]|None=None):
        if not name:
            name = self.__class__.__name__
        self.name = name
        self.description = description
        if children is None:
            children = []
        self.children = children

    def __str__(self):
        return self.name

    def _len(self) -> int:
        """Returns the length of the feature (implementation).

        By default this actually gets the feature and then returns its length.
        """
        return len(self.get())

    def __len__(self) -> int:
        """Returns the length of the feature"""
        if self.children:
            return sum(len(c) for c in self.children)
        return self._len()

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array (implementation)"""
        raise NotImplementedError()

    def get(self) -> np.ndarray:
        """Returns the feature as a numpy array.

        If it's a composite feature, it will return the concatenation of the children's features.
        Else, it will call the implementation of `_get`.
        """
        if self.children:
            ret: list[np.ndarray]|np.ndarray = [c.get() for c in self.children]
            for child, f in zip(self.children, ret):
                self.validate(f, child)
            return np.concatenate(ret)
        ret = self._get()
        self.validate(ret, self)
        return ret

    def validate(self, arr, feat):
        """Validates the feature `arr` obtained from feature `feat`."""
        try:
            assert len(arr) > 0, f"Feature {feat} has length 0"
        except Exception as e:
            assert False, f"{type(e)} with feature {feat}: value {arr}"

    def update(self, **kw):
        """Updates the feature"""
        raise NotImplementedError()


class ConstantFeature(Feature):
    """A constant feature (could be many dims)"""
    def __init__(self, values: int|float|Sequence[int|float], **kw):
        """Initializes the constant feature"""
        super().__init__(**kw)
        if isinstance(values, (int, float)):
            values = [values]
        self.values = values

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        return np.array(self.values)


class EnumFeature(Feature):
    """A feature that encodes an enum/categorical value.

    Supports multiple encoding types:
    - onehot: One-hot encoding (binary vector with 1 at category position) [default]
    - label: Simple integer label encoding
    - binary: Binary encoding using log2(n) dimensions
    - target: Replace category with mean target value
    - hash: Hash encoding to fixed number of bins
    """
    def __init__(self,
                 value: str|int,
                 enum_values: list[str|int],
                 encoding: str='onehot',
                 target_values: dict[str|int, float]|None=None,
                 n_hash_bins: int=8,
                 **kw):
        """Initialize enum feature encoder.

        Args:
        - value: The enum value to encode
        - enum_values: List of all possible enum values
        - encoding: One of 'onehot', 'label', 'binary', 'target', 'hash'
        - target_values: Dict mapping enum values to target means (for target encoding)
        - n_hash_bins: Number of bins for hash encoding
        """
        super().__init__(**kw)
        if encoding not in ['onehot', 'label', 'binary', 'target', 'hash']:
            raise ValueError(f"Unknown encoding type: {encoding}")
        if encoding == 'target' and not target_values:
            raise ValueError("Target encoding requires target_values dict")
            
        # Compute encoded value immediately
        match encoding:
            case 'onehot':
                idx = enum_values.index(value)
                arr = np.zeros(len(enum_values))
                arr[idx] = 1
                self._encoded = arr
            case 'label':
                self._encoded = np.array([enum_values.index(value)])
            case 'binary':
                idx = enum_values.index(value)
                n_bits = int(np.ceil(np.log2(len(enum_values))))
                binary = format(idx, f'0{n_bits}b')
                self._encoded = np.array([int(b) for b in binary])
            case 'target':
                assert target_values is not None
                self._encoded = np.array([target_values[value]])
            case 'hash':
                hash_val = hash(str(value))
                bin_idx = hash_val % n_hash_bins
                arr = np.zeros(n_hash_bins)
                arr[bin_idx] = 1
                self._encoded = arr

    def _get(self) -> np.ndarray:
        """Returns the encoded feature as a numpy array."""
        return self._encoded


T = TypeVar('T')

class PairwiseMax(Feature, Generic[T]):
    """Computes max over two sets of keys that can be compared pairwise.

    You can use this to compute the max similarity between two sets of keys.
    Each set of keys is a sequence of objects of type T, and is compared using a similarity function
    that returns a float, of which we take the maximum.
    """
    def __init__(self,
                 keys1: Sequence[T],
                 keys2: Sequence[T],
                 sim_func: Callable[[T, T], float],
                 default: float=-1.0,
                 **kw):
        super().__init__(**kw)
        self.keys1 = keys1
        self.keys2 = keys2
        self.sim_func = sim_func
        self.default = default

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        try:
            keys = [(k1, k2) for k1 in self.keys1 for k2 in self.keys2]
            values = [self.sim_func(k1, k2) for k1, k2 in keys]
            if self.name.startswith('img_text_similarity') and max(values) > 0.4:
                print(f"For {self.name} got keys and values: {[(k, v) for k,v in zip(keys, values) if v > 0.4]}")
            return np.array([max(values)])
        except ValueError:
            return np.array([self.default])

    def _len(self) -> int:
        """Returns the length of the feature"""
        return 1


class TimeContext(Feature):
    """Computes some features based on the time context"""
    def __init__(self, ts: float|str, fields=['dow', 'hour'], **kw):
        """Initializes the feature with a timestamp.

        The types of fields that we populate are:
        - 'dow': day of week (0-6)
        - 'hour': hour of day (0-23)
        """
        super().__init__(**kw)
        self.ts = parse_ts(ts)
        self.fields = fields
        for field in fields:
            if field not in ['dow', 'hour']:
                raise NotImplementedError(f"Unknown field {field}")

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        ret = []
        for field in self.fields:
            if field == 'dow':
                ret.append(datetime.datetime.fromtimestamp(self.ts).weekday())
            elif field == 'hour':
                ret.append(datetime.datetime.fromtimestamp(self.ts).hour)
            else:
                raise ValueError(f"Unknown field {field}")
        return np.array(ret)


class Recency(Feature):
    """Computes the recency between two timestamps"""
    def __init__(self, a: float|str, b: float|str, apply_log: bool=True, **kw):
        """The dates can either be as floats (epoch seconds) or as strings (ISO format).

        Note that this is NOT symmetric, we assume `a` is the more recent date.
        If `apply_log` is True, we apply log1p to the difference (in that case clamping to positive
        first).

        Internally we convert them to floats.
        """
        super().__init__(**kw)
        self.a = parse_ts(a)
        self.b = parse_ts(b)
        self.apply_log = apply_log

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        try:
            ret = self.a - self.b
        except Exception as e:
            logger.error(f"{type(e)} Error with {self.a} - {self.b}: {e}")
            raise

        if self.apply_log:
            ret = np.log1p(max(ret, 0.0))
        return np.array([ret])

    def _len(self) -> int:
        """Returns the length of the feature"""
        return 1


class MappingFeature(Feature, Generic[T]):
    """A feature that is stored in a Mapping (dict-like) object.

    The generic type T is the type of the key in the mapping.
    """
    def __init__(self, d: Mapping, key: T, **kw):
        """Initializes the feature with a mapping object `d` and a `key`."""
        super().__init__(**kw)
        self.d = d
        self.key = key

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        return np.array(self.d[self.key])


class FunctionFeature(Feature):
    """A feature that is computed by a function."""
    def __init__(self,
                 func: Callable,
                 func_args: Sequence[Any]=(),
                 func_kwargs: Mapping[str, Any]={},
                 **kw):
        """Initializes the feature with a function `func` and the args and kwargs to call it with."""
        super().__init__(**kw)
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        return self.func(*self.func_args, **self.func_kwargs)


def save_as_image(path: str, arr: np.ndarray, n_cols=100, cmap='viridis'):
    """Fold the input array to have `n_cols` columns and save as an image"""
    logger.info(f'Saving {arr.shape} array to {path} with {n_cols} columns')
    n_rows = len(arr) // n_cols
    arr = arr[:n_rows*n_cols].reshape(n_rows, n_cols, -1)
    # scale arr to be between 0 and 1 using np functions
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    try:
        os.makedirs(dirname(path), exist_ok=True)
    except Exception as e:
        pass
    imsave(path, arr, cmap=cmap)


KeyT = TypeVar('KeyT')
class FeatureMap(Mapping, Generic[KeyT]):
    """A map from keys to Feature objects, that returns feature vectors."""
    def __init__(self, d: dict[KeyT, Feature]|None=None):
        """Initializes the feature map with a dictionary of features."""
        if d is None:
            d = {}
        self._d = d

    def __getitem__(self, key: KeyT) -> np.ndarray:
        """Returns the feature for the given key."""
        return self._d[key].get()

    def __iter__(self) -> Iterator[KeyT]:
        """Returns an iterator over the keys of the map."""
        return iter(self._d)

    def __len__(self) -> int:
        """Returns the number of features in the map."""
        return len(self._d)

    def __contains__(self, key: object) -> bool:
        """Checks if the map contains the given key."""
        return key in self._d
