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




    A Feature instance represents one logical feature, but can be many dimensions.

    This class provides a common interface for working with features that can be:
    - Single values or arrays
    - Computed on demand or cached
    - Combined hierarchically (via children features)

    Attributes:
    - name (str): Name of the feature, defaults to class name
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
        feat = MyFeature(name='example')
        arr = feat.get()  # Returns numpy array

        # Composite features
        parent = ParentFeature(children=[
            ChildFeature1(),
            ChildFeature2()
        ])
        arr = parent.get()  # Returns concatenated arrays from children

"""

from __future__ import annotations

import datetime
import logging
import os
import time

from abc import ABC
from collections.abc import Mapping, MutableMapping
from os.path import dirname
from typing import Any, Sequence, TypeVar, Generic, Callable, Iterator, Hashable, Type

import numpy as np

from matplotlib.pyplot import imsave

from nkpylib.time_utils import parse_ts

logger = logging.getLogger(__name__)

__all__ = [
    'Template', 'Feature', 'CompositeFeature', 'ConstantFeature', 'EnumFeature', 'PairwiseMax', 'TimeContext', 'Recency',
    'MappingFeature', 'FunctionFeature', 'FeatureMap'
]


class Template(Mapping):
    """Base class for feature templates that hold shared parameters."""
    def __init__(self, feature_class: Type[Feature], **shared_params: Any) -> None:
        """Create template for given feature class with shared parameters."""
        self.feature_class = feature_class
        self.shared_params = shared_params
        self._instances: list[Feature] = []  # Track all instances created from this template

    def create(self, **instance_params: Any) -> Feature:
        """Create new feature instance from this template."""
        instance = self.feature_class(template=self, **instance_params)
        self._instances.append(instance)
        return instance

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to shared params."""
        return self.shared_params[key]

    def __len__(self) -> int:
        """Return number of shared parameters."""
        return len(self.shared_params)

    def __iter__(self) -> Iterator[str]:
        """Iterate over shared parameter keys."""
        return iter(self.shared_params)

    @property
    def num_instances(self) -> int:
        """Return number of instances created from this template."""
        return len(self._instances)

    def iter_instances(self) -> Iterator[Feature]:
        """Iterate over all instances created from this template."""
        return iter(self._instances)


class Feature(ABC):
    """Base class for all features.
    """
    SCHEMA: list = []

    @classmethod
    def define_schema(cls, schema_list):
        """Called by subclasses to set up schema."""
        cls.SCHEMA = schema_list

    def __init__(self,
                 template: Template=None,
                 name: str=''):
        self._template = template
        self.name = name or self.__class__.__name__
        # Initialize schema-based features if schema exists
        if not self.SCHEMA:
            self.__class__.define_schema()
        if self.SCHEMA:
            # Pre-allocate array for all schema features
            self._children = [None] * len(self.SCHEMA)

    def __getattr__(self, name):
        """Check template for attributes not found in instance."""
        if self._template is not None and name in self._template:
            return self._template[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Don't store instance attributes that are already in template"""
        # Always allow setting the template itself
        if name == '_template':
            super().__setattr__(name, value)
            return
        # Don't store instance attributes that are already in template
        if hasattr(self, '_template') and self._template is not None and name in self._template:
            return  # Skip storing this attribute
        super().__setattr__(name, value)

    @property
    def instance_number(self):
        """Get this instance's number within its template (0-indexed)."""
        if self._template is None:
            return None
        return self._template._instances.index(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} [{len(self)}]>'

    def _len(self) -> int:
        """Returns the length of the feature (implementation).

        By default this actually gets the feature and then returns its length.
        """
        return len(self.get())

    @property
    def children(self) -> list[Feature]:
        """Dynamic children list based on schema order."""
        if not self.SCHEMA:
            return []
        return self._children

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

    def _set(self, name: str, *args, **kwargs) -> Feature:
        """Set a schema (child) feature with given name and arguments."""
        if not self.SCHEMA:
            raise ValueError("_set() can only be used with schema-based features")

        # Find template and index for this name
        template = None
        idx = None
        for i, (schema_name, schema_template) in enumerate(self.SCHEMA):
            if schema_name == name:
                template = schema_template
                idx = i
                break
        if template is None:
            raise ValueError(f"Unknown feature name: {name}")
        # Create feature using template
        feature = template.create(name=name, *args, **kwargs)
        self._children[idx] = feature
        return feature

    def validate_complete(self) -> None:
        """Ensure all schema features have been initialized."""
        if not self.SCHEMA:
            return
        missing = []
        for i, (name, _) in enumerate(self.SCHEMA):
            if self._children[i] is None:
                missing.append(name)
        if missing:
            raise ValueError(f"Uninitialized features: {missing}")

    def update(self, **kw):
        """Updates the feature and validates completion for schema-based features."""
        if self.SCHEMA:
            self.validate_complete()


class ConstantFeature(Feature):
    """A constant feature (could be many dims)"""
    def __init__(self, values: int|float|Sequence[int|float], **kw):
        """Initializes the constant feature"""
        super().__init__(**kw)
        if isinstance(values, (int, float)):
            values = [values]
        self.values = np.array(values)

    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array"""
        return self.values


EnumT = TypeVar('EnumT', bound=Hashable)
class EnumFeature(Feature, Generic[EnumT]):
    """A feature that encodes an enum/categorical value.

    Supports multiple encoding types:
    - onehot: One-hot encoding (binary vector with 1 at category position) [default]
    - int: Simple integer encoding
    - binary: Binary encoding using log2(n) dimensions
    - target: Replace category with mean target value
    - hash: Hash encoding to fixed number of bins
    """
    def __init__(self,
                 value: EnumT,
                 enum_values: Sequence[EnumT] | Mapping[EnumT, float|int|Sequence[float]] | int,
                 encoding: str='onehot',
                 **kw):
        """Initialize enum feature encoder.

        Args:
        - value: The enum value to encode
        - encoding: One of 'onehot', 'int', 'binary', 'target', 'hash'
        - enum_values:
          - if encoding is 'onehot', 'int', or 'binary', this is the list of all possible enum
            values, in the order to use for encoding
          - if encoding is 'target', this is a mapping from enum values to target value. The target
            value can be a single number or an array of numbers (e.g. an embedding)
          - if encoding is 'hash', this is the number of number of hash bins to use.
            we use the built-in hash function and modulo to get a bin index.

        We encode the value according to the specified encoding scheme immediately, storing only the
        final numpy array. Note that if you are encoding multiple values of the same enum, you
        should ensure that `enum_values` stays the same.
        """
        super().__init__(**kw)
        match encoding:
            case 'onehot':
                idx = enum_values.index(value)
                arr = np.zeros(len(enum_values))
                arr[idx] = 1
                self._encoded = arr
            case 'int':
                self._encoded = np.array([enum_values.index(value)])
            case 'binary':
                idx = enum_values.index(value)
                n_bits = int(np.ceil(np.log2(len(enum_values))))
                binary = format(idx, f'0{n_bits}b')
                self._encoded = np.array([int(b) for b in binary])
            case 'target':
                v = enum_values[value]
                if isinstance(v, (int, float)):
                    v = [v]
                self._encoded = np.array(v)
            case 'hash':
                hash_val = hash(str(value))
                bin_idx = hash_val % enum_values
                arr = np.zeros(enum_values)
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


''' IGNORE FOR NOW
class MovieFeature(Feature):
    """Movie Feature Vector.

    This includes for each movie [feature dim]:
    - year [1]
    - runtime [1]
    - rating, votes, log(votes), popularity [4 x 5 = 20]
      - by source (imdb, tmdb, lb, rt critics, rt audience)
    - num of {actors, actresses, directors, writers} [4]
    - tmdb budget and revenue in millions [2 x 2 = 4]
      - and their logs
    - content rating (G, PG, etc) as int [1]
    """
    def __init__(self, m: Movie, **kw):
        super().__init__(name=f"MovieFeature<{m.title_id}>")
        self.m = m
        self.update()

    def update(self, **kw):
        """Actually generates the features.

        Note that we have to be very careful to generate things in the right order.
        """
        self.children = []
        m = self.m
        def try_float(obj, attr, default=0.0):
            """Basically `float(obj.attr)` but returns default on any error or if None"""
            try:
                x = getattr(obj, attr)
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        C = lambda f: self.children.append(f) # shortcut to add a child feature
        C(ConstantFeature(name='year', values=m.year if m.year else 0))
        C(ConstantFeature(name='runtime', values=m.runtime if m.runtime else 0))
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        ratings_by_source = {r.source: r for r in m.ratings}
        fields = ['rating', 'votes', 'popularity']
        for src in rating_sources:
            rating, votes, popularity = 0.0, 0.0, 0.0
            r = ratings_by_source.get(src, None)
            for field in fields:
                v = try_float(r, field)
                C(ConstantFeature(name=f'{src}_{field}', values=v))
                if field == 'votes':
                    C(ConstantFeature(name=f'{src}_{log_votes}', values=np.log1p(v)))
        # count of people by job
        job_counts = Counter(tp.job_id.name for tp in m.people)
        for job in ['actor', 'actress', 'director', 'writer']:
            C(ConstantFeature(name=f'num_{job}s', values=job_counts.get(job, 0.0)))
        # tags
        budget, revenue = 0.0, 0.0
        million = 1_000_000.0
        content_rating = None
        # accumulate relevant tags
        ratings = [None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR']
        for t in m.tags:
            match (t.source, t.type):
                case ('tmdb', 'budget'):
                    budget = try_float(t.value) / million
                case ('tmdb', 'revenue'):
                    revenue = try_float(t.value)
                case ('rotten_tomatoes', 'content_rating'):
                    content_rating = t.value if t.value in ratings else None
                case _:
                    pass
        # actually add features now, to make sure they're in a consistent order
        C(ConstantFeature(name=f'tmdb_budget', values=budget))
        C(ConstantFeature(name=f'tmdb_log_budget', values=np.log1p(budget)))
        C(ConstantFeature(name=f'tmdb_revenue', values=revenue))
        C(ConstantFeature(name=f'tmdb_log_revenue', values=np.log1p(revenue)))
        C(EnumFeature(name=f'rt_content_rating', value=content_rating, enum_values=ratings, encoding='int'))
'''
