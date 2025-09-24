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

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
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
    """Base class for all features"""
    def __init__(self, template: Template=None, name: str=''):
        self._template = template
        self.name = name or self.__class__.__name__

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

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return len(self.get())

    @abstractmethod
    def _get(self) -> np.ndarray:
        """Returns the feature as a numpy array (implementation)"""
        raise NotImplementedError()

    def get(self) -> np.ndarray:
        """Returns the feature as a numpy array with validation."""
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
        """Updates the feature - override in subclasses as needed."""
        pass


class CompositeFeature(Feature):
    """Feature with children, defined via schema."""
    SCHEMA: OrderedDict = OrderedDict()

    @classmethod
    @abstractmethod
    def define_schema(cls):
        """Must be defined by subclasses to define the schema.

        They should call `set_schema()` with a list of (name, template) tuples.
        """
        pass

    @classmethod
    def set_schema(cls, schema_list):
        """Called by subclasses to set up schema."""
        cls.SCHEMA = OrderedDict(schema_list)

    def __init__(self, **kw):
        """Initialize this composite feature.

        This will first initialize our schema using `define_schema()` if it hasn't been done yet.
        It will also pre-allocate an array for all schema features.
        """
        super().__init__(**kw)
        if not self.SCHEMA:
            self.__class__.define_schema()
        self._children = [None] * len(self.SCHEMA)

    @property
    def children(self) -> list[Feature]:
        """Dynamic children list based on schema order."""
        return [f for f in self._children]

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return sum(len(c) for c in self.children)

    def get(self) -> np.ndarray:
        """Composite features concatenate children."""
        arrays = [c.get() for c in self.children]
        for child, arr in zip(self.children, arrays):
            self.validate(arr, child)
        return np.concatenate(arrays)

    def _set(self, name: str, *args, **kwargs) -> Feature:
        """Set a schema (child) feature with given name and arguments."""
        if name not in self.SCHEMA:
            raise ValueError(f"Unknown feature name: {name}")
        template = self.SCHEMA[name]
        # Create feature using template
        feature = template.create(name=name, *args, **kwargs)
        idx = list(self.SCHEMA).index(name)
        self._children[idx] = feature
        return feature

    def validate_complete(self) -> None:
        """Ensure all schema features have been initialized."""
        if not self.SCHEMA:
            return
        missing = []
        for i, name in enumerate(self.SCHEMA):
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


class MovieFeature(CompositeFeature):
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
    @classmethod
    def define_schema(cls):
        """Define the schema for movie features."""
        constant = Template(ConstantFeature)
        schema = [
            ('year', constant),
            ('runtime', constant),
        ]
        # Add rating features for each source
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                schema.append((f'{src}_{field}', constant))
            schema.append((f'{src}_log_votes', constant))
        # Add job count features
        for job in ['actor', 'actress', 'director', 'writer']:
            schema.append((f'num_{job}', constant))
        # Add financial features
        schema.extend([
            ('tmdb_budget', constant),
            ('tmdb_log_budget', constant),
            ('tmdb_revenue', constant),
            ('tmdb_log_revenue', constant),
        ])
        # add content-rating features
        rating_enum = Template(EnumFeature,
                             enum_values=[None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR'],
                             encoding='int')
        schema.append(('rt_content_rating', rating_enum))
        super().set_schema(schema)

    def __init__(self, m, **kw):
        super().__init__(**kw)
        self.m = m
        self.update()

    def _try_float(self, obj, attr, default=0.0):
        """Basically `float(obj.attr)` but returns default on any error or if None"""
        try:
            x = getattr(obj, attr)
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _extract_content_rating(self, m):
        """Extract content rating from movie tags."""
        ratings = [None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR']
        for t in m.tags:
            if t.source == 'rotten_tomatoes' and t.type == 'content_rating':
                return t.value if t.value in ratings else None
        return None

    def _extract_financials(self, m):
        """Extract budget and revenue from movie tags."""
        budget, revenue = 0.0, 0.0
        million = 1_000_000.0
        for t in m.tags:
            if t.source == 'tmdb' and t.type == 'budget':
                budget = self._try_float(t, 'value') / million
            elif t.source == 'tmdb' and t.type == 'revenue':
                revenue = self._try_float(t, 'value') / million
        return budget, revenue

    def update(self, **kw):
        """Actually generates the features using the schema."""
        m = self.m
        # Basic features
        self._set('year', m.year if m.year else 0)
        self._set('runtime', m.runtime if m.runtime else 0)
        # Rating features
        for r in m.ratings:
            for field in ['rating', 'votes', 'popularity']:
                v = self._try_float(r, field)
                self._set(f'{r.source}_{field}', v)
                if field == 'votes':
                    self._set(f'{r.source}_log_votes', np.log1p(v))
        # Job counts
        job_counts = Counter(tp.job_id.name for tp in m.people)
        for job, count in job_counts.items():
            if job in self.SCHEMA:
                self._set(f'num_{job}s', count)
        # Financial features
        budget, revenue = self._extract_financials(m)
        self._set('tmdb_budget', budget)
        self._set('tmdb_log_budget', np.log1p(budget))
        self._set('tmdb_revenue', revenue)
        self._set('tmdb_log_revenue', np.log1p(revenue))
        # Content rating
        content_rating = self._extract_content_rating(m)
        self._set('rt_content_rating', content_rating)
        # Validate all features are set
        super().update()
