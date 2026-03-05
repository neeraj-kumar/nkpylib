"""Basic classes and utilities for ML features.

## Core Concepts
Ultimately, the goal is to have an object that takes in a set of inputs and outputs a numpy feature
vector, in a functional way (i.e., the same set of inputs yields the same output feature vector). In
many cases, you might need some bit of configuration to determine how to compute the features. So
the general pattern is:

    f = Feature(config)
    all_vectors = [f.get(input) for input in inputs]

To make this easier in the common case where lots of features are quite similar, varying only in
some config parameters, this library uses a template-based architecture for efficient feature
computation.

**Templates**: Reusable feature configurations that define shared parameters across multiple feature
instances. Templates act as factories that create feature instances with common settings, reducing
memory usage and ensuring consistency.

**Features**: Individual feature computors that return numpy arrays. Features can be simple (single
values) or composite (combining multiple sub-features). All features implement a common interface
with `get()` returning validated numpy arrays.

**Composite Features**: Schema-driven features that combine multiple sub-features in a predefined
structure. The schema defines what features are included and their order, enabling consistent
feature vectors across instances.

**Feature Maps**: Dictionary-like containers that map keys to feature vectors, providing a clean
interface for batch feature computation.


## Usage Pattern

1. **Define templates** with shared parameters for feature types you'll use repeatedly
2. **Create composite features** using schemas to define consistent feature structures
  - You will typically have a single top-level composite feature that contains all sub-features for
    your task. You might also have other composite features nested within it.
3. **Use feature maps** to organize and access features by key
4. **Call `get()`** on any feature to obtain its numpy array representation


## Class Details

**Template**: Factory for creating feature instances with shared parameters. Tracks all instances
created. Initialize with the feature class and all shared parameters as keyword arguments. Then you
can call `create()` to make new feature instances, optionally with additional instance-specific
parameters.

**Feature**: Abstract base class defining the feature interface. Subclasses implement `_get()` to
return numpy arrays. The public interface is to call `get()`, which also performs validation (and
optional caching, in subclasses). Features can have names and access attributes from their
templates if not found in the instance. They can also be updated via `update()`.

**CompositeFeature**: Schema-based feature that combines multiple sub-features. A schema is an
`OrderedDict` of fields mapped to `Template` instances, stored in the class variable `SCHEMA`. Each
subclass must have a `define_schema()` method that populates the `SCHEMA` by adding feature
templates. For convenience (and some validation), rather than adding to `SCHEMA` directly,
subclasses can call the `add_schema()` method to add individual templates to the schema.

During initialization, the composite feature pre-allocates space for all schema features. To
populate the sub-features, you call the `_set()` method with the name of the schema field and any
arguments needed to create the feature from the corresponding template. After setting all features,
call `update()` to ensure that all schema features have been initialized, using any provided
defaults.

Finally, call `get()` (as with all Features) to get the final feature vector, which concatenates the
results from all child features.

**ConstantFeature**: Returns constant values as numpy arrays.

**EnumFeature**: Encodes categorical values using various encoding schemes (onehot, integer, binary,
target, hash).

**PairwiseMax**: Computes maximum similarity between two sets using a comparison function.

**TimeContext**: Extracts temporal features from timestamps (day of week, hour, etc).

**Recency**: Computes time differences between timestamps with optional log transformation.

**MappingFeature**: Wraps dictionary-like objects as features.

**FunctionFeature**: Wraps arbitrary functions as features.

**FeatureMap**: Dictionary-like container mapping keys to feature vectors.


For groups of features put together, as well as storage and retrieval, see feature_set.py


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
import json
import logging
import os
import time

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from os.path import dirname
from typing import Any, Sequence, TypeVar, Generic, Callable, Iterator, Hashable, Type

import numpy as np

from matplotlib.pyplot import imsave

from nkpylib.time_utils import parse_ts

logger = logging.getLogger(__name__)

__all__ = [
    'Feature', 'CompositeFeature', # base classes
    'ConstantFeature', 'EnumFeature', 'PairwiseMax', 'TimeContext', 'Recency', # specific features
    'MappingFeature', 'FunctionFeature', 'FeatureMap', # utility classes
]

class PrintableJSONEncoder(json.JSONEncoder):
    """A JSON encoder that takes every non-serializable object and converts it to a string."""
    def default(self, obj):
        """A non-serializable type is converted to a string."""
        if is_dataclass(obj):
            data = asdict(obj)
            return self._convert_tuple_keys(data)
        elif isinstance(obj, Enum):
            return obj.value
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

    def _convert_tuple_keys(self, obj):
        """Convert tuple keys in dicts to strings recursively."""
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    k = str(k)
                if isinstance(v, dict):
                    v = self._convert_tuple_keys(v)
                new_obj[k] = v
            return new_obj
        elif isinstance(obj, (list, tuple)):
            return [self._convert_tuple_keys(v) for v in obj]
        return obj

    def encode(self, obj):
        """Dicts are normally serializable, but we have to make sure the keys are str"""
        if isinstance(obj, dict):
            obj = {str(k): v for k, v in obj.items()}
            # do it recursively
            for k, v in obj.items():
                if isinstance(v, dict):
                    obj[k] = self.encode(v)
        return super().encode(obj)


def recursive_json(o: Any) -> Any:
    """Recursively convert strings that are JSON to objects"""
    def try_json(v: str) -> Any:
        try:
            return json.loads(v)
        except Exception:
            return v
    if isinstance(o, dict):
        return {k: recursive_json(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [recursive_json(v) for v in o]
    elif isinstance(o, str):
        return try_json(o)
    else:
        return o

def make_jsonable(obj: Any, **kw) -> Any:
    """Convert an object to a JSON-serializable form.

    This uses the `PrintableJSONEncoder` to convert any non-serializable objects to strings. It
    will first serialize to JSON string and then parse back to a Python object.
    """
    # first try a direct json.dumps
    try:
        json.dumps(obj)
        ret = obj
    except Exception:
        pass
    # now do it via our encoder
    ret = json.loads(json.dumps(obj, cls=PrintableJSONEncoder, **kw))
    # recursively convert strings that are JSON to objects
    return recursive_json(ret)




class Feature(ABC):
    """Base class for all features"""
    def __init__(self, name: str=''):
        self.name = name or self.__class__.__name__

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name} [{len(self)}]>'

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return len(self.get())

    @abstractmethod
    def _get(self, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array (implementation)"""
        raise NotImplementedError()

    def get(self, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array with validation."""
        ret = self._get(*args, **kw)
        self.validate(ret, self)
        return ret

    def validate(self, arr, feat):
        """Validates the feature `arr` obtained from feature `feat`."""
        try:
            assert len(arr) > 0, f"Feature {feat} has length 0"
        except Exception as e:
            assert False, f"{type(e)} with feature {feat}: value {arr}"


class CompositeFeature(Feature):
    """Feature with children that concatenates their outputs."""

    def __init__(self, **kw):
        """Initialize this composite feature."""
        super().__init__(**kw)
        self._children: list[Feature] = []

    @property
    def children(self) -> list[Feature]:
        """Get list of child features."""
        return self._children

    def add_child(self, feature: Feature) -> None:
        """Add a child feature."""
        self._children.append(feature)

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return sum(len(c) for c in self.children)

    def _get(self, *args, **kw) -> np.ndarray:
        """Composite features concatenate children."""
        arrays = [c.get(*args, **kw) for c in self.children]
        for child, arr in zip(self.children, arrays):
            self.validate(arr, child)
        return np.concatenate(arrays)


class ConstantFeature(Feature):
    """A constant feature (could be many dims)"""
    def __init__(self, values: int|float|Sequence[int|float]=None, **kw):
        """Initializes the constant feature"""
        super().__init__(**kw)
        if values is not None:
            if isinstance(values, (int, float)):
                values = [values]
            self.values = np.array(values)
        else:
            self.values = None

    def _get(self, values: int|float|Sequence[int|float]=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided values or fall back to stored ones
        vals = values if values is not None else self.values
        if vals is None:
            raise ValueError("No values provided to ConstantFeature")
        if isinstance(vals, (int, float)):
            vals = [vals]
        return np.array(vals)


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
                 enum_values: Sequence[EnumT] | Mapping[EnumT, float|int|Sequence[float]] | int,
                 encoding: str='onehot',
                 **kw):
        """Initialize enum feature encoder.

        Args:
        - enum_values:
          - if encoding is 'onehot', 'int', or 'binary', this is the list of all possible enum
            values, in the order to use for encoding
          - if encoding is 'target', this is a mapping from enum values to target value. The target
            value can be a single number or an array of numbers (e.g. an embedding)
          - if encoding is 'hash', this is the number of number of hash bins to use.
            we use the built-in hash function and modulo to get a bin index.
        - encoding: One of 'onehot', 'int', 'binary', 'target', 'hash'
        """
        super().__init__(**kw)
        self.enum_values = enum_values
        self.encoding = encoding

    def _get(self, val: EnumT, *args, **kw) -> np.ndarray:
        """Returns the encoded feature as a numpy array."""
        if val is None:
            raise ValueError("No value provided to EnumFeature")
        match self.encoding:
            case 'onehot':
                idx = self.enum_values.index(val)
                arr = np.zeros(len(self.enum_values))
                arr[idx] = 1
                return arr
            case 'int':
                return np.array([self.enum_values.index(val)])
            case 'binary':
                idx = self.enum_values.index(val)
                n_bits = int(np.ceil(np.log2(len(self.enum_values))))
                binary = format(idx, f'0{n_bits}b')
                return np.array([int(b) for b in binary])
            case 'target':
                v = self.enum_values[val]
                if isinstance(v, (int, float)):
                    v = [v]
                return np.array(v)
            case 'hash':
                hash_val = hash(str(val))
                bin_idx = hash_val % self.enum_values
                arr = np.zeros(self.enum_values)
                arr[bin_idx] = 1
                return arr


T = TypeVar('T')
class PairwiseMax(Feature, Generic[T]):
    """Computes max over two sets of keys that can be compared pairwise.

    You can use this to compute the max similarity between two sets of keys.
    Each set of keys is a sequence of objects of type T, and is compared using a similarity function
    that returns a float, of which we take the maximum.
    """
    #TODO what is the default needed for?
    def __init__(self,
                 sim_func: Callable[[T, T], float],
                 default: float=-1.0,
                 **kw):
        super().__init__(**kw)
        self.sim_func = sim_func
        self.default = default

    def _get(self, keys1: Sequence[T]=None, keys2: Sequence[T]=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        if keys1 is None or keys2 is None:
            raise ValueError("No keys provided to PairwiseMax")
        try:
            keys = [(key1, key2) for key1 in keys1 for key2 in keys2]
            values = [self.sim_func(key1, key2) for key1, key2 in keys]
            return np.array([max(values)])
        except ValueError:
            return np.array([self.default])

    def _len(self) -> int:
        """Returns the length of the feature"""
        return 1


class TimeContext(Feature):
    """Computes some features based on the time context"""
    def __init__(self, fields=('dow', 'hour'), **kw):
        """Initializes the feature with field configuration.

        The types of fields that we populate are:
        - 'dow': day of week (0-6)
        - 'hour': hour of day (0-23)
        """
        super().__init__(**kw)
        self.fields = fields
        for field in fields:
            if field not in ['dow', 'hour']:
                raise NotImplementedError(f"Unknown field {field}")

    def _get(self, ts: float|str, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        parsed_ts = parse_ts(ts)
        ret = []
        for field in self.fields:
            if field == 'dow':
                ret.append(datetime.datetime.fromtimestamp(parsed_ts).weekday())
            elif field == 'hour':
                ret.append(datetime.datetime.fromtimestamp(parsed_ts).hour)
            else:
                raise ValueError(f"Unknown field {field}")
        return np.array(ret)


class Recency(Feature):
    """Computes the recency between two timestamps.

    Internally we convert them to floats.
    """
    def __init__(self, apply_log: bool=True, **kw):
        """

        If `apply_log` is True, we apply log1p to the difference (in that case clamping to positive
        first).
        """
        super().__init__(**kw)
        self.apply_log = apply_log

    def _get(self, a: float|str, b: float|str, *args, **kw) -> np.ndarray:
        """Returns the recency (i.e., time difference) as a numpy array.

        The timestamps `a` and `b` can either be floats (epoch seconds) or strings (ISO format).
        Note that this is NOT symmetric, we assume `a` is the more recent date.
        """
        assert a is not None
        assert b is not None
        parsed_a = parse_ts(a)
        parsed_b = parse_ts(b)
        try:
            ret = parsed_a - parsed_b
        except Exception as e:
            logger.error(f"{type(e)} Error with {parsed_a} - {parsed_b}: {e}")
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
    def __init__(self, mapping: Mapping[T, Any], **kw):
        """Initializes the feature with a mapping object."""
        super().__init__(**kw)
        self.mapping = mapping

    def _get(self, key: T, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        return np.array(self.mapping[key])


class FunctionFeature(Feature):
    """A feature that is computed by a function."""
    def __init__(self, func: Callable, **kw):
        """Initializes the feature with a function `func`."""
        super().__init__(**kw)
        self.func = func

    def _get(self, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided args/kwargs or fall back to stored ones
        return np.array(self.func(*args, **kw))


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
    """Movie Features.

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
    from nkpylib.ml.feature_set import NumpyLmdb

    def __init__(self, m, *, enum_dbs: dict[str, NumpyLmdb], **kw):
        super().__init__(**kw)
        self.m = m
        self.enum_dbs = enum_dbs
        self._setup_features()

    def _setup_features(self):
        """Set up all child features."""
        # Basic features
        self.add_child(ConstantFeature(name='year'))
        self.add_child(ConstantFeature(name='runtime'))

        # Rating features
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                self.add_child(ConstantFeature(name=f'{src}_{field}'))
            self.add_child(ConstantFeature(name=f'{src}_log_votes'))

        # Job count features
        for job in ['actor', 'actress', 'director', 'writer']:
            self.add_child(ConstantFeature(name=f'num_{job}'))

        # Financial features
        self.add_child(ConstantFeature(name='tmdb_budget'))
        self.add_child(ConstantFeature(name='tmdb_log_budget'))
        self.add_child(ConstantFeature(name='tmdb_revenue'))
        self.add_child(ConstantFeature(name='tmdb_log_revenue'))

        # Content rating
        self.add_child(EnumFeature(
            enum_values=[None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR'],
            encoding='int',
            name='rt_content_rating'
        ))

        # Enum embeddings
        for key, db in sorted(self.enum_dbs.items()):
            self.add_child(ConstantFeature(name=f'{key}_emb'))

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

    def _extract_enum_embs(self, m):
        """Extract enum embeddings from movie tags."""
        enum_embs = defaultdict(set)
        for t in m.tags:
            key = f'{t.source}-{t.type}'
            if key in self.enum_dbs:
                enum_embs[key].add(t.value)
        return enum_embs

    def _get(self, *args, **kw) -> np.ndarray:
        """Compute all movie features and concatenate them."""
        m = self.m
        feature_values = []

        # Basic features
        feature_values.append([m.year if m.year else 0])
        feature_values.append([m.runtime if m.runtime else 0])

        # Rating features
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        rating_values = {src: {'rating': 0.0, 'votes': 0.0, 'popularity': 0.0} for src in rating_sources}

        for r in m.ratings:
            if r.source in rating_values:
                for field in ['rating', 'votes', 'popularity']:
                    rating_values[r.source][field] = self._try_float(r, field)

        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                feature_values.append([rating_values[src][field]])
            feature_values.append([np.log1p(rating_values[src]['votes'])])

        # Job counts
        job_counts = Counter(tp.job_id.name for tp in m.people)
        for job in ['actor', 'actress', 'director', 'writer']:
            feature_values.append([job_counts.get(job, 0)])

        # Financial features
        budget, revenue = self._extract_financials(m)
        feature_values.append([budget])
        feature_values.append([np.log1p(budget)])
        feature_values.append([revenue])
        feature_values.append([np.log1p(revenue)])

        # Content rating
        content_rating = self._extract_content_rating(m)
        rating_enum = EnumFeature(
            enum_values=[None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR'],
            encoding='int'
        )
        feature_values.append(rating_enum.get(content_rating))

        # Enum embeddings
        enum_embs = self._extract_enum_embs(m)
        for key, db in sorted(self.enum_dbs.items()):
            values = enum_embs.get(key, set())
            embs = [db[v] for v in values if v in db]
            value = np.mean(embs, axis=0) if len(embs) > 0 else np.zeros(db.n_dims, dtype=np.float32)
            feature_values.append(value)

        # Concatenate all features
        arrays = [np.array(fv) for fv in feature_values]
        return np.concatenate(arrays)
