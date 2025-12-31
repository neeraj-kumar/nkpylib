"""Basic classes and utilities for ML features.

## Core Concepts

Ultimately, the goal is to have an object that takes in a set of inputs and outputs a numpy feature
vector. One approach would be to have a Feature class define a `get()` method that takes in the
inputs and computes the output ("functionally", i.e., without state). However, this can be
inefficient if we have many features that share common parameters or configurations.

However, since features come in many different

However, given that features are often defined in families where there's a lot of
commonalities between them, this library uses a template-based architecture for efficient feature
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
    'Template', 'Feature', 'CompositeFeature', # base classes
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


class Template(Mapping):
    """Base class for feature templates that hold shared parameters."""
    def __init__(self, feature_class: Type[Feature], **shared_params: Any) -> None:
        """Create template for given feature class with shared parameters."""
        self.feature_class = feature_class
        self.shared_params = shared_params
        self._instances: list[Feature] = []  # Track all instances created from this template

    def create(self, *args, **instance_params: Any) -> Feature:
        """Create new feature instance from this template."""
        instance = self.feature_class(template=self, *args, **instance_params, **self.shared_params)
        self._instances.append(instance)
        return instance

    def __repr__(self) -> str:
        return f'<Template {self.feature_class.__name__}: {len(self.shared_params)} kw, {len(self._instances)} instances>'

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

    def get_schema(self, jsonable:bool=False) -> dict:
        """Return schema of shared parameters."""
        ret = dict(type='template',
                   id=id(self),
                   feature_class=self.feature_class.__name__,
                   feature_schema=self.feature_class.get_schema(jsonable=jsonable) if hasattr(self.feature_class, 'get_schema') else None,
                   n_instances=self.num_instances,
                   params=self.shared_params)
        if jsonable:
            ret = make_jsonable(ret)
        return ret


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

    def update(self, **kw):
        """Updates the feature - override in subclasses as needed."""
        pass


class CompositeFeature(Feature):
    """Feature with children, defined via schema."""
    SCHEMA: OrderedDict = OrderedDict()
    DEFAULTS: dict = {}  # Store default args/kwargs by name

    @classmethod
    @abstractmethod
    def define_schema(cls):
        """Must be defined by subclasses to define the schema.

        They should call `add_schema()` for each feature.
        """
        pass

    @classmethod
    def add_schema(cls, name: str, template: Template, default: dict = None):
        """Add a feature to the schema with given `name`, initialized from given `template`.

        These are stored in order of addition.

        When the class is initialized, and update() is called, we ensure that all schema
        features have been initialized. If you want a schema item to be auto-initialized, you can
        provide `default`, a dict of kwargs to pass to the template when creating the feature during
        update().
        """
        assert name not in cls.SCHEMA
        cls.SCHEMA[name] = template
        if default is not None:
            cls.DEFAULTS[name] = default

    @classmethod
    def get_schema(cls, jsonable:bool=False) -> dict:
        """Return the schema of this class as a dict."""
        ret = dict(type='composite_feature',
                   feature_class=cls.__name__,
                   schema=[{
                       'name': name,
                       'template': template.get_schema(jsonable=jsonable),
                       'default': cls.DEFAULTS.get(name),
                   } for name, template in cls.SCHEMA.items()])
        if jsonable:
            ret = make_jsonable(ret)
        return ret

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

    def _get(self) -> np.ndarray:
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
        """Ensure all schema features have been initialized, using defaults if available."""
        if not self.SCHEMA:
            return
        for i, name in enumerate(self.SCHEMA):
            if self._children[i] is None:
                if name in self.DEFAULTS:
                    # Create default feature using template and default args
                    template = self.SCHEMA[name]
                    default_kwargs = self.DEFAULTS[name]
                    self._children[i] = template.create(name=name, **default_kwargs)
                else:
                    # No default provided - this is an error
                    raise ValueError(f"Uninitialized feature with no default: {name}")

    def update(self, **kw):
        """Updates the feature and validates completion for schema-based features."""
        if self.SCHEMA:
            self.validate_complete()


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
                 value: EnumT=None,
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
        - value: The enum value to encode (can be provided here or in get())
        """
        super().__init__(**kw)
        self.enum_values = enum_values
        self.encoding = encoding
        self.value = value

    def _get(self, value: EnumT=None, *args, **kw) -> np.ndarray:
        """Returns the encoded feature as a numpy array."""
        # Use provided value or fall back to stored one
        val = value if value is not None else self.value
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
    def __init__(self,
                 sim_func: Callable[[T, T], float],
                 default: float=-1.0,
                 keys1: Sequence[T]=None,
                 keys2: Sequence[T]=None,
                 **kw):
        super().__init__(**kw)
        self.sim_func = sim_func
        self.default = default
        self.keys1 = keys1
        self.keys2 = keys2

    def _get(self, keys1: Sequence[T]=None, keys2: Sequence[T]=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided keys or fall back to stored ones
        k1 = keys1 if keys1 is not None else self.keys1
        k2 = keys2 if keys2 is not None else self.keys2
        if k1 is None or k2 is None:
            raise ValueError("No keys provided to PairwiseMax")
        
        try:
            keys = [(key1, key2) for key1 in k1 for key2 in k2]
            values = [self.sim_func(key1, key2) for key1, key2 in keys]
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
    def __init__(self, fields=['dow', 'hour'], ts: float|str=None, **kw):
        """Initializes the feature with field configuration.

        The types of fields that we populate are:
        - 'dow': day of week (0-6)
        - 'hour': hour of day (0-23)
        """
        super().__init__(**kw)
        self.fields = fields
        self.ts = parse_ts(ts) if ts is not None else None
        for field in fields:
            if field not in ['dow', 'hour']:
                raise NotImplementedError(f"Unknown field {field}")

    def _get(self, ts: float|str=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided timestamp or fall back to stored one
        timestamp = ts if ts is not None else self.ts
        if timestamp is None:
            raise ValueError("No timestamp provided to TimeContext")
        
        parsed_ts = parse_ts(timestamp)
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
    """Computes the recency between two timestamps"""
    def __init__(self, apply_log: bool=True, a: float|str=None, b: float|str=None, **kw):
        """The dates can either be as floats (epoch seconds) or as strings (ISO format).

        Note that this is NOT symmetric, we assume `a` is the more recent date.
        If `apply_log` is True, we apply log1p to the difference (in that case clamping to positive
        first).

        Internally we convert them to floats.
        """
        super().__init__(**kw)
        self.apply_log = apply_log
        self.a = parse_ts(a) if a is not None else None
        self.b = parse_ts(b) if b is not None else None

    def _get(self, a: float|str=None, b: float|str=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided timestamps or fall back to stored ones
        ts_a = a if a is not None else self.a
        ts_b = b if b is not None else self.b
        if ts_a is None or ts_b is None:
            raise ValueError("No timestamps provided to Recency")
        
        parsed_a = parse_ts(ts_a)
        parsed_b = parse_ts(ts_b)
        
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
    def __init__(self, d: Mapping=None, key: T=None, **kw):
        """Initializes the feature with a mapping object `d` and a `key`."""
        super().__init__(**kw)
        self.d = d
        self.key = key

    def _get(self, d: Mapping=None, key: T=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided mapping/key or fall back to stored ones
        mapping = d if d is not None else self.d
        k = key if key is not None else self.key
        if mapping is None or k is None:
            raise ValueError("No mapping or key provided to MappingFeature")
        return np.array(mapping[k])


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

    def _get(self, func_args: Sequence[Any]=None, func_kwargs: Mapping[str, Any]=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided args/kwargs or fall back to stored ones
        f_args = func_args if func_args is not None else self.func_args
        f_kwargs = func_kwargs if func_kwargs is not None else self.func_kwargs
        return self.func(*f_args, **f_kwargs)


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
    from nkpylib.ml.feature_set import NumpyLmdb
    @classmethod
    def define_schema(cls, enum_dbs):
        """Define the schema for movie features."""
        # lambda to create a constant feature of given length
        C = lambda n_dims: Template(ConstantFeature, n_dims=n_dims)
        # Features that should always exist (no defaults)
        cls.add_schema('year', C(1))
        cls.add_schema('runtime', C(1))
        # Rating features with defaults for missing sources
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                cls.add_schema(f'{src}_{field}', C(1), default={'values': 0.0})
            cls.add_schema(f'{src}_log_votes', C(1), default={'values': 0.0})
        # Job count features with defaults
        for job in ['actor', 'actress', 'director', 'writer']:
            cls.add_schema(f'num_{job}', C(1), default={'values': 0.0})
        # Financial features with defaults
        cls.add_schema('tmdb_budget', C(1), default={'values': 0.0})
        cls.add_schema('tmdb_log_budget', C(1), default={'values': 0.0})
        cls.add_schema('tmdb_revenue', C(1), default={'values': 0.0})
        cls.add_schema('tmdb_log_revenue', C(1), default={'values': 0.0})
        # Content rating with default
        rating_enum = Template(EnumFeature,
                             enum_values=[None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR'],
                             encoding='int')
        cls.add_schema('rt_content_rating', rating_enum, default={'value': None})
        for key, db in sorted(enum_dbs.items()):
            cls.add_schema(f'{key}_emb', C(db.n_dims),
                           default={'values': np.zeros(db.n_dims, dtype=np.float32)})

    def __init__(self, m, *, enum_dbs: dict[str, NumpyLmdb], **kw):
        super().__init__(**kw)
        self.m = m
        self.enum_dbs = enum_dbs
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

    def _extract_enum_embs(self, m):
        """Extract enum embeddings from movie tags."""
        enum_embs = defaultdict(set)
        for t in m.tags:
            key = f'{t.source}-{t.type}'
            if key in self.enum_dbs:
                enum_embs[key].add(t.value)
        return enum_embs

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
            key = f'num_{job}'
            if key in self.SCHEMA:
                self._set(key, count)
        # Financial features
        budget, revenue = self._extract_financials(m)
        self._set('tmdb_budget', budget)
        self._set('tmdb_log_budget', np.log1p(budget))
        self._set('tmdb_revenue', revenue)
        self._set('tmdb_log_revenue', np.log1p(revenue))
        # Content rating
        content_rating = self._extract_content_rating(m)
        self._set('rt_content_rating', content_rating)
        enum_embs = self._extract_enum_embs(m)
        # Enum embeddings
        for key, values in enum_embs.items():
            db = self.enum_dbs[key]
            embs = [db[v] for v in values if v in db]
            # average all the embeddings together
            value = np.mean(embs, axis=0) if len(embs) > 0 else np.zeros(db.n_dims, dtype=np.float32)
            self._set(f'{key}_emb', values=value)
        # Validate all features are set
        super().update()



NO_TIMESTAMP = time.time() - (5*YEAR_SECS) # see if this messes up scaling

AirtableRow = dict[str, Any]

class FoodDishFeature(Feature):
    all_dists: dict[tuple[str, str], float] | None  = None
    text_col = load_chroma_db().get_collection('recipe-texts')
    img_col = load_chroma_db().get_collection('recipe-images')
    lock = Lock()

    def __init__(self, fl_row: AirtableRow, dish_row: AirtableRow, **kw):
        """Given a food log row and a dish row, initializes this feature"""
        super().__init__(name=f"FoodDishFeature<{fl_row['ID']}, {dish_row['path']}>")
        self.fl_row: Optional[AirtableRow] = None
        self.dish_row: Optional[AirtableRow] = None
        t0 = time.time()
        self.update(fl_row, dish_row)
        t1 = time.time()
        #logger.info(f'Took {t1-t0:.4f}s to initialize {self.name}')

    def photo_names(self, row: AirtableRow, field: str):
        """Given an airtable row and a field with photos, returns unquoted filenames"""
        return [unquote_plus(p['filename']) for p in row.get(field, [])]

    def clip_similarity(self, key1: str, key2: str) -> float:
        """Returns the similarity between the two clip embedding keys.

        We precompute this the first time for all pairs and then store it as a dict from
        (key1, key2) to similarity.
        """
        with self.__class__.lock:
            if self.__class__.all_dists is None:
                self.__class__.all_dists = {}
                # first get food log photos (these are all source=mine)
                resp = self.img_col.get(include=['metadatas', 'embeddings'], where={'source': 'mine'})
                fl_keys = resp['ids']
                fl_features = np.array(resp['embeddings'])
                logger.info(f'Got food log photo {fl_features.shape} features in {self.img_col.name}')
                # now get all photos
                resp = self.img_col.get(include=['metadatas', 'embeddings'])
                img_keys = resp['ids']
                img_features = np.array(resp['embeddings'])
                # now get all texts
                resp = self.text_col.get(include=['metadatas', 'embeddings'])
                text_keys = resp['ids']
                text_features = np.array(resp['embeddings'])
                # concat all ids and features for matching
                match_keys = img_keys + text_keys
                match_features = np.vstack([img_features, text_features])
                logger.info(f'Got {match_features.shape} match features')
                dists = cdist(fl_features, match_features, 'cosine')
                for i, key1 in enumerate(fl_keys):
                    for j, key2 in enumerate(match_keys):
                        self.__class__.all_dists[key1, key2] = float(1 - dists[i, j])
            try:
                return self.__class__.all_dists[key1, key2]
            except KeyError:
                return -1.0

    def update(self, fl_row: AirtableRow, dish_row: AirtableRow, **kw): # type: ignore
        if fl_row == self.fl_row or dish_row == self.dish_row:
            return
        self.fl_row = fl_row
        self.dish_row = dish_row
        # get photos from fl (mine), dish (recipe), dish food log (mine)
        fl_photos = [f"My cooking/{p.split('-', 1)[0]}/{p}" for p in self.photo_names(fl_row, 'Photos')]
        hash = self.dish_row['sha256']
        d_recipe_photos = [f"by_hash/{hash}/images/{p}" for p in self.photo_names(dish_row, 'recipe images')]
        d_my_photos = []
        # get all dish -> food log rows
        matching_fls = []
        for fl in dish_row.get('Food log', []):
            d_my_photos.extend(self.photo_names(fl, 'Photos'))
            matching_fls.append(fl)
        # get latest dish -> food log row
        latest_fl = None
        if matching_fls:
            latest_fl = max(matching_fls, key=lambda fl: fl['Date'])
        # get all dish -> food log rows within last 2 weeks of our food log row
        recent_fls = [fl for fl in matching_fls if parse_ts(fl['Date']) > (parse_ts(fl_row['Date']) - (14*DAY_SECS))]
        text_keys = get_dish_text_fields(dish_row)
        all_text_types = ['path', 'tags', 'site', 'recipe url', 'cuisines', 'diets', 'dish types', 'title', 'name']
        text_features = [PairwiseMax(
            name=f'img_text_similarity:{text_type}',
            keys1=fl_photos,
            keys2=[tk for tk in text_keys if tk.startswith(f'{text_type}:')],
            sim_func=self.clip_similarity,
            default=-1,
        ) for text_type in all_text_types]
        # setup our children
        self.children = [
            PairwiseMax(
                name='recipe_img_similarity',
                keys1=fl_photos,
                keys2=d_recipe_photos,
                sim_func=self.clip_similarity,
                default=-1,
            ),
            PairwiseMax(
                name='my_img_similarity',
                keys1=fl_photos,
                keys2=d_my_photos,
                sim_func=self.clip_similarity,
                default=-1,
            ),
            *text_features,
            Recency(
                name='added_recency',
                a=fl_row['Date'],
                b=dish_row.get('Recipe added', NO_TIMESTAMP),
            ),
            Recency(
                name='comment_recency',
                a=fl_row['Date'],
                b=dish_row.get('last comment', NO_TIMESTAMP),
            ),
            Recency(
                name='dish_last_fl_recency',
                a=fl_row['Date'],
                b=latest_fl.get('Date', NO_TIMESTAMP) if latest_fl else NO_TIMESTAMP,
            ),
            ConstantFeature(
                name='dish_fl_counts',
                values=(len(recent_fls), len(matching_fls)),
            ),
            TimeContext(
                name='fl_time',
                ts=fl_row['Date'],
            ),
        ]
