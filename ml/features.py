"""Basic classes and utilities for ML features.

## Core Concepts
Ultimately, the goal is to have an object that takes in a set of inputs and outputs a numpy feature
vector, in a functional way (i.e., the same set of inputs yields the same output feature vector). In
many cases, you might need some bit of configuration to determine how to compute the features. So
the general pattern is:

    f = Feature(config)
    all_vectors = [f.get(input) for input in inputs]

**Features**: Individual feature computors that return numpy arrays. Features can be simple (single
values) or composite (combining multiple sub-features). All features implement a common interface
with `get()` returning validated numpy arrays.

**Composite Features**: Features that combine multiple child features using a compute-then-delegate
pattern. Child features are stored in an OrderedDict by name, and the composite feature computes
all input values upfront via `values_to_dict()`, then delegates to each child feature in order.

**Feature Maps**: Dictionary-like containers that map keys to feature vectors, providing a clean
interface for batch feature computation.


## Usage Pattern

1. **Create composite features** by subclassing `CompositeFeature` and implementing `values_to_dict()`
  - Add child features using `add()` method during initialization
  - The composite feature will automatically concatenate child outputs in order
2. **Use feature maps** to organize and access features by key
3. **Call `get()`** on any feature to obtain its numpy array representation


## Class Details

**Feature**: Abstract base class defining the feature interface. Subclasses implement `_get()` to
return numpy arrays. The public interface is to call `get()`, which also performs validation.
Key methods:
- `get(*args, **kwargs)`: Returns the feature as a numpy array with validation
- `schema()`: Returns JSON-serializable metadata about the feature
- `from_schema(schema)`: Class method to reconstruct feature from schema

**CompositeFeature**: Base class for features that combine multiple child features. Uses a
compute-then-delegate pattern where `values_to_dict()` computes all input values upfront, then
each child feature processes its corresponding value. Child features are stored by name in an
OrderedDict and processed in order. Key methods:
- `values_to_dict(*args, **kwargs)`: Must be implemented by subclasses to compute input values
- `add(feature)`: Add a child feature by name
- `explain(vector)`: Map feature vector back to child feature contributions

**ConstantFeature**: Returns constant numerical values as numpy arrays. Can store values at
initialization or accept them dynamically at runtime.

**EnumFeature**: Encodes categorical values using various encoding schemes (onehot, integer, binary,
target, hash). Supports different enum value formats depending on encoding type.

**PairwiseMax**: Computes maximum similarity between two sets using a comparison function.

**TimeContext**: Extracts temporal features from timestamps (day of week, hour, etc).

**Recency**: Computes time differences between timestamps with optional log transformation.

**MappingFeature**: Wraps dictionary-like objects as features for key-based lookups.

**FunctionFeature**: Wraps arbitrary functions as features.

**FeatureMap**: Dictionary-like container mapping keys to feature vectors.

**MovieFeature**: Example composite feature that extracts comprehensive movie features including
ratings, financial data, cast/crew counts, and embeddings from enum databases.


For groups of features put together, as well as storage and retrieval, see feature_set.py


Key methods that all features implement:
- `get(*args, **kwargs)`: Returns the feature as a numpy array with validation
- `__len__()`: Returns the expected length of the feature vector
- `schema()`: Returns JSON-serializable metadata about the feature structure
- `from_schema(schema)`: Class method to reconstruct feature from schema dictionary

Optional methods subclasses may implement:
- `validate(arr, feat)`: Custom validation logic (default checks length consistency)

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

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the feature"""
        raise NotImplementedError()

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
            assert len(feat) == len(arr), f"Feature {feat} expected length {len(feat)}, got {len(arr)}"
        except Exception as e:
            assert False, f"{type(e)} with feature {feat}: value {arr}"

    def schema(self) -> dict:
        """Get schema information for this feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and other metadata
        """
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'dims': len(self),
        }

    @classmethod
    def from_schema(cls, schema: dict) -> Feature:
        """Create a feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - Feature instance
        """
        return cls(name=schema.get('name', ''))


class CompositeFeature(Feature):
    """Feature with children that concatenates their outputs."""
    def __init__(self, **kw):
        """Initialize this composite feature."""
        super().__init__(**kw)
        self._children: OrderedDict[str, Feature] = OrderedDict()

    @property
    def children(self) -> list[Feature]:
        """Get list of child features."""
        return list(self._children.values())

    def add(self, feature: Feature) -> None:
        """Add a child feature."""
        self._children[feature.name] = feature

    def __getitem__(self, name: str) -> Feature:
        """Get a child feature by name"""
        return self._children[name]

    def __iter__(self) -> Iterator[Feature]:
        """Iterate over child features."""
        return iter(self.children)

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return sum(len(c) for c in self.children)

    @abstractmethod
    def values_to_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Compute all feature inputs and return as name->value mapping.

        Subclasses should override this method to convert input args into this dict.
        """
        raise NotImplementedError("Subclasses must implement values_to_dict()")

    def _get(self, *args, **kwargs) -> np.ndarray:
        """Computes the feature vector for this composite feature.

        This generates a dictionary of values for all child features using `values_to_dict()`, then
        iterates through the child features in order, calling `get()` on each with the corresponding
        value from the dictionary, and concatenates the results into a single feature vector. It
        also validates that all values from the dictionary are used and logs any errors encountered
        during feature computation.
        """
        values = self.values_to_dict(*args, **kwargs)
        arrays = []
        for feature in self:
            value = values.pop(feature.name)
            try:
                arrays.append(feature.get(value))
            except Exception as e:
                logger.error(f"Error computing feature {feature.name} with value {value}: {e}")
                raise
        assert not values, f"Unused values in feature computation: {values}"
        return np.concatenate(arrays)

    def explain(self, v: np.ndarray) -> dict[str, list[float]]:
        """Explains the feature vector `v` by mapping it back to child feature values.

        This returns a dict that maps from child feature names to their corresponding slices of the
        input vector `v`, based on the lengths of the child features. This can be useful for
        debugging and interpretability.

        Args:
        - v: Feature vector to explain, must have length equal to len(self)

        Returns:
        - Dictionary mapping child feature names to their corresponding slices of v
        """
        if len(v) != len(self):
            raise ValueError(f"Feature vector length {len(v)} doesn't match composite feature length {len(self)}")
        explanation = {}
        start_idx = 0
        for name, child in self._children.items():
            child_len = len(child)
            end_idx = start_idx + child_len
            explanation[name] = [float(x) for x in v[start_idx:end_idx]]
            start_idx = end_idx
        return explanation

    def schema(self) -> dict:
        """Get schema information for this composite feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and child schemas
        """
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'dims': len(self),
            'children': {name: child.schema() for name, child in self._children.items()},
            'num_children': len(self.children),
        }

    @classmethod
    def from_schema(cls, schema: dict) -> CompositeFeature:
        """Create a composite feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - CompositeFeature instance with reconstructed children
        """
        feature = cls(name=schema.get('name', ''))
        children_data = schema.get('children', {})
        for name, child_schema in children_data.items():
            child_type = child_schema['type']
            feature_class = globals().get(child_type)
            if feature_class and issubclass(feature_class, Feature):
                child = feature_class.from_schema(child_schema)
                feature.add(child)
        return feature


class ConstantFeature(Feature):
    """A constant feature (could be many dims)"""
    def __init__(self, values: int|float|Sequence[int|float]|None=None, dims: int=0, **kw):
        """Initializes the constant feature.

        You must provide either `values` (a single number or a sequence of numbers) or `dims` (the
        number of dimensions that will be provided to `get()` at runtime.
        """
        super().__init__(**kw)
        assert values is not None or dims > 0, "Must provide either values or dims"
        self.values: None|np.ndarray
        if values is not None:
            if isinstance(values, (int, float)):
                values = [values]
            self.values = np.array(values)
            self.dims = len(self.values)
        else:
            self.values = None
            self.dims = dims

    def __len__(self) -> int:
        """Returns the length of the feature"""
        return self.dims

    def _get(self, values: int|float|Sequence[int|float]|None=None, *args, **kw) -> np.ndarray:
        """Returns the feature as a numpy array"""
        # Use provided values or fall back to stored ones
        vals = values if values is not None else self.values
        if vals is None:
            raise ValueError("No values provided to ConstantFeature")
        if isinstance(vals, (int, float)):
            vals = [vals]
        return np.array(vals)

    def schema(self) -> dict:
        """Get schema information for this constant feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and storage info
        """
        return {
            'type': 'ConstantFeature',
            'name': self.name,
            'dims': len(self),
            'stored_values': self.values.tolist() if self.values is not None else None,
        }

    @classmethod
    def from_schema(cls, schema: dict) -> ConstantFeature:
        """Create a constant feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - ConstantFeature instance
        """
        stored_values = schema.get('stored_values')
        return cls(values=stored_values, dims=schema.get('dims'), name=schema.get('name', ''))


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

    def __len__(self) -> int:
        """Returns the length of the feature"""
        match self.encoding:
            case 'onehot':
                return len(self.enum_values)
            case 'int':
                return 1
            case 'binary':
                return int(np.ceil(np.log2(len(self.enum_values))))
            case 'target':
                return 1  # Could be more for embeddings, but we don't know without sample
            case 'hash':
                return self.enum_values  # num_bins
            case _:
                return 0

    def _get(self, val: EnumT, *args, **kw) -> np.ndarray:
        """Returns the encoded feature as a numpy array."""
        if val is None and None not in self.enum_values:
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

    def schema(self) -> dict:
        """Get schema information for this enum feature.

        Returns:
        - Dictionary with feature type, name, encoding, enum values, and dimensions
        """
        schema = {
            'type': 'EnumFeature',
            'name': self.name,
            'encoding': self.encoding,
            'dims': len(self),
        }

        # Add enum_values based on encoding type
        match self.encoding:
            case 'onehot' | 'int' | 'binary':
                schema['enum_values'] = list(self.enum_values) if hasattr(self.enum_values, '__iter__') else [self.enum_values]
            case 'target':
                schema['target_mapping'] = dict(self.enum_values) if hasattr(self.enum_values, 'items') else self.enum_values
            case 'hash':
                schema['num_bins'] = self.enum_values

        return schema

    @classmethod
    def from_schema(cls, schema: dict) -> EnumFeature:
        """Create an enum feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - EnumFeature instance
        """
        encoding = schema.get('encoding', 'onehot')

        match encoding:
            case 'onehot' | 'int' | 'binary':
                enum_values = schema.get('enum_values', [])
            case 'target':
                enum_values = schema.get('target_mapping', {})
            case 'hash':
                enum_values = schema.get('num_bins', 10)
            case _:
                enum_values = schema.get('enum_values', [])

        return cls(enum_values=enum_values, encoding=encoding, name=schema.get('name', ''))


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

    def schema(self) -> dict:
        """Get schema information for this pairwise max feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and default value
        """
        return {
            'type': 'PairwiseMax',
            'name': self.name,
            'dims': len(self),
            'default': self.default,
        }

    @classmethod
    def from_schema(cls, schema: dict) -> PairwiseMax:
        """Create a pairwise max feature instance from a schema dictionary.

        Note: The similarity function cannot be serialized, so this creates
        a placeholder that will need to be set manually.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - PairwiseMax instance with placeholder similarity function
        """
        def placeholder_sim_func(a, b):
            return 0.0

        return cls(
            sim_func=placeholder_sim_func,
            default=schema.get('default', -1.0),
            name=schema.get('name', '')
        )

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

    def schema(self) -> dict:
        """Get schema information for this time context feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and fields
        """
        return {
            'type': 'TimeContext',
            'name': self.name,
            'dims': len(self),
            'fields': list(self.fields),
        }

    @classmethod
    def from_schema(cls, schema: dict) -> TimeContext:
        """Create a time context feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - TimeContext instance
        """
        return cls(
            fields=tuple(schema.get('fields', ('dow', 'hour'))),
            name=schema.get('name', '')
        )


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

    def schema(self) -> dict:
        """Get schema information for this recency feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and log setting
        """
        return {
            'type': 'Recency',
            'name': self.name,
            'dims': len(self),
            'apply_log': self.apply_log,
        }

    @classmethod
    def from_schema(cls, schema: dict) -> Recency:
        """Create a recency feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - Recency instance
        """
        return cls(
            apply_log=schema.get('apply_log', True),
            name=schema.get('name', '')
        )

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

    def schema(self) -> dict:
        """Get schema information for this mapping feature.

        Returns:
        - Dictionary with feature type, name, dimensions, and mapping info
        """
        return {
            'type': 'MappingFeature',
            'name': self.name,
            'dims': len(self),
            'num_keys': len(self.mapping),
            'sample_keys': list(self.mapping.keys())[:5] if self.mapping else [],
        }

    @classmethod
    def from_schema(cls, schema: dict) -> MappingFeature:
        """Create a mapping feature instance from a schema dictionary.

        Note: The actual mapping data cannot be serialized, so this creates
        an empty mapping that will need to be populated manually.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - MappingFeature instance with empty mapping
        """
        return cls(mapping={}, name=schema.get('name', ''))


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

    def schema(self) -> dict:
        """Get schema information for this function feature.

        Returns:
        - Dictionary with feature type, name, and dimensions
        """
        return {
            'type': 'FunctionFeature',
            'name': self.name,
            'dims': len(self),
        }

    @classmethod
    def from_schema(cls, schema: dict) -> FunctionFeature:
        """Create a function feature instance from a schema dictionary.

        Note: The function cannot be serialized, so this creates
        a placeholder that will need to be set manually.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - FunctionFeature instance with placeholder function
        """
        def placeholder_func(*args, **kwargs):
            return [0.0]

        return cls(func=placeholder_func, name=schema.get('name', ''))


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

    def __init__(self, *, enum_dbs: dict[str, NumpyLmdb], **kw):
        super().__init__(**kw)
        self.enum_dbs = enum_dbs
        self._setup_features()

    def _setup_features(self):
        """Set up all child features."""
        C = lambda name, dims=1: self.add(ConstantFeature(name=name, dims=dims))
        for field in ['year', 'runtime']: # Basic features
            C(field)
        # Rating features
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                C(f'{src}_{field}')
            C(f'{src}_log_votes')
        # Job count features
        for job in ['actor', 'actress', 'director', 'writer']:
            C(f'num_{job}')
        # Financial features
        for field in ['tmdb_budget', 'tmdb_log_budget', 'tmdb_revenue', 'tmdb_log_revenue']:
            C(field)
        # Content rating
        self.add(EnumFeature(
            enum_values=[None, 'G', 'PG', 'PG-13', 'R', 'NC17', 'NR'],
            encoding='int',
            name='rt_content_rating'
        ))
        # Enum embeddings
        for key, db in sorted(self.enum_dbs.items()):
            self.add(ConstantFeature(name=f'{key}_emb', dims=db.n_dims))

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
        ratings = self['rt_content_rating'].enum_values
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

    def values_to_dict(self, m, *args, **kwargs) -> dict[str, Any]:
        """Compute all movie feature values and return as name->value mapping."""
        d = {}
        d['year'] = m.year if m.year else 0
        d['runtime'] = m.runtime if m.runtime else 0
        # Rating features
        rating_sources = ['imdb', 'tmdb', 'letterboxd', 'rotten_tomatoes_critics', 'rotten_tomatoes_audience']
        rating_values = {src: {'rating': 0.0, 'votes': 0.0, 'popularity': 0.0} for src in rating_sources}
        for r in m.ratings:
            if r.source in rating_values:
                for field in ['rating', 'votes', 'popularity']:
                    rating_values[r.source][field] = self._try_float(r, field)
        for src in rating_sources:
            for field in ['rating', 'votes', 'popularity']:
                d[f'{src}_{field}'] = rating_values[src][field]
            d[f'{src}_log_votes'] = np.log1p(rating_values[src]['votes'])
        # Job counts
        job_counts = Counter(tp.job_id.name for tp in m.people)
        for job in ['actor', 'actress', 'director', 'writer']:
            d[f'num_{job}'] = job_counts.get(job, 0)
        # Financial features
        budget, revenue = self._extract_financials(m)
        d['tmdb_budget'] = budget
        d['tmdb_log_budget'] = np.log1p(budget)
        d['tmdb_revenue'] = revenue
        d['tmdb_log_revenue'] = np.log1p(revenue)
        # Content rating
        d['rt_content_rating'] = self._extract_content_rating(m)
        # Enum embeddings
        enum_embs = self._extract_enum_embs(m)
        for key, db in sorted(self.enum_dbs.items()):
            values = enum_embs.get(key, set())
            embs = [db[v] for v in values if v in db]
            value = np.mean(embs, axis=0) if len(embs) > 0 else np.zeros(db.n_dims, dtype=np.float32)
            d[f'{key}_emb'] = value
        return d

    def schema(self) -> dict:
        """Get schema information for this movie feature.

        This just uses the base class implementation with the addition of the enum_dbs keys.
        """
        base_schema = super().schema()
        base_schema['enum_dbs'] = list(self.enum_dbs.keys())
        return base_schema

    @classmethod
    def from_schema(cls, schema: dict) -> MovieFeature:
        """Create a movie feature instance from a schema dictionary.

        Args:
        - schema: Dictionary containing feature configuration

        Returns:
        - MovieFeature instance
        """
        return cls(enum_dbs=schema['enum_dbs'], name=schema.get('name', ''))
