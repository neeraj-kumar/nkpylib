"""A simple feature container."""

from __future__ import annotations

import datetime
import logging
import os
import time

from abc import ABC
from os.path import dirname
from typing import Any, Sequence, TypeVar, Generic, Callable

import numpy as np

from matplotlib.pyplot import imsave

from nkpylib.time_utils import parse_ts

logger = logging.getLogger(__name__)

class Feature(ABC):
    """Base class for all features"""
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
        """Returns the length of the feature (implementation)"""
        raise NotImplementedError()

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
        """Validates the feature array"""
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


T = TypeVar('T')

class PairwiseMax(Feature, Generic[T]):
    def __init__(self,
                 keys1: Sequence[T],
                 keys2: Sequence[T],
                 sim_func: Callable[[T, T], float],
                 default: float=-1.0,
                 **kw):
        """Computes max over two sets of keys that can be compared pairwise using `sim_func`"""
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
        """Initializes the feature with a timestamp"""
        super().__init__(**kw)
        self.ts = parse_ts(ts)
        self.fields = fields

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
