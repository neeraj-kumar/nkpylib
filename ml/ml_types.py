"""Various types for ML-related funcs"""
from __future__ import annotations

from functools import reduce
from typing import Union, Tuple, Sequence

import numpy as np

FLOAT_TYPES = (float, np.float32, np.float64)
NUMERIC_TYPES = FLOAT_TYPES + (int, np.int32, np.int64)
NumericT = Union[NUMERIC_TYPES]

nparray1d = np.ndarray
nparray2d = np.ndarray

array1d = nparray1d | Sequence[float]
array2d = nparray2d | Sequence[Sequence[float]]

