"""Various ml-related utils.

These should eventually get refactored more properly.
"""

from __future__ import annotations

import functools
import gc
import logging
import time

from abc import abstractmethod
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from queue import Queue, Full, Empty
from threading import Thread
from typing import Callable, Sequence, Any, Iterator

import joblib # type: ignore
import numpy as np
import numpy.random as npr
import psutil
import torch
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv # type: ignore
from torch_geometric.nn.models import GAT # type: ignore
from torch_geometric.datasets import Planetoid # type: ignore
from torch_geometric.transforms import NormalizeFeatures # type: ignore
from torch_geometric.data import Data # type: ignore
from tqdm import tqdm

from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
)

logger = logging.getLogger(__name__)

def list_gpu_tensors():
    """Lists all tensors that are currently on the GPU."""
    gpu_tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            gpu_tensors.append(obj)
    return gpu_tensors


def explain_args(*args, **kwargs) -> dict[str|int, Any]:
    """This takes an arbitrary set of input `args` and `kwargs` and "explains" them.

    That means for each one:
    - if it's a scalar, then just return it as-is
    - if it's a list/array, then return its type and shape
    - if it's numpy array or torch tensor, then return its type and shape

    The output is a dict mapping argument index (for `args`) or name (for `kwargs`) to the explanation.
    """
    result: dict[str|int, str] = {}
    def process(arg):
        if hasattr(arg, 'shape'):  # numpy arrays, torch tensors
            ret = f'{type(arg).__name__}{arg.shape}'
            if hasattr(arg, 'dtype'):
                ret += f',{arg.dtype}'
        elif isinstance(arg, (list, tuple)):
            return f'{type(arg).__name__}[{len(arg)}]'
        elif isinstance(arg, dict):
            return f'dict[{len(arg)}]'
        else:  # scalars
            return arg

    # Process positional arguments
    for i, arg in enumerate(args):
        result[i] = process(arg)
    # Process keyword arguments
    for name, arg in kwargs.items():
        result[name] = process(arg)
    return result

def trace(func):
    """Decorator that tracks total time and memory delta for a function.

    Handles both regular functions and generator functions.
    For generators, traces the entire iteration lifecycle.
    """
    import inspect

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        def finish(result, suffix=''):
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            time_delta = end_time - start_time
            memory_delta = end_memory - start_memory
            # Temporarily override findCaller to return the traced function
            original_findCaller = logger.findCaller
            def fake_findCaller(*args, **kwargs):
                return ('', 0, func.__name__, None)

            logger.findCaller = fake_findCaller
            try:
                logger.info(f"{time_delta:.3f}s, {start_memory:.2f}GB -> {end_memory:.2f}GB (Δ{memory_delta:+.2f}GB) {suffix}")
                logger.debug(f'Inputs: {explain_args(*args, **kwargs)}, Outputs: {explain_args(result)}')
            finally:
                logger.findCaller = original_findCaller

        try:
            result = func(*args, **kwargs)

            # Check if result is a generator
            if inspect.isgenerator(result):
                def traced_generator():
                    try:
                        yield_count = 0
                        while True:
                            value = next(result)
                            yield_count += 1
                            yield value
                    except StopIteration:
                        finish(result, f' [gen, {yield_count}x]')
                        return

                return traced_generator()
            else:
                # Regular function
                finish(result)
                return result
        except Exception as e:
            finish(e, ' [FAILED]')
            raise

    return wrapper

