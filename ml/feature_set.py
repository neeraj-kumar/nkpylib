"""Groups of features put together.

Feature Management:
- FeatureSet: Collection of features with similarity search capabilities
- Support for multiple input sources (files or mapping objects)
- Automatic key intersection across sources

"""

from __future__ import annotations

import logging
import os
import sys
import time

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, Iterator

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm import tqdm

from nkpylib.ml.ml_types import nparray1d, nparray2d, array1d, array2d
from nkpylib.ml.nklmdb import PickleableLmdb, JsonLmdb, MetadataLmdb, NumpyLmdb

logger = logging.getLogger(__name__)

KeyT = TypeVar('KeyT')


class WeightTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for applying feature weights."""
    def __init__(self, weights: nparray1d):
        """Initialize with feature weights.

        - `weights`: 1D array of weights to multiply each feature by
        """
        self.weights = weights

    def fit(self, X: np.ndarray, y=None):
        """Fit the transformer (no-op for weighting)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature weights to input array."""
        return X * self.weights


class FeaturePipelineBuilder:
    """Builder for creating sklearn pipelines for feature transformation."""
    def __init__(self):
        self.steps = []

    def normalize(self, norm: str='l2'):
        """Add normalization step.

        - `norm`: Type of normalization ('l1', 'l2', or 'max')
        """
        self.steps.append(('normalize', Normalizer(norm=norm)))
        return self

    def scale(self, with_mean: bool=True, with_std: bool=True):
        """Add standard scaling step.

        - `with_mean`: Whether to center data to zero mean
        - `with_std`: Whether to scale data to unit variance
        """
        self.steps.append(('scale', StandardScaler(with_mean=with_mean, with_std=with_std)))
        return self

    def rbf_sample(self, n_components: int=4000, gamma: float|str='scale'):
        """Add RBF sampling step for kernel approximation.

        - `n_components`: Number of components to sample
        - `gamma`: Parameter of the RBF kernel, default
        """
        self.steps.append(('rbf', RBFSampler(n_components=n_components, gamma=gamma)))
        return self

    def pca(self, n_components: int=512):
        """Add PCA dimensionality reduction.

        - `n_components`: Number of principal components to keep
        """
        self.steps.append(('pca', PCA(n_components=n_components)))
        return self

    def weight(self, weights: np.ndarray):
        """Add feature weighting step.

        - `weights`: 1D array of weights to multiply each feature by
        """
        self.steps.append(('weight', WeightTransformer(weights)))
        return self

    def build(self) -> Pipeline:
        """Build the final sklearn Pipeline."""
        return Pipeline(self.steps)

class FeatureSet(Mapping, Generic[KeyT]):
    """A set of features that you can do stuff with.

    It is accessible as a mapping of `KeyT` to `np.ndarray`.

    The inputs should be a list of mapping-like objects, or paths to numpy-encoded lmdb files.
    """
    def __init__(self, inputs: list[Any], dtype=np.float32, **kw):
        """Loads features from given list of `inputs`.

        The inputs should either be mapping-like objects, or paths to numpy-encoded lmdb files.
        We compute the intersection of the keys in all inputs, and use that as our list of _keys.
        """
        self.orig_inputs = inputs
        # remap any path inputs to NumpyLmdb objects
        self.dtype = dtype
        self.inputs = [NumpyLmdb.open(inp, flag='r', dtype=dtype) if isinstance(inp, str) else inp
                       for inp in inputs]
        self.reload_keys(reload_lmdb=False)

    def reload_keys(self, reload_lmdb:bool=True) -> None:
        """Reloads our keys"""
        # first reload all our lmdbs
        def rel_inp(i):
            if isinstance(i, NumpyLmdb):
                i.close()
                ret = NumpyLmdb.open(i.path, flag='r', dtype=self.dtype)
                del i
                return ret
            else:
                return i

        if reload_lmdb:
            self.inputs = [rel_inp(inp) for inp in self.inputs]
        logger.debug(f'Reloading keys for FeatureSet with {len(self.inputs)} inputs: {self.inputs}')
        self._keys = self.get_keys()
        self.n_dims = 0
        for key, value in self.items():
            self.n_dims = len(value)
            break

    def __repr__(self) -> str:
        return f'FeatureSet<{len(self.inputs)} inputs, {len(self)} keys, {self.n_dims} dims>'

    def __getstate__(self) -> dict[str, Any]:
        """Returns state of this suitable for pickling.

        This just returns a dict with `inputs` and `dtype`. We replace any NumpyLmdb inputs with
        their paths. If an input is of a non-pickleable type, it will raise an error when you try to
        pickle this (not in this function).

        When you unpickle this, setstate will simply rerun initialization with these.
        """
        return dict(
            inputs=[inp.path if isinstance(inp, NumpyLmdb) else inp for inp in self.inputs],
            dtype=self.dtype,
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Sets state of this from given `state` dict.

        This simply reruns initialization with the given inputs and dtype.
        """
        self.__init__(**state)

    def get_keys(self) -> list[KeyT]:
        """Gets the intersection of all keys by reading all our inputs.

        Useful if the underlying databases might change over time.
        Note that we make no guarantees on correctness due to changing databases!
        """
        keys = []
        for i, inp in enumerate(self.inputs):
            if i == 0:
                keys = list(inp.keys())
            else:
                cur_keys = set(inp.keys())
                keys = [k for k in keys if k in cur_keys]
        return keys

    def __iter__(self) -> Iterator[KeyT]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __getitem__(self, key: KeyT) -> np.ndarray:
        cur = np.hstack([inp[key] for inp in self.inputs])
        return cur

    def fast_get_multi(self, keys: list[KeyT]) -> dict[KeyT, np.ndarray]:
        """A faster way to get multiple keys"""
        assert len(self.inputs) == 1 and isinstance(self.inputs[0], PickleableLmdb), "fast_get_multi only supports single-input FeatureSets"
        inp = self.inputs[0]
        key_set = set(keys)
        return {k: v for k, v in inp.items() if k in key_set}

    def keys_vecs(self, keys: list[KeyT]|None=None) -> tuple[list[KeyT], np.ndarray]:
        """Returns a list of keys and a numpy array of vectors.

        By default we return embeddings for all our keys, but you can optionally pass in a list of
        keys to get embeddings for. Note that these are futher filtered to those we have in our set.
        """
        if keys is None:
            _keys, _vecs = zip(*list(self.items()))
            keys = list(_keys)
            vecs = np.vstack(_vecs)
        else:
            self_keys = set(self._keys)
            done = set()
            new_keys = []
            for k in keys:
                if k in self_keys and k not in done:
                    new_keys.append(k)
                    done.add(k)
            keys = new_keys
            if keys:
                vecs = np.vstack([self[k] for k in keys])
            else:
                vecs = np.zeros((0, self.n_dims), dtype=self.dtype)
        return keys, vecs

    def get_keys_embeddings(self,
                            keys: list[KeyT]|None=None,
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True,
                            scaler:StandardScaler|None=None,
                            return_scaler:bool=False) -> tuple[list[KeyT], np.ndarray]:
        """Returns a list of keys and a numpy array of embeddings.

        By default we return embeddings for all our keys, but you can optionally pass in a list of
        keys to get embeddings for. Note that these are futher filtered to those we have in our set.

        You can optionally set the following flags:
        - `normed`: Normalize embeddings to unit length.
        - `scale_mean`: Scale embeddings to have zero mean.
        - `scale_std`: Scale embeddings to have unit variance.

        Note that the scalings are applied only to the set of keys you fetch embeddings for, so it
        might be degenerate if you request too few keys.

        You can also pass in your own scaler. In that case, we still apply normalization first (if
        `normed` is True), and then apply the given scaler to the embeddings, ignoring the
        `scale_mean` and `scale_std` params.

        If you set `return_scaler` to True, we also return the scaler object used for scaling as the
        last item in the return tuple.
        """
        times = [time.time()]
        keys, embs = self.keys_vecs(keys)
        times.append(time.time())
        if not keys: # early exit
            return (keys, embs, scaler) if return_scaler else (keys, embs)
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        times.append(time.time())
        if scaler is None:
            if scale_mean or scale_std:
                scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
                embs = scaler.fit_transform(embs)
        else:
            embs = scaler.transform(embs)
        times.append(time.time())
        time_names = ['get_keys_embeddings', 'norming', 'scaling']
        timings = {n: times[i+1]-times[i] for i, n in enumerate(time_names)}
        logger.debug(f'timings: {timings}')
        if return_scaler:
            return keys, embs, scaler
        else:
            return keys, embs

    def keys_final_vecs(self,
                        keys: list[KeyT]|None=None,
                        pipeline: Pipeline|None=None,
                        fit_pipeline: bool=True) -> tuple[list[KeyT], np.ndarray]:
        """Returns a list of keys and a numpy array of final transformed vectors.

        By default we return vectors for all our keys, but you can optionally pass in a list of
        keys to get vectors for. Note that these are further filtered to those we have in our set.

        - `pipeline`: sklearn Pipeline to apply transformations to the raw vectors
        - `fit_pipeline`: Whether to fit the pipeline on the data (True) or just transform (False)
        """
        keys, raw_vecs = self.keys_vecs(keys)

        if pipeline is None or not keys:
            return keys, raw_vecs

        if fit_pipeline:
            transformed = pipeline.fit_transform(raw_vecs)
        else:
            transformed = pipeline.transform(raw_vecs)

        return keys, transformed


def create_basic_pipeline() -> Pipeline:
    """Create a standard normalization + scaling pipeline."""
    return FeaturePipelineBuilder().normalize().scale().build()


def create_rbf_pipeline(n_components: int=1000, gamma: float=0.1) -> Pipeline:
    """Create a pipeline with RBF sampling for kernel approximation.

    - `n_components`: Number of RBF components to sample
    - `gamma`: Parameter of the RBF kernel
    """
    return (FeaturePipelineBuilder()
           .normalize()
           .scale()
           .rbf_sample(n_components=n_components, gamma=gamma)
           .build())


def create_dimensionality_reduction_pipeline(final_dims: int=512) -> Pipeline:
    """Create a pipeline that reduces dimensionality via PCA.

    - `final_dims`: Final number of dimensions after PCA
    """
    return (FeaturePipelineBuilder()
           .normalize()
           .scale()
           .pca(n_components=final_dims)
           .build())
