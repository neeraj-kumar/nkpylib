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

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nkpylib.ml.ml_types import nparray1d, nparray2d, array1d, array2d
from nkpylib.ml.nklmdb import PickleableLmdb, JsonLmdb, MetadataLmdb, NumpyLmdb

logger = logging.getLogger(__name__)

KeyT = TypeVar('KeyT')

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
        self.cached: dict[str, Any] = dict()
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
        logger.info(f'Reloading keys for FeatureSet with {len(self.inputs)} inputs: {self.inputs}')
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
        return np.hstack([inp[key] for inp in self.inputs])

    def get_keys_embeddings(self,
                            keys: list[KeyT]|None=None,
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True,
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

        The keys and embeddings are cached for future calls with the same flags (only if requesting
        all keys).

        If you set `return_scaler` to True, we also return the scaler object used for scaling as the
        last item in the return tuple.
        """
        if keys is None:
            if 0: #TODO caching temporarily disabled
                cache_kw = dict(normed=normed, scale_mean=scale_mean, scale_std=scale_std)
                if self.cached and all(self.cached[k] == v for k, v in cache_kw.items()):
                    return self.cached['keys'], self.cached['embs']
            _keys, _embs = zip(*list(self.items()))
            keys = list(_keys)
            embs = np.vstack(_embs)
        else:
            keys = [k for k in keys if k in self]
            embs = np.vstack([self[k] for k in keys])
        scaler: StandardScaler|None = None
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        if scale_mean or scale_std:
            scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
            embs = scaler.fit_transform(embs)
        if 0 and len(keys) == len(self): # cache these
            self.cached.update(keys=keys, embs=embs, scaler=scaler, **cache_kw)
        if return_scaler:
            return keys, embs, scaler
        else:
            return keys, embs
