"""Utilities to deal with embeddings"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lmdbm import Lmdb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

2darray = np.ndarray

class Embeddings(Mapping):
    """Wrapper class around embeddings.

    This functions as a read-only Mapping.
    """
    def __init__(self, path: str, mode: str='r'):
        """Loads embeddings from given `path`.

        This must be LMDB format for now.
        """
        self.path = path
        self.db = Lmdb.load(path, mode)

    def __iter__(self):
        for key in self.db:
            yield key.decode('utf-8')

    def __len__(self):
        return len(self.db)

    def __getitem__(self, key: str) -> np.ndarray:
        a = np.frombuffer(self.db[key.encode('utf-8')], dtype=np.float32)
        return a

    def __contains__(self, key: str) -> bool:
        return key.encode('utf-8') in self.db

    def __repr__(self):
        return f'Embeddings({self.path!r})'

    def __str__(self):
        return f'Embeddings({self.path!r})'

    def __del__(self):
        self.db.close()

    def get_keys_embeddings(self,
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True) -> tuple[list[str], 2darray]:
        """Returns a list of string keys and a numpy array of embeddings."""
        keys, embs = zip(*list(self.items()))
        embs = np.vstack(embs)
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        if scale_mean or scale_std:
            scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
            embs = scaler.fit_transform(embs)
        return keys, embs
