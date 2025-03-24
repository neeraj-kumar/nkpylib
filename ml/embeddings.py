"""Utilities to deal with embeddings"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lmdbm import Lmdb
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering
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
        # we cache the order of our keys
        self._keys = [k.decode('utf-8') for k in self.db.keys()]
        self.n_dims = 0
        for key, value in self.items():
            self.n_dims = len(value)
            break
        self.cached = dict()

    def __iter__(self):
        return iter(self._keys)

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
        """Returns a list of string keys and a numpy array of embeddings.

        You can optionally set the following flags:
        - `normed`: Normalize embeddings to unit length.
        - `scale_mean`: Scale embeddings to have zero mean.
        - `scale_std`: Scale embeddings to have unit variance.

        The keys and embeddings are cached for future calls with the same flags.
        """
        cache_kw = dict(normed=normed, scale_mean=scale_mean, scale_std=scale_std)
        if self.cached and all(self.cached[k] == v for k, v in cache_kw.items()):
            return self.cached['keys'], self.cached['embs']
        keys, embs = zip(*list(self.items()))
        keys = list(keys)
        embs = np.vstack(embs)
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        if scale_mean or scale_std:
            scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
            embs = scaler.fit_transform(embs)
        # cache these
        self.cached.update(keys=keys, embs=embs, **cache_kw)
        return keys, embs

    def cluster(self, n_clusters=-1, method='kmeans', **kwargs) -> list[list[str]]:
        """Clusters our embeddings.

        If `n_clusters` is not positive (default), we set it to the sqrt of the number of
        embeddings we have.

        Returns a list of lists of keys, where each list is a cluster; in order from largest to smallest.
        """
        keys, embs = self.get_keys_embeddings(normed=False, scale_mean=True, scale_std=True)
        if n_clusters <= 0:
            n_clusters = int(np.sqrt(len(keys)))
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters)
        elif method in ('agg', 'average'):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        elif method == 'affinity':
            clusterer = AffinityPropagation()
        else:
            raise NotImplementedError(f'Clustering method {method!r} not implemented.')
        labels = clusterer.fit_predict(embs)
        uniques = set(labels)
        clusters = {i: [] for i in uniques}
        for key, label in zip(keys, labels):
            clusters[label].append(key)
        return sorted(clusters.values(), key=len, reverse=True)
