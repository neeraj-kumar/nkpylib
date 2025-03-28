"""Utilities to deal with embeddings"""

from __future__ import annotations

import logging
import random

from collections.abc import Mapping
from collections import Counter
from typing import Any, Sequence

import numpy as np

from lmdbm import Lmdb
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

nparray1d = np.ndarray
nparray2d = np.ndarray

array1d = nparray1d | Sequence[float]
array2d = nparray2d | Sequence[Sequence[float]]

logger = logging.getLogger(__name__)

class Embeddings(Mapping):
    """Wrapper class around embeddings.

    This functions as a read-only Mapping.
    """
    def __init__(self, path: str, mode: str='r'):
        """Loads embeddings from given `path`.

        This must be LMDB format for now.
        """
        self.path = path
        self.db = Lmdb.open(path, mode)
        # we cache the order of our keys
        self._keys = [k.decode('utf-8') for k in self.db.keys()]
        self.n_dims = 0
        for key, value in self.items():
            self.n_dims = len(value)
            break
        self.cached: dict[str, Any] = dict()

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, key: str) -> np.ndarray:
        a = np.frombuffer(self.db[key.encode('utf-8')], dtype=np.float32)
        return a

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key.encode('utf-8') in self.db
        elif isinstance(key, bytes):
            return key in self.db
        else:
            raise TypeError(f'Key must be str or bytes, not {type(key)}')

    def __repr__(self):
        return f'Embeddings({self.path!r})'

    def __str__(self):
        return f'Embeddings({self.path!r})'

    def get_keys_embeddings(self,
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True) -> tuple[list[str], np.ndarray]:
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
        _keys, _embs = zip(*list(self.items()))
        keys = list(_keys)
        embs = np.vstack(_embs)
        scaler: StandardScaler|None = None
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        if scale_mean or scale_std:
            scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
            embs = scaler.fit_transform(embs)
        # cache these
        self.cached.update(keys=keys, embs=embs, scaler=scaler, **cache_kw)
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
            clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        elif method in ('agg', 'average'):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        elif method == 'affinity':
            clusterer = AffinityPropagation()
        else:
            raise NotImplementedError(f'Clustering method {method!r} not implemented.')
        labels = clusterer.fit_predict(embs)
        uniques = set(labels)
        clusters: dict[int, list[str]] = {i: [] for i in uniques}
        for key, label in zip(keys, labels):
            clusters[label].append(key)
        return sorted(clusters.values(), key=len, reverse=True)

    def similar(self,
                queries: list[str]|array2d,
                weights: list[float]|None=None,
                n_neg: int=1000,
                method: str='rbf',
                min_score: float=-0.1,
                **kw) -> list[tuple[float, str]]:
        """Returns the most similar keys and scores to the given `queries`.

        This is a wrapper on top of `nearest_neighbors()` and `make_classifier()`.

        The queries can either be keys from this class, or embedding vectors.

        Returns (score, key) tuples in descending order of score.
        """
        assert len(queries) > 0, 'Must provide at least one query.'
        assert len({type(q) for q in queries}) == 1, 'All queries must be of the same type.'
        if isinstance(queries[0], str):
            assert all(q in self for q in queries), f'All queries must be in the embeddings.'
        #TODO normalize queries if not in dataset
        keys, embs = self.get_keys_embeddings(normed=True, scale_mean=False, scale_std=True)
        pos: Any
        if method == 'nn':
            if isinstance(queries[0], str):
                _pos = np.array([i for i, k in enumerate(keys) if k in queries])
                pos = embs[_pos]
            else:
                pos = queries
            _ret = self.nearest_neighbors(pos, n_neighbors=n_neg, **kw)
        else:
            # train a classifier with these as positive and some randomly chosen as negative
            if isinstance(queries[0], str):
                pos = [i for i, k in enumerate(keys) if k in queries]
                neg = [i for i in range(len(keys)) if i not in pos]
                neg = random.sample(neg, n_neg)
                X = embs[pos + neg]
            else:
                pos = queries
                neg = random.sample(range(len(embs)), n_neg)
                X = np.vstack([queries, embs[neg]])
            y = [1]*len(pos) + [-1]*len(neg)
            clf = self.make_classifier(X, y, method=method, **kw)
            scores = clf.decision_function(embs)
            logger.debug(f'Got scores {scores.shape}: {scores}')
            _ret = [(s, k) for s, k in zip(scores, keys) if s > min_score]
            logger.debug(f'got _ret: {len(_ret)}: {_ret[:10]}')
        # sort results by score (desc) and filter out queries
        ret = sorted([(float(s), k) for s, k in _ret if k not in queries], reverse=True)
        return ret

    def nearest_neighbors(self, pos: array2d, n_neighbors:int=1000, metric='cosine', **kw):
        """Runs nearest neighbors with given `pos` embeddings, aggregating scores."""
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        keys, embs = self.get_keys_embeddings(normed=True, scale_mean=False, scale_std=True)
        nn.fit(embs)
        scores, indices = nn.kneighbors(pos, n_neighbors, return_distance=True)
        # aggregate scores for each index over all queries
        score_by_index: Counter = Counter()
        for i, s in zip(indices, scores):
            for j, k in zip(i, s): # for each query, add the score to the index
                score_by_index[j] += k
        ret = [(score, keys[idx]) for idx, score in score_by_index.most_common()]
        return ret

    def make_classifier(self,
                        X: nparray2d,
                        y: Sequence[float|int],
                        weights: Sequence[float]|None=None,
                        method: str='rbf',
                        C=10,
                        class_weight='balanced',
                        **kw) -> Any:
        """Makes a classifier with given `method`, trains it on X, y, and returns it.

        If `weights` is provided, it should be of the same length as `keys` and is a weight for key.
        These can be negative as well.

        """
        assert len(X) == len(y), f'Length of X {len(X)} must match y {len(y)}'
        if weights is not None:
            assert len(X) == len(weights), f'Length of weights {len(weights)} must match X {len(X)}'
        logger.debug(f'training labels {Counter(y).most_common()}, X: {X.shape}, {X}')
        clf_kw = dict(class_weight=class_weight, **kw)
        if method == 'rbf':
            clf = SVC(kernel='rbf', C=C, **clf_kw)
        elif method == 'linear':
            clf = SVC(kernel='linear', C=C, **clf_kw)
        elif method == 'sgd':
            clf = SGDClassifier(**clf_kw)
        else:
            raise NotImplementedError(f'Classifier method {method!r} not implemented.')
        clf.fit(X, y, sample_weight=weights)
        return clf
