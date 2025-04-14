"""Utilities to deal with embeddings"""

#TODO overall metadata
#TODO metadata for each embedding
#TODO sparse
#TODO in-memory

from __future__ import annotations

import logging
import random

from collections.abc import Mapping
from collections import Counter
from typing import Any, cast, Sequence, Generic, TypeVar, Union, Iterator

import numpy as np

from lmdbm import Lmdb
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.linear_model import SGDClassifier # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore
from tqdm import tqdm

nparray1d = np.ndarray
nparray2d = np.ndarray

array1d = nparray1d | Sequence[float]
array2d = nparray2d | Sequence[Sequence[float]]

logger = logging.getLogger(__name__)


class NumpyLmdb(Lmdb):
    """Subclass of LMDB database that stores numpy arrays with utf-8 encoded string keys.
    """
    dtype: Any
    path: str

    @classmethod
    def open(cls, file: str, mode: str='r', dtype=np.float32, **kw) -> NumpyLmdb: # type: ignore[override]
        """Opens the LMDB database at given `file` path.

        The mode is one of:
        - 'r': read-only, existing
        - 'w': read and write, existing
        - 'c': read and write, create if not exists
        - 'n': read and write, overwrite

        We enforce that all np array values will be of `dtype` type.
        """
        if 'map_size' not in kw:
            kw['map_size'] = 2 ** 25 # lmdbm only grows up to 12 factors, and defaults to 2e20
        ret = cast(NumpyLmdb, super().open(file, mode, **kw))
        ret.dtype = dtype
        ret.path = file
        return ret

    def _pre_key(self, key: str) -> bytes:
        return key.encode('utf-8', 'ignore')

    def _post_key(self, key: bytes) -> str:
        return key.decode('utf-8', 'ignore')

    def _pre_value(self, value: np.ndarray) -> bytes:
        assert isinstance(value, np.ndarray), f'Value must be a numpy array, not {type(value)}'
        assert value.dtype == self.dtype, f'Value must be of type {self.dtype}, not {value.dtype}'
        return value.tobytes()

    def _post_value(self, value: bytes) -> np.ndarray:
        a = np.frombuffer(value, dtype=self.dtype)
        return a

    def __repr__(self):
        return f'NumpyLmdb<{self.path}>'


    @classmethod
    def concat_multiple(cls, paths: list[str], output_path: str, dtype=np.float32) -> None:
        """Loads ands concatenates multiple lmdbs from given `paths`, writing to `output_path`.

        This writes a single lmdb with only those keys that are in all files.
        """
        vecs = {}
        for i, path in tqdm(enumerate(paths)):
            cur = NumpyLmdb.open(path, mode='r', dtype=dtype)
            if i == 0:
                vecs = dict(cur.items())
            else:
                cur_keys = set(cur.keys())
                # remove keys that are not in all veceddings
                to_del = set(vecs.keys()) - cur_keys
                for k in to_del:
                    del vecs[k]
                # now concatenate
                for k, existing in vecs.items():
                    vecs[k] = np.hstack([existing, cur[k]])
        # write to output path
        with cls.open(output_path, 'c', dtype=dtype) as db:
            db.update({key: vec for key, vec in vecs.items()})

KeyT = TypeVar('KeyT')

def is_mapping(obj):
    """Returns True if the given `obj` is a mapping (dict-like).

    This checks for various methods, including __getitem__, __iter__, and __len__, keys(), items(),
    values(), etc.
    """
    to_check = ['__getitem__', '__iter__', '__len__', 'keys', 'items', 'values']
    for method in to_check:
        if not hasattr(obj, method):
            return False
    return True


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
        # remap any path inputs to NumpyLmdb objects
        self.inputs = [
            inp if is_mapping(inp) else NumpyLmdb.open(inp, mode='r', dtype=dtype)
            for inp in inputs
        ]
        self._keys = self.get_keys()
        self.n_dims = 0
        for key, value in self.items():
            self.n_dims = len(value)
            break
        self.cached: dict[str, Any] = dict()

    def get_keys(self) -> list[KeyT]:
        """Gets the intersection of all keys by reading all our inputs again.

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
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True) -> tuple[list[KeyT], np.ndarray]:
        """Returns a list of keys and a numpy array of embeddings.

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

    def cluster(self, n_clusters=-1, method='kmeans', **kwargs) -> list[list[KeyT]]:
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
        clusters: dict[int, list[KeyT]] = {i: [] for i in uniques}
        for key, label in zip(keys, labels):
            clusters[label].append(key)
        return sorted(clusters.values(), key=len, reverse=True)

    def similar(self,
                queries: list[KeyT]|array2d|BaseEstimator,
                weights: list[float]|None=None,
                n_neg: int=1000,
                method: str='rbf',
                min_score: float=-0.1,
                **kw) -> list[tuple[float, KeyT]]:
        """Returns the most similar keys and scores to the given `queries`.

        This is a wrapper on top of `nearest_neighbors()` and `make_classifier()`.

        The queries can either be keys from this class, or embedding vectors.

        Returns (score, key) tuples in descending order of score.
        """
        if not isinstance(queries, BaseEstimator):
            assert len(queries) > 0, 'Must provide at least one query.'
            assert len({type(q) for q in queries}) == 1, 'All queries must be of the same type.'
            if queries[0] in self:
                assert all(q in self for q in queries), f'All queries must be in the embeddings.'
        #TODO normalize queries if not in dataset
        keys, embs = self.get_keys_embeddings(normed=True, scale_mean=False, scale_std=True)
        pos: Any
        if method == 'nn': # queries must not be estimator
            if queries[0] in self:
                _pos = np.array([i for i, k in enumerate(keys) if k in queries])
                pos = embs[_pos]
            else:
                pos = queries
            _ret = self.nearest_neighbors(pos, n_neighbors=n_neg, **kw)
        else:
            if isinstance(queries, BaseEstimator):
                clf = queries
            else:
                # train a classifier with these as positive and some randomly chosen as negative
                if queries[0] in self:
                    pos = [i for i, k in enumerate(keys) if k in queries]
                    neg = [i for i in range(len(keys)) if i not in pos]
                    neg = random.sample(neg, n_neg)
                    X = embs[pos + neg]
                else:
                    # at this point, we know queries is a 2d array
                    pos = queries
                    neg = random.sample(range(len(embs)), n_neg)
                    X = np.vstack([queries, embs[neg]]) # type: ignore[list-item]
                y = [1]*len(pos) + [-1]*len(neg)
                clf = self.make_classifier(X, y, method=method, **kw)
            scores = clf.decision_function(embs)
            logger.debug(f'Got scores {scores.shape}: {scores}')
            _ret = [(s, k) for s, k in zip(scores, keys) if s > min_score]
            logger.debug(f'got _ret: {len(_ret)}: {_ret[:10]}')
        # sort results by score (desc) and filter out queries (if applicable)
        if isinstance(queries, BaseEstimator):
            ret = sorted([(float(s), k) for s, k in _ret], reverse=True)
        else:
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


if __name__ == '__main__':
    import sys, time
    db = NumpyLmdb.open(sys.argv[1])
    print(f'Opened {db}, {len(db)} items, {db.dtype} dtype, {db.map_size} map size.')
    for key, value in db.items():
        print(f'Key: {key}, Value: {value}')
        break
