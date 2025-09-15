"""Embeddings and utilities.

This module provides tools for working with high-dimensional embeddings, particularly
for machine learning applications. Key functionality includes:

Storage:
- JsonLmdb: LMDB database for JSON-serialized data
- MetadataLmdb: LMDB with additional metadata support
- NumpyLmdb: Efficient storage of numpy arrays in LMDB

Feature Management:
- FeatureSet: Collection of features with similarity search capabilities
- Support for multiple input sources (files or mapping objects)
- Automatic key intersection across sources

Operations:
- Clustering with multiple algorithms (k-means, agglomerative, affinity propagation)
- Nearest neighbor search with customizable metrics
- Classification-based similarity search
- Feature extraction from images

Ways to validate embeddings:

Single embedding, no labels
- Clustering
  - cluster size distribution
  - are there tight clusters and well-separated?
  - Can use silhouette score, Davies-Bouldin index, etc.
- knn-graph
  - See what are selected as neighbors
  - assess connectivity and local density — good embeddings will have more meaningful local neighborhoods
- visualize
  - t-SNE, UMAP, Isomap, PCA, etc. Overlaps, clusters, "cloudiness" tells a lot
- By dimension:
  - min-spectrum-max
  - And after PCA?
  - mean/std, min, max values
  - histogram of values
    - particularly 0s
  - PCA can reveal how much variance each dimension explains
  - VAE is good non-linear way to disentangle dimensions (particularly Beta-VAE)
- Compute pairwise Pearson/Spearman correlations between dimensions. Strong correlations suggest redundant or entangled dimensions.
- How compressible are the embeddings?
  - PCA, autoencoder, quantization, Kernel PCA?, VAE
  - estimate "intrinsic dimensionality" of the embeddings via MLE or TwoNN
- Look at distances from examples
  - e.g., how many neighbors are within 0.1, 0.2, etc. distance?
  - How similar are different examples' distances
  - Aggregate distances across all examples into histograms
- Check various stats:
  - Mean/std
  - Norm
- Compare cosine vs euclidean distances
  - How correlated
  - Pairs with big differences between two metrics
- Look at local neighborhoods
  - Compare geodesic (manifold) distances in high-dimensional space vs. Euclidean distances — e.g., via Isomap. Indicates whether embeddings preserve underlying structure
  - Use Locally Linear Embedding (LLE) to test whether local patches of embedding space behave linearly — good embeddings often have locally linear structure.
- Detect outliers
  - Run unsupervised outlier detection algorithms (e.g., Isolation Forest, LOF) on the embedding space — helpful to spot anomalies or collapsed modes.
- Visualize full pairwise cosine similarity heatmaps — useful for spotting large dense cliques (bad) or disconnected islands (good/bad, depending).
- Compute pairwise angles between random vector pairs. For high-quality, high-dimensional embeddings, the distribution should be tightly centered.
- 

Multiple embeddings, no labels
- compare clusters
- compare knn-graphs
  - i.e., common neighbors, reciprocal ranks
- embedding similarity
  - use procrustes or CCA
- assess drift over time or other meaningful slices of the data between embeddings

Single embedding, with labels
- Clustering purity on metadata - compute purity/ARI/NMI
- For items with labels, compute proportion of neighbors with the same label
- Does label propagation work?
- Train simple classifier/regressor on attributes
  - Helpful even if attributes were used to train embeddings - how well does it learn the attr?
  - Look at PR-curves, AUC, ROC
- Recommendation system
- Few-shot classifier
- Pairwise/triplet-loss task eval
- Analogies, e.g., "comedy - dark + romance" ≈ "romcom"
- Sequence modeling using embeddings as input
- Calibration of similarity scores
  - Compare histograms or CDFs of same-class vs diff-class scores
- Look at NNs of examples with labels - closest, spectrum, farthest
- Simulate synthetic user with known prefs and test how well NN align with prefs?
- Sort by label and plot value of each dimension of embedding
- Sort by each dim of embedding and plot label value
- See which dims correlate strongest with labels
  - E.g. dim 45 = horror movies

Misc
- Can you get text or images out of the embeddings?
  - Or map from embeddings to text/images?

"""

#TODO sparse
#TODO in-memory

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time

from argparse import ArgumentParser
from collections.abc import Mapping
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os.path import abspath, dirname, exists, join
from typing import Any, cast, Sequence, Generic, TypeVar, Iterator

import numpy as np

from lmdbm import Lmdb
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.linear_model import SGDClassifier # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.thread_utils import CollectionUpdater

nparray1d = np.ndarray
nparray2d = np.ndarray

array1d = nparray1d | Sequence[float]
array2d = nparray2d | Sequence[Sequence[float]]

logger = logging.getLogger(__name__)

class JsonLmdb(Lmdb):
    """Keys are utf-8 encoded strings, values are JSON-encoded objects as utf-8 encoded strings."""
    def _pre_key(self, key):
        return key.encode("utf-8")

    def _post_key(self, key):
        return key.decode("utf-8")

    def _pre_value(self, value):
        return json.dumps(value).encode("utf-8")

    def _post_value(self, value):
        return json.loads(value.decode("utf-8"))


class MetadataLmdb(Lmdb):
    """Subclass of LMDB database that stores JSON metadata for each key inside a 2nd database.

    In the main database, the keys are utf-8 encoded strings, and the values are arbitrary objects.
    (Override _pre_value and _post_value as needed in your subclass.)

    We add a few new methods, all prefixed with md_ (for metadata):
    - md_get(key): Get metadata for given key.
    - md_set(key, **kw): Set (replace) metadata for given key.
    - md_update(key, **kw): Update metadata for given key, keeping existing values.
    - md_batch_set(dict): Set multiple metadata entries at once (replacing existing values).
    - md_delete(key): Delete metadata for given key.
    - md_iter(): Iterate over all metadata keys and values.
    - md_iter_all(): Iterate over (key, value, metadata) triplets

    You can also access the metadata database directly via the `md_db` property, e.g. to check for
    existence of certain keys, etc.

    There's also a property `global_key` which is a special key name used for global metadata.

    Note that the metadata is not indexed, i.e. there's no way to search for keys based on metadata.
    """
    path: str
    md_path: str
    md_db: JsonLmdb

    @classmethod
    def open(cls, file: str, flag: str='r', **kw) -> MetadataLmdb: # type: ignore[override]
        """Opens the main LMDB database at given `file` path.

        Opens a 2nd metadata-only LMDB database inside the first one, with name 'metadata.lmdb',
        with the same `flag` and `kw` as the main db.

        The flag is one of:
        - 'r': read-only, existing
        - 'w': read and write, existing
        - 'c': read and write, create if not exists
        - 'n': read and write, overwrite

        By default, we set `map_size` to 2 ** 25, which is 32 MiB. LMDB only grows up to 12 factors,
        which would be 2 ** 25 * 2 ** 12 = 16 TiB, so this is a reasonable default.

        Note that as the original code says, keeping autogrow=True (the default) means that there
        could be problems with multiple writers.
        """
        if 'map_size' not in kw:
            kw['map_size'] = 2 ** 25 # lmdbm only grows up to 12 factors, and defaults to 2e20
        # make dirs if needed
        if flag != 'r':
            try:
                os.makedirs(dirname(file), exist_ok=True)
            except Exception:
                pass
        ret = cast(MetadataLmdb, super().open(file, flag, **kw))
        ret.path = file
        # now the metadata db
        ret.md_path = join(file, 'metadata.lmdb')
        ret.md_db = cast(JsonLmdb, JsonLmdb.open(ret.md_path, flag=flag, **kw))
        return ret

    def _pre_key(self, key: str) -> bytes:
        return key.encode('utf-8', 'ignore')

    def _post_key(self, key: bytes) -> str:
        return key.decode('utf-8', 'ignore')

    @property
    def global_key(self) -> str:
        """Special key name used for global metadata."""
        return '__global__'

    def md_get(self, key: str) -> dict:
        """Get metadata for given key, or empty dict if not present."""
        try:
            return self.md_db[key]
        except KeyError:
            return {}

    def md_set(self, key: str, **kw) -> None:
        """Set (replace) metadata for given key."""
        self.md_db[key] = kw

    def md_update(self, key: str, **kw) -> None:
        """Update metadata for given key, keeping existing values."""
        md = self.md_get(key)
        md.update(kw)
        self.md_db[key] = md

    def md_batch_set(self, data: dict[str, dict]) -> None:
        """Set multiple metadata entries at once (replacing existing values)."""
        self.md_db.update(data)

    def md_delete(self, key: str) -> None:
        """Delete metadata for given key."""
        try:
            del self.md_db[key]
        except KeyError:
            pass

    def md_iter(self) -> Iterator[tuple[str, dict]]:
        """Iterate over all metadata keys and values."""
        return self.md_db.items()

    def md_iter_all(self) -> Iterator[tuple[str, Any, dict]]:
        """Iterate over (key, value, metadata) triplets.

        Note that this only iterates over keys in the main database.
        If you set metadata on other keys (or the global one), this doesn't include them.
        """
        for key in self:
            yield key, self[key], self.md_get(key)

    def sync(self) -> None:
        """Sync both the main database and the metadata database."""
        super().sync()
        self.md_db.sync()

    def close(self) -> None:
        """Close both the main database and the metadata database."""
        super().close()
        self.md_db.close()

    def __repr__(self):
        return f'MetadataLmdb<{self.path}>'


class NumpyLmdb(MetadataLmdb):
    """Subclass of MetadataLmdb database that stores numpy arrays with utf-8 encoded string keys."""
    dtype: Any
    path: str

    @classmethod
    def open(cls, file: str, flag: str='r', dtype=np.float32, **kw) -> NumpyLmdb: # type: ignore[override]
        """Opens the LMDB database at given `file` path.

        The flag is one of:
        - 'r': read-only, existing
        - 'w': read and write, existing
        - 'c': read and write, create if not exists
        - 'n': read and write, overwrite

        We enforce that all np array values will be of `dtype` type.
        """
        ret = cast(NumpyLmdb, super().open(file, flag, **kw))
        ret.dtype = dtype
        ret.path = file
        return ret

    def _pre_value(self, value: np.ndarray) -> bytes:
        value = np.array(value, dtype=self.dtype)
        assert isinstance(value, np.ndarray), f'Value must be a numpy array, not {type(value)}'
        assert value.dtype == self.dtype, f'Value must be of type {self.dtype}, not {value.dtype}'
        #print(f'set: value of type {type(value)} with shape {value.shape} and dtype {value.dtype}: {value}')
        return value.tobytes()

    def _post_value(self, value: bytes) -> np.ndarray:
        a = np.frombuffer(value, dtype=self.dtype)
        #print(f'get: value of type {type(a)} with shape {a.shape} and dtype {a.dtype}: {a}')
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
            cur = NumpyLmdb.open(path, flag='r', dtype=dtype)
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

class LmdbUpdater(CollectionUpdater):
    """Helper class to update an LMDB database in parallel.

    This is a subclass of CollectionUpdater that takes an LMDB path as input.

    Use like this:

        updater = LmdbUpdater(db_path, NumpyLmdb.open, n_procs=4, batch_size=1000)
        updater.add('id1', embedding=np.array([...]))
        updater.add('id2', metadata=dict(key='value'))
        updater.add('id3', embedding=np.array([...]), metadata=dict(key='value'))

    """
    def __init__(self,
                 db_path: str,
                 init_fn: Callable,
                 map_size: int=2e32,
                 n_procs: int=4,
                 **kw):
        """Initialize the updater with the given db_path and `init_fn` (to pick which flavor)

        If you specify `debug=True`, then commit messages will be printed using logger.info()

        All other `kw` are passed to the CollectionUpdater constructor.
        """
        self.pool = ProcessPoolExecutor(max_workers=n_procs, initializer=self.init_worker, initargs=(db_path, init_fn, map_size))
        super().__init__(add_fn=self._add_fn, **kw)
        self.futures = []

    @staticmethod
    def init_worker(db_path: str, init_fn: Callable, map_size: int=2**32) -> None:
        """Initializes a worker process with the given db_path and init_fn."""
        global db, my_id
        my_id = os.getpid()
        print(f'In child {my_id}: initializing db at {db_path} with {init_fn} and map_size {map_size}', flush=True)
        db = init_fn(db_path, flag='c', map_size=map_size, autogrow=False)

    def add(self, id: str, embedding=None, metadata=None):
        """Add an item with given `id`, `embedding` (np array) and/or `metadata` (dict)."""
        obj: dict[str, Any] = {}
        if embedding is not None:
            obj['embedding'] = embedding
        if metadata is not None:
            obj['metadata'] = metadata
        #print(f'Added id {id} with obj {obj}')
        super().add(id, obj)

    @staticmethod
    def set_emb(id, emb):
        db.__setitem__(id, np.array(emb, dtype=db.dtype))

    @staticmethod
    def set_md(id, md):
        db.md_set(id, **md)

    @staticmethod
    def db_sync():
        db.sync()
        return my_id

    @staticmethod
    def batch_add(ids, objects):
        md_updates = {}
        emb_updates = {}
        for id, obj in zip(ids, objects):
            md = obj.get('metadata', None)
            if md is not None:
                md_updates[id] = md
            emb = obj.get('embedding', None)
            if emb is not None:
                emb_updates[id] = np.array(emb, dtype=db.dtype)
        if md_updates:
            db.md_batch_set(md_updates)
        if emb_updates:
            db.update(emb_updates)
        db.sync()

    def _add_fn(self, to_add: dict[str, list]) -> None:
        future = self.pool.submit(self.batch_add, to_add['ids'], to_add['objects'])
        self.futures.append(future)

    def sync(self):
        """Makes sure all pending writes are done."""
        print(f'Calling sync')
        for f in self.futures:
            f.result()
        self.futures = []


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
            inp if is_mapping(inp) else NumpyLmdb.open(inp, flag='r', dtype=dtype)
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

def quick_test(path: str, **kw):
    """Prints the first embeddings from given `path`"""
    db = NumpyLmdb.open(path)
    print(f'Opened {db}, {len(db)} items, {db.dtype} dtype, {db.map_size} map size.')
    num = 0
    for key, value in db.items():
        print(f'Key: {key}, Value: {value}')
        num += 1
        if num > 2:
            break

def image_extract(path: str, flag: str='c', model='jina', batch_size=20, **kw):
    """Does image feature extraction with given `model`.

    Reads filenames from stdin and writes the embeddings to the given `path`.
    """
    from tqdm import tqdm
    from nkpylib.ml.client import chunked, embed_image
    db = NumpyLmdb.open(path, flag=flag)
    paths = []
    for line in sys.stdin:
        path = line.strip()
        if path in db:
            continue
        paths.append(path)
    print(f'Got {len(db)} keys in db {db.path} with dtype {db.dtype} and flag {flag}, {len(paths)} new images to process.')
    num = 0
    # create a progress bar we can manually move
    bar = tqdm(total=len(paths), desc='Extracting features from images', unit='image')
    # interleave reading files from stdin and processing them, in batches
    for batch in chunked(paths, batch_size):
        futures = embed_image.batch_futures([abspath(path) for path in batch], model=model, use_cache=False)
        for path, future in zip(batch, futures):
            try:
                emb = future.result()
                if emb is None:
                    logger.warning(f'No embedding for {path}')
                    continue
                db[path] = np.array(emb, dtype=db.dtype)
                num += 1
                bar.update(1)
            except Exception as e:
                logger.error(f'Error extracting {path}: {e}', exc_info=True)
        db.sync()
    db.sync()
    print(f'Wrote {num} items to {db.path}.')


if __name__ == '__main__':
    funcs = {f.__name__: f for f in [quick_test, image_extract]}
    parser = ArgumentParser(description='Test embeddings utilities.')
    parser.add_argument('func', choices=funcs, help='Function to run')
    parser.add_argument('path', help='Path to the embeddings lmdb file')
    parser.add_argument('-f', '--flag', default='r', choices=['r', 'w', 'c', 'n'],
                        help='Flag to open the lmdb file (default: r)')
    parser.add_argument('keyvalue', nargs='*', help='Key=value pairs to pass to the function')
    args = parser.parse_args()
    kwargs = vars(args)
    for keyvalue in kwargs.pop('keyvalue', []):
        if '=' not in keyvalue:
            raise ValueError(f'Invalid key=value pair: {keyvalue}')
        key, value = keyvalue.split('=', 1)
        value = specialize(value)
        kwargs[key] = value
    func = funcs[kwargs.pop('func')]
    func(**kwargs) # type: ignore[operator]
