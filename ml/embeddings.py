"""Embeddings and utilities.

This module provides tools for working with high-dimensional embeddings, particularly
for machine learning applications. Key functionality includes:

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
from os.path import abspath, dirname, exists, join
from typing import Any, Sequence, Generic, TypeVar

import numpy as np

from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.linear_model import SGDClassifier # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    FeatureSet,
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)

logger = logging.getLogger(__name__)

KeyT = TypeVar('KeyT')

class FeatureSetOperations(FeatureSet, Generic[KeyT]):
    """A set of features that you can do stuff with."""
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
    funcs = {f.__name__: f for f in []}
    parser = ArgumentParser(description='Test embeddings')
    parser.add_argument('func', choices=funcs, help='Function to run')
    parser.add_argument('path', help='Path to the embeddings lmdb file')
    parser.add_argument('-f', '--flag', default='r', choices=['r', 'w', 'c', 'n'],
                        help='Flag to open the lmdb file (default: r)')
    parser.add_argument('-t', '--tag_path', default='', help='Path to the tags sqlite')
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
