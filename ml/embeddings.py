"""Embeddings and utilities.

This module provides tools for working with high-dimensional embeddings, particularly
for machine learning applications. Key functionality includes:

Operations:
- Clustering with multiple algorithms (k-means, agglomerative, affinity propagation)
- Nearest neighbor search with customizable metrics
- Classification-based similarity search
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
from typing import Any, Sequence, Generic, TypeVar, Hashable

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.decomposition import TruncatedSVD # type: ignore
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
)
from nkpylib.ml.nklmdb import (
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)

logger = logging.getLogger(__name__)

KeyT = TypeVar('KeyT')

class Embeddings(FeatureSet, Generic[KeyT]):
    """A set of features that you can do stuff with."""
    def get_clusterer(self, method: str='kmeans', n_clusters: int=-1, **kw) -> Any:
        if method == 'kmeans':
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', **kw)
        elif method in ('agg', 'average'):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', **kw)
        elif method == 'affinity':
            clusterer = AffinityPropagation(**kw)
        else:
            raise NotImplementedError(f'Clustering method {method!r} not implemented.')
        return clusterer

    def guided_clustering(self,
                          labels: dict[KeyT, int],
                          keys: list[KeyT]|None=None,
                          n_clusters=-1,
                          method='kmeans',
                          **kwargs) -> dict[KeyT, dict]:
        """You provide a few cluster assignments, and we fill in the rest.

        Returns a dict of key to {num, score}, where score is how confident we are.
        """
        clusters: dict[KeyT, dict] = {}
        if method == 'random': # randomly assign, purely for testing UI
            for key in keys:
                if key in labels:
                    clusters[key] = dict(num=labels[key], score=1.0)
                else:
                    clusters[key] = dict(num=random.randint(1, n_clusters), score=random.uniform(0, 1))
        else:
            # apply clustering method repeatedly until we have no conflicts with labels
            clusterer = self.get_clusterer(method=method, n_clusters=n_clusters)
            keys_all, embs = self.get_keys_embeddings(keys=keys, normed=False, scale_mean=True, scale_std=True)
            #print(keys_all, embs)
            labels_array = np.array([labels[key] if key in labels else -1 for key in keys_all])
            print(labels_array)
            for _ in range(5):
                # do a clustering
                pred_labels = clusterer.fit_predict(embs)
                # check for conflicts
                conflict = False
                for i, key in enumerate(keys_all):
                    if key in labels and labels[key] != pred_labels[i]:
                        conflict = True
                        break
                if not conflict:
                    break
            # assign scores based on distance to cluster center
            centers = clusterer.cluster_centers_
            for i, key in enumerate(keys_all):
                center = centers[pred_labels[i]]
                dist = np.linalg.norm(embs[i] - center)
                score = 1 / (1 + dist)
                clusters[key] = dict(num=int(pred_labels[i])+1, score=float(score))
        return clusters

    def cluster(self, n_clusters=-1, method='kmeans', **kwargs) -> list[list[KeyT]]:
        """Clusters our embeddings.

        If `n_clusters` is not positive (default), we set it to the sqrt of the number of
        embeddings we have.

        Returns a list of lists of keys, where each list is a cluster; in order from largest to smallest.
        """
        keys, embs = self.get_keys_embeddings(normed=False, scale_mean=True, scale_std=True)
        if n_clusters <= 0:
            n_clusters = int(np.sqrt(len(keys)))
        clusterer = self.get_clusterer(method=method, n_clusters=n_clusters, **kwargs)
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
                all_keys: list[KeyT]|None=None,
                **kw) -> list[tuple[float, KeyT]]:
        """Returns the most similar keys and scores to the given `queries`.

        This is a wrapper on top of `nearest_neighbors()` (method='nn') and `make_classifier()`
        (method='rbf').

        The queries can either be keys from this class, or embedding vectors.

        You can set the "universe" of keys to search over using `all_keys`. By default, we search
        over all keys in this class.

        Returns (score, key) tuples in descending order of score.
        """
        if not isinstance(queries, BaseEstimator):
            assert len(queries) > 0, 'Must provide at least one query.'
            assert len({type(q) for q in queries}) == 1, 'All queries must be of the same type.'
            if queries[0] in self:
                assert all(q in self for q in queries), f'All queries must be in the embeddings.'
        #TODO normalize queries if not in dataset
        keys, embs = self.get_keys_embeddings(keys=all_keys, normed=True, scale_mean=False, scale_std=True)
        pos: Any
        if method == 'nn': # queries must not be estimator
            if queries[0] in self:
                logger.debug(f'{len(queries)} Pos: {queries}')
                _pos = np.array([i for i, k in enumerate(keys) if k in queries])
                pos = embs[_pos]
            else:
                pos = queries
            _ret = self.nearest_neighbors(pos, n_neighbors=n_neg, all_keys=all_keys, **kw)
        else:
            if isinstance(queries, BaseEstimator):
                clf = queries
            else:
                # train a classifier with these as positive and some randomly chosen as negative
                if queries[0] in self:
                    pos = [i for i, k in enumerate(keys) if k in queries]
                    neg = [i for i in range(len(keys)) if i not in pos]
                    neg = random.sample(neg, min(len(neg), n_neg))
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

    def nearest_neighbors(self, pos: array2d, n_neighbors:int=1000, metric='l2', all_keys=None, **kw):
        """Runs nearest neighbors with given `pos` embeddings, aggregating scores."""
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        #keys, embs = self.get_keys_embeddings(keys=all_keys, normed=True, scale_mean=False, scale_std=True)
        keys, embs = self.get_keys_embeddings(keys=all_keys, normed=False, scale_mean=False, scale_std=False)
        logger.debug(f'first keys and embs: {keys[:5]}, {embs[:5]}')
        nn.fit(embs)
        scores, indices = nn.kneighbors(pos, min(n_neighbors, len(keys)), return_distance=True)
        # aggregate scores for each index over all queries
        score_by_index: Counter = Counter()
        for i, s in zip(indices, scores):
            for j, k in zip(i, s): # for each query, take the best score
                cur = 1 - k
                if j not in score_by_index:
                    score_by_index[j] = cur
                score_by_index[j] = max(score_by_index[j], cur)
        ret = [(score, keys[idx]) for idx, score in score_by_index.most_common()]
        return ret

    def simple_nearest_neighbors(self, pos: array2d, n_neighbors:int=1000, metric='cosine', all_keys=None, **kw):
        """Runs nearest neighbors with given `pos` embeddings, aggregating scores.

        This version uses cdist directly.
        """
        keys, embs = self.get_keys_embeddings(keys=all_keys, normed=True, scale_mean=False, scale_std=False)
        logger.debug(f'first keys and embs: {keys[:5]}, {embs[:5]}')
        scores = cdist(pos, embs, metric=metric)
        logger.debug(f'got scores: {scores.shape}: {scores}')
        # aggregate scores for each index over all queries
        score_by_index: Counter = Counter()
        for row in scores:
            for j, s in enumerate(row):
                score_by_index[j] += 1 - s
        logger.debug(f'got score by index: {score_by_index.most_common(10)}')
        ret = [(score, keys[idx]) for idx, score in score_by_index.most_common()]
        logger.debug(f'got final ret: {ret[:10]}')
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

# hashable bound
T = TypeVar('T', bound=Hashable)
def generate_cooccurence_embeddings(
        data: list[list[T]],
        existing: Mapping[T, array1d]|None=None,
        min_variance: float = 0.9,
        shifted_ppmi_k: float = 0.0,
        ) -> tuple[dict[T, array1d], list[float]]:
    """Generates embeddings based on co-occurence.

    Returns a tuple of the embeddings dict {tag: embedding} and the cumulative variances list.

    The idea is that if two items co-occur often, they should be closer in embedding space.

    These are useful for things like tags, etc., that you might otherwise 1-hot encode. Probably
    anything with more than 10 items is worth doing. (We'll refer to the items as tags below, for
    convenience.)

    In practice, there are a few main approaches for generating these:
    1. Count co-occurences and use SVD to reduce dimensionality. This has the advantage that it's
    well understood, principled, fast, globally optimal, and has a direct way of figuring out
    dimensionality. It also works quite stably when updating with new data or even new tags. The
    downsides are that it can be slow if used on very large data (since we recompute it from scratch
    each time) and that in practice, it might not be as good as word2vec style embeddings for
    downstream classification tasks.

    2. Word2vec-style embeddings. These are often faster to train (at scale) and might be higher
    signal for downstream tasks. The downsides are that you have to figure out dimensionality
    yourself, and updates with new tags have to be done very carefully due to randomization/etc.

    This function uses the first approach. We use PPMI to normalize the co-occurence matrix, and
    then use SVD to reduce dimensionality. You can optionally provide the `shifted_ppmi_k parameter`
    (set it to > 0) to use shifted ppmi (with log(k) as the shift factor). We pick the number of
    dimensions based on the amount of variance we want to capture (default 90%).

    The input `data` is a list of lists of tags. Each inner list is a set of tags that co-occurred
    together. Tags can be any hashable type.

    Note that if you had previously computed embeddings with older data (perhaps even with new tags
    this time), the embeddings should still not drift much.

    You can optionally provide `existing` embeddings to update as a dict from tag to embedding. In
    that case, the `data` should be a snapshot of all data (old + new), as we simply recompute
    things from scratch.
    """
    #FIXME how do we get consistent indices per tag?
    tag_to_idx: dict[T, int] = {}
    if existing is not None:
        for tag in existing:
            tag_to_idx[tag] = len(tag_to_idx)
    for tags in data:
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
    n_tags = len(tag_to_idx)
    logger.info(f'Got {n_tags} unique tags from {len(data)} data points.')
    cooccur = np.zeros((n_tags, n_tags), dtype=np.float32)
    if 0: # sequential version
        for tags in tqdm(data, desc='Counting co-occurences'):
            indices = [tag_to_idx[tag] for tag in tags]
            for i in indices:
                for j in indices:
                    if i != j:
                        cooccur[i, j] += 1
    else: # vectorized version
        for tags in tqdm(data, desc='Counting co-occurences'):
            indices = [tag_to_idx[tag] for tag in tags]
            if len(indices) > 1:
                arr = np.array(indices)
                cooccur[np.ix_(arr, arr)] += 1
                np.fill_diagonal(cooccur, 0)
    logger.info(f'Co-occurence matrix has {np.count_nonzero(cooccur)} non-zero entries '
                f'out of {cooccur.size} ({100*np.count_nonzero(cooccur)/cooccur.size:.2f}%)')
    # PPMI
    logger.info('Computing PPMI...')
    if shifted_ppmi_k > 0:
        logger.info(f'Using shifted PPMI with k={shifted_ppmi_k}')
    row_sums = cooccur.sum(axis=1, keepdims=True)
    col_sums = cooccur.sum(axis=0, keepdims=True)
    total = row_sums.sum()
    expected = row_sums @ col_sums / total
    ppmi = np.zeros_like(cooccur)
    nz_i, nz_j = np.nonzero(cooccur)
    for i, j in zip(nz_i, nz_j):
        p_ij = cooccur[i, j] / total
        p_i = row_sums[i, 0] / total
        p_j = col_sums[0, j] / total
        denom = p_i * p_j
        if denom > 0:
            val = np.log2(p_ij / denom)
            if shifted_ppmi_k > 0:
                val -= np.log2(shifted_ppmi_k)
            ppmi[i, j] = max(val, 0.0)
    cooccur = ppmi
    # SVD
    logger.info('Computing SVD...')
    # note that we norm afterwards, so we don't have scale the matrix here
    if 1: # using sklearn
        # btw the docs say that there's a sign ambiguity, but if you use arpack they have a
        # sign_flip method that makes it deterministic
        svd = TruncatedSVD(n_components=min(256, n_tags-1), algorithm='arpack')
        U = svd.fit_transform(cooccur)
        S = svd.singular_values_
    else: # numpy version (has mem issues sometimes)
        U, S, VT = np.linalg.svd(cooccur, full_matrices=False)
    # pick the dimensionality
    total_variance = sum(S**2)
    variance = 0.0
    dim = 0
    cumvars = []
    while variance / total_variance < min_variance and dim < len(S):
        variance += S[dim]**2
        cumvars.append(float(variance / total_variance))
        dim += 1
    assert 2 <= dim <= 256, f'Unreasonable dimensionality {dim}'
    logger.info(f'Using {dim} dimensions to capture {100*variance/total_variance:.2f}% of variance')
    var_s = ' '.join(f'{v:.2f}' for v in cumvars)
    logger.info(f'Cumulative variances: {var_s}')
    embeddings = U[:, :dim]
    # normalize embeddings and return as dict
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    ret: dict[T, array1d] = {}
    for tag, idx in tag_to_idx.items():
        ret[tag] = embeddings[idx]
    return ret, cumvars

def gen_tag_embeddings(input_path: str, dlm: str='\t'):
    """Generates tag embeddings from the given `input_path`

    The input should be a text file with each line containing a list of `dlm`-separated tags that
    have co-occurred together.
    """
    with open(input_path) as f:
        data = [line.strip().split(dlm) for line in f if line.strip()]
    embs = generate_cooccurence_embeddings(data)
    print(embs.items()[:5])


if __name__ == '__main__':
    funcs = {f.__name__: f for f in [gen_tag_embeddings]}
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
