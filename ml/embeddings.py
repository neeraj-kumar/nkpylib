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

import faiss
import joblib
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.base import clone, BaseEstimator, ClassifierMixin, ClusterMixin # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.decomposition import TruncatedSVD # type: ignore
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier # type: ignore
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score # type: ignore
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import LinearSVC, SVC # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray2d,
    FeatureSet,
)
from nkpylib.ml.nklmdb import (
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)

logger = logging.getLogger(__name__)


def apply_cls_weights(X: array2d, classifier: BaseEstimator) -> nparray2d:
    """Apply classifier feature weights to feature vectors.

    - X: 2D sequence/array of shape `(n_samples, n_features)`
    - classifier: Trained `sklearn` classifier
      - For now, we only handle `SVC` with `kernel='linear'` or `SGDClassifier`

    Returns a reweighted `np.ndarray` of shape `(n_samples, n_features)`, produced by element-wise
    multiplication of each feature column by the corresponding classifier weight. For multiclass
    models, uses the mean absolute coefficient across classes to obtain a single weight per feature.
    """
    #FIXME note that if the classifier was trained using RBF sampler, we have to replicate that
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f'X must be 2D (n_samples, n_features); got shape {X_arr.shape}')
    coef: np.ndarray|None = None
    match classifier:
        case SVC() if getattr(classifier, 'kernel', None) == 'linear':
            coef = getattr(classifier, 'coef_', None)
        case SGDClassifier() | LinearSVC():
            coef = getattr(classifier, 'coef_', None)
        case _:
            raise NotImplementedError('apply_cls_weights supports only linear SVC or SGDClassifier.')
    if coef is None:
        raise ValueError('Classifier does not expose coef_. Ensure it is a trained linear model.')
    coef_arr = np.asarray(coef)
    if coef_arr.ndim == 1:
        weights = coef_arr
    else: # Aggregate multiclass weights to a single per-feature vector
        weights = np.mean(np.abs(coef_arr), axis=0)
    weights = np.abs(weights) # take absolute value to get importance regardless of direction
    if weights.shape[0] != X_arr.shape[1]:
        raise ValueError(f'Feature dimension mismatch: weights has {weights.shape[0]} dims, X has {X_arr.shape[1]}.')
    return X_arr * weights


class MixtureOfLinear(BaseEstimator, ClassifierMixin):
    """A Mixture-of-Linears classifier for binary classification.

    This performs clustering on the positive examples, and then trains a linear SVM for each cluster
    vs the negatives.

    You can optionally do an RBF transform on the input features before training the SVMs, which can
    help if the data is not linearly separable. This is done using `RBFSampler`, which approximates
    the RBF kernel.
    """
    #FIXME get this working
    def __init__(self,
                 clusterer:ClusterMixin|None=None,
                 classifier: ClassifierMixin|None=None,
                 n_rbf_components=0, #FIXME
                 gamma="scale"):
        """Give it the `clusterer` and `classifier` you want to use, and other params.

        - The default clusterer is `MiniBatchKMeans` with 5 clusters
        - The default classifier is Linear SGD with balanced class
        - The default RBF transform is 2000 components with gamma='scale'
          - Set n_rbf_components to 0 to disable the RBF transform
        """
        if clusterer is None:
            clusterer = MiniBatchKMeans(n_clusters=5)
            self.n_clusters = 5
        else:
            self.n_clusters = clusterer.n_clusters if hasattr(clusterer, 'n_clusters') else None
        self.clusterer = clusterer
        if classifier is None:
            classifier = SGDClassifier(class_weight="balanced")
        self.classifier = classifier
        self.n_rbf_components = n_rbf_components
        self.gamma = gamma

    def fit(self, X, y, **kw):
        logger.info(f'Input to MixtureOfLinear fit: X {X.shape}, y {Counter(y).most_common()}')
        X_pos = np.vstack([row for row, label in zip(X, y) if label > 0])
        X_neg = np.vstack([row for row, label in zip(X, y) if label <= 0])
        logger.info(f'Positive examples: {X_pos.shape}, Negative examples: {X_neg.shape}')
        pos_clusters = self.clusterer.fit_predict(X_pos)
        self.models_ = []
        for k in range(len(set(pos_clusters))):
            Xp = X_pos[pos_clusters == k]
            Xk = np.vstack([Xp, X_neg])
            yk = np.hstack([np.ones(len(Xp)), np.zeros(len(X_neg))])
            pipe = Pipeline([("clf", clone(self.classifier))])
            if self.n_rbf_components > 0:
                pipe.insert(0, ("rbf", RBFSampler(
                    gamma=self.gamma,
                    n_components=self.n_rbf_components,
                )))
            pipe.fit(Xk, yk)
            self.models_.append(pipe)
        return self

    def decision_function(self, X):
        scores = np.column_stack([
            m.decision_function(X) for m in self.models_
        ])
        return scores.max(axis=1)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


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

    def _create_classifier(self, method: str='rbf', C=1, class_weight='balanced', **kw) -> Any:
        """Creates a classifier instance without training it."""
        clf_kw = dict(class_weight=class_weight, **kw)
        match method:
            case 'rbf':
                return SVC(kernel='rbf', C=C, **clf_kw)
            case 'linear':
                return SVC(kernel='linear', C=C, **clf_kw)
            case 'sgd':
                return SGDClassifier(**clf_kw)
            case 'mixlinear':
                return MixtureOfLinear(classifier=SGDClassifier(**clf_kw), **kw)
            case _:
                raise NotImplementedError(f'Classifier method {method!r} not implemented.')

    def count_clustering_conflicts(self, labels: dict[KeyT, int], predictions: dict[KeyT, int]) -> dict[str, int]:
        """Count conflicts between labeled and predicted cluster assignments.

        Args:
            labels: Ground truth cluster assignments {item_id: cluster_num}
            predictions: Predicted cluster assignments {item_id: cluster_num}

        Returns:
            Dictionary with conflict counts:
            - separation_conflicts: labeled same-cluster pairs that got separated
            - merge_conflicts: labeled different-cluster pairs that got merged
            - total_conflicts: sum of above two
            - total_pairs_checked: total number of labeled pairs examined
            - conflict_rate: total_conflicts / total_pairs_checked
        """
        separation_conflicts = 0
        merge_conflicts = 0
        total_pairs_checked = 0
        # Get all items that have both labels and predictions
        common_items = set(labels.keys()) & set(predictions.keys())
        common_items = list(common_items)
        # Check all pairs of labeled items
        for i in range(len(common_items)):
            for j in range(i + 1, len(common_items)):
                item1, item2 = common_items[i], common_items[j]
                total_pairs_checked += 1
                # Check if they should be in same cluster (according to labels)
                same_label_cluster = labels[item1] == labels[item2]
                same_pred_cluster = predictions[item1] == predictions[item2]
                if same_label_cluster and not same_pred_cluster:
                    separation_conflicts += 1
                elif not same_label_cluster and same_pred_cluster:
                    merge_conflicts += 1
        total_conflicts = separation_conflicts + merge_conflicts
        conflict_rate = total_conflicts / total_pairs_checked if total_pairs_checked > 0 else 0.0
        return dict(
            separation_conflicts=separation_conflicts,
            merge_conflicts=merge_conflicts,
            total_conflicts=total_conflicts,
            total_pairs_checked=total_pairs_checked,
            conflict_rate=conflict_rate
        )

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
        elif method == 'rbf': # apply multiclass rbf classifier
            keys_all, embs = self.get_keys_embeddings(keys=keys, normed=False, scale_mean=True, scale_std=True)
            # if we don't have any labels for cluster=1, then add a few dummy examples at low weight
            orig_labels = labels.copy()
            key_order = list(labels)
            if 1 not in labels.values():
                neg_keys = random.sample([k for k in keys_all if k not in labels], min(5, len(self)//10))
                for nk in neg_keys:
                    labels[nk] = 1
                key_order = list(labels)
                kwargs['weights'] = [0.1 if k in neg_keys else 1.0 for k in key_order]
            cls_kw = dict(
                X=np.vstack([embs[keys_all.index(key)] for key in key_order]),
                y=[labels[key] for key in key_order],
                method='rbf',
                **kwargs)
            cls = self.train_classifier(C=1, **cls_kw)
            logger.info(f'Training {method} with labels {labels}: {cls_kw}')
            scores = cls.decision_function(embs)
            pred_labels = cls.predict(embs)
            print(f'got labels {pred_labels}, {pred_labels.shape}, scores of shape {scores.shape}')
            for i, key in enumerate(keys_all):
                if len(cls.classes_) == 2:
                    score = scores[1]
                else: # multiclass, look up score for predicted class
                    pred_cls_idx = list(cls.classes_).index(pred_labels[i])
                    score = scores[i][pred_cls_idx]
                clusters[key] = dict(num=int(pred_labels[i]), score=float(score))
        else:
            # apply clustering method repeatedly until we have no conflicts with labels
            clusterer = self.get_clusterer(method=method, n_clusters=n_clusters)
            keys_all, embs = self.get_keys_embeddings(keys=keys, normed=False, scale_mean=True, scale_std=True)
            #print(keys_all, embs)
            labels_array = np.array([labels[key] if key in labels else -1 for key in keys_all])
            print(labels_array)
            for iteration in range(5):
                # do a clustering
                pred_labels = clusterer.fit_predict(embs)
                # create predictions dict for conflict checking
                predictions = {key: int(pred_labels[i]) for i, key in enumerate(keys_all)}
                # check for conflicts using the new method
                conflict_info = self.count_clustering_conflicts(labels, predictions)
                print(f"Iteration {iteration}: {conflict_info}")
                if conflict_info['total_conflicts'] == 0:
                    break
            # assign scores based on distance to cluster center
            centers = clusterer.cluster_centers_
            for i, key in enumerate(keys_all):
                center = centers[pred_labels[i]]
                dist = np.linalg.norm(embs[i] - center)
                score = 1 / (1 + dist)
                clusters[key] = dict(num=int(pred_labels[i])+1, score=float(score))
        return clusters

    def cluster(self, n_clusters=-1, method='kmeans', all_keys=None, **kwargs) -> list[list[KeyT]]:
        """Clusters our embeddings.

        If `n_clusters` is not positive (default), we set it to the sqrt of the number of
        embeddings we have.

        Returns a list of lists of keys, where each list is a cluster; in order from largest to smallest.
        """
        keys, embs = self.get_keys_embeddings(keys=all_keys, normed=False, scale_mean=True, scale_std=True)
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

        This is a wrapper on top of `nearest_neighbors()` (method='nn') and `train_classifier()`
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
                clf = self.train_classifier(X, y, method=method, **kw)
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

    def train_and_run_classifier(self,
                                 pos: list[KeyT],
                                 neg: list[KeyT],
                                 to_cls: list[KeyT],
                                 neg_weight: float=1.0,
                                 method: str='rbf',
                                 C=1,
                                 cv: int=0,
                                 **kw) -> tuple[BaseEstimator, dict[KeyT, float], dict[str, Any]]:
        """High-level function to train a classifier with given `pos` and `neg` and run on `to_cls`.

        The params `method`, `C`, and `kw` are fed into `train_classifier()`, which in turn uses them
        in `_create_classifier()`.

        Returns `(classifier, scores_dict, other_stuff)`, where `scores_dict` is a dict of key to
        score, and `other_stuff` includes:
        - times: dict with timing info for training and inference
        - scaler: the scaler used (if any)
        - cv: if `cv` > 0, the list of cross-validation scores for `cv` folds
        """
        other_stuff = {}
        times = [time.time()]
        assert len(to_cls) > 0
        # we get initial embeddings for all keys to normalize correctly.
        keys, embs, scaler = self.get_keys_embeddings(
            keys=pos+neg+to_cls,
            normed=False,
            scale_mean=True,
            scale_std=True,
            return_scaler=True,
        )
        times.append(time.time())
        other_stuff['scaler'] = scaler
        pos_set = set(pos)
        neg_set = set(neg)
        to_cls = set(to_cls)
        train_X = []
        y = []
        weights = []
        test_keys = []
        test_X = []
        logger.info(f'Got {len(keys)} keys, {len(set(keys))} unique keys')
        for k, emb in zip(keys, embs):
            if k in pos_set or k in neg_set:
                train_X.append(emb)
                y.append(1 if k in pos_set else -1)
                weights.append(1.0 if k in pos_set else neg_weight)
            if k in to_cls:
                test_keys.append(k)
                test_X.append(emb)
        train_X = np.vstack(train_X)
        #sampler = RBFSampler(gamma=1.0/len(emb), n_components=5000)
        sampler = RBFSampler(gamma='scale', n_components=4000)
        if method in ('linear', 'sgd'):
            other_stuff['sampler'] = sampler
            train_X = sampler.fit_transform(train_X)
        times.append(time.time())
        logger.info(f'In training, trainX: {train_X.shape}, y: {Counter(y).most_common()}')
        # Perform cross-validation if requested
        if cv > 0:
            cv_classifier = self._create_classifier(method=method, C=C, **kw)
            cv_scores = cross_val_score(cv_classifier, train_X, y, cv=cv, params=dict(sample_weight=weights))
            other_stuff['cv'] = [float(s) for s in cv_scores]
            logger.info(f'Cross-validation scores: {cv_scores}, mean: {cv_scores.mean():.3f}')
        cls = self.train_classifier(train_X, y, weights=weights, method=method, C=C, **kw)
        times.append(time.time())
        test_X = np.vstack(test_X)
        test_X = sampler.transform(test_X)
        scores = {key: float(s) for key, s in zip(test_keys, cls.decision_function(test_X))}
        times.append(time.time())
        other_stuff['times'] = dict(
            training=times[3] - times[2],
            inference=times[4] - times[3]
        )
        logger.info(f'train_and_run_classifier times: {[t1-t0 for t0, t1 in zip(times, times[1:])]}')
        return cls, scores, other_stuff

    def run_classifier(self,
                       to_cls: list[KeyT],
                       classifier: BaseEstimator,
                       scaler: StandardScaler|None=None,
                       sampler: RBFSampler|None=None) -> dict[KeyT, Any]:
        """Runs `classifier` on `to_cls`, returning dict of key to score."""
        logger.debug(f'running inference on {len(to_cls)}: {to_cls[:5]}...')
        keys, embs, scaler = self.get_keys_embeddings(
            keys=to_cls,
            normed=False,
            scale_mean=True,
            scale_std=True,
            scaler=scaler,
            return_scaler=True,
        )
        if not keys:
            return {}
        if sampler is not None:
            embs = sampler.transform(embs)
        scores_array = classifier.decision_function(embs)
        return {key: float(score) for key, score in zip(keys, scores_array)}

    def train_classifier(self,
                         X: nparray2d,
                         y: Sequence[float|int],
                         weights: Sequence[float]|None=None,
                         method: str='rbf',
                         C=1,
                         class_weight='balanced',
                         **kw) -> Any:
        """Makes a classifier with given `method`, trains it on X, y, and returns it.

        This is a somewhat low-level method; for common simple cases, you might want to use
        `train_and_run_classifier()` instead.

        If `weights` is provided, it should be of the same length as `keys` and is a weight for key.
        These can be negative as well.

        Note that we don't do any preprocessing of X here; you should do that before calling this.
        For most methods, it helps to scale dimensions to have mean 0 and std 1. Use a
        `StandardScaler` to do this.
        """
        assert len(X) == len(y), f'Length of X {len(X)} must match y {len(y)}'
        if weights is not None:
            assert len(X) == len(weights), f'Length of weights {len(weights)} must match X {len(X)}'
        X = np.asarray(X)
        logger.debug(f'training labels {Counter(y).most_common()}, X: {X.shape}, {X}')
        clf = self._create_classifier(method=method, C=C, class_weight=class_weight, **kw)
        clf.fit(X, y, sample_weight=weights)
        return clf

    def rescore_by_nn(self,
                     scores: dict[KeyT, float],
                     pos: list[KeyT],
                     min_score: float=1.0,
                     k: int=20,
                     metric: str='l2') -> dict[KeyT, float]:
        """Given some existing `scores`, reweight the high ones based on number of NN in `radius`.

        This is typically for SVMs, for which high scores are not discriminative enough (i.e., not
        based on density of nearby positives), since it's optimizing for pos/neg separation.

        So for all scores higher than `min_score`, we instead compute `k`-nearest neighbors and use
        the median distance to get a new score.
        """
        all_keys = set(pos)
        n_high = 0
        for key, s in scores.items():
            if s >= min_score: # we only care about those above min_score
                n_high += 1
                all_keys.add(key)
        keys, embs = self.get_keys_embeddings(keys=list(all_keys), normed=True, scale_mean=False, scale_std=False)
        logger.info(f'In rescore, {len(scores)} scores, {n_high} high (>= {min_score}), {len(pos)} pos, {len(all_keys)} all keys -> {embs.shape}')
        logger.debug(f'  First scores: {list(scores.items())[:5]}, first pos: {pos[:5]}')
        if len(embs) == 0:
            return scores
        search_indices = [i for i, key in enumerate(keys) if key in scores and scores[key] >= min_score]
        logger.debug(f'Got {len(search_indices)} search indices for rescoring: {search_indices[:5]}, {[keys[x] for x in search_indices[:5]]}.')
        to_search = np.vstack([embs[i] for i in search_indices])
        logger.debug(f'Searching {to_search.shape} embeddings for {k} neighbors')
        if 0:
            nn = NearestNeighbors(n_neighbors=k, metric=metric)
            nn.fit(embs)
            distances, indices = nn.kneighbors(to_search, n_neighbors=min(k, len(to_search)), return_distance=True)
        else: # faiss
            times = [time.time()]
            # IndexFlatIP(d) for cosine sim with normed vectors
            # IndexHNSWFlat(d, 32) for faster approximate search
            # IndexFlatL2(d) for l2 distance
            index = faiss.IndexHNSWFlat(embs.shape[1], 32)
            index.add(embs)
            times.append(time.time())
            distances, indices = index.search(to_search, min(k, len(to_search)))
            times.append(time.time())
            logger.debug(f'Faiss {index} times: {[(t1-t0) for t0, t1 in zip(times, times[1:])]}')
        logging.debug(f'New dists: {distances}')
        distances = np.array([np.sum(row) for row in np.exp(-distances)]) / k
        logging.debug(f'Scaled dists: {distances}')
        #logging.info(f'New dists: {np.array([len(ind) for ind in indices])}')
        # add these as increments on min_score to existing high scores
        ret: dict[KeyT, float] = dict(**scores)
        for dist, idx in zip(distances, search_indices):
            key = keys[idx]
            ret[key] = min_score + float(dist)
            assert ret[key] <= scores[key]+1.0, f'Rescored {key} too high: {ret[key]} vs {scores[key]}, min={min_score}'
            #logger.info(f'  Rescored {key} ({min_score}, {dist}): {scores[key]} -> {ret[key]}')
        return ret

    def save_classifier(self, path: str, classifier: BaseEstimator, **kw) -> dict[str, Any]:
        """Save classifier with additional metadata using joblib.

        - path: Path where to save the classifier (will add .joblib extension if missing)
        - classifier: The trained classifier to save
        - **kw: Additional metadata to save with the classifier

        Returns the saved data dictionary.
        """
        # Ensure path has the correct extension
        if not path.endswith('.joblib'):
            path = path + '.joblib'
        try:
            os.makedirs(dirname(path), exist_ok=True)
        except Exception:
            pass
        save_data = dict(classifier=classifier, **kw)
        if 'created_time' not in kw:
            save_data['created_at'] = time.time()
        # Save using joblib
        joblib.dump(save_data, path)
        logger.debug(f"Saved classifier to {path}")
        return save_data

    @staticmethod
    def load_and_setup_classifier(path: str) -> dict[str, Any]:
        """Load classifier from `path`, returning the saved data dict"""
        # Load the saved data
        saved_data = joblib.load(path)
        logger.debug(f"Loaded classifier from {path}")
        return saved_data


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

def compute_binary_classifier_stats(y_true: list[int], y_scores: list[float]) -> dict[str, float]:
    """Compute comprehensive binary classification statistics.

    Args:
    - y_true: True binary labels (should be 1 for positive, -1 or 0 for negative)
    - y_scores: Classifier scores (higher scores indicate positive prediction)

    Returns a dictionary with evaluation metrics including accuracy, precision, recall, F1, etc.
    """
    if not y_true or not y_scores:
        return {}
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} items, y_scores has {len(y_scores)}")
    # Convert scores to binary predictions (positive if score > 0)
    y_pred = [1 if score > 0 else -1 for score in y_scores]
    # Convert -1/1 labels to 0/1 for some metrics that require it
    y_true_binary = [1 if label == 1 else 0 for label in y_true]
    y_pred_binary = [1 if pred == 1 else 0 for pred in y_pred]
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary'
    )
    # ROC AUC using raw scores
    try:
        auc = roc_auc_score(y_true_binary, y_scores)
    except ValueError:
        auc = 0.0  # In case of issues with AUC calculation
    # Compute ranking metrics
    # Sort by score descending
    sorted_items = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    # Precision at different cutoffs
    def precision_at_k(k):
        if k > len(sorted_items):
            k = len(sorted_items)
        top_k = sorted_items[:k]
        correct = sum(1 for _, label in top_k if label == 1)
        return correct / k if k > 0 else 0.0

    p_at_10 = precision_at_k(10)
    p_at_50 = precision_at_k(50)
    p_at_100 = precision_at_k(100)
    # Mean Average Precision (MAP)
    def mean_average_precision():
        precisions = []
        correct = 0
        for i, (_, label) in enumerate(sorted_items):
            if label == 1:
                correct += 1
                precisions.append(correct / (i + 1))
        return sum(precisions) / len(precisions) if precisions else 0.0

    map_score = mean_average_precision()
    # Count positives and negatives
    n_positive = sum(1 for label in y_true if label == 1)
    n_negative = len(y_true) - n_positive
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'precision_at_10': float(p_at_10),
        'precision_at_50': float(p_at_50),
        'precision_at_100': float(p_at_100),
        'mean_average_precision': float(map_score),
        'n_items': len(y_true),
        'n_positive': n_positive,
        'n_negative': n_negative,
    }


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
