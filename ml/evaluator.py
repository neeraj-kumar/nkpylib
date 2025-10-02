"""A generic embeddings evaluator.

This does a bunch of semi-automated things to test out embeddings and see how good they are.
This is mostly in reference to a set of labels you provide, in the form of a tags database.

TODO:

- Collect all errors (particularly task failures) in one place
- Restartability
- Lots of ways of mapping embeddings to other embeddings, should be common procedure
  - PCA, K-PCA, ISOMAP, t-SNE, UMAP, LLE, Beta-VAE, etc
  - Also labeled methods like CCA and PLS
  - skip ICA, it's more for source separation
  - Don't repeat work
  - Correlate embedding dimensions before/after transformation
  - estimate "intrinsic dimensionality" of the embeddings via MLE or TwoNN
- Classification/regression against labels
  - pytorch nn?
- For dims, we can also compute histograms
  - Standard stats on histograms: bin sizes, lop-sidedness, normality
  - highlight 0s and other outliers
- Compare labels of same type but different keys, e.g. genre
  - Look at confusion matrices
  - Also clustering metrics for label similarity?
  - With small number of labels, no need to embed labels, just do one-hot
- For numerical labels, look at orig values, log(val) and exp(val) where relevant
- Zeros in large numbers/parts of embeddings
- Distances between labeled points
  - For multiclass, we might need embeddings for labels to get distances
  - combine distances across labels
    - Have to be careful about scaling
    - Join in neighbor-space? Distance-space?
    - Might have multiple combined distances to compare
- What to do with distances
  - compare distribution histograms
  - Also between different embedding distances (e.g. euclidean vs cosine)
  - Visualize full pairwise cosine similarity heatmaps — useful for spotting large dense cliques (bad) or disconnected islands (good/bad, depending).
  - Compute pairwise angles between random vector pairs. For high-quality, high-dimensional embeddings, the distribution should be tightly centered.
  - Can also do classification (near/far)/regression on distances
    - metric learning using triplet or contrastive loss?
- Clustering
  - Inputs of distance or neighbors for all methods
  - Look at cluster size distributions
  - Tightness and separation of clusters
    - Can use silhouette score, Davies-Bouldin index, etc.
  - Apply clusterings to embeddings and to labels (based on distances/neighbors)
  - Use standard clustering metrics
  - Modified confusion matrix: 1 row per cluster, 1 col per label, show purity/counts/etc per cell
  - Does label prop work?
- Outlier detection
  - Run unsupervised outlier detection algorithms (e.g., Isolation Forest, LOF) on the embedding space — helpful to spot anomalies or collapsed modes.
- Maybe more generally for each Label, output best predictors/correlators/etc
- Each op should define criteria for highlighting things and add them to a special obj in OM
  - Remember to add back in top embedding dims with correlation
- In the future, do ML doctor stuff
- Feature selection
  - E.g. we have budget/revenue, but only as one or two dims
- Performance
  - sigopt for hyperparameter tuning (including which classifier to use)
    - different rbf params (C, alpha)
  - Figure out how to order different operations, including not evaluating things if already
    promising alternatives

Old stuff:
- Recommendation system
- Few-shot classifier
- Analogies, e.g., "comedy - dark + romance" ≈ "romcom"
- Sequence modeling using embeddings as input
- Calibration of similarity scores
  - Compare histograms or CDFs of same-class vs diff-class scores
- Simulate synthetic user with known prefs and test how well NN align with prefs?
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
import time
import warnings

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections.abc import Mapping
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import abspath, dirname, exists, join
from pprint import pprint as _pprint, pformat
from typing import Any, Literal, Sequence, Generic, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn

from pony.orm import * # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from scipy.special import kl_div
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from sklearn.linear_model import Ridge, SGDClassifier # type: ignore
from sklearn.metrics import recall_score, r2_score, balanced_accuracy_score, accuracy_score # type: ignore
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold # type: ignore
from sklearn.metrics import recall_score, r2_score, balanced_accuracy_score, accuracy_score # type: ignore
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from sklearn.linear_model import Ridge, SGDClassifier # type: ignore
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR # type: ignore
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.ml.evaluator_ops import Op, OpManager, error_logger, result_logger
from nkpylib.ml.feature_set import (
    FeatureSet,
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)
from nkpylib.ml.types import (
    NUMERIC_TYPES,
    FLOAT_TYPES,
    array1d,
    array2d,
    nparray1d,
    nparray2d,
    )

from nkpylib.ml.tag_db import Tag, get_all_tags, init_tag_db

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Create specialized loggers for different components
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("evaluator.op")
eval_logger = logging.getLogger("evaluator.eval")

# get console width from system
CONSOLE_WIDTH = os.get_terminal_size().columns

# pprint should use full width
pprint = lambda x: _pprint(x, width=CONSOLE_WIDTH)

# a distance tuple has (id1, id2, distance)
DistTuple = tuple[str, str, float]

# All distances is a dict with various fields
AllDists = dict[str, Any]

# stats are for now just a dict of strings
Stats = dict[str, Any]

# Define a literal type for task types
PTaskType = Literal['classification', 'regression']

# Define a type for the task data (numpy array)
PTaskData = np.ndarray

# Define the task tuple type
PTask = tuple[PTaskData, PTaskType]

# Define the tasks dictionary type
PTasks = dict[str, PTask]

def train_and_predict(model, X_train, y_train, X_test):
    """Simple train and predict function for use in multiprocessing."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

def get_array1d_stats(x: nparray2d) -> list[Stats]:
    """Returns a list of 1d array stats for each row of an input 2D numpy array."""
    S = dict() # temporary storage of stats arrays
    S['mean'] = np.mean(x, axis=1)
    S['std'] = np.std(x, axis=1)
    S['min'] = np.min(x, axis=1)
    S['max'] = np.max(x, axis=1)
    S['median'] = np.median(x, axis=1)
    # Percentiles - computed for all rows at once
    percentiles = np.percentile(x, [1, 5, 25, 75, 95, 99], axis=1)
    S['p1'], S['p5'], S['p25'], S['p75'], S['p95'], S['p99'] = percentiles
    # Count-based stats
    S['n_neg'] = np.sum(x < 0, axis=1)
    S['n_zero'] = np.sum(x == 0, axis=1)
    S['n_pos'] = np.sum(x > 0, axis=1)
    # Statistical measures
    S['kurtose'] = stats.kurtosis(x, axis=1)
    S['gmean'] = stats.gmean(x, axis=1)
    S['skew'] = stats.skew(x, axis=1)
    S['entropy'] = np.array([stats.entropy(row) for row in x])
    ret = [{k: v[i] for k, v in S.items()} for i in range(x.shape[0])]
    return ret

def compare_array1d_stats(a: array1d, b: array1d, *,
                          stats_a: Stats|None=None, stats_b: Stats|None=None) -> Stats:
    """Returns comparison stats between two 1D arrays.

    This computes some pairwise measures between `a` and `b`, such as:
    - Pearson correlation coefficient
    - Spearman's rank correlation coefficient
    - Kendall's tau (rank correlation)
    - KL divergence (treating as distributions)
    - R^2 value of computing a linear regression of `a` vs `b`

    In addition, if you provide 1d array stats for `a` and `b`, it will compute the the differences
    between the stats (b-a) and include them in the output dict, with keys prefixed by `diff_`.

    These are all returned in a dict.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    assert a.ndim == b.ndim == 1
    ret = dict(
        pearson=float(np.corrcoef(a, b)[0, 1]),
        spearman=float(stats.spearmanr(a, b).statistic),
        tau=float(stats.kendalltau(a, b).statistic),
        kl_div=float(stats.entropy(a, b)),
    )
    # compute least squares linear fit to get rvalue
    try:
        res = stats.linregress(a, b)
        ret.update(linear_least_square_r2=float(res.rvalue)**2.0)
    except Exception as e:
        error_logger.exception(e)
        ret.update(linear_least_square_r2=float('nan'))
    if stats_a is not None and stats_b is not None:
        ret.update({f'diff_{k}': stats_b[k] - stats_a[k] for k in stats_a})
    return ret

def join_mpl_figs(figs: list[mpl.figure.Figure], scaling: float=5) -> mpl.figure.Figure:
    """Joins multiple matplotlib figures into one figure with subplots.

    This tries to make something as close to square as possible.
    """
    raise NotImplementedError("This function is broken, needs fixing")
    n = len(figs)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n/rows))

    # Create new figure with subplots
    fig_combined, axes = plt.subplots(rows, cols, figsize=(cols*scaling, rows*scaling))
    axes = axes.flat  # Flatten axes array for easier indexing

    # Copy contents of each figure to a subplot
    for i, fig in enumerate(figs):
        # Get the contents from original figure
        original_ax = fig.axes[0]
        # Copy to new subplot
        axes[i].get_figure().canvas.draw()
        axes[i].imshow(np.asarray(fig.canvas.renderer.buffer_rgba()))
    return fig_combined


class Labels(ABC):
    """A base class for different types of labels that we get from tags.

    This stores metadata about the types of labels, and the specialized values.
    The 'tag_type' is often generic (e.g. 'genre') and could come from multiple sources, and so the
    'key' is more specific (e.g. 'imdb-genre'). All labels with the same key should have the same
    tag type, but not vice versa. There should be exactly one `Labels` instance per key.
    """
    def __init__(self, tag_type: str, key: str, *, ids: list[str], values: Any, norm_type: str='raw'):
        self.tag_type = tag_type
        self.key = key
        self.ids = ids
        self.values = values
        self.norm_type = norm_type

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.tag_type} {self.key} ({len(self.ids)} labels)>'

    def get_distance(self, idx1: int, idx2: int, **kw) -> float:
        """Returns distance between two id indices.

        This is implemented by subclasses to define their specific distance metric.
        Distance should ideally be normalized between 0 (identical) and 1 (maximally different).
        """
        raise NotImplementedError()

    def get_all_distances(self, n_pts: int, keys: list[str], matrix: nparray2d, perc_close: float = -1, **kw) -> AllDists:
        """Returns all pairwise distances between `n_pts` points.

        We try to sample at least `perc_close` points that are "close" according to the label type's
        definition of closeness. < 0 means we don't care about closeness (default).

        This is a naive implementation that ignores `perc_close` and just does random sampling.
        It also computes distances one-by-one using `get_distance()`. It passes all kw to
        `get_distance()`.

        Returns a dict with the following fields:
        - `sub_keys` is the list of overlapping keys
        - `label_distances` is a 2d np array of distances between the overlapping keys
          - Shape `(len(sub_keys), len(sub_keys))`
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
          - Shape `(len(sub_keys), matrix.shape[1])`
        """
        assert n_pts > 1, 'Must have at least 2 points to compute distances'
        ids = [id for id in self.ids if id in keys]
        if n_pts > len(ids):
            n_pts = len(ids)
        ids = sorted(random.sample(ids, n_pts))
        id_indices, sub_keys, sub_matrix = self.get_matching_matrix(keys, matrix, ids=ids)
        op_logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
        dists = self.compute_all_distances(ids, **kw)
        assert len(sub_keys) == len(sub_matrix) == len(ids) == dists.shape[0] == dists.shape[1]
        assert sub_matrix.shape[1] == matrix.shape[1]
        return dict(
            sub_keys=sub_keys,
            label_distances=dists,
            sub_matrix=sub_matrix,
        )

    def compute_all_distances(self, ids: list[str], **kw) -> array2d:
        """Computes all distances in `ids` using `get_distance(id1, id2, **kw)`."""
        n_pts = len(ids)
        dists = np.zeros((n_pts, n_pts), dtype=np.float32)
        for i, id1 in enumerate(ids):
            idx1 = self.ids.index(id1)
            for j in range(i+1, n_pts):
                id2 = ids[j]
                idx2 = self.ids.index(id2)
                dists[i, j] = dists[j, i] = dist = self.get_distance(idx1, idx2, **kw)
        return dists

    @abstractmethod
    def get_pair_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[DistTuple]:
        """Returns `n_pairs` of `(id1, id2, distance)` tuples.

        You can specify `perc_close` which is the minimum percentage of pairs that should be "close"
        according to the label type's definition of closeness. < 0 means we don't care about
        closeness (default). Note that there might be more close pairs than this, since for the
        "non-close" pairs we randomly sample, and those might be close too.

        Returns a list of `(id1, id2, distance)` tuples. A distance of 0 implies the points are
        identical (according to this distance metric), but the upper-bound is variable, depending on
        the specific subclass/etc.
        """
        raise NotImplementedError()

    def get_matching_matrix(self, keys: list[str], matrix: array2d, ids: list[str]|None=None) -> tuple[array1d, list[str], nparray2d]:
        """Returns matching submatrix based on overlapping keys.

        This does a set intersection between our ids and the given `keys`, and returns a tuple of
        `(id_indices, sub_keys, sub_matrix)`, where:
        - `id_indices` is the list of indices into our `self.ids` array corresponding to the
          overlapping keys, so that `sub_matrix[i]` corresponds to `self.ids[id_indices[i]]`,
        - `sub_keys` is the filtered list of keys that correspond to the intersection
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys, with same
          dimensionality as the input `matrix`.

        If `ids` is given, it is used instead of `self.ids` to find the intersection.

        In other words, you can iterate through all 3 in parallel.

        Note that if ids repeat in `self.ids`, this will use the first matching index.
        """
        assert len(keys) == len(matrix), f'Keys {len(keys)} and matrix ({matrix.shape}) rows must match'
        # get row indices of our ids in keys and in self
        mat_indices = []
        id_indices = []
        assert len(keys) == len(set(keys)), 'Keys should be unique'
        if ids is None:
            ids = list(self.ids)
        common = set(keys) & set(ids)
        assert common, 'No matching ids found between {len(keys)} input keys and {ids}'
        sub_keys = []
        for mat_idx, id in enumerate(keys):
            if id not in common:
                continue
            label_idx = self.ids.index(id)
            mat_indices.append(mat_idx)
            id_indices.append(label_idx)
            sub_keys.append(id)
        op_logger.debug(f'  Found {len(common)} matching ids in embeddings')
        id_indices = np.asarray(id_indices)
        sub_matrix = matrix[mat_indices, :]
        op_logger.debug(f'Got sub matrix of shape {sub_matrix.shape}: {sub_matrix}')
        assert sub_matrix.shape == (len(id_indices), matrix.shape[1])
        assert len(id_indices) == len(sub_keys) == len(sub_matrix)
        return id_indices, sub_keys, sub_matrix

    @abstractmethod
    def get_label_arrays(self, keys: list[str], matrix: nparray2d) -> dict[str, Any]:
        """Returns a list of 1d arrays of numeric values using the given `keys` to filter down.

        This checks for overlap between the given `keys` and our `self.ids`, and returns 1 or more
        1d arrays, packed into a numpy 2d array. These might correspond to a different label, or to
        different transformations of our underlying data, but in any case they are given unique
        names.

        It returns a dict with the fillowing keys:
        - `sub_keys` is the list of overlapping keys
        - `label_names` is a list of names, one for each row of `label_arrays`
        - `label_arrays` is a 2d array of values corresponding to each name and the overlapping keys
          - Shape `(len(label_names), len(sub_keys))`
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
          - Shape `(len(sub_keys), matrix.shape[1])`
        """
        raise NotImplementedError()

    def _pairs_to_list(self, pairs: dict[frozenset[str], float]) -> list[DistTuple]:
        """Converts `pairs` dict to list of `(id1, id2, distance)` tuples."""
        ret = []
        for spair, dist in pairs.items():
            id1, id2 = sorted(spair)
            ret.append((id1, id2, dist))
        ret.sort(key=lambda x: x[2])
        return ret


class NumericLabels(Labels):
    """A class for numeric labels. For now we convert them to floats.

    This stores ids as a list and values as a numpy array, where values[i] is the value for ids[i].
    """
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Union[NUMERIC_TYPES]]]):
        ids = [id for id, v in ids_values]
        assert len(ids) == len(set(ids)), 'Ids should be unique'
        values = np.array([v for id, v in ids_values], dtype=np.float32)
        fix_norm = lambda f: f if f > 0 else 1
        self.norm_factors = dict(
            range=fix_norm(np.max(values) - np.min(values)),
            std=fix_norm(np.std(values)),
            raw=1.0,
        )
        super().__init__(tag_type, key, ids=ids, values=values)

    def get_label_arrays(self, keys: list[str], matrix: nparray2d) -> dict[str, Any]:
        """Returns a list of 1d arrays of numeric values using the given `keys` to filter down.

        This checks for overlap between the given `keys` and our `self.ids`, and returns 1 or more
        1d arrays, packed into a numpy 2d array. These might correspond to a different label, or to
        different transformations of our underlying data, but in any case they are given unique
        names.

        It returns a dict with the fillowing keys:
        - `sub_keys` is the list of overlapping keys
        - `label_names` is a list of names, one for each row of `label_arrays`
        - `label_arrays` is a 2d array of values corresponding to each name and the overlapping keys
          - Shape `(len(label_names), len(sub_keys))`
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
          - Shape `(len(sub_keys), matrix.shape[1])`
        """
        ret = dict(label_names=['value'])
        id_indices, ret['sub_keys'], ret['sub_matrix'] = self.get_matching_matrix(keys, matrix)
        # convert the values into a 2d array with one row
        ret['label_arrays'] = self.values[id_indices].reshape((1, -1))
        assert len(ret['label_arrays']) == len(ret['label_names'])
        assert len(ret['label_arrays'][0]) == len(ret['sub_keys']) == len(ret['sub_matrix'])
        assert ret['sub_matrix'].shape[1] == matrix.shape[1]
        return ret

    def get_distance(self, idx1: int, idx2: int, norm_type: str='raw', **kw) -> float:
        """Returns distance between two id indices.

        You can specify `norm_type`:
        - 'raw' (default): absolute difference between values
        - 'range': absolute difference divided by (max-min)
        - 'std': absolute difference divided by stddev
        """
        dist = abs(self.values[idx1] - self.values[idx2]) / self.norm_factors[norm_type]
        return dist

    def get_pair_distances(self, n_pairs: int, perc_close: float = -1,
                      norm_type: str='raw', close_thresh=0.2, **kw) -> list[DistTuple]:
        """Returns `n_pairs` of `(id1, id2, distance)` tuples.

        For numeric labels, we have the 'raw' distance which just the absolute difference between
        values. You can choose to normalize this using either 'range' (max-min) or 'std' (stddev) by
        specifying `norm_type`.

        You can specify `perc_close` which is the percentage of pairs that should be "close", where
        this is any normalized distance <= `close_thresh`. `perc_close < 0` means we don't care about
        closeness (default) when generating pairs.

        Note that if the number of total possible pairs or the number of close pairs is less than
        requested, we will return fewer pairs.

        Returns a list of `(id1, id2, distance)` tuples. A distance of 0 implies the points are
        identical (according to these labels), but the upper-bound is variable, depending on
        the normalization used.
        """
        # Compute all pairwise differences for consecutive values and then cumsums
        sorted_indices = np.argsort(self.values)
        sorted_values = self.values[sorted_indices]
        diffs = np.diff(sorted_values)
        diffs = diffs / self.norm_factors[norm_type]
        cum_diffs = np.cumsum(diffs)

        # we store pairs as a frozenset pair of ids mapping to distance
        pairs: dict[frozenset[str], float] = {}
        def add_pairs(lst: list[Any], n_requested: int, *, is_indices:bool):
            """Samples up to `n_requested` items from `lst` (pairs) and adds them.

            If `is_indices` is True, the pairs are indices into sorted_values/indices, otherwise
            they are the ids themselves.
            """
            if not lst:
                return
            pair_indices = random.sample(lst, min(n_requested, len(lst)))
            for i, j in pair_indices:
                if is_indices:
                    id1 = self.ids[sorted_indices[i]]
                    id2 = self.ids[sorted_indices[j]]
                    dist = abs(sorted_values[j] - sorted_values[i])
                else:
                    id1, id2 = i, j
                    dist = abs(self.values[self.ids.index(id1)] - self.values[self.ids.index(id2)])
                spair = frozenset((id1, id2))
                pairs[spair] = dist / norm_factor

        if perc_close >= 0:
            n_close = int(n_pairs * perc_close)
            # Find all possible close pairs by sliding window, then sample from them
            close_pairs = []
            for i in range(len(sorted_values)):
                # Use cumsum to find rightmost index where distance is still <= close_thresh
                j = i + 1
                while j < len(sorted_values) and (
                        cum_diffs[j-1] - (cum_diffs[i-1] if i > 0 else 0) <= close_thresh):
                    close_pairs.append((i, j))
                    j += 1
            add_pairs(close_pairs, n_close, is_indices=True)
        # Fill remaining pairs with random sampling
        n_remaining = n_pairs - len(pairs)
        if n_remaining > 0:
            # Generate all possible pairs, removing those we've done, and sample from them
            poss = ((id1, id2) for i, id1 in enumerate(self.ids) for j, id2 in enumerate(self.ids) if i < j)
            poss = [p for p in poss if frozenset(p) not in pairs]
            add_pairs(poss, n_remaining, is_indices=False)
        return self._pairs_to_list(pairs)


class MulticlassBase(Labels):
    """Some common code for multiclass/multilabel labels."""
    def by_label(self) -> dict[Any, set[str]]:
        raise NotImplementedError()

    def get_all_distances(self, n_pts: int, keys: list[str], matrix: nparray2d, perc_close: float = -1, **kw) -> AllDists:
        """Returns all pairwise distances between `n_pts` points.

        We try to sample at least `perc_close` points that are "close" according to the label type's
        definition of closeness. < 0 means we don't care about closeness (default).

        This implementation samples points from the same label groups if `perc_close > 0` to try to
        get points which share at least one label in common, in rough proportion to the size of each
        label group.

        Returns a dict with the following fields:
        - `sub_keys` is the list of overlapping keys
        - `label_distances` is a 2d np array of distances between the overlapping keys
          - Shape `(len(sub_keys), len(sub_keys))`
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
          - Shape `(len(sub_keys), matrix.shape[1])`
        """
        assert n_pts > 1, 'Must have at least 2 points to compute distances'
        valid_ids = [id for id in self.ids if id in keys]
        if n_pts > len(valid_ids):
            n_pts = len(valid_ids)
        ids = set()
        n_close = int(n_pts * perc_close)
        if n_close > 0:
            groups = {}
            for label, ids in self.by_label().items():
                cur_ids = sorted(id for id in ids if id in valid_ids)
                if len(cur_ids) >= 2:
                    groups[label] = cur_ids
            labels = sorted(groups.keys())
            counts = [len(groups[label]) for label in labels]
            # we want to sample from each group in proportion to its size, but at least 2 from each
            sample = Counter(random.sample(labels, min(n_close, sum(counts)), counts=counts))
            for label, n in sample.items():
                n = max(min(n, len(groups[label])), 2)
                ids.update(random.sample(groups[label], n))
        if len(ids) < n_pts:
            remaining_ids = set(valid_ids) - ids
            ids.update(random.sample(sorted(remaining_ids), n_pts - len(ids)))
        # at this point we should have all our ids
        ids = sorted(ids)
        id_indices, sub_keys, sub_matrix = self.get_matching_matrix(keys, matrix, ids=ids)
        op_logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
        dists = self.compute_all_distances(ids, **kw)
        return dict(
            sub_keys=sub_keys,
            label_distances=dists,
            sub_matrix=sub_matrix,
        )

    def get_pair_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[DistTuple]:
        """Returns `n_pairs` of `(id1, id2, distance)` tuples in sorted order by distance.

        If perc_close >= 0, ensures that proportion of pairs are "close" according to
        the label type's definition of closeness.

        This implementation handles the pair generation and sampling strategy,
        while subclasses implement get_distance() for their specific distance metric.
        """
        # we store pairs as a frozenset pair of ids mapping to distance
        pairs: dict[frozenset[str], float] = {}
        def add_pairs(lst: list[Any], n_requested: int):
            """Samples up to `n_requested` items from `lst` (pairs) and adds them."""
            if not lst:
                return
            pair_indices = random.sample(lst, min(n_requested, len(lst)))
            for id1, id2 in pair_indices:
                if id1 == id2:
                    continue
                spair = frozenset((id1, id2))
                i1, i2 = self.ids.index(id1), self.ids.index(id2)
                dist = self.get_distance(i1, i2)
                pairs[spair] = dist

        n_close = int(n_pairs * perc_close)
        if n_close > 0:
            # First get close pairs by sampling within each label group (since they by definition
            # share at least one label in common)
            cands = []
            for ids in self.by_label().values():
                ids = list(ids)
                cands.extend((id1, id2) for idx, id1 in enumerate(ids) for id2 in ids[idx+1:])
            add_pairs(cands, n_close)
        # Fill remaining with random pairs
        n_remaining = n_pairs - len(pairs)
        if n_remaining > 0:
            # Generate all possible pairs, removing those we've done, and sample from them
            unique_ids = set(self.ids)
            poss = ((id1, id2) for i, id1 in enumerate(unique_ids) for j, id2 in enumerate(unique_ids) if i < j)
            poss = [p for p in poss if frozenset(p) not in pairs]
            add_pairs(poss, n_remaining)
        return self._pairs_to_list(pairs)

    def get_label_arrays(self, keys: list[str], matrix: nparray2d) -> dict[str, Any]:
        """Returns a list of 1d arrays of numeric values using the given `keys` to filter down.

        This checks for overlap between the given `keys` and our `self.ids`, and returns 1 or more
        1d arrays, packed into a numpy 2d array. These might correspond to a different label, or to
        different transformations of our underlying data, but in any case they are given unique
        names.

        It returns a dict with the fillowing keys:
        - `sub_keys` is the list of overlapping keys
        - `label_names` is a list of names, one for each row of `label_arrays`
        - `label_arrays` is a 2d array of values corresponding to each name and the overlapping keys
          - Shape `(len(label_names), len(sub_keys))`
        - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
          - Shape `(len(sub_keys), matrix.shape[1])`
        """
        ret = dict(label_names=[])
        id_indices, ret['sub_keys'], ret['sub_matrix'] = self.get_matching_matrix(keys, matrix)
        # For each specific label value, create +1/-1 array
        label_arrays = []
        for label_name, ids in self.by_label().items():
            binary_array = np.array([1.0 if self.ids[i] in ids else -1.0 for i in id_indices])
            ret['label_names'].append(label_name)
            label_arrays.append(binary_array)
        ret['label_arrays'] = np.array(label_arrays)
        assert len(ret['label_arrays']) == len(ret['label_names'])
        assert len(ret['label_arrays'][0]) == len(ret['sub_keys']) == len(ret['sub_matrix'])
        assert ret['sub_matrix'].shape[1] == matrix.shape[1]
        return ret


class MulticlassLabels(MulticlassBase):
    """A class for multiclass (mutually-exclusive) labels.

    This stores ids as a list and values as a list, where values[i] is the label for ids[i].
    """
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
        ids = [id for id, v in ids_values]
        values = [v for id, v in ids_values]
        types = Counter(type(v) for v in values)
        assert len(types) == 1
        t = type(values[0])
        assert t not in FLOAT_TYPES, f'No floats in multiclass label: {types}'
        super().__init__(tag_type, key, ids=ids, values=values)

    def by_label(self) -> dict[Any, set[str]]:
        """Returns a dictionary mapping each label to the set of ids that have that label."""
        ret = defaultdict(set)
        for id, v in zip(self.ids, self.values):
            ret[v].add(id)
        return dict(ret)

    def get_distance(self, idx1: int, idx2: int, **kw) -> float:
        """Returns distance between two id indices.

        Distance is 0 for same class, 1 for different class.
        """
        return 0.0 if self.values[idx1] == self.values[idx2] else 1.0


class MultilabelLabels(MulticlassBase):
    """A class for multilabel (non-mutually-exclusive) labels.

    This stores ids as a list and values as a dictionary mapping id -> list of labels.
    """
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
        ids = set()
        values = defaultdict(list)
        types = Counter()
        for id, v in ids_values:
            ids.add(id)
            values[id].append(v)
            types[type(v)] += 1
        ids = sorted(ids)
        assert len(types) == 1
        t = type(values[0])
        assert t not in FLOAT_TYPES, f'Not floats in multilabel label: {types}'
        super().__init__(tag_type, key, ids=ids, values=dict(values))

    def by_label(self) -> dict[Any, set[str]]:
        """Returns a dictionary mapping each label to the set of ids that have that label."""
        ret = defaultdict(set)
        for id, vs in self.values.items():
            for v in vs:
                ret[v].add(id)
        return dict(ret)

    def get_distance(self, idx1: int, idx2: int, **kw) -> float:
        """Returns distance between two id indices.

        Distance is Jaccard distance: 1 - |intersection|/|union|.
        """
        set1 = set(self.values[self.ids[idx1]])
        set2 = set(self.values[self.ids[idx2]])
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)


def parse_into_labels(tag_type: str,
                      key: str,
                      ids_values: list[tuple[str, Any]],
                      impure_thresh=0.1) -> Labels|None:
    """Parses our (id, value) pairs into a Labels object of the appropriate type."""
    ids = [id for id, v in ids_values]
    values = [v for id, v in ids_values]
    types = Counter(type(v) for v in values)
    most_t, n_most = types.most_common(1)[0]
    op_logger.debug(f'For {(tag_type, key)} got {len(ids)} ids, types: {types.most_common()}')
    # if we have less than `impure_thresh` of other types, ignore them
    if len(types) > 1:
        impure = 1.0 - (n_most / len(ids))
        op_logger.debug(f'  Most common (purity): {n_most}/{len(ids)} -> {impure}')
        if impure < impure_thresh:
            new_ids_values = [(id, v) for id, v in ids_values if type(v) == most_t]
            return parse_into_labels(tag_type, key, new_ids_values, impure_thresh=impure_thresh)
        else:
            raise NotImplementedError(f'Cannot handle mixed types: {types.most_common()}')
    # at this point we should have exactly one type
    #print(f'Got {len(ids)} ids, {len(set(ids))} unique ids, type: {most_t}')
    if len(set(ids)) != len(ids): # we have duplicate ids
        # check for impurity level
        impure = 1.0 - (len(set(ids)) / len(ids))
        op_logger.debug(f'  Multilabel impurity {impure}: {len(set(ids))}/{len(ids)}')
        if impure < impure_thresh:
            seen_ids = set()
            new_ids_values = []
            for id, v in ids_values:
                if id in seen_ids:
                    continue
                seen_ids.add(id)
                new_ids_values.append((id, v))
            ids_values = new_ids_values
        else:
            return MultilabelLabels(tag_type, key, ids_values)
    if most_t in NUMERIC_TYPES: # numeric
        return NumericLabels(tag_type, key, ids_values)
    else: # categorical
        if len(set(values)) == len(values):
            op_logger.debug(f'  All values unique, treating as id')
            return None
        return MulticlassLabels(tag_type, key, ids_values)


class StartValidatorOp(Op):
    """Starting point for running embeddings validation.

    Passes through kw version of parsed args from ArgumentParser.
    """
    name = 'start_validator'
    input_types = set()
    output_types = {"argparse"}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any], **kwargs) -> Any:
        return inputs


class ParseTagsOp(Op):
    """Parses our tags from the tag db"""
    name = 'parse_tags'
    input_types = {'argparse'}
    output_types = {'labels'}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any], **kwargs) -> Any:
        tag_path = inputs['argparse'].get('tag_path')
        if not tag_path:
            raise ValueError('No tag_path provided')
        tag_db = init_tag_db(tag_path)
        grouped = defaultdict(list)
        with db_session:
            # get all tags, group by (type, key)
            tags = Tag.select()
            for t in Tag.select():
                key = (t.type, t.key)
                v = specialize(t.value)
                grouped[key].append((t.id, v))
        op_logger.info(f'Loaded {len(grouped)} types of tags from {tag_path}')
        labels = {}
        for (tag_type, key), ids_values in grouped.items():
            if key == 'title':
                continue
            if cur := parse_into_labels(tag_type, key, ids_values):
                labels[key] = cur
        return labels

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        ret = {}
        labels = results
        ret = {}
        for name, label in labels.items():
            ret[name] = dict(
                tag_type=label.tag_type,
                n_ids=len(label.ids),
                n_unique_ids=len(set(label.ids)),
                norm_type=getattr(label, 'norm_type', None),
                n_unique_values=len(set(label.values)) if isinstance(label.values, list) else None,
            }
        return ret


class LoadEmbeddingsOp(Op):
    """Load embeddings from paths into a FeatureSet."""
    name = 'load_embeddings'
    input_types = {'argparse'}
    output_types = {"feature_set"}
    is_intermediate = True

    #TODO return cartesian product of inputs as variants
    def _execute(self, inputs: dict[str, Any], **kwargs) -> Any:
        paths = inputs['argparse']['paths']
        return FeatureSet(paths, **kwargs)

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        fs: FeatureSet = results
        dims = Counter()
        for key, emb in fs.items():
            dims[len(emb)] += 1
        return dict(
            n_embeddings=len(fs),
            dimension_counts=dict(dims),
        )

class CheckDimensionsOp(Op):
    """Check that all embeddings have consistent dimensions."""
    #run_mode = 'process'
    name = "check_dimensions"
    input_types = {"feature_set"}
    output_types = {"dimension_check_result"}

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        dims = Counter()
        for key, emb in fs.items():
            dims[len(emb)] += 1
        return {
            "dimension_counts": dict(dims),
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        dims = Counter(results['dimension_counts'])
        return dict(
            is_consistent=(len(dims) == 1),
            error_message: None if is_consistent else f"Inconsistent embedding dimensions: {dims.most_common()}",
        )

class CheckNaNsOp(Op):
    """Check for NaN values in embeddings."""

    #run_mode = 'process'
    name = "check_nans"
    input_types = {"feature_set"}
    output_types = {"nan_check_result"}

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        n_nans = 0
        nan_keys = []
        for key, emb in fs.items():
            key_nans = np.sum(np.isnan(emb))
            n_nans += key_nans
            if key_nans > 0:
                nan_keys.append((key, int(key_nans)))
        return {
            "total_nans": int(n_nans),
            "nan_keys": nan_keys,
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        n_nans = results['total_nans']
        return dict(
            has_nans=n_nans > 0,
            error_message=None if n_nans == 0 else f"Found {n_nans} NaNs in embeddings",
        )



class BasicChecksOp(Op):
    """Combine dimension and NaN checks into a single basic validation."""

    name = "basic_checks"
    input_types = {"dimension_check_result", "nan_check_result"}
    output_types = {"basic_checks_report"}
    #run_mode = 'process'

    def _execute(self, inputs: dict[str, Any]) -> Any:
        dim_result = inputs["dimension_check_result"]
        nan_result = inputs["nan_check_result"]
        errors = []
        if not dim_result["is_consistent"]:
            errors.append(dim_result["error_message"])
        if nan_result["has_nans"]:
            errors.append(nan_result["error_message"])
        return {
            "errors": errors,
            "dimension_check": dim_result,
            "nan_check": nan_result
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        errors = results.get('errors', [])
        return dict(passed= (len(errors) == 0))


class NormalizeOp(Op):
    """Normalize embeddings from a FeatureSet based on normalization parameters.

    Note that this takes 'labels' as input so that we filter down to keys that have any labels,
    """

    name = "normalize"
    input_types = {"feature_set", "labels"}
    output_types = {"normalized_embeddings"}
    is_intermediate = True
    run_mode = 'main'

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants of this op based on normalization options."""
        return None #FIXME
        ret = {}
        for normed, scale_mean, scale_std in product([True, False], repeat=3):
            variant_name = f"normed:{int(normed)}_mean:{int(scale_mean)}_std:{int(scale_std)}"
            ret[variant_name] = {
                "normed": normed,
                "scale_mean": scale_mean,
                "scale_std": scale_std
            }
        return ret

    def __init__(self, normed: bool = False, scale_mean: bool = True, scale_std: bool = True, **kw):
        self.normed = normed
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs, labels = inputs["feature_set"], inputs['labels']
        valid_keys = set()
        for label in labels.values():
            valid_keys.update(label.ids)
        keys, emb = fs.get_keys_embeddings(
            keys=sorted(valid_keys),
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
        )
        op_logger.info(f'from {len(valid_keys)} got {len(keys)} embeddings {emb.shape}')
        return (keys, emb)

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        keys, emb = results
        return dict(
            variant=self.variant,
            n_embeddings=len(keys),
            embedding_shape=emb.shape,
        )


class LabelOp(Op):
    """A label op is an abstract class that defines one variant per label_key"""

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants of this op, one per label key."""
        labels = inputs.get("labels", {})
        assert labels, 'Must have some labels!'
        ret = {}
        for key in labels:
            variant_name = f"label_key:{key}"
            ret[variant_name] = {"label_key": key}
            if len(ret) > 3: #FIXME temporary
                break
        op_logger.info(f'Got {len(ret)} variants for {cls.name}: {labels}, {ret}')
        return ret

    def __init__(self, label_key: str, **kw):
        self.label_key = label_key
        super().__init__(**kw)


class GetLabelArraysOp(LabelOp):
    """Extract label arrays from labels, determining the intersection with embeddings.

    Each label key gets its own variant of this op, so we can process each one separately, and
    because they have different intersections with the embeddings. But each label can have 1 or more
    rows of labels (which get stacked together into 2d numpy array), each with their own name.

    Returns a dict with:
    - `label_key`, which is the variant name, corresponding to the label that these arrays are from.
    - `sub_keys` is the list of overlapping keys
    - `label_names` is a list of names, one for each row of `label_arrays`
    - `label_arrays` is a 2d array of values corresponding to each name and the overlapping keys
      - Shape `(len(label_names), len(sub_keys))`
    - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
      - Shape `(len(sub_keys), matrix.shape[1])`
    """
    #enabled = False

    name = "get_label_arrays"
    input_types = {"normalized_embeddings", "labels"}
    output_types = {"label_arrays_data"}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        keys, matrix = inputs["normalized_embeddings"]
        label = inputs["labels"][self.label_key]
        ret = label.get_label_arrays(keys, matrix)
        ret['label_key'] = self.label_key
        return ret

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.get('label_key'),
            n_sub_keys=len(results.get('sub_keys', [])),
            n_label_names=len(results.get('label_names', [])),
            label_array_shape=results.get('label_arrays', []).shape if 'label_arrays' in results else None,
            sub_matrix_shape=results.get('sub_matrix', []).shape if 'sub_matrix' in results else None,
        )

class GetLabelDistancesOp(LabelOp):
    """Generate distance matrices from labels.

    Each label key gets its own variant of this op, so we can process each one separately.

    Returns a dict with:
    - `label_key`: The variant name, corresponding to the label that these distances are from
    - `sub_keys` is the list of overlapping keys used in the distance matrix(rows/cols)
    - `label_distances`: The distance matrix between ids
    - `sub_matrix` is the submatrix of `matrix` corresponding to the overlapping keys.
      - Shape `(len(sub_keys), matrix.shape[1])`
    """
    name = "get_label_distances"
    input_types = {"labels", "normalized_embeddings"}
    output_types = {"label_distances", 'distances'}
    is_intermediate = True

    def __init__(self, label_key: str, n_pts: int = 200, perc_close: float = 0.5, **kw):
        self.label_key = label_key
        self.n_pts = n_pts
        self.perc_close = perc_close
        super().__init__(label_key=label_key, **kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        label = inputs["labels"][self.label_key]
        keys, matrix = inputs["normalized_embeddings"]
        ret = label.get_all_distances(
            n_pts=self.n_pts,
            keys=keys,
            matrix=matrix,
            perc_close=self.perc_close,
        )
        ret['distances'] = ret['label_distances']
        ret['label_key'] = self.label_key
        return ret

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.get('label_key'),
            n_sub_keys=len(results.get('sub_keys', [])),
            label_distances_shape=results.get('label_distances', []).shape if 'label_distances' in results else None,
            sub_matrix_shape=results.get('sub_matrix', []).shape if 'sub_matrix' in results else None,
        )

class GetEmbeddingDimsOp(Op):
    """Extract embedding dimensions from filtered label data for consistency.

    Uses the keys and matrix from label_arrays_data to ensure both arrays
    have the same samples in the same order.
    """
    #enabled = False
    name = "get_embedding_dims"
    input_types = {"label_arrays_data"}
    output_types = {"embedding_dims"}
    is_intermediate = True

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants for embedding dimension processing."""
        ret = {
            "raw": {"transform": "raw"},
            #"log": {"transform": "log"},
        }
        return ret

    def __init__(self, transform: str = "raw", **kw):
        self.transform = transform
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        label_data = inputs["label_arrays_data"]
        matrix = label_data["sub_matrix"]  # Already filtered to the right intersection
        dims = matrix.T  # Each row is one dimension across all samples
        # Apply transformation based on variant
        if self.transform == "log":
            # Apply log transform, handling negative values
            dims = np.log1p(np.abs(dims))
        # "raw" needs no transformation
        return dict(dims_matrix=dims, transform=self.transform, label_key=label_data["label_key"])

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.get('label_key'),
            dims_shape=results.get('dims_matrix', []).shape if 'dims_matrix' in results else None,
            transform=results.get('transform'),
        )


class GetEmbeddingDistancesOp(Op):
    """Generate distance matrices from embeddings using various distance metrics.

    Takes a list of ids (from label_distances) and normalized embeddings, and computes distance
    matrices using different metrics (cosine, dot product, etc.)
    """
    name = "get_embedding_distances"
    input_types = {"normalized_embeddings", "label_distances"}
    output_types = {"embedding_distances", 'distances'}
    run_mode = "process"
    is_intermediate = True

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants for distance metrics."""
        metrics = ['cosine', 'dot_product', 'euclidean']
        metrics = ['cosine', 'dot_product'] #TODO 'euclidean' is too similar to cosine
        label_key = inputs['label_distances']['label_key']
        return {f'label:{label_key}_metric:{metric}': dict(metric=metric) for metric in metrics}

    def __init__(self, metric, **kw):
        self.metric = metric
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        keys, matrix = inputs["normalized_embeddings"]
        label_data = inputs["label_distances"]

        # Get the ids from label distances to ensure consistency
        ret = dict(label_key=label_data["label_key"], metric=self.metric, sub_keys=label_data["sub_keys"])
        matrix = label_data["sub_matrix"]  # Already filtered to the right intersection
        op_logger.info(f'Computing {self.metric} dists for label {label_data["label_key"]} with {matrix.shape} matrix')

        # Compute distance matrix based on metric
        if self.metric == "cosine":
            distances = squareform(pdist(matrix, 'cosine'))
        elif self.metric == "dot_product":
            # Dot product similarity (convert to distance by 1-sim)
            distances = 1.0 - (matrix @ matrix.T)
        elif self.metric == "euclidean":
            distances = squareform(pdist(matrix, 'euclidean'))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        ret['distances'] = ret['embedding_distances'] = distances
        return ret

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.get('label_key'),
            metric=results.get('metric'),
            n_sub_keys=len(results.get('sub_keys', [])),
            distances_shape=results.get('distances', []).shape if 'distances' in results else None,
        )


class GetNeighborsOp(Op):
    """Generate nearest neighbors from distance matrices.

    Takes a distance matrix and computes the K nearest neighbors for each point.

    Returns a dict with:
    - `distance_type`: The type of distance matrix used ('label' or 'embedding')
    - `label_key`: The label key this is for
    - `metric`: The distance metric used (for embedding distances)
    - `neighbors`: A 2D array of neighbor indices, shape (n_samples, K)
    - `distances`: A 2D array of distances to neighbors, shape (n_samples, K)
    - `keys`: The list of keys corresponding to the rows/columns of the distance matrix
    """
    #enabled = False
    name = "get_neighbors"
    input_types = {'distances'}
    output_types = {"neighbors_data"}
    #run_mode = "process"
    is_intermediate = True

    def __init__(self, k: int = 20, **kw):
        """Initialize with number of neighbors to compute.

        - `k`: Number of nearest neighbors to compute (default: 20)
        """
        self.k = k
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Determine which type of distance matrix we're using
        data = inputs['distances']
        op_logger.info(f'Computing neighbors, got data {data.keys()}, k={self.k}')
        distances = data["distances"]  # Get the actual distance matrix
        # clamp distances to be positive
        distances = np.maximum(distances, 0.0)
        keys = data["sub_keys"]
        if "label_distances" in data:
            distance_type = "label"
            metric = None
        elif "embedding_distances" in data:
            distance_type = "embedding"
            metric = data.get("metric")
        else:
            raise ValueError("Distance data must contain either 'label_distances' or 'embedding_distances'")

        # Ensure we don't request more neighbors than we have points
        k = min(self.k + 1, distances.shape[0])  # +1 because the point itself is included

        # Use sklearn's NearestNeighbors with precomputed distances
        nn_cls = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nn_cls.fit(distances)
        neighbor_dists, neighbor_indices = nn_cls.kneighbors(distances)
        return {
            "distance_type": distance_type,
            "label_key": data.get("label_key"),
            "metric": metric,
            "neighbors": neighbor_indices,
            "distances": neighbor_dists,
            "keys": keys,
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            distance_type=results.get('distance_type'),
            label_key=results.get('label_key'),
            metric=results.get('metric'),
            n_keys=len(results.get('keys', [])),
            neighbors_shape=results.get('neighbors', []).shape if 'neighbors' in results else None,
            distances_shape=results.get('distances', []).shape if 'distances' in results else None,
        )


class GenPredictionTasksOp(LabelOp):
    """Generate prediction tasks from labels.

    For numeric labels, generates original values and log/exp transformations.
    For multiclass labels, generates original values and binarized versions for each class.
    For multilabel labels, generates binarized versions for each label.

    Returns a dict with:
    - `label_key`: The label key these tasks are for
    - `tasks`: Dict mapping task name to tuple of (label array, task type)
    - `sub_keys`: List of keys used in the tasks
    - `sub_matrix`: Submatrix of embeddings corresponding to the keys
    """
    name = "gen_prediction_tasks"
    input_types = {"labels", "normalized_embeddings"}
    output_types = {"prediction_tasks"}
    #run_mode = "process"
    is_intermediate = True

    def __init__(self, label_key: str, min_pos: int = 10, max_tasks: int = 10, **kw):
        """Initialize with parameters for task generation.

        - `label_key`: The label key to generate tasks for
        - `min_pos`: Minimum number of positive examples for classification tasks
        - `max_tasks`: Maximum number of tasks to generate
        """
        self.min_pos = min_pos
        self.max_tasks = max_tasks
        super().__init__(label_key=label_key, **kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        label = inputs["labels"][self.label_key]
        keys, matrix = inputs["normalized_embeddings"]

        # Get intersection of keys with label ids
        valid_ids = [id for id in label.ids if id in keys]
        id_indices = [keys.index(id) for id in valid_ids]
        sub_matrix = matrix[id_indices]
        tasks: PTasks = {}
        if isinstance(label, NumericLabels):
            # For numeric labels, generate original, log, and exp transformations
            values = np.array([label.values[label.ids.index(id)] for id in valid_ids])
            tasks['orig-num'] = (values, 'regression')
            # Log transformation (shift to positive)
            tasks['log'] = (np.log1p(values - np.min(values) + 1.0), 'regression')
            # Exp transformation
            tasks['exp'] = (np.expm1(values - np.min(values) + 1.0), 'regression')
        elif isinstance(label, MulticlassLabels):
            # For multiclass labels, generate original and binarized versions
            values = np.array([label.values[label.ids.index(id)] for id in valid_ids])
            tasks['orig-cls'] = (values, 'classification')

            # Generate binary tasks for each class
            counts = Counter(values)
            for v, _ in counts.most_common(self.max_tasks):
                bin_values = np.array([int(val == v) for val in values])
                if np.sum(bin_values) >= self.min_pos:
                    tasks[f'binarized-{v}'] = (bin_values, 'classification')
        elif isinstance(label, MultilabelLabels):
            # For multilabel labels, generate binarized versions for each label
            counts = Counter()
            for id in valid_ids:
                counts.update(label.values.get(id, []))

            for v, _ in counts.most_common(self.max_tasks):
                bin_values = np.array([int(v in label.values.get(id, [])) for id in valid_ids])
                if np.sum(bin_values) >= self.min_pos:
                    tasks[f'binarized-{v}'] = (bin_values, 'classification')
        return {
            "label_key": self.label_key,
            "tasks": tasks,
            "sub_keys": valid_ids,
            "sub_matrix": sub_matrix
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.get('label_key'),
            n_tasks=len(results.get('tasks', {})),
            n_sub_keys=len(results.get('sub_keys', [])),
            sub_matrix_shape=results.get('sub_matrix', []).shape if 'sub_matrix' in results else None,
            task_counts={k: v[1] for k, v in results.get('tasks', {}).items()},
        )


class RunPredictionOp(Op):
    """Run prediction models on tasks generated by GenPredictionTasksOp.

    Runs different models based on task type (regression or classification).
    Uses cross-validation to evaluate model performance.

    Returns a dict with:
    - `label_key`: The label key these predictions are for
    - `task_name`: The specific task name
    - `model_name`: The model used
    - `score`: The evaluation score (R² for regression, balanced accuracy for classification)
    - `score_type`: The type of score ('r2' or 'balanced_accuracy')
    - `n_classes`: Number of classes (for classification tasks)
    - `predictions`: Cross-validated predictions
    - `true_values`: True values
    """
    name = "run_prediction"
    input_types = {"prediction_tasks"}
    output_types = {"prediction_results"}
    run_mode = "process"

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants for model types and tasks."""
        tasks = inputs["prediction_tasks"]["tasks"]
        label_key = inputs["prediction_tasks"]["label_key"]

        variants = {}
        for task_name, (_, task_type) in tasks.items():
            if task_type == 'regression':
                models = {
                    "ridge": {"model_type": "ridge"},
                    "rbf_svr": {"model_type": "rbf_svr"},
                    "linear_svr": {"model_type": "linear_svr"},
                    "knn_reg": {"model_type": "knn_reg"}
                }
            else:  # classification
                models = {
                    "rbf_svm": {"model_type": "rbf_svm"},
                    "linear_svm": {"model_type": "linear_svm"},
                    "knn_cls": {"model_type": "knn_cls"}
                }

            for model_name, model_params in models.items():
                variant_name = f"label:{label_key}_task:{task_name}_model:{model_name}"
                variants[variant_name] = {
                    "task_name": task_name,
                    "model_type": model_params["model_type"]
                }

        return variants

    def __init__(self, task_name: str, model_type: str, n_splits: int = 4, **kw):
        """Initialize with task and model parameters.

        - `task_name`: The specific task to run
        - `model_type`: The model type to use
        - `n_splits`: Number of cross-validation splits
        """
        self.task_name = task_name
        self.model_type = model_type
        self.n_splits = n_splits
        super().__init__(**kw)

    def _get_model(self):
        """Get the appropriate model based on model_type."""
        match self.model_type:
            case "ridge":
                return Ridge(alpha=1.0)
            case "rbf_svr":
                return SVR(kernel='rbf', C=1.0, epsilon=0.1)
            case "linear_svr":
                return LinearSVR(C=1.0, epsilon=0.1, dual='auto')
            case "knn_reg":
                return KNeighborsRegressor(n_neighbors=10)
            case "rbf_svm":
                return SVC(kernel='rbf', C=1.0, probability=True)
            case "linear_svm":
                return LinearSVC(C=1.0, max_iter=200, dual='auto')
            case "knn_cls":
                return KNeighborsClassifier(n_neighbors=10)
            case _:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task_data = inputs["prediction_tasks"]
        label_key, tasks = task_data["label_key"], task_data['tasks']

        assert self.task_name in tasks, f"PTask {self.task_name} not found in available tasks"

        X = task_data["sub_matrix"]
        y, task_type = tasks[self.task_name]

        # Get appropriate model and cross-validation strategy
        model = self._get_model()
        is_regression = task_type == 'regression'
        cv_cls = KFold if is_regression else StratifiedKFold
        cv = cv_cls(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Run cross-validation
        all_preds = []
        all_true = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            try:
                preds = model.predict(X_test)
            except:
                # Some models like LinearSVC don't have predict_proba
                if hasattr(model, "decision_function"):
                    decisions = model.decision_function(X_test)
                    preds = (decisions > 0).astype(int)
                else:
                    raise

            all_preds.extend(preds)
            all_true.extend(y_test)

        # Calculate score
        if is_regression:
            score = r2_score(all_true, all_preds)
            score_type = 'r2'
            n_classes = None
        else:
            score = balanced_accuracy_score(all_true, all_preds)
            score_type = 'balanced_accuracy'
            n_classes = len(np.unique(y))

        return {
            "label_key": label_key,
            "task_name": self.task_name,
            "model_name": self.model_type,
            "score": float(score),
            "score_type": score_type,
            "n_classes": n_classes,
            "predictions": np.array(all_preds),
            "true_values": np.array(all_true)
        }

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes prediction results and identifies notable outcomes.
        
        Checks if the prediction score exceeds a threshold (0.7 by default) and
        generates appropriate warnings for high-performing models.
        """
        threshold = 0.7  # Score threshold for highlighting results
        
        # Extract key information from results
        score = results.get("score", 0.0)
        score_type = results.get("score_type", "")
        label_key = results.get("label_key", "")
        task_name = results.get("task_name", "")
        model_name = results.get("model_name", "")
        n_classes = results.get("n_classes")
        
        # Create analysis dict
        analysis = {
            "score": score,
            "score_type": score_type,
            "warnings": []
        }
        
        # Check if score exceeds threshold
        if score > threshold:
            warning = {
                "unit": "prediction",
                "key": label_key,
                "task": task_name,
                "method": model_name,
                "value": score,
                "n_classes": n_classes,
                "score": 3,  # Importance score
                "warning": f"High prediction {score_type} {score:.3f} for {label_key} using {model_name}"
            }
            analysis["warnings"].append(warning)
            
        return analysis




class CompareNeighborsOp(Op):
    """Compare nearest neighbors from different distance metrics.

    Takes two sets of neighbors (typically from label distances and embedding distances)
    and computes comparison metrics like recall@K, MRR@K, and Jaccard similarity.

    Returns a dict with:
    - `label_key`: The label key this comparison is for
    - `embedding_metric`: The embedding distance metric used
    - `metrics`: Dict of metrics computed (recall@K, MRR@K, jaccard@K for different K values)
    - `per_item_metrics`: Optional detailed metrics for each item
    """
    #enabled = False
    name = "compare_neighbors"
    input_types = {
        ("neighbors_data", "neighbors_data"): {
            "consistency_fields": ["label_key"]
        }
    }
    output_types = {"neighbor_comparison"}

    @classmethod
    def get_variants(cls, inputs: dict[str, Any], k_values: list[int]=[1, 5, 10, 20]) -> dict[str, Any]|None:
        """Returns different variants for K values to compare."""
        return {f"k:{k}": {"k": k} for k in k_values}

    def __init__(self, k: int, detailed: bool = True, **kw):
        """Initialize with K value to use for comparison.

        - `k`: Number of neighbors to compare
        - `detailed`: Whether to include per-item metrics (default: False)
        """
        self.k = k
        self.detailed = detailed
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Find which input is label neighbors and which is embedding neighbors
        neighbors_a, neighbors_b = list(inputs["neighbors_data"])

        if 0: #TODO actually i think we wanted to compare all distances against each other?
            if neighbors_a["distance_type"] == "label" and neighbors_b["distance_type"] == "embedding":
                label_neighbors = neighbors_a
                embedding_neighbors = neighbors_b
            elif neighbors_b["distance_type"] == "label" and neighbors_a["distance_type"] == "embedding":
                label_neighbors = neighbors_b
                embedding_neighbors = neighbors_a
            else:
                raise ValueError(f"Need one label and one embedding neighbors, got {neighbors_a['distance_type']} and {neighbors_b['distance_type']}")

        # Get the neighbor indices
        l_nn = neighbors_a["neighbors"]
        m_nn = neighbors_b["neighbors"]
        op_logger.info(f'In CompareNeighborsOp with {neighbors_a.keys()} and {neighbors_b.keys()}, k={self.k}, shapes {l_nn.shape}, {m_nn.shape}')
        #TODO clamp values up to 0?

        # Compute metrics for different K values
        metrics = {}
        per_item_metrics = defaultdict(list) if self.detailed else None

        # Use min of requested K and available neighbors
        max_k = min(self.k, min(l_nn.shape[1], m_nn.shape[1]) - 1)
        k_values = [min(k, max_k) for k in [1, 5, 10, 20] if k <= max_k]

        for idx in range(l_nn.shape[0]):
            l_row = l_nn[idx]
            m_row = m_nn[idx]
            # Remove self-references (at index 0)
            if l_row[0] == idx:
                l_row = l_row[1:]
            if m_row[0] == idx:
                m_row = m_row[1:]
            for k in k_values:
                # Compute recall (how many of the label neighbors are in the embedding neighbors)
                l_set = set(l_row[:k])
                m_set = set(m_row[:k])
                recall = len(l_set & m_set) / len(l_set) if l_set else 0.0
                # Compute MRR (Mean Reciprocal Rank)
                mrr = 0.0
                if recall > 0:
                    for rank, nbr in enumerate(m_row[:k], start=1):
                        if nbr in l_set:
                            mrr = 1.0 / rank
                            break
                # Compute Jaccard similarity
                jaccard = len(l_set & m_set) / len(l_set | m_set) if (l_set | m_set) else 0.0
                # Store per-item metrics if detailed
                if self.detailed:
                    per_item_metrics[f"recall@{k}"].append(recall)
                    per_item_metrics[f"mrr@{k}"].append(mrr)
                    per_item_metrics[f"jaccard@{k}"].append(jaccard)
                # Update running averages
                metrics.setdefault(f"recall@{k}", []).append(recall)
                metrics.setdefault(f"mrr@{k}", []).append(mrr)
                metrics.setdefault(f"jaccard@{k}", []).append(jaccard)
        # Calculate averages
        avg_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}
        result = dict(
            label_key=neighbors_a["label_key"],
            embedding_metric_a=neighbors_a.get('metric'),
            embedding_metric_b=neighbors_b.get('metric'),
            metrics=avg_metrics,
        )
        if self.detailed:
            result["per_item_metrics"] = dict(per_item_metrics)
        return result


class CompareStatsOp(Op):
    """Compare various statistics between cartesian product of rows from two 2D arrays.

    Returns a dict with:
    - stats_a: list of stats dicts for each row in array A
    - stats_b: list of stats dicts for each row in array B
    - shape_a: shape of array A
    - shape_b: shape of array B
    - comparisons: dict mapping (i,j) to comparison stats between row i of A and row j of B
    - n_comparisons: total number of comparisons made
    """
    #enabled = False
    name = "compare_stats"
    input_types = {
        ('many_array1d_a', 'many_array1d_b'): {},
        ("label_arrays_data", "embedding_dims"): {
            "consistency_fields": ["label_key"]
        },
        ("label_distances", "embedding_distances"): {
            "consistency_fields": ["label_key"]
        },
    }
    output_types = {"stats_comparison"}
    #run_mode = 'process'

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if 'many_array1d_a' in inputs:
            arrays_a = inputs['many_array1d_a']
        elif 'label_arrays_data' in inputs:
            arrays_a = inputs['label_arrays_data']['label_arrays']
        elif 'label_distances' in inputs:
            arrays_a = inputs['label_distances']['label_distances']
            #iu = np.triu_indices(label_dists.shape[0], k=1); label_dists=label_dists #TODO?
            if 0:
                plots = self.make_plots(label_dists, m, a_name=f'Label dists for {key}', b_name=f'{k} dists for {key}')
                # display each plot interactively
                for name, plot in plots.items():
                    plt.figure(plot.number)
                    plt.show()
        else:
            raise NotImplementedError(f'Cannot handle inputs {inputs.keys()} for array A')
        if 'many_array1d_b' in inputs:
            arrays_b = inputs['many_array1d_b']
        elif 'embedding_dims' in inputs:
            arrays_b = inputs['embedding_dims']['dims_matrix']
        elif 'embedding_distances' in inputs:
            arrays_b = inputs['embedding_distances']['embedding_distances']
        assert arrays_a.ndim == 2
        assert arrays_b.ndim == 2
        assert arrays_a.shape[1] == arrays_b.shape[1], f'Arrays must have same number of columns, got {arrays_a.shape} vs {arrays_b.shape}'
        ret = dict(stats_a=get_array1d_stats(arrays_a),
                   stats_b=get_array1d_stats(arrays_b),
                   shape_a=arrays_a.shape,
                   shape_b=arrays_b.shape,
                   comparisons={})
        # compare cartesian product of rows
        for i, a in enumerate(arrays_a):
            for j, b in enumerate(arrays_b):
                ret['comparisons'][(i,j)] = compare_array1d_stats(
                    a, b, stats_a=ret['stats_a'][i], stats_b=ret['stats_b'][j]
                )
        ret['n_comparisons'] = len(ret['comparisons'])
        return ret

def init_logging(log_names=('tasks', 'perf', 'op', 'results', 'errors', 'eval'),
                 stderr_loggers=('op', 'errors', 'eval'),
                 file_mode='w'): #FIXME change this to 'a'
    """initializes all our logging

    Args:
        log_names: Names of all loggers to initialize
        stderr_loggers: Names of loggers that should also write to stderr
    """
    fmt = '\t'.join(['%(asctime)s', '%(levelname)s', '%(name)s', '%(process)d:%(thread)d', '%(module)s:%(lineno)d', '%(funcName)s'])+'\n%(message)s\n'
    #logging.basicConfig(format=fmt, level=logging.INFO)

    # Configure specialized loggers
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(fmt)

    # Configure each logger
    for name in log_names:
        logger = logging.getLogger(f"evaluator.{name}")
        logger.setLevel(logging.INFO)

        # Always add a file handler
        file_handler = logging.FileHandler(f"{log_dir}/{name}.log", mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Add stderr handler for specified loggers
        if name in stderr_loggers:
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            logger.addHandler(stderr_handler)


if __name__ == '__main__':
    init_logging()
    parser = ArgumentParser(description='Embeddings evaluator')
    parser.add_argument('paths', nargs='+', help='Paths to the embeddings lmdb file')
    parser.add_argument('-t', '--tag_path', help='Path to the tags sqlite db')
    args = parser.parse_args()
    om = OpManager.get()
    om.start(StartValidatorOp, vars(args))
    for r in om._results.values():
        result_logger.info(f"Result: {r.key} - {r.op.name}")
        #result_logger.info(pformat(make_jsonable(result.to_dict())))
