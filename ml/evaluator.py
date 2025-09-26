"""A generic embeddings evaluator.

This does a bunch of semi-automated things to test out embeddings and see how good they are.
This is mostly in reference to a set of labels you provide, in the form of a tags database.

TODO:

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

from argparse import ArgumentParser
from collections.abc import Mapping
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import abspath, dirname, exists, join
from pprint import pprint as _pprint
from typing import Any, Sequence, Generic, TypeVar, Union

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
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.ml.evaluator_ops import Op, OpManager
from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
    FeatureSet,
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)
from nkpylib.ml.tag_db import Tag, get_all_tags, init_tag_db

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)

# get console width from system
CONSOLE_WIDTH = os.get_terminal_size().columns

# pprint should use full width
pprint = lambda x: _pprint(x, width=CONSOLE_WIDTH)

FLOAT_TYPES = (float, np.float32, np.float64)
NUMERIC_TYPES = FLOAT_TYPES + (int, np.int32, np.int64)

# correlations are returns as a tuple of `(pearson, spearman)`
# where each contains a list of (correlation, dim) tuples
Correlations = tuple[list[float, int], list[tuple[float, int]]]

# a distance tuple has (id1, id2, distance)
DistTuple = tuple[str, str, float]

# All distances is a tuple of (list of ids, complete distance matrix)
AllDists = tuple[list[str], array2d]

# stats are for now just a dict of strings
Stats = dict[str, Any]

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
    res = stats.linregress(a, b)
    stats_cmp.update(dict(
        linear_least_square_r2=float(res.rvalue)**2.0,
    ))
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


class Labels:
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

    def get_all_distances(self, n_pts: int, perc_close: float = -1, **kw) -> AllDists:
        """Returns all pairwise distances between `n_pts` points.

        We try to sample at least `perc_close` points that are "close" according to the label type's
        definition of closeness. < 0 means we don't care about closeness (default).

        Returns a pair of `(list of ids, complete distance matrix)`. The distance matrix rows/cols
        are in the same order as the list of ids.

        This is a naive implementation that ignores `perc_close` and just does random sampling.
        It also computes distances one-by-one using `get_distance()`. It passes all kw to
        `get_distance()`.
        """
        assert n_pts > 1, 'Must have at least 2 points to compute distances'
        if n_pts > len(self.ids):
            n_pts = len(self.ids)
        ids = sorted(random.sample(self.ids, n_pts))
        logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
        dists = self.compute_all_distances(ids, **kw)
        return ids, dists

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

    def get_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[DistTuple]:
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

    def get_matching_matrix(self, keys: list[str], matrix: array2d) -> tuple[array2d, array1d]:
        """Returns matching submatrix based on overlapping keys.

        This does a set intersection between our ids and the given `keys`, and returns a tuple of
        `(sub_matrix, id_indices)`, where `sub_matrix` is the submatrix of `matrix` corresponding
        to the overlapping keys, and `id_indices` is the list of indices into our `self.ids`
        array corresponding to the overlapping keys, so that `sub_matrix[i]` corresponds to
        `self.ids[id_indices[i]]`. The output `sub_matrix` has the same dimensionality as the
        input `matrix`.

        Note that if ids repeat in `self.ids`, this will use the first matching index.
        """
        assert len(keys) == len(matrix)
        # get row indices of our ids in keys and in self
        mat_indices = []
        id_indices = []
        assert len(keys) == len(set(keys)), 'Keys should be unique'
        common = set(keys) & set(self.ids)
        for mat_idx, id in enumerate(keys):
            if id not in common:
                continue
            label_idx = self.ids.index(id)
            mat_indices.append(mat_idx)
            id_indices.append(label_idx)
        logger.debug(f'  Found {len(common)} matching ids in embeddings')
        id_indices = np.asarray(id_indices)
        sub_matrix = matrix[mat_indices, :]
        logger.debug(f'Got sub matrix of shape {sub_matrix.shape}: {sub_matrix}')
        assert sub_matrix.shape == (len(id_indices), matrix.shape[1])
        return sub_matrix, id_indices

    def get_correlations(self, sub_matrix: array2d, sub_labels: array1d, n_top: int=10) -> Correlations:
        """Computes correlations between each dimension of `sub_matrix` and `sub_labels`.

        Returns 2 lists of (correlation, dim) tuples, one using Pearson correlation and the other
        using Spearman correlation.
        """
        def format_results(ret):
            ret = [(corr, dim) for dim, corr in enumerate(ret) if not np.isnan(corr)]
            ret = sorted(ret, key=lambda x: abs(x[0]), reverse=True)[:n_top]
            return ret

        funcs = [
            lambda i: np.corrcoef(sub_matrix[:, i], sub_labels)[0, 1],
            lambda i: stats.spearmanr(sub_matrix[:, i], sub_labels).statistic,
        ]
        ret = [format_results([func(i) for i in range(sub_matrix.shape[1])]) for func in funcs]
        return tuple(ret)

    def check_correlations(self, keys: list[str], matrix: array2d, n_top:int=10) -> Correlations:
        """Checks correlations between our labels and each dimension of the given `matrix`"""
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

    def check_correlations(self, keys: list[str], matrix: array2d, n_top:int=10) -> Correlations:
        """Checks correlations between our labels and each dimension of the given `matrix`.

        The input `keys` should be the list of ids corresponding to the rows of `matrix`.

        Returns 2 lists of upto `n_top` (correlation, dim) tuples, sorted by highest absolute
        correlation, one using Pearson correlation and the other using Spearman correlation.
        The latter is better for ordinal or non-normally-distributed data.
        """
        sub_matrix, id_indices = self.get_matching_matrix(keys, matrix)
        sub_labels = self.values[id_indices]
        logger.debug(f'Got sub labels of shape {sub_labels.shape}: {sub_labels}')
        return self.get_correlations(sub_matrix, sub_labels, n_top=n_top)

    def get_distance(self, idx1: int, idx2: int, norm_type: str='raw', **kw) -> float:
        """Returns distance between two id indices.

        You can specify `norm_type`:
        - 'raw' (default): absolute difference between values
        - 'range': absolute difference divided by (max-min)
        - 'std': absolute difference divided by stddev
        """
        dist = abs(self.values[idx1] - self.values[idx2]) / self.norm_factors[norm_type]
        return dist

    def get_distances(self, n_pairs: int, perc_close: float = -1,
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

    def get_all_distances(self, n_pts: int, perc_close: float = -1, **kw) -> AllDists:
        """Returns all pairwise distances between `n_pts` points.

        We try to sample at least `perc_close` points that are "close" according to the label type's
        definition of closeness. < 0 means we don't care about closeness (default).

        Returns a pair of `(list of ids, complete distance matrix)`. The distance matrix rows/cols
        are in the same order as the list of ids.

        This implementation samples points from the same label groups if `perc_close > 0` to try to
        get points which share at least one label in common, in rough proportion to the size of each
        label group.
        """
        assert n_pts > 1, 'Must have at least 2 points to compute distances'
        if n_pts > len(self.ids):
            n_pts = len(self.ids)
        ids = set()
        n_close = int(n_pts * perc_close)
        if n_close > 0:
            groups = {label: sorted(ids) for label, ids in self.by_label().items() if len(ids) >= 2}
            labels = sorted(groups.keys())
            counts = [len(groups[label]) for label in labels]
            # we want to sample from each group in proportion to its size, but at least 2 from each
            sample = Counter(random.sample(labels, min(n_close, sum(counts)), counts=counts))
            for label, n in sample.items():
                n = max(min(n, len(groups[label])), 2)
                ids.update(random.sample(groups[label], n))
        if len(ids) < n_pts:
            remaining_ids = set(self.ids) - ids
            ids.update(random.sample(sorted(remaining_ids), n_pts - len(ids)))
        ids = sorted(ids)
        logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
        dists = self.compute_all_distances(ids, **kw)
        return ids, dists

    def get_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[DistTuple]:
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

    def check_correlations(self, keys: list[str], matrix: array2d, n_top:int=10) -> dict[str, Correlations]:
        """Checks correlations between our labels and each dimension of the given `matrix`.

        We use the `by_labels` representation, where for each label, we have a set of ids that have
        that label and the rest (implicitly) don't. We then compute a +1/-1 correlation for each
        label against each dimension of the matrix, and return the top `n_top` correlations.

        The input `keys` should be the list of ids corresponding to the rows of `matrix`.

        Returns a dict mapping labels to `(pearson, spearman)` correlations results, each of which
        has upto `n_top` (correlation, dim) tuples, sorted by highest absolute correlation.
        """
        sub_matrix, id_indices = self.get_matching_matrix(keys, matrix)
        ret = {}
        # iterate through each label and construct +1/-1 labels
        for label, ids in self.by_label().items():
            sub_labels = np.array([1.0 if self.ids[i] in ids else -1.0 for i in id_indices])
            logger.debug(f'  {self.key}: Checking label {label} with {len(ids)} ids: {sub_labels}')
            assert sub_labels.shape == (len(id_indices),)
            ret[label] = self.get_correlations(sub_matrix, sub_labels, n_top=n_top)
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
    logger.debug(f'For {(tag_type, key)} got {len(ids)} ids, types: {types.most_common()}')
    # if we have less than `impure_thresh` of other types, ignore them
    if len(types) > 1:
        impure = 1.0 - (n_most / len(ids))
        logger.debug(f'  Most common (purity): {n_most}/{len(ids)} -> {impure}')
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
        logger.debug(f'  Multilabel impurity {impure}: {len(set(ids))}/{len(ids)}')
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
            logger.debug(f'  All values unique, treating as id')
            return None
        return MulticlassLabels(tag_type, key, ids_values)



class EmbeddingsValidator:
    """A class to validate/evaluate embeddings in various ways, semi-automatically."""
    def __init__(self, paths: list[str], tag_path: str, **kw):
        self.paths = paths
        self.fs = FeatureSet(paths, **kw)
        self.labels = {}
        self.parse_tags(tag_path)
        self.msgs = []

    def add_msg(self, **msg) -> None:
        """Adds a message to our internal message list."""
        logger.debug(f'Adding msg {msg}')
        self.msgs.append(msg)

    def init_fig(self):
        plt.figure(figsize=(10, 10))
        plt.grid(True)
        plt.tight_layout()

    def make_plots(self, a: array1d, b: array1d, a_name: str='a', b_name:str='b') -> dict[str, mpl.figure.Figure]:
        """Makes various plots comparing 1D array `a` to `b`.

        Returns a dict mapping plot type to figure.
        """
        ret = {}
        if 0:
            # make a scatter plot of a vs b
            self.init_fig()
            plt.scatter(a, b)
            plt.xlabel(a_name)
            plt.ylabel(b_name)
            plt.title(f'Scatter plot of {a_name} vs {b_name}')
            ret['scatter'] = plt.gcf()
        # make a q-q plot to compare distributions
        self.init_fig()
        stats.probplot(a, dist="norm", plot=plt)
        stats.probplot(b, dist="norm", plot=plt)
        plt.title(f'Q-Q plot of {a_name} vs {b_name}')
        ret['qq'] = plt.gcf()
        return ret

    def check_distances(self, n: int=200, sample_ids:bool=True) -> None:
        """Does various tests based on distances.

        This uses our labels to generate sets of `n` items, and then compute all-pairs distances
        between them. We then compute all-pairs distances on our embeddings, and can the compare
        these against the label distances. We can also compute neighbors from distances and run
        neighbor-based tests.
        """
        fs_keys = set(self.fs.keys())
        for key, label in self.labels.items():
            kw = dict(close_thresh=.4, perc_close=0.5, norm_type='std')
            if sample_ids: # all pairs for a set of ids
                ids, label_dists = label.get_all_distances(n, **kw)
                common_indices = np.array([i for i, id in enumerate(ids) if id in fs_keys])
                if len(common_indices) < 2:
                    continue
                ids = [ids[i] for i in common_indices]
                label_dists = label_dists[np.ix_(common_indices, common_indices)]
                print(f'\nFor {key} got {len(ids)} ids: {ids[:3]}')
                # get embedding matrix to then compute all pairs distances
                keys, emb = self.fs.get_keys_embeddings(keys=ids, normed=False, scale_mean=False, scale_std=False)
                dists = dict(
                    dot_prod=1.0 - (emb @ emb.T),
                    cos_sim=squareform(pdist(emb, 'cosine')),
                    #euc_dist=squareform(pdist(emb, 'euclidean')),
                )
            else: # sample pairs
                pairs = label.get_distances(n, **kw)
                # filter by keys in our embeddings
                pairs = [(id1, id2, dist) for id1, id2, dist in pairs if id1 in fs_keys and id2 in fs_keys]
                if not pairs:
                    continue
                a_keys, b_keys, label_dists = zip(*pairs)
                label_dists = np.asarray(label_dists)
                print(f'\nFor {key} got {len(pairs)} pairs: {pairs[:3]}')
                ids = sorted(set(a_keys) | set(b_keys))
                # get A and B matrix of embeddings to then compute distances
                keys, emb = self.fs.get_keys_embeddings(keys=ids, normed=False, scale_mean=False, scale_std=False)
                A = np.array([emb[keys.index(id)] for id in a_keys])
                B = np.array([emb[keys.index(id)] for id in b_keys])
                logger.debug(f'  {label_dists.shape}, {A.shape}, {B.shape}, {label_dists}, {A}, {B}')
                # compute dot product, cosine similarity, euclidean distance (all small=similar)
                dists = dict(
                    dot_prod=1.0 - np.einsum('ij,ij->i', A, B),
                    cos_sim=1.0 - np.einsum('ij,ij->i', A, B) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)),
                    #euc_dist=np.linalg.norm(A - B, axis=1),
                )
            if np.allclose(dists['dot_prod'], dists['cos_sim'], rtol=1e-2, atol=1e-2):
                del dists['cos_sim']
            # now compare each one to label_dists
            if label_dists.ndim == 2:
                # sample upper triangle without diagonal
                iu = np.triu_indices(label_dists.shape[0], k=1)
                label_dists = label_dists[iu]
            for k, m in dists.items():
                if m.ndim == 2:
                    # sample identically as label_dists
                    m = m[iu]
                print(f'  {k}: {m.shape}: {m}')
                label_stats, dist_stats, cmp_stats = self.compare_stats(label_dists, m)
                for s in cmp_stats:
                    if s not in 'pearson spearman tau kl_div'.split():
                        continue
                    if s in label_stats:
                        print(f'    {s:>10}:\t{label_stats[s]: .3f}\t{dist_stats[s]: .3f}\t{cmp_stats[s]: .3f}')
                    else:
                        print(f'    {s:>10}:\t{cmp_stats[s]: .3f}')
                if 0:
                    plots = self.make_plots(label_dists, m, a_name=f'Label dists for {key}', b_name=f'{k} dists for {key}')
                    # display each plot interactively
                    for name, plot in plots.items():
                        plt.figure(plot.number)
                        plt.show()
            if sample_ids:
                self.check_neighbors(ids, label_dists, dists, key=key)
            #break

    def check_neighbors(self,
                        ids: list[str],
                        label_dists: array1d,
                        dists: dict[str, array1d],
                        key: str,
                        n_neighbors:int = 20,
                        **kw) -> None:
        """Checks neighbors based on distances.

        Given a set of `ids`, the pairwise `label_dists` between them, and a dict of distances
        computed via different methods, we generate upto `n_neighbors` nearest neighbors for each
        method and compare them to the label-based neighbors using various metrics.
        """
        n = len(ids)
        id_indices = {id: i for i, id in enumerate(ids)}
        def upper_tri_to_full(m):
            """Convert upper triangle 1-d array back to full matrix"""
            if m.ndim == 2 and m.shape[0] == m.shape[1]:
                return m
            full = np.zeros((n, n), dtype=label_dists.dtype)
            iu = np.triu_indices(n, k=1)
            full[iu] = m
            full = full + full.T
            return full

        def upper_tri_to_neighbors(m):
            m = upper_tri_to_full(m)
            # clamp values up to 0
            m = np.maximum(m, 0)
            nn_cls = NearestNeighbors(n_neighbors=n_neighbors+1, metric='precomputed')
            nn_cls.fit(m)
            dists, nn = nn_cls.kneighbors(m)
            #print(f'  Got {nn.shape} neighbors, {dists.shape} dists: {nn}, {dists}')
            return dists, nn

        l_dists, l_nn = upper_tri_to_neighbors(label_dists)
        for method, m in dists.items():
            m_dists, m_nn = upper_tri_to_neighbors(m)
            print(f'  [{method}] Comparing neighbors for {key}:')
            # compute recalls, mrr, jaccard
            counts = defaultdict(list)
            for idx, id in enumerate(ids):
                l_row = l_nn[idx]
                if l_row[0] == idx:
                    l_row = l_row[1:]
                m_row = m_nn[idx]
                if m_row[0] == idx:
                    m_row = m_row[1:]
                for k in [1, 5, 10, 20]:
                    recall = recall_score(l_row[:k], m_row[:k], average='micro', zero_division=0)
                    counts[f'recall@{k}'].append(recall)
                    if recall > 0:
                        # compute mrr
                        for rank, nbr in enumerate(m_row[:k], start=1):
                            if nbr in l_row[:k]:
                                counts[f'mrr@{k}'].append(1.0 / rank)
                                break
                    else:
                        counts[f'mrr@{k}'].append(0.0)
                    # compute jaccard
                    set_l = set(l_row[:k])
                    set_m = set(m_row[:k])
                    jaccard = len(set_l & set_m) / len(set_l | set_m) if set_l | set_m else 0.0
                    counts[f'jaccard@{k}'].append(jaccard)
            for k, v in counts.items():
                avg = sum(v) / len(v) if v else 0.0
                print(f'    {k:>10}: {avg:.3f}')
                if avg > 0.5:
                    self.add_msg(unit='neighbors',
                                 key=key,
                                 method=method,
                                 metric=k,
                                 value=avg,
                                 score=3,
                                 warning=f'High neighbor {k}={avg:.3f} for {key} using {method}')

    def test_distances(self) -> None:
        """Quick test to see if distances are working"""
        print(', '.join(self.labels))
        label = self.labels['ml-genre']
        kw = dict(close_thresh=.4, perc_close=0.5, norm_type='std')
        pairs = label.get_distances(20, **kw)
        def print_pairs(pairs):
            n_close = sum(1 for id1, id2, dist in pairs if dist <= kw['close_thresh'])
            print(f'\n{len(pairs)} pairs for label {label.key} ({n_close} close within {kw["close_thresh"]}, {int(len(pairs)*kw["perc_close"])} requested):')
            with db_session:
                for id1, id2, dist in pairs:
                    title1 = Tag.get(key='title', id=id1).value
                    title2 = Tag.get(key='title', id=id2).value
                    if isinstance(label, MultilabelLabels):
                        val1 = ','.join(sorted(label.values[id1]))
                        val2 = ','.join(sorted(label.values[id2]))
                    else:
                        val1 = label.values[label.ids.index(id1)]
                        val2 = label.values[label.ids.index(id2)]
                    #print(f'  {title1} ({id1}) - {title2} ({id2}): {dist:.3f}')
                    print(f'  {title1} ({val1}) - {title2} ({val2}): {dist:.3f}')

        print_pairs(pairs)

    def gen_prediction_tasks(self, ids: list[str], label: Labels, min_pos:int=10, max_tasks:int=10) -> dict[str, array1d]:
        """Generates prediction tasks derived from a set of `ids` and a `Labels` instance.

        For numeric labels, this takes the original values and also generates log- and
        exp-transformed versions of them, and returns all 3 as separate tasks.

        For multiclass labels, this takes the original values and also generates binarized versions
        for each label class. We only generate upto `max_tasks` binarized tasks, choosing the ones
        with the most positive examples (at least `min_pos`).

        We return a dict mapping task name to the array of values (training labels) for each id.
        """
        ret = {}
        if isinstance(label, NumericLabels):
            values = np.array([label.values[label.ids.index(id)] for id in ids])
            ret['orig-num'] = values
            ret['log'] = np.log1p(values - np.min(values) + 1.0)
            ret['exp'] = np.expm1(values - np.min(values) + 1.0)
            #TODO other kinds of normalizations (e.g., currently the values are very large)
            #     - also might want to e.g. std-normalize before applying log/exp?
            #TODO also generate classification tasks
        elif isinstance(label, MulticlassLabels):
            values = np.array([label.values[label.ids.index(id)] for id in ids])
            ret['orig-cls'] = values
            counts = Counter(values)
            for v, _ in counts.most_common(max_tasks):
                bin_values = np.array([int(val == v) for val in values])
                if np.sum(bin_values) >= min_pos:
                    ret[f'binarized-{v}'] = bin_values
        elif isinstance(label, MultilabelLabels):
            # label.values maps from id to a list of labels
            counts = Counter()
            for id in ids:
                counts.update(label.values.get(id, []))
            for v, _ in counts.most_common(max_tasks):
                bin_values = np.array([int(v in label.values.get(id, [])) for id in ids])
                if np.sum(bin_values) >= min_pos:
                    ret[f'binarized-{v}'] = bin_values
        return ret

    def check_prediction(self, n_jobs=10) -> None:
        """Does prediction tests on our labels.

        For each label, we generate a number of prediction tasks. For numeric labels, these are
        regression, and for categorical labels, these are classification tasks. We then run a number
        of different prediction methods on each task, using cross-validation, and report the results.

        For classification tasks, we report balanced accuracy, and for regression tasks, we report
        R^2.
        """
        fs_keys = set(self.fs.keys())
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        for key, label in self.labels.items():
            print(f'\nRunning prediction tests on {key} of type {label.__class__.__name__} with {len(label.ids)} ids')
            assert len(label.ids) == len(set(label.ids))
            method_kw = {} #dict(n_jobs=n_jobs)
            if isinstance(label, NumericLabels):
                methods = dict(
                    ridge=Ridge(alpha=1.0),
                    #forest_reg=RandomForestRegressor(n_estimators=100, **method_kw),
                    rbf_svr=SVR(kernel='rbf', C=1.0, epsilon=0.1),
                    linear_svr=LinearSVR(C=1.0, epsilon=0.1, dual='auto'),
                    neighbors_reg=KNeighborsRegressor(n_neighbors=10, **method_kw),
                )
            else:
                methods = dict(
                    #forest_cls=RandomForestClassifier(n_estimators=100, max_depth=5, **method_kw),
                    rbf_svm=SVC(kernel='rbf', C=1.0, probability=True),
                    linear_svm=LinearSVC(C=1.0, max_iter=200, dual='auto'),
                    neighbors_cls=KNeighborsClassifier(n_neighbors=10, **method_kw),
                )
            # filter ids to those in our embeddings and get embeddings
            ids = [id for id in label.ids if id in fs_keys]
            keys, emb = self.fs.get_keys_embeddings(keys=ids, normed=False, scale_mean=True, scale_std=True)
            if len(keys) < 10:
                print(f'  Only {len(keys)} embeddings, skipping')
                continue
            assert keys == ids
            # generate prediction tasks (different sets of labels) and run each one in our pool
            tasks = self.gen_prediction_tasks(ids, label)
            print(f'  Generated {len(tasks)} tasks for {key} x {len(methods)} methods = {len(tasks)*len(methods)}')
            todo = []
            for task_name, values in tasks.items():
                for method_name, model in methods.items():
                    y = []
                    KF_cls = KFold if isinstance(label, NumericLabels) else StratifiedKFold
                    kf = KF_cls(n_splits=4, shuffle=True)
                    futures = []
                    for train, test in kf.split(emb, values):
                        X_train, X_test = emb[train], emb[test]
                        y_train, y_test = values[train], values[test]
                        y.extend(y_test)
                        #print(f'Submitting task {task_name} x {method_name} with {X_train.shape} train, {X_test.shape} test')
                        futures.append(pool.submit(train_and_predict, model, X_train, y_train, X_test))
                    todo.append((task_name, np.array(y), method_name, futures))
            for task_name, y, method_name, futures in todo:
                s = f'    {task_name} x {method_name} - '
                preds = []
                try:
                    for f in futures:
                            preds.extend(f.result())
                except Exception as e:
                    print(f'{s} FAILED!: {e}')
                    continue
                preds = np.array(preds)
                if isinstance(label, NumericLabels):
                    score = r2_score(y, preds)
                    score_type = 'R^2'
                    print(f'{s} R^2: {score:.3f}')
                    n_classes = None
                else:
                    score = balanced_accuracy_score(y, preds)
                    score_type = 'balanced_accuracy'
                    n_classes = len(set(y))
                    print(f'{s} Balanced Accuracy: {score:.3f} {n_classes} classes')
                if score > 0.7:
                    self.add_msg(unit='prediction',
                                 key=key,
                                 task=task_name,
                                 method=method_name,
                                 value=score,
                                 n_classes=n_classes,
                                 score=3,
                                 warning=f'High prediction {score_type} {score:.3f} for {key} using {method_name}')

    def run(self) -> None:
        logger.info(f'Validating embeddings in {self.paths}, {len(self.fs)}')
        #self.check_distances(n=200, sample_ids=True)
        self.check_prediction()
        return
        # raw correlations
        # norming hurts correlations
        # scaling mean/std doesn't do anything, because correlation is scale-invariant
        keys, emb = self.fs.get_keys_embeddings(normed=False, scale_mean=False, scale_std=False)
        n_top = 10
        if 0:
            # pca correlations
            pca = PCA(n_components=min(n_top, emb.shape[1]))
            trans = pca.fit_transform(emb)
            # if the first pca dimension explained variance is too high, add a message about it
            if pca.explained_variance_ratio_[0] > 0.5:
                self.add_msg(unit='pca', warning=f'PCA first dimension explains too much variance: {pca.explained_variance_ratio_[0]:.3f}', score=-2)
            print(f'PCA with {pca.n_components_} comps, explained variance: {pca.explained_variance_ratio_}: {emb.shape} -> {trans.shape}')

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
        logger.info(f'Loaded {len(grouped)} types of tags from {tag_path}')
        labels = {}
        for (tag_type, key), ids_values in grouped.items():
            if key == 'title':
                continue
            if cur := parse_into_labels(tag_type, key, ids_values):
                labels[key] = cur
        return labels


class CheckDimensionsOp(Op):
    """Check that all embeddings have consistent dimensions."""

    enabled = True
    run_mode = 'process'

    name = "check_dimensions"
    input_types = {"feature_set"}
    output_types = {"dimension_check_result"}

    def _execute(self, inputs: dict[str, Any]) -> Any:
        fs = inputs["feature_set"]
        dims = Counter()
        for key, emb in fs.items():
            dims[len(emb)] += 1
        is_consistent = len(dims) == 1
        return {
            "is_consistent": is_consistent,
            "dimension_counts": dict(dims),
            "error_message": None if is_consistent else f"Inconsistent embedding dimensions: {dims.most_common()}"
        }


class CheckNaNsOp(Op):
    """Check for NaN values in embeddings."""

    enabled = True
    run_mode = 'process'

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
        has_nans = n_nans > 0
        return {
            "has_nans": has_nans,
            "total_nans": int(n_nans),
            "nan_keys": nan_keys,
            "error_message": None if not has_nans else f"Found {n_nans} NaNs in embeddings"
        }


class BasicChecksOp(Op):
    """Combine dimension and NaN checks into a single basic validation."""

    name = "basic_checks"
    input_types = {"dimension_check_result", "nan_check_result"}
    output_types = {"basic_checks_report"}
    run_mode = 'process'

    def _execute(self, inputs: dict[str, Any]) -> Any:
        dim_result = inputs["dimension_check_result"]
        nan_result = inputs["nan_check_result"]
        errors = []
        if not dim_result["is_consistent"]:
            errors.append(dim_result["error_message"])
        if nan_result["has_nans"]:
            errors.append(nan_result["error_message"])
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "dimension_check": dim_result,
            "nan_check": nan_result
        }


class NormalizeOp(Op):
    """Normalize embeddings from a FeatureSet based on normalization parameters."""

    name = "normalize"
    input_types = {"feature_set"}
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
        fs = inputs["feature_set"]
        keys, emb = fs.get_keys_embeddings(
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
        )
        return (keys, emb)


class CheckCorrelationsOp(Op):
    """Check correlations between embeddings and labels."""

    name = "check_correlations"
    input_types = {"normalized_embeddings", "labels"}
    output_types = {"correlation_results"}
    run_mode = 'main'

    def __init__(self, n_top: int = 10, **kw):
        self.n_top = n_top
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> Any:
        (keys, matrix), labels = inputs['normalized_embeddings'], inputs['labels']
        print(f'Checking correlations for {len(labels)} labels on {matrix.shape} embeddings')
        results = dict(by_label_key={}, warnings=[])
        for key, label in labels.items():
            print(f'  Checking correlations for label {label} with {len(label.ids)} ids')
            correlations = label.check_correlations(keys, matrix, n_top=self.n_top)
            results['by_label_key'][key] = correlations
            # Check for high correlations and create warnings
            if not isinstance(correlations, dict):
                correlations = {"all": correlations}
            for label_name, (pearson_list, spearman_list) in correlations.items():
                for method, corr_list in [("pearson", pearson_list), ("spearman", spearman_list)]:
                    for corr, dim in corr_list:
                        if abs(corr) > 0.5:
                            warning = f'High correlation {corr:.3f} for {key}={label_name} at dim {dim}'
                            results["warnings"].append({
                                "unit": "correlation",
                                "label": label_name,
                                "key": key,
                                "dim": dim,
                                "method": method,
                                "value": corr,
                                "score": 3,
                                "warning": warning
                            })
        return results

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
    name = "compare_stats"
    input_types = {"many_array1d_a", "many_array1d_b"}
    output_types = {"stats_comparison"}

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        arrays_a = inputs["many_array1d_a"]  # 2D numpy array, each row is a 1D array
        arrays_b = inputs["many_array1d_b"]  # 2D numpy array, each row is a 1D array
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


if __name__ == '__main__':
    fmt_els = ['%(asctime)s', '%(process)d:%(thread)d', '%(levelname)s', '%(funcName)s', '%(message)s']
    fmt_els = ['%(asctime)s', '%(levelname)s', '%(funcName)s', '%(message)s']
    logging.basicConfig(format='\t'.join(fmt_els), level=logging.INFO)
    parser = ArgumentParser(description='Embeddings evaluator')
    parser.add_argument('paths', nargs='+', help='Paths to the embeddings lmdb file')
    parser.add_argument('-t', '--tag_path', help='Path to the tags sqlite db')
    args = parser.parse_args()
    if 0:
        ev = EmbeddingsValidator(args.paths, tag_path=args.tag_path)
        ev.run()
    else:
        om = OpManager.get()
        om.start(StartValidatorOp, vars(args))
