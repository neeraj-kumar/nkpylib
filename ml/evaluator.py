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
  - Some std methods: linear svm, rbf svm, svr, forests, pytorch nn?
  - look at binary and one-vs-all
  - don't do special multilabel stuff
- For dims, we can also compute histograms
  - And min/max/mean/std/etc
  - Standard stats on histograms: kurtosis, bin sizes, lop-sidedness, normality
  - highlight 0s other outliers
- Compare labels of same type but different keys, e.g. genre
  - Look at confusion matrices
  - Also clustering metrics for label similarity?
  - With small number of labels, no need to embed labels, just do one-hot
- For numerical labels, look at orig values, log(val) and exp(val) where relevant
- Distances between labeled points
  - Actually, we might want to pick a bunch of ids, and then all-pairs distances to get something
    closer to reality
  - For multiclass, we might need embeddings for labels to get distances
  - combine distances across labels
    - Have to be careful about scaling
    - Join in neighbor-space? Distance-space?
    - Might have multiple combined distances to compare
  - For converting distances to neighbors, maybe use KNeighborsTransformer or RadiusNeighborsTransformer
- What to do with distances
  - Probably want to randomly sample pairs (focusing more on neighbors)
  - Each pair has a set of label distances (per label, different aggregations)
  - For each pair, we can also compute embedding distances in various ways
    - euclidean, cosine, manhattan
  - Then compute correlations between label distances and embedding distances
  - Also between different embedding distances (e.g. euclidean vs cosine)
    - Find outliers
  - Visualize full pairwise cosine similarity heatmaps — useful for spotting large dense cliques (bad) or disconnected islands (good/bad, depending).
  - Compute pairwise angles between random vector pairs. For high-quality, high-dimensional embeddings, the distribution should be tightly centered.
  - Can also do classification (near/far)/regression on distances
    - metric learning using triplet or contrastive loss?
  - Also look at neighbor similarity
    - jaccard, precision/recall, rank-biased overlap
  - Compare histograms of distances (from labels vs embeddings)
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
- In the future, do ML doctor stuff


Old stuff:
- Recommendation system
- Few-shot classifier
- Pairwise/triplet-loss task eval
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
import random
import sys
import time

from argparse import ArgumentParser
from collections.abc import Mapping
from collections import Counter, defaultdict
from os.path import abspath, dirname, exists, join
from pprint import pprint as _pprint
from typing import Any, Sequence, Generic, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from pony.orm import * # type: ignore
from scipy.stats import spearmanr # type: ignore
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, KMeans, AgglomerativeClustering, MiniBatchKMeans # type: ignore
from sklearn.decomposition import PCA # type: ignore
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
from nkpylib.ml.tag_db import Tag, get_all_tags, init_tag_db

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

# stats are for now just a dict of strings
Stats = dict[str, Any]


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
        axes[i].imshow(np.array(fig.canvas.renderer.buffer_rgba()))
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
        id_indices = np.array(id_indices)
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
            lambda i: spearmanr(sub_matrix[:, i], sub_labels).statistic,
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
        sorted_indices = np.argsort(self.values)
        sorted_values = self.values[sorted_indices]

        # Compute all pairwise differences for consecutive values
        diffs = np.diff(sorted_values)

        # Normalize differences based on specified method
        if norm_type == 'range':
            norm_factor = np.max(self.values) - np.min(self.values)
        elif norm_type == 'std':
            norm_factor = np.std(self.values)
        elif norm_type == 'raw':
            norm_factor = 1
        else:
            raise NotImplementedError(f'Unknown norm_type: {norm_type}')
        if norm_factor == 0:
            norm_factor = 1
        diffs = diffs / norm_factor

        # Compute cumulative sums to find ranges of close pairs
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

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Returns distance between two id indices.

        This is implemented by subclasses to define their specific distance metric.
        Distance should be normalized between 0 (identical) and 1 (maximally different).
        """
        raise NotImplementedError()

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

        if perc_close > 0:
            # First get close pairs by sampling within each label group (since they by definition
            # share at least one label in common)
            n_close = int(n_pairs * perc_close)
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

    def get_distance(self, idx1: int, idx2: int) -> float:
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
        super().__init__(tag_type, key, ids=ids, values=values)

    def by_label(self) -> dict[Any, set[str]]:
        """Returns a dictionary mapping each label to the set of ids that have that label."""
        ret = defaultdict(set)
        for id, vs in self.values.items():
            for v in vs:
                ret[v].add(id)
        return dict(ret)

    def get_distance(self, idx1: int, idx2: int) -> float:
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
        logger.info(f'Adding msg {msg}')
        self.msgs.append(msg)

    def parse_tags(self, tag_path: str) -> None:
        """Parses our tags from the tag db"""
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
        for (tag_type, key), ids_values in grouped.items():
            if cur := parse_into_labels(tag_type, key, ids_values):
                self.labels[key] = cur
        #pprint(self.labels)

    def check_correlations(self, keys, matrix: array2D, n_top:int=10) -> None:
        """Checks correlations against all numeric labels"""
        logger.debug(f'Got matrix: {matrix.shape}, {len(keys)} keys: {keys[:10]}, {matrix}')
        for key, labels in self.labels.items():
            top = labels.check_correlations(keys, matrix, n_top=n_top)
            if not isinstance(top, dict):
                top = dict(all=top)
            for label, lsts in top.items():
                for name, lst in zip(['pearson', 'spearman'], lsts):
                    for corr, dim in lst:
                        if abs(corr) > 0.5:
                            self.add_msg(unit='correlation',
                                         label=label,
                                         key=key,
                                         dim=dim,
                                         method=name,
                                         value=corr,
                                         score=3,
                                         warning=f'High correlation {corr:.3f} for {key}={label} at dim {dim}')

    def compare_stats(self, a: array1d, b: array1d) -> tuple[Stats, Stats, Stats]:
        """Compares various stats between two 1D arrays.

        Returns 3 dicts of stats, one for each array and one for the comparison.
        """
        def get_stats(x: array1d) -> Stats:
            return dict(
                mean=float(np.mean(x)),
                std=float(np.std(x)),
                min=float(np.min(x)),
                max=float(np.max(x)),
                median=float(np.median(x)),
                n_zeros=int(np.sum(x == 0)),
                n_neg=int(np.sum(x < 0)),
                n_pos=int(np.sum(x > 0)),
                kurtosis=float(stats.kurtosis(x)),
                gmean=float(stats.gmean(x)),
                skew=float(stats.skew(x)),
                p1=float(np.percentile(x, 1)),
                p5=float(np.percentile(x, 5)),
                p25=float(np.percentile(x, 25)),
                p75=float(np.percentile(x, 75)),
                p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)),
            )
        stats_a = get_stats(a)
        stats_b = get_stats(b)
        stats_cmp = {k: stats_a[k] - stats_b[k] for k in stats_a}
        # add comparison-only stats
        stats_cmp.update(dict(
            pearson=float(np.corrcoef(a, b)[0, 1]),
            spearman=float(spearmanr(a, b).statistic),
        ))
        # compute least squares linear fit to get rvalue
        res = stats.linregress(a, b)
        stats_cmp.update(dict(
            linear_least_square_r2=float(res.rvalue)**2.0,
        ))

        return stats_a, stats_b, stats_cmp

    def init_fig(self):
        plt.figure(figsize=(10, 10))
        plt.grid(True)
        plt.tight_layout()

    def make_plots(self, a: array1d, b: array1d, a_name: str='a', b_name:str='b') -> dict[str, mpl.figure.Figure]:
        """Makes various plots comparing 1D array `a` to `b`.

        Returns a dict mapping plot type to figure.
        """
        ret = {}
        # make a scatter plot of a vs b
        self.init_fig()
        plt.scatter(a, b)
        plt.xlabel(a_name)
        plt.ylabel(b_name)
        plt.title(f'Scatter plot of {a_name} vs {b_name}')
        #ret['scatter'] = plt.gcf()
        # make a q-q plot to compare distributions
        self.init_fig()
        stats.probplot(a, dist="norm", plot=plt)
        stats.probplot(b, dist="norm", plot=plt)
        plt.title(f'Q-Q plot of {a_name} vs {b_name}')
        ret['qq'] = plt.gcf()
        return ret

    def check_distances(self, n_pairs: int=1000) -> None:
        """Does various tests based on distances"""
        fs_keys = set(self.fs.keys())
        for key, label in self.labels.items():
            kw = dict(close_thresh=.4, perc_close=0.5, norm_type='std')
            pairs = label.get_distances(n_pairs, **kw)
            # filter by keys in our embeddings
            pairs = [(id1, id2, dist) for id1, id2, dist in pairs if id1 in fs_keys and id2 in fs_keys]
            if not pairs:
                continue
            a_keys, b_keys, label_dists = zip(*pairs)
            label_dists = np.array(label_dists)
            print(f'\nFor {key} got {len(pairs)} pairs: {pairs[:3]}')
            ids = sorted(set(a_keys) | set(b_keys))
            # get A and B matrix of embeddings to then compute distances
            keys, fvecs = self.fs.get_keys_embeddings(keys=sorted(ids), normed=False, scale_mean=False, scale_std=False)
            A = np.array([fvecs[keys.index(id)] for id in a_keys])
            B = np.array([fvecs[keys.index(id)] for id in b_keys])
            # compute dot product, cosine similarity, euclidean distance (all small=similar)
            dists = dict(
                dot_prod=1.0 - np.einsum('ij,ij->i', A, B),
                cos_sim=1.0 - np.einsum('ij,ij->i', A, B) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)),
                #euc_dist=np.linalg.norm(A - B, axis=1),
            )
            if np.allclose(dists['dot_prod'], dists['cos_sim']):
                del dists['cos_sim']
            # now compare each one to label_dists
            for k, m in dists.items():
                assert m.shape == (len(pairs),), f'Bad shape for {k}: {m.shape} vs {len(pairs)}'
                print(f'  {k}')
                label_stats, dist_stats, cmp_stats = self.compare_stats(label_dists, m)
                for s in cmp_stats:
                    if s in label_stats:
                        print(f'    {s:>10}:\t{label_stats[s]: .3f}\t{dist_stats[s]: .3f}\t{cmp_stats[s]: .3f}')
                    else:
                        print(f'    {s:>10}:\t{cmp_stats[s]: .3f}')
                plots = self.make_plots(label_dists, m, a_name=f'Label dists for {key}', b_name=f'{k} dists for {key}')
                # display each plot interactively
                for name, plot in plots.items():



    def test_distances(self) -> None:
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

    def run(self) -> None:
        logger.info(f'Validating embeddings in {self.paths}, {len(self.fs)}')
        self.check_distances()
        return
        # raw correlations
        # norming hurts correlations
        # scaling mean/std doesn't do anything, because correlation is scale-invariant
        keys, fvecs = self.fs.get_keys_embeddings(normed=False, scale_mean=False, scale_std=False)
        n_top = 10
        self.check_correlations(keys, fvecs, n_top=n_top)
        if 0:
            # pca correlations
            pca = PCA(n_components=min(n_top, fvecs.shape[1]))
            trans = pca.fit_transform(fvecs)
            # if the first pca dimension explained variance is too high, add a message about it
            if pca.explained_variance_ratio_[0] > 0.5:
                self.add_msg(unit='pca', warning=f'PCA first dimension explains too much variance: {pca.explained_variance_ratio_[0]:.3f}', score=-2)
            print(f'PCA with {pca.n_components_} comps, explained variance: {pca.explained_variance_ratio_}: {fvecs.shape} -> {trans.shape}')
            self.check_correlations(keys, trans, n_top=n_top)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    parser = ArgumentParser(description='Embeddings evaluator')
    parser.add_argument('paths', nargs='+', help='Paths to the embeddings lmdb file')
    parser.add_argument('-t', '--tag_path', help='Path to the tags sqlite db')
    args = parser.parse_args()
    ev = EmbeddingsValidator(args.paths, tag_path=args.tag_path)
    ev.run()
