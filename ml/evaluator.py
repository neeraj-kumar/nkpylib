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
  - Given distances, we can get neighbors
  - For numerical, can directly compare values to get distances
  - For multiclass, we might need embeddings for labels to get distances. Otherwise we just have
    same/different class
    - Might get these from text embeddings of labels
  - For multilabel, can do jaccard or hamming distance of labels on a point
  - Do this per label, but then also combine distances across labels
    - Have to be careful about scaling
    - Join in neighbor-space? Distance-space?
    - Might have multiple combined distances to compare
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
from typing import Any, Sequence, Generic, TypeVar

import numpy as np

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

    def get_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[tuple[str, str, float]]:
        """Returns `n_pairs` of `(id1, id2, distance)` tuples.

        You can specify `perc_close` which is the percentage of pairs that should be "close"
        according to the label type's definition of closeness. < 0 means we don't care about
        closeness (default).

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


class NumericLabels(Labels):
    """A class for numeric labels. For now we convert them to floats.

    This stores ids as a list and values as a numpy array, where values[i] is the value for ids[i].
    """
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
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

    def get_distance(self, val1: Any, val2: Any) -> float:
        """Returns normalized distance between two numeric values."""
        if self.norm_type == 'range':
            norm_factor = np.max(self.values) - np.min(self.values)
        elif self.norm_type == 'std':
            norm_factor = np.std(self.values)
        else:  # raw
            norm_factor = 1
        if norm_factor == 0:
            norm_factor = 1
        return abs(val1 - val2) / norm_factor


class MulticlassBase(Labels):
    """Some common code for multiclass/multilabel labels."""
    def by_label(self) -> dict[Any, set[str]]:
        raise NotImplementedError()
        
    def get_distance(self, val1: Any, val2: Any) -> float:
        """Returns distance between two values.
        
        This is implemented by subclasses to define their specific distance metric.
        Distance should be normalized between 0 (identical) and 1 (maximally different).
        """
        raise NotImplementedError()
        
    def get_distances(self, n_pairs: int, perc_close: float = -1, **kw) -> list[tuple[str, str, float]]:
        """Returns n_pairs of (id1, id2, distance) tuples.
        
        If perc_close >= 0, ensures that proportion of pairs are "close" according to 
        the label type's definition of closeness.
        
        This implementation handles the pair generation and sampling strategy,
        while subclasses implement get_distance() for their specific distance metric.
        """
        pairs = []
        done = set()
        
        if perc_close > 0:
            # First get close pairs by sampling within groups that tend to be close
            n_close = int(n_pairs * perc_close)
            by_group = self.by_label()
            close_candidates = []
            
            # Generate candidates from each group
            for ids in by_group.values():
                if len(ids) >= 2:
                    ids = list(ids)
                    # Take all pairs within this group
                    close_candidates.extend((i, j) for idx, i in enumerate(ids) 
                                         for j in ids[idx+1:])
                                         
            # Sample from candidates
            if close_candidates:
                sampled = random.sample(close_candidates, 
                                      min(n_close, len(close_candidates)))
                for id1, id2 in sampled:
                    i1, i2 = self.ids.index(id1), self.ids.index(id2)
                    dist = self.get_distance(self.values[i1], self.values[i2])
                    spair = frozenset((id1, id2))
                    if spair not in done:
                        done.add(spair)
                        pairs.append((id1, id2, dist))
        
        # Fill remaining with random pairs
        remaining = n_pairs - len(pairs)
        if remaining > 0:
            attempts = 0
            max_attempts = remaining * 3  # Avoid infinite loop
            while len(pairs) < n_pairs and attempts < max_attempts:
                attempts += 1
                id1, id2 = random.sample(self.ids, 2)
                spair = frozenset((id1, id2))
                if spair in done:
                    continue
                i1, i2 = self.ids.index(id1), self.ids.index(id2)
                dist = self.get_distance(self.values[i1], self.values[i2])
                done.add(spair)
                pairs.append((id1, id2, dist))
                
        return pairs

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

    def get_distance(self, val1: Any, val2: Any) -> float:
        """Returns distance between two categorical values.
        
        Distance is 0 for same class, 1 for different class.
        """
        return 0.0 if val1 == val2 else 1.0


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

    def get_distance(self, val1: Any, val2: Any) -> float:
        """Returns distance between two sets of labels.
        
        Distance is Jaccard distance: 1 - |intersection|/|union|.
        """
        set1 = set(val1)
        set2 = set(val2)
        union = len(set1 | set2)
        if union == 0:  # Handle empty sets
            return 0.0 if not set1 and not set2 else 1.0
        return 1.0 - len(set1 & set2) / union


def parse_into_labels(tag_type: str,
                      key: str,
                      ids_values: list[tuple[str, Any]],
                      impure_thresh=0.1) -> Labels:
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
            if 'revenue' in key:
                print(f'{(tag_type, key)} -> {ids_values[:20]}')
            self.labels[key] = parse_into_labels(tag_type, key, ids_values)
        pprint(self.labels)

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
                            self.add_msg(unit='correlation', label=label, key=key, dim=dim, method=name, value=corr, score=3,
                                         warning=f'High correlation {corr:.3f} for {key}={label} at dim {dim}')


    def run(self) -> None:
        logger.info(f'Validating embeddings in {self.paths}, {len(self.fs)}')
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
