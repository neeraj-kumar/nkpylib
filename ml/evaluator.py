"""A generic embeddings evaluator.

This does a bunch of semi-automated things to test out embeddings and see how good they are.
This is mostly in reference to a set of labels you provide, in the form of a tags database.

- Lots of ways of mapping embeddings to other embeddings, should be common procedure
  - PCA, K-PCA, ISOMAP, t-SNE, UMAP, LLE, etc
  - skip ICA, it's more for source separation
  - Don't repeat work
  - Correlate embedding dimensions before/after transformation
- PCA with only one variance-explaining dimension is bad
- Classification/regression against labels
  - Some std methods: linear svm, rbf svm, svr, forests, pytorch nn?
  - look at binary and one-vs-all
  - don't do special multilabel stuff
- For correlation, pick things with corr > 0.5 and highlight those dims somehow
  - Maybe more generally for each Label, output best predictors/correlators/etc
- Output is partially a report
  - In the future, do ML doctor stuff
  - Highlight clear problems/wins
- For dims, we can also compute histograms
  - Standard stats on histograms: kurtosis, bin sizes, lop-sidedness, normality
  - highlight 0s other outliers

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
    """
    def __init__(self, tag_type: str, key: str, *, ids: list[str], values: Any):
        self.tag_type = tag_type
        self.key = key
        self.ids = ids
        self.values = values

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.tag_type} {self.key} ({len(self.ids)} labels)>'

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


class NumericLabels(Labels):
    """A class for numeric labels. For now we convert them to floats.

    This stores ids as a list and values as a numpy array, where values[i] is the value for ids[i].
    """
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
        ids = [id for id, v in ids_values]
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


class MulticlassBase(Labels):
    """Some common code for multiclass/multilabel labels."""
    def by_label(self) -> dict[Any, set[str]]:
        raise NotImplementedError()

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
            if len(top) <= 20:
                print(f'  Top for {key}:')
                pprint(top)


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
