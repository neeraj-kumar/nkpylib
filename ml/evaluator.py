"""A generic embeddings evaluator.

This does a bunch of semi-automated things to test out embeddings and see how good they are.
This is mostly in reference to a set of labels you provide, in the form of a tags database.
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
from pprint import pprint
from typing import Any, Sequence, Generic, TypeVar

import numpy as np

from pony.orm import * # type: ignore
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

FLOAT_TYPES = (float, np.float32, np.float64)
NUMERIC_TYPES = FLOAT_TYPES + (int, np.int32, np.int64)

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

class NumericLabels(Labels):
    """A class for numeric labels. For now we convert them to floats."""
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
        ids = [id for id, v in ids_values]
        values = np.array([v for id, v in ids_values], dtype=np.float32)
        super().__init__(tag_type, key, ids=ids, values=values)

class MulticlassLabels(Labels):
    """A class for multiclass (mutually-exclusive) labels."""
    def __init__(self, tag_type: str, key: str, ids_values: list[tuple[str, Any]]):
        ids = [id for id, v in ids_values]
        values = [v for id, v in ids_values]
        types = Counter(type(v) for v in values)
        assert len(types) == 1
        t = type(values[0])
        assert t not in FLOAT_TYPES, f'No floats in multiclass label: {types}'
        super().__init__(tag_type, key, ids=ids, values=values)

class MultilabelLabels(Labels):
    """A class for multilabel (non-mutually-exclusive) labels."""
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
            if not isinstance(labels, NumericLabels):
                continue
            logger.debug(f'Checking correlation for {key} with {len(labels.ids)} labels: {labels.ids[:5]}, {labels.values[:5]}')
            # get row indices of our ids in keys and in labels
            mat_indices = []
            label_indices = []
            common = set(keys) & set(labels.ids)
            for mat_idx, id in enumerate(keys):
                if id not in common:
                    continue
                label_idx = labels.ids.index(id)
                mat_indices.append(mat_idx)
                label_indices.append(label_idx)
            logger.debug(f'  Found {len(common)} matching ids in embeddings')
            sub_matrix = matrix[mat_indices, :]
            sub_labels = labels.values[label_indices]
            logger.debug(f'Got sub matrix of shape {sub_matrix.shape}: {sub_matrix}')
            logger.debug(f'Got sub labels of shape {sub_labels.shape}: {sub_labels}')
            # compute correlation between each column of sub_matrix and all of sub_labels
            corrs = np.array([np.corrcoef(sub_matrix[:, i], sub_labels)[0, 1] for i in range(sub_matrix.shape[1])])
            logger.debug(f'  Correlations ({corrs.shape}): {corrs}')
            corrs = [(corr, dim) for dim, corr in enumerate(corrs) if not np.isnan(corr)]
            top = sorted(corrs, key=lambda x: abs(x[0]), reverse=True)[:n_top]
            print(f'Top correlations for {key}:')
            for corr, dim in top:
                print(f'    Dim {dim}: {corr}')


    def run(self) -> None:
        logger.info(f'Validating embeddings in {self.paths}, {len(self.fs)}')
        # raw correlations
        # norming hurts correlations
        # scaling mean/std doesn't do anything, because correlation is scale-invariant
        keys, fvecs = self.fs.get_keys_embeddings(normed=False, scale_mean=False, scale_std=False)
        n_top = 10
        self.check_correlations(keys, fvecs, n_top=n_top)
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
