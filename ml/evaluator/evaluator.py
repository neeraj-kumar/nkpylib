"""A generic embeddings evaluator.

This does a bunch of semi-automated things to test out embeddings and see how good they are.
This is mostly in reference to a set of labels you provide, in the form of a tags database.

TODO:

- I think the duplicate results are perhaps coming from multiple "label_name" per label_key
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
import logging.handlers
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
from dataclasses import dataclass
from enum import Enum
from itertools import product
from os.path import abspath, dirname, exists, join
from pprint import pprint as _pprint, pformat
from queue import Queue
from typing import Any, Literal, Sequence, Generic, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn # type: ignore

from pony.orm import db_session
from scipy.spatial.distance import pdist, squareform # type: ignore
from scipy.special import kl_div
from sklearn.exceptions import ConvergenceWarning # type: ignore
from sklearn.base import BaseEstimator # type: ignore
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, MiniBatchKMeans, DBSCAN # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from sklearn.linear_model import Ridge, SGDClassifier # type: ignore
from sklearn.metrics import recall_score, r2_score, balanced_accuracy_score, accuracy_score # type: ignore
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix # type: ignore
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold # type: ignore
from sklearn.metrics import recall_score, r2_score, balanced_accuracy_score, accuracy_score # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score # type: ignore
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from sklearn.linear_model import Ridge, SGDClassifier # type: ignore
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR # type: ignore
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR # type: ignore
from tqdm import tqdm

from nkpylib.utils import specialize
from nkpylib.ml.evaluator.evaluator_ops import Op, OpManager, OpResult, result_logger
from nkpylib.ml.evaluator.labels import parse_into_labels, Labels, MulticlassLabels, MultilabelLabels, NumericLabels
from nkpylib.ml.feature_set import (
    FeatureSet,
    JsonLmdb,
    LmdbUpdater,
    MetadataLmdb,
    NumpyLmdb,
)
from nkpylib.ml.ml_types import (
    NUMERIC_TYPES,
    FLOAT_TYPES,
    array1d,
    array2d,
    nparray1d,
    nparray2d,
    )

from numpy.typing import NDArray
import numpy as np

from nkpylib.ml.tag_db import Tag, get_all_tags, init_tag_db

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Create specialized loggers for different components
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("evaluator.op")
eval_logger = logging.getLogger("evaluator.eval")
error_logger = logging.getLogger("evaluator.error")

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

class ClusteringAlgorithm(Enum):
    """Enumeration of available clustering algorithms."""
    MINIBATCH_KMEANS = "minibatch_kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    AFFINITY_PROPAGATION = "affinity_propagation"

    @property
    def is_distance_based(self) -> bool:
        """Returns True if this algorithm requires distance matrices."""
        return self in {self.AGGLOMERATIVE, self.AFFINITY_PROPAGATION}

    @property
    def is_embedding_based(self) -> bool:
        """Returns True if this algorithm works on raw embeddings."""
        return self in {self.MINIBATCH_KMEANS, self.DBSCAN}

class PredictionAlgorithm(Enum):
    """Enumeration of available prediction algorithms."""
    # Regression algorithms
    RIDGE = "ridge"
    RBF_SVR = "rbf_svr"
    LINEAR_SVR = "linear_svr"
    KNN_REG = "knn_reg"

    # Classification algorithms
    RBF_SVM = "rbf_svm"
    LINEAR_SVM = "linear_svm"
    KNN_CLS = "knn_cls"

    @property
    def is_regression(self) -> bool:
        """Returns True if this is a regression algorithm."""
        return self in {self.RIDGE, self.RBF_SVR, self.LINEAR_SVR, self.KNN_REG}

    @property
    def is_classification(self) -> bool:
        """Returns True if this is a classification algorithm."""
        return self in {self.RBF_SVM, self.LINEAR_SVM, self.KNN_CLS}

@dataclass
class DimensionCheckResult:
    """Result from CheckDimensionsOp."""
    dimension_counts: dict[int, int]
    is_consistent: bool

@dataclass
class NaNCheckResult:
    """Result from CheckNaNsOp."""
    total_nans: int
    nan_keys: list[tuple[str, int]]
    has_nans: bool

@dataclass
class BasicChecksReport:
    """Result from BasicChecksOp."""
    errors: list[str]
    dimension_check: DimensionCheckResult
    nan_check: NaNCheckResult

@dataclass
class NormalizedEmbeddings:
    """Result from NormalizeOp."""
    keys: list[str]
    embeddings: nparray2d
    normed: bool
    scale_mean: bool
    scale_std: bool

@dataclass
class LabelArraysData:
    """Result from GetLabelArraysOp."""
    label_obj: Labels
    label_key: str
    sub_keys: list[str]
    label_names: list[str]
    label_arrays: nparray2d
    sub_matrix: nparray2d

@dataclass
class LabelDistancesData:
    """Result from GetLabelDistancesOp."""
    label_key: str
    sub_keys: list[str]
    label_distances: nparray2d
    sub_matrix: nparray2d

@dataclass
class EmbeddingDimsData:
    """Result from GetEmbeddingDimsOp."""
    dims_matrix: nparray2d
    transform: str
    label_key: str

@dataclass
class EmbeddingDistancesData:
    """Result from GetEmbeddingDistancesOp."""
    label_key: str
    metric: str
    sub_keys: list[str]
    embedding_distances: nparray2d

@dataclass
class NeighborsData:
    """Result from GetNeighborsOp."""
    distance_type: str
    label_key: str
    metric: str | None
    neighbors: nparray2d
    distances: nparray2d
    keys: list[str]

@dataclass
class PredictionTasksData:
    """Result from GenPredictionTasksOp."""
    label_key: str
    tasks: PTasks
    sub_keys: list[str]
    sub_matrix: nparray2d
    label_name: str | None = None

@dataclass
class PredictionResult:
    """Result from RunPredictionOp."""
    label_key: str
    label_name: str
    task_name: str
    model_name: str
    score: float
    score_type: str
    n_classes: int | None
    predictions: nparray1d
    true_values: nparray1d

@dataclass
class NeighborComparison:
    """Result from CompareNeighborsOp."""
    label_key: str
    embedding_metric_a: str | None
    embedding_metric_b: str | None
    metrics: dict[str, float]
    per_item_metrics: dict[str, list[float]] | None = None

@dataclass
class StatsComparison:
    """Result from CompareStatsOp."""
    stats_a: list[Stats]
    stats_b: list[Stats]
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    comparisons: dict[tuple[int, int], Stats]
    n_comparisons: int

@dataclass
class ClusteringResult:
    """Result from RunClusteringOp."""
    algorithm: str
    labels: nparray1d
    n_clusters: int
    centroids: nparray2d | None
    parameters: dict[str, Any]
    keys: list[str]
    label_key: str
    inertia: float | None = None
    true_labels: nparray2d | None = None

@dataclass
class Warning:
    """Standardized warning structure for analysis results.

    This dataclass captures all the warning information generated by various ops
    with appropriate defaults for optional fields.
    """
    unit: str  # Type of analysis unit (e.g., 'prediction', 'clustering', 'neighbors')
    warning: str  # Human-readable warning message
    score: int = 1  # Importance score (1=low, 2=medium, 3=high)
    label_key: str = ""  # Label key this warning relates to
    algorithm: str = ""  # Algorithm name if applicable
    metric: str = ""  # Metric name if applicable
    value: float | int | None = None  # Numeric value associated with warning
    task: str = ""  # Task name if applicable
    issue: str = ""  # Issue type/category
    n_classes: int | None = None  # Number of classes for classification tasks
    count: int | None = None  # Count of items if applicable
    total_comparisons: int | None = None  # Total comparisons made
    noise_ratio: float | None = None  # Noise ratio for clustering
    clusters: dict[int, float] | None = None  # Cluster-specific data
    metrics: list[str] | None = None  # List of metrics involved
    key: str = ""  # Generic key field

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}

    @classmethod
    def add_warning(cls, *, unit: str, warning: str, analysis: dict[str, Any], pred: bool=True, **kwargs) -> None]:
        """Create a Warning instance and adds it to `analysis['warnings']` as a dict if `pred`."""
        if not pred or not analysis:
            return
        instance = cls(unit=unit, warning=warning, **kwargs)
        analysis.setdefault('warnings', []).append(instance.to_dict())

@dataclass
class ClusterLabelAnalysis:
    """Result from ClusterLabelAnalysisOp."""
    cluster_labels: nparray1d
    label_names: list[str]
    confusion_matrix: nparray2d
    cluster_purities: dict[int, float]
    dominant_labels: dict[int, str]
    adjusted_rand_index: float
    normalized_mutual_info: float
    label_key: str
    true_labels: nparray2d | None = None

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
    funcs = dict(
            pearson=lambda: float(np.corrcoef(a, b)[0, 1]),
            spearman=lambda: float(stats.spearmanr(a, b).statistic),
            tau=lambda: float(stats.kendalltau(a, b).statistic),
            kl_div=lambda: float(stats.entropy(a, b)),
            linear_least_square_slope=lambda: float(stats.linregress(a, b).rvalue)**2.0,
    )
    ret = {}
    for name, func in funcs.items():
        try:
            ret[name] = func()
        except Exception as e:
            error_logger.exception(e)
            ret[name] = float('nan')
    if stats_a is not None and stats_b is not None:
        for k in stats_a:
            key = f'diff_{k}'
            try:
                ret[key] = stats_b[k] - stats_a[k]
            except Exception as e:
                error_logger.exception(e)
                ret[key] = float('nan')
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



class StartValidatorOp(Op):
    """Starting point for running embeddings validation.

    Passes through kw version of parsed args from ArgumentParser.
    """
    name = 'start_validator'
    input_types: set[str] = set()
    output_types = {"argparse"}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any], **kwargs) -> OpResult:
        return inputs


class ParseTagsOp(Op):
    """Parses our tags from the tag db"""
    name = 'parse_tags'
    input_types = {'argparse'}
    output_types = {'labels'}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any], **kwargs) -> OpResult:
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
        labels = results
        ret: dict[str, Any] = {}
        for name, label in labels.items():
            ret[name] = dict(
                tag_type=label.tag_type,
                n_ids=len(label.ids),
                n_unique_ids=len(set(label.ids)),
                norm_type=getattr(label, 'norm_type', None),
                n_unique_values=len(set(label.values)) if isinstance(label.values, list) else None,
            )
        return ret


class LoadEmbeddingsOp(Op):
    """Load embeddings from paths into a FeatureSet."""
    name = 'load_embeddings'
    input_types = {'argparse'}
    output_types = {"feature_set"}
    is_intermediate = True

    #TODO return cartesian product of inputs as variants
    def _execute(self, inputs: dict[str, Any], **kwargs) -> OpResult:
        paths = inputs['argparse']['paths']
        return FeatureSet(paths, **kwargs)

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        fs: FeatureSet = results
        return dict(
            n_embeddings=len(fs),
        )

class CheckDimensionsOp(Op):
    """Check that all embeddings have consistent dimensions."""
    #run_mode = 'process'
    name = "check_dimensions"
    input_types = {"feature_set"}
    output_types = {"dimension_check_result"}

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        fs = inputs["feature_set"]
        dims: Counter[int] = Counter()
        for key, emb in fs.items():
            dims[len(emb)] += 1
        return DimensionCheckResult(
            dimension_counts=dict(dims),
            is_consistent=(len(dims) == 1),
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        dims = Counter(results.dimension_counts)
        return dict(
            is_consistent=(len(dims) == 1),
            error_message=f"Inconsistent embedding dimensions: {dims.most_common()}" if len(dims) > 1 else None,
        )

class CheckNaNsOp(Op):
    """Check for NaN values in embeddings."""

    run_mode = 'main'  # Changed from 'process' to avoid pool issues
    name = "check_nans"
    input_types = {"feature_set"}
    output_types = {"nan_check_result"}

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        fs = inputs["feature_set"]
        n_nans = 0
        nan_keys = []
        for key, emb in fs.items():
            key_nans = np.sum(np.isnan(emb))
            n_nans += key_nans
            if key_nans > 0:
                nan_keys.append((key, int(key_nans)))
        return NaNCheckResult(
            total_nans=int(n_nans),
            nan_keys=nan_keys,
            has_nans=n_nans > 0,
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        n_nans = results.total_nans
        return dict(
            has_nans=n_nans > 0,
            error_message=None if n_nans == 0 else f"Found {n_nans} NaNs in embeddings",
        )



class BasicChecksOp(Op):
    """Combine dimension and NaN checks into a single basic validation."""

    name = "basic_checks"
    input_types = {"dimension_check_result", "nan_check_result"}
    output_types = {"basic_checks_report"}
    run_mode = 'main'  # Changed from 'process' to avoid pool issues

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        dim_result = inputs["dimension_check_result"]
        nan_result = inputs["nan_check_result"]
        errors = []
        if not dim_result.is_consistent:
            errors.append(f"Inconsistent embedding dimensions: {dim_result.dimension_counts}")
        if nan_result.has_nans:
            errors.append(f"Found {nan_result.total_nans} NaNs in embeddings")
        return BasicChecksReport(
            errors=errors,
            dimension_check=dim_result,
            nan_check=nan_result
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        errors = results.errors
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

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
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
        return NormalizedEmbeddings(keys=keys, embeddings=emb, normed=self.normed,
                                    scale_mean=self.scale_mean, scale_std=self.scale_std)

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            n_embeddings=len(results.keys),
            embedding_shape=results.embeddings.shape,
            normed=self.normed,
            scale_mean=self.scale_mean,
            scale_std=self.scale_std
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
            #if len(ret) > 3: break #FIXME temporary
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

    name = "get_label_arrays"
    input_types = {"normalized_embeddings", "labels"}
    output_types = {"label_arrays_data"}
    is_intermediate = True

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        norm_emb = inputs["normalized_embeddings"]
        keys, matrix = norm_emb.keys, norm_emb.embeddings
        label = inputs["labels"][self.label_key]
        result = label.get_label_arrays(keys, matrix)
        return LabelArraysData(
            label_obj=label,
            label_key=self.label_key,
            sub_keys=result.sub_keys,
            label_names=result.label_names,
            label_arrays=result.label_arrays,
            sub_matrix=result.sub_matrix,
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.label_key,
            n_sub_keys=len(results.sub_keys),
            n_label_names=len(results.label_names),
            label_array_shape=results.label_arrays.shape,
            sub_matrix_shape=results.sub_matrix.shape,
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
    run_mode = "main"  # Changed from 'process' to avoid pool issues

    def __init__(self, label_key: str, n_pts: int = 200, perc_close: float = 0.5, **kw):
        self.label_key = label_key
        self.n_pts = n_pts
        self.perc_close = perc_close
        super().__init__(label_key=label_key, **kw)

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        label = inputs["labels"][self.label_key]
        norm_emb = inputs["normalized_embeddings"]
        keys, matrix = norm_emb.keys, norm_emb.embeddings
        result = label.get_all_distances(
            n_pts=self.n_pts,
            keys=keys,
            matrix=matrix,
            perc_close=self.perc_close,
        )
        # clamp distances to be positive
        result.label_distances = np.clip(result.label_distances, a_min=0.0, a_max=None)

        return LabelDistancesData(
            label_key=self.label_key,
            sub_keys=result.sub_keys,
            label_distances=result.label_distances,
            sub_matrix=result.sub_matrix,
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.label_key,
            n_sub_keys=len(results.sub_keys or []),
            label_distances_shape=results.label_distances.shape,
            sub_matrix_shape=results.sub_matrix.shape,
        )

class GetEmbeddingDimsOp(Op):
    """Extract embedding dimensions from filtered label data for consistency.

    Uses the keys and matrix from label_arrays_data to ensure both arrays
    have the same samples in the same order.
    """
    name = "get_embedding_dims"
    input_types = {"label_arrays_data"}
    output_types = {"embedding_dims"}
    is_intermediate = True
    run_mode = "main"  # Changed from 'process' to avoid pool issues

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

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        label_data = inputs["label_arrays_data"]
        matrix = label_data.sub_matrix  # Already filtered to the right intersection
        dims = matrix.T  # Each row is one dimension across all samples
        # Apply transformation based on variant
        if self.transform == "log":
            # Apply log transform, handling negative values
            dims = np.log1p(np.abs(dims))
        # "raw" needs no transformation
        return EmbeddingDimsData(
            dims_matrix=dims,
            transform=self.transform,
            label_key=label_data.label_key
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.label_key,
            dims_shape=results.dims_matrix.shape,
            transform=results.transform,
        )


class GetEmbeddingDistancesOp(Op):
    """Generate distance matrices from embeddings using various distance metrics.

    Takes a list of ids (from label_distances) and normalized embeddings, and computes distance
    matrices using different metrics (cosine, dot product, etc.)
    """
    name = "get_embedding_distances"
    input_types = {"normalized_embeddings", "label_distances"}
    output_types = {"embedding_distances", 'distances'}
    run_mode = "main"  # Changed from 'process' to avoid pool issues
    is_intermediate = True

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants for distance metrics."""
        metrics = ['cosine', 'dot_product', 'euclidean']
        metrics = ['cosine', 'dot_product'] #TODO 'euclidean' is too similar to cosine
        label_key = inputs['label_distances'].label_key
        return {f'label:{label_key}_metric:{metric}': dict(metric=metric) for metric in metrics}

    def __init__(self, metric, **kw):
        self.metric = metric
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        norm_emb = inputs["normalized_embeddings"]
        label_data = inputs["label_distances"]

        # Get the ids from label distances to ensure consistency
        matrix = label_data.sub_matrix  # Already filtered to the right intersection
        op_logger.info(f'Computing {self.metric} dists for label {label_data.label_key} with {matrix.shape} matrix')

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

        # clamp distances to be positive
        distances = np.clip(distances, a_min=0.0, a_max=None)

        return EmbeddingDistancesData(
            label_key=label_data.label_key,
            metric=self.metric,
            sub_keys=label_data.sub_keys,
            embedding_distances=distances
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.label_key,
            metric=results.metric,
            n_sub_keys=len(results.sub_keys),
            distances_shape=results.embedding_distances.shape,
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
    name = "get_neighbors"
    input_types = {'distances'}
    output_types = {"neighbors_data", "neighbors_data_a", "neighbors_data_b"} # aliases
    #run_mode = "process"
    is_intermediate = True

    def __init__(self, k: int = 20, **kw):
        """Initialize with number of neighbors to compute.

        - `k`: Number of nearest neighbors to compute (default: 20)
        """
        self.k = k
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        # Determine which type of distance matrix we're using
        data = inputs['distances']
        op_logger.info(f'Computing neighbors, got data type {type(data)}, k={self.k}')

        if isinstance(data, LabelDistancesData):
            distances = data.label_distances
            keys = data.sub_keys
            distance_type = "label"
            metric = None
            label_key = data.label_key
        elif isinstance(data, EmbeddingDistancesData):
            distances = data.embedding_distances
            keys = data.sub_keys
            distance_type = "embedding"
            metric = data.metric
            label_key = data.label_key
        else:
            raise ValueError(f"Distance data must be LabelDistancesData or EmbeddingDistancesData, got {type(data)}")

        # Ensure we don't request more neighbors than we have points
        k = min(self.k + 1, distances.shape[0])  # +1 because the point itself is included

        # Use sklearn's NearestNeighbors with precomputed distances
        nn_cls = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nn_cls.fit(distances)
        neighbor_dists, neighbor_indices = nn_cls.kneighbors(distances)

        return NeighborsData(
            distance_type=distance_type,
            label_key=label_key,
            metric=metric,
            neighbors=neighbor_indices,
            distances=neighbor_dists,
            keys=keys,
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            distance_type=results.distance_type,
            label_key=results.label_key,
            metric=results.metric,
            n_keys=len(results.keys),
            neighbors_shape=results.neighbors.shape,
            distances_shape=results.distances.shape,
        )


class GenPredictionTasksOp(Op):
    """Generate prediction tasks from processed label arrays.

    Uses the processed label arrays from GetLabelArraysOp to generate prediction tasks.
    Creates variants for each label_name in the label arrays, allowing fine-grained
    prediction tasks for multilabel scenarios.

    Returns a dict with:
    - `label_key`: The label key these tasks are for
    - `label_name`: The specific label name this variant handles
    - `tasks`: Dict mapping task name to tuple of (label array, task type)
    - `sub_keys`: List of keys used in the tasks
    - `sub_matrix`: Submatrix of embeddings corresponding to the keys
    """
    name = "gen_prediction_tasks"
    input_types = {"label_arrays_data"}
    output_types = {"prediction_tasks"}
    run_mode = "main"
    is_intermediate = True

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants based on label names in label_arrays_data."""
        label_data = inputs.get("label_arrays_data")
        if not label_data:
            return None
        variants = {}
        label_key = label_data.label_key
        # Create one variant per label_name
        for i, label_name in enumerate(label_data.label_names):
            variant_name = f"label_key:{label_key}_label_name:{label_name}"
            variants[variant_name] = {
                "label_key": label_key,
                "label_name": label_name,
                "label_index": i
            }
        return variants

    def __init__(self,
                 label_key: str,
                 label_name: str,
                 label_index: int,
                 min_pos: int = 10,
                 max_tasks: int = 10,
                 **kw):
        """Initialize with parameters for task generation.

        - `label_key`: The label key to generate tasks for
        - `label_name`: The specific label name to generate tasks for
        - `label_index`: Index of this label in the label_arrays
        - `min_pos`: Minimum number of positive examples for classification tasks
        - `max_tasks`: Maximum number of tasks to generate
        """
        self.label_key = label_key
        self.label_name = label_name
        self.label_index = label_index
        self.min_pos = min_pos
        self.max_tasks = max_tasks
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        label_data = inputs["label_arrays_data"]
        # Get the specific label array for this variant
        label_obj = label_data.label_obj
        label_array = label_data.label_arrays[self.label_index]
        tasks: PTasks = {}
        # Determine what kind of task this is
        if isinstance(label_obj, NumericLabels):
            # For numeric labels, generate original, log, and exp transformations
            values = label_array.astype(float)
            tasks[f'{self.label_name}-regression-orig'] = (values, 'regression')
            # Log transformation (shift to positive)
            tasks[f'{self.label_name}-regression-log'] = (np.log1p(values - np.min(values) + 1.0), 'regression')
            # Exp transformation
            tasks[f'{self.label_name}-regression-exp'] = (np.expm1(values - np.min(values) + 1.0), 'regression')
        else: # multiclass or multilabel
            counts: Counter[str] = Counter()
            unique_values = set(label_array)
            if len(unique_values) == 2 and set(unique_values).issubset({-1.0, 1.0}): # binary labels
                n_positive = np.sum(label_array > 0)
                if n_positive >= self.min_pos:
                    key = f'{self.label_key}-binary-{self.label_name}'
                    tasks[key] = (label_array, 'classification')
                    counts[key] = n_positive
                # keep only the top `self.max_tasks` most common binary classes (by # positive)
                for i, (key, _) in enumerate(counts.most_common()):
                    if i < self.max_tasks:
                        continue
                    op_logger.debug(f'Deleting task {key} because {i} > {self.max_tasks}')
                    del tasks[key]
            else: # multiclass
                tasks[f'{self.label_name}-multiclass-orig'] = (label_array, 'classification')
        return PredictionTasksData(
            label_key=self.label_key,
            tasks=tasks,
            sub_keys=label_data.sub_keys,
            sub_matrix=label_data.sub_matrix,
            label_name=self.label_name
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes `results` from executing this op with given `inputs`."""
        return dict(
            variant=self.variant,
            label_key=results.label_key,
            label_name=self.label_name,
            label_index=self.label_index,
            n_tasks=len(results.tasks),
            n_sub_keys=len(results.sub_keys),
            sub_matrix_shape=results.sub_matrix.shape,
            task_counts={k: v[1] for k, v in results.tasks.items()},
            task_names=list(results.tasks.keys()),
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
    run_mode = "main"

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        """Returns different variants for model types and tasks."""
        pt = inputs.get("prediction_tasks")
        assert pt is not None
        tasks = pt.tasks
        label_key, label_name = pt.label_key, pt.label_name
        variants = {}
        for task_name, (_, task_type) in tasks.items():
            if task_type == 'regression':
                models = {
                    "ridge": {"model_type": PredictionAlgorithm.RIDGE},
                    "rbf_svr": {"model_type": PredictionAlgorithm.RBF_SVR},
                    "linear_svr": {"model_type": PredictionAlgorithm.LINEAR_SVR},
                    "knn_reg": {"model_type": PredictionAlgorithm.KNN_REG}
                }
            else:  # classification
                models = {
                    "rbf_svm": {"model_type": PredictionAlgorithm.RBF_SVM},
                    "linear_svm": {"model_type": PredictionAlgorithm.LINEAR_SVM},
                    "knn_cls": {"model_type": PredictionAlgorithm.KNN_CLS}
                }
            for model_name, model_params in models.items():
                variant_name = f"label_key:{label_key}_label_name:{label_name}_task:{task_name}_model:{model_name}"
                variants[variant_name] = {
                    "task_name": task_name,
                    "model_type": model_params["model_type"]
                }
        #print(f'RunPredictionOp got {len(variants)} variants for label_key:{label_key}, label_name:{label_name}: {variants}')
        return variants

    def __init__(self, task_name: str, model_type: PredictionAlgorithm, n_splits: int = 4, **kw):
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
            case PredictionAlgorithm.RIDGE:
                return Ridge(alpha=1.0)
            case PredictionAlgorithm.RBF_SVR:
                return SVR(kernel='rbf', C=1.0, epsilon=0.1)
            case PredictionAlgorithm.LINEAR_SVR:
                return LinearSVR(C=1.0, epsilon=0.1, dual='auto')
            case PredictionAlgorithm.KNN_REG:
                return KNeighborsRegressor(n_neighbors=10)
            case PredictionAlgorithm.RBF_SVM:
                return SVC(kernel='rbf', C=1.0, probability=True)
            case PredictionAlgorithm.LINEAR_SVM:
                return LinearSVC(C=1.0, max_iter=200, dual='auto')
            case PredictionAlgorithm.KNN_CLS:
                return KNeighborsClassifier(n_neighbors=10)
            case _:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        task_data = inputs["prediction_tasks"]
        tasks = task_data.tasks
        assert self.task_name in tasks, f"PTask {self.task_name} not found in available tasks"
        X = task_data.sub_matrix
        y, task_type = tasks[self.task_name]
        # Get appropriate model and cross-validation strategy
        model = self._get_model()
        is_regression = self.model_type.is_regression
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
        return PredictionResult(
            label_key=task_data.label_key,
            label_name=task_data.label_name,
            task_name=self.task_name,
            model_name=self.model_type.value,
            score=float(score),
            score_type=score_type,
            n_classes=n_classes,
            predictions=np.array(all_preds),
            true_values=np.array(all_true)
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any], threshold = 0.7) -> dict[str, Any]:
        """Analyzes prediction results and identifies notable outcomes.

        Checks if the prediction score exceeds a threshold (0.7 by default) and
        generates appropriate warnings for high-performing models.
        """
        # Extract key information from results
        R = results
        score = results.score
        score_type = results.score_type
        label_key = results.label_key
        label_name = results.label_name
        task_name = results.task_name
        model_name = results.model_name
        n_classes = results.n_classes

        # Create analysis dict and add warnings
        analysis = {
            "score": score,
            "score_type": score_type,
        }

        # add warnings
        Warning.add_warning(
            analysis=analysis,
            pred=score > threshold,
            unit="prediction",
            warning=f"High prediction {score_type} {score:.3f} for {label_key}:{label_name} using {model_name}",
            label_key=f'{label_key}:{label_name}',
            task=task_name,
            algorithm=model_name,
            value=score,
            n_classes=n_classes,
            score=3,
            metric=score_type
        )
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
    name = "compare_neighbors"
    input_types = {
        ("neighbors_data_a", "neighbors_data_b"): {
            "consistency_fields": ["label_key"]
        }
    }
    output_types = {"neighbor_comparison"}
    run_mode = "main"

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

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        # Find which input is label neighbors and which is embedding neighbors
        neighbors_a, neighbors_b = inputs["neighbors_data_a"], inputs["neighbors_data_b"]

        # Get the neighbor indices
        l_nn = neighbors_a.neighbors
        m_nn = neighbors_b.neighbors
        op_logger.info(f'In CompareNeighborsOp with {type(neighbors_a)} and {type(neighbors_b)}, k={self.k}, shapes {l_nn.shape}, {m_nn.shape}')
        #TODO clamp values up to 0?

        # Compute metrics for different K values
        metrics: dict[str, list[float]]  = {}
        per_item_metrics: defaultdict[str, list[float]]|None  = defaultdict(list) if self.detailed else None

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
                if per_item_metrics:
                    per_item_metrics[f"recall@{k}"].append(recall)
                    per_item_metrics[f"mrr@{k}"].append(mrr)
                    per_item_metrics[f"jaccard@{k}"].append(jaccard)
                # Update running averages
                metrics.setdefault(f"recall@{k}", []).append(recall)
                metrics.setdefault(f"mrr@{k}", []).append(mrr)
                metrics.setdefault(f"jaccard@{k}", []).append(jaccard)
        # Calculate averages
        avg_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

        return NeighborComparison(
            label_key=neighbors_a.label_key,
            embedding_metric_a=neighbors_a.metric,
            embedding_metric_b=neighbors_b.metric,
            metrics=avg_metrics,
            per_item_metrics=dict(per_item_metrics) if per_item_metrics else None
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes neighbor comparison results and identifies notable patterns.

        Checks for high recall, MRR, or Jaccard similarity values and generates
        appropriate warnings for significant neighbor agreement between distance metrics.
        """
        threshold = 0.5  # Threshold for highlighting high similarity metrics

        # Extract key metrics
        metrics = results.metrics
        label_key = results.label_key
        embedding_metric_a = results.embedding_metric_a
        embedding_metric_b = results.embedding_metric_b

        # Create analysis dict
        analysis = {
            "k_value": self.k,
            "metrics_summary": metrics,
            "warnings": []
        }

        # Check for high similarity metrics
        for metric_name, value in metrics.items():
            if value > threshold:
                metric_type, k = metric_name.split('@')
                Warning.add_warning(
                    analysis=analysis,
                    pred=True,
                    unit="neighbors",
                    warning=f"High neighbor {metric_type} ({value:.3f}) at k={k} between {embedding_metric_a or 'label'} and {embedding_metric_b or 'label'} for {label_key}",
                    label_key=label_key,
                    metric=metric_name,
                    metrics=[embedding_metric_a, embedding_metric_b],
                    value=value,
                    score=2  # Importance score
                )

        return analysis


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
    run_mode = 'process'

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        if 'many_array1d_a' in inputs:
            arrays_a = inputs['many_array1d_a']
        elif 'label_arrays_data' in inputs:
            arrays_a = inputs['label_arrays_data'].label_arrays
        elif 'label_distances' in inputs:
            arrays_a = inputs['label_distances'].label_distances
            #iu = np.triu_indices(label_dists.shape[0], k=1); label_dists=label_dists #TODO?
        else:
            raise NotImplementedError(f'Cannot handle inputs {inputs.keys()} for array A')
        if 'many_array1d_b' in inputs:
            arrays_b = inputs['many_array1d_b']
        elif 'embedding_dims' in inputs:
            arrays_b = inputs['embedding_dims'].dims_matrix
        elif 'embedding_distances' in inputs:
            arrays_b = inputs['embedding_distances'].embedding_distances
        assert arrays_a.ndim == 2
        assert arrays_b.ndim == 2
        assert arrays_a.shape[1] == arrays_b.shape[1], f'Arrays must have same number of columns, got {arrays_a.shape} vs {arrays_b.shape}'

        stats_a = get_array1d_stats(arrays_a)
        stats_b = get_array1d_stats(arrays_b)
        comparisons = {}

        # compare cartesian product of rows
        for i, a in enumerate(arrays_a):
            for j, b in enumerate(arrays_b):
                comparisons[(i,j)] = compare_array1d_stats(
                    a, b, stats_a=stats_a[i], stats_b=stats_b[j]
                )

        return StatsComparison(
            stats_a=stats_a,
            stats_b=stats_b,
            shape_a=arrays_a.shape,
            shape_b=arrays_b.shape,
            comparisons=comparisons,
            n_comparisons=len(comparisons)
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyzes statistical comparison results and identifies notable patterns.

        Checks for high correlations, significant differences, and other statistical patterns
        between the compared arrays.
        """
        comparisons = results.comparisons
        if not comparisons:
            return dict(warnings=[])

        # Extract correlation values and other metrics
        correlations = []
        r2_values = []
        kl_divs = []
        warnings = []

        for (i, j), comp_stats in comparisons.items():
            pearson = comp_stats.get('pearson', 0.0)
            r2 = comp_stats.get('linear_least_square_r2', 0.0)
            kl_div = comp_stats.get('kl_div', float('inf'))

            correlations.append(abs(pearson))
            r2_values.append(r2)
            if not np.isinf(kl_div) and not np.isnan(kl_div):
                kl_divs.append(kl_div)

        # Check for high correlations (threshold: 0.7)
        high_corr_threshold = 0.7
        high_correlations = [c for c in correlations if c > high_corr_threshold]

        analysis = {"warnings": []}
        Warning.add_warning(
            analysis=analysis,
            pred=bool(high_correlations),
            unit="stats_comparison",
            warning=f"High correlation found: {max(high_correlations):.3f} ({len(high_correlations)}/{len(comparisons)} comparisons > {high_corr_threshold})" if high_correlations else "",
            key="correlation",
            value=max(high_correlations) if high_correlations else None,
            count=len(high_correlations),
            total_comparisons=len(comparisons),
            score=3  # High importance
        )
        warnings = analysis["warnings"]

        # Check for high R² values (threshold: 0.6)
        high_r2_threshold = 0.6
        high_r2_values = [r for r in r2_values if r > high_r2_threshold]

        Warning.add_warning(
            analysis=analysis,
            pred=bool(high_r2_values),
            unit="stats_comparison",
            warning=f"High R² found: {max(high_r2_values):.3f} ({len(high_r2_values)}/{len(comparisons)} comparisons > {high_r2_threshold})" if high_r2_values else "",
            key="r2",
            value=max(high_r2_values) if high_r2_values else None,
            count=len(high_r2_values),
            total_comparisons=len(comparisons),
            score=2  # Medium importance
        )

        # Summary statistics
        analysis = dict(
            n_comparisons=len(comparisons),
            correlation_stats=dict(
                mean=np.mean(correlations) if correlations else 0.0,
                max=max(correlations) if correlations else 0.0,
                min=min(correlations) if correlations else 0.0,
                std=np.std(correlations) if correlations else 0.0
            ),
            r2_stats=dict(
                mean=np.mean(r2_values) if r2_values else 0.0,
                max=max(r2_values) if r2_values else 0.0,
                min=min(r2_values) if r2_values else 0.0,
                std=np.std(r2_values) if r2_values else 0.0
            ),
            warnings=warnings
        )

        return analysis


class RunClusteringOp(Op):
    """Run clustering algorithms on embeddings or distance matrices.

    Automatically selects appropriate algorithms based on available inputs:
    - If normalized_embeddings available: MiniBatchKMeans, DBSCAN, GaussianMixture
    - If distances available: AgglomerativeClustering, AffinityPropagation, SpectralClustering
    """
    name = "run_clustering"
    input_types = {
        ("normalized_embeddings",): {},
        ("distances",): {}
    }
    output_types = {"clustering_results"}
    #run_mode = "process"

    @classmethod
    def get_variants(cls, inputs: dict[str, Any], cluster_sizes=(5,20,100)) -> dict[str, Any]|None:
        variants: dict[str, dict[str, Any]] = {}

        # Algorithms that work on raw embeddings
        if "normalized_embeddings" in inputs:
            #TODO when we generate multiple embeddings, this should incorporate those into
            #FIXME re-enable dbscan?
            embedding_algorithms = {
                #"dbscan_0.3": {"algorithm": ClusteringAlgorithm.DBSCAN, "eps": 0.3, "min_samples": 5},
                #"dbscan_0.5": {"algorithm": ClusteringAlgorithm.DBSCAN, "eps": 0.5, "min_samples": 5},
            }
            for k in cluster_sizes:
                embedding_algorithms[f'kmeans_{k}'] = {"algorithm": ClusteringAlgorithm.MINIBATCH_KMEANS, "n_clusters": k, "n_init": 'auto'}
            variants.update(embedding_algorithms)

        # Algorithms that work on distance matrices
        if "distances" in inputs:
            label_key = inputs["distances"].label_key
            distance_algorithms = {
                f"{label_key}-affinity_prop": {
                    "algorithm": ClusteringAlgorithm.AFFINITY_PROPAGATION,
                    'label_key': label_key,
                },
            }
            for k in cluster_sizes:
                distance_algorithms[f"{label_key}-agglomerative_{k}"] = {
                    "algorithm": ClusteringAlgorithm.AGGLOMERATIVE,
                    "n_clusters": k,
                    'label_key':label_key,
                }
            variants.update(distance_algorithms)

        return variants

    def __init__(self, algorithm: ClusteringAlgorithm, label_key='', **params):
        self.algorithm = algorithm
        self.label_key = label_key
        self.params = params
        super().__init__()

    def _get_clusterer(self):
        """Get the appropriate clustering algorithm."""
        match self.algorithm:
            case ClusteringAlgorithm.MINIBATCH_KMEANS:
                return MiniBatchKMeans(random_state=42, n_clusters=self.params["n_clusters"], n_init=self.params.get("n_init", 'auto'))
            case ClusteringAlgorithm.DBSCAN:
                return DBSCAN(eps=self.params["eps"], min_samples=self.params.get("min_samples", 5))
            case ClusteringAlgorithm.AGGLOMERATIVE:
                return AgglomerativeClustering(n_clusters=self.params["n_clusters"], metric='precomputed', linkage='average')
            case ClusteringAlgorithm.AFFINITY_PROPAGATION:
                return AffinityPropagation(affinity='precomputed', random_state=42)
            case _:
                raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        clusterer = self._get_clusterer()

        # Determine input data based on algorithm requirements
        true_labels = None
        label_key = None
        if self.algorithm.is_embedding_based:
            # Use normalized embeddings
            norm_emb = inputs["normalized_embeddings"]
            data = norm_emb.embeddings
            keys = norm_emb.keys
        else:
            # Use distance matrix
            distances_data = inputs["distances"]
            label_key = distances_data.label_key
            keys = distances_data.sub_keys
            if isinstance(distances_data, LabelDistancesData):
                data = distances_data.label_distances
                # Try to get true labels from the labels input if available
                if "labels" in inputs and label_key in inputs["labels"]:
                    label_obj = inputs["labels"][label_key]
                    # Get true labels for the same keys
                    try:
                        if isinstance(label_obj, NumericLabels):
                            true_labels = np.array([label_obj.values[label_obj.ids.index(key)]
                                                  for key in keys if key in label_obj.ids])
                        elif isinstance(label_obj, MulticlassLabels):
                            true_labels = np.array([label_obj.values[label_obj.ids.index(key)]
                                                  for key in keys if key in label_obj.ids])
                        # For multilabel, we'll use the first label as a proxy
                        elif isinstance(label_obj, MultilabelLabels):
                            # Convert to binary for most common label
                            all_labels = []
                            for key in keys:
                                if key in label_obj.ids:
                                    labels_for_key = label_obj.values.get(key, [])
                                    all_labels.extend(labels_for_key)
                            if all_labels:
                                most_common_label = Counter(all_labels).most_common(1)[0][0]
                                true_labels = np.array([int(most_common_label in label_obj.values.get(key, []))
                                                      for key in keys if key in label_obj.ids])
                    except Exception as e:
                        op_logger.warning(f"Could not extract true labels: {e}")
            elif isinstance(distances_data, EmbeddingDistancesData):
                data = distances_data.embedding_distances
            else:
                raise ValueError(f"Unknown distances data type: {type(distances_data)}")

        op_logger.info(f'Running {self.algorithm.value} clustering on {data.shape} data')
        # Fit the clustering algorithm
        cluster_labels = clusterer.fit_predict(data)
        # Extract results
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))  # Exclude noise points (-1)
        centroids = getattr(clusterer, 'cluster_centers_', None)
        inertia = getattr(clusterer, 'inertia_', None)
        return ClusteringResult(
            algorithm=self.algorithm.value,
            labels=cluster_labels,
            n_clusters=n_clusters,
            centroids=centroids,
            parameters=self.params,
            label_key=label_key,
            keys=keys,
            inertia=inertia,
            true_labels=true_labels,
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyze clustering results and compute quality metrics."""
        # Get the data that was clustered
        if self.algorithm.is_embedding_based:
            data = inputs["normalized_embeddings"].embeddings
        else:
            distances_data = inputs["distances"]
            if isinstance(distances_data, LabelDistancesData):
                data = distances_data.label_distances
            elif isinstance(distances_data, EmbeddingDistancesData):
                data = distances_data.embedding_distances
            else:
                data = None
        cluster_labels = results.labels
        n_clusters = results.n_clusters
        n_noise = np.sum(cluster_labels == -1)  # DBSCAN noise points
        true_labels = results.true_labels
        # Compute cluster size statistics
        unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))
        analysis = {
            "algorithm": self.algorithm.value,
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "cluster_sizes": cluster_sizes,
            "largest_cluster_size": int(max(counts)) if len(counts) > 0 else 0,
            "smallest_cluster_size": int(min(counts)) if len(counts) > 0 else 0,
            "true_labels": results.true_labels.tolist() if results.true_labels is not None else None,
            "warnings": []
        }

        # Add supervised metrics if we have true labels
        if true_labels is not None and len(true_labels) > 0:
            # Remove noise points for supervised metrics
            valid_mask = cluster_labels >= 0
            if np.sum(valid_mask) > 1 and len(true_labels) == len(cluster_labels):
                cluster_labels_clean = cluster_labels[valid_mask]
                true_labels_clean = true_labels[valid_mask]

                if len(cluster_labels_clean) > 0 and len(set(true_labels_clean)) > 1:
                    try:
                        ari = adjusted_rand_score(true_labels_clean, cluster_labels_clean)
                        nmi = normalized_mutual_info_score(true_labels_clean, cluster_labels_clean)
                        analysis["adjusted_rand_index"] = float(ari)
                        analysis["normalized_mutual_info"] = float(nmi)

                        # Compute cluster purities
                        cluster_purities = {}
                        for cluster_id in np.unique(cluster_labels_clean):
                            cluster_mask = cluster_labels_clean == cluster_id
                            cluster_true_labels = true_labels_clean[cluster_mask]
                            if len(cluster_true_labels) > 0:
                                unique_vals, counts_vals = np.unique(cluster_true_labels, return_counts=True)
                                purity = max(counts_vals) / len(cluster_true_labels)
                                cluster_purities[int(cluster_id)] = float(purity)

                        analysis["cluster_purities"] = cluster_purities
                        analysis["average_purity"] = float(np.mean(list(cluster_purities.values()))) if cluster_purities else 0.0

                    except Exception as e:
                        op_logger.warning(f"Could not compute supervised clustering metrics: {e}")

        # Compute quality metrics if we have valid clusters and data
        if n_clusters > 1 and data is not None and len(cluster_labels[cluster_labels >= 0]) > 1:
            try:
                # For distance-based algorithms, we need to handle precomputed distances
                if self.algorithm.is_distance_based:
                    # Convert distance matrix to similarity for silhouette score
                    # Use negative distances as similarities (closer = higher similarity)
                    similarities = -data
                    valid_mask = cluster_labels >= 0
                    if np.sum(valid_mask) > 1:
                        sil_score = silhouette_score(similarities[np.ix_(valid_mask, valid_mask)],
                                                   cluster_labels[valid_mask], metric='precomputed')
                        analysis["silhouette_score"] = float(sil_score)
                else:
                    # For embedding-based algorithms
                    valid_mask = cluster_labels >= 0
                    if np.sum(valid_mask) > 1:
                        sil_score = silhouette_score(data[valid_mask], cluster_labels[valid_mask])
                        db_score = davies_bouldin_score(data[valid_mask], cluster_labels[valid_mask])
                        analysis["silhouette_score"] = float(sil_score)
                        analysis["davies_bouldin_score"] = float(db_score)
            except Exception as e:
                op_logger.warning(f"Could not compute clustering quality metrics: {e}")

        # Generate warnings for notable results
        Warning.add_warning(
            analysis=analysis,
            pred=n_clusters == 1,
            unit="clustering",
            warning=f"Clustering produced only 1 cluster with {self.algorithm.value}",
            algorithm=self.algorithm.value,
            issue="single_cluster",
            score=-1  # Negative score for potential error condition
        )

        Warning.add_warning(
            analysis=analysis,
            pred=n_noise > len(cluster_labels) * 0.5,
            unit="clustering",
            warning=f"High noise ratio: {n_noise}/{len(cluster_labels)} points classified as noise",
            algorithm=self.algorithm.value,
            issue="high_noise",
            noise_ratio=n_noise / len(cluster_labels),
            score=-2  # Negative score for potential error condition
        )

        # Check for very unbalanced clusters
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
            Warning.add_warning(
                analysis=analysis,
                pred=imbalance_ratio > 10,
                unit="clustering",
                warning=f"Highly imbalanced clusters: largest/smallest = {imbalance_ratio:.1f}",
                label_key=self.label_key,
                algorithm=self.algorithm.value,
                issue="imbalanced_clusters",
                value=float(imbalance_ratio),
                score=-1  # Negative score for potential error condition
            )

        # High silhouette score warning
        Warning.add_warning(
            analysis=analysis,
            pred="silhouette_score" in analysis and analysis["silhouette_score"] > 0.7,
            unit="clustering",
            warning=f"High silhouette score ({analysis.get('silhouette_score', 0):.3f}) indicates well-separated clusters",
            algorithm=self.algorithm.value,
            metric="silhouette_score",
            value=analysis.get("silhouette_score"),
            score=3
        )

        return analysis


class ClusterLabelAnalysisOp(Op):
    """Analyze clustering results against ground truth labels.

    Computes confusion matrix, cluster purity, and supervised clustering metrics
    like adjusted rand index and normalized mutual information.
    """
    name = "cluster_label_analysis"
    input_types = {
        ("clustering_results", "label_arrays_data"): {
            "consistency_fields": ["label_key"]
        }
    }
    output_types = {"cluster_label_analysis"}

    @classmethod
    def get_variants(cls, inputs: dict[str, Any]) -> dict[str, Any]|None:
        # One variant per label type in label_arrays_data
        label_data = inputs.get("label_arrays_data")
        if not label_data:
            return None

        label_key = label_data.label_key
        return {f"label_key:{label_key}": {"label_key": label_key}}

    def __init__(self, label_key: str, **kw):
        self.label_key = label_key
        super().__init__(**kw)

    def _execute(self, inputs: dict[str, Any]) -> OpResult:
        clustering_result = inputs["clustering_results"]
        label_data = inputs["label_arrays_data"]
        label_key = label_data.label_key

        # Get cluster labels and true labels for the same keys
        cluster_labels = clustering_result.labels
        cluster_keys = clustering_result.keys

        # Find intersection of keys
        common_keys = set(cluster_keys) & set(label_data.sub_keys)
        if not common_keys:
            raise ValueError("No common keys between clustering results and labels")

        # Get indices for common keys
        cluster_indices = [cluster_keys.index(key) for key in common_keys if key in cluster_keys]
        label_indices = [label_data.sub_keys.index(key) for key in common_keys if key in label_data.sub_keys]

        # Extract corresponding labels
        cluster_labels_subset = cluster_labels[cluster_indices]

        # For multilabel/multiclass labels, we'll analyze each label array separately
        # For now, take the first label array (could be extended to handle multiple)
        true_labels_subset = label_data.label_arrays[0][label_indices]  # First label array

        # Convert continuous labels to discrete if needed (for numeric labels)
        if len(np.unique(true_labels_subset)) > 20:  # Likely continuous
            # Discretize into quartiles
            quartiles = np.percentile(true_labels_subset, [25, 50, 75])
            true_labels_discrete = np.digitize(true_labels_subset, quartiles)
        else:
            true_labels_discrete = true_labels_subset.astype(int)

        # Remove noise points (-1) from clustering for supervised metrics
        valid_mask = cluster_labels_subset >= 0
        cluster_labels_clean = cluster_labels_subset[valid_mask]
        true_labels_clean = true_labels_discrete[valid_mask]

        if len(cluster_labels_clean) == 0:
            raise ValueError("No valid cluster assignments (all noise)")

        # Compute metrics
        ari = adjusted_rand_score(true_labels_clean, cluster_labels_clean)
        nmi = normalized_mutual_info_score(true_labels_clean, cluster_labels_clean)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_labels_clean, cluster_labels_clean)

        # Compute cluster purities
        cluster_purities = {}
        dominant_labels = {}

        for cluster_id in np.unique(cluster_labels_clean):
            cluster_mask = cluster_labels_clean == cluster_id
            cluster_true_labels = true_labels_clean[cluster_mask]

            if len(cluster_true_labels) > 0:
                # Find most common true label in this cluster
                unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                dominant_label_idx = np.argmax(counts)
                dominant_label = unique_labels[dominant_label_idx]
                purity = counts[dominant_label_idx] / len(cluster_true_labels)

                cluster_purities[int(cluster_id)] = float(purity)
                dominant_labels[int(cluster_id)] = str(dominant_label)

        return ClusterLabelAnalysis(
            cluster_labels=cluster_labels_subset,
            true_labels=true_labels_subset.reshape(1, -1),  # Keep as 2D for consistency
            label_names=[label_data.label_names[0]],  # First label name
            label_key=label_key,
            confusion_matrix=conf_matrix,
            cluster_purities=cluster_purities,
            dominant_labels=dominant_labels,
            adjusted_rand_index=ari,
            normalized_mutual_info=nmi
        )

    def analyze_results(self, results: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Analyze cluster-label correspondence and identify notable patterns."""
        if results is None or isinstance(results, Exception):
            analysis = {"warnings": []}
            Warning.add_warning(
                analysis=analysis,
                pred=True,
                unit="cluster_analysis",
                warning=f"Error during cluster-label analysis: {results}",
                algorithm=getattr(self, 'algorithm', {}).value if hasattr(getattr(self, 'algorithm', {}), 'value') else '',
                label_key=self.label_key,
                issue="error",
                score=-3  # Negative score for error condition
            )
            return analysis
        ari = results.adjusted_rand_index
        nmi = results.normalized_mutual_info
        purities = results.cluster_purities

        # Compute average purity
        avg_purity = np.mean(list(purities.values())) if purities else 0.0
        max_purity = max(purities.values()) if purities else 0.0
        min_purity = min(purities.values()) if purities else 0.0

        analysis = {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "average_purity": avg_purity,
            "max_purity": max_purity,
            "min_purity": min_purity,
            "n_clusters": len(purities),
            "true_labels": results.true_labels.tolist() if results.true_labels is not None else None,
            "warnings": []
        }

        # Generate warnings for notable results
        Warning.add_warning(
            analysis=analysis,
            pred=ari > 0.5,
            unit="cluster_analysis",
            warning=f"High adjusted rand index ({ari:.3f}) indicates good cluster-label correspondence",
            label_key=self.label_key,
            algorithm=getattr(self, 'algorithm', {}).value if hasattr(getattr(self, 'algorithm', {}), 'value') else '',
            issue="high adjusted_rand_index",
            metric="adjusted_rand_index",
            value=ari,
            score=3
        )

        Warning.add_warning(
            analysis=analysis,
            pred=avg_purity > 0.8,
            unit="cluster_analysis",
            warning=f"High average cluster purity ({avg_purity:.3f}) indicates homogeneous clusters",
            label_key=self.label_key,
            algorithm=getattr(self, 'algorithm', {}).value if hasattr(getattr(self, 'algorithm', {}), 'value') else '',
            issue="high average_purity",
            metric="average_purity",
            value=avg_purity,
            score=2
        )

        # Check for very pure individual clusters
        high_purity_clusters = {k: v for k, v in purities.items() if v > 0.9}
        Warning.add_warning(
            analysis=analysis,
            pred=bool(high_purity_clusters),
            unit="cluster_analysis",
            warning=f"{len(high_purity_clusters)} clusters with >90% purity: {high_purity_clusters}",
            label_key=self.label_key,
            algorithm=getattr(self, 'algorithm', {}).value if hasattr(getattr(self, 'algorithm', {}), 'value') else '',
            issue="high individual_cluster_purity",
            metric="individual_purity",
            clusters=high_purity_clusters,
            score=2
        )

        return analysis


def init_logging(log_names=('tasks', 'perf', 'op', 'results', 'errors', 'eval', 'labels'),
                 stderr_loggers=('op', 'errors', 'eval'),
                 file_mode='w', #FIXME change this to 'a'
                 async_logging = True):
    """initializes all our logging

    Args:
    - log_names: Names of all loggers to initialize
    - stderr_loggers: Names of loggers that should also write to stderr
    - file_mode: Mode to open log files with ('w' for write, 'a' for append)
    - async_logging: Whether to use asynchronous logging handlers
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
        logger.setLevel(logging.DEBUG if name == 'task' else logging.INFO)
        file_handler = logging.FileHandler(f"{log_dir}/{name}.log", mode=file_mode)
        file_handler.setFormatter(formatter)
        handlers = [file_handler]
        # Stderr handler if needed
        if name in stderr_loggers:
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            handlers.append(stderr_handler)
        if async_logging: # Create separate queue and handler for this logger
            log_queue = Queue()
            queue_handler = logging.handlers.QueueHandler(log_queue)
            logger.addHandler(queue_handler)
            # Create and start listener for this logger's queue
            listener = logging.handlers.QueueListener(log_queue, *handlers)
            listener.start()
        else: # Non-async version - add handlers directly to logger
            for handler in handlers:
                logger.addHandler(handler)


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
