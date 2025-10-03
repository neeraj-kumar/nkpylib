import random

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Union

import numpy as np

from nkpylib.ml.ml_types import NUMERIC_TYPES, FLOAT_TYPES, array1d, array2d, nparray2d

# Type aliases used in this module
DistTuple = tuple[str, str, float]
AllDists = dict[str, Any]

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
        # op_logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
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
        # op_logger.debug(f'  Found {len(common)} matching ids in embeddings')
        id_indices = np.asarray(id_indices)
        sub_matrix = matrix[mat_indices, :]
        # op_logger.debug(f'Got sub matrix of shape {sub_matrix.shape}: {sub_matrix}')
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
                pairs[spair] = dist / self.norm_factors[norm_type]

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
        # op_logger.debug(f'Sampled {n_pts} ids for all-pairs distance: {ids[:10]}...')
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
    # op_logger.debug(f'For {(tag_type, key)} got {len(ids)} ids, types: {types.most_common()}')
    # if we have less than `impure_thresh` of other types, ignore them
    if len(types) > 1:
        impure = 1.0 - (n_most / len(ids))
        # op_logger.debug(f'  Most common (purity): {n_most}/{len(ids)} -> {impure}')
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
        # op_logger.debug(f'  Multilabel impurity {impure}: {len(set(ids))}/{len(ids)}')
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
            # op_logger.debug(f'  All values unique, treating as id')
            return None
        return MulticlassLabels(tag_type, key, ids_values)

