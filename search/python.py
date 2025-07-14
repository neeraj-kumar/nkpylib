from __future__ import annotations

import logging

from typing import Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from nkpylib.search.searcher import (
    Array1D,
    JoinCond,
    JoinType,
    Op,
    OpCond,
    SearchCond,
    SearchImpl,
    SearchResult,
)

logger = logging.getLogger(__name__)

class PythonSearch(SearchImpl):
    """A search implementation that searches an in-memory Python data structure."""
    def __init__(self, items: list[dict], id_field:str=''):
        """Initializes with a list of `items` to search.

        You can optionally provide a `id_field` that uniquely identifies each item.
        If not given, then we use the index of the item in the list as its ID.
        """
        self.items = items
        self.id_field = id_field

    def get_id(self, item: dict) -> str:
        """Returns the ID of an item, either from the specified id_field or its index."""
        if self.id_field:
            return str(item.get(self.id_field, ''))
        return str(self.items.index(item))

    def search(self, cond: SearchCond, n_results: int=15, **kw) -> list[SearchResult]:
        """Searches our list of `items` with given `cond`."""
        results = []
        for idx, item in enumerate(self.items):
            if self._matches_condition(item, cond):
                results.append(SearchResult(id=self.get_id(item), score=1.0, obj=item))
            if len(results) >= n_results:
                break
        return results

    def _matches_condition(self, item: dict, cond: SearchCond) -> bool:
        """Check if an item matches the given condition"""
        if isinstance(cond, OpCond):
            return self._matches_op_cond(item, cond)
        elif isinstance(cond, JoinCond):
            return self._matches_join_cond(item, cond)
        else:
            raise NotImplementedError(f"Unknown condition type: {type(cond)}")

    def _matches_join_cond(self, item: dict, cond: JoinCond) -> bool:
        """Check if an item matches a join condition"""
        if cond.join == JoinType.AND:
            return all(self._matches_condition(item, c) for c in cond.conds)
        elif cond.join == JoinType.OR:
            return any(self._matches_condition(item, c) for c in cond.conds)
        elif cond.join == JoinType.NOT:
            return not self._matches_condition(item, cond.conds[0])
        else:
            raise NotImplementedError(f"Unknown join type: {cond.join}")

    def _matches_op_cond(self, item: dict, cond: OpCond) -> bool:
        """Check if an item matches an operator condition"""
        # Handle EXISTS and NULL checks first
        if cond.op in (Op.EXISTS, Op.IS_NOT_NULL):
            return cond.field in item and item[cond.field] is not None
        elif cond.op in (Op.NOT_EXISTS, Op.IS_NULL):
            return cond.field not in item or item[cond.field] is None

        # For all other operators, field must exist and not be None
        if cond.field not in item or item[cond.field] is None:
            return False

        val = item[cond.field]
        
        # Handle each operator type
        if cond.op == Op.EQ:
            return val == cond.value
        elif cond.op == Op.NEQ:
            return val != cond.value
        elif cond.op == Op.GT:
            return val > cond.value
        elif cond.op == Op.GTE:
            return val >= cond.value
        elif cond.op == Op.LT:
            return val < cond.value
        elif cond.op == Op.LTE:
            return val <= cond.value
        elif cond.op == Op.LIKE:
            return self._matches_like(str(val), str(cond.value))
        elif cond.op == Op.NOT_LIKE:
            return not self._matches_like(str(val), str(cond.value))
        elif cond.op == Op.IN:
            return val in cond.value
        elif cond.op == Op.NOT_IN:
            return val not in cond.value
        elif cond.op == Op.CLOSE_TO:
            if not isinstance(val, (list, tuple)) or not isinstance(cond.value, (list, tuple)):
                return False
            # Simple Euclidean distance threshold for now
            return self._euclidean_distance(val, cond.value) < 0.1
        else:
            raise ValueError(f"Unknown operator: {cond.op}")

    def _matches_like(self, val: str, pattern: str) -> bool:
        """Check if value matches SQL LIKE pattern"""
        # Convert SQL LIKE pattern to regex
        import re
        pattern = re.escape(pattern)
        pattern = pattern.replace('%', '.*').replace('_', '.')
        return bool(re.match(f'^{pattern}$', val))

    def _euclidean_distance(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if len(v1) != len(v2):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
