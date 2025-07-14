"""A search implementation that searches an in-memory objects.

This assumes you have a sequence of items that you want to search over. The common case is that they
are dicts, but we don't actually enforce that -- as long as things like `value = item[field]` or
`field in item` work, the code should work fine.

We also support parallel searching through the `n_processes` argument. This forks the process using
a multiprocessing.Pool (which should preserve Copy-On-Write semantics to avoid copying the items
explicitly).
"""
from __future__ import annotations

import logging
import re
from multiprocessing import get_context
from typing import Sequence

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

class ObjSearch(SearchImpl):
    def __init__(self, items: list[dict], id_field:str='', n_processes: int=1):
        """Initializes with a list of `items` to search.

        You can optionally provide a `id_field` that uniquely identifies each item.
        If not given, then we use the index of the item in the list as its ID.

        You can set n_processes (number of parallel processes to use, default 1 = sequential).
        """
        self.items = items
        self.id_field = id_field
        self.n_processes = n_processes

    def get_id(self, item: dict) -> str:
        """Returns the ID of an item, either from the specified id_field or its index."""
        if self.id_field:
            return str(item.get(self.id_field, ''))
        return str(self.items.index(item))

    def search(self, cond: SearchCond, n_results: int=15, **kw) -> list[SearchResult]:
        """Searches our list of `items` with given `cond`.

        Args:
            cond: The search condition to apply
            n_results: Maximum number of results to return
        """
        func = self._search_sequential if self.n_processes <= 1 else self._search_parallel
        ret = func(cond, n_results)
        #TODO rank the results
        return ret[:n_results]  # Limit to n_results


    def _search_sequential(self, cond: SearchCond, n_results: int) -> list[SearchResult]:
        """Sequential search implementation"""
        results = []
        for idx, item in enumerate(self.items):
            if self._matches_condition(item, cond):
                results.append(SearchResult(id=self.get_id(item), score=1.0, metadata=item))
        return results

    def _search_chunk(self, chunk_data: dict) -> list[SearchResult]:
        """Search a chunk of items - called by each worker process"""
        start, end = chunk_data['range']
        results = []
        for item in self.items[start:end]:
            if self._matches_condition(item, chunk_data['cond']):
                results.append(SearchResult(id=self.get_id(item), score=1.0, metadata=item))
        return results

    def _search_parallel(self, cond: SearchCond, n_results: int) -> list[SearchResult]:
        """Parallel search using process pool with fork"""
        # Split items into chunks
        chunk_size = len(self.items) // self.n_processes
        if chunk_size == 0:
            return self._search_sequential(cond, n_results)

        chunks = []
        for i in range(0, len(self.items), chunk_size):
            end = min(i + chunk_size, len(self.items))
            chunks.append({
                'range': (i, end),
                'cond': cond,
            })

        # Use fork context for COW memory sharing
        ctx = get_context('fork')
        with ctx.Pool(processes=self.n_processes) as pool:
            chunk_results = pool.map(self._search_chunk, chunks)

        # Combine results
        all_results = []
        for results in chunk_results:
            all_results.extend(results)
        return all_results

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
        # Handle EXISTS checks
        if cond.op == Op.EXISTS:
            return cond.field in item
        elif cond.op == Op.NOT_EXISTS:
            return cond.field not in item
        # Handle NULL checks
        elif cond.op == Op.IS_NOT_NULL:
            return cond.field in item and item[cond.field] is not None
        elif cond.op == Op.IS_NULL:
            return cond.field in item and item[cond.field] is None
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
        elif cond.op == Op.HAS:
            return cond.value in val
        elif cond.op == Op.NOT_HAS:
            return cond.value not in val
        elif cond.op == Op.CLOSE_TO:
            # deal with this separately
            pass
        else:
            raise NotImplementedError(f"Unknown operator: {cond.op}")

    def _matches_like(self, val: str, pattern: str) -> bool:
        """Check if value matches SQL LIKE pattern"""
        # Convert SQL LIKE pattern to regex
        pattern = re.escape(pattern)
        pattern = pattern.replace('%', '.*').replace('_', '.')
        return bool(re.match(f'^{pattern}$', val))

    def _close_to(self, v1: Array1D, v2: Array1D) -> float:
        """Calculate Euclidean distance between two vectors"""
        if len(v1) != len(v2):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
