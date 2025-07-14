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
    def __init__(self, obj: list[dict]):
        self.obj = obj

    def search(self, cond: SearchCond, n_results: int=15, **kw) -> list[SearchResult]:
        """Searches our `obj` with given `cond`."""
