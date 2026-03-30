"""SQL search implementation"""

from __future__ import annotations

import asyncio
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

class SqlSearchImpl(SearchImpl):
    """Search implementation that uses SQL queries"""

