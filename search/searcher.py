"""Searcher base classes"""

from __future__ import annotations

import json
import logging
import sys
import time

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Sequence, Any, Callable, Generator

import numpy as np

logger = logging.getLogger(__name__)

J = lambda x: pprint(x)

Array1D = list[float] | tuple[float] | np.ndarray
Array2D = list[Array1D] | np.ndarray # for multiple embeddings

@dataclass
class SearchCond:
    """Base class for a search condition"""
    def filter(cls, pred: Callable[[SearchCond], bool]) -> SearchCond|None:
        """Filter this search condition based on a predicate, including all children.

        Returns a new SearchCond that only includes conditions that match the predicate.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def walk(self, fn: Callable[[SearchCond], Any]) -> Generator[Any]:
        """Walk this search condition and call `fn` on each condition, yielding the output."""
        yield fn(self)

    def simplify(self) -> None:
        """Simplify this search condition if possible (in-place)"""
        pass


# define enum for operators
class Op(Enum):
    """Ops for search conditions"""
    EQ = '='
    NEQ = '!='
    GT = '>'
    GTE = '>='
    LT = '<'
    LTE = '<='
    LIKE = 'like' # sql "like"
    NOT_LIKE = 'not like' # sql "not like"
    IN = 'in' # value is a list
    NOT_IN = 'not in' # value is a list
    CLOSE_TO = 'close to' # for vector similarity
    EXISTS = 'exists' # key exists, value ignored
    NOT_EXISTS = 'not exists' # key doesn't exist, value ignored
    IS_NULL = 'is null' # key is null, value ignored
    IS_NOT_NULL = 'is not null' # key is not null, value ignored

    def __str__(self):
        return self.value

@dataclass
class OpCond(SearchCond):
    """A search condition that compares a field with a value using an operator"""
    field: str
    op: Op
    value: Any

    def filter(self, pred: Callable[[SearchCond], bool]) -> SearchCond|None:
        """Filter this condition based on a predicate."""
        if pred(self):
            return self
        return None

    def __repr__(self):
        val = str(self.value)
        if len(val) > 50:
            val = val[:50] + '...'
        return f"{self.field} {self.op} {val}"

class JoinType(Enum):
    """Types of joins for search conditions"""
    AND = 'and'
    OR = 'or'
    NOT = 'not'

@dataclass
class JoinCond(SearchCond):
    """A search condition that joins multiple conditions together"""
    join: JoinType
    conds: Sequence[SearchCond|None]

    def filter(self, pred: Callable[[SearchCond], bool]) -> SearchCond|None:
        """Filter this condition based on a predicate."""
        filtered_conds = [c.filter(pred) for c in self.conds if c is not None]
        filtered_conds = [c for c in filtered_conds if c is not None]
        if not filtered_conds:
            return None
        if len(filtered_conds) == 1:
            ret = filtered_conds[0]
        else:
            ret = JoinCond(join=self.join, conds=filtered_conds)
        ret.simplify()
        return ret

    def walk(self, fn: Callable[[SearchCond], Any]) -> Generator[Any]:
        """Walk this condition and call `fn` on ourself, and on each condition."""
        yield fn(self)
        for c in self.conds:
            if c is not None:
                yield from c.walk(fn)

    def simplify(self) -> None:
        """Simplify this condition by removing None conditions and merging similar ones."""
        # remove None conditions
        self.conds = [c for c in self.conds if c is not None]

    def __repr__(self):
        dlm = self.join.value.upper()
        return '(' + f'\n  {dlm} '.join(repr(c) for c in self.conds) + ')'


@dataclass
class SearchResult:
    """A single search result.

    Guaranteed to have an id and a score, but can also have metadata, document text, and vector.
    """
    id: str | int
    score: float
    metadata: dict[str, Any]|None = None
    document: str| None = None
    vector: Array1D | None = None

    def __repr__(self):
        md = ''
        if self.metadata:
            md = str(self.metadata)[:100]
        return f"SR<{self.id}|{self.score:.3f}|{md}>"


class SearchImpl(ABC):
    """Base class for search implementations.

    This class provides a context manager `timed` to measure execution time of code blocks.
    It updates `timing_times` and `timing_counts` dictionaries with the time taken and count of
    executions for each label.

    These are used to access different kinds of databases (chroma, qdrant, etc), using a common
    interface.
    """
    def __init__(self):
        self.timing_times = {}
        self.timing_counts = {}

    @contextmanager
    def timed(self, label: str):
        """Context manager to time a block of code and update timing statistics."""
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        if label not in self.timing_times:
            self.timing_times[label] = 0.0
            self.timing_counts[label] = 0
        self.timing_times[label] += elapsed_time
        self.timing_counts[label] += 1


class Searcher:
    """Base class for searchers"""
    pass



