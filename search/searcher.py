"""Searcher data classes and utils.

This module provides classes and utilities for building and executing search queries:

Classes:
- SearchCond: Base class for search conditions
- OpCond: Condition comparing a field with a value using an operator
- JoinCond: Combines multiple conditions with AND/OR/NOT logic
- SearchResult: Container for search results with id, score and optional metadata
- SearchImpl: Abstract base class for search implementations

Enums:
- Op: Search operators (=, !=, >, >=, etc.)
- JoinType: Types of condition joins (AND, OR, NOT)

Types:
- Array1D: Type alias for 1D numeric arrays/lists
- Array2D: Type alias for 2D numeric arrays/lists

The search conditions form a tree structure that can be walked and filtered.
SearchImpl provides timing utilities for concrete implementations.
"""


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

    def __hash__(self):
        """Hash this condition for use in sets or dicts."""
        raise NotImplementedError("Subclasses must implement __hash__")


# define enum for operators
class Op(Enum):
    """Ops for search conditions"""
    EQ = '='
    NEQ = '!='
    GT = '>'
    GTE = '>='
    LT = '<'
    LTE = '<='
    LIKE = '~' # sql "like"
    NOT_LIKE = '!~' # sql "not like"
    IN = ':' # key is in value (which is a list)
    NOT_IN = '!:' # key is not in value (which is a list)
    HAS = '@' # key (which is a list) contains value (not a list)
    NOT_HAS = '!@' # key (which is a list) does not contain value (not a list)
    CLOSE_TO = '~=' # key is similar to value (both are vectors)
    EXISTS = '?' # key exists, value ignored
    NOT_EXISTS = '!?' # key doesn't exist, value ignored
    IS_NULL = '!?+' # key is null, value ignored
    IS_NOT_NULL = '?+' # key is not null, value ignored

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

    def __hash__(self):
        """Hash this condition for use in sets or dicts."""
        return hash((self.field, self.op, json.dumps(self.value, sort_keys=True)))


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

    def __hash__(self):
        """Hash this condition for use in sets or dicts."""
        return hash((self.join, tuple(hash(c) for c in self.conds)))


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
            md = str(self.metadata)[:150]
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
