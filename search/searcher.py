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


def parse_token(token: str) -> str|int|float|list:
    """Parse a token into the appropriate type."""
    # Handle quoted strings
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    
    # Handle lists
    if token.startswith('[') and token.endswith(']'):
        items = [t.strip() for t in token[1:-1].split(',')]
        return [parse_token(item) for item in items]
        
    # Try numeric conversion
    try:
        if '.' in token:
            return float(token)
        return int(token)
    except ValueError:
        return token

def parse_op(op_str: str) -> Op:
    """Convert string operator to Op enum."""
    op_map = {
        '=': Op.EQ,
        '!=': Op.NEQ,
        '>': Op.GT,
        '>=': Op.GTE,
        '<': Op.LT,
        '<=': Op.LTE,
        'like': Op.LIKE,
        'not like': Op.NOT_LIKE,
        'in': Op.IN,
        'not in': Op.NOT_IN,
        'close to': Op.CLOSE_TO,
        'exists': Op.EXISTS,
        'not exists': Op.NOT_EXISTS,
        'is null': Op.IS_NULL,
        'is not null': Op.IS_NOT_NULL
    }
    return op_map[op_str.lower()]

def parse_cond(query: str) -> SearchCond:
    """Parse a query string into a SearchCond.
    
    Syntax:
    - Basic condition: key op value
    - AND: expr1, expr2
    - OR: expr1 | expr2 
    - NOT: !expr
    
    Examples:
    name = "John"
    age > 25, status = "active"
    category = "books" | category = "magazines"
    !(status = "deleted")
    name like "Jo%", (age < 20 | age > 60)
    """
    def parse_basic(s: str) -> OpCond:
        """Parse a basic condition like 'key op value'"""
        # Split on first operator we find
        for op_str in ['!=', '>=', '<=', '=', '>', '<', 'not like', 'like', 
                      'not in', 'in', 'close to', 'not exists', 'exists',
                      'is not null', 'is null']:
            if op_str in s:
                key, val = s.split(op_str, 1)
                return OpCond(
                    field=key.strip(),
                    op=parse_op(op_str),
                    value=parse_token(val.strip())
                )
        raise ValueError(f"No operator found in: {s}")
    
    def parse_expr(s: str) -> SearchCond:
        """Parse a full expression, handling NOT/AND/OR"""
        s = s.strip()
        
        # Handle empty or None
        if not s:
            return None
            
        # Handle NOT
        if s.startswith('!'):
            if not (s[1] == '(' and s[-1] == ')'):
                raise ValueError(f"NOT expression must be in parentheses: {s}")
            inner = parse_expr(s[2:-1])
            return JoinCond(JoinType.NOT, [inner])
            
        # Remove outer parens if present
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()
            
        # Split on OR first
        parts = []
        current = []
        paren_count = 0
        
        for c in s + '|':  # Add sentinel
            if c == '|' and paren_count == 0:
                if current:
                    parts.append(''.join(current).strip())
                    current = []
            else:
                if c == '(':
                    paren_count += 1
                elif c == ')':
                    paren_count -= 1
                current.append(c)
                
        if len(parts) > 1:
            return JoinCond(JoinType.OR, [parse_expr(p) for p in parts])
            
        # Then split on AND
        parts = []
        current = []
        paren_count = 0
        
        for c in s + ',':  # Add sentinel
            if c == ',' and paren_count == 0:
                if current:
                    parts.append(''.join(current).strip())
                    current = []
            else:
                if c == '(':
                    paren_count += 1
                elif c == ')':
                    paren_count -= 1
                current.append(c)
                
        if len(parts) > 1:
            return JoinCond(JoinType.AND, [parse_expr(p) for p in parts])
            
        # Must be a basic condition
        return parse_basic(s)
        
    return parse_expr(query)


class Searcher:
    """Base class for searchers"""
    pass
