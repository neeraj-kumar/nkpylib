"""Tests for nkpylib.search.searcher"""

from __future__ import annotations

import pytest

from nkpylib.search.searcher import SearchCond, Op, OpCond, JoinType, JoinCond, SearchResult, SearchImpl, Searcher

def test_parse_basic():
    """Test basic condition parsing"""
    # Test different operators
    assert Searcher.parse_cond('name = "John"') == OpCond('name', Op.EQ, 'John')
    assert Searcher.parse_cond('age > 25') == OpCond('age', Op.GT, 25)
    assert Searcher.parse_cond('price >= 99.99') == OpCond('price', Op.GTE, 99.99)
    assert Searcher.parse_cond('status != "deleted"') == OpCond('status', Op.NEQ, 'deleted')
    assert Searcher.parse_cond('name ~ "Jo%"') == OpCond('name', Op.LIKE, 'Jo%')
    assert Searcher.parse_cond('tags : ["red", "blue"]') == OpCond('tags', Op.IN, ['red', 'blue'])
    assert Searcher.parse_cond('embedding ~= [0.1, 0.2]') == OpCond('embedding', Op.CLOSE_TO, [0.1, 0.2])
    assert Searcher.parse_cond('phone ?') == OpCond('phone', Op.EXISTS, None)
    assert Searcher.parse_cond('status !?+') == OpCond('status', Op.IS_NULL, None)

def test_parse_and():
    """Test AND conditions"""
    # Simple AND
    cond = Searcher.parse_cond('name = "John", age > 25')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert cond.conds[1] == OpCond('age', Op.GT, 25)

    # Multiple AND
    cond = Searcher.parse_cond('name = "John", age > 25, status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

def test_parse_or():
    """Test OR conditions"""
    # Simple OR
    cond = Searcher.parse_cond('status = "new" | status = "pending"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('status', Op.EQ, 'new')
    assert cond.conds[1] == OpCond('status', Op.EQ, 'pending')

    # Multiple OR
    cond = Searcher.parse_cond('status = "new" | status = "pending" | status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 3

def test_parse_not():
    """Test NOT conditions"""
    # Simple NOT
    cond = Searcher.parse_cond('!(status = "deleted")')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.NOT
    assert len(cond.conds) == 1
    assert cond.conds[0] == OpCond('status', Op.EQ, 'deleted')

def test_parse_nested():
    """Test nested conditions"""
    # AND with nested OR
    cond = Searcher.parse_cond('name = "John", (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert isinstance(cond.conds[1], JoinCond)
    assert cond.conds[1].join == JoinType.OR

    # Complex nested
    cond = Searcher.parse_cond('name ~ "Jo%", !(status = "deleted"), (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

def test_parse_errors():
    """Test error conditions"""
    with pytest.raises(ValueError, match="No operator found"):
        Searcher.parse_cond('invalid query')
    
    with pytest.raises(ValueError, match="NOT expression must be in parentheses"):
        Searcher.parse_cond('!status = "deleted"')

def test_parse_whitespace():
    """Test whitespace handling"""
    # Extra spaces shouldn't matter
    assert Searcher.parse_cond('name="John"') == Searcher.parse_cond('name = "John"')
    assert Searcher.parse_cond('age>25') == Searcher.parse_cond('age > 25')
    assert Searcher.parse_cond('name="John",age>25') == Searcher.parse_cond('name = "John", age > 25')
