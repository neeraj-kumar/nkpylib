"""Tests for nkpylib.search.searcher"""

from __future__ import annotations

import pytest

from nkpylib.search.searcher import (
    SearchCond, Op, OpCond, JoinType, JoinCond, SearchResult, SearchImpl, 
    Searcher, LarkSearcher
)

def test_parse_basic():
    """Test basic condition parsing"""
    # Test different operators
    assert LarkSearcher.parse_cond('name = "John"') == OpCond('name', Op.EQ, 'John')
    assert LarkSearcher.parse_cond('age > 25') == OpCond('age', Op.GT, 25)
    assert LarkSearcher.parse_cond('price >= 99.99') == OpCond('price', Op.GTE, 99.99)
    assert LarkSearcher.parse_cond('status != "deleted"') == OpCond('status', Op.NEQ, 'deleted')
    assert LarkSearcher.parse_cond('name ~ "Jo%"') == OpCond('name', Op.LIKE, 'Jo%')
    assert LarkSearcher.parse_cond('tags : ["red", "blue"]') == OpCond('tags', Op.IN, ['red', 'blue'])
    assert LarkSearcher.parse_cond('embedding ~= [0.1, 0.2]') == OpCond('embedding', Op.CLOSE_TO, [0.1, 0.2])
    assert LarkSearcher.parse_cond('phone ?') == OpCond('phone', Op.EXISTS, None)
    assert LarkSearcher.parse_cond('status !?+') == OpCond('status', Op.IS_NULL, None)

def test_parse_and():
    """Test AND conditions"""
    # Simple AND
    cond = LarkSearcher.parse_cond('name = "John", age > 25')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert cond.conds[1] == OpCond('age', Op.GT, 25)

    # Multiple AND
    cond = LarkSearcher.parse_cond('name = "John", age > 25, status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

def test_parse_or():
    """Test OR conditions"""
    # Simple OR
    cond = LarkSearcher.parse_cond('status = "new" | status = "pending"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('status', Op.EQ, 'new')
    assert cond.conds[1] == OpCond('status', Op.EQ, 'pending')

    # Multiple OR
    cond = LarkSearcher.parse_cond('status = "new" | status = "pending" | status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 3

def test_parse_not():
    """Test NOT conditions"""
    # Simple NOT
    cond = LarkSearcher.parse_cond('!(status = "deleted")')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.NOT
    assert len(cond.conds) == 1
    assert cond.conds[0] == OpCond('status', Op.EQ, 'deleted')

def test_parse_nested():
    """Test nested conditions"""
    # AND with nested OR
    cond = LarkSearcher.parse_cond('name = "John", (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert isinstance(cond.conds[1], JoinCond)
    assert cond.conds[1].join == JoinType.OR

    # Complex nested
    cond = LarkSearcher.parse_cond('name ~ "Jo%", !(status = "deleted"), (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

    # Test optional parentheses
    assert LarkSearcher.parse_cond('(name = "John", age > 25)') == LarkSearcher.parse_cond('name = "John", age > 25')
    assert LarkSearcher.parse_cond('(status = "new" | status = "pending")') == LarkSearcher.parse_cond('status = "new" | status = "pending"')

def test_parse_errors():
    """Test error conditions"""
    with pytest.raises(Exception):
        LarkSearcher.parse_cond('invalid query')

    with pytest.raises(Exception):
        LarkSearcher.parse_cond('!status = "deleted"')  # NOT must use !()

    with pytest.raises(Exception):
        LarkSearcher.parse_cond('name = "John" age > 25')  # Missing comma

    with pytest.raises(Exception):
        LarkSearcher.parse_cond('name @ "John"')  # Invalid operator

    with pytest.raises(Exception):
        LarkSearcher.parse_cond('name = "John")')  # Unmatched parenthesis

    with pytest.raises(Exception):
        r = LarkSearcher.parse_cond('(name = "John"')  # Unmatched parenthesis
        print(f'Parsed condition: {r}')

def test_parse_whitespace():
    """Test whitespace handling"""
    # Extra spaces shouldn't matter
    assert LarkSearcher.parse_cond('name="John"') == LarkSearcher.parse_cond('name = "John"')
    assert LarkSearcher.parse_cond('age>25') == LarkSearcher.parse_cond('age > 25')
    assert LarkSearcher.parse_cond('name="John",age>25') == LarkSearcher.parse_cond('name = "John", age > 25')

def test_unquoted_strings():
    """Test unquoted string values"""
    # Single words don't need quotes
    assert LarkSearcher.parse_cond('name = John') == LarkSearcher.parse_cond('name = "John"')
    assert LarkSearcher.parse_cond('status = active') == LarkSearcher.parse_cond('status = "active"')
    # Lists can mix quoted and unquoted
    assert LarkSearcher.parse_cond('tags : [red, "blue green"]') == OpCond('tags', Op.IN, ['red', 'blue green'])
