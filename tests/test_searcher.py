"""Tests for nkpylib.search.searcher"""

from __future__ import annotations

from typing import Any
import pytest

from nkpylib.search.searcher import (
    JoinCond,
    JoinType,
    LarkSearcher,
    Op,
    OpCond,
    SearchCond,
    Searcher,
    SearchImpl,
    SearchResult,
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

def test_boolean_values():
    """Test boolean values"""
    assert LarkSearcher.parse_cond('is_active = true') == OpCond('is_active', Op.EQ, True)
    assert LarkSearcher.parse_cond('is_deleted = false') == OpCond('is_deleted', Op.EQ, False)
    assert LarkSearcher.parse_cond('verified != true') == OpCond('verified', Op.NEQ, True)


def is_compatible(op: str, val: str) -> bool:
    """Check if an operator and value combination is valid"""
    # Exists/Null operators don't take values
    if op in ('?', '!?', '?+', '!?+'):
        return val == ''
    # List operators only work with list values
    if op in (':', '!:', '~='):
        return val.startswith('[')
    # Everything else is compatible
    return True

def _generate_test_cases():
    """Generate test cases for operator combinations"""
    cases = []
    fields = ['name', 'age', 'status', 'tags', 'price', 'is_active', 'verified', 'embedding']
    operators = [
        ('=', Op.EQ),
        ('!=', Op.NEQ),
        ('>', Op.GT),
        ('>=', Op.GTE),
        ('<', Op.LT),
        ('<=', Op.LTE),
        ('~', Op.LIKE),
        ('!~', Op.NOT_LIKE),
        (':', Op.IN),
        ('!:', Op.NOT_IN),
        ('~=', Op.CLOSE_TO),
        ('?', Op.EXISTS),
        ('!?', Op.NOT_EXISTS),
        ('!?+', Op.IS_NULL),
        ('?+', Op.IS_NOT_NULL),
    ]
    values = [
        ('"John"', 'John'),
        ('active', 'active'),
        ('25', 25),
        ('99.99', 99.99),
        ('[1, 2, 3]', [1, 2, 3]),
        ('["red", "blue"]', ['red', 'blue']),
        ('true', True),
        ('false', False),
    ]

    for f in fields:
        for os, oe in operators:
            for vs, ve in values:
                if is_compatible(os, vs):
                    cases.append((f, os, oe, vs, ve))
    return cases

@pytest.mark.parametrize('field,op_str,op_enum,val_str,val_expected', _generate_test_cases())
def test_op_combinations(field: str, op_str: str, op_enum: Op, val_str: str, val_expected: Any):
    """Test various combinations of fields, operators and values"""
    # Skip value for exists/null operators
    if op_str in ('?', '!?', '?+', '!?+'):
        query = f'{field} {op_str}'
        val_expected = None
    else:
        query = f'{field} {op_str} {val_str}'

    expected = OpCond(field, op_enum, val_expected)
    result = LarkSearcher.parse_cond(query)
    assert result == expected, f"Failed parsing '{query}'"
