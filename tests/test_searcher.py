"""Tests for nkpylib.search.searcher"""

from __future__ import annotations

import time

from typing import Any

import pytest

from nkpylib.search.parser import parse_query_into_cond
from nkpylib.search.searcher import (
    JoinCond,
    JoinType,
    Op,
    OpCond,
    SearchCond,
    SearchImpl,
    SearchResult,
)
from nkpylib.search.obj import ObjSearch

TEST_UNICODE = False

def test_parse_basic():
    """Test basic condition parsing"""
    # Test different operators
    assert parse_query_into_cond('name = "John"') == OpCond('name', Op.EQ, 'John')
    assert parse_query_into_cond('age > 25') == OpCond('age', Op.GT, 25)
    assert parse_query_into_cond('price >= 99.99') == OpCond('price', Op.GTE, 99.99)
    assert parse_query_into_cond('status != "deleted"') == OpCond('status', Op.NEQ, 'deleted')
    assert parse_query_into_cond('name ~ "Jo%"') == OpCond('name', Op.LIKE, 'Jo%')
    assert parse_query_into_cond('tags : ["red", "blue"]') == OpCond('tags', Op.IN, ['red', 'blue'])
    assert parse_query_into_cond('embedding ~= [0.1, 0.2]') == OpCond('embedding', Op.CLOSE_TO, [0.1, 0.2])
    assert parse_query_into_cond('phone ?') == OpCond('phone', Op.EXISTS, None)
    assert parse_query_into_cond('status !?+') == OpCond('status', Op.IS_NULL, None)

def test_parse_and():
    """Test AND conditions"""
    # Simple AND
    cond = parse_query_into_cond('name = "John", age > 25')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert cond.conds[1] == OpCond('age', Op.GT, 25)

    # Multiple AND
    cond = parse_query_into_cond('name = "John", age > 25, status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

def test_parse_or():
    """Test OR conditions"""
    # Simple OR
    cond = parse_query_into_cond('status = "new" | status = "pending"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('status', Op.EQ, 'new')
    assert cond.conds[1] == OpCond('status', Op.EQ, 'pending')

    # Multiple OR
    cond = parse_query_into_cond('status = "new" | status = "pending" | status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.OR
    assert len(cond.conds) == 3

def test_parse_not():
    """Test NOT conditions"""
    # Simple NOT
    cond = parse_query_into_cond('!(status = "deleted")')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.NOT
    assert len(cond.conds) == 1
    assert cond.conds[0] == OpCond('status', Op.EQ, 'deleted')

def test_parse_nested():
    """Test nested conditions"""
    # AND with nested OR
    cond = parse_query_into_cond('name = "John", (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert isinstance(cond.conds[1], JoinCond)
    assert cond.conds[1].join == JoinType.OR

    # Complex nested
    cond = parse_query_into_cond('name ~ "Jo%", !(status = "deleted"), (age < 20 | age > 60)')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

    # Test optional parentheses
    assert parse_query_into_cond('(name = "John", age > 25)') == parse_query_into_cond('name = "John", age > 25')
    assert parse_query_into_cond('(status = "new" | status = "pending")') == parse_query_into_cond('status = "new" | status = "pending"')

def test_parse_errors():
    """Test error conditions"""
    with pytest.raises(Exception):
        parse_query_into_cond('invalid query')

    if 0: #FIXME
        with pytest.raises(Exception):
            parse_query_into_cond('!status = "deleted"')  # NOT must use !()

    with pytest.raises(Exception):
        parse_query_into_cond('name = "John" age > 25')  # Missing comma

    with pytest.raises(Exception):
        cond = parse_query_into_cond('name ^ "John"')  # Invalid operator

    with pytest.raises(Exception):
        parse_query_into_cond('name = "John")')  # Unmatched parenthesis

    with pytest.raises(Exception):
        r = parse_query_into_cond('(name = "John"')  # Unmatched parenthesis
        print(f'Parsed condition: {r}')

def test_parse_whitespace():
    """Test whitespace handling"""
    # Extra spaces shouldn't matter
    assert parse_query_into_cond('name="John"') == parse_query_into_cond('name = "John"')
    assert parse_query_into_cond('age>25') == parse_query_into_cond('age > 25')
    assert parse_query_into_cond('name="John",age>25') == parse_query_into_cond('name = "John", age > 25')

def test_unquoted_strings():
    """Test unquoted string values"""
    # Single words don't need quotes
    assert parse_query_into_cond('name = John') == parse_query_into_cond('name = "John"')
    assert parse_query_into_cond('status = active') == parse_query_into_cond('status = "active"')
    # Lists can mix quoted and unquoted
    assert parse_query_into_cond('tags : [red, "blue green"]') == OpCond('tags', Op.IN, ['red', 'blue green'])

def test_boolean_values():
    """Test boolean values"""
    assert parse_query_into_cond('is_active = true') == OpCond('is_active', Op.EQ, True)
    assert parse_query_into_cond('is_deleted = false') == OpCond('is_deleted', Op.EQ, False)
    assert parse_query_into_cond('verified != true') == OpCond('verified', Op.NEQ, True)


def test_python_search():
    """Test ObjSearch implementation"""
    data = [
        {'name': 'John', 'age': 30, 'tags': ['a', 'b'], 'status': 'active', 'zodiac': 'gemini'},
        {'name': 'Jane', 'age': 25, 'tags': ['b', 'c'], 'status': None, 'zodiac': 'libra'},
        {'name': 'Bob', 'age': 35, 'tags': ['a', 'c'], 'status': 'inactive', 'zodiac': 'scorpio'},
        {'name': 'Alice', 'age': 28, 'status': None, 'zodiac': 'virgo'},  # No tags
    ]
    searcher = ObjSearch(data, id_field='name')

    test_cases = [
        # Basic operators
        ('age > 30', {'Bob'}),
        ('name ~ "J%"', {'Jane', 'John'}),

        # AND/OR conditions
        ('age > 25, name ~ "J%"', {'John'}),
        ('age < 30 | age > 33', {'Alice', 'Bob', 'Jane'}),

        # EXISTS/NULL checks
        ('tags ?', {'Bob', 'Jane', 'John'}),
        ('tags !?', {'Alice'}),
        ('tags ?+', {'Bob', 'Jane', 'John'}), # exists and is not null

        # NULL checks
        ('status !?+', {'Jane', 'Alice'}),  # is null
        ('status ?+', {'John', 'Bob'}),  # is not null

        # IN/NOT_IN operators
        ('zodiac : ["gemini", "scorpio"]', {'Bob', 'John'}),
        ('name !: ["John", "Jane"]', {'Alice', 'Bob'}),

        # HAS/NOT_HAS operators
        ('tags @ a', {'Bob', 'John'}),
        ('tags !@ "b"', {'Bob'}), #TODO note that alice doesn't have tags at all
    ]

    # Test parallel search
    t0 = time.time()
    parallel_results = searcher.search(parse_query_into_cond('age > 25'), n_processes=2)
    t1 = time.time()
    sequential_results = searcher.search(parse_query_into_cond('age > 25'), n_processes=1)
    t2 = time.time()
    assert {r.id for r in parallel_results} == {r.id for r in sequential_results}
    print(f'Parallel search time: {t1 - t0:.3f}s, Sequential search time: {t2 - t1:.3f}s')

    for query, expected_names in test_cases:
        cond = parse_query_into_cond(query)
        print(f'Parsed query {query} -> {cond}')
        result_names = {r.id for r in searcher.search(cond)}
        assert result_names == expected_names, \
            f"Query '{query}' returned {result_names}, expected {expected_names}"


def is_compatible(op: str, val: str) -> bool:
    """Check if an operator and value combination is valid"""
    # Exists/Null operators don't take values
    if op in ('?', '!?', '?+', '!?+'):
        return val == ''
    # List operators only work with list values
    if op in (':', '!:', '~='):
        return val.startswith('[')
    # Simple operators and HAS operators don't work with lists
    if op in ('=', '!=', '>', '>=', '<', '<=', '~', '!~', '@', '!@'):
        return not val.startswith('[')
    # Everything else is compatible
    return True

def _generate_test_cases():
    """Generate test cases for operator combinations"""
    cases = []
    # Field names with various challenges
    fields = [
        'name',  # simple
        'user_id',  # underscore
        'firstName',  # camelCase
        'API_KEY',  # uppercase with underscore
        'email2',  # with number
        'has_value',  # looks like keyword
        'true_col',  # looks like boolean
        'null_field',  # looks like null
        'in_array',  # looks like operator
        '_private',  # leading underscore
        'x',  # single char
        'very_very_very_long_field_name_that_goes_on',  # very long
    ]

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

    # Values with various challenges
    values = [
        # Strings with special characters
        ('"John Doe"', 'John Doe'),  # space
        ('"O\'Reilly"', "O'Reilly"),  # single quote
        ('"tab\there"', 'tab\there'),  # tab
        (r'"new\nline"', 'new\nline'),  # newline
        ('simple', 'simple'),  # unquoted simple
        ('has_underscore', 'has_underscore'),  # unquoted with underscore
        ('hasNumbers123', 'hasNumbers123'),  # unquoted with numbers

        # Numbers
        ('0', 0),  # zero
        ('-1', -1),  # negative
        ('3.14159', 3.14159),  # pi
        ('1e6', 1000000.0),  # scientific notation
        ('-0.0001', -0.0001),  # small decimal
        ('9999999999', 9999999999),  # large number

        # Lists
        ('[]', []),  # empty
        ('[1]', [1]),  # single item
        ('[1, 2, 3]', [1, 2, 3]),  # numbers
        ('["a", "b", "c"]', ['a', 'b', 'c']),  # strings
        ('[1, "two", 3.14]', [1, 'two', 3.14]),  # mixed
        ('["spaces here", "and,comma"]', ['spaces here', 'and,comma']),  # complex strings
        ('[" leading", "trailing ", "  both  "]', [' leading', 'trailing ', '  both  ']),  # whitespace

        # Booleans
        ('true', True),
        ('false', False),
        ('TRUE', True),  # uppercase
        ('False', False),  # mixed case
    ]
    if TEST_UNICODE:
        values += [
            ('Unicode \u2605 Star', 'Unicode ★ Star'),  # unicode without quotes
            ('"∑∏∐∅∈∉√"', '∑∏∐∅∈∉√'),  # math symbols
            ('"中文"', '中文'),  # Chinese
            ('"🌟 emoji"', '🌟 emoji'),  # emoji
            ('["★", "∑", "🌟"]', ['★', '∑', '🌟']),  # unicode in list
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
    result = parse_query_into_cond(query)
    assert result == expected, f"Failed parsing '{query}'"
