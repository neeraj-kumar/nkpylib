"""Tests for nkpylib.search.searcher"""

from __future__ import annotations

import json
import logging
import time

from os.path import dirname, join
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
from nkpylib.search.sql import SqlSearchImpl

logger = logging.getLogger(__name__)

TEST_UNICODE = False

def test_parse_basic():
    """Test basic condition parsing"""
    # Test different operators
    assert parse_query_into_cond('name = "John"') == OpCond('name', Op.EQ, 'John')
    assert parse_query_into_cond('(name = "John")') == OpCond('name', Op.EQ, 'John')
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
    # Simple AND with comma
    cond = parse_query_into_cond('name = "John", age > 25')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert cond.conds[1] == OpCond('age', Op.GT, 25)

    # Simple AND with &
    cond = parse_query_into_cond('name = "John" & age > 25')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 2
    assert cond.conds[0] == OpCond('name', Op.EQ, 'John')
    assert cond.conds[1] == OpCond('age', Op.GT, 25)

    # Multiple AND with comma
    cond = parse_query_into_cond('name = "John", age > 25, status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

    # Multiple AND with &
    cond = parse_query_into_cond('name = "John" & age > 25 & status = "active"')
    assert isinstance(cond, JoinCond)
    assert cond.join == JoinType.AND
    assert len(cond.conds) == 3

    # Mixed AND operators
    cond = parse_query_into_cond('name = "John" & age > 25, status = "active"')
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
        ('name ~ "J"', {'Jane', 'John'}),

        # AND/OR conditions
        ('age > 25, name ~ "J"', {'John'}),
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



def test_sql_searcher(db_path=join(dirname(__file__), 'movie-collection.sqlite')):
    from nkpylib.nkcollections.nkcollections import init_sql_db, db_session, Item
    db = init_sql_db(db_path)
    print(f"Database: {db}")

    ssi = SqlSearchImpl(db=db, table_name='item', other_tables=[
        ('rel', 'src'), ('rel', 'tgt'), ('score', 'id')],
    )
    print(f"SqlSearchImpl: {ssi}")

    # Test queries using compact JSON syntax
    queries = [
        # Basic field filters (like QueryBuilder's source, otype filters)
        dict(q=["source", "=", "imdb"], name="Filter by source", num=1236),
        dict(q=["otype", "=", "content_rating"], name="Filter by object type", ids=[83,115,130,148,425]),
        dict(q=["name", "~", "long"], name="Search name containing 'long'", ids=[543, 1681, 702, 1941]),

        # Numeric filters (like QueryBuilder's ts, added_ts filters)
        dict(q=["ts", "<", 1773727880.02], name="Items before Tuesday, March 17, 2026", ids=[1,2,3,4,5,6,7]),
        dict(q=["id", ":", [2, 3, 4, 9]], name="Items with specific IDs", ids=[2,3,4,9]),

        # JSON field access (like QueryBuilder's md field access)
        dict(q=["md.birth_year", ">=", 1999], name="Items with birth year >= 1999", ids=[48, 1073, 1587, 1690, 1972]),
        dict(q=["md.imdb_id", "=", 'tt1849718'], name="Specific imdb id", ids=[2]),

        # Related table queries (like QueryBuilder's rel filters)
        dict(q=["score.tag", "=", "nkrating"], name="Items with nkrating scores", num=247),
        dict(q=["score.score", "=", 1], name="Items with score=1", ids=[247, 915, 924, 1004, 1217, 1417, 1478, 1776]),
        dict(q=["rel_src.rtype", "=", "has_genre"], name="Items that have a genre", num=250),

        # Complex AND/OR queries (like QueryBuilder's complex filters)
        dict(q=["&", ["otype", "=", "genre"], ["source", "=", "imdb"]], name="Imdb genre", ids=[12, 17, 19, 54, 58, 84, 105, 119, 154, 157, 188, 227, 243, 283, 303, 415, 530, 587, 750, 952, 1014, 1206]),
        dict(q=["&", ["otype", "=", "person"], ["md.birth_year", ">", 1999]], name="Young people", ids=[1073,1690,1972]),
        dict(q=["|", ["source", "=", "letterboxd"], ["source", "=", "movielens"]], name="Letterboxd or movielens items", num=126),

        # NOT queries
        dict(q=["!", ["otype", "=", "movie"]], name="Non-movie items", num=1746),

        # Existence checks
        dict(q=["md.imdb_id", "?"], name="Items with imdb_id metadata", num=250),
        dict(q=["embed_ts", "?+"], name="Items with embeddings", ids=[]),

        # Score-based queries (budget, revenue, ratings stored in Score table)
        dict(q=["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "budget"],
          ["score.score", ">", 100000000]
         ], name="Big budget movies (>$100M)", ids=[559, 892, 1737, 1860]),

        dict(q=["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "revenue"],
          ["score.score", ">", 500000000]
         ], name="High-grossing movies (>$500M)", ids=[1737, 1860]),

        dict(q=["&",
          ["otype", "=", "person"],
          ["rel_tgt.rtype", "=", "director"]
         ], name="People that are directors", num=213),

        # Multiple score conditions using numbered syntax
        dict(q=["&",
          ["otype", "=", "movie"],
          ["score.1.tag", "=", "rating"],
          ["score.1.score", ">=", 7.0],
          ["score.2.tag", "=", "budget"],
          ["score.2.score", "<", 100000]
         ], name="Good low-budget movies", ids=[813, 1152, 1328, 1719, 1953]),

        # Complex OR with nested AND conditions using actual schema
        dict(q=["|",
          ["&", ["otype", "=", "person"], ["name", "~", "Nolan"]],
          ["&", ["otype", "=", "person"], ["name", "~", "Tarantino"]],
          ["&", ["otype", "=", "genre"], ["name", "=", "Sci-Fi"]]
         ], name="Nolan OR Tarantino OR Sci-Fi genre", ids=[1741, 169, 105, 196]),

        # List operations with multiple values
        dict(q=["otype", ":", ["movie", "person", "genre"]], name="Movies, people, or genres", num=1292),
        dict(q=["source", ":", ["letterboxd", "movielens"]], name="Items from specific sources", num=126),
        dict(q=["id", "!:", [1, 2, 3, 100, 200]], name="Exclude specific IDs", num=1991),

        # Numeric comparisons with actual fields
        dict(q=["md.runtime", ":", [120, 150]], name="Movies with specific runtimes", ids=[659, 1305, 1774, 1847]),
        dict(q=["md.runtime", ">=", 220], name="Long movies", ids=[168]),

        # Complex existence and null checks
        dict(q=["&",
          ["md.imdb_id", "?"],
          ["!", ["embed_ts", "?"]]
         ], name="Items with IMDB ID but no embeddings", num=250),

        # Multi-table joins with complex conditions using actual schema
        dict(q=["&",
          ["otype", "=", "movie"],
          ["rel_src.rtype", "=", "actor"],
          ["rel_src.tgt.name", "~", "tom"],
          ["score.tag", "=", "revenue"],
          ["score.score", ">", 10000]
         ], name="$10k+ movies with actors named 'tom'", ids=[1408, 1704]),

        # Revenue vs budget analysis using numbered syntax
        dict(q=["&",
          ["otype", "=", "movie"],
          ["score.1.tag", "=", "revenue"],
          ["score.1.score", ">", 100000000],
          ["score.2.tag", "=", "budget"],
          ["score.2.score", "<", 30000000]
         ], name="Profitable movies (high revenue, low budget)", ids=[997, 1646, 1671, 1697]),

        # Time-based queries
        dict(q=["&",
          ["ts", ">", 1640995200], # Jan 1, 2022
          ["embed_ts", "?+"],
          ["seen_ts", "!?"]
         ], name="Recent items with embeddings but never seen", ids=[]),

        # Complex string matching with actual fields
        dict(q=["&",
          ["name", "~", "and"],
          ["!", ["name", "~", "the"]],
          ["otype", "=", "movie"]
         ], name="Movies with 'and' but not 'the'", ids=[247, 310, 629, 908, 1035, 1188, 1254, 1384, 1426]),

        # Genre-specific queries using relationships
        dict(q=["&",
          ["otype", "=", "genre"],
          ["name", ":", ["Action", "Thriller", "Crime"]]
         ], name="Action, thriller, or crime genres", ids=[5, 7, 13, 17, 19, 22, 54, 57, 69]),

        # Person-specific queries
        dict(q=["&",
          ["otype", "=", "person"],
          ["name", "~", "Tom"]
         ], name="People named Tom", ids=[347, 661, 1147, 1162, 1413, 1435, 1706]),

        # Cross-entity relationship queries using aliases
        dict(q=["&",
          ["rel_src.rtype", "=", "has_genre"],
          ["rel_src.tgt", "=", 52]  # movielens Mystery
         ], name="Movies with specific genre (using alias)", ids=[46, 138, 512, 852, 860, 964, 1074, 1296, 1462, 1470, 1609, 1884]),

        # Movies with specific genre by name (nested query)
        dict(q=["&",
          ["otype", "=", "movie"],
          ["rel_src.rtype", "=", "has_genre"],
          ["rel_src.tgt.name", "=", "War"]
         ], name="Movies with specific genre (by name)", ids=[223, 345, 394, 555, 622, 723, 766, 964, 986, 1481, 1841]),

        # Genres used by specific movies
        dict(q=["&",
          ["otype", "=", "genre"],
          ["source", "=", 'tmdb'],
          ["rel_tgt.rtype", "=", "has_genre"],
          ["rel_tgt.src", ":", [2, 46, 223]]  # Agneepath, Bullet train, Tropic thunder
         ], name="TMDB Genres used by specific movies", ids=[5, 7, 14, 63, 69, 81, 231]),

        # Magic field examples
        dict(q=["&",
          ["otype", "=", "movie"],
          ["$limit", "=", 5],
          ["$order", "=", "-ts"]
         ], name="5 most recent movies", ids=[1994, 1988, 1984, 1975, 1969]),

        dict(q=["&",
          ["otype", "=", "person"],
          ["$limit", "=", 5],
          ["$offset", "=", 20],
          ["$order", "=", "name"]
         ], name="People 21-26 ordered by name", ids=[1178, 1531, 42, 396, 535]),

        dict(q=["&",
          ["otype", "=", "movie"],
          ["md.runtime", "?+"],
          ["$order", "=", "-md.runtime"],
          ["$limit", "=", 3]
         ], name="3 longest movies by runtime", ids=[168, 1685, 1105]),
    ]
    print(f"\nTesting {len(queries)} queries:")
    for i, row in enumerate(queries):
        query, name = row['q'], row['name']
        print(f"\n{i+1}. {name}: {query}")
        results = ssi.search(query, n_results=50000)
        print(f"{len(results)} Results found")
        try:
            if 'num' in row:
                assert len(results) == row['num'], f"Expected {row['num']} results, got {len(results)}"
            elif 'ids' in row:
                ids = row['ids']
                assert set(r.id for r in results) == set(ids), f"For '{name}' expected IDs {ids}, got {len(results)}: {[r.id for r in results]}"
        except Exception:
            for j, r in enumerate(results[:5]):
                with db_session:
                    item = json.dumps(Item[r.id].to_dict(), indent=2)
                logger.info(f"  {j+1}. {r}\n{item}")
            raise


def test_sql_searcher_with_aliases(db_path=join(dirname(__file__), 'movie-collection.sqlite')):
    """Test SqlSearchImpl with different alias configurations"""
    from nkpylib.nkcollections.nkcollections import init_sql_db, db_session, Item
    db = init_sql_db(db_path)
    
    # Test different alias configurations
    alias_test_cases = [
        {
            'name': 'Simple field aliases',
            'aliases': {
                'imdb_id': 'md.imdb_id',
                'runtime': 'md.runtime',
                'birth_year': 'md.birth_year',
            },
            'queries': [
                (["imdb_id", "=", "tt1849718"], "Query by IMDB ID alias", [2]),
                (["runtime", ">=", 220], "Long movies via runtime alias", [168]),
                (["birth_year", ">=", 1999], "Young people via birth_year alias", [48, 1073, 1587, 1690, 1972]),
            ]
        },
        {
            'name': 'Score condition aliases',
            'aliases': {
                'rating': {
                    'type': 'score_condition',
                    'tag_field': 'rating',
                    'score_field': 'score',
                    'table': 'score'
                },
                'budget': {
                    'type': 'score_condition', 
                    'tag_field': 'budget',
                    'score_field': 'score',
                    'table': 'score'
                },
                'revenue': {
                    'type': 'score_condition',
                    'tag_field': 'revenue', 
                    'score_field': 'score',
                    'table': 'score'
                }
            },
            'queries': [
                (["&", ["otype", "=", "movie"], ["rating", ">=", 7.0]], "High rated movies", [813, 1152, 1328, 1719, 1953]),
                (["&", ["otype", "=", "movie"], ["budget", ">", 100000000]], "Big budget movies", [559, 892, 1737, 1860]),
                (["&", ["otype", "=", "movie"], ["revenue", ">", 500000000]], "High grossing movies", [1737, 1860]),
            ]
        },
        {
            'name': 'Multi-score condition aliases',
            'aliases': {
                'profitable': {
                    'type': 'multi_score_condition',
                    'conditions': [
                        {'tag_field': 'revenue', 'op': '>', 'value': 100000000},
                        {'tag_field': 'budget', 'op': '<', 'value': 30000000}
                    ]
                },
                'blockbuster': {
                    'type': 'multi_score_condition', 
                    'conditions': [
                        {'tag_field': 'budget', 'op': '>', 'value': 100000000},
                        {'tag_field': 'revenue', 'op': '>', 'value': 500000000}
                    ]
                }
            },
            'queries': [
                (["&", ["otype", "=", "movie"], ["profitable", "=", True]], "Profitable movies", [997, 1646, 1671, 1697]),
                (["&", ["otype", "=", "movie"], ["blockbuster", "=", True]], "Blockbuster movies", [1737, 1860]),
            ]
        },
        {
            'name': 'Nested relation aliases',
            'aliases': {
                'genre_name': {
                    'type': 'nested_relation',
                    'path': 'rel_src.tgt.name'
                },
                'director_name': {
                    'type': 'nested_relation',
                    'path': 'rel_src.tgt.name'  # Would need different logic for director vs genre
                }
            },
            'queries': [
                (["&", ["otype", "=", "movie"], ["genre_name", "=", "War"]], "War movies via alias", [223, 345, 394, 555, 622, 723, 766, 964, 986, 1481, 1841]),
            ]
        },
        {
            'name': 'Mixed aliases',
            'aliases': {
                # Simple aliases
                'imdb_id': 'md.imdb_id',
                'runtime': 'md.runtime',
                # Score condition alias
                'rating': {
                    'type': 'score_condition',
                    'tag_field': 'rating',
                    'score_field': 'score', 
                    'table': 'score'
                },
                # Multi-condition alias
                'good_and_cheap': {
                    'type': 'multi_score_condition',
                    'conditions': [
                        {'tag_field': 'rating', 'op': '>=', 'value': 7.0},
                        {'tag_field': 'budget', 'op': '<', 'value': 100000}
                    ]
                }
            },
            'queries': [
                (["&", ["otype", "=", "movie"], ["imdb_id", "?"], ["rating", ">=", 8.0]], "High rated movies with IMDB ID", []),
                (["&", ["otype", "=", "movie"], ["runtime", ">=", 120], ["good_and_cheap", "=", True]], "Long, good, cheap movies", [813, 1152, 1328, 1719, 1953]),
            ]
        }
    ]
    
    for test_case in alias_test_cases:
        print(f"\n=== Testing {test_case['name']} ===")
        
        # Create SqlSearchImpl with specific aliases
        ssi = SqlSearchImpl(
            db=db, 
            table_name='item', 
            other_tables=[('rel', 'src'), ('rel', 'tgt'), ('score', 'id')],
            aliases=test_case['aliases']
        )
        
        # Test each query in this alias configuration
        for query, description, expected_ids in test_case['queries']:
            print(f"\n  {description}: {query}")
            results = ssi.search(query, n_results=50000)
            result_ids = [r.id for r in results]
            print(f"  Found {len(results)} results: {result_ids[:10]}{'...' if len(result_ids) > 10 else ''}")
            
            try:
                if expected_ids:
                    assert set(result_ids) == set(expected_ids), \
                        f"Expected IDs {expected_ids}, got {result_ids}"
                    print(f"  ✓ Matched expected {len(expected_ids)} results")
                else:
                    print(f"  ✓ Query executed successfully")
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
                # Show some sample results for debugging
                for j, r in enumerate(results[:3]):
                    with db_session:
                        item = Item[r.id].to_dict()
                        print(f"    Sample {j+1}: {item}")
                raise
