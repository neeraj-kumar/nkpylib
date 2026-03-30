"""Full search parser.

This uses Lark to parse a terse but powerful search language and convert it to a SearchCond.
We support arbitrarily nested clauses joined via AND (, or &), OR (|), and NOT (~) operators. Each
clause has a field, op, and optional value.

See the global `OP_MAP` for the operators available, and the grammar for the syntax.
"""

from __future__ import annotations

import json
import logging

from typing import Any

from lark import Lark, Transformer

from .searcher import SearchCond, Op, OpCond, JoinType, JoinCond

#TODO deal with unicode

logger = logging.getLogger(__name__)

OP_MAP = {
    '=': Op.EQ,          # exact equality
    '!=': Op.NEQ,        # not equal
    '>': Op.GT,          # greater than
    '>=': Op.GTE,        # greater than or equal
    '<': Op.LT,          # less than
    '<=': Op.LTE,        # less than or equal
    '~': Op.LIKE,        # case-insensitive like (soft equality)
    '!~': Op.NOT_LIKE,   # not like
    ':': Op.IN,          # field in list
    '!:': Op.NOT_IN,     # field not in list
    '@': Op.HAS,         # list has value (inverse of IN)
    '!@': Op.NOT_HAS,    # list does not have value
    '~=': Op.CLOSE_TO,   # close to (for numbers/vectors, within some threshold)
    '?': Op.EXISTS,      # field exists (might or might not be null)
    '!?': Op.NOT_EXISTS, # field does not exist
    '!?+': Op.IS_NULL,   # field doesn't exists or is null
    '?+': Op.IS_NOT_NULL # field exists and is not null
}

#op_cond: "(" field op [value] ")" | field op [value]
# Define the grammar for our search language
GRAMMAR = """
?start: expr
?expr: op_cond | "(" op_cond ")" | and_cond | or_cond | not_cond
and_cond:  "(" expr ("," | "&") expr (("," | "&") expr)* ")"
or_cond:    "(" expr "|" expr ("|" expr)* ")"
not_cond: "!(" expr ")"
op_cond: field op [value]
field: CNAME
op: OP
value: string | unquoted_string | number | list | boolean
string: ESCAPED_STRING
unquoted_string: CNAME
number: SIGNED_NUMBER
list: "[" [value ("," value)*] "]"

boolean: TRUE | FALSE
TRUE: "true"i
FALSE: "false"i
OP: "=" | "!=" | ">" | ">=" | "<" | "<=" | "~" | "!~" | ":" | "!:" | "@" | "!@" | "~=" | "?" | "!?" | "?+" | "!?+"

%import common.CNAME
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""

PROMPT_FMT = """Convert the following natural language query into a structured search query string.
The query language consists of a number of clauses, which can be arbitrarily nested with different
join operators (described below). Each clause should be surrounded with parentheses, except the
outermost one, where they are optional.

Each clause consists of a field, an operator, and an optional value. The field is always a string.
The query language supports these operators (clause format in parens):

Basic Comparisons:
- = : Exact equality (field = value)
- != : Not equal (field != value)
- > : Greater than (field > value)
- >= : Greater than or equal (field >= value)
- < : Less than (field < value)
- <= : Less than or equal (field <= value)

Text Matching:
- ~ : Case-insensitive like/contains (field ~ "pattern")
- !~ : Not like/not contains (field !~ "pattern")

List Operations:
- : : Field (scalar) In list (field : [val1, val2, ...])
- !: : Field (scalar) Not in list (field !: [val1, val2])
- @ : Field (list) Has value (scalar) (field @ value)
- !@ : Field (list) Does not have value (scalar) (field !@ value)

Existence Checks:
- ? : Field exists (field ?)
- !? : Field does not exist (field !?)
- ?+ : Field exists and is not null (field ?+)
- !?+ : Field is null (field !?+)

Join Operations:
- & : AND (cond1 & cond2)
- | : OR (cond1 | cond2)
- !( ) : NOT !(cond)

Values can be:
- Quoted strings: "example"
- Unquoted strings (if no spaces): simple_value
- Numbers: 42 or 3.14
- Lists: [1, 2, "three"]
- Booleans: true or false

Examples:
- name = "John" & age > 25
- status : ["active", "pending"] & !(deleted ?+)
- (price >= 100 | rating > 4.5), category = "electronics"
- title ~ "robot" & year >= 2020
- tags @ "urgent" & !(assigned_to ?)
- department = "sales", (region = "west" | region = "north")

Convert the natural language query to these patterns, preserving the logical structure and using
appropriate operators.
"""

class SearchTransformer(Transformer):
    """Transforms parse tree into SearchCond objects"""
    def string(self, items):
        logger.debug(f"Parsing string: {items}")
        # Get the string without outer quotes
        s = str(items[0][1:-1])
        # Handle escape sequences
        result = s.encode('utf-8').decode('unicode_escape')
        logger.debug(f"String result: {result!r}")
        return result

    def number(self, items):
        logger.debug(f"Parsing number: {items}")
        val = float(items[0])
        result = int(val) if val.is_integer() else val
        logger.debug(f"Number result: {result}")
        return result

    def list(self, items):
        logger.debug(f"Parsing list: {items}")
        # Handle empty list case
        if not items:
            result = []
        elif items[0] is None:
            result = []
        else:
            result = list(items)
        logger.debug(f"List result: {result}")
        return result

    def unquoted_string(self, items):
        logger.debug(f"Parsing unquoted string: {items}")
        result = str(items[0])
        logger.debug(f"Unquoted string result: {result}")
        return result

    def boolean(self, items):
        logger.debug(f"Parsing boolean: {items}")
        result = items[0].lower() == "true"
        logger.debug(f"Boolean result: {result}")
        return result

    def value(self, items):
        logger.debug(f"Parsing value: {items}")
        result = items[0]
        logger.debug(f"Value result: {result}")
        return result

    def field(self, items):
        logger.debug(f"Parsing field: {items}")
        result = str(items[0])
        logger.debug(f"Field result: {result}")
        return result

    def op(self, items):
        logger.debug(f'Parsing op with items: {items!r}')
        if not items:
            raise ValueError("No operator token found in parse tree")
        op_str = str(items[0])
        logger.debug(f'Operator string: {op_str!r}')
        if op_str not in OP_MAP:
            raise ValueError(f"Unknown operator: {op_str!r}")
        result = OP_MAP[op_str]
        logger.debug(f'Mapped to Op: {result}')
        return result

    def op_cond(self, items):
        logger.debug(f"Parsing op_cond: {items}")
        if len(items) == 2:
            field, op = items
            value = None
        else:
            field, op, value = items
        result = OpCond(field=field, op=op, value=value)
        logger.debug(f"OpCond result: {result}")
        return result

    def and_cond(self, items):
        logger.debug(f"Parsing and_cond: {items}")
        # If first item is already an AND condition, extend it
        if isinstance(items[0], JoinCond) and items[0].join == JoinType.AND:
            items[0].conds.extend(items[1:])
            result = items[0]
        else:
            result = JoinCond(JoinType.AND, list(items))
        logger.debug(f"AND result: {result}")
        return result

    def or_cond(self, items):
        logger.debug(f"Parsing or_cond: {items}")
        # If first item is already an OR condition, extend it
        if isinstance(items[0], JoinCond) and items[0].join == JoinType.OR:
            items[0].conds.extend(items[1:])
            result = items[0]
        else:
            result = JoinCond(JoinType.OR, list(items))
        logger.debug(f"OR result: {result}")
        return result

    def not_cond(self, items):
        logger.debug(f"Parsing not_cond: {items}")
        result = JoinCond(JoinType.NOT, [items[0]])
        logger.debug(f"NOT result: {result}")
        return result


# Create parser once as module variable
parser = Lark(GRAMMAR, parser='lalr', propagate_positions=True, transformer=SearchTransformer())

def parse_query_into_cond(query: str) -> SearchCond:
    """Parse a query string into a `SearchCond`.

    This first tries to parse it using JSON, then falls back to the Lark grammar if that fails.
    """
    query = query.strip()
    try: # first try json format parsing
        print(f'Trying to load {query}')
        json_cond = json.loads(query)
        cond = parse_json_into_cond(json_cond)
    except Exception: # if that fails, try query string format parsing
        cond = parse_grammar_str_into_cond(query)
    return cond

def parse_grammar_str_into_cond(query: str) -> SearchCond:
    """Parses a query string in our search grammar into a SearchCond using Lark."""
    def parse(q: str):
        if 0: #FIXME...what was this for?
            tree = parser.parse(q)
            logger.debug(f"Parse tree:\n{tree.pretty()}")
            r = result = SearchTransformer().transform(tree)
            logger.debug(f"Transformed result:{type(result)}\n{result}")
        else:
            result = parser.parse(q)
        return result
    try:
        return parse(query)
    except Exception:
        # add outer parens if necessary
        try:
            return parse('('+query+')')
            return parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} -> {e}")
            raise e

def parse_json_into_cond(data: list | dict) -> SearchCond:
    """Parse a JSON array/dict into a `SearchCond` using compact array/dict formats.

    Supports formats:
    - [field, op, value] for operations
    - [field, op] for operations without values (like existence checks)
    - [join_type, cond1, cond2, ...] for joins where join_type is one of:
      - '&', ',', or 'and' (any case)
      - '|' or 'or' (any case)
      - '!' or 'not' (any case, only takes one condition)
    - {field_op: value, ...} for dict format where field_op is "field" or "field+op"
      - Each field is treated as a separate condition, and all conditions are ANDed together
      - If no operator suffix, defaults to '='
      - Example: {'name~': 'john', 'age>': 5, 'eyes:': ['blue', 'brown']}
        - "name~": "john" -> name LIKE "john"
        - "age>": 5 -> age > 5
        - "eyes:": ["blue", "brown"] -> eyes IN ["blue", "brown"]

    For example:
        ["&", ["title", "~", "machine learning"], ["year", ">=", 2020]]
        ["&", ["!", ["draft", "=", true]], ["published_at", "?+"]]
        {"name~": "john", "age>": 25, "status": "active"}
        ["|", {"name~": "john", "age>": 25}, ["status", "=", "active"]]
    """
    logger.debug(f"Parsing JSON: {data}")
    ANDs = {'&', ',', 'and'}
    ORs = {'|', 'or'}
    NOTs = {'!', 'not'}
    def parse_item(item):
        # Handle dict format in subqueries
        if isinstance(item, dict):
            conditions = []
            for field_op, value in item.items():
                # Parse field and operator from key
                field, op = _parse_field_op(field_op)
                conditions.append(OpCond(field, op, value))
            if len(conditions) == 1:
                return conditions[0]
            else:
                return JoinCond(JoinType.AND, conditions)
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            raise ValueError(f"Invalid format: must be a dict or non-empty list or tuple, got {item} (type: {type(item)})")
        first = item[0]
        fl = first.lower().strip()
        if fl in ANDs | ORs | NOTs: # Join condition
            if fl in ANDs:
                join_type = JoinType.AND
            elif fl in ORs:
                join_type = JoinType.OR
            elif fl in NOTs:
                join_type = JoinType.NOT
            else:
                raise ValueError(f"Unknown join operator: {first}")
            if len(item) < 2:
                raise ValueError(f"Join condition '{first}' requires at least one sub-condition")
            conditions = [parse_item(c) for c in item[1:]]
            return JoinCond(join_type, conditions)
        elif 3 >= len(item) >= 2: # Operation condition: [field, op, value] or [field, op]
            field = item[0]
            op_str = item[1]
            if op_str not in OP_MAP:
                raise ValueError(f"Unknown operator: {op_str}")
            op = OP_MAP[op_str]
            value = item[2] if len(item) > 2 else None
            return OpCond(field, op, value)
        else:
            raise ValueError(f"Invalid array format: {item}")
    result = parse_item(data)
    logger.debug(f"JSON parse result: {result}")
    return result

def _parse_field_op(field_op: str) -> tuple[str, Op]:
    """Parse a field+operator string into separate `(field, operator)`.

    Examples:
    - 'name' -> ('name', Op.EQ)
    - 'name~' -> ('name', Op.LIKE)
    - 'age>' -> ('age', Op.GT)
    - 'eyes:' -> ('eyes', Op.IN)
    """
    # Check operators in order (multi-character first, then single-character)
    ops = ['>=', '<=', '!=', '!~', '!:', '!@', '~=', '!?', '?+', '!?+', '=', '>', '<', '~', ':', '@', '?']
    for op_str in ops:
        if field_op.endswith(op_str):
            field = field_op[:-len(op_str)]
            if field and op_str in OP_MAP:
                return field, OP_MAP[op_str]
    # No operator found, default to equality
    return field_op, Op.EQ
