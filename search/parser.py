"""Full search parser.

This uses Lark to parse a terse but powerful search language and convert it to a SearchCond.
We support arbitrarily nested clauses joined via AND (, or &), OR (|), and NOT (~) operators. Each clause has a
field, op, and optional value.

See the global `OP_MAP` for the operators available, and the grammar for the syntax.
"""

from __future__ import annotations

import logging

from typing import Any

from lark import Lark, Transformer

from .searcher import SearchCond, Op, OpCond, JoinType, JoinCond

#TODO deal with unicode

logger = logging.getLogger(__name__)

OP_MAP = {
    '=': Op.EQ,
    '!=': Op.NEQ,
    '>': Op.GT,
    '>=': Op.GTE,
    '<': Op.LT,
    '<=': Op.LTE,
    '~': Op.LIKE,
    '!~': Op.NOT_LIKE,
    ':': Op.IN,
    '!:': Op.NOT_IN,
    '@': Op.HAS,
    '!@': Op.NOT_HAS,
    '~=': Op.CLOSE_TO,
    '?': Op.EXISTS,
    '!?': Op.NOT_EXISTS,
    '!?+': Op.IS_NULL,
    '?+': Op.IS_NOT_NULL
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
    """Parse a query string into a `SearchCond` using our Lark parser"""
    query = query.strip()
    logger.debug(f"Parsing query: {query}")
    def parse(q: str):
        if 0:
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
