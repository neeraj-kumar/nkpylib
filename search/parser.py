"""Full search parser"""

from __future__ import annotations

import logging

from typing import Any

from lark import Lark, Transformer

from .searcher import SearchCond, Op, OpCond, JoinType, JoinCond

logger = logging.getLogger(__name__)

# Define the grammar for our search language
GRAMMAR = """
?start: expr
?expr: op_cond | and_cond | or_cond | not_cond
and_cond: expr ("," expr)+
or_cond: expr ("|" expr)+
not_cond: "!(" expr ")"
op_cond: field op value
field: CNAME
op: "=" | "!=" | ">" | ">=" | "<" | "<=" | "~" | "!~" | ":" | "!:" | "~=" | "?" | "!?" | "?+" | "!?+"
value: string | number | list
string: ESCAPED_STRING
number: SIGNED_NUMBER
list: "[" [value ("," value)*] "]"

%import common.CNAME
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""

# Create parser once as module variable
parser = Lark(GRAMMAR, parser='lalr', propagate_positions=True)

class SearchTransformer(Transformer):
    """Transforms parse tree into SearchCond objects"""
    def string(self, items):
        logger.debug(f"Parsing string: {items}")
        result = str(items[0][1:-1])  # Remove quotes
        logger.debug(f"String result: {result}")
        return result

    def number(self, items):
        logger.debug(f"Parsing number: {items}")
        val = float(items[0])
        result = int(val) if val.is_integer() else val
        logger.debug(f"Number result: {result}")
        return result

    def list(self, items):
        logger.debug(f"Parsing list: {items}")
        result = list(items)
        logger.debug(f"List result: {result}")
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
        op_str = str(items[0])
        op_map = {
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
            '~=': Op.CLOSE_TO,
            '?': Op.EXISTS,
            '!?': Op.NOT_EXISTS,
            '!?+': Op.IS_NULL,
            '?+': Op.IS_NOT_NULL
        }
        logger.debug(f'Parsing op: {op_str}')
        if op_str not in op_map:
            raise ValueError(f"Unknown operator: {op_str}")
        return op_map[op_str]

    def op_cond(self, items):
        logger.debug(f"Parsing op_cond: {items}")
        field, op, value = items
        result = OpCond(field=field, op=op, value=value)
        logger.debug(f"OpCond result: {result}")
        return result

    def and_cond(self, items):
        logger.debug(f"Parsing and_cond: {items}")
        result = JoinCond(JoinType.AND, list(items))
        logger.debug(f"AND result: {result}")
        return result

    def or_cond(self, items):
        logger.debug(f"Parsing or_cond: {items}")
        result = JoinCond(JoinType.OR, list(items))
        logger.debug(f"OR result: {result}")
        return result

    def not_cond(self, items):
        logger.debug(f"Parsing not_cond: {items}")
        result = JoinCond(JoinType.NOT, [items[0]])
        logger.debug(f"NOT result: {result}")
        return result


def parse_cond(query: str) -> SearchCond:
    """Parse a query string into a SearchCond using Lark parser."""
    logger.debug(f"Parsing query: {query}")
    try:
        tree = parser.parse(query)
        logger.debug(f"Parse tree:\n{tree.pretty()}")
        result = SearchTransformer().transform(tree)
        logger.debug(f"Transformed result:\n{result}")
        return result
    except Exception as e:
        logger.error(f"Failed to parse query: {query}")
        logger.error(f"Error: {e}")
        raise
