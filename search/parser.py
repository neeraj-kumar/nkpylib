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
?expr: basic_cond | and_cond | or_cond | not_cond
and_cond: expr ("," expr)+
or_cond: expr ("|" expr)+
not_cond: "!(" expr ")"
basic_cond: field op value
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
        return str(items[0][1:-1])  # Remove quotes
        
    def number(self, items):
        val = float(items[0])
        if val.is_integer():
            return int(val)
        return val
        
    def list(self, items):
        return list(items)
        
    def value(self, items):
        return items[0]
        
    def field(self, items):
        return str(items[0])
        
    def op(self, items):
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
        return op_map[str(items[0])]
        
    def basic_cond(self, items):
        field, op, value = items
        return OpCond(field=field, op=op, value=value)
        
    def and_cond(self, items):
        return JoinCond(JoinType.AND, list(items))
        
    def or_cond(self, items):
        return JoinCond(JoinType.OR, list(items))
        
    def not_cond(self, items):
        return JoinCond(JoinType.NOT, [items[0]])

def parse_cond(query: str) -> SearchCond:
    """Parse a query string into a SearchCond using Lark parser."""
    tree = parser.parse(query)
    return SearchTransformer().transform(tree)

