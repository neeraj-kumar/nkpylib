"""SQL search implementation.

Because sql is so complex and flexible, this search implementation has many strong constraints to
make it more tractable and fit more easily in the general searcher framework.

The key limitation is that we only return results from a single table. The queries can include other
tables, but they must all reference this primary table.

The result of a search is always the primary key of the main table, and an optional score (which is
always 1.0 for now).
"""

from __future__ import annotations

import asyncio
import logging
import re
from multiprocessing import get_context
from typing import Sequence

from pony.orm import Database, db_session

from nkpylib.search.searcher import (
    Array1D,
    JoinCond,
    JoinType,
    Op,
    OpCond,
    SearchCond,
    SearchImpl,
    SearchResult,
)

logger = logging.getLogger(__name__)

class SqlSearchImpl(SearchImpl):
    """Search implementation that uses SQL queries with automatic schema discovery"""
    def __init__(self,
                 *,
                 db: Database,
                 table_name: str,
                 id_field: str = 'id',
                 other_tables: list[tuple[str, str]] | None = None):
        """Init with a Pony ORM `Database` instance and the `table_name` to return results from.

        The `id_field` is the primary key field in the main table (default 'id').

        You can also pass in other tables that reference the primary table. Pass these as a list of
        tuples that contain the table name and the foreign key field that references the primary
        table. You can pass in the same table multiple times if it contains multiple foreign keys to
        the primary table.
        """
        super().__init__()
        self.db = db
        self.table_name = table_name
        self.id_field = id_field
        self.other_tables = other_tables or []

        # Generate aliases for other tables
        self.table_aliases = self._generate_aliases(self.other_tables)

        tables = list({table_name} | {t[0] for t in self.other_tables})

        # Auto-discover schema
        self.table_json_fields = {table: self._discover_json_fields(table) for table in tables}
        logger.info(f"Discovered JSON fields: {self.table_json_fields}")
        logger.info(f"Generated table aliases: {self.table_aliases}")

        # Initialize magic field storage
        self._reset_magic_fields()

    def _generate_aliases(self, other_tables: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """Generate aliases automatically based on table and fk_field"""
        from collections import Counter

        # Count how many times each table appears
        table_counts = Counter(table for table, fk_field in other_tables)

        aliases = {}
        for table, fk_field in other_tables:
            if table_counts[table] > 1:
                # Multiple entries for this table - use table_fkfield format
                alias = f"{table}_{fk_field}"
            else:
                # Single entry - just use table name
                alias = table
            aliases[(table, fk_field)] = alias

        return aliases

    def _reset_magic_fields(self):
        """Reset magic field values for a new query"""
        self._query_limit = None
        self._query_offset = None
        self._query_order = None

    def _discover_json_fields(self, table: str) -> set[str]:
        """Find columns in `table` that contain JSON data.

        This is only tested on sqlite for now.
        """
        with db_session:
            try:
                result = self.db.execute(f"PRAGMA table_info({table})")
                return {row[1] for row in result if row[2].upper() == 'JSON'}
            except Exception as e:
                logger.warning(f"Could not discover JSON fields in {table}: {e}")
                return set()

    async def _async_search(self, cond: SearchCond, n_results: int = 15, **kw) -> list[SearchResult]:
        """Execute search using SQL queries"""
        with db_session:
            # Reset magic fields and parameter counter for each search
            self._reset_magic_fields()
            self._param_counter = 0
            where_clause, param_dict, joins = self._build_where_clause(cond)

            # Use magic field values or defaults
            limit = self._query_limit if self._query_limit is not None else n_results
            offset = self._query_offset if self._query_offset is not None else 0
            order = self._query_order
            # Build complete SQL query
            join_clause = ' '.join(joins) if joins else ''
            where_part = f"WHERE {where_clause}" if where_clause else ""
            # Build ORDER BY clause
            order_clause = ""
            if order:
                if order.startswith('-'):
                    direction = "DESC"
                    field = order[1:]
                else:
                    direction = "ASC"
                    field = order
                # Handle JSON field ordering
                if '.' in field:
                    base_field, *path_parts = field.split('.')
                    if base_field in self.table_json_fields.get(self.table_name, set()):
                        json_path = '$.' + '.'.join(path_parts)
                        path_param = f"order_path_{self._param_counter}"
                        self._param_counter += 1
                        param_dict[path_param] = json_path
                        order_clause = f"ORDER BY json_extract({self.table_name}.{base_field}, ${path_param}) {direction}"
                    else:
                        order_clause = f"ORDER BY {self.table_name}.{field} {direction}"
                else:
                    order_clause = f"ORDER BY {self.table_name}.{field} {direction}"
            sql = f"""
            SELECT DISTINCT {self.table_name}.{self.id_field}
            FROM {self.table_name}
            {join_clause}
            {where_part}
            {order_clause}
            LIMIT $limit OFFSET $offset
            """
            param_dict['limit'] = limit
            param_dict['offset'] = offset
            logger.info(f"Executing SQL: {sql}")
            logger.info(f"With params: {param_dict}")
            cursor = self.db.execute(sql, {}, param_dict)
            rows = list(cursor)
            results = [self._row_to_result(row) for row in rows]
            return results

    def _build_where_clause(self, cond: SearchCond) -> tuple[str, dict, list]:
        """Build WHERE clause, parameters dict, and JOINs from SearchCond"""
        if isinstance(cond, OpCond):
            where_part, params, joins = self._build_op_clause(cond)
            return where_part, params, joins
        elif isinstance(cond, JoinCond):
            return self._build_join_clause(cond)
        else:
            return "", {}, []

    def _next_param_name(self) -> str:
        """Generate next parameter name and increment counter"""
        self._param_counter += 1
        return f"p{self._param_counter}"

    def _build_operator_condition(self, where_clause: str, cond: OpCond) -> tuple[str, dict]:
        """Build the SQL condition and parameters for any operator"""
        match cond.op:
            case Op.EQ:
                param = self._next_param_name()
                return f"{where_clause} = ${param}", {param: cond.value}
            case Op.NEQ:
                param = self._next_param_name()
                return f"{where_clause} != ${param}", {param: cond.value}
            case Op.GT:
                param = self._next_param_name()
                return f"CAST({where_clause} AS REAL) > ${param}", {param: cond.value}
            case Op.GTE:
                param = self._next_param_name()
                return f"CAST({where_clause} AS REAL) >= ${param}", {param: cond.value}
            case Op.LT:
                param = self._next_param_name()
                return f"CAST({where_clause} AS REAL) < ${param}", {param: cond.value}
            case Op.LTE:
                param = self._next_param_name()
                return f"CAST({where_clause} AS REAL) <= ${param}", {param: cond.value}
            case Op.LIKE:
                param = self._next_param_name()
                return f"{where_clause} LIKE ${param}", {param: f"%{cond.value}%"}
            case Op.NOT_LIKE:
                param = self._next_param_name()
                return f"{where_clause} NOT LIKE ${param}", {param: f"%{cond.value}%"}
            case Op.IN:
                params = {}
                placeholders = []
                for val in cond.value:
                    param = self._next_param_name()
                    params[param] = val
                    placeholders.append(f"${param}")
                return f"{where_clause} IN ({','.join(placeholders)})", params
            case Op.NOT_IN:
                params = {}
                placeholders = []
                for val in cond.value:
                    param = self._next_param_name()
                    params[param] = val
                    placeholders.append(f"${param}")
                return f"{where_clause} NOT IN ({','.join(placeholders)})", params
            case Op.EXISTS:
                return f"{where_clause} IS NOT NULL", {}
            case Op.NOT_EXISTS:
                return f"{where_clause} IS NULL", {}
            case Op.IS_NULL:
                return f"{where_clause} IS NULL", {}
            case Op.IS_NOT_NULL:
                return f"{where_clause} IS NOT NULL", {}
            case _:
                raise NotImplementedError(f"Operator {cond.op} not implemented")

    def _build_json_condition(self, table_alias: str, json_field: str, path_parts: list[str], cond: OpCond) -> tuple[str, dict]:
        """Build JSON field condition with proper parameterization"""
        json_path = '$.' + '.'.join(path_parts)
        path_param = self._next_param_name()
        where_clause = f"json_extract({table_alias}.{json_field}, ${path_param})"
        params = {path_param: json_path}

        condition_sql, condition_params = self._build_operator_condition(where_clause, cond)
        params.update(condition_params)
        return condition_sql, params

    def _build_op_clause(self, cond: OpCond) -> tuple[str, dict, list]:
        """Build SQL for a single operation condition"""
        field = cond.field
        joins_needed = []

        # Handle magic fields first
        if field.startswith('$'):
            if field == '$limit':
                self._query_limit = int(cond.value)
                return "", {}, []  # Don't add to WHERE clause
            elif field == '$offset':
                self._query_offset = int(cond.value)
                return "", {}, []
            elif field == '$order':
                self._query_order = str(cond.value)
                return "", {}, []
            else:
                raise ValueError(f"Unknown magic field: {field}")

        # Handle table references (numbered, aliased, or nested)
        if '.' in field:
            parts = field.split('.')
            base_field = parts[0]

            # Check for numbered table reference pattern: table.number.field
            if len(parts) >= 3 and parts[1].isdigit():
                table_name = base_field
                table_number = parts[1]
                remaining_parts = parts[2:]

                # Check if this is a known related table
                related_table_info = None
                for table, fk_field in self.other_tables:
                    if table == table_name:
                        related_table_info = (table, fk_field)
                        break

                if related_table_info:
                    table, fk_field = related_table_info
                    alias = f"{table}_{table_number}"
                    joins_needed.append(f"JOIN {table} AS {alias} ON {alias}.{fk_field} = {self.table_name}.{self.id_field}")
                    
                    if len(remaining_parts) == 1:
                        # Simple field in related table (e.g., score.1.tag)
                        where_clause = f"{alias}.{remaining_parts[0]}"
                        condition_sql, params = self._build_operator_condition(where_clause, cond)
                        return condition_sql, params, joins_needed
                    else:
                        # Check if this is JSON field access in related table
                        json_field = remaining_parts[0]
                        if json_field in self.table_json_fields.get(table, set()):
                            # JSON field in related table
                            condition_sql, params = self._build_json_condition(alias, json_field, remaining_parts[1:], cond)
                            return condition_sql, params, joins_needed
                        else:
                            # Regular field in related table
                            where_clause = f"{alias}.{json_field}"
                            condition_sql, params = self._build_operator_condition(where_clause, cond)
                            return condition_sql, params, joins_needed
                else:
                    raise ValueError(f"Unknown numbered table reference: {table_name}")

            # Check for aliased table reference pattern: alias.field or alias.nested.field
            elif base_field in self.table_aliases.values():
                # Find the table info for this alias
                related_table_info = None
                for (table, fk_field), alias in self.table_aliases.items():
                    if alias == base_field:
                        related_table_info = (table, fk_field, alias)
                        break

                if related_table_info:
                    table, fk_field, alias = related_table_info
                    remaining_parts = parts[1:]

                    # Check for nested field access like rel_src.tgt.name
                    if len(remaining_parts) >= 2 and remaining_parts[0] in ['src', 'tgt']:
                        # This is a nested reference to another item through a relationship
                        rel_field = remaining_parts[0]  # 'src' or 'tgt'
                        target_field = remaining_parts[1]  # 'name', 'otype', etc.

                        # Create joins: main_table -> rel_table -> target_item
                        target_alias = f"{alias}_target"
                        joins_needed.append(f"JOIN {table} AS {alias} ON {alias}.{fk_field} = {self.table_name}.{self.id_field}")
                        joins_needed.append(f"JOIN {self.table_name} AS {target_alias} ON {target_alias}.{self.id_field} = {alias}.{rel_field}")

                        # Handle JSON field access in target item
                        if len(remaining_parts) > 2:
                            json_field = target_field
                            if json_field in self.table_json_fields.get(self.table_name, set()):
                                condition_sql, params = self._build_json_condition(target_alias, json_field, remaining_parts[2:], cond)
                                return condition_sql, params, joins_needed
                            else:
                                where_clause = f"{target_alias}.{json_field}"
                        else:
                            # Simple field in target item
                            where_clause = f"{target_alias}.{target_field}"

                        # Build condition based on operator for the target field
                        condition_sql, params = self._build_operator_condition(where_clause, cond)
                        return condition_sql, params, joins_needed
                    else:
                        # Regular aliased table field access
                        joins_needed.append(f"JOIN {table} AS {alias} ON {alias}.{fk_field} = {self.table_name}.{self.id_field}")

                    if len(remaining_parts) == 1:
                        # Simple field in related table (e.g., score.1.tag)
                        where_clause = f"{alias}.{remaining_parts[0]}"
                        condition_sql, params = self._build_operator_condition(where_clause, cond)
                        return condition_sql, params, joins_needed
                    else:
                        # Check if this is JSON field access in related table
                        json_field = remaining_parts[0]
                        if json_field in self.table_json_fields.get(table, set()):
                            # JSON field in related table
                            condition_sql, params = self._build_json_condition(alias, json_field, remaining_parts[1:], cond)
                            return condition_sql, params, joins_needed
                        else:
                            # Regular field in related table
                            where_clause = f"{alias}.{json_field}"
                            condition_sql, params = self._build_operator_condition(where_clause, cond)
                            return condition_sql, params, joins_needed
                else:
                    raise ValueError(f"Unknown numbered table reference: {table_name}")

            # Handle regular JSON field access (e.g., md.stats.n_images)
            elif base_field in self.table_json_fields.get(self.table_name, set()):
                # Handle JSON field access
                path_parts = parts[1:]
                condition_sql, params = self._build_json_condition(self.table_name, base_field, path_parts, cond)
                return condition_sql, params, joins_needed

            else:
                # Check if this is a field in a related table (legacy support)
                related_table = None
                fk_field = None
                for table_name, foreign_key in self.other_tables:
                    if base_field == table_name:
                        related_table = table_name
                        fk_field = foreign_key
                        break

                if related_table:
                    # Handle related table field access (legacy)
                    joins_needed.append(f"JOIN {related_table} ON {related_table}.{fk_field} = {self.table_name}.{self.id_field}")
                    path_parts = parts[1:]

                    if len(path_parts) == 1:
                        # Simple field in related table
                        where_clause = f"{related_table}.{path_parts[0]}"
                    else:
                        # Check if this is JSON field access in related table
                        json_field = path_parts[0]
                        if json_field in self.table_json_fields.get(related_table, set()):
                            # JSON field in related table - use refactored method
                            condition_sql, params = self._build_json_condition(related_table, json_field, path_parts[1:], cond)
                            return condition_sql, params, joins_needed
                        else:
                            # Regular field in related table
                            where_clause = f"{related_table}.{json_field}"
                else:
                    raise ValueError(f"Unknown field or table: {base_field}")
        else:
            # Simple field on main table
            where_clause = f"{self.table_name}.{field}"

        # Build condition based on operator
        condition_sql, params = self._build_operator_condition(where_clause, cond)
        return condition_sql, params, joins_needed

    def _build_join_clause(self, cond: JoinCond) -> tuple[str, dict, list]:
        """Build SQL for joined conditions (AND/OR/NOT)"""
        if not cond.conds:
            return "", {}, []

        all_where_parts = []
        all_params = {}
        all_joins = []

        for subcond in cond.conds:
            if subcond is not None:
                where_part, params, joins = self._build_where_clause(subcond)
                if where_part:
                    all_where_parts.append(where_part)
                    all_params.update(params)
                    all_joins.extend(joins)

        if not all_where_parts:
            return "", {}, []

        # Combine conditions based on join type
        if cond.join == JoinType.AND:
            combined_where = ' AND '.join(f"({part})" for part in all_where_parts)
        elif cond.join == JoinType.OR:
            combined_where = ' OR '.join(f"({part})" for part in all_where_parts)
        elif cond.join == JoinType.NOT:
            if len(all_where_parts) == 1:
                combined_where = f"NOT ({all_where_parts[0]})"
            else:
                # NOT of multiple conditions - apply NOT to the AND of all
                and_clause = ' AND '.join(f"({part})" for part in all_where_parts)
                combined_where = f"NOT ({and_clause})"
        else:
            raise NotImplementedError(f"Join type {cond.join} not implemented")

        # Remove duplicate joins
        unique_joins = list(dict.fromkeys(all_joins))

        return combined_where, all_params, unique_joins

    def _row_to_result(self, row) -> SearchResult:
        """Convert database row to SearchResult"""
        return SearchResult(
            id=row[0],
            score=1.0,
            #metadata=row_dict
        )
