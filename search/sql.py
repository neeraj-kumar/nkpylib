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
        tables = list({table_name} | {t[0] for t in self.other_tables})

        # Auto-discover schema
        self.table_json_fields = {table: self._discover_json_fields(table) for table in tables}
        logger.info(f"Discovered JSON fields: {self.table_json_fields}")

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
            # Reset parameter counter for each search
            self._param_counter = 0
            where_clause, param_dict, joins = self._build_where_clause(cond)
            # Build complete SQL query
            join_clause = ' '.join(joins) if joins else ''
            where_part = f"WHERE {where_clause}" if where_clause else ""
            sql = f"""
            SELECT DISTINCT {self.table_name}.{self.id_field}
            FROM {self.table_name}
            {join_clause}
            {where_part}
            LIMIT $limit
            """
            #FIXME do ordering better: ORDER BY {self.table_name}.{self.id_field} DESC
            param_dict['limit'] = n_results

            logger.info(f"Executing SQL: {sql}")
            logger.info(f"With params: {param_dict}")

            cursor = self.db.execute(sql, {}, param_dict)
            rows = list(cursor)

            # Convert rows to SearchResult objects
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

    def _build_op_clause(self, cond: OpCond) -> tuple[str, dict, list]:
        """Build SQL for a single operation condition"""
        field = cond.field
        joins_needed = []
        param_counter = getattr(self, '_param_counter', 0)
        def next_param_name():
            nonlocal param_counter
            param_counter += 1
            self._param_counter = param_counter
            return f"p{param_counter}"

        # Handle numbered table references (e.g., score.1.tag, score.2.score)
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
                    else:
                        # Check if this is JSON field access in related table
                        json_field = remaining_parts[0]
                        if json_field in self.table_json_fields.get(table, set()):
                            # JSON field in related table
                            json_path = '$.' + '.'.join(remaining_parts[1:])
                            path_param = next_param_name()
                            where_clause = f"json_extract({alias}.{json_field}, ${path_param})"
                            params = {path_param: json_path}
                            
                            # Build JSON condition based on operator
                            if cond.op == Op.EQ:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"{where_clause} = ${value_param}", params, joins_needed
                            elif cond.op == Op.NEQ:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"{where_clause} != ${value_param}", params, joins_needed
                            elif cond.op == Op.GT:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) > ${value_param}", params, joins_needed
                            elif cond.op == Op.GTE:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) >= ${value_param}", params, joins_needed
                            elif cond.op == Op.LT:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) < ${value_param}", params, joins_needed
                            elif cond.op == Op.LTE:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) <= ${value_param}", params, joins_needed
                            elif cond.op == Op.EXISTS:
                                return f"{where_clause} IS NOT NULL", params, joins_needed
                            elif cond.op == Op.NOT_EXISTS:
                                return f"{where_clause} IS NULL", params, joins_needed
                            elif cond.op == Op.LIKE:
                                value_param = next_param_name()
                                params[value_param] = f"%{cond.value}%"
                                return f"{where_clause} LIKE ${value_param}", params, joins_needed
                            elif cond.op == Op.NOT_LIKE:
                                value_param = next_param_name()
                                params[value_param] = f"%{cond.value}%"
                                return f"{where_clause} NOT LIKE ${value_param}", params, joins_needed
                        else:
                            # Regular field in related table
                            where_clause = f"{alias}.{json_field}"
                else:
                    raise ValueError(f"Unknown numbered table reference: {table_name}")
            
            # Handle regular JSON field access (e.g., md.stats.n_images)
            elif base_field in self.table_json_fields.get(self.table_name, set()):
                # Handle JSON field access
                path_parts = parts[1:]
                json_path = '$.' + '.'.join(path_parts)
                path_param = next_param_name()
                where_clause = f"json_extract({self.table_name}.{base_field}, ${path_param})"
                params = {path_param: json_path}

                # Build condition based on operator
                if cond.op == Op.EQ:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"{where_clause} = ${value_param}", params, joins_needed
                elif cond.op == Op.NEQ:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"{where_clause} != ${value_param}", params, joins_needed
                elif cond.op == Op.GT:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"CAST({where_clause} AS REAL) > ${value_param}", params, joins_needed
                elif cond.op == Op.GTE:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"CAST({where_clause} AS REAL) >= ${value_param}", params, joins_needed
                elif cond.op == Op.LT:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"CAST({where_clause} AS REAL) < ${value_param}", params, joins_needed
                elif cond.op == Op.LTE:
                    value_param = next_param_name()
                    params[value_param] = cond.value
                    return f"CAST({where_clause} AS REAL) <= ${value_param}", params, joins_needed
                elif cond.op == Op.EXISTS:
                    return f"{where_clause} IS NOT NULL", params, joins_needed
                elif cond.op == Op.NOT_EXISTS:
                    return f"{where_clause} IS NULL", params, joins_needed
                elif cond.op == Op.LIKE:
                    value_param = next_param_name()
                    params[value_param] = f"%{cond.value}%"
                    return f"{where_clause} LIKE ${value_param}", params, joins_needed
                elif cond.op == Op.NOT_LIKE:
                    value_param = next_param_name()
                    params[value_param] = f"%{cond.value}%"
                    return f"{where_clause} NOT LIKE ${value_param}", params, joins_needed

            else:
                # Check if this is a field in a related table
                related_table = None
                fk_field = None
                for table_name, foreign_key in self.other_tables:
                    if base_field == table_name:
                        related_table = table_name
                        fk_field = foreign_key
                        break

                if related_table:
                    # Handle related table field access
                    joins_needed.append(f"JOIN {related_table} ON {related_table}.{fk_field} = {self.table_name}.{self.id_field}")
                    path_parts = parts[1:]

                    if len(path_parts) == 1:
                        # Simple field in related table
                        where_clause = f"{related_table}.{path_parts[0]}"
                    else:
                        # Check if this is JSON field access in related table
                        json_field = path_parts[0]
                        if json_field in self.table_json_fields.get(related_table, set()):
                            # JSON field in related table
                            json_path = '$.' + '.'.join(path_parts[1:])
                            path_param = next_param_name()
                            where_clause = f"json_extract({related_table}.{json_field}, ${path_param})"
                            params = {path_param: json_path}

                            # Build JSON condition based on operator
                            if cond.op == Op.EQ:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"{where_clause} = ${value_param}", params, joins_needed
                            elif cond.op == Op.NEQ:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"{where_clause} != ${value_param}", params, joins_needed
                            elif cond.op == Op.GT:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) > ${value_param}", params, joins_needed
                            elif cond.op == Op.GTE:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) >= ${value_param}", params, joins_needed
                            elif cond.op == Op.LT:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) < ${value_param}", params, joins_needed
                            elif cond.op == Op.LTE:
                                value_param = next_param_name()
                                params[value_param] = cond.value
                                return f"CAST({where_clause} AS REAL) <= ${value_param}", params, joins_needed
                            elif cond.op == Op.EXISTS:
                                return f"{where_clause} IS NOT NULL", params, joins_needed
                            elif cond.op == Op.NOT_EXISTS:
                                return f"{where_clause} IS NULL", params, joins_needed
                            elif cond.op == Op.LIKE:
                                value_param = next_param_name()
                                params[value_param] = f"%{cond.value}%"
                                return f"{where_clause} LIKE ${value_param}", params, joins_needed
                            elif cond.op == Op.NOT_LIKE:
                                value_param = next_param_name()
                                params[value_param] = f"%{cond.value}%"
                                return f"{where_clause} NOT LIKE ${value_param}", params, joins_needed
                        else:
                            # Regular field in related table
                            where_clause = f"{related_table}.{json_field}"
                else:
                    raise ValueError(f"Unknown field or table: {base_field}")
        else:
            # Simple field on main table
            where_clause = f"{self.table_name}.{field}"

        # Build condition based on operator
        if cond.op == Op.EQ:
            value_param = next_param_name()
            return f"{where_clause} = ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.NEQ:
            value_param = next_param_name()
            return f"{where_clause} != ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.GT:
            value_param = next_param_name()
            return f"{where_clause} > ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.GTE:
            value_param = next_param_name()
            return f"{where_clause} >= ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.LT:
            value_param = next_param_name()
            return f"{where_clause} < ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.LTE:
            value_param = next_param_name()
            return f"{where_clause} <= ${value_param}", {value_param: cond.value}, joins_needed
        elif cond.op == Op.LIKE:
            value_param = next_param_name()
            return f"{where_clause} LIKE ${value_param}", {value_param: f"%{cond.value}%"}, joins_needed
        elif cond.op == Op.NOT_LIKE:
            value_param = next_param_name()
            return f"{where_clause} NOT LIKE ${value_param}", {value_param: f"%{cond.value}%"}, joins_needed
        elif cond.op == Op.IN:
            params = {}
            placeholders = []
            for i, val in enumerate(cond.value):
                param_name = next_param_name()
                params[param_name] = val
                placeholders.append(f"${param_name}")
            return f"{where_clause} IN ({','.join(placeholders)})", params, joins_needed
        elif cond.op == Op.NOT_IN:
            params = {}
            placeholders = []
            for i, val in enumerate(cond.value):
                param_name = next_param_name()
                params[param_name] = val
                placeholders.append(f"${param_name}")
            return f"{where_clause} NOT IN ({','.join(placeholders)})", params, joins_needed
        elif cond.op == Op.EXISTS:
            return f"{where_clause} IS NOT NULL", {}, joins_needed
        elif cond.op == Op.NOT_EXISTS:
            return f"{where_clause} IS NULL", {}, joins_needed
        elif cond.op == Op.IS_NULL:
            return f"{where_clause} IS NULL", {}, joins_needed
        elif cond.op == Op.IS_NOT_NULL:
            return f"{where_clause} IS NOT NULL", {}, joins_needed
        else:
            raise NotImplementedError(f"Operator {cond.op} not implemented")

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
