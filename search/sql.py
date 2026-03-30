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

from collections import Counter
from enum import Enum
from multiprocessing import get_context
from typing import Any, Sequence

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

class SearchError(Exception):
    """Base exception for search-related errors"""
    pass

class ValidationError(SearchError):
    """Raised when search query validation fails"""
    pass

class DatabaseError(SearchError):
    """Raised when database operations fail"""
    pass

class FieldType(Enum):
    """Types of field references in search queries"""
    MAGIC = "magic"
    NUMBERED = "numbered"
    ALIASED = "aliased"
    NESTED = "nested"
    JSON = "json"
    SIMPLE = "simple"

class ParameterManager:
    """Manages SQL parameter generation and storage"""
    def __init__(self):
        self.counter = 0
        self.params: dict[str, Any] = {}

    def add_param(self, value: Any, prefix: str = "p") -> str:
        """Add a parameter and return its placeholder name"""
        self.counter += 1
        param_name = f"{prefix}{self.counter}"
        self.params[param_name] = value
        return param_name

    def add_json_path(self, path: str) -> str:
        """Add a JSON path parameter"""
        return self.add_param(path, "path")

    def reset(self):
        """Reset counter and clear parameters"""
        self.counter = 0
        self.params.clear()

class JoinBuilder:
    """Builds and manages SQL JOINs"""
    def __init__(self, main_table: str, id_field: str):
        self.main_table = main_table
        self.id_field = id_field
        self.joins: list[str] = []

    def add_table_join(self, table: str, alias: str, fk_field: str) -> str:
        """Add a table join and return the alias"""
        join = f"JOIN {table} AS {alias} ON {alias}.{fk_field} = {self.main_table}.{self.id_field}"
        if join not in self.joins:
            self.joins.append(join)
        return alias

    def add_nested_join(self, rel_alias: str, target_alias: str, rel_field: str) -> str:
        """Add a nested join through a relationship and return the target alias"""
        join = f"JOIN {self.main_table} AS {target_alias} ON {target_alias}.{self.id_field} = {rel_alias}.{rel_field}"
        if join not in self.joins:
            self.joins.append(join)
        return target_alias

    def get_joins(self) -> list[str]:
        """Get all unique joins"""
        return list(dict.fromkeys(self.joins))

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

        # Initialize parameter manager
        self.param_manager = ParameterManager()

        # Initialize field handler mapping
        self._field_handlers = {
            FieldType.MAGIC: self._handle_magic_field,
            FieldType.NUMBERED: self._handle_numbered_reference,
            FieldType.ALIASED: self._handle_aliased_reference,
            FieldType.NESTED: self._handle_nested_reference,
            FieldType.JSON: self._handle_json_reference,
            FieldType.SIMPLE: self._handle_simple_field,
        }

    def _generate_aliases(self, other_tables: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """Generate aliases automatically based on table and fk_field"""

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

    def _validate_field_reference(self, field: str) -> None:
        """Validate field reference for common issues"""
        if not field or not field.strip():
            raise ValidationError("Empty field name")

        if len(field) > 200:
            raise ValidationError(f"Field name too long: {field[:50]}...")

        parts = field.split('.')
        if len(parts) > 10:
            raise ValidationError(f"Field reference too deep (max 10 levels): {field}")

        # Check for invalid characters in field names
        for part in parts:
            if not part or not part.strip():
                raise ValidationError(f"Empty field part in: {field}")

    def _classify_field(self, field: str) -> FieldType:
        """Classify the type of field reference"""
        self._validate_field_reference(field)

        if field.startswith('$'):
            return FieldType.MAGIC
        elif '.' in field:
            parts = field.split('.')
            base_field = parts[0]
            if len(parts) >= 3 and parts[1].isdigit():
                return FieldType.NUMBERED
            elif base_field in self.table_aliases.values():
                remaining_parts = parts[1:]
                if len(remaining_parts) >= 2 and remaining_parts[0] in ['src', 'tgt']:
                    return FieldType.NESTED
                else:
                    return FieldType.ALIASED
            elif base_field in self.table_json_fields.get(self.table_name, set()):
                return FieldType.JSON
            else:
                return FieldType.SIMPLE  # Legacy table reference
        else:
            return FieldType.SIMPLE

    def _validate_table_name(self, table: str) -> None:
        """Validate table name for SQL injection and basic format"""
        if not table or not table.strip():
            raise ValidationError("Empty table name")

        if len(table) > 64:
            raise ValidationError(f"Table name too long: {table}")

        # Basic SQL identifier validation
        if not table.replace('_', '').replace('-', '').isalnum():
            raise ValidationError(f"Invalid table name format: {table}")

    @db_session
    def _discover_json_fields(self, table: str) -> set[str]:
        """Find columns in `table` that contain JSON data.

        This is only tested on sqlite for now.
        """
        self._validate_table_name(table)

        try:
            # Use parameterized query to prevent SQL injection
            result = self.db.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in result if row[2].upper() == 'JSON'}
        except Exception as e:
            logger.warning(f"Could not discover JSON fields in {table}: {e}")
            raise DatabaseError(f"Failed to discover schema for table {table}: {e}") from e

    def _validate_search_params(self, n_results: int, **kw) -> None:
        """Validate search parameters"""
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValidationError(f"n_results must be a positive integer, got: {n_results}")

        if n_results > 50000:
            raise ValidationError(f"n_results too large (max 50000), got: {n_results}")

    def _safe_execute_sql(self, sql: str, params: dict) -> list:
        """Execute SQL with proper error handling and logging"""
        try:
            logger.debug(f"Executing SQL: {sql[:200]}{'...' if len(sql) > 200 else ''}")
            logger.debug(f"With params: {params}")
            cursor = self.db.execute(sql, {}, params)
            return list(cursor)
        except Exception as e:
            logger.error(f"SQL execution failed: {sql[:100]}...")
            logger.error(f"Parameters: {params}")
            raise DatabaseError(f"Database query failed: {e}") from e

    async def _async_search(self, cond: SearchCond, n_results: int = 15, **kw) -> list[SearchResult]:
        """Execute search using SQL queries. This is the main entry point."""
        self._validate_search_params(n_results, **kw)

        try:
            with db_session:
                # Reset magic fields and parameter manager for each search
                self._reset_magic_fields()
                self.param_manager.reset()
                where_clause, params, joins = self._build_where_clause(cond)

                # Use magic field values or defaults
                limit = self._query_limit if self._query_limit is not None else n_results
                offset = self._query_offset if self._query_offset is not None else 0
                order = self._query_order

                # Validate final parameters
                self._validate_final_params(limit, offset)

                # Build ORDER BY clause
                order_clause = self._build_order_clause(order, params)

                # Build final SQL
                sql = self._build_final_sql(joins, where_clause, order_clause, limit, offset)

                params.update(limit=limit, offset=offset)
                rows = self._safe_execute_sql(sql, params)
                results = [self._row_to_result(row) for row in rows]
                return results
        except (ValidationError, DatabaseError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise SearchError(f"Search operation failed: {e}") from e

    def _build_where_clause(self, cond: SearchCond) -> tuple[str, dict, list]:
        """Build WHERE clause, parameters dict, and JOINs from SearchCond"""
        if isinstance(cond, OpCond):
            where_part, params, joins = self._build_op_clause(cond)
            return where_part, params, joins
        elif isinstance(cond, JoinCond):
            return self._build_join_clause(cond)
        else:
            return "", {}, []

    def _build_order_clause(self, order: str|None, param_dict: dict) -> str:
        """Build ORDER BY clause with JSON field support"""
        if not order:
            return ""

        direction = "DESC" if order.startswith('-') else "ASC"
        field = order.lstrip('-')

        if '.' in field:
            return self._build_json_order_clause(field, direction, param_dict)
        else:
            return f"ORDER BY {self.table_name}.{field} {direction}"

    def _build_json_order_clause(self, field: str, direction: str, param_dict: dict) -> str:
        """Build ORDER BY clause for JSON fields"""
        base_field, *path_parts = field.split('.')
        if base_field in self.table_json_fields.get(self.table_name, set()):
            json_path = '$.' + '.'.join(path_parts)
            path_param = self.param_manager.add_json_path(json_path)
            param_dict[path_param] = json_path
            return f"ORDER BY json_extract({self.table_name}.{base_field}, ${path_param}) {direction}"
        else:
            return f"ORDER BY {self.table_name}.{field} {direction}"

    def _validate_final_params(self, limit: int, offset: int) -> None:
        """Validate final query parameters"""
        if not isinstance(limit, int) or limit <= 0:
            raise ValidationError(f"Invalid limit: {limit}")

        if limit > 50000:
            raise ValidationError(f"Limit too large (max 50000): {limit}")

        if not isinstance(offset, int) or offset < 0:
            raise ValidationError(f"Invalid offset: {offset}")

        if offset > 1000000:
            raise ValidationError(f"Offset too large (max 1000000): {offset}")

    def _build_final_sql(self, joins: list[str], where_clause: str,
                        order_clause: str, limit: int, offset: int) -> str:
        """Assemble the final SQL query"""
        # Validate inputs
        if joins and not all(isinstance(j, str) and j.strip() for j in joins):
            raise ValidationError("Invalid JOIN clauses")

        if where_clause and not isinstance(where_clause, str):
            raise ValidationError("Invalid WHERE clause")

        join_clause = ' '.join(joins) if joins else ''
        where_part = f"WHERE {where_clause}" if where_clause else ""

        return f"""
        SELECT DISTINCT {self.table_name}.{self.id_field}
        FROM {self.table_name}
        {join_clause}
        {where_part}
        {order_clause}
        LIMIT $limit OFFSET $offset
        """.strip()

    def _validate_operator_condition(self, where_clause: str, cond: OpCond) -> None:
        """Validate operator condition inputs"""
        if not where_clause or not where_clause.strip():
            raise ValidationError("Empty WHERE clause")

        if not isinstance(cond.op, Op):
            raise ValidationError(f"Invalid operator: {cond.op}")

        # Validate value based on operator
        if cond.op in {Op.IN, Op.NOT_IN}:
            if not isinstance(cond.value, (list, tuple)):
                raise ValidationError(f"Operator {cond.op} requires list/tuple value, got: {type(cond.value)}")
            if len(cond.value) > 1000:
                raise ValidationError(f"Too many values in IN clause (max 1000): {len(cond.value)}")

        if cond.op in {Op.EXISTS, Op.NOT_EXISTS, Op.IS_NULL, Op.IS_NOT_NULL}:
            # These operators don't use the value
            pass
        elif cond.value is None and cond.op not in {Op.EXISTS, Op.NOT_EXISTS, Op.IS_NULL, Op.IS_NOT_NULL}:
            raise ValidationError(f"Operator {cond.op} requires a value")

    def _build_operator_condition(self, where_clause: str, cond: OpCond) -> tuple[str, dict]:
        """Build the SQL condition and parameters for any operator"""
        self._validate_operator_condition(where_clause, cond)

        match cond.op:
            case Op.EQ:
                param = self.param_manager.add_param(cond.value)
                return f"{where_clause} = ${param}", {param: cond.value}
            case Op.NEQ:
                param = self.param_manager.add_param(cond.value)
                return f"{where_clause} != ${param}", {param: cond.value}
            case Op.GT:
                param = self.param_manager.add_param(cond.value)
                return f"CAST({where_clause} AS REAL) > ${param}", {param: cond.value}
            case Op.GTE:
                param = self.param_manager.add_param(cond.value)
                return f"CAST({where_clause} AS REAL) >= ${param}", {param: cond.value}
            case Op.LT:
                param = self.param_manager.add_param(cond.value)
                return f"CAST({where_clause} AS REAL) < ${param}", {param: cond.value}
            case Op.LTE:
                param = self.param_manager.add_param(cond.value)
                return f"CAST({where_clause} AS REAL) <= ${param}", {param: cond.value}
            case Op.LIKE:
                param = self.param_manager.add_param(f"%{cond.value}%")
                return f"{where_clause} LIKE ${param}", {param: f"%{cond.value}%"}
            case Op.NOT_LIKE:
                param = self.param_manager.add_param(f"%{cond.value}%")
                return f"{where_clause} NOT LIKE ${param}", {param: f"%{cond.value}%"}
            case Op.IN:
                params = {}
                placeholders = []
                for val in cond.value:
                    param = self.param_manager.add_param(val)
                    params[param] = val
                    placeholders.append(f"${param}")
                return f"{where_clause} IN ({','.join(placeholders)})", params
            case Op.NOT_IN:
                params = {}
                placeholders = []
                for val in cond.value:
                    param = self.param_manager.add_param(val)
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
        path_param = self.param_manager.add_json_path(json_path)
        where_clause = f"json_extract({table_alias}.{json_field}, ${path_param})"
        params = {path_param: json_path}

        condition_sql, condition_params = self._build_operator_condition(where_clause, cond)
        params.update(condition_params)
        return condition_sql, params

    def _build_op_clause(self, cond: OpCond) -> tuple[str, dict, list]:
        """Build SQL for a single operation condition"""
        field_type = self._classify_field(cond.field)
        handler = self._field_handlers[field_type]
        return handler(cond.field, cond)

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

    def _validate_magic_field(self, field: str, cond: OpCond) -> None:
        """Validate magic field values"""
        match field:
            case '$limit':
                try:
                    limit = int(cond.value)
                    if limit <= 0 or limit > 50000:
                        raise ValidationError(f"Invalid $limit value (must be 1-50000): {limit}")
                except (ValueError, TypeError):
                    raise ValidationError(f"$limit must be an integer, got: {cond.value}")
            case '$offset':
                try:
                    offset = int(cond.value)
                    if offset < 0 or offset > 1000000:
                        raise ValidationError(f"Invalid $offset value (must be 0-1000000): {offset}")
                except (ValueError, TypeError):
                    raise ValidationError(f"$offset must be an integer, got: {cond.value}")
            case '$order':
                if not isinstance(cond.value, str) or not cond.value.strip():
                    raise ValidationError(f"$order must be a non-empty string, got: {cond.value}")
                # Basic validation of order field format
                order_field = str(cond.value).lstrip('-')
                if not order_field or len(order_field) > 100:
                    raise ValidationError(f"Invalid $order field: {cond.value}")

    def _handle_magic_field(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle magic fields like $limit, $offset, $order"""
        self._validate_magic_field(field, cond)

        match field:
            case '$limit':
                self._query_limit = int(cond.value)
            case '$offset':
                self._query_offset = int(cond.value)
            case '$order':
                self._query_order = str(cond.value)
            case _:
                raise ValidationError(f"Unknown magic field: {field}")
        return "", {}, []

    def _validate_numbered_reference(self, field: str, parts: list[str]) -> None:
        """Validate numbered table reference format"""
        if len(parts) < 3:
            raise ValidationError(f"Invalid numbered reference format: {field}")

        table_name = parts[0]
        table_number = parts[1]

        if not table_number.isdigit():
            raise ValidationError(f"Table number must be numeric: {table_number}")

        num = int(table_number)
        if num < 1 or num > 99:
            raise ValidationError(f"Table number out of range (1-99): {num}")

        self._validate_table_name(table_name)

    def _handle_numbered_reference(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle numbered table references like score.1.tag"""
        parts = field.split('.')
        self._validate_numbered_reference(field, parts)

        table_name = parts[0]
        table_number = parts[1]
        remaining_parts = parts[2:]

        # Find the table info
        related_table_info = None
        for table, fk_field in self.other_tables:
            if table == table_name:
                related_table_info = (table, fk_field)
                break

        if not related_table_info:
            raise ValidationError(f"Unknown numbered table reference: {table_name}")
        table, fk_field = related_table_info
        alias = f"{table}_{table_number}"
        join_builder = JoinBuilder(self.table_name, self.id_field)
        join_builder.add_table_join(table, alias, fk_field)
        if len(remaining_parts) == 1:
            # Simple field in related table
            where_clause = f"{alias}.{remaining_parts[0]}"
            condition_sql, params = self._build_operator_condition(where_clause, cond)
            return condition_sql, params, join_builder.get_joins()
        else:
            # JSON field access in related table
            json_field = remaining_parts[0]
            if json_field in self.table_json_fields.get(table, set()):
                condition_sql, params = self._build_json_condition(alias, json_field, remaining_parts[1:], cond)
                return condition_sql, params, join_builder.get_joins()
            else:
                where_clause = f"{alias}.{json_field}"
                condition_sql, params = self._build_operator_condition(where_clause, cond)
                return condition_sql, params, join_builder.get_joins()

    def _handle_aliased_reference(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle aliased table references like rel_src.rtype"""
        parts = field.split('.')
        base_field = parts[0]
        remaining_parts = parts[1:]

        if not remaining_parts:
            raise ValidationError(f"Aliased reference missing field: {field}")

        # Find the table info for this alias
        related_table_info = None
        for (table, fk_field), alias in self.table_aliases.items():
            if alias == base_field:
                related_table_info = (table, fk_field, alias)
                break

        if not related_table_info:
            raise ValidationError(f"Unknown alias reference: {base_field}")
        table, fk_field, alias = related_table_info
        join_builder = JoinBuilder(self.table_name, self.id_field)
        join_builder.add_table_join(table, alias, fk_field)
        if len(remaining_parts) == 1:
            # Simple field in related table
            where_clause = f"{alias}.{remaining_parts[0]}"
            condition_sql, params = self._build_operator_condition(where_clause, cond)
            return condition_sql, params, join_builder.get_joins()
        else:
            # JSON field access in related table
            json_field = remaining_parts[0]
            if json_field in self.table_json_fields.get(table, set()):
                condition_sql, params = self._build_json_condition(alias, json_field, remaining_parts[1:], cond)
                return condition_sql, params, join_builder.get_joins()
            else:
                where_clause = f"{alias}.{json_field}"
                condition_sql, params = self._build_operator_condition(where_clause, cond)
                return condition_sql, params, join_builder.get_joins()

    def _validate_nested_reference(self, field: str, remaining_parts: list[str]) -> None:
        """Validate nested reference format"""
        if len(remaining_parts) < 2:
            raise ValidationError(f"Nested reference needs at least 2 parts: {field}")

        rel_field = remaining_parts[0]
        if rel_field not in {'src', 'tgt'}:
            raise ValidationError(f"Invalid relationship field (must be 'src' or 'tgt'): {rel_field}")

    def _handle_nested_reference(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle nested references like rel_src.tgt.name"""
        parts = field.split('.')
        base_field = parts[0]
        remaining_parts = parts[1:]

        self._validate_nested_reference(field, remaining_parts)

        # Find the table info for this alias
        related_table_info = None
        for (table, fk_field), alias in self.table_aliases.items():
            if alias == base_field:
                related_table_info = (table, fk_field, alias)
                break

        if not related_table_info:
            raise ValidationError(f"Unknown alias reference: {base_field}")

        table, fk_field, alias = related_table_info
        rel_field = remaining_parts[0]  # 'src' or 'tgt'
        target_field = remaining_parts[1]  # 'name', 'otype', etc.
        # Create joins: main_table -> rel_table -> target_item
        target_alias = f"{alias}_target"
        join_builder = JoinBuilder(self.table_name, self.id_field)
        join_builder.add_table_join(table, alias, fk_field)
        join_builder.add_nested_join(alias, target_alias, rel_field)
        # Handle JSON field access in target item
        if len(remaining_parts) > 2:
            json_field = target_field
            if json_field in self.table_json_fields.get(self.table_name, set()):
                condition_sql, params = self._build_json_condition(target_alias, json_field, remaining_parts[2:], cond)
                return condition_sql, params, join_builder.get_joins()
            else:
                where_clause = f"{target_alias}.{json_field}"
        else:
            # Simple field in target item
            where_clause = f"{target_alias}.{target_field}"
        condition_sql, params = self._build_operator_condition(where_clause, cond)
        return condition_sql, params, join_builder.get_joins()

    def _handle_json_reference(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle JSON field references like md.runtime"""
        parts = field.split('.')
        base_field = parts[0]
        path_parts = parts[1:]
        condition_sql, params = self._build_json_condition(self.table_name, base_field, path_parts, cond)
        return condition_sql, params, []

    def _handle_simple_field(self, field: str, cond: OpCond) -> tuple[str, dict, list]:
        """Handle simple field references or legacy table references"""
        if '.' in field: # Legacy table reference
            parts = field.split('.')
            base_field = parts[0]

            if not parts[1:]:
                raise ValidationError(f"Legacy table reference missing field: {field}")

            matching_entries = [(t, fk) for t, fk in self.other_tables if t == base_field]
            if len(matching_entries) > 1:
                # Multiple FKs to same table - require explicit alias
                available_aliases = [alias for (table, fk), alias in self.table_aliases.items() if table == base_field]
                raise ValidationError(f"Ambiguous table reference '{base_field}'. "
                                   f"Use explicit alias: {', '.join(available_aliases)}")
            elif len(matching_entries) == 1:
                related_table, fk_field = matching_entries[0]
                path_parts = parts[1:]

                joins_needed = [f"JOIN {related_table} ON {related_table}.{fk_field} = {self.table_name}.{self.id_field}"]

                if len(path_parts) == 1:
                    where_clause = f"{related_table}.{path_parts[0]}"
                else:
                    json_field = path_parts[0]
                    if json_field in self.table_json_fields.get(related_table, set()):
                        condition_sql, params = self._build_json_condition(related_table, json_field, path_parts[1:], cond)
                        return condition_sql, params, joins_needed
                    else:
                        where_clause = f"{related_table}.{json_field}"
                condition_sql, params = self._build_operator_condition(where_clause, cond)
                return condition_sql, params, joins_needed
            else:
                raise ValidationError(f"Unknown field or table: {base_field}")
        else:
            # Simple field on main table
            if not field.replace('_', '').replace('-', '').isalnum():
                raise ValidationError(f"Invalid field name format: {field}")

            where_clause = f"{self.table_name}.{field}"
            condition_sql, params = self._build_operator_condition(where_clause, cond)
            return condition_sql, params, []

    def _row_to_result(self, row) -> SearchResult:
        """Convert database row to SearchResult"""
        return SearchResult(
            id=row[0],
            score=1.0,
            #metadata=row_dict
        )
