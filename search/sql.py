"""SQL search implementation"""

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
    
    def __init__(self, db: Database, table_name: str, id_field: str = 'id'):
        super().__init__()
        self.db = db
        self.table_name = table_name
        self.id_field = id_field
        
        # Auto-discover schema
        self.json_fields = self._discover_json_fields()
        self.related_tables = self._discover_foreign_keys()
        logger.info(f"Discovered JSON fields: {self.json_fields}")
        logger.info(f"Discovered related tables: {self.related_tables}")
    
    def _discover_json_fields(self) -> set[str]:
        """Find columns that likely contain JSON data"""
        with db_session:
            try:
                # Get table schema from SQLite
                result = self.db.execute(f"PRAGMA table_info({self.table_name})")
                json_fields = set()
                
                for row in result:
                    column_name, data_type = row[1], row[2]
                    # In SQLite, JSON is often stored as TEXT
                    # We'll assume TEXT fields might be JSON (could be refined)
                    if data_type.upper() in ('JSON', 'TEXT'):
                        # Additional heuristic: check if column name suggests JSON
                        if column_name in ('md', 'metadata', 'data', 'json'):
                            json_fields.add(column_name)
                
                return json_fields
            except Exception as e:
                logger.warning(f"Could not discover JSON fields: {e}")
                return set()
    
    def _discover_foreign_keys(self) -> dict[str, str]:
        """Find tables that reference this table via foreign keys"""
        with db_session:
            try:
                # Find tables that have foreign keys pointing to our table
                result = self.db.execute("""
                    SELECT m.name as table_name, p."from" as fk_column, p."to" as pk_column
                    FROM sqlite_master m
                    JOIN pragma_foreign_key_list(m.name) p ON m.type = 'table'
                    WHERE p."table" = ?
                """, [self.table_name])
                
                related = {}
                for row in result:
                    ref_table, fk_col, pk_col = row
                    if pk_col == self.id_field:
                        related[ref_table] = fk_col
                
                return related
            except Exception as e:
                logger.warning(f"Could not discover foreign keys: {e}")
                return {}
    
    async def _async_search(self, cond: SearchCond, n_results: int = 15, **kw) -> list[SearchResult]:
        """Execute search using SQL queries"""
        with db_session:
            where_clause, params, joins = self._build_where_clause(cond)
            
            # Build complete SQL query
            join_clause = ' '.join(joins) if joins else ''
            sql = f"""
            SELECT DISTINCT {self.table_name}.* 
            FROM {self.table_name}
            {join_clause}
            {where_clause}
            ORDER BY {self.table_name}.{self.id_field} DESC
            LIMIT ?
            """
            params.append(n_results)
            
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"With params: {params}")
            
            rows = self.db.execute(sql, params).fetchall()
            
            # Convert rows to SearchResult objects
            results = []
            for row in rows:
                results.append(self._row_to_result(row))
            
            return results
    
    def _build_where_clause(self, cond: SearchCond) -> tuple[str, list, list]:
        """Build WHERE clause, parameters, and JOINs from SearchCond"""
        if isinstance(cond, OpCond):
            return self._build_op_clause(cond)
        elif isinstance(cond, JoinCond):
            return self._build_join_clause(cond)
        else:
            return "", [], []
    
    def _build_op_clause(self, cond: OpCond) -> tuple[str, list, list]:
        """Build SQL for a single operation condition"""
        field = cond.field
        joins_needed = []
        
        # Handle JSON field access (e.g., md.stats.n_images)
        if '.' in field:
            base_field, *path_parts = field.split('.')
            
            if base_field in self.json_fields:
                # Handle JSON field access
                json_path = '$.' + '.'.join(path_parts)
                where_clause = f"json_extract({self.table_name}.{base_field}, ?)"
                params = [json_path]
                
                # Build condition based on operator
                if cond.op == Op.EQ:
                    return f"{where_clause} = ?", params + [cond.value], joins_needed
                elif cond.op == Op.NEQ:
                    return f"{where_clause} != ?", params + [cond.value], joins_needed
                elif cond.op == Op.GT:
                    return f"CAST({where_clause} AS REAL) > ?", params + [cond.value], joins_needed
                elif cond.op == Op.GTE:
                    return f"CAST({where_clause} AS REAL) >= ?", params + [cond.value], joins_needed
                elif cond.op == Op.LT:
                    return f"CAST({where_clause} AS REAL) < ?", params + [cond.value], joins_needed
                elif cond.op == Op.LTE:
                    return f"CAST({where_clause} AS REAL) <= ?", params + [cond.value], joins_needed
                elif cond.op == Op.EXISTS:
                    return f"{where_clause} IS NOT NULL", params, joins_needed
                elif cond.op == Op.NOT_EXISTS:
                    return f"{where_clause} IS NULL", params, joins_needed
                elif cond.op == Op.LIKE:
                    return f"{where_clause} LIKE ?", params + [f"%{cond.value}%"], joins_needed
                elif cond.op == Op.NOT_LIKE:
                    return f"{where_clause} NOT LIKE ?", params + [f"%{cond.value}%"], joins_needed
            
            elif base_field in self.related_tables:
                # Handle related table field access
                fk_field = self.related_tables[base_field]
                joins_needed.append(f"JOIN {base_field} ON {base_field}.{fk_field} = {self.table_name}.{self.id_field}")
                
                if len(path_parts) == 1:
                    where_clause = f"{base_field}.{path_parts[0]}"
                else:
                    # Nested field in related table (might be JSON)
                    where_clause = f"{base_field}.{path_parts[0]}"
                    # Could extend this for JSON in related tables
            else:
                raise ValueError(f"Unknown field or table: {base_field}")
        else:
            # Simple field on main table
            where_clause = f"{self.table_name}.{field}"
        
        # Build condition based on operator
        if cond.op == Op.EQ:
            return f"{where_clause} = ?", [cond.value], joins_needed
        elif cond.op == Op.NEQ:
            return f"{where_clause} != ?", [cond.value], joins_needed
        elif cond.op == Op.GT:
            return f"{where_clause} > ?", [cond.value], joins_needed
        elif cond.op == Op.GTE:
            return f"{where_clause} >= ?", [cond.value], joins_needed
        elif cond.op == Op.LT:
            return f"{where_clause} < ?", [cond.value], joins_needed
        elif cond.op == Op.LTE:
            return f"{where_clause} <= ?", [cond.value], joins_needed
        elif cond.op == Op.LIKE:
            return f"{where_clause} LIKE ?", [f"%{cond.value}%"], joins_needed
        elif cond.op == Op.NOT_LIKE:
            return f"{where_clause} NOT LIKE ?", [f"%{cond.value}%"], joins_needed
        elif cond.op == Op.IN:
            placeholders = ','.join(['?' for _ in cond.value])
            return f"{where_clause} IN ({placeholders})", list(cond.value), joins_needed
        elif cond.op == Op.NOT_IN:
            placeholders = ','.join(['?' for _ in cond.value])
            return f"{where_clause} NOT IN ({placeholders})", list(cond.value), joins_needed
        elif cond.op == Op.EXISTS:
            return f"{where_clause} IS NOT NULL", [], joins_needed
        elif cond.op == Op.NOT_EXISTS:
            return f"{where_clause} IS NULL", [], joins_needed
        elif cond.op == Op.IS_NULL:
            return f"{where_clause} IS NULL", [], joins_needed
        elif cond.op == Op.IS_NOT_NULL:
            return f"{where_clause} IS NOT NULL", [], joins_needed
        else:
            raise NotImplementedError(f"Operator {cond.op} not implemented")
    
    def _build_join_clause(self, cond: JoinCond) -> tuple[str, list, list]:
        """Build SQL for joined conditions (AND/OR/NOT)"""
        if not cond.conds:
            return "", [], []
        
        all_where_parts = []
        all_params = []
        all_joins = []
        
        for subcond in cond.conds:
            if subcond is not None:
                where_part, params, joins = self._build_where_clause(subcond)
                if where_part:
                    all_where_parts.append(where_part)
                    all_params.extend(params)
                    all_joins.extend(joins)
        
        if not all_where_parts:
            return "", [], []
        
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
        
        return f"WHERE {combined_where}", all_params, unique_joins
    
    def _row_to_result(self, row) -> SearchResult:
        """Convert database row to SearchResult"""
        row_dict = dict(row)
        item_id = row_dict.pop(self.id_field)
        
        return SearchResult(
            id=item_id,
            score=1.0,
            metadata=row_dict
        )

