"""An abstraction over collections to make it easy to filter/sort/etc.

"""
#TODO similar users
#TODO   by similarity of their images/embeddings
#TODO   or if we have same metadata/scores as items, then we can apply exactly the same machinery
#TODO cluster users
#TODO compute my preferred tags and preferred image embeddings over time based on likes ts
#TODO better feature extraction pipeline
#TODO separate out config on sources vs overall
#TODO investigate multiple linear classifiers
#TODO remove bad images
#TODO diversity on likes classifier?
#TODO handle reblog keys
#TODO transfer likes between related items?
#TODO enrich queued posts separately to not delay get()
#TODO global clustering of images?
#TODO clickable tags
#TODO debug desc errors
#TODO propagate likes to source sites if possible
#TODO import tumblr likes
#TODO import google history
#TODO backups
#TODO make cacheking allow checking for mtime when loading from file

from __future__ import annotations

import abc
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback

from argparse import ArgumentParser
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from multiprocessing import Process
from os.path import abspath, exists, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread
from typing import Any, Callable

import joblib # type: ignore
import termcolor
import tornado.web

from tornado.web import RequestHandler
from pony.orm import (
    composite_index,
    commit,
    Database,
    db_session,
    desc,
    distinct,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
    exists as pony_exists,
    set_sql_debug
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore
from tqdm import tqdm # type: ignore

from nkpylib.ml.client import embed_text, call_llm
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.llm_utils import load_llm_json
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.embeddings import IMAGE_SUFFIX, cleanup_embeddings
from nkpylib.nkcollections.model import init_sql_db, Item, Rel, Score, Source, J, timed, ACTIONS
from nkpylib.nkcollections.workers import CollectionsWorker, get_like_scores
from nkpylib.nkpony import recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)


def query_step(step_name: str, sql_debug: bool = False, profile: bool = False):
    """Decorator for QueryBuilder methods with exception handling and profiling."""
    def decorator(func):
        async def wrapper(self, kw: dict) -> 'QueryBuilder':
            if profile:
                start_time = time.time()
            # Enable SQL debugging if requested
            if sql_debug:
                set_sql_debug(True)
            try:
                result = await func(self, kw)
                if profile:
                    elapsed = time.time() - start_time
                    logger.info(f"QueryBuilder.{step_name}: {elapsed:.3f}s")
                return result
            except Exception as e:
                logger.error(f"Error in QueryBuilder.{step_name}: {e}")
                # Continue with unchanged query on error rather than failing
                return self
            finally: # Restore SQL debug setting
                if sql_debug:
                    set_sql_debug(False)
        return wrapper
    return decorator


@cache
def get_all_tags() -> list[str]:
    """Get all available tags from Score table."""
    logger.info(f'Getting all tags')
    with db_session:
        all_tags = sorted(set(select(s.tag for s in Score if s.ttype.startswith('tag:'))))
    return all_tags


class QueryBuilder:
    """Builds database queries from filter parameters with a fluent interface."""
    def __init__(self, embs: Embeddings):
        self.embs = embs
        self.query: Query = Item.select()
        self.converted_to_list = False
        self.needs_ordering = True
        self.manual_reverse = False
        self.filters_applied: list[str] = []

    @classmethod
    def create(cls, embs: Embeddings) -> 'QueryBuilder':
        """Factory method to create a new QueryBuilder."""
        return cls(embs)

    async def build(self, kw: dict[str, Any]) -> Query|list:
        """Main build method with fluent chaining."""
        logger.info(f'Building query with filters: {kw}')
        builder = self
        builder = await builder.apply_basic_filters(kw)
        builder = await builder.apply_relationship_filters(kw)
        builder = await builder.apply_numeric_filters(kw)
        builder = await builder.apply_rel_filters(kw)
        builder = await builder.apply_new_search_filters(kw)
        builder = await builder.apply_score_filters(kw)
        builder = await builder.apply_ordering(kw)
        builder = await builder.apply_pagination(kw)
        return builder.finalize()

    @query_step("basic_filters", profile=True)
    async def apply_basic_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply string and simple field filters."""
        if self.converted_to_list:
            return self
        # Handle IDs
        if 'ids' in kw:
            ids = self._parse_ids_parameter(kw['ids'])
            self.query = self.query.filter(lambda c: c.id in ids)
            self.filters_applied.append('ids')
        # Handle string fields
        string_fields = ['source', 'stype', 'otype', 'url', 'name']
        for field in string_fields:
            if field in kw:
                value = kw[field]
                if isinstance(value, list):
                    self.query = self.query.filter(lambda c: getattr(c, field) in value)
                else:
                    self.query = self.query.filter(lambda c: getattr(c, field) == value)
                self.filters_applied.append(field)
        return self

    @query_step("relationship_filters", profile=True)
    async def apply_relationship_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply parent/ancestor/same_user filters."""
        if self.converted_to_list:
            return self
        if 'parent' in kw:
            parent_id = int(kw['parent'])
            self.query = self.query.filter(lambda c: c.parent and c.parent.id == parent_id)
            self.filters_applied.append('parent')
        if 'ancestor' in kw:
            ancestor_id = int(kw['ancestor'])
            self.query = self.query.filter(lambda c: c.parent and (c.parent.id == ancestor_id or (c.parent.parent and c.parent.parent.id == ancestor_id)))
            self.filters_applied.append('ancestor')
        if 'same_user' in kw:
            ancestor_id = self._resolve_same_user_ancestor(kw['same_user'])
            if ancestor_id:
                self.query = self.query.filter(lambda c: c.parent and (c.parent.id == ancestor_id or (c.parent.parent and c.parent.parent.id == ancestor_id)))
                self.filters_applied.append('same_user')
        if 'mn' in kw:
            self.query = self.query.filter(lambda c: c.md['like-benchmark-20260207'])
            self.filters_applied.append('mn')
        return self

    @query_step("numeric_filters", profile=True)
    async def apply_numeric_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply numeric field filters with operators."""
        if self.converted_to_list:
            return self
        numeric_fields = ['ts', 'added_ts', 'explored_ts', 'seen_ts', 'embed_ts']
        for field in numeric_fields:
            if field not in kw:
                continue
            value = kw[field]
            if isinstance(value, str):
                operator, threshold = self._parse_numeric_operator(value)
                self.query = self._apply_numeric_operator(self.query, field, operator, threshold)
            else:
                self.query = self.query.filter(lambda c: getattr(c, field) == value)
            self.filters_applied.append(field)
        return self

    @query_step("rel_filters", sql_debug=True, profile=True)
    async def apply_rel_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply relationship-based filters."""
        if self.converted_to_list:
            return self
        rel_filters = {k: v for k, v in kw.items() if k.startswith('rels.')}
        if rel_filters:
            me = Item.get_me()
            for filter_key, filter_value in rel_filters.items():
                parts = filter_key.split('.')
                if len(parts) < 2:
                    continue
                rtype = parts[1]
                if len(parts) == 2:
                    # Simple existence check
                    if isinstance(filter_value, bool):
                        if filter_value:
                            self.query = self.query.filter(lambda c: pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))
                        else:
                            self.query = self.query.filter(lambda c: not pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))
                elif len(parts) == 3:
                    # Property-based filter - need post-processing
                    property_name = parts[2]
                    if property_name in ['count', 'ts']:
                        self.query = self.query.filter(lambda c: pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))
                        if not hasattr(self.query, '_rel_property_filters'):
                            self.query._rel_property_filters = []
                        self.query._rel_property_filters.append((rtype, property_name, filter_value))
                self.filters_applied.append(filter_key)
        return self

    @query_step("search_filters", profile=True)
    async def apply_search_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply search-based filters using LLM tag parsing."""
        logger.warning('WHOAAAA')
        search_query = kw.get('search', '').strip()
        if not search_query:
            return self
        logger.info(f'Applying search filter: {search_query}')
        all_tags = get_all_tags()
        if not all_tags:
            logger.warning('No tags found in Score table for search')
            return self
        # Use LLM to parse search query into relevant tags
        prompt = f"""Given this search query: "{search_query}"

Return a JSON list of tags from the following available tags that best match what the user is searching for. Only return tags that are directly relevant to the search query, and a minimal set. If no tags match, return an empty list.

Available tags: {', '.join(all_tags)}

Return only the JSON list, no other text."""
        response = await call_llm.single_async(prompt, model='fast', use_cache=False)
        try:
            parsed_tags = load_llm_json(response)
        except Exception as e:
            logger.warning(f'Failed to parse LLM response for search query: {e}\nResponse was: {response}')
            parsed_tags = []
        if not parsed_tags:
            logger.info('No relevant tags found for search query')
            return self
        logger.info(f'LLM parsed search into tags: {parsed_tags}')
        # Apply tag filters using SQL joins (OR logic - item must have ANY of the tags with score > min_score)
        min_item_score = float(kw.get('min_item_score', 0.7))
        min_user_score = float(kw.get('min_user_score', 150))
        min_score = min_user_score if kw.get('otype') == 'user' else min_item_score
        ttype = f'tag:{IMAGE_SUFFIX}'
        if not self.converted_to_list:
            # Create OR condition for any of the tags with score > min_score
            self.query = self.query.filter(lambda item:
                pony_exists(select(s for s in Score
                                 if s.id == item and
                                   s.ttype == ttype and
                                   s.tag in parsed_tags and
                                   s.score > min_score)))
            self.filters_applied.append('search')
            logger.info(f'Applied search filter for {len(parsed_tags)} tags via SQL joins (OR logic, score > {min_score})')
        else:
            # If already converted to list, filter the list using OR logic with score threshold
            filtered_ids = set()
            for tag in parsed_tags:
                with db_session:
                    tag_item_ids = {s.id.id for s in Score.select(lambda s: s.ttype == ttype and s.tag == tag and s.score > min_score)}
                filtered_ids |= tag_item_ids

            self.query = [id for id in self.query if id in filtered_ids]
            self.filters_applied.append('search')
            logger.info(f'Applied search filter for {len(parsed_tags)} tags on list (OR logic, score > {min_score}), {len(self.query)} items remain')
        return self

    async def apply_new_search_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply advanced semantic search with AND+OR logic and score aggregation.

        Parses search query into semantic groups where each group represents OR alternatives
        for a concept, then uses AND logic across groups with score aggregation.

        Example: "cute cat" -> [["adorable", "sweet"], ["feline", "kitten"]]
        Scoring: max(adorable_score, sweet_score) + max(feline_score, kitten_score)
        """
        search_query = kw.get('search', '').strip()
        if not search_query:
            return self
        logger.info(f'Applying new semantic search filter: {search_query}')
        # Parse query into semantic groups via LLM
        semantic_groups = await self._parse_semantic_groups(search_query)
        if not semantic_groups:
            logger.info('No semantic groups found for search query')
            return self
        logger.info(f'Parsed into semantic groups: {semantic_groups}')
        # Get top-K scored items using complex scoring
        search_limit = int(kw.get('search_limit', 1000))
        min_score = float(kw.get('search_min_score', 0.8))
        if not self.converted_to_list:
            # Use subquery approach for efficiency
            scored_items = self._get_semantic_scored_items(semantic_groups, search_limit, min_score)

            if scored_items:
                scored_ids = [item_id for item_id, score in scored_items]
                self.query = self.query.filter(lambda item: item.id in scored_ids)

                # Store scores for potential ordering
                self._semantic_scores = {item_id: score for item_id, score in scored_items}

                # Convert to list and apply semantic ordering
                if isinstance(self.query, Query):
                    self.query = self.query.without_distinct()
                items_list = [item.id for item in self.query]

                # Sort by semantic scores
                items_list.sort(key=lambda id: self._semantic_scores.get(id, 0), reverse=True)
                self.query = items_list
                self.converted_to_list = True
                self.needs_ordering = False

                logger.info(f'Applied semantic search: {len(scored_items)} items found')
            else:
                # No matches found, return empty result
                self.query = []
                self.converted_to_list = True
                self.needs_ordering = False
                logger.info('No items matched semantic search criteria')
        else:
            # Already converted to list, filter existing items
            if hasattr(self, '_semantic_scores'):
                # Use existing scores if available
                self.query = [id for id in self.query if id in self._semantic_scores]
                self.query.sort(key=lambda id: self._semantic_scores.get(id, 0), reverse=True)
            else:
                # Compute scores for current list
                scored_items = self._get_semantic_scored_items(semantic_groups, len(self.query), min_score, self.query)
                if scored_items:
                    score_dict = {item_id: score for item_id, score in scored_items}
                    self.query = [id for id in self.query if id in score_dict]
                    self.query.sort(key=lambda id: score_dict.get(id, 0), reverse=True)
                else:
                    self.query = []

            logger.info(f'Applied semantic search to list: {len(self.query)} items remain')

        self.filters_applied.append('new_search')
        return self

    async def _parse_semantic_groups(self, query: str) -> list[list[str]]:
        """Parse search query into semantic groups with OR alternatives."""
        all_tags = get_all_tags()
        if not all_tags:
            return []
        prompt = f"""Parse this search query into semantic concept groups: "{query}"

For each main concept in the query, provide alternative tags that mean the same thing.
Return as JSON array of arrays: [["tag1a", "tag1b"], ["tag2a", "tag2b"]]

Rules:
- Each inner array represents ONE concept with alternative tags (OR logic)
- Only use tags from the available list
- Keep groups minimal and focused
- If no relevant tags exist, return empty array

Available tags: {', '.join(all_tags)}

Return only the JSON array, no other text."""
        try:
            response = await call_llm.single_async(prompt, model='fast', use_cache=False)
            semantic_groups = load_llm_json(response)
            # Validate structure
            if not isinstance(semantic_groups, list):
                logger.warning(f'Invalid semantic groups structure: {semantic_groups}')
                return []
            logger.info(f'Parsed {query} into semantic groups: {semantic_groups}')
            # Filter out empty groups and validate tags
            valid_groups = []
            for group in semantic_groups:
                if isinstance(group, list) and group:
                    valid_tags = [tag for tag in group if tag in all_tags]
                    if valid_tags:
                        valid_groups.append(valid_tags)

            return valid_groups

        except Exception as e:
            logger.warning(f'Failed to parse semantic groups: {e}')
            return []

    def _get_semantic_scored_items(self,
                                   semantic_groups: list[list[str]],
                                   limit: int,
                                   min_score: float,
                                   filter_ids: list[int]|None = None) -> list[tuple[int, float]]:
        """Get top-K items by semantic scoring using database queries.

        Returns list of (item_id, aggregated_score) tuples sorted by score descending.
        """
        if not semantic_groups:
            return []

        ttype = f'tag:{IMAGE_SUFFIX}'

        with db_session:
            # Get all relevant scores
            all_tags = [tag for group in semantic_groups for tag in group]

            if filter_ids:
                # Filter to specific items
                score_query = select((s.id.id, s.tag, s.score) for s in Score
                                   if s.ttype == ttype and
                                      s.tag in all_tags and
                                      s.score > min_score and
                                      s.id.id in filter_ids)
            else:
                score_query = select((s.id.id, s.tag, s.score) for s in Score
                                   if s.ttype == ttype and
                                      s.tag in all_tags and
                                      s.score > min_score)

            # Group scores by item and semantic group
            item_group_scores = defaultdict(lambda: defaultdict(list))

            for item_id, tag, score in score_query:
                # Find which semantic group this tag belongs to
                for group_idx, group_tags in enumerate(semantic_groups):
                    if tag in group_tags:
                        item_group_scores[item_id][group_idx].append(score)
                        break

            # Compute aggregated scores
            item_scores = []
            required_groups = len(semantic_groups)

            for item_id, group_scores in item_group_scores.items():
                # Check if item has scores for ALL semantic groups (AND logic)
                if len(group_scores) == required_groups:
                    # For each group, take the maximum score (OR logic within group)
                    group_maxes = [max(scores) for scores in group_scores.values()]

                    # Aggregate across groups (AND logic)
                    aggregated_score = self._aggregate_group_scores(group_maxes)
                    item_scores.append((item_id, aggregated_score))

            # Sort by score and limit
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:limit]

    def _aggregate_group_scores(self, group_scores: list[float]) -> float:
        """Aggregate scores across semantic groups.

        Different aggregation strategies for AND logic across groups.
        """
        if not group_scores:
            return 0.0

        # Method 1: Sum of max scores (additive)
        return sum(group_scores)

        # Alternative methods (can be made configurable):
        # Method 2: Geometric mean (penalizes weak groups)
        # return (reduce(lambda x, y: x * y, group_scores)) ** (1/len(group_scores))

        # Method 3: Minimum (all groups must be strong)
        # return min(group_scores)

        # Method 4: Weighted average (if we had group weights)
        # weights = [1.0] * len(group_scores)  # Equal weights for now
        # return sum(score * weight for score, weight in zip(group_scores, weights)) / sum(weights)

    @query_step("score_filters", sql_debug=True, profile=True)
    async def apply_score_filters(self, kw: dict) -> 'QueryBuilder':
        """Apply ML-based filters (min_like, pos)."""
        if not any(k in kw for k in ['min_like', 'pos']):
            return self

        # Handle min_like filter with SQL join (more efficient)
        if 'min_like' in kw and not self.converted_to_list:
            min_like = float(kw['min_like'])
            logger.info(f'Applying min_like filter with SQL join: {min_like}')
            from nkpylib.nkcollections.workers import LIKES_TTYPE
            # Join with Score table to filter by like scores
            self.query = self.query.filter(lambda item:
                pony_exists(select(s for s in Score
                                 if s.id == item and
                                    s.ttype == LIKES_TTYPE and
                                    s.tag == 'like' and
                                    s.score >= min_like)))
            self.filters_applied.append('min_like')
            logger.info(f'  Applied min_like filter via SQL join')

        # For pos filter or if we already converted to list, use the original approach
        if 'pos' in kw or (self.converted_to_list and 'min_like' in kw):
            # Convert query to ID list for score-based filtering
            if not self.converted_to_list:
                if isinstance(self.query, Query):
                    self.query = self.query.without_distinct()
                ids_only = [item.id for item in self.query]
                self.query = ids_only
                self.converted_to_list = True
                logger.info(f'  Got {len(ids_only)} candidate ids for score filtering')

            # Handle min_like on converted list (fallback case)
            if 'min_like' in kw and 'min_like' not in self.filters_applied:
                min_like = float(kw['min_like'])
                logger.info(f'Applying min_like filter on list: {min_like}')
                scores = get_like_scores(ids=self.query)
                self.query = [id for id in self.query if scores.get(id, 0.0) >= min_like]
                logger.info(f'  Filtered to {len(self.query)} items with min_like {min_like}')
                self.filters_applied.append('min_like')

            if 'pos' in kw:
                pos_raw = kw['pos']
                # Ensure pos is a flat list of integers/strings
                if isinstance(pos_raw, list):
                    pos = pos_raw
                else:
                    pos = [pos_raw]
                logger.info(f'Applying pos filter: {pos}')
                self.embs.reload_keys()
                if kw.get('otype') == 'image':
                    sim = find_similar(pos, embs=self.embs, cur_ids=self.query, app=None)
                    scores = sim['scores']
                    min_score = min(scores.values()) if scores else 0.0
                    self.query = sorted(self.query, key=lambda id: scores.get(id, min_score-10), reverse=True)
                elif kw.get('otype') == 'user':
                    sim = find_similar(pos, embs=self.embs, cur_ids=None, app=None)
                    logger.info(f'For user: found {len(sim["scores"])} similar items for pos {pos}')
                    user_scores = self._aggregate_user_scores(sim['scores'], kw)
                    self.query = sorted(self.query, key=lambda id: user_scores.get(id, -10000), reverse=True)
                    logger.info(f'  Found {len(user_scores)} users with similarity scores: {Counter(user_scores).most_common(5)}')

                self.needs_ordering = False
                self.filters_applied.append('pos')

        return self

    @query_step("ordering", profile=True)
    async def apply_ordering(self, kw: dict) -> 'QueryBuilder':
        """Apply ordering to the query."""
        if not self.needs_ordering:
            return self

        order_field = kw.get('order', '-id')

        if not self.converted_to_list:
            # Query object ordering
            if '[' in order_field:
                if order_field.startswith('-'):
                    self.manual_reverse = True
                    order_field = order_field[1:]
                self.query = eval(f'self.query.order_by({order_field})')
            else:
                if order_field.startswith('-'):
                    self.query = self.query.order_by(lambda c: desc(getattr(c, order_field[1:])))
                else:
                    self.query = self.query.order_by(lambda c: getattr(c, order_field))
        else:
            # List ordering
            if self.query and isinstance(self.query[0], int):
                items = [Item[id] for id in self.query]
            else:
                items = self.query

            def key_func(item):
                if '[' in order_field:
                    return eval(f'item.{order_field}')
                else:
                    return getattr(item, order_field.lstrip('-'))

            items.sort(key=key_func, reverse=order_field.startswith('-'))
            self.query = items

        return self

    @query_step("pagination", profile=True)
    async def apply_pagination(self, kw: dict) -> 'QueryBuilder':
        """Apply limit and offset."""
        limit = int(kw.get('limit', 10000000))
        offset = int(kw.get('offset', 0))
        if 'limit' in kw:
            if self.manual_reverse: # Fetch all items and reverse
                if isinstance(self.query, Query):
                    self.query = self.query[:]
                    self.query = self.query[::-1]
                self.query = self.query[offset:limit+offset]
            else:
                if isinstance(self.query, Query):
                    self.query = self.query.limit(limit, offset=offset)
                else:
                    self.query = self.query[offset:limit+offset]
        return self

    def finalize(self) -> Query|list:
        """Convert to final result format."""
        if not isinstance(self.query, Query) and self.query and isinstance(self.query[0], int):
            self.query = [Item[id] for id in self.query]
        return self.query

    def _parse_ids_parameter(self, ids_value: Any) -> list[int]:
        """Parse various formats of ids parameter into list of ints."""
        if isinstance(ids_value, (int, str)):
            if isinstance(ids_value, str):
                return parse_num_spec(ids_value)
            else:
                return [ids_value]
        elif isinstance(ids_value, list):
            return [int(id) for id in ids_value]
        else:
            raise ValueError(f"Invalid ids parameter type: {type(ids_value)}")

    def _resolve_same_user_ancestor(self, item_id: int) -> int|None:
        """Resolve same_user filter to actual ancestor ID."""
        with db_session:
            item = Item[int(item_id)]
            user = item.get_closest(otype='user')
            return user.id if user else None

    def _parse_numeric_operator(self, value: str) -> tuple[str, float]:
        """Parse operator strings like '>=123', '<=456'."""
        value = value.replace(' ', '')
        for op in ['>=', '<=', '!=', '>', '<', '<>']:
            if value.startswith(op):
                return op, float(value[len(op):])
        return '==', float(value)

    def _apply_numeric_operator(self, query: Query, field: str, operator: str, threshold: float) -> Query:
        """Apply numeric operator to query."""
        if operator == '>=':
            return query.filter(lambda c: getattr(c, field) >= threshold)
        elif operator == '<=':
            return query.filter(lambda c: getattr(c, field) <= threshold)
        elif operator == '!=':
            return query.filter(lambda c: getattr(c, field) != threshold)
        elif operator == '>':
            return query.filter(lambda c: getattr(c, field) > threshold)
        elif operator == '<':
            return query.filter(lambda c: getattr(c, field) < threshold)
        elif operator == '<>':
            return query.filter(lambda c: getattr(c, field) is None)
        else:  # '=='
            return query.filter(lambda c: getattr(c, field) == threshold)

    def _aggregate_user_scores(self, image_scores: dict[int, float], kw: dict) -> dict[int, float]:
        """Aggregate image scores by user for user similarity search."""
        user_scores = Counter()
        with db_session:
            for image_id, score in image_scores.items():
                image_item = Item[image_id]
                user = image_item.get_closest(otype='user')
                if user:
                    if 'min_images' in kw:
                        if 'stats' not in user.md:
                            continue
                        n_images = user.md['stats'].get('n_images', 0)
                        if n_images < int(kw['min_images']):
                            continue
                    user_scores[user.id] += score
        return dict(user_scores)

    def get_debug_info(self) -> dict:
        """Get information about what filters were applied."""
        return {
            'filters_applied': self.filters_applied,
            'converted_to_list': self.converted_to_list,
            'needs_ordering': self.needs_ordering,
            'query_type': type(self.query).__name__
        }

class CachedFileLoader(abc.ABC):
    """Base class for loading files with mtime-based caching.

    Maintains instance variables for last modification time and last returned object.
    Subclasses must implement the `load()` method to define how to load the file.
    """
    def __init__(self, path: str, default_object: Any=None):
        self.path = path
        self.last_mtime: float = 0
        self.last_object = default_object

    @abc.abstractmethod
    def load(self) -> Any:
        """Load and return the object from the file.

        This method must be implemented by subclasses to define how to load
        the specific file format and return the appropriate object.
        """
        ...

    def get(self) -> Any:
        """Get the object, loading from file if it has been modified.

        Returns the cached object if the file hasn't changed since last load,
        otherwise loads and caches the new object.
        """
        try:
            if not os.path.exists(self.path):
                return self.last_object
            file_mtime = os.path.getmtime(self.path)
            if file_mtime <= self.last_mtime: # File hasn't changed
                return self.last_object
            # File has been modified, load new object
            self.last_object = self.load()
            self.last_mtime = file_mtime
            return self.last_object
        except Exception as e:
            logger.warning(f"Failed to load from {self.path}: {e}")
            return self.last_object


class CachedScoresLoader(CachedFileLoader):
    """Cached loader for scores from joblib classifier files."""
    def load(self) -> dict[str, float]:
        """Load scores from saved classifier using joblib."""
        saved_data = joblib.load(self.path)
        scores = saved_data.get('scores', {})
        # Convert keys to ints and values to floats for consistency
        scores = {int(k): float(v) for k, v in scores.items()}
        logger.debug(f"Loaded {len(scores)} scores from classifier {self.path} (mtime: {self.last_mtime})")
        return scores


class MyBaseHandler(BaseHandler):
    @property
    def sql_db(self) -> Database:
        return self.application.sql_db # type: ignore[attr-defined]

    @property
    def lmdb(self) -> NumpyLmdb:
        return self.application.lmdb # type: ignore[attr-defined]

    @property
    def embs(self) -> Embeddings:
        return self.application.embs # type: ignore[attr-defined]

    @property
    def likes_worker(self) -> CollectionsWorker:
        if hasattr(self.application, 'likes_worker'):
            return self.application.likes_worker # type: ignore[attr-defined]
        else:
            raise NotImplementedError("likes_worker not available in this application")

    @property
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Item)) # type: ignore[attr-defined]
            return otypes


class GetHandler(MyBaseHandler):
    async def build_query(self, kw: dict[str, Any]) -> Query:
        return await QueryBuilder.create(self.embs).build(kw)


    @classmethod
    @timed
    async def query_to_web(cls, q: Query, assemble_posts:bool=True) -> tuple[dict[int, dict], list[int]]:
        """Converts a query to a dict of items suitable for web output.

        Returns a tuple of (row_by_id, cur_ids), where the latter is in order.
        """
        times = [time.time()]
        items = q[:]
        if hasattr(q, '_rel_property_filters'):
            items = cls._apply_rel_property_filters(items, q._rel_property_filters)
        cur_ids = [item.id for item in items]
        times.append(time.time())
        if assemble_posts:
            ret = {r['id']: r for r in Source.assemble_posts(items)}
        else:
            ret = {r.id: recursive_to_dict(r) for r in items}
        times.append(time.time())
        for item in items:
            await item.for_web(ret[item.id])
        times.append(time.time())
        logger.info(f'  query_to_web times: {[(t1 - t0) for t0, t1 in zip(times, times[1:])]}')
        return (ret, cur_ids)

    @classmethod
    @db_session
    def _apply_rel_property_filters(cls, items: list[Item], property_filters: list[tuple[str, str, Any]]) -> list[Item]:
        """Apply rel property filters that couldn't be done in SQL."""
        if not property_filters:
            return items

        me = Item.get_me()
        filtered_items = []

        for item in items:
            # Create a temporary dict to hold processed rels (like rels_for_web does)
            temp_dict: dict[str, Any] = {'rels': {}}
            item.rels_for_web(temp_dict)
            processed_rels = temp_dict['rels']
            # Check if item passes all property filters
            passes_all_filters = True
            for rtype, property_name, filter_value in property_filters:
                if rtype not in processed_rels:
                    passes_all_filters = False
                    break
                rel_data = processed_rels[rtype]
                actual_value = rel_data.get(property_name)
                if actual_value is None:
                    passes_all_filters = False
                    break
                # Apply the filter based on the operator in filter_value
                if isinstance(filter_value, str) and any(op in filter_value for op in ['>=', '<=', '!=', '>', '<']):
                    filter_value = filter_value.replace(' ', '')
                    if filter_value.startswith('>='):
                        threshold = float(filter_value[2:])
                        if not (actual_value >= threshold):
                            passes_all_filters = False
                            break
                    elif filter_value.startswith('<='):
                        threshold = float(filter_value[2:])
                        if not (actual_value <= threshold):
                            passes_all_filters = False
                            break
                    elif filter_value.startswith('!='):
                        threshold = float(filter_value[2:])
                        if not (actual_value != threshold):
                            passes_all_filters = False
                            break
                    elif filter_value.startswith('>'):
                        threshold = float(filter_value[1:])
                        if not (actual_value > threshold):
                            passes_all_filters = False
                            break
                    elif filter_value.startswith('<'):
                        threshold = float(filter_value[1:])
                        if not (actual_value < threshold):
                            passes_all_filters = False
                            break
                    else:
                        if actual_value != float(filter_value):
                            passes_all_filters = False
                            break
                else:
                    # Direct comparison
                    if actual_value != filter_value:
                        passes_all_filters = False
                        break

            if passes_all_filters:
                filtered_items.append(item)

        return filtered_items

    async def post(self):
        data = json.loads(self.request.body)
        # Build query conditions
        with db_session:
            times = [time.time()]
            q = await self.build_query(data)
            times.append(time.time())
            row_by_id, cur_ids = await self.query_to_web(q, assemble_posts=data.get('assemble_posts', True))
            times.append(time.time())
            # count the number of un-embedded images
            n_unembedded = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None) .count()
            times.append(time.time())
            logger.info(f'Times: {[(t1-t0) for t0, t1 in zip(times, times[1:])]}')
        msg = f'Got {len(row_by_id)} items, {n_unembedded} un-embedded images'
        self.write(dict(msg=msg,
                        row_by_id=row_by_id,
                        cur_ids=cur_ids,
                        allOtypes=self.all_otypes))

class SourceHandler(MyBaseHandler):
    async def post(self):
        """Set a source url to parse."""
        data = json.loads(self.request.body)
        url = data.pop('url', '')
        logger.info(f'SourceHandler got url={url}, {data}')
        # find a source that can parse this url
        parsed = await Source.handle_url(url, **data)
        logger.info(f'parsed to {parsed}')
        if 0:
            parsed_params = '&'.join([f'{k}={v}' for k, v in parsed.items()])
            self.redirect(f"/get/0-100000?{parsed_params}")
        else:
            # send the parsed result to the client
            self.write(parsed)

class DwellHandler(MyBaseHandler):
    """Update dwell times of objects"""
    @db_session
    async def post(self):
        """Update dwell times for items.

        Expects POST data with 'increments' field containing a dict mapping
        item IDs to dwell time increments in seconds.
        """
        data = json.loads(self.request.body)
        increments = data.get('increments', {})
        if not increments:
            self.write(dict(msg='No increments provided', updated_count=0))
            return
        logger.info(f'DwellHandler updating dwell times for {len(increments)} items')
        updated_count = 0
        for item_id_str, increment in increments.items():
            try:
                item_id, increment = int(item_id_str), float(increment)
                if increment <= 0:
                    continue
                item = Item.get(id=item_id)
                if item:
                    # Initialize dwell_time if it doesn't exist
                    if not hasattr(item, 'dwell_time') or item.dwell_time is None:
                        item.dwell_time = 0.0
                    # Increment the dwell time
                    item.dwell_time += increment
                    updated_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f'Invalid dwell increment for item {item_id_str}: {increment}, error: {e}')
                continue
        msg = f'Updated dwell times for {updated_count} items'
        logger.info(msg)
        self.write(dict(msg=msg, updated_count=updated_count))

class ActionHandler(MyBaseHandler):
    """The user took some action, which we will store in our `rels` table."""
    async def post(self):
        """Input data should include 'action' and 'ids' (of the target items)."""
        data = json.loads(self.request.body)
        action = data.pop('action', '')
        logger.info(f'ActionHandler got action={action}, {data}')
        assert action in ACTIONS
        ids = [int(i) for i in data.pop('ids')]
        # Get the generic result (source-specific processing happens later)
        await Rel.handle_me_action(ids=ids, action=action, **data)
        with db_session:
            q = Item.select(lambda c: c.id in ids)
            updated_rows, _ = await GetHandler.query_to_web(q)
            self.write(dict(
                action=action,
                msg=f'Took action {action} on {ids}',
                updated_rows=updated_rows,
            ))


class FilterHandler(MyBaseHandler):
    async def post(self):
        data = json.loads(self.request.body)
        q, cur_ids = data.pop('q'), data.pop('cur_ids')
        q = q.strip()
        if not q.strip():
            self.write(dict(msg='No query provided', q=q, scores={}))
            return
        logger.info(f'FilterHandler got q {q}, {len(cur_ids)} cur ids, {data}')
        # embed the query
        if q.startswith('-') or q.startswith('not '):
            is_neg = True
            if q.startswith('-'):
                q = q[1:].strip()
            elif q.startswith('not '):
                q = q[4:].strip()
        else:
            is_neg = False
        q_emb = await embed_text.single_async(q, timeout=5, model='clip')
        self.embs.reload_keys()
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
        results = self.embs.simple_nearest_neighbors(pos=[q_emb], n_neighbors=1000, metric='cosine', all_keys=all_keys)
        # returns list of (score, key)
        scores = {key.split(':')[0]: score**(1.0/5) for score, key in results}
        if is_neg:
            # invert scores
            scores = {id: 1.0 - score for id, score in scores.items()}
        msg = f'FilterHandler got {len(scores)} scores for query "{q}"'
        self.write(dict(msg=msg, q=q, scores=scores))


def find_similar(pos: list[str|int], *, embs: Embeddings, cur_ids: list[int]|None, app=None) -> dict[str, Any]:
    """Searches for similarity to `pos` amongst `cur_ids` using `embs`"""
    
    # Load pipeline from the last saved likes classifier
    pipeline = None
    if app and hasattr(app, 'classifiers_dir'):
        try:
            classifier_path = join(app.classifiers_dir, 'likes-mn_image.joblib')
            saved_data = embs.load_classifier(classifier_path)
            pipeline = saved_data.get('pipeline')
            logger.debug(f"Loaded pipeline from {classifier_path}")
        except Exception as e:
            logger.warning(f"Could not load pipeline from classifier: {e}")
    
    pos = [f'{p}:{IMAGE_SUFFIX}' for p in pos]
    if cur_ids is None:
        all_keys = [k for k in embs if k.endswith(f':{IMAGE_SUFFIX}')]
    else:
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
    logger.info(f'got pos={pos}, {len(all_keys)} all keys: {all_keys[:5]}...')
    ret = embs.similar(pos, all_keys=all_keys, method='nn', pipeline=pipeline)
    scores, curIds = zip(*ret)
    return dict(
        pos=pos,
        scores={int(id.split(':')[0]): score for id, score in zip(curIds, scores)},
        msg=f'Classified {len(scores)} items with pos {pos}',
    )

class ClassifyHandler(MyBaseHandler):
    async def _handle_pos(self,
                          cur_ids: list[int]|None=None,
                          **data):
        """Simple positive only classifier"""
        pos = data.get('pos', [])
        return find_similar(pos, embs=self.embs, cur_ids=cur_ids, app=self.application)

    async def _handle_likes(self,
                            cur_ids: list[int]|None=None,
                            otypes=['image'],
                            **kw):
        """Gets the latest likes scores from Score table"""
        scores = get_like_scores(ids=cur_ids)
        return dict(
            msg=f'Likes scores for {len(scores)} items',
            scores=scores
        )

    async def _handle_clusters(self,
                               cur_ids: list[int],
                               n_clusters: int=5,
                               method: str='kmeans',
                               **kw):
        """Does auto-clustering with `n_clusters`"""
        cur_ids = [int(id) for id in cur_ids]
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
        ret = self.embs.cluster(all_keys=all_keys, method=method, n_clusters=n_clusters)
        clusters = {}
        # For each cluster, analyze tags and create names
        with db_session:
            for i, lst in enumerate(ret):
                ids = [int(key.split(':')[0]) for key in lst]
                # Get top 5 most common tags for cluster name
                tag_counts = Score.get_top_tags(ids=ids)
                logger.debug(f'Cluster {i} has tag counts: {tag_counts.most_common(5)}')
                top_tags = [tag for tag, count in tag_counts.most_common(5)]
                name = f'Cluster {i}'
                if top_tags:
                    name += f" ({', '.join(top_tags)})"
                clusters[name] = ids
        msg = f'Clustered {len(cur_ids)} ids into {len(clusters)} clusters (req: {n_clusters}) using method {method}'
        return dict(msg=msg, clusters=clusters)

    async def post(self):
        #self.embs.reload_keys()
        # figure out what kind of classification we're doing
        data = json.loads(self.request.body)
        logger.info(f'ClassifyHandler got data {data}')
        cls_type = data.get('type', '')
        func_by_name = dict(
            likes=self._handle_likes,
            pos=self._handle_pos,
            clusters=self._handle_clusters,
        )
        self.embs.reload_keys()
        ret = await func_by_name[cls_type](**data)
        if not ret:
            return
        if ret.get('msg'):
            logger.info(ret['msg'])
        self.write(ret)


class ClusterHandler(MyBaseHandler):
    """Cluster objects semi-automatically.

    Call with manually labeled clusters: {id: cluster_num, ...} and ids (list of all ids to
    consider), and returns suggested clusters for all ids with scores, as:
        clusters={id: {num=cluster_num, score=score}, ...}

    """
    def post(self):
        data = json.loads(self.request.body)
        logger.info(f'In clustering, got manual clusters {data["clusters"]}')
        # randomly assign cluster nums and scores for now, making sure that the manually labeled
        # clusters are preserved
        self.embs.reload_keys()
        manual_clusters = data.get('clusters', {})
        labels = {f'{id}:text': num for id, num in manual_clusters.items()}
        keys = {f'{id}:text' for id in data.get('ids', [])}
        method = data.get('method', 'rbf')
        # number of clusters is the max of the manual cluster num, unless the method is random
        n_clusters = max(manual_clusters.values())
        if method == 'random':
            n_clusters = 5
        clusters = self.embs.guided_clustering(labels=labels,
                                               keys=keys,
                                               method=method,
                                               n_clusters=n_clusters)
        clusters = {key.split(':')[0]: v for key, v in clusters.items()}
        ret = dict(msg=f'method: {method}', clusters=clusters)
        self.write(ret)

def web_main(port: int=12555, with_worker: bool=False, sqlite_path:str='', lmdb_path:str='', **kw):
    # load the data file from first arg
    parser = ArgumentParser(description="NK collections main")
    if sqlite_path:
        parser.add_argument('--sqlite_path', default=sqlite_path, help="The path to the sqlite database")
    else:
        parser.add_argument('sqlite_path', help="The path to the sqlite database")
    if lmdb_path:
        parser.add_argument('--lmdb_path', default=lmdb_path, help="The path to the lmdb database")
    else:
        parser.add_argument('lmdb_path', help="The path to the lmdb database")
    parser.add_argument('-w', '--worker', default=with_worker, action='store_true', help="Whether to start the worker process")
    parser.add_argument('ignore', nargs='*', help="Ignore extra args")
    #FIXME add images dir and make it accessible via a static path
    kw = {}
    def post_parse_fn(args):
        logger.info(f'Got args {args}')

    def on_start(app, args):
        app.sql_db = init_sql_db(args.sqlite_path)
        temp = NumpyLmdb.open(args.lmdb_path, flag='c')
        del temp
        app.embs = Embeddings([args.lmdb_path])
        # Initialize scores file path for reading from worker process
        sources = list(Source._registry.values())
        if sources:
            app.classifiers_dir = sources[0].classifiers_dir
        else:
            assert False, "No sources registered, cannot determine classifiers_dir"
        if args.worker: # version with likes workers
            app.likes_worker = CollectionsWorker(embs=app.embs, classifiers_dir=app.classifiers_dir)
            app.likes_worker.start()
            app.likes_worker.add_task('update')  # Start the main loop
            logger.info("CollectionsWorker started successfully")
        else: # without likes worker
            app.likes_worker = None

    more_handlers = [
        (r'/get', GetHandler),
        (r'/source', SourceHandler),
        (r'/action', ActionHandler),
        (r'/dwell', DwellHandler),
        (r'/classify', ClassifyHandler),
        (r'/filter', FilterHandler),
        (r'/cluster', ClusterHandler),
    ]

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/collections.jsx',
                                css_filename=f'collections.css',
                                port=port,
                                more_handlers=more_handlers,
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw,
                                on_start=on_start)

def embeddings_main(batch_size: int=20,
                    loop_delay: float=1,
                    source_timeout_factor: float=0.5,
                    loop_callback: Callable|None=None,
                    cleanup_freq: int=100,
                    **kw):
    """Runs embedding updates from the command line in an infinite loop.

    You probably want to call this from your subclass, after having initialized your Source.

    Params:
    - batch_size: The number of embeddings to process per source per otype per loop iteration
    - loop_delay: The desired max delay between loop iterations, in seconds
    - source_timeout_factor: How long to wait for each source to do one round of updates. This is
      number of seconds * batch_size.
    - loop_callback: An optional callback to call at the end of each loop iteration, given the
      counts of embeddings updated. If this returns a dict, then we replace our kw with those.
    - kw: Any other kw are passed to Source.update_embeddings
    """
    sources = list(Source._registry.values())
    logger.info(f'Initialized embeddings main with {len(sources)} sources: {sources}')
    executor = ThreadPoolExecutor()
    per_timeout = source_timeout_factor * batch_size
    i = 0
    while 1:
        with db_session:
            commit()
            if i % cleanup_freq == 0:
                for s in sources:
                    cleanup_embeddings(s.lmdb_path)
                commit()
        counts: Counter = Counter()
        t0 = time.time()
        futures = {}
        for s in sources:
            future = executor.submit(s.update_embeddings, limit=batch_size, **kw)
            futures[future] = s
        def finish_future(future):
            if not future.done():
                return
            try:
                cur = future.result()
                s = futures[future]
                if sum(cur.values()) > 0:
                    logger.info(f'  Updated embeddings for source {s}, got counts {cur}')
                for k, v in cur.items():
                    counts[k] += v
            except Exception as e:
                logger.warning(f'Error updating embeddings for source {s}: {e}')
                print(traceback.format_exc())

        try:
            # Wait for at most per_timeout seconds for the first future to complete
            completed_future = next(as_completed(futures, timeout=per_timeout))
            finish_future(completed_future)
        except StopIteration:
            logger.warning('No futures completed')
        except TimeoutError:
            logger.warning(f'No source completed within {per_timeout}s')
        except Exception as e:
            logger.warning(f'Error in embeddings main loop: {e}')
            print(traceback.format_exc())
        finally:
            # Finish/cancel all remaining futures
            for future in futures:
                if future.done():
                    finish_future(future)
                else:
                    future.cancel()
        if loop_callback:
            out = loop_callback(counts)
            if isinstance(out, dict):
                kw = out
        elapsed = time.time() - t0
        diff = loop_delay - elapsed
        time.sleep(max(0, diff))


def worker_main(sqlite_path: str, lmdb_path: str, classifiers_dir: str, image_suffix: str='mn_image', **kw) -> None:
    """Standalone process that runs just the CollectionsWorker.

    - sqlite_path: Path to the SQLite database
    - lmdb_path: Path to the LMDB embeddings database
    - classifiers_dir: Directory where classifiers are saved
    """
    logger.info(f"Starting worker process with sqlite={sqlite_path}, lmdb={lmdb_path}, image_suffix={image_suffix}")
    try:
        # Initialize database and embeddings in this process
        sql_db = init_sql_db(sqlite_path)
        embs: Embeddings = Embeddings([lmdb_path])
        # Create and start worker
        likes_worker = CollectionsWorker(
            embs=embs,
            classifiers_dir=classifiers_dir,
            image_suffix=image_suffix,
        )
        likes_worker.add_task('update')  # Start the main loop
        likes_worker.run()
        logger.info("CollectionsWorker started successfully")
    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    parser = ArgumentParser(description="NK collections")
    parser.add_argument('mode', choices=['worker', 'server'], help="Run mode: worker or server")
    parser.add_argument('--sqlite_path', required=True, help="Path to SQLite database")
    parser.add_argument('--lmdb_path', required=True, help="Path to LMDB database")
    parser.add_argument('--classifiers_dir', help="Directory for classifiers (worker mode only)")
    parser.add_argument('--scores_path', help="Path for scores JSON file (worker mode only)")
    parser.add_argument('--port', type=int, default=12555, help="Server port (server mode only)")
    args = parser.parse_args()
    if args.mode == 'worker':
        if not args.classifiers_dir:
            parser.error("--classifiers_dir is required for worker mode")
        worker_main(**vars(args))
    elif args.mode == 'server':
        web_main(port=args.port, sqlite_path=args.sqlite_path, lmdb_path=args.lmdb_path)
