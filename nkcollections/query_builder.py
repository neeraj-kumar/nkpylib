import logging
import time

from collections import Counter, defaultdict
from functools import cache, reduce
from typing import Any

from pony.orm.core import Query # type: ignore
from pony.orm import (
    db_session,
    desc,
    select,
    exists as pony_exists,
    set_sql_debug,
) # type: ignore

from nkpylib.ml.client import call_llm
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.llm_utils import load_llm_json
from nkpylib.nkcollections.model import (
    Item,
    Rel,
    Score,
    LIKES_TTYPE,
    CFG,
)
from nkpylib.stringutils import parse_num_spec

logger = logging.getLogger(__name__)

@cache
def get_all_tags() -> list[str]:
    """Get all available tags from Score table."""
    logger.info(f'Getting all tags')
    with db_session:
        all_tags = sorted(set(select(s.tag for s in Score if s.ttype.startswith('tag:'))))
    return all_tags


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
                #TODO for now we just rewrite the query by hand, since we know what it is
                print(f'Query is : {self.query.get_sql()}')
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
                    sim = find_similar(pos, embs=self.embs, cur_ids=self.query)
                    scores = sim['scores']
                    min_score = min(scores.values()) if scores else 0.0
                    self.query = sorted(self.query, key=lambda id: scores.get(id, min_score-10), reverse=True)
                elif kw.get('otype') == 'user':
                    sim = find_similar(pos, embs=self.embs, cur_ids=None)
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


