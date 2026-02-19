"""An abstraction over collections to make it easy to filter/sort/etc.

"""
#TODO faster search-by-example using intersection of sql + scores?
#TODO separate out config on sources vs overall
#TODO investigate multiple linear classifiers
#TODO general slowness
#TODO remove bad images
#TODO diversity on likes classifier?
#TODO handle reblog keys
#TODO put tags in sql
#TODO   get tag list from descs
#TODO   aggregate to user
#TODO search texts queue
#TODO transfer likes between related items
#TODO multiple searches
#TODO   more like this on objects
#TODO   clickable tags
#TODO compute dwell times
#TODO debug desc errors
#TODO propagate likes to source sites if possible
#TODO import tumblr likes
#TODO import google history
#TODO similar users
#TODO adding custom clip embeddings
#TODO faster/cached embeddings scaling for get_keys_embeddings()
#TODO backups
#TODO make cacheking allow checking for mtime when loading from file

from __future__ import annotations

import abc
import asyncio
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

import joblib
import termcolor
import tornado.web

from tornado.web import RequestHandler
from pony.orm import (
    composite_index,
    commit,
    Database,
    db_session,
    desc,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
    exists as pony_exists,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore

from nkpylib.ml.client import embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import init_sql_db, Item, Rel, Source, J, timed, IMAGE_SUFFIX, ACTIONS
from nkpylib.nkcollections.workers import CollectionsWorker
from nkpylib.nkpony import recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

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
    @db_session
    def build_query(self, kw: dict[str, Any]) -> Query:
        """Builds up the database query to get items matching the given kw filters.

        - For string fields, the value can be a string (exact match) or a list of strings (any of).
        - For numeric fields, the value can be a number (exact match) or a string with an operator
          such as '>=123', '<=456', '>789', '<1011', '!=1213'.
        - Use 'parent' or 'ancestor' filters to filter by parent or any ancestor item.
          - For now, ancestor only searches parent + grandparent
        - Use 'ids' with a num spec string like '1,2,5-10' to filter by specific item ids.
        - Use 'order' with a field name, optionally prefixed by '-' for descending
          - For JSON fields, use the format field[key], e.g. md[like-benchmark-20260207].
        - For rel-based filters, use rels.{rtype}.{property} format:
          - rels.like=True/False (existence)
          - rels.queue.count>=2 (rel metadata)
          - rels.queue.ts>1234567890 (rel timestamp)
        - Use 'min_like' to filter by a minimum like score from the likes classifier.
        """
        logger.info(f'Building query with filters: {kw}')
        t0 = time.time()
        q = Item.select()

        # Handle rel-based filters first (they may need to modify the query significantly)
        rel_filters = {k: v for k, v in kw.items() if k.startswith('rels.')}
        if rel_filters:
            q = self._apply_rel_filters(q, rel_filters)

        # Handle ids parameter (num spec)
        if 'ids' in kw:
            ids = parse_num_spec(kw['ids'])
            q = q.filter(lambda c: c.id in ids)
        # Handle string fields
        string_fields = ['source', 'stype', 'otype', 'url', 'name']
        for field in string_fields:
            if field in kw:
                value = kw[field]
                if isinstance(value, list):
                    q = q.filter(lambda c: getattr(c, field) in value)
                else:
                    q = q.filter(lambda c: getattr(c, field) == value)
        # Handle parent field
        if 'parent' in kw:
            parent_id = int(kw['parent'])
            q = q.filter(lambda c: c.parent and c.parent.id == parent_id)
        if 'ancestor' in kw:
            ancestor_id = int(kw['ancestor'])
            def has_ancestor(c):
                p = c.parent
                while p:
                    if p.id == ancestor_id:
                        return True
                    p = p.parent
                return False

            #q = q.filter(has_ancestor) #TODO doesn't work
            # partial workaround (one-level up only)
            q = q.filter(lambda c: c.parent and (c.parent.id == ancestor_id or (c.parent.parent and c.parent.parent.id == ancestor_id)))
        if 'mn' in kw:
            q = q.filter(lambda c: c.md['like-benchmark-20260207'])
        # Handle numeric fields
        # TODO how to handle JSON fields (particularly md)?
        numeric_fields = ['ts', 'added_ts', 'explored_ts', 'seen_ts', 'embed_ts']
        for field in numeric_fields:
            if field in kw:
                value = kw[field]
                if isinstance(value, str): # Parse operator
                    value = value.replace(' ', '')
                    if value.startswith('>='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) >= threshold)
                    elif value.startswith('<>'): # doesn't exist
                        q = q.filter(lambda c: getattr(c, field) is None)
                    elif value.startswith('<='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) <= threshold)
                    elif value.startswith('!='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) != threshold)
                    elif value.startswith('>'):
                        threshold = float(value[1:])
                        q = q.filter(lambda c: getattr(c, field) > threshold)
                    elif value.startswith('<'):
                        threshold = float(value[1:])
                        q = q.filter(lambda c: getattr(c, field) < threshold)
                    else: # No operator, treat as exact match
                        q = q.filter(lambda c: getattr(c, field) == float(value))
                else: # Exact match
                    q = q.filter(lambda c: getattr(c, field) == value)
        limit = int(kw.get('limit', 10000000))
        offset = int(kw.get('offset', 0))
        do_ordering = True
        # score/ml-based filters
        if 'min_like' in kw or 'pos' in kw:
            logger.info(f'Applying score/ml-based filters')
            # get only ids
            if isinstance(q, Query):
                q = q.without_distinct()
                # pre-limit the query if we have a limit and min_like filter
                if 'min_like' in kw and 'pos' not in kw and limit:
                    q = q.limit(int((limit+offset)/ 0.1))
            ids_only = [item.id for item in q]
            logger.info(f'  Got {len(ids_only)} candidate ids')
            # check for min_like score
            if 'min_like' in kw:
                min_like = float(kw['min_like'])
                logger.info(f'Checking for min like')
                scores = self.application.get_scores()
                out_ids = [id for id in ids_only if scores.get(id, 0.0) >= min_like]
                logger.info(f'  Filtered from {len(ids_only)} -> {len(out_ids)} items with min_like {min_like}')
            # check for sorting by pos
            if 'pos' in kw:
                pos = kw['pos']
                logger.info(f'Finding similar from {pos} to {len(ids_only)}')
                self.embs.reload_keys()
                sim = find_similar(pos, embs=self.embs, cur_ids=ids_only)
                scores = sim['scores']
                min_score = min(scores.values()) if scores else 0.0
                out_ids = sorted(ids_only, key=lambda id: scores.get(id, min_score-10), reverse=True)
                do_ordering = False
            q = out_ids
        manual_reverse = False
        if do_ordering: # if we haven't already ordered by a score-based method
            # order value will be a field name, optionally prefixed by - for descending
            order_field = kw.get('order', '-id')
            if isinstance(q, Query): # for query objects
                if '[' in order_field: # JSON access, so do it via an eval() call
                    if order_field.startswith('-'):
                        # using desc() does not work for JSON field ordering
                        manual_reverse = True
                        order_field = order_field[1:]
                    q = eval(f'q.order_by({order_field})')
                else:
                    if order_field.startswith('-'):
                        q = q.order_by(lambda c: desc(getattr(c, order_field[1:])))
                    else:
                        q = q.order_by(lambda c: getattr(c, order_field))
            else: # the query is a list of items/objects
                if q and isinstance(q[0], int):
                    q = [Item[id] for id in q]
                def key_func(item):
                    if '[' in order_field:
                        return eval(f'item.{order_field}')
                    else:
                        return getattr(item, order_field.lstrip('-'))

                q.sort(key=key_func, reverse=order_field.startswith('-'))
            t1 = time.time()
            if manual_reverse: # fetch all items, reverse
                q = q[:]
                q = q[::-1]
        # if there was a limit parameter, set it
        if 'limit' in kw:
            if manual_reverse: # we've already fetch items and reversed them, so just limit
                q = q[offset:limit+offset]
            else:
                if isinstance(q, Query):
                    q = q.limit(limit, offset=offset)
                else:
                    q = q[offset:limit+offset]
        t2 = time.time()
        # now convert ids to full items, if needed
        if not isinstance(q, Query) and q and isinstance(q[0], int):
            q = [Item[id] for id in q]
        #logger.info(f'Built query in {t1 - t0:.3f}s, applied limit in {t2 - t1:.3f}s, output type: {type(q)}')
        #print(f'q2: {q}')
        return q

    @db_session
    def _apply_rel_filters(self, q: Query, rel_filters: dict[str, Any]) -> Query:
        """Apply rel-based filters to the query.

        Handles filters like:
        - rels.like=True/False (existence)
        - rels.queue.count>=2 (rel metadata)
        - rels.queue.ts>1234567890 (rel timestamp)
        """
        me = Item.get_me()
        for filter_key, filter_value in rel_filters.items():
            # Parse the filter key: rels.{rtype}[.{property}]
            parts = filter_key.split('.')
            if len(parts) < 2:
                continue
            rtype = parts[1]  # e.g., 'like', 'queue'
            if len(parts) == 2:
                # Simple existence check: rels.like=True/False
                if isinstance(filter_value, bool):
                    if filter_value:
                        # Must have this rel type
                        q = q.filter(lambda c: pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))
                    else:
                        # Must NOT have this rel type
                        q = q.filter(lambda c: not pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))
            elif len(parts) == 3:
                # Property-based filter: rels.queue.count>=2
                property_name = parts[2]  # e.g., 'count', 'ts'

                # For complex rel filters, we need to post-process since we can't easily
                # filter on processed rel data in the SQL query. We'll get candidate items
                # that have the rel type, then filter in Python.
                if property_name in ['count', 'ts']:
                    # First filter to items that have this rel type
                    q = q.filter(lambda c: pony_exists(Rel.select(lambda r: r.src == me and r.tgt == c and r.rtype == rtype)))

                    # Store the property filter for post-processing
                    if not hasattr(q, '_rel_property_filters'):
                        q._rel_property_filters = []
                    q._rel_property_filters.append((rtype, property_name, filter_value))
        return q

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
            # Get all rels for this item
            rels = list(Rel.select(lambda r: r.src == me and r.tgt == item))

            # Create a temporary dict to hold processed rels (like rels_for_web does)
            temp_dict = {'rels': {}}
            item.rels_for_web(temp_dict, rels)
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
        logger.info(f'GetHandler got data={data}')
        # Build query conditions
        with db_session:
            times = [time.time()]
            q = self.build_query(data)
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
    def post(self):
        pass

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


def find_similar(pos: list[str|int], *, embs: Embeddings, cur_ids: list[int]|None) -> dict[str, Any]:
    """Searches for similarity to `pos` amongst `cur_ids` using `embs`"""
    pos = [f'{p}:{IMAGE_SUFFIX}' for p in pos]
    if cur_ids is None:
        all_keys = [k for k in embs if k.endswith(f':{IMAGE_SUFFIX}')]
    else:
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
    logger.info(f'got pos={pos}, {len(all_keys)} all keys: {all_keys[:5]}...')
    ret = embs.similar(pos, all_keys=all_keys, method='nn')
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
        return find_similar(pos, embs=self.embs, cur_ids=cur_ids)

    async def _handle_likes(self,
                            cur_ids: list[int]|None=None,
                            otypes=['image'],
                            **kw):
        """Gets the latest likes scores from cached loader"""
        scores = self.application.get_scores()
        if cur_ids is not None:
            cur_ids = [int(id) for id in cur_ids]
            scores = {id: score for id, score in scores.items() if int(id) in cur_ids}
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
        #TODO get cluster names
        #TODO weight features differently, based on likes classifier?
        cur_ids = [int(id) for id in cur_ids]
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
        ret = self.embs.cluster(all_keys=all_keys, method=method, n_clusters=n_clusters)
        clusters = {}
        for i, lst in enumerate(ret):
            lst = [int(key.split(':')[0]) for key in lst]
            clusters[i] = lst
        msg = f'Clustered {len(cur_ids)} ids into {len(clusters)} clusters (req: {n_clusters}) using method {method} and kw {kw}'
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
            classifiers_dir = sources[0].classifiers_dir
        else:
            assert False, "No sources registered, cannot determine classifiers_dir"
        if args.worker: # version with likes workers
            app.likes_worker = CollectionsWorker(embs=app.embs, classifiers_dir=classifiers_dir)
            app.likes_worker.start()
            app.likes_worker.add_task('update')  # Start the main loop
            logger.info("CollectionsWorker started successfully")
        else: # without likes worker
            app.likes_worker = None
            classifier_path = join(classifiers_dir, 'likes-mn_image.joblib')
            app.cached_score_loader = CachedScoresLoader(classifier_path, {})
            logger.info(f"Will read scores from: {classifier_path}")
        def app_get_scores(app):
            """Returns the latest scores from the `app`"""
            if hasattr(app, 'cached_score_loader'):
                return app.cached_score_loader.get()
            else:
                return app.likes_worker.get_scores()

        app.get_scores = lambda: app_get_scores(app)



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
                    s.cleanup_embeddings(s.lmdb_path)
                commit()
        counts = Counter()
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
        embs = Embeddings([lmdb_path])
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
