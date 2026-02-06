"""An abstraction over collections to make it easy to filter/sort/etc.

"""
#TODO benchmark mobilenet
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

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import init_sql_db, Item, Rel, Source, J, timed
from nkpylib.nkcollections.workers import LikesWorker
from nkpylib.nkpony import recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

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
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Item)) # type: ignore[attr-defined]
            return otypes


class GetHandler(MyBaseHandler):
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

    @db_session
    def build_query(self, kw: dict[str, Any]) -> Query:
        """Builds up the database query to get items matching the given kw filters.

        For string fields, the value can be a string (exact match) or a list of strings (any of).
        For numeric fields, the value can be a number (exact match) or a string with an operator
        such as '>=123', '<=456', '>789', '<1011', '!=1213'.

        For rel-based filters, use rels.{rtype}.{property} format:
        - rels.like=True/False (existence)
        - rels.queue.count>=2 (rel metadata)
        - rels.queue.ts>1234567890 (rel timestamp)
        """
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
        # order value will be a field name, optionally prefixed by - for descending
        order_field = kw.get('order', '-id')
        manual_reverse = False
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
        #print(f'q1: {q.get_sql()}')
        t1 = time.time()
        if manual_reverse: # fetch all items, reverse
            q = q[:]
            q = q[::-1]
        # if there was a limit parameter, set it
        if 'limit' in kw:
            limit = int(kw['limit'])
            offset = int(kw.get('offset', 0))
            if manual_reverse: # we've already fetch items and reversed them, so just limit
                q = q[offset:limit+offset]
            else:
                q = q.limit(limit, offset=offset)
        # now check for min_like score
        if 'min_like' in kw:
            min_like = float(kw['min_like'])
            # Load scores from file if available
            if hasattr(self, 'application') and hasattr(self.application, 'scores_path') and self.application.scores_path:
                scores_dict, new_mtime = maybe_load_scores(
                    self.application.scores_path,
                    self.application.cached_scores,
                    self.application.scores_mtime
                )
                self.application.cached_scores = scores_dict
                self.application.scores_mtime = new_mtime
                scores = {int(k): v for k, v in scores_dict.items()}
            else:
                scores = {}
            newq = [item for item in q if scores.get(item.id, 0.0) >= min_like]
            logger.info(f'  Filtered from {len(q)} -> {len(newq)} items with min_like {min_like}')
            q = newq
        t2 = time.time()
        #logger.info(f'Built query in {t1 - t0:.3f}s, applied limit in {t2 - t1:.3f}s, output type: {type(q)}')
        #print(f'q2: {q}')
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
        # fetch rels
        if 0:
            # fetch all rels with source = me and tgt in ids
            me = Item.get_me()
            rels_by_id = defaultdict(list)
            for r in Rel.select(lambda r: r.src == me and r.tgt.id in cur_ids):
                rels_by_id[r.tgt.id].append(r)
        else:
            # all rels related to this item
            rels_by_id = defaultdict(list)
            for r in Rel.select(lambda r: r.tgt.id in cur_ids or r.src.id in cur_ids):
                if r.src.id in cur_ids:
                    rels_by_id[r.src.id].append(r)
                elif r.tgt.id in cur_ids:
                    rels_by_id[r.tgt.id].append(r)
        times.append(time.time())
        # prepare items for web
        for item in items:
            await item.for_web(ret[item.id], rels=rels_by_id[item.id])
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
        assert action in 'like unlike dislike undislike queue unqueue explore'.split()
        ids = [int(i) for i in data.pop('ids')]
        # Start the generator and get the immediate result
        gen = Rel.handle_me_action(ids=ids, action=action, **data)
        rels_by_item_by_source = await gen.__anext__()
        # Respond to client immediately after database updates
        with db_session:
            q = Item.select(lambda c: c.id in ids)
            updated_rows, _ = await GetHandler.query_to_web(q)
            self.write(dict(
                action=action,
                msg=f'Took action {action} on {ids}',
                updated_rows=updated_rows,
            ))
        # Fire-and-forget the rest of the generator (source processing)
        async def finish_source_processing():
            try:
                await gen.__anext__()  # This will finish the source processing
            except StopAsyncIteration:
                pass  # Generator finished normally

        self.background_task(finish_source_processing())

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
        q_emb = await embed_text.single_async(q, model='clip')
        self.embs.reload_keys()
        all_keys = [f'{id}:image' for id in cur_ids]
        results = self.embs.simple_nearest_neighbors(pos=[q_emb], n_neighbors=1000, metric='cosine', all_keys=all_keys)
        # returns list of (score, key)
        scores = {key.split(':')[0]: score**(1.0/5) for score, key in results}
        if is_neg:
            # invert scores
            scores = {id: 1.0 - score for id, score in scores.items()}
        msg = f'FilterHandler got {len(scores)} scores for query "{q}"'
        self.write(dict(msg=msg, q=q, scores=scores))

class ClassifyHandler(MyBaseHandler):
    def _handle_pos(self, pos):
        """Simple positive only classifier"""
        # for now, we use the first pos to set the otype to search over
        with db_session:
            otype = Item[pos[0]].otype
        pos = [f'{p}:{otype}' for p in pos]
        all_keys = [k for k in self.embs if k.endswith(f':{otype}')]
        logger.info(f'ClassifyHandler got pos={pos}, {otype}, {len(all_keys)} total keys: {all_keys[:5]}...')
         # get similar from embs
        ret = self.embs.similar(pos, all_keys=all_keys, method='nn')
        logger.info(f'Got ret {ret}')
        scores, curIds = zip(*ret)
        curIds = [p.split(':')[0] for p in curIds]
        self.write(dict(pos=pos,
                        scores={id: score for id, score in zip(curIds, scores)},
                        curIds=curIds))

    async def _handle_likes(self,
                            cur_ids: list[int]|None=None,
                            otypes=['image'],
                            feature_types=None,
                            method: str='rbf',
                            neg_factor: float=10,
                            **kw):
        """Gets the latest likes scores from file"""
        # Load scores from file if available
        if hasattr(self.application, 'scores_path') and self.application.scores_path:
            scores, new_mtime = maybe_load_scores(
                self.application.scores_path,
                self.application.cached_scores,
                self.application.scores_mtime
            )
            self.application.cached_scores = scores
            self.application.scores_mtime = new_mtime
        else:
            scores = {}
            
        # Convert string keys to int keys for consistency
        scores = {int(k): v for k, v in scores.items()}
        
        # filter down to cur_ids if given
        if cur_ids is not None:
            cur_ids = [int(id) for id in cur_ids]
            scores = {id: score for id, score in scores.items() if int(id) in cur_ids}
            
        self.write(dict(
            msg=f'Likes scores for {len(scores)} items',
            scores=scores
        ))

    async def post(self):
        #self.embs.reload_keys()
        # figure out what kind of classification we're doing
        data = json.loads(self.request.body)
        pos = data.get('pos', [])
        if pos:
            return self._handle_pos(pos)
        cls_type = data.get('type', '')
        if cls_type == 'likes':
            return await self._handle_likes(**data)


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


def web_main(port: int=12555, sqlite_path:str='', lmdb_path:str='', **kw):
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
            app.scores_path = join(classifiers_dir, 'scores.json')
            app.cached_scores = {}
            app.scores_mtime = 0
            logger.info(f"Will read scores from: {app.scores_path}")
        else:
            logger.warning("No sources available, scores path not set")
            app.scores_path = None
            app.cached_scores = {}
            app.scores_mtime = 0

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

def embeddings_main(batch_size: int=20, loop_delay: float=10, loop_callback: Callable|None=None,
                    **kw):
    """Runs embedding updates from the command line in an infinite loop.

    You probably want to call this from your subclass, after having initialized your Source.

    Params:
    - batch_size: The number of embeddings to process per source per otype per loop iteration
    - loop_delay: The desired max delay between loop iterations, in seconds
    - loop_callback: An optional callback to call at the end of each loop iteration, given the
      counts of embeddings updated. If this returns a dict, then we replace our kw with those.
    - kw: Any other kw are passed to Source.update_embeddings
    """
    sources = list(Source._registry.values())
    logger.info(f'Initialized embeddings main with {len(sources)} sources: {sources}')
    for s in sources:
        s.cleanup_embeddings(s.lmdb_path)
    while 1:
        counts = Counter()
        t0 = time.time()
        try:
            for s in sources:
                cur = s.update_embeddings(limit=batch_size, **kw)
                for k, v in cur.items():
                    counts[k] += v
        except Exception as e:
            logger.warning(f'Error in embeddings main loop: {e}')
            print(traceback.format_exc())
        if loop_callback:
            out = loop_callback(counts)
            if isinstance(out, dict):
                kw = out
        elapsed = time.time() - t0
        diff = loop_delay - elapsed
        time.sleep(max(0, diff))


def save_scores(scores: dict[str, float], path: str) -> None:
    """Save scores dict to a JSON file atomically."""
    import tempfile
    
    # Write to temporary file first, then rename for atomicity
    temp_path = path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(scores, f)
        os.rename(temp_path, path)
        logger.debug(f"Saved {len(scores)} scores to {path}")
    except Exception as e:
        logger.error(f"Failed to save scores to {path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def maybe_load_scores(path: str, current_scores: dict[str, float], last_mtime: float = 0) -> tuple[dict[str, float], float]:
    """Load scores from file if it has been modified since last_mtime.
    
    Returns tuple of (scores_dict, new_mtime).
    If file hasn't changed, returns (current_scores, last_mtime).
    """
    try:
        if not os.path.exists(path):
            return current_scores, last_mtime
            
        file_mtime = os.path.getmtime(path)
        if file_mtime <= last_mtime:
            # File hasn't changed
            return current_scores, last_mtime
            
        # File has been modified, load new scores
        with open(path, 'r') as f:
            new_scores = json.load(f)
            
        # Convert keys to strings and values to floats for consistency
        new_scores = {str(k): float(v) for k, v in new_scores.items()}
        logger.debug(f"Loaded {len(new_scores)} scores from {path} (mtime: {file_mtime})")
        return new_scores, file_mtime
        
    except Exception as e:
        logger.warning(f"Failed to load scores from {path}: {e}")
        return current_scores, last_mtime


def worker_main(sqlite_path: str, lmdb_path: str, classifiers_dir: str, scores_path: str = '') -> None:
    """Standalone process that runs just the LikesWorker.
    
    - sqlite_path: Path to the SQLite database
    - lmdb_path: Path to the LMDB embeddings database  
    - classifiers_dir: Directory where classifiers are saved
    - scores_path: Path where scores JSON file should be saved (defaults to classifiers_dir/scores.json)
    """
    if not scores_path:
        scores_path = join(classifiers_dir, 'scores.json')
        
    logger.info(f"Starting worker process with sqlite={sqlite_path}, lmdb={lmdb_path}, scores={scores_path}")
    
    try:
        # Initialize database and embeddings in this process
        sql_db = init_sql_db(sqlite_path)
        embs = Embeddings([lmdb_path])
        
        # Create custom LikesWorker that saves scores periodically
        class ScoringLikesWorker(LikesWorker):
            def __init__(self, *args, scores_save_path: str, **kwargs):
                super().__init__(*args, **kwargs)
                self.scores_save_path = scores_save_path
                self.last_save_time = 0
                self.save_interval = 30.0  # Save every 30 seconds
                
            def process_task(self, task):
                result = super().process_task(task)
                
                # Save scores periodically
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    try:
                        save_scores(self.scores, self.scores_save_path)
                        self.last_save_time = current_time
                    except Exception as e:
                        logger.error(f"Failed to save scores in worker: {e}")
                        
                return result
        
        # Create and start worker
        likes_worker = ScoringLikesWorker(
            embs=embs,
            classifiers_dir=classifiers_dir,
            scores_save_path=scores_path
        )
        likes_worker.start()
        likes_worker.add_task('update')  # Start the main loop
        
        logger.info("LikesWorker started successfully")
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down worker...")
            likes_worker.stop()
            # Save scores one final time
            try:
                save_scores(likes_worker.scores, scores_path)
                logger.info("Final scores saved")
            except Exception as e:
                logger.error(f"Failed to save final scores: {e}")
                
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
        worker_main(args.sqlite_path, args.lmdb_path, args.classifiers_dir, args.scores_path or '')
    elif args.mode == 'server':
        web_main(port=args.port, sqlite_path=args.sqlite_path, lmdb_path=args.lmdb_path)
