"""An abstraction over collections to make it easy to filter/sort/etc.

"""
#TODO next items queue
#TODO   with img previews
#TODO search texts queue
#TODO rels cleanup as Rel or Item method
#TODO fast scanning/detector of all images?
#TODO   explore grid segmentation - there are various options for the actual segmentation
#TODO   explore grid clip
#TODO   explore multimachine processing - prolly around $10/day for a good compute machine
#TODO   explore external api
#TODO   it seems like the way to go to get 10x speedup is to use mobilenet v3 embeddings -- see chatgpt for code
#TODO remove bad images
#TODO handle reblog keys
#TODO are svm scores higher if close to many nn?
#TODO   use NN aggregation for high svm scores
#TODO put tags in sql
#TODO   get tag list from descs
#TODO   aggregate to user
#TODO link videos to their poster images
#TODO transfer likes between related items
#TODO quality scores
#TODO multiple searches
#TODO   more like this on objects
#TODO   clickable tags
#TODO   custom search text
#TODO compute dwell times
#TODO aggregate like scores per user
#TODO list of recent users - figure out how to display these in ux
#TODO debug desc errors
#TODO propagate likes to source sites if possible
#TODO import tumblr likes
#TODO import google history

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
    Database,
    db_session,
    desc,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import init_sql_db, Item, Rel, Source
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
    def likes_worker(self) -> LikesWorker:
        return self.application.likes_worker

    @property
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Item)) # type: ignore[attr-defined]
            return otypes

class GetHandler(MyBaseHandler):
    @db_session
    def build_query(self, data: dict[str, Any]) -> Query:
        """Builds up the database query to get items matching the given data filters.

        For string fields, the value can be a string (exact match) or a list of strings (any of).
        For numeric fields, the value can be a number (exact match) or a string with an operator
        such as '>=123', '<=456', '>789', '<1011', '!=1213'.
        """
        q = Item.select()
        # Handle ids parameter (num spec)
        if 'ids' in data:
            ids = parse_num_spec(data['ids'])
            q = q.filter(lambda c: c.id in ids)
        # Handle string fields
        string_fields = ['source', 'stype', 'otype', 'url', 'name']
        for field in string_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    q = q.filter(lambda c: getattr(c, field) in value)
                else:
                    q = q.filter(lambda c: getattr(c, field) == value)
        # Handle parent field
        if 'parent' in data:
            parent_id = int(data['parent'])
            q = q.filter(lambda c: c.parent and c.parent.id == parent_id)
        if 'ancestor' in data:
            ancestor_id = int(data['ancestor'])
            def has_ancestor(c):
                p = c.parent
                while p:
                    if p.id == ancestor_id:
                        return True
                    p = p.parent
                return False

            #q = q.filter(has_ancestor) #TODO doesn't work
            # workaround:
            q = q.filter(lambda c: c.parent and (c.parent.id == ancestor_id or (c.parent.parent and c.parent.parent.id == ancestor_id)))
        # Handle numeric fields
        numeric_fields = ['ts', 'added_ts', 'explored_ts', 'seen_ts', 'embed_ts']
        for field in numeric_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str): # Parse operator
                    value = value.replace(' ', '')
                    if value.startswith('>='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) >= threshold)
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
        if 'order' in data: # check for ordering info
            # order value will be a field name, optionally prefixed by - for descending
            order_field = data['order']
            if order_field.startswith('-'):
                q = q.order_by(lambda c: desc(getattr(c, order_field[1:])))
            else:
                q = q.order_by(lambda c: getattr(c, order_field))
        else: # sort by id descending
            q = q.order_by(lambda c: desc(c.id))
        # if there was a limit parameter, set it
        if 'limit' in data:
            q = q.limit(int(data['limit']))
        return q

    def post(self):
        data = json.loads(self.request.body)
        print(f'GetHandler got data={data}')
        # Build query conditions
        with db_session:
            q = self.build_query(data)
            items = q[:]
            if data.get('assemble_posts', False):
                rows = {r['id']: r for r in Source.assemble_posts(items)}
            else:
                rows = {r.id: recursive_to_dict(r) for r in q}
            cur_ids = list(rows.keys())
            # Add local_path for images with positive embed_ts and parent_url for items with parents
            for item in items:
                if item.otype == 'image' and item.embed_ts and item.embed_ts > 0:
                    # Find the appropriate source to get images_dir
                    source = Source._registry.get(item.source)
                    if source:
                        local_path = Item.image_path(item, images_dir=source.images_dir)
                        rows[item.id]['local_path'] = os.path.relpath(local_path)
                # Add parent_url if item has a parent
                if item.parent:
                    rows[item.id]['parent_url'] = item.parent.url
            # fetch all rels with source = me and tgt in ids and update the appropriate rows
            me = Item.get_me()
            rels = Rel.select(lambda r: r.src == me and r.tgt.id in cur_ids)
            for rel in rels:
                tgt_id = rel.tgt.id
                if 'rels' not in rows[tgt_id]:
                    rows[tgt_id]['rels'] = {}
                rel_md = rel.md or {}
                rel_md['ts'] = rel.ts
                rows[tgt_id]['rels'][rel.rtype] = rel_md
            # count the number of un-embedded images
            n_unembedded = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None) .count()
        msg = f'Got {len(rows)} items, {n_unembedded} un-embedded images'
        self.write(dict(msg=msg, rows=rows, allOtypes=self.all_otypes))

class SourceHandler(MyBaseHandler):
    def post(self):
        """Set a source url to parse."""
        data = json.loads(self.request.body)
        url = data.pop('url', '')
        print(f'SourceHandler got url={url}, {data}')
        # find a source that can parse this url
        parsed = Source.handle_url(url, **data)
        print(f'parsed to {parsed}')
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
    """The user took some action, which we will store in our `rels` table"""
    def post(self):
        data = json.loads(self.request.body)
        action = data.get('action', '')
        print(f'ActionHandler got action={action}, {data}')
        self.write(dict(action=action, status='ok'))
        # create a new rel
        with db_session:
            me = Item.get_me()
            logger.debug(f'Got me={me}')
            get_kw = dict(src=me, tgt=Item[int(data['id'])], rtype=action)
            match action:
                case 'like': # create or update the rel (only 1 like possible)
                    r = Rel.upsert(get_kw=get_kw, ts=int(time.time()))
                case 'unlike': # delete the rel if it exists
                    get_kw['rtype'] = 'like'
                    r = Rel.get(**get_kw)
                    if r:
                        r.delete()
                case _:
                    print(f'Unknown action {action}')

class ClassifyHandler(MyBaseHandler):
    def _handle_pos(self, pos):
        """Simple positive only classifier"""
        # for now, we use the first pos to set the otype to search over
        with db_session:
            otype = Item[pos[0]].otype
        pos = [f'{p}:{otype}' for p in pos]
        all_keys = [k for k in self.embs if k.endswith(f':{otype}')]
        print(f'ClassifyHandler got pos={pos}, {otype}, {len(all_keys)} total keys: {all_keys[:5]}...')
         # get similar from embs
        ret = self.embs.similar(pos, all_keys=all_keys, method='nn')
        print(f'Got ret {ret}')
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
        """Gets the latest likes scores"""
        scores = self.likes_worker.get_scores()
        #logger.info(f'Got {len(scores)} scores {list(scores.items())[:10]}')
        self.write(dict(
            msg=f'Likes scores for {len(scores)} items',
            #msg=f'Likes image classifier with {len(pos)} pos, {len(neg)} neg, {len(scores)} scores in {t1 - t0:.2f}s (training: {times_dict["training"]:.2f}s, inference: {times_dict["inference"]:.2f}s)',
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
        print(f'In clustering, got manual clusters {data["clusters"]}')
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
        print(f'Got args {args}')

    def on_start(app, args):
        app.sql_db = init_sql_db(args.sqlite_path)
        temp = NumpyLmdb.open(args.lmdb_path, flag='c')
        del temp
        app.embs = Embeddings([args.lmdb_path])

        # Initialize LikesWorker with first source's classifiers_dir
        sources = list(Source._registry.values())
        if sources:
            classifiers_dir = sources[0].classifiers_dir
            app.likes_worker = LikesWorker(
                embs=app.embs,
                classifiers_dir=classifiers_dir,
            )
            app.likes_worker.start()
            # kick it off by putting a likes task in the queue
            app.likes_worker.add_task('update')
            logger.info(f"Started LikesWorker with classifiers_dir: {classifiers_dir}")
        else:
            logger.warning("No sources available, LikesWorker not initialized")

    more_handlers = [
        (r'/get', GetHandler),
        (r'/source', SourceHandler),
        (r'/action', ActionHandler),
        (r'/dwell', DwellHandler),
        (r'/classify', ClassifyHandler),
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    web_main()
