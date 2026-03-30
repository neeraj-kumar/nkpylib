"""An abstraction over collections to make it easy to filter/sort/etc.

"""
#TODO rescore runs only on newer items?
#TODO toggle different functions on/off interactively or via config file
#TODO implement other classifiers/etc as subclasses of Worker
#TODO similar users
#TODO   by similarity of their images/embeddings
#TODO   or if we have same metadata/scores as items, then we can apply exactly the same machinery
#TODO cluster users
#TODO compute my preferred tags and preferred image embeddings over time based on likes ts
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
from nkpylib.nkcollections.embeddings import cleanup_embeddings, find_similar
from nkpylib.nkcollections.model import init_sql_db, Item, Rel, Score, Source, J, timed, ACTIONS, LIKES_TTYPE, CFG, IMAGE_SUFFIX, get_like_scores
from nkpylib.nkcollections.query_builder import QueryBuilder, make_search_argparser
from nkpylib.nkcollections.workers import CollectionsWorker, make_worker_argparser
from nkpylib.nkpony import recursive_to_dict
from nkpylib.script_utils import NestedNamespace, YamlConfigManager
from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

def parse_config(config_path: str, input_args=None) -> NestedNamespace:
    """This parses config from `config_path` and returns it.

    It also sets the config as global CFG.
    """
    with YamlConfigManager(config_path) as config_manager:
        search = config_manager.add_parser('search', make_search_argparser())
        web = config_manager.add_parser('web', make_web_argparser())
        db = config_manager.add_parser('db', make_db_argparser())
        worker = config_manager.add_parser('worker', make_worker_argparser())
        frontend = config_manager.add_parser('frontend', make_frontend_argparser())
    cfg = config_manager.parse_all(input_args)
    CFG._update(cfg)
    print(f'Got frontend cfg: {vars(CFG.frontend)}')
    return cfg

def try_parse_key_value(s):
    """Tries to parse a key-value pair from a string of the form "key=value".

    If the string is not in this format, sets key to None and value to `s`."""
    print(f'Trying to parse key-value from "{s}"')
    if '=' in s:
        key, value = s.split('=', 1)
        return key.strip(), value.strip()
    else:
        return None, s.strip()

def make_frontend_argparser() -> ArgumentParser:
    """Makes a parser for the frontend"""
    parser = ArgumentParser(description="NK collections frontend")
    return parser

def make_web_argparser() -> ArgumentParser:
    """Makes the argparser for the web"""
    parser = ArgumentParser(description="NK collections web server")
    parser.add_argument('--port', type=int, default=12555,
                        help='Port for the web server')
    parser.add_argument('--with_worker', action='store_true', default=False,
                        help='Whether to start the worker process')
    return parser

def make_db_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="NK collections database")
    parser.add_argument('--sqlite_path', type=str, help='Path to the SQLite database')
    parser.add_argument('--lmdb_path', action='append', default=[], type=try_parse_key_value,
                        help='Path to an LMDB database (can specify multiple, each as just the path, or "{name}={path}")')
    parser.add_argument('--classifiers_dir', type=str, help='Where classifiers are stored')
    return parser

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

class ConfigHandler(MyBaseHandler):
    def _get(self):
        """Returns the config and dynamic code components"""
        comp_paths = CFG.frontend.component_paths or []
        components = ''
        for path in comp_paths:
            try:
                with open(path, 'r') as f:
                    components += f.read() + '\n'
            except Exception as e:
                logger.warning(f'Failed to load component from {path}: {e}')
        self.write(dict(config=CFG.to_dict(), components=components))

    def get(self):
        return self._get()

    def post(self):
        return self._get()

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


class ClassifyHandler(MyBaseHandler):
    async def _handle_pos(self,
                          cur_ids: list[int]|None=None,
                          **data):
        """Simple positive only classifier"""
        pos = data.get('pos', [])
        classifier_path = join(self.application.classifiers_dir, 'likes-mn_image.joblib')
        return find_similar(pos, embs=self.embs, cur_ids=cur_ids, classifier_path=classifier_path)

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

def web_main(cfg_path: str, **kw):
    parse_config(cfg_path, **kw)
    print(f'Got cfg {CFG}')
    #FIXME add images dir and make it accessible via a static path

    def on_start(app, args):
        app.sql_db = init_sql_db(CFG.db.sqlite_path)
        assert len(CFG.db.lmdb_path) == 1
        lmdb_path = CFG.db.lmdb_path[0][1]
        temp = NumpyLmdb.open(lmdb_path, flag='c')
        del temp
        app.embs = Embeddings([lmdb_path])
        # Initialize scores file path for reading from worker process
        sources = list(Source._registry.values())
        app.classifiers_dir = CFG.db.classifiers_dir
        if CFG.web.with_worker: # version with likes workers
            app.likes_worker = CollectionsWorker(
                embs=app.embs,
                classifiers_dir=app.classifiers_dir,
                **CFG.worker.to_dict(), # Pass worker-specific config parameters
            )
            app.likes_worker.start() # start the main loop
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
        (r'/config', ConfigHandler),
        (r'/cluster', ClusterHandler),
    ]

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/collections.jsx',
                                css_filename=f'collections.css',
                                port=CFG.web.port,
                                more_handlers=more_handlers,
                                parse_args=False,
                                on_start=on_start)

def worker_main(cfg_path: str, **kw) -> None:
    """Standalone process that runs just the CollectionsWorker.

    - sqlite_path: Path to the SQLite database
    - lmdb_path: Path to the LMDB embeddings database
    - classifiers_dir: Directory where classifiers are saved
    """
    parse_config(cfg_path, **kw)
    logger.info(f"Starting worker process with sqlite={CFG.db.sqlite_path}, lmdb={CFG.db.lmdb_path}, image_suffix={CFG.worker.image_suffix}")
    try:
        # Initialize database and embeddings in this process
        sql_db = init_sql_db(CFG.db.sqlite_path)
        lmdb_path = CFG.db.lmdb_path[0] # Assuming one LMDB path for worker #FIXME
        assert len(lmdb_path) > 5, f'LMDB path seems too short: {lmdb_path}'
        embs: Embeddings = Embeddings([lmdb_path])
        # Create and start worker
        likes_worker = CollectionsWorker(
            embs=embs,
            classifiers_dir=CFG.db.classifiers_dir,
            **CFG.worker.to_dict(), # Pass worker-specific config parameters
        )
        likes_worker.run()
        logger.info("CollectionsWorker started successfully")
    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        traceback.print_exc()
        raise

def test_sql_search(db_path='db/nkmovies/embeddings/movie-collection.sqlite'):
    """Tests out the sql searcher"""
    from nkpylib.search.sql import SqlSearchImpl
    db = init_sql_db(db_path)
    print(f"Database: {db}")

    ssi = SqlSearchImpl(db=db, table_name='item', other_tables=[
        ('rel', 'src'), ('rel', 'tgt'), ('score', 'id')])
    print(f"SqlSearchImpl: {ssi}")

    # Test queries using compact JSON syntax, mimicking QueryBuilder functionality
    queries = [
        # Basic field filters (like QueryBuilder's source, otype filters)
        (["source", "=", "imdb"], "Filter by source"),
        (["otype", "=", "movie"], "Filter by object type"),
        (["name", "~", "long"], "Search name containing 'long'"),

        # Numeric filters (like QueryBuilder's ts, added_ts filters)
        (["ts", "<", 1773729138], "Items before Tuesday, March 17, 2026"),
        (["id", ":", [1, 2, 3, 4, 5]], "Items with specific IDs"),

        # JSON field access (like QueryBuilder's md field access)
        (["md.birth_year", ">=", 1999], "Items with birth year >= 1999"),
        (["md.imdb_id", "=", 'tt1849718'], "Specific imdb id"),

        # Related table queries (like QueryBuilder's rel filters)
        (["score.tag", "=", "nkrating"], "Items with nkrating scores"),
        (["score.score", "<", 0.2], "Items with low scores"),
        (["rel.rtype", "=", "has_genre"], "Items that have a genre"),
        #TODO we're currently returning rel.src for this query

        # Complex AND/OR queries (like QueryBuilder's complex filters)
        (["&", ["otype", "=", "genre"], ["source", "=", "imdb"]], "Imdb genre"),
        (["&", ["otype", "=", "person"], ["md.birth_year", ">", 1999]], "Young people"),
        (["|", ["source", "=", "letterboxd"], ["source", "=", "movielens"]], "Letterboxd or movielens items"),

        # NOT queries
        (["!", ["otype", "=", "movie"]], "Non-movie items"),

        # Existence checks
        (["md.imdb_id", "?"], "Items with imdb_id metadata"),
        (["embed_ts", "?+"], "Items with embeddings"),

        # Score-based queries (budget, revenue, ratings stored in Score table)
        (["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "budget"],
          ["score.score", ">", 100000000]
         ], "Big budget movies (>$100M)"),

        (["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "revenue"],
          ["score.score", ">", 500000000]
         ], "High-grossing movies (>$500M)"),

        # Relationship-based queries (genres, cast as separate items with rels)
        (["&",
          ["otype", "=", "movie"],
          ["rel.rtype", "=", "has_genre"]
         ], "Movies with genre relationships"),

        (["&",
          ["otype", "=", "movie"],
          ["rel.rtype", "=", "directed_by"]
         ], "Movies with known directors"),

        (["&",
          ["otype", "=", "movie"],
          ["rel.rtype", "=", "acted_in"]
         ], "Movies with known cast"),

        # Multiple score conditions
        (["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "imdb_rating"],
          ["score.score", ">=", 7.0],
          ["score.tag", "=", "budget"],
          ["score.score", "<", 10000000]
         ], "Good low-budget movies"),

        # Complex OR with nested AND conditions using actual schema
        (["|",
          ["&", ["otype", "=", "person"], ["name", "~", "Nolan"]],
          ["&", ["otype", "=", "person"], ["name", "~", "Tarantino"]],
          ["&", ["otype", "=", "genre"], ["name", "=", "Sci-Fi"]]
         ], "Nolan OR Tarantino OR Sci-Fi genre"),

        # List operations with multiple values
        (["otype", ":", ["movie", "person", "genre"]], "Movies, people, or genres"),
        (["source", ":", ["imdb", "letterboxd", "movielens"]], "Items from specific sources"),
        (["id", "!:", [1, 2, 3, 100, 200]], "Exclude specific IDs"),

        # Numeric comparisons with actual fields
        (["md.runtime", ":", [90, 120, 150]], "Movies with specific runtimes"),
        (["md.runtime", ">=", 120], "Long movies"),

        # Complex existence and null checks
        (["&",
          ["md.imdb_id", "?"],
          ["!", ["embed_ts", "?"]]
         ], "Items with IMDB ID but no embeddings"),

        # Multi-table joins with complex conditions using actual schema
        (["&",
          ["otype", "=", "movie"],
          ["rel.rtype", "=", "acted_in"],
          ["score.tag", "=", "revenue"],
          ["score.score", ">", 1000000000]
         ], "Billion-dollar movies with known cast"),

        # Revenue vs budget analysis
        (["&",
          ["otype", "=", "movie"],
          ["score.tag", "=", "revenue"],
          ["score.score", ">", 100000000],
          ["score.tag", "=", "budget"],
          ["score.score", "<", 50000000]
         ], "Profitable movies (high revenue, low budget)"),

        # Time-based queries
        (["&",
          ["ts", ">", 1640995200], # Jan 1, 2022
          ["embed_ts", "?+"],
          ["seen_ts", "!?"]
         ], "Recent items with embeddings but never seen"),

        # Complex string matching with actual fields
        (["&",
          ["name", "~", "The"],
          ["!", ["name", "~", "The End"]],
          ["otype", "=", "movie"]
         ], "Movies starting with 'The' but not ending movies"),

        # Genre-specific queries using relationships
        (["&",
          ["otype", "=", "genre"],
          ["name", ":", ["Action", "Thriller", "Crime"]]
         ], "Action, thriller, or crime genres"),

        # Person-specific queries
        (["&",
          ["otype", "=", "person"],
          ["name", "~", "Tom"]
         ], "People named Tom"),

        # Cross-entity relationship queries
        (["&",
          ["rel.rtype", "=", "has_genre"],
          ["rel.tgt", "=", 42]  # Assuming genre ID 42 is Horror
         ], "Items with Horror genre relationship"),
    ]

    print(f"\nTesting {len(queries)} queries:")
    for i, (query, description) in enumerate(queries):
        if i < 16:
            continue
        print(f"\n{i+1}. {description}: {query}")
        results = ssi.search(query, n_results=50000)
        print(f"{len(results)} Results found")
        for j, r in enumerate(results[:5]):
            with db_session:
                item = json.dumps(Item[r.id].to_dict(), indent=2)
            print(f"  {j+1}. {r}\n{item}")
    sys.exit()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    test_sql_search()
    parser = ArgumentParser(description="NK collections")
    parser.add_argument('mode', choices=['worker', 'server'], help="Run mode: worker or server")
    parser.add_argument('config_path', help="Config file path")
    parser.add_argument('--port', type=int, default=12555, help="Server port (server mode only)")
    args = parser.parse_args()
    if args.mode == 'worker':
        worker_main(args.config_path)
    elif args.mode == 'server':
        web_main(args.config_path, input_args=[])
