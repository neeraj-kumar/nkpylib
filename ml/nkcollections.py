"""An abstraction over collections to make it easy to filter/sort/etc

"""
#TODO Embeddings status page
#TODO propagate likes to source sites if possible

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import time

from argparse import ArgumentParser
from collections import defaultdict
from functools import cache
from os.path import abspath, exists, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread
from typing import Any

import pony.orm.core
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

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

sql_db = Database()

J = lambda obj: json.dumps(obj, indent=2)


async def maybe_dl(url: str, path: str, fetch_delay: float=0.1) -> bool:
    """Downloads the given url to the given dir if it doesn't already exist there (and is not empty).

    Returns if we actually downloaded the file.
    """
    if exists(path) and os.path.getsize(path) > 0:
        return False
    logger.debug(f'downloading image {url} -> {path}')
    r = await make_request_async(url, headers={'Accept': 'image/*,video/*'}, min_delay=fetch_delay)
    try:
        os.makedirs(dirname(path), exist_ok=True)
    except Exception as e:
        pass
    with open(path, 'wb') as f:
        f.write(r.content)
    return True

class Item(sql_db.Entity, GetMixin):
    """Each individual item, which can include users, posts, images, links, etc."""
    id = PrimaryKey(int, auto=True)
    source = Required(str)
    stype = Required(str)
    otype = Required(str, index=True)
    url = Required(str, index=True)
    composite_index(source, stype, otype, url)
    name = Optional(str, index=True)
    parent = Optional('Item', reverse='children', index=True)
    # time of the actual item
    ts = Required(float, default=lambda: time.time(), index=True)
    # time we added this to our database
    added_ts = Required(float, default=lambda: time.time())
    # time we last explored this item
    explored_ts = Optional(float, index=True)
    # time we last saw this item
    seen_ts = Optional(float, index=True)
    # time we last extracted embeddings for this item
    embed_ts = Optional(float, index=True)
    # all other metadata
    md = Optional(Json)
    # cumulative seconds spent on this item
    dwell_time = Required(float, default=0.0)
    children = Set('Item', reverse='parent')
    rel_srcs = Set('Rel', reverse='src')
    rel_tgts = Set('Rel', reverse='tgt')

    @classmethod
    def get_me(cls):
        """Returns the "me" Item row."""
        with db_session:
            me = cls.get(source='me')
            return me

    @classmethod
    def image_path(cls, row, images_dir: str='') -> str:
        """Returns the image path for a given image `row`"""
        if not row:
            return ''
        url = row.url
        ext = row.md.get('ext', url.split('.')[-1])
        mk = row.md.get('media_key', row.id)
        path = abspath(join(images_dir, f'{mk}.{ext}'))
        return path

    @classmethod
    async def update_text_embeddings(cls, q, limit: int, **kw) -> int:
        """Updates text embeddings for the given query `q`.

        This select 'text' and 'link' otypes and updates their embeddings.
        Returns the number of embeddings updated.
        """
        with db_session:
            rows = q.filter(lambda c: c.otype in ('text', 'link') and not c.embed_ts).limit(limit)
            if not rows:
                return 0
            logger.info(f'Updating embeddings for upto {len(rows)} text rows: {rows[:5]}...')
            inputs = []
            row_by_key = {}
            for r in rows:
                if not r.md:
                    continue
                key = f'{r.id}:text'
                row_by_key[key] = r
                if r.otype == 'text' and 'text' in r.md:
                    inputs.append((key, r.md['text']))
                elif r.otype == 'link' and 'title' in r.md:
                    inputs.append((key, f"{r.md['title']}: {r.url}"))

        def md_func(key, input):
            row = row_by_key[key]
            ts = int(time.time())
            with db_session:
                row.embed_ts = ts
            return dict(embed_ts=ts)

        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        n_text = await loop.run_in_executor(
            None,
            lambda: batch_extract_embeddings(inputs=inputs, embedding_type='text', md_func=md_func, **kw)
        )
        if n_text > 0:
            logger.info(f'  Updated embeddings for {n_text} text rows')
        return n_text

    @classmethod
    async def update_image_embeddings(cls, q, lmdb_path, images_dir: str, limit: int, fetch_delay: float=0.1, **kw) -> int:
        """Updates images embeddings for the given query `q`.

        This select 'image' rows, downloads them if needed, and updates their embeddings.
        Returns the number of embeddings updated.
        """
        with db_session:
            rows = q.filter(lambda c: c.otype == 'image' and not c.embed_ts).limit(limit)
        if not rows:
            return 0
        updater = LmdbUpdater(lmdb_path)
        logger.info(f'Updating embeddings for upto {len(rows)} image rows: {rows[:5]}...')
        # kick off downloads
        async def dl_image(row):
            with db_session:
                path = cls.image_path(row, images_dir=images_dir)
                await maybe_dl(row.url, path, fetch_delay=fetch_delay)
            return row, path

        download_tasks = [dl_image(row) for row in rows]
        # kick off embeddings as downloads complete
        embedding_tasks = []
        async def embed_image_task(row, key, path):
            try:
                emb = await embed_image.single_async(path, model='clip', use_cache=kw.get('use_cache', True))
            except Exception as e:
                logger.warning(f'Error embedding image for row id={row.id}, path={path}: {e}')
                emb = None
            return (row, key, emb)

        done = set()
        for task in asyncio.as_completed(download_tasks):
            row, path = await task
            key = f'{row.id}:image'
            if key in updater or key in done:
                continue
            done.add(key)
            embedding_tasks.append(embed_image_task(row, key, path))
        # kick off postprocessing of embeddings as they complete
        n_images = 0
        for task in asyncio.as_completed(embedding_tasks):
            row, key, emb = await task
            ts = int(time.time())
            with db_session:
                # update the lmdb and sqlite
                logger.debug(f' emb for image {row}, key={key}, {emb[:10] if emb is not None else "failed"}')
                if emb is None: # error
                    updater.add(key, metadata=dict(embed_ts=ts, error='image embedding failed'))
                    row.embed_ts = -1 #TODO in the future decrement this
                else:
                    updater.add(key, embedding=emb, metadata=dict(embed_ts=ts))
                    row.embed_ts = ts
            n_images += 1
        updater.commit()
        if n_images > 0:
            logger.info(f'  Updated embeddings for {n_images} images')
        return n_images

    @classmethod
    async def update_image_descriptions(cls,
                                        q,
                                        lmdb_path: str,
                                        vlm_prompt: str,
                                        sys_prompt: str,
                                        vlm_model: str,
                                        images_dir: str,
                                        limit: int,
                                        **kw) -> int:
        """Updates image descriptions using VLM for images that have been explored.

        Filters to images where explored_ts is not null, generates descriptions via VLM,
        embeds the descriptions, and updates both LMDB and SQLite metadata.

        Returns the number of descriptions updated.
        """
        updater = LmdbUpdater(lmdb_path)
        with db_session:
            rows = q.filter(lambda c: c.otype == 'image' and c.embed_ts is not None and c.embed_ts > 0 and c.explored_ts is None).limit(limit)
            if not rows:
                return 0
            logger.info(f'Updating descriptions for {len(rows)} image rows: {rows[:5]}...')
        # Create VLM tasks for all images
        async def vlm_task(row):
            if sys_prompt:
                messages = [
                    dict(role='system', content=sys_prompt),
                    dict(role='user', content=vlm_prompt)
                ]
            else:
                messages = vlm_prompt
            path = cls.image_path(row, images_dir=images_dir)
            try:
                desc = await call_vlm.single_async((path, messages), model=vlm_model)
            except Exception as e:
                logger.warning(f'Error generating desc for image {row}, path={path}: {e}')
                desc = ''
            return row, desc

        vlm_tasks = [vlm_task(row) for row in rows]
        # Process VLM results as they complete
        n_descs = 0
        for vlm_task in asyncio.as_completed(vlm_tasks):
            row, desc = await vlm_task
            key = f'{row.id}:text'
            with db_session:
                if desc:
                    row.md['desc'] = desc
                    try:
                        text_embedding = embed_text.single(desc, model='qwen_emb')
                        ts = int(time.time())
                        updater.add(key, embedding=text_embedding, metadata=dict(desc=desc, embed_ts=ts))
                        row.explored_ts = ts
                        n_descs += 1
                    except Exception as e:
                        logger.warning(f'Error embedding description {desc} for {row}: {e}')
                        updater.add(key, metadata=dict(desc=desc, embed_ts=time.time(), error='text embedding failed'))
                else: # failed to get description
                    row.explored_ts = -1 #TODO decrement on more errors
        updater.commit()
        if n_descs > 0:
            logger.info(f'Updated descriptions for {n_descs} images')
        return n_descs

    @classmethod
    async def update_embeddings_async(cls,
                          lmdb_path: str,
                          images_dir: str,
                          ids: list[int]|None=None,
                          vlm_prompt: str|None='briefly describe this image',
                          sys_prompt: str|None=None,
                          vlm_model: str='fastvlm',
                          limit: int=-1,
                          fetch_delay: float=0.1,
                          **kw) -> int:
        """Updates the embeddings for all relevant rows in our table.

        The embeddings are stored in a NumpyLmdb at the given `lmdb_path`.
        For images, we first download them to the given `images_dir`.

        If `ids` is given, we only update embeddings for those ids, else all ids that don't exist in
        the lmdb.

        By default we use the given `vlm_prompt` to generate image descriptions for images. If you
        want, you can override this and also optionally override the `sys_prompt`. If `vlm_prompt`
        is empty or None, we don't generate descriptions.

        If you specify a positive `limit`, we only update upto that many embeddings, per otype.

        Any kw are passed to `batch_extract_embeddings`.

        We return the number of embeddings updated.
        """
        if limit <= 0:
            limit = 10000000
        q = cls.select(lambda c: (ids is None or c.id in ids))
        q = q.order_by(desc(Item.id))
        
        # Run text and image embedding updates in parallel
        text_task = asyncio.create_task(
            cls.update_text_embeddings(q=q, db_path=lmdb_path, limit=limit, **kw)
        )
        image_task = asyncio.create_task(
            cls.update_image_embeddings(q=q,
                                       lmdb_path=lmdb_path,
                                       images_dir=images_dir,
                                       fetch_delay=fetch_delay,
                                       limit=limit,
                                       **kw)
        )
        
        # Wait for both to complete
        n_text, n_images = await asyncio.gather(text_task, image_task)
        
        # Image descriptions run after text/image embeddings complete
        n_descs = await cls.update_image_descriptions(q=q,
                                                      lmdb_path=lmdb_path,
                                                      vlm_prompt=vlm_prompt,
                                                      sys_prompt=sys_prompt,
                                                      vlm_model=vlm_model,
                                                      images_dir=images_dir,
                                                      limit=limit,
                                                      **kw)
        return n_text + n_images + n_descs

    @classmethod
    def update_embeddings(cls,
                          lmdb_path: str,
                          images_dir: str,
                          ids: list[int]|None=None,
                          vlm_prompt: str|None='briefly describe this image',
                          sys_prompt: str|None=None,
                          vlm_model: str='fastvlm',
                          limit: int=-1,
                          fetch_delay: float=0.1,
                          **kw) -> int:
        """Calls the async version"""
        return run_async(cls.update_embeddings_async(
            lmdb_path=lmdb_path,
            images_dir=images_dir,
            ids=ids,
            vlm_prompt=vlm_prompt,
            sys_prompt=sys_prompt,
            vlm_model=vlm_model,
            limit=limit,
            fetch_delay=fetch_delay,
            **kw
        ))


class Rel(sql_db.Entity, GetMixin):
    """Relations between items"""
    src = Required('Item', reverse='rel_srcs')
    tgt = Required('Item', reverse='rel_tgts')
    rtype = Required(str)
    ts = Required(int)
    PrimaryKey(src, tgt, rtype, ts)
    md = Optional(Json)


def init_sql_db(path: str) -> Database:
    """Initializes the sqlite database at the given `path`"""
    for func in sqlite_pragmas:
        sql_db.on_connect(provider='sqlite')(func)
    # create parent dirs
    path = abspath(path)
    try:
        os.makedirs(dirname(path), exist_ok=True)
    except Exception as e:
        pass
    try:
        sql_db.bind('sqlite', path, create_db=True)
        #set_sql_debug(True)
        sql_db.generate_mapping(create_tables=True)
        # add an initial row for 'me'
        with db_session:
            Item.upsert(get_kw=dict(
                source='me',
                stype='user',
                otype='user',
                url='me'
            ))
    except pony.orm.core.BindingError:
        pass
    return sql_db


class Source:
    """Base class for all sources. Subclass this.

    Implement can_parse() and parse() methods if you want to handle custom inputs.
    """
    _registry = {}  # Class variable to maintain map from names to Source classes

    def __init__(self,
                 name: str,
                 data_dir: str,
                 sqlite_path: str='',
                 lmdb_path: str='',
                 images_dir: str='',
                 **kw):
        self.name = name
        self.data_dir = data_dir
        self.sqlite_path = sqlite_path or join(data_dir, 'collection.sqlite')
        self.lmdb_path = lmdb_path or join(data_dir, 'embeddings.lmdb')
        self.images_dir = images_dir or join(data_dir, 'images')
        init_sql_db(self.sqlite_path)
        Source._registry[name] = self

    def __repr__(self) -> str:
        return f'Source<{self.name}>'

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return False

    def parse(self, url: str, **kw) -> Any:
        """Parses the given url and does whatever it wants."""
        raise NotImplementedError()

    @staticmethod
    def handle_url(url: str, **data) -> dict[str, Any]:
        """This is the main entry point to handle a given url.

        This finds the appropriate Source subclass that can parse the given url,
        and calls its parse() method with the given url and data.
        """
        for source_cls in Source.__subclasses__():
            if source_cls.can_parse(url):
                source = source_cls()
                result = source.parse(url, **data)
                return result
        raise NotImplementedError(f'No source found to parse url {url}')

    @classmethod
    def assemble_post(cls, post, children) -> dict:
        """Assembles a post, generically.

        This method can be overridden by subclasses for custom behavior.
        In this version, we take all children and add them to a subkey called "children".
        """
        assembled_post = recursive_to_dict(post)
        assembled_post['children'] = [recursive_to_dict(child) for child in children]
        # Extract media blocks for carousel functionality
        media_blocks = []
        for child in children:
            if child.otype in ['image', 'video']:
                if child.otype == 'image' and child.md and 'poster_for' in child.md:
                    continue
                media_blocks.append(dict(
                    type=child.otype,
                    data=recursive_to_dict(child)
                ))
        assembled_post['media_blocks'] = media_blocks
        return assembled_post

    @classmethod
    @db_session
    def assemble_posts(cls, posts: list[Item]) -> list[dict]:
        """Assemble complete posts with their children content.

        Takes a list of post `Item`s and returns a list of assembled post dictionaries
        with their children content nested appropriately based on source type.
        """
        assembled_posts = []
        #print(f'Got registry: {Source._registry}, {posts}')
        # for each post, get its children and assemble based on the source type
        for post in posts:
            src = Source._registry.get(post.source, Source)
            #print(f'for post source {post.source}, using src={src}, {Source._registry}')
            assembled_posts.append(src.assemble_post(post, post.children.select()))
        return assembled_posts

    @db_session
    def update_embeddings(self, **kw):
        """Updates the embeddings for this Source.

        By default, this just calls Item.update_embeddings, filtered to the ids of items from
        this source. If 'ids' is given in kw, we further filter to those ids.

        We pass all `kw` to Item.update_embeddings.
        """
        source_ids = select(c.id for c in Item if c.source == self.name)[:]
        # if we had a list of input ids, filter to those
        if 'ids' in kw and kw['ids'] is not None:
            ids = [id for id in source_ids if id in kw['ids']]
        else:
            ids = source_ids
        Item.update_embeddings(lmdb_path=self.lmdb_path, images_dir=self.images_dir, ids=ids, **kw)


class MyBaseHandler(BaseHandler):
    @property
    def sql_db(self) -> Database:
        return self.application.sql_db

    @property
    def lmdb(self) -> NumpyLmdb:
        return self.application.lmdb

    @property
    def embs(self) -> Embeddings:
        return self.application.embs

    @property
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Item))
            return otypes

class GetHandler(MyBaseHandler):
    @db_session
    def build_query(self, data: dict[str, Any]) -> pony.orm.core.Query:
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
        self.write(dict(rows=rows, allOtypes=self.all_otypes))

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
            print(f'Got me={me}')
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

    async def _handle_likes(self, data):
        """Likes-based classifier"""
        assert data['otype'] == 'image'
        images = set()
        with db_session:
            # first get pos images from likes
            me = Item.get_me()
            like_rels = Rel.select(lambda r: r.src == me and r.rtype == 'like')[:]
            for r in like_rels:
                try:
                    ot = r.tgt.otype
                except pony.orm.core.UnrepeatableReadError: # tgt was deleted
                    continue
                if r.tgt.otype == 'image':
                    images.add(r.tgt)
                elif r.tgt.otype == 'post':
                    for child in r.tgt.children.select():
                        if child.otype == 'image':
                            images.add(child)
            # filter down to only those with embeddings
            pos = [img for img in images if img.embed_ts]
            pos_ids = [p.id for p in pos]
            # get a bunch of random negative images
            neg = list(Item.select(lambda c: c.otype == 'image' and c.embed_ts > 0 and c.id not in pos_ids))
            neg = random.sample(neg, min(len(neg), len(pos)*5))
        # train and run the classifier
        pos = [f'{r.id}:image' for r in pos]
        neg = [f'{r.id}:image' for r in neg]
        to_cls = [k for k in self.embs if k.endswith(':image')]
        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        t0 = time.time()
        cls, scores = await loop.run_in_executor(
            None,
            lambda: self.embs.train_and_run_classifier(pos=pos, neg=neg, to_cls=to_cls, method='rbf')
        )
        t1 = time.time()
        scores = {k.split(':')[0]: v for k, v in scores.items()}
        self.write(dict(
            msg=f'Likes image classifier with {len(pos)} pos, {len(neg)} neg, {len(scores)} scores in {t1 - t0:.2f}s',
            pos=pos, neg=neg, scores=scores))

    async def post(self):
        self.embs.reload_keys()
        # figure out what kind of classification we're doing
        data = json.loads(self.request.body)
        pos = data.get('pos', [])
        if pos:
            return self._handle_pos(pos)
        cls_type = data.get('type', '')
        if cls_type == 'likes':
            return await self._handle_likes(data)


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

    more_handlers = [
        (r'/get', GetHandler),
        (r'/source', SourceHandler),
        (r'/action', ActionHandler),
        (r'/dwell', DwellHandler),
        (r'/classify', ClassifyHandler),
        (r'/cluster', ClusterHandler),
    ]

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/nkcollections.jsx',
                                port=port,
                                more_handlers=more_handlers,
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw,
                                on_start=on_start)

def embeddings_main(batch_size: int=10, loop_delay: float=10, **kw):
    """Runs embedding updates from the command line in an infinite loop.

    You probably want to call this from your subclass, after having initialized your Source.
    """
    sources = list(Source._registry.values())
    logger.info(f'Initialized embeddings main with {len(sources)} sources: {sources}')
    while 1:
        t0 = time.time()
        for s in sources:
            s.update_embeddings(limit=batch_size, **kw)
        elapsed = time.time() - t0
        diff = loop_delay - elapsed
        time.sleep(max(0, diff))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    web_main()
