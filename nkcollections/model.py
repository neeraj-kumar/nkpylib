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
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkpony import init_sqlite_db, GetMixin, recursive_to_dict
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

class Item(sql_db.Entity, GetMixin): # type: ignore[name-defined]
    """Each individual item, which can include users, posts, images, links, etc."""
    id = PrimaryKey(int, auto=True)
    source = Required(str)
    stype = Required(str)
    otype = Required(str, index=True)
    url = Required(str, index=True)
    composite_index(source, stype, otype, url)
    name = Optional(str, index=True)
    parent = Optional('Item', reverse='children', index=True) # type: ignore[var-annotated]
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
    children = Set('Item', reverse='parent') # type: ignore[var-annotated]
    rel_srcs = Set('Rel', reverse='src') # type: ignore[var-annotated]
    rel_tgts = Set('Rel', reverse='tgt') # type: ignore[var-annotated]

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

    def get_closest(self, **kw) -> Item|None:
        """Finds the first item that matches the given `kw`, searching up the parent chain.

        That can include the item itself. If none is found, returns None.
        """
        item: Item|None = self
        while item:
            match = True
            for k, v in kw.items():
                if getattr(item, k) != v:
                    match = False
                    break
            if match:
                return item
            item = item.parent
        return None

    def for_web(self, r: dict[str, Any], rels: list[Rel]) -> None:
        """Cleans up this item for web use.

        The web representation of this object is in `r`, which this modifies. Also pass in the list
        of `rels` where this item is the target (for now we assume the src was 'me'). We assume rels
        is in sorted order.

        Item changes:
        - for images with embeddings, adds 'local_path' relative to cwd
        - if this item has a parent, adds 'parent_url' with the parent's url
        - if this item has an ancestor that's a 'user', adds 'user_name' and 'user_url'
        - if this item is a user, then we add fields 'compact' and 'detailed" with
          strings of what to display

        We also deal with rels:
        - adds a 'rels' sub-dict to `r` with keys being the rtype and values being dicts
        - each rel dict has 'ts' and any metadata from the rel's md
        - special processing:
          - for 'like' rels, we only keep the latest one (highest ts)
          - for 'queue'/'unqueue' rels:
            - anytime we see an 'unqueue', we remove it and all 'queue' rels before it
            - for the remaining queues, we keep only the latest one (highest ts), but add a 'count'
        """
        # add local image path if we have it
        if self.otype == 'image' and self.embed_ts and self.embed_ts > 0:
            # Find the appropriate source to get images_dir
            source = Source._registry.get(self.source)
            if source:
                local_path = self.image_path(self, images_dir=source.images_dir)
                r['local_path'] = os.path.relpath(local_path)
        # Add parent_url if self has a parent
        if self.parent:
            r['parent_url'] = self.parent.url
        # Add user_name and user_url if we have an ancestor user
        ancestor = self.parent
        while ancestor:
            if ancestor.otype == 'user':
                r['user_id'] = ancestor.id
                r['user_name'] = ancestor.name
                r['user_url'] = ancestor.url
                break
            ancestor = ancestor.parent
        # if this is a user, add compact and detailed strings
        if self.otype == 'user':
            compact = f'{self.source}: {self.name or self.url}'
            if self.seen_ts:
                compact += f'<br>Last seen: {time.ctime(self.seen_ts)}'
            else:
                compact += f'<br>Never seen'
            qrbs = self.md.get('n_queued_reblogs', 0)
            if qrbs:
                compact += f'<br>queued reblogs: {qrbs}'
            # add the first image if we have one
            images = Item.select(parent=self, otype='image')[:1]
            if not images:
                # find where grandparent is this user
                images = Item.select(lambda i: i.parent.parent == self and i.otype == 'image')[:1]
            if images:
                compact += f'<br><img src="{images[0].url}" />'
            detailed = compact #TODO
            r['compact'] = compact
            r['detailed'] = detailed
        self.rels_for_web(r, rels)
        # call the source-specific version of this function
        source = Source._registry.get(self.source)
        if source:
            source.item_for_web(self, r, rels)

    def rels_for_web(self, r: dict[str, Any], rels: list[Rel]) -> None:
        """Deal with rels for web representation."""
        R = r['rels'] = {}
        for rel in rels:
            # all rels should map from rtype to a dict, including at least the ts
            md = R[rel.rtype] = dict(ts=rel.ts)
            if rel.md:
                md.update(rel.md)

    @classmethod
    async def update_text_embeddings(cls, q: Query, limit: int, lmdb_path: str, **kw) -> int:
        """Updates text embeddings for the given query `q`.

        This select 'text' and 'link' otypes and updates their embeddings.
        Returns the number of embeddings updated.
        """
        return 0 #FIXME
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
            lambda: batch_extract_embeddings(inputs=inputs, embedding_type='text', md_func=md_func, db_path=lmdb_path, **kw)
        )
        if n_text > 0:
            logger.info(f'  Updated embeddings for {n_text} text rows')
        return n_text

    @classmethod
    async def update_image_embeddings(cls,
                                      q: Query,
                                      lmdb_path: str,
                                      images_dir: str,
                                      limit: int,
                                      fetch_delay: float=0.1,
                                      **kw) -> int:
        """Updates images embeddings for the given query `q`.

        This select 'image' rows, downloads them if needed, and updates their embeddings.
        Returns the number of embeddings updated.
        """
        with db_session:
            rows = q.filter(lambda c: c.otype == 'image' and not c.embed_ts).limit(limit)
        if not rows:
            return 0
        updater = LmdbUpdater(lmdb_path, n_procs=1)
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
                logger.info(f' skipping image {row}, key={key} already in lmdb')
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
                                        images_dir: str,
                                        limit: int,
                                        vlm_prompt: str|None='briefly describe this image',
                                        sys_prompt: str|None=None,
                                        vlm_model: str='fastvlm',
                                        **kw) -> int:
        """Updates image descriptions using VLM for images that have been explored.

        Filters to images where explored_ts is not null, generates descriptions via VLM,
        embeds the descriptions, and updates both LMDB and SQLite metadata.

        Returns the number of descriptions updated.
        """
        return 0 #FIXME
        if not vlm_prompt or not vlm_model:
            return 0
        updater = LmdbUpdater(lmdb_path, n_procs=1)
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
        for task in asyncio.as_completed(vlm_tasks):
            row, desc = await task
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
                          **kw) -> dict[str, int]:
        """Updates the embeddings for all relevant rows in our table.

        This does embeddings of:
        - otype=text: text embeddings of the 'text' field in md
        - otype=link: text embeddings of the 'title' field + url in md
        - otype=image: image embeddings of the image at url (downloaded to images_dir if needed)
        - otype=image: text embeddings of image descriptions generated via VLM

        The embeddings are stored in a NumpyLmdb at the given `lmdb_path`.
        For images, we first download them to the given `images_dir`.

        If `ids` is given, we only update embeddings for those ids, else all ids that don't exist in
        the lmdb. If you specify a positive `limit`, we only update upto that many embeddings, per
        otype. Note that because image embeddings are done locally and are much slower, we apply a
        factor of 2x for the other two functions. In general, we skip rows that are already marked
        done in the sql database (via the `embed_ts` or `explored_ts` fields), and in the case of
        text, we also skip keys that are already in the lmdb.

        By default we use the given `vlm_prompt` to generate image descriptions for images. If you
        want, you can override this and also optionally override the `sys_prompt`. If `vlm_prompt`
        is empty or None, we don't generate descriptions.

        We run the 3 subfunctions (text+link, image embeddings, image descriptions+text embeddings)
        asynchronously in parallel.

        Any kw are passed to the subfunctions, some of which call `batch_extract_embeddings`.

        We return a dict with the number of embeddings updated for each type
        """
        if limit <= 0:
            limit = 10000000
        q = cls.select(lambda c: (ids is None or c.id in ids))
        q = q.order_by(desc(Item.id))
        common_kw = dict(q=q, lmdb_path=lmdb_path, **kw)
        # start async tasks for all 3 subfunctions
        text_task = asyncio.create_task(cls.update_text_embeddings(limit=limit, **common_kw))
        q2 = q.limit(limit)
        #print(f'here with {common_kw}: {q2}')
        image_task = asyncio.create_task(
            cls.update_image_embeddings(images_dir=images_dir, fetch_delay=fetch_delay, limit=limit, **common_kw)
        )
        desc_task = asyncio.create_task(
            cls.update_image_descriptions(vlm_prompt=vlm_prompt,
                                          sys_prompt=sys_prompt,
                                          vlm_model=vlm_model,
                                          images_dir=images_dir,
                                          limit=limit,
                                          **common_kw)
        )
        ret = {}
        ret['n_text'], ret['n_images'], ret['n_descs'] = await asyncio.gather(text_task, image_task, desc_task)
        return ret

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
                          **kw) -> dict[str, int]:
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


class Rel(sql_db.Entity, GetMixin): # type: ignore[name-defined]
    """Relations between items"""
    src = Required('Item', reverse='rel_srcs') # type: ignore[var-annotated]
    tgt = Required('Item', reverse='rel_tgts') # type: ignore[var-annotated]
    rtype = Required(str)
    ts = Required(int)
    PrimaryKey(src, tgt, rtype, ts)
    md = Optional(Json)

    @classmethod
    @db_session
    def get_likes(cls, valid_types: list[str]|None=None) -> list[Item]:
        """Returns Items I've liked, optionally filtered to the given `valid_types`."""
        me = Item.get_me()
        like_rels = cls.select(lambda r: r.src == me and r.rtype == 'like')[:]
        ret = set()
        def maybe_add(obj):
            if valid_types is None or obj.otype in valid_types:
                ret.add(obj)

        for r in like_rels:
            # check the item itself
            try:
                ot = r.tgt.otype
            except UnrepeatableReadError: # tgt was deleted
                continue
            maybe_add(r.tgt)
            # also check its children
            for child in r.tgt.children.select():
                maybe_add(child)
        return list(ret)

    @classmethod
    @db_session
    async def handle_me_action(cls, ids: list[int], action: str, **kw) -> None:
        """Handles an action (e.g. 'like' or 'unlike') from "me" on the given list of `items`."""
        items = Item.select(lambda c: c.id in ids)[:]
        me = Item.get_me()
        ts = int(time.time())
        rels_by_item_by_source = defaultdict(dict)
        for item in items:
            r: None|Rel = None
            match action:
                case 'like': # create or update the rel (only 1 like possible)
                    get_kw = dict(src=me, rtype='like', tgt=item)
                    print(f'in like, got get_kw={get_kw}')
                    if not Rel.get(**get_kw):
                        r = Rel(**get_kw, ts=ts)
                case 'unlike': # delete the rel if it exists
                    get_kw = dict(src=me, rtype='like', tgt=item)
                    r = Rel.get(**get_kw)
                    if r:
                        r.delete()
                case 'queue': # increment count or create new queue rel
                    get_kw = dict(src=me, rtype='queue', tgt=item)
                    r = Rel.get(**get_kw)
                    if r: # Increment count
                        if not r.md:
                            r.md = {}
                        r.md['count'] = r.md.get('count', 1) + 1
                        r.ts = ts  # Update timestamp
                    else: # Create new queue rel with count=1
                        r = Rel(**get_kw, ts=ts, md=dict(count=1))
                case 'unqueue': # remove the queue rel entirely
                    get_kw = dict(src=me, rtype='queue', tgt=item)
                    r = Rel.get(**get_kw)
                    if r:
                        r.delete()
                case 'explore': # explore the given item...source-dependent only
                    pass
                case _:
                    logger.info(f'Unknown me action {action}')
            by_src = rels_by_item_by_source[item.source]
            by_src[item] = []
            if r is not None:
                by_src[item].append(r)
        # now call this method on each source for custom handling
        for source, rels_by_item in rels_by_item_by_source.items():
            src = Source._registry.get(source)
            if src:
                commit()
                await src.handle_me_action(rels_by_item, action, **kw)


def init_sql_db(path: str) -> Database:
    """Initializes the sqlite database at the given `path`"""
    init_sqlite_db(path, db=sql_db)
    with db_session:
        Item.upsert(get_kw=dict(
            source='me',
            stype='user',
            otype='user',
            url='me'
        ))
    return sql_db



class Source(abc.ABC):
    """Base class for all sources. Subclass this.

    Implement can_parse() and parse() methods if you want to handle custom inputs.
    """
    _registry: dict[str, Source] = {}  # Class variable to maintain map from names to Source classes

    def __init__(self,
                 name: str,
                 data_dir: str,
                 sqlite_path: str='',
                 lmdb_path: str='',
                 images_dir: str='',
                 classifiers_dir: str='',
                 **kw):
        self.name = name
        self.data_dir = data_dir
        self.sqlite_path = sqlite_path or join(data_dir, 'collection.sqlite')
        self.lmdb_path = lmdb_path or join(data_dir, 'embeddings.lmdb')
        self.images_dir = images_dir or join(data_dir, 'images')
        self.classifiers_dir = classifiers_dir or join(data_dir, 'classifiers')
        init_sql_db(self.sqlite_path)
        Source._registry[name] = self

    def __repr__(self) -> str:
        return f'Source<{self.name}>'

    def item_for_web(self, item: Item, r: dict[str, Any], rels: list[Rel]) -> None:
        """Source-specific processing of an item for web representation.

        This is called from Item.for_web after generic processing.
        Subclasses can override this to add custom fields to `r`.
        """
        pass

    @classmethod
    def iter_sources(cls) -> list[Source]:
        """Iterates over all registered Source subclasses."""
        return list(cls._registry.values())

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return False

    async def parse(self, url: str, **kw) -> Any:
        """Parses the given url and does whatever it wants."""
        raise NotImplementedError()

    async def handle_me_action(self, rels_by_item: dict[Item, list[Rel]], action: str, **kw) -> None:
        """Handles an action (e.g. 'like' or 'unlike') from "me" on the given list of `items`.

        This is called after generic processing, with the dict mapping from original item to list of
        rels created/deleted/updated for that item.

        The default implementation does nothing, but subclasses can override this.
        """
        pass

    @staticmethod
    async def handle_url(url: str, **data) -> dict[str, Any]:
        """This is the main entry point to handle a given url.

        This finds the appropriate Source subclass that can parse the given url,
        and calls its parse() method with the given url and data.
        """
        for source_cls in Source.__subclasses__():
            if source_cls.can_parse(url):
                # subclasses must be instantiable with no args
                source = source_cls() # type: ignore[call-arg]
                result = await source.parse(url, **data)
                return result
        raise NotImplementedError(f'No source found to parse url {url}')

    @classmethod
    def assemble_post(cls, post, children) -> dict:
        """Assembles a post, generically.

        This method can be overridden by subclasses for custom behavior.
        In this version, we take all children and add them to a subkey called "children".
        """
        assembled_post = recursive_to_dict(post)
        if post.otype != 'post':
            return assembled_post
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

    @classmethod
    def cleanup_embeddings(cls, lmdb_path: str):
        """Cleans up discrepancies between our sqlite and lmdb.

        Note that this doesn't modify the lmdb at all, only the sqlite.
        """
        db = NumpyLmdb.open(lmdb_path, flag='r')
        keys_in_db = set(db.keys())
        n_missing = 0
        n_done = 0
        with db_session:
            def fix(rows: list[Item], key_suffix: str, ts_field: str, fix_missing: bool) -> int:
                """Fix synchronization between sqlite and lmdb.

                - fix_missing: If True, fix items marked done in sqlite but missing in lmdb. If
                  False, fix items present in lmdb but not marked done in sqlite
                """
                n = 0
                for row in rows:
                    key = f'{row.id}:{key_suffix}'
                    if fix_missing:
                        # Fix wrongly marked as done in sqlite but missing in lmdb
                        if key not in keys_in_db:
                            logger.debug(f'Cleaning up {row} with missing key {key}')
                            setattr(row, ts_field, None)
                            n += 1
                    else:
                        # Fix present in lmdb but not marked done in sqlite
                        if key in keys_in_db:
                            logger.debug(f'Marking done for {row} with existing key {key}')
                            d = db.md_get(key)
                            ts = d.get('embed_ts', d.get('embedding_ts', int(time.time())))
                            setattr(row, ts_field, int(time.time()))
                            n += 1
                return n

            # first deal with embeddings wrongly marked as done in sqlite but missing in lmdb
            rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype in ('text', 'link'))
            n_missing += fix(rows, 'text', 'embed_ts', fix_missing=True)
            rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype == 'image')
            n_missing += fix(rows, 'image', 'embed_ts', fix_missing=True)
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is not None and c.explored_ts > 0)
            n_missing += fix(rows, 'text', 'explored_ts', fix_missing=True)
            # now deal with embeddings present in lmdb but not marked done in sqlite
            rows = Item.select(lambda c: c.otype in ('text', 'link') and c.embed_ts is None)
            n_done += fix(rows, 'text', 'embed_ts', fix_missing=False)
            rows = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None)
            n_done += fix(rows, 'image', 'embed_ts', fix_missing=False)
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is None)
            n_done += fix(rows, 'text', 'explored_ts', fix_missing=False)
        del db
        logger.info(f'Cleaned up {n_missing} missing and {n_done} done embeddings')

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
        # if we have a limit, then check our likes to see if those need to be prioritized
        limit = kw.get('limit', None)
        if 0 and limit:
            by_type = defaultdict(list)
            n = 0
            for item in Rel.get_likes():
                if item.embed_ts:
                    continue
                if item.id in ids:
                    by_type[item.otype].append(item)
                    n += 1
            # reset ids to just 'limit' number of these, by type
            if n:
                ids = []
                for otype, items in by_type.items():
                    ids.extend([item.id for item in items])
                logger.info(f'Found {n} liked undone items for source {self.name}, prioritizing {len(ids)}')
        return Item.update_embeddings(lmdb_path=self.lmdb_path, images_dir=self.images_dir, ids=ids, **kw)
