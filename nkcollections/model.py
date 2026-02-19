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
from nkpylib.thread_utils import run_async, background_task, classify_func_output
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

sql_db = Database()

J = lambda obj: json.dumps(obj, indent=2)

IMAGE_SUFFIX = 'mn_image'

ACTIONS = 'like unlike dislike undislike queue unqueue explore'.split()


def elapsed_str(ts: float) -> str:
    """Returns a compact human readable string showing elapsed time since the given timestamp.

    - ts: Unix timestamp (seconds since epoch)

    Returns the largest unit that has a value > 1, or the next smaller unit.
    Values are rounded down (no decimals).
    """
    if ts is None or ts == 0:
        return 'null'
    if ts < 0:
        return f'error: {ts}'
    now = time.time()
    diff_secs = int(now - ts)
    if diff_secs < 0:
        return '0s ago'  # Handle future dates
    diff_mins = diff_secs // 60
    diff_hours = diff_mins // 60
    diff_days = diff_hours // 24
    diff_weeks = diff_days // 7
    diff_months = diff_days // 30  # Approximate
    diff_years = diff_days // 365  # Approximate
    # Return the largest unit that has a value > 1, or the next smaller unit
    if diff_years > 1:
        return f'{diff_years}y ago'
    if diff_months > 1:
        return f'{diff_months}mo ago'
    if diff_weeks > 1:
        return f'{diff_weeks}w ago'
    if diff_days > 1:
        return f'{diff_days}d ago'
    if diff_hours > 1:
        return f'{diff_hours}h ago'
    if diff_mins > 1:
        return f'{diff_mins}m ago'
    return f'{diff_secs}s ago'


async def ret_immediate(func_output) -> Any:
    """Given some `func_output`, we want to return something asap.

    If the function is not a generator, then we just return it (running the async part if needed.
    If the function is a generator, we get the first returned value and will return that, after
    running the rest of the function in an background task.
    """
    is_async, is_gen = classify_func_output(func_output)
    print(f'ret_immediate: is_async={is_async}, is_gen={is_gen}')
    if not is_gen: # Not a generator - return as-is
        if is_async:
            return await func_output
        else:
            return func_output
    # at this point, we know it's a generator
    if is_async:
        # Async generator - get first value and schedule rest in background
        async def handle_async_gen():
            try:
                first_value = await func_output.__anext__()
                print(f'Got first value from async generator: {first_value}')
                # Schedule the rest to run in background
                background_task(consume_async_generator(func_output))
                return first_value
            except StopAsyncIteration:
                return None

        return await handle_async_gen()
    else:
        # Sync generator - get first value and schedule rest in background
        try:
            first_value = next(func_output)
            # Schedule the rest to run in background
            background_task(lambda:consume_sync_generator(func_output))
            return first_value
        except StopIteration:
            return None


async def consume_async_generator(async_gen):
    """Consume the rest of an async generator in the background."""
    print(f'in consume_async_generator with {async_gen}')
    a = await async_gen.__anext__() # just to check if we can get a value, will be consumed in the loop below
    print(f'got next val {a}')
    try:
        async for _ in async_gen:
            print(f'consuming async generator, got value {_}')
            pass  # Just consume, don't do anything with the values
    except Exception as e:
        logger.warning(f"Error consuming async generator: {e}")


def consume_sync_generator(gen):
    """Consume the rest of a sync generator in the background."""
    try:
        for _ in gen:
            pass  # Just consume, don't do anything with the values
    except Exception as e:
        logger.warning(f"Error consuming sync generator: {e}")


def timed(func: Callable) -> Callable:
    """Decorator to time a function and log its duration."""
    async def async_wrapper(*args, **kw):
        start = time.time()
        result = await func(*args, **kw)
        end = time.time()
        logger.info(f'Function {func.__name__} took {end - start:.2f} seconds')
        return result

    def sync_wrapper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        logger.info(f'Function {func.__name__} took {end - start:.2f} seconds')
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

async def maybe_dl(url: str, path: str, fetch_delay: float=0.1, timeout: float=-1) -> bool:
    """Downloads the given url to the given dir if it doesn't already exist there (and is not empty).

    - fetch_delay: minimum delay in seconds between fetches, to avoid overwhelming servers. This
      limit is per domain.
    - timeout: if > 0, the maximum time in seconds to wait for the download. If the download doesn't
      complete in that time then raises TimeoutError. If <= 0, no timeout is applied.

    Returns if we actually downloaded the file.
    """
    if exists(path) and os.path.getsize(path) > 0:
        return False
    logger.debug(f'downloading image {url} -> {path}')
    r = await asyncio.wait_for(
            make_request_async(
                url,
                headers={'Accept': 'image/*,video/*'},
                min_delay=fetch_delay,
            ), timeout if timeout > 0 else None)
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

    def get_source(self) -> Source|None:
        """Returns the Source object for this item, if available."""
        return Source._registry.get(self.source)

    def image_path(self, images_dir: str|None=None) -> str:
        """Returns the image path for our row.

        If `images_dir` is None, we try to find the appropriate source to get the images_dir, and if
        we can't find it, we just return a path in the current directory.
        """
        if images_dir is None:
            source = self.get_source()
            if source:
                images_dir = source.images_dir
            else:
                images_dir = ''
        url = self.url
        ext = self.md.get('ext', url.split('.')[-1])
        mk = self.md.get('media_key', self.id)
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

    async def for_web(self, r: dict[str, Any]) -> None:
        """Cleans up this item for web use.

        The web representation of this object is in `r`, which this modifies. Rels are fetched
        and processed internally, including merging with containing post rels.

        Item changes:
        - for images with embeddings, adds 'local_path' relative to cwd
        - if this item has a parent, adds 'parent_url' with the parent's url
        - if this item has an ancestor that's a 'user', adds 'user_name' and 'user_url'
        - if this item is a user, then we add fields 'compact' and 'detailed" with
          strings of what to display

        We also deal with rels by calling `rels_for_web`
        """
        # add local image path if we have it
        if self.otype == 'image' and self.embed_ts and self.embed_ts > 0:
            # Find the appropriate source to get images_dir
            local_path = self.image_path()
            if exists(local_path):
                r['local_path'] = os.path.relpath(local_path)
                if 1: # replace with our image resizer
                    r['local_path'] = 'http://192.168.1.135:8183/thumbs/w300/' + r['local_path']
                else: # serve directly from here
                    r['local_path'] = '/data/'+r['local_path']
                #print(f'Got local path {r["local_path"]}')
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
            compact = f'{self.source}: <a href="{self.url}" target="_blank">{self.name or self.url}</a>'
            if self.explored_ts:
                if self.explored_ts > 0:
                    compact += f'<br>Last explored: {elapsed_str(self.explored_ts)}'
                else:
                    compact += f'<br>Error'
            else:
                compact += f'<br>Never explored'
            stats = dict(**self.md.get('stats', {}))
            now = time.time()
            if stats:
                image_url = stats.pop('image_url', None)
                compact += f'<br><ul class="user-stats">'
                for k, v in stats.items():
                    if k == 'ts': # skip the update time, we don't care
                        continue
                    if k.endswith('_ts'):
                        v = f'{elapsed_str(v)}'
                    if isinstance(v, float):
                        v = f'{v:.2f}'
                    compact += f'<li>{k}: {v}</li>'
                compact += '</ul></pre>'
                if image_url:
                    compact += f'<img src="{image_url}" />'
            detailed = compact #TODO
            r['compact'] = compact
            #r['detailed'] = detailed
        self.rels_for_web(r)
        # call the source-specific version of this function
        source = Source._registry.get(self.source)
        if source:
            source.item_for_web(self, r)

    def rels_for_web(self, r: dict[str, Any]) -> None:
        """Deal with rels for web representation.

        This does:
        - fetches rels for this item and its containing post (if different)
        - merges them with item rels taking precedence over post rels for same rtype
        - adds a 'rels' sub-dict to `r` with keys being the rtype and values being dicts or lists
        - if there's only one rel of a given type, the value is a dict with 'ts' and any metadata
        - if there are multiple rels of a given type, the value is a list of such dicts
        - special processing:
          - for 'like' rels, we only keep the latest one (highest ts)
        """
        me = Item.get_me()
        # Get rels for this item
        item_rels = list(Rel.select(lambda r: r.src == self or r.tgt == self))
        # Get rels for containing post if different from this item
        post_rels = []
        post = self.get_closest(otype='post')
        if post and post.id != self.id:
            post_rels = list(Rel.select(lambda r: r.src == post or r.tgt == post))
        # Group rels by type
        item_rels_by_type = defaultdict(list)
        post_rels_by_type = defaultdict(list)
        for rel in item_rels:
            item_rels_by_type[rel.rtype].append(rel)
        for rel in post_rels:
            post_rels_by_type[rel.rtype].append(rel)
        # Merge: item rels override post rels for same rtype
        merged_rels_by_type = {}
        all_rtypes = set(item_rels_by_type.keys()) | set(post_rels_by_type.keys())
        for rtype in all_rtypes:
            if rtype in item_rels_by_type:
                # Item has rels of this type, use them
                merged_rels_by_type[rtype] = item_rels_by_type[rtype]
            else:
                # Only post has rels of this type, use post's
                merged_rels_by_type[rtype] = post_rels_by_type[rtype]
        # Process each rel type for web output
        R = r['rels'] = {}
        for rtype, rel_list in merged_rels_by_type.items():
            if len(rel_list) == 1: # Single rel: store as dict
                rel = rel_list[0]
                md = dict(ts=rel.ts, src_id=rel.src.id, tgt_id=rel.tgt.id)
                if rel.md:
                    md.update(rel.md)
                R[rtype] = md
            else: # Multiple rels: store as list of dicts
                rel_dicts = []
                for rel in rel_list:
                    md = dict(ts=rel.ts, src_id=rel.src.id, tgt_id=rel.tgt.id)
                    if rel.md:
                        md.update(rel.md)
                    rel_dicts.append(md)
                R[rtype] = rel_dicts

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
                path = row.image_path()
                try:
                    await maybe_dl(row.url, path, fetch_delay=fetch_delay, timeout=30)
                except Exception as e:
                    logger.warning(f'Error downloading image for row id={row.id}, url={row.url}, path={path}: {e}')
                    path = ''
            return row, path

        download_tasks = [dl_image(row) for row in rows]
        # kick off embeddings as downloads complete
        embedding_tasks = []
        async def embed_image_task(row, key, path):
            try:
                if not path or not exists(path) or os.path.getsize(path) == 0:
                    raise FileNotFoundError(f'File not found or empty')
                emb = await embed_image.single_async(path, timeout=5, model='mobilenet', use_cache=kw.get('use_cache', True))
                #FIXME emb = await embed_image.single_async(path, model='clip', use_cache=kw.get('use_cache', True))
            except Exception as e:
                logger.warning(f'Error embedding image for row id={row.id}, path={path}: {e}')
                print(traceback.format_exc())
                emb = None
            return (row, key, emb)

        done = set()
        for task in asyncio.as_completed(download_tasks):
            row, path = await task
            key = f'{row.id}:{IMAGE_SUFFIX}'
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
                                        limit: int,
                                        vlm_prompt: str|None='Briefly describe this image. Include a list of tags at the end.',
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
            path = row.image_path()
            try:
                desc = await call_vlm.single_async((path, messages), timeout=30, model=vlm_model)
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
                        text_embedding = await embed_text.single_async(desc, timeout=5, model='qwen_emb')
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
                          source: str|None=None,
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
        if source is not None:
            q = q.filter(lambda c: c.source == source)
        q = q.order_by(desc(Item.id))
        common_kw = dict(q=q, lmdb_path=lmdb_path, **kw)
        # start async tasks for all 3 subfunctions
        text_task = asyncio.create_task(cls.update_text_embeddings(limit=limit, **common_kw))
        q2 = q.limit(limit)
        #print(f'here with {common_kw}: {q2}')
        image_task = asyncio.create_task(
            cls.update_image_embeddings(fetch_delay=fetch_delay, limit=limit, **common_kw)
        )
        desc_task = asyncio.create_task(
            cls.update_image_descriptions(vlm_prompt=vlm_prompt,
                                          sys_prompt=sys_prompt,
                                          vlm_model=vlm_model,
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
        ret = run_async(cls.update_embeddings_async(
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
        #print(f'Sync Done with update_embeddings in {cls}, got {ret}')
        return ret


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
    async def handle_me_action(cls, ids: list[int], action: str, **kw) -> None:
        """Handles an action (e.g. 'like' or 'unlike') from "me" on the given list of `items`."""
        with db_session:
            items = Item.select(lambda c: c.id in ids)[:]
            me = Item.get_me()
            ts = int(time.time())
            rels_by_item_by_source = defaultdict(dict)
            for item in items:
                # Update seen_ts for any action
                item.seen_ts = ts
                r: None|Rel = None
                match action:
                    case 'like': # create or update the rel (only 1 like possible)
                        get_kw = dict(src=me, rtype='like', tgt=item)
                        if not Rel.get(**get_kw):
                            r = Rel(**get_kw, ts=ts)
                        # also remove any 'dislike' rels for this item
                        r = Rel.get(src=me, rtype='dislike', tgt=item)
                        if r:
                            r.delete()
                    case 'unlike': # delete the rel if it exists
                        get_kw = dict(src=me, rtype='like', tgt=item)
                        r = Rel.get(**get_kw)
                        if r:
                            r.delete()
                    case 'dislike': # create or update the rel (only 1 like possible)
                        get_kw = dict(src=me, rtype='dislike', tgt=item)
                        if not Rel.get(**get_kw):
                            r = Rel(**get_kw, ts=ts)
                        # also remove any 'like' rels for this item
                        r = Rel.get(src=me, rtype='like', tgt=item)
                        if r:
                            r.delete()
                    case 'undislike': # delete the rel if it exists
                        get_kw = dict(src=me, rtype='dislike', tgt=item)
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
                    case 'explore': # explore the given item: just add a new rel
                        r = Rel(src=me, rtype='explore', tgt=item, ts=ts)
                    case _:
                        logger.info(f'Unknown me action {action}')
        return


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

    def item_for_web(self, item: Item, r: dict[str, Any]) -> None:
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
    def first_source(cls) -> Source|None:
        """Returns the first source in our registry, or None if no sources."""
        return next(iter(cls._registry.values()), None)

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return False

    async def parse(self, url: str, **kw) -> dict[str, Any]:
        """Parses the given url and returns GetHandler params.

        The function can either return the params directly, or for efficiency, it can yield the
        params quickly (as soon as it knows them) and then do the rest of the processing after that.
        """
        raise NotImplementedError()

    async def handle_me_action(self, ids: list[int], action: str, **kw) -> None:
        """Handles an action (e.g. 'like' or 'unlike') from "me" on the given list of `items`.

        This is called at some point after generic processing, with a list of item ids.

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
                source = Source._registry.get(source_cls.NAME)
                print(f'Got source {source}')
                if not source:
                    continue
                result = await ret_immediate(source.parse(url, **data))
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
            def fix(rows: list[Item], key_suffix: str, ts_field: str, fix_missing: bool, db) -> int:
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
            n_missing += fix(rows, 'text', 'embed_ts', fix_missing=True, db=db)
            rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype == 'image')
            n_missing += fix(rows, IMAGE_SUFFIX, 'embed_ts', fix_missing=True, db=db)
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is not None and c.explored_ts > 0)
            n_missing += fix(rows, 'text', 'explored_ts', fix_missing=True, db=db)
            # now deal with embeddings present in lmdb but not marked done in sqlite
            rows = Item.select(lambda c: c.otype in ('text', 'link') and c.embed_ts is None)
            n_done += fix(rows, 'text', 'embed_ts', fix_missing=False, db=db)
            rows = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None)
            n_done += fix(rows, IMAGE_SUFFIX, 'embed_ts', fix_missing=False, db=db)
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is None)
            n_done += fix(rows, 'text', 'explored_ts', fix_missing=False, db=db)
        del db
        logger.info(f'Cleaned up {n_missing} missing and {n_done} done embeddings')

    @db_session
    def update_embeddings(self, **kw):
        """Updates the embeddings for this Source.

        By default, this just calls Item.update_embeddings, with the `source` explicitly set to our
        source.

        We pass all `kw` to Item.update_embeddings.
        """
        if 'source' not in kw:
            kw['source'] = self.name
        #logger.info(f'In {self}, updating embeddings for {len(ids)} items')
        return Item.update_embeddings(lmdb_path=self.lmdb_path, images_dir=self.images_dir, **kw)
