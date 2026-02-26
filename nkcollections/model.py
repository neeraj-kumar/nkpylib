from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import time

from collections import defaultdict
from os.path import abspath, exists, join, dirname
from typing import Any, Callable

from pony.orm import (
    composite_index,
    commit,
    Database,
    db_session,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
    set_sql_debug,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore

from nkpylib.nkpony import init_sqlite_db, GetMixin, recursive_to_dict
from nkpylib.thread_utils import (
    background_task,
    classify_func_output,
    consume_async_generator,
    consume_sync_generator,
)
from nkpylib.time_utils import elapsed_str, timed

logger = logging.getLogger(__name__)

sql_db = Database()

J = lambda obj: json.dumps(obj, indent=2)

ACTIONS = 'like unlike dislike undislike queue unqueue explore'.split()

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


ItemUserSQL = """CREATE VIEW ItemUser AS
SELECT i.id,
CASE
    WHEN p.otype='user' then p.id
    WHEN gp.otype='user' then gp.id
    ELSE NULL END
AS user_id
FROM item i
LEFT JOIN item p ON i.parent = p.id
LEFT JOIN item gp ON p.parent = gp.id
WHERE p.otype='user' OR gp.otype='user';"""

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
    composite_index(source, otype)
    # all other metadata
    md = Optional(Json)
    # cumulative seconds spent on this item
    dwell_time = Required(float, default=0.0)
    children = Set('Item', reverse='parent') # type: ignore[var-annotated]
    rel_srcs = Set('Rel', reverse='src') # type: ignore[var-annotated]
    rel_tgts = Set('Rel', reverse='tgt') # type: ignore[var-annotated]
    scores = Set('Score', reverse='id') # type: ignore[var-annotated]

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

        We also deal with rels by calling `rels_for_web` and scores with `scores_for_web`.
        """
        # add local image path if we have it
        if self.otype == 'image' and self.embed_ts and self.embed_ts > 0:
            # Find the appropriate source to get images_dir
            local_path = self.image_path()
            if exists(local_path):
                r['local_path'] = os.path.relpath(local_path)
                if 1: # replace with our image resizer
                    r['local_path'] = 'http://192.168.1.135:8183/thumbs/w300/' + r['local_path']
                    #r['local_path'] = 'http://aphex.local:8183/thumbs/w300/' + r['local_path']
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
        self.scores_for_web(r)
        # call the source-specific version of this function
        source = Source._registry.get(self.source)
        if source:
            await source.item_for_web(self, r)

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

    def scores_for_web(self, r: dict[str, Any]) -> None:
        """Deal with scores for web representation.

        This fetches scores for this item and adds a 'scores' sub-dict to `r` with keys being the
        ttype:tag and values being dicts with 'score', 'ts', and any metadata.
        """
        S = r['scores'] = {}
        for score in self.scores.select():
            cur = S.setdefault(score.ttype, {})
            cur[score.tag] = score.score


class Score(sql_db.Entity, GetMixin): # type: ignore[name-defined]
    """Scores for items of various types"""
    id = Required(Item) # type: ignore[var-annotated]
    ttype = Required(str)
    tag = Required(str)
    score = Required(float, index=True)
    ts = Required(float, default=lambda: time.time())
    md = Optional(Json)
    PrimaryKey(id, ttype, tag)
    #composite_index(id, ttype, tag, score) # can't do this in pony, but sqlite does it


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
    NAME = 'Source' # override this

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

    async def item_for_web(self, item: Item, r: dict[str, Any]) -> None:
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

    @db_session
    def update_embeddings(self, **kw):
        """Updates the embeddings for this Source.

        By default, this just calls the embeddings module functions, with the `source` explicitly
        set to our source.

        We pass all `kw` to the embeddings functions.
        """
        from nkpylib.nkcollections.embeddings import update_embeddings
        if 'source' not in kw:
            kw['source'] = self.name
        #logger.info(f'In {self}, updating embeddings for {len(ids)} items')
        return update_embeddings(lmdb_path=self.lmdb_path, images_dir=self.images_dir, **kw)
