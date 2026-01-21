"""An abstraction over collections to make it easy to filter/sort/etc

"""

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

from nkpylib.ml.client import call_vlm
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings
from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request

logger = logging.getLogger(__name__)

sql_db = Database()

J = lambda obj: json.dumps(obj, indent=2)


def maybe_dl(url: str, path: str) -> bool:
    """Downloads the given url to the given dir if it doesn't already exist there (and is not empty).

    Returns if we actually downloaded the file.
    """
    if exists(path) and os.path.getsize(path) > 0:
        return False
    logger.debug(f'downloading image {url} -> {path}')
    r = make_request(url, headers={'Accept': 'image/*,video/*'})
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
    def update_embeddings(cls,
                          lmdb_path: str,
                          images_dir: str,
                          ids: list[int]|None=None,
                          vlm_prompt: str|None='briefly describe this image',
                          sys_prompt: str|None=None,
                          vlm_model: str='fastvlm',
                          **kw) -> int:
        """Updates the embeddings for all relevant rows in our table.

        The embeddings are stored in a NumpyLmdb at the given `lmdb_path`.
        For images, we first download them to the given `images_dir`.

        If `ids` is given, we only update embeddings for those ids, else all ids that don't exist in
        the lmdb.

        By default we use the given `vlm_prompt` to generate image descriptions for images. If you
        want, you can override this and also optionally override the `sys_prompt`. If `vlm_prompt`
        is empty or None, we don't generate descriptions.

        Any kw are passed to `batch_extract_embeddings`.

        We return the number of embeddings updated.
        """
        #FIXME this can be quite slow right now
        #FIXME this is rather complicated
        #FIXME we want this to be async-friendly, as well as batchable/resumable
        #FIXME we also maybe want this to run periodically and automatically?
        db = NumpyLmdb.open(lmdb_path, flag='r')
        #TODO filter by ids
        def postprocess_rows(rows):
            with db_session:
                for row in rows:
                    row.embed_ts = time.time()

        # first do text rows
        with db_session:
            rows = cls.select(lambda c: c.otype == 'text' and not c.embed_ts)
            logger.info(f'Updating embeddings for upto {rows.count()} text rows: {rows[:5]}...')
            inputs = [(f'{c.id}:text', c.md['text']) for c in rows if c.md and 'text' in c.md]
        def md_func(key, input):
            return dict(embedding_ts=int(time.time()))

        n_text = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='text', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_text} text rows')
        postprocess_rows(rows)
        # now do links
        with db_session:
            rows = cls.select(lambda c: c.otype == 'link' and not c.embed_ts)
            logger.info(f'Updating embeddings for upto {rows.count()} link rows: {rows[:5]}...')
            inputs = [(f'{c.id}:text', f"{c.md['title']}: {c.url}") for c in rows if c.md and 'title' in c.md]
        n_text = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='text', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_text} text rows')
        postprocess_rows(rows)
        # now do image rows. first we have to download them.
        with db_session:
            rows = cls.select(lambda c: c.otype == 'image' and not c.embed_ts)
            logger.info(f'Updating embeddings for upto {rows.count()} image rows: {rows[:5]}...')
            inputs = []
            futures = {}
            descriptions = {}
            done = {}
            for c in rows:
                url = c.url
                ext = c.md.get('ext', url.split('.')[-1])
                mk = c.md.get('media_key', c.id)
                path = abspath(join(images_dir, f'{mk}.{ext}'))
                downloaded = maybe_dl(url, path)
                key = f'{c.id}:image'
                inputs.append((key, path))
                if vlm_prompt:
                    desc_key = f'{c.id}:text'
                    desc = db.md_get(desc_key)
                    if not desc: # start computing text description for this image
                        if sys_prompt:
                            messages = [dict(role='system', content=sys_prompt), dict(role='user', content=vlm_prompt)]
                        else:
                            messages = vlm_prompt
                        logger.debug(f'Calling VLM for image description for key={key}, path={path}, image_data[:30]')
                        futures[key] = call_vlm.single_future((path, messages), model=vlm_model)

                    else: # we already have the desc
                        done[desc_key] = desc['desc']
        def md_func(key, input):
            desc_key = key.split(':')[0]+':text'
            if desc_key in done: # we already have the desc
                desc = done[desc_key]
            else: # wait for description
                if key in futures:
                    descriptions[desc_key] = desc = futures[key].result()
                else:
                    desc = ''
            return dict(embedding_ts=int(time.time()), desc=desc)

        n_image = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='image', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_image} image rows')
        postprocess_rows(rows)
        # finally, add embeddings for the descriptions
        inputs = list(descriptions.items())
        logger.info(f'Updating embeddings for upto {len(inputs)} image descs: {inputs[:3]}')
        def md_func(key, input):
            return dict(embedding_ts=int(time.time()), desc=input)

        n_descs = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='text', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_descs} image descs')
        return n_text + n_image + n_descs


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

    def __init__(self, name: str, sqlite_path: str, **kw):
        self.name = name
        self.sqlite_path = sqlite_path
        init_sql_db(sqlite_path)
        Source._registry[name] = self.__class__

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

class ActionHandler(MyBaseHandler):
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
    def post(self):
        # get pos from json argument
        data = json.loads(self.request.body)
        pos = data.get('pos', [])
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

class ClusterHandler(MyBaseHandler):
    def post(self):
        data = json.loads(self.request.body)
        print(f'In clustering, got data {data}')
        ret = dict(msg='hi', **data)
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    web_main()
