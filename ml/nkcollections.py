"""An abstraction over collections to make it easy to filter/sort/etc

"""

import json
import logging
import os
import re
import shutil
import sys
import time

from argparse import ArgumentParser
from collections import defaultdict
from os.path import abspath, exists, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread
from functools import cache

import termcolor
import tornado.web

from tornado.web import RequestHandler
from pony.orm import (
    composite_index,
    Database,
    db_session,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
) # type: ignore

from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.ml.client import call_vlm
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings
from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request

logger = logging.getLogger(__name__)

sql_db = Database()

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

class Collection(sql_db.Entity, GetMixin):
    id = PrimaryKey(int, auto=True)
    source = Required(str)
    stype = Required(str)
    otype = Required(str, index=True)
    url = Required(str)
    parent = Optional('Collection', reverse='children')
    ts = Required(int, default=lambda: time.time(), index=True)
    added_ts = Required(int, default=lambda: int(time.time()))
    md = Optional(Json)
    children = Set('Collection', reverse='parent')
    composite_index(source, stype, otype)

    @classmethod
    def update_embeddings(cls, lmdb_path: str, images_dir: str, **kw) -> int:
        """Updates the embeddings for all relevant rows in our table.

        The embeddings are stored in a NumpyLmdb at the given `lmdb_path`.
        For images, we first download them to the given `images_dir`.

        Any kw are passed to `batch_extract_embeddings`.

        We return the number of embeddings updated.
        """
        # first do text rows
        rows = cls.select(lambda c: c.otype == 'text')
        logger.info(f'Updating embeddings for upto {rows.count()} text rows: {rows[:5]}...')
        inputs = [(f'{c.id}:text', c.md['text']) for c in rows if c.md and 'text' in c.md]
        def md_func(key, input):
            return dict(embedding_ts=int(time.time()))

        n_text = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='text', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_text} text rows')
        # now do image rows. first we have to download them.
        rows = cls.select(lambda c: c.otype == 'image')
        logger.info(f'Updating embeddings for upto {rows.count()} image rows: {rows[:5]}...')
        inputs = []
        vlm_prompt = 'briefly describe this image'
        futures = {}
        for c in rows:
            url = c.url
            ext = url.split('.')[-1]
            mk = c.md.get('media_key', c.id)
            path = abspath(join(images_dir, f'{mk}.{ext}'))
            downloaded = maybe_dl(url, path)
            key = f'{c.id}:image'
            inputs.append((key, path))
            # start computing text description for this image
            futures[key] = call_vlm.single_future((path, vlm_prompt), model='fastvlm')
        descriptions = {}
        def md_func(key, input):
            # wait for description
            descriptions[key.split(':')[0]+':text'] = desc = futures[key].result()
            return dict(embedding_ts=int(time.time()), desc=desc)

        n_image = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='image', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_image} image rows')
        # finally, add embeddings for the descriptions
        inputs = list(descriptions.items())
        logger.info(f'Updating embeddings for upto {len(inputs)} image descs: {inputs[:3]}')
        def md_func(key, input):
            return dict(embedding_ts=int(time.time()), desc=input)

        n_descs = batch_extract_embeddings(inputs=inputs, db_path=lmdb_path, embedding_type='text', md_func=md_func, **kw)
        logger.info(f'  Updated embeddings for {n_descs} image descs')
        return n_text + n_image + n_descs


def init_sql_db(path: str) -> Database:
    """Initializes the sqlite database at the given `path`"""
    for func in sqlite_pragmas:
        sql_db.on_connect(provider='sqlite')(func)
    sql_db.bind('sqlite', abspath(path), create_db=True)
    #set_sql_debug(True)
    sql_db.generate_mapping(create_tables=True)
    return sql_db

class MyBaseHandler(BaseHandler):
    @property
    def sql_db(self) -> Database:
        return self.application.sql_db

    @property
    def lmdb(self) -> NumpyLmdb:
        return self.application.lmdb

    @property
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Collection))
            return otypes

class GetHandler(MyBaseHandler):
    def get(self, indices):
        otypes = self.get_argument('otypes', ','.join(self.all_otypes)).split(',')
        ids = parse_num_spec(indices)
        # select from the Collection table where ids in ids
        with db_session:
            rows = {r.id: recursive_to_dict(r) for r in Collection.select(lambda c: c.id in ids and (c.otype in otypes))}
            msg = f'hello'
        self.write(dict(msg=msg,
                        indices=indices,
                        rows=rows))

def web_main():
    # load the data file from first arg
    parser = ArgumentParser(description="NK collections main")
    parser.add_argument('sqlite_path', help="The path to the sqlite database")
    parser.add_argument('lmdb_path', help="The path to the lmdb database")
    kw = {}
    def post_parse_fn(args):
        print(f'Got args {args}')

    def on_start(app, args):
        app.sql_db = init_sql_db(args.sqlite_path)
        app.lmdb = NumpyLmdb.open(args.lmdb_path, flag='r')

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/nkcollections.jsx',
                                port=12555,
                                more_handlers=[(r'/get/(.*)', GetHandler)],
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw,
                                on_start=on_start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    web_main()
