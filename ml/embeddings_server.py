"""Server for exploring embeddings"""

from __future__ import annotations

import logging
import os
import sys
import time

from argparse import ArgumentParser
from os.path import abspath, join, dirname, exists

import tornado, tornado.web

from pony.orm import * # type: ignore
from pony.orm.core import Entity, EntityMeta, SetInstance # type: ignore
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

from nkpylib.ml.embeddings import FeatureSet, NumpyLmdb, LmdbUpdater
from nkpylib.pony import GetMixin, recursive_to_dict, sqlite_pragmas
from nkpylib.utils import specialize
from nkpylib.web_utils import (
    BaseHandler,
    make_request_async,
    simple_react_tornado_server,
    setup_and_run_server,
)

logger = logging.getLogger(__name__)

tag_db = Database() # Pony Tag database


class Tag(tag_db.Entity, GetMixin):
    tag_id = PrimaryKey(int, auto=True)
    id = Required(str, index=True)
    key = Required(str, index=True)
    value = Optional(str, default='')
    type = Optional(str, default='')
    composite_index(id, type, key, value)
    composite_index(type, key, value)
    composite_index(key, value)

    def __repr__(self):
        return f'<{self.id} {self.type}: {self.key}={self.value}>'

def init_tag_db(path: str) -> Database:
    """Initializes our tag database at given `path`"""
    for func in sqlite_pragmas:
        tag_db.on_connect(provider='sqlite')(func)
    tag_db.bind('sqlite', abspath(path), create_db=True)
    #set_sql_debug(True)
    tag_db.generate_mapping(create_tables=True)
    return tag_db


class MyBaseHandler(BaseHandler):
    @property
    def db(self) -> NumpyLmdb:
        return self.application.db

    def by_dim(self, dim: int=0) -> tuple[list[str], list[float]]:
        """Returns list of ids sorted by the given dimension"""
        values, ids = zip(*sorted([(self.db[k][dim], k) for k in self.db.keys() if k.startswith('tt')]))
        print(f'Got values: {values[:10]}, ids: {ids[:10]}')
        return list(ids), [float(v) for v in values]


class IndexHandler(MyBaseHandler):
    def get(self):
        ids, values = self.by_dim(0)
        self.write(dict(status='ok', n=len(self.db), ids=ids, values={id: v for id, v in zip(ids, values)}))

class PointMetadataHandler(MyBaseHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        ids = data.get('ids', [])
        ret = {id: self.db.md_get(id) for id in ids}
        self.write(dict(status='ok', data=ret))


class TagsHandler(MyBaseHandler):
    def get(self):
        # group tags by id
        tags = {}
        with db_session:
            for tag in Tag.select():
                tags.setdefault(tag.id, []).append(dict(key=tag.key, value=tag.value, type=tag.type))
        self.write(dict(status='ok', tags=tags))

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        key = data.get('key')
        value = data.get('value', '')
        type = data.get('type', '')
        if not key:
            self.write(dict(status='error', message='Missing key'))
            return
        tag = Tag.get_or_create(key=key, value=value, type=type)
        self.write(dict(status='ok', tag=recursive_to_dict(tag)))

class Application(tornado.web.Application):
    def __init__(self:self, db: NumpyLmdb, **kw):
        handlers = [
            (r"/index/", IndexHandler),
            (r"/pt_md/", PointMetadataHandler),
            (r"/tags/", TagsHandler),
            #(r'/favicon.ico', tornado.web.StaticFileHandler, {'path': 'static/favicon.ico'}),
        ]
        settings = {
            "debug": True,
            "static_path": os.path.join(os.path.dirname(__file__), "static"),
            "compress_response": True,
        }
        self.db = db
        super().__init__(handlers, **settings)


def start_server(path: str, parser, tag_path: str, **kw):
    print(f'starting server with path {path}, {tag_path}, and kw {kw}')
    jsx_path = join(dirname(__file__), 'static', 'embeddings.jsx')
    init_tag_db(tag_path)
    db = NumpyLmdb.open(path, 'r')
    setup_and_run_server(parser=parser, make_app=lambda: Application(db), default_port=8908)


if __name__ == '__main__':
    parser = ArgumentParser(description='Embeddings Exploration Server')
    parser.add_argument('path', help='Path to the embeddings lmdb file')
    parser.add_argument('-t', '--tag_path', help='Path to the tags sqlite')
    parser.add_argument('keyvalue', nargs='*', help='Key=value pairs to pass to the function')
    args = parser.parse_args()
    kwargs = vars(args)
    for keyvalue in kwargs.pop('keyvalue', []):
        if '=' not in keyvalue:
            raise ValueError(f'Invalid key=value pair: {keyvalue}')
        key, value = keyvalue.split('=', 1)
        value = specialize(value)
        kwargs[key] = value
    start_server(parser=parser, **kwargs)
