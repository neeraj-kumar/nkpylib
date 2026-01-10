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
from os.path import abspath, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread

import termcolor
import tornado.web

from tornado.web import RequestHandler
from pony.orm import * # type: ignore
from pony.orm.core import Entity, EntityMeta, SetInstance # type: ignore

from nkpylib.nkpony import sqlite_pragmas, GetMixin
from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server

logger = logging.getLogger(__name__)

sql_db = Database()

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

def init_sql_db(path: str) -> Database:
    """Initializes the sqlite database at the given `path`"""
    for func in sqlite_pragmas:
        sql_db.on_connect(provider='sqlite')(func)
    sql_db.bind('sqlite', abspath(path), create_db=True)
    #set_sql_debug(True)
    sql_db.generate_mapping(create_tables=True)
    return sql_db


class GetHandler(BaseHandler):
    def get(self, indices):
        msg = f'hello'
        self.write(dict(msg=msg, indices=indices))

def web_main():
    # load the data file from first arg
    parser = ArgumentParser(description="NK collections main")
    #parser.add_argument('data_path', help="The path to the data file")
    kw = {}
    def post_parse_fn(args):
        pass

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/nkcollections.jsx',
                                port=12555,
                                more_handlers=[(r'/get/(.*)', GetHandler)],
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw)


if __name__ == '__main__':
    web_main()
