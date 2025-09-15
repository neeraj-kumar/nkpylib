"""Server for exploring embeddings"""

from __future__ import annotations

import os
import sys
import time

from argparse import ArgumentParser
from os.path import join, dirname, exists

import tornado, tornado.web

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

from nkpylib.web_utils import (
    BaseHandler,
    make_request_async,
    simple_react_tornado_server,
)
from nkpylib.utils import specialize

class DataHandler(BaseHandler):
    def get(self):
        return {"data": "This is some data from the server"}


def start_server(path: str, parser, **kw):
    more_handlers = [
        (r"/data", DataHandler),
        #(r"/update", UpdateHandler),
        #(r"/poster/(.*).jpg", MoviePosterHandler),
        #(r"/summary/(.*)", MovieSummaryHandler),
        #(r"/to_watch/(.*)", ToWatchHandler),
        #(r"/letterboxd/(.*)", LetterboxdHandler),
        #(r'/favicon.ico', tornado.web.StaticFileHandler, {'path': 'static/favicon.ico'}),
    ]
    print(f'starting server with path {path} and kw {kw}')
    jsx_path = join(dirname(__file__), 'static', 'embeddings.jsx')
    simple_react_tornado_server(
        parser=parser,
        jsx_path=jsx_path,
        css_filename='embeddings.css',
        port=8908,
        more_handlers=more_handlers,
    )


if __name__ == '__main__':
    parser = ArgumentParser(description='Embeddings Exploration Server')
    parser.add_argument('path', help='Path to the embeddings lmdb file')
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
