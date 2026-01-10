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
from os.path import join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread

import termcolor
import tornado.web

from tornado.web import RequestHandler

from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server

logger = logging.getLogger(__name__)


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
