"""Twitter wrapper"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time

from argparse import ArgumentParser
from datetime import datetime
from os.path import exists, dirname, join, abspath
from urllib.parse import urlencode

import requests

from pony.orm import db_session, select
from pony.orm.core import Entity
from pyquery import PyQuery as pq # type: ignore
from tqdm import tqdm

from nkpylib.script_utils import cli_runner
from nkpylib.stringutils import save_json
from nkpylib.web_utils import make_request
from nkpylib.ml.nkcollections import Item, init_sql_db, Source, web_main

logger = logging.getLogger(__name__)

class Twitter(Source):
    """A wrapper for processing twitter into a collection"""
    NAME = 'twitter'
    DIR = 'db/twitter'
    IMAGES_DIR = join(DIR, 'images/')
    SQLITE_PATH = join(DIR, 'twitter_collection.sqlite')
    LMDB_PATH = join(DIR, 'twitter_embeddings.lmdb')

    def __init__(self, **kw):
        """Initializes twitter source."""
        super().__init__(sqlite_path=self.SQLITE_PATH)

    @db_session
    def create_collection_from_archive(self, path: str, **kw) -> list[Entity]:
        """Creates `Item` rows from a twitter archive.

        The archive is in my nkbase, accessible via http://localhost:10002/p/x.com

        A single tweet in the archive looks something like this currently:

        {
          "avatar": "https://pbs.twimg.com/profile_images/1361631454795546624/7fnmx7Fq_normal.jpg",
          "db_path": "raw/x.com/paddycosgrave/status/2011738746035847243",
          "display_name": "Paddy Cosgrave",
          "handle": "paddycosgrave",
          "image_urls": [
            "https://pbs.twimg.com/media/G-shuRAXEAAgiDJ?format=jpg&name=360x360",
            "https://pbs.twimg.com/media/G-shuRcbQAUSJNI?format=jpg&name=360x360"
          ],
          "iso_ts": "2026-01-15T09:54:28.000Z",
          "likes": 897,
          "rel_time": "19h",
          "replies": 47,
          "reposter_display_name": "",
          "reposter_handle": "",
          "reposts": 218,
          "text": "Scholar: In India alone, British empire killed 165 million in just 40 years & stole trillions. Irish clown: British weren\u2019t so bad, even not that bad in Ireland compared to those who fought for Irish freedom. Yikes",
          "url": "https://x.com/paddycosgrave/status/2011738746035847243",
          "views": 0
        }

        """
        with open(path) as f:
            tweets = [t for t in json.load(f)['objs'] if '/status/' in t['url']]
        # sort tweets by ts
        tweets.sort(key=lambda t: t.get('iso_ts', ''))
        md_fields = 'handle display_name likes replies reposts views'.split()
        ret = []
        for i, t in tqdm(enumerate(tweets)):
            # add the tweet id
            t['id'] = t['url'].split('/')[-1]
            get_kw = dict(
                url=t['url'],
            )
            if Item.get(**get_kw):
                continue
            set_kw = dict(
                source=self.NAME,
                stype='blog',
                otype='post',
                name=t['id'],
                ts=datetime.fromisoformat(t['iso_ts'].replace('Z', '+00:00')).timestamp(),
                md={field: t[field] for field in md_fields if field in t},
            )
            item = Item(**get_kw, **set_kw)
            ret.append(item)
            # add text item
            child_kw = dict(
                source=item.source,
                stype=item.stype,
                ts=item.ts,
                parent=item,
            )
            if t.get('text', ''):
                t_item = Item(
                    otype='text',
                    url=item.url+'#text',
                    name=f"tweet text from {t['id']}",
                    md={'text': t['text']},
                    **child_kw
                )
                ret.append(t_item)
            # add images
            for j, im in enumerate(t.get('image_urls', [])):
                mk = im.split('/')[-1]
                md = dict()
                # image urls come in a few formats
                if '.' in mk: # has extension, so use it directly
                    mk, ext = mk.rsplit('.', 1)
                    md = dict(media_key=mk, ext=ext)
                elif 'format=' in mk: # no extension, but has format (ext)
                    mk, params = mk.split('?', 1)
                    params = dict([p.split('=') for p in params.split('&')])
                    params['ext'] = params.pop('format')
                    md = dict(media_key=mk, **params)
                else:
                    logger.warning(f'Unknown image url format: {im}')
                    continue
                im_item = Item(
                    otype='image',
                    url=im,
                    name=f"image {j+1} from {t['id']}",
                    md=md,
                    **child_kw
                )
                ret.append(im_item)
        logging.info(f'Created {len(ret)} items from archive {path}')
        return ret

def read_archive(path: str='db/twitter/20260116-0028.json', **kw):
    """Reads a twitter archive from given path and prints summary info."""
    twitter = Twitter()
    items = twitter.create_collection_from_archive(path, **kw)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    cli_runner([read_archive, web_main])
