"""Twitter wrapper


To convert user ids to usernames, use the following:

https://www.postel.app/twitter-user-id-converter

curl 'https://www.postel.app/api/twitter-user-id-converter/convert' \
  --compressed \
  -X POST \
  -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:146.0) Gecko/20100101 Firefox/146.0' \
  -H 'Accept: */*' \
  -H 'Accept-Language: en-US,en;q=0.5' \
  -H 'Accept-Encoding: gzip, deflate, br, zstd' \
  -H 'Content-Type: application/json' \
  -H 'Referer: https://www.postel.app/twitter-user-id-converter' \
  -H 'sentry-trace: 75cc6122221c4f48a89b1c36c079676e-aeaeba6415e4b41d-0' \
  -H 'baggage: sentry-environment=production,sentry-public_key=97303b17ebf99a7ba23abea0fe383b93,sentry-trace_id=75cc6122221c4f48a89b1c36c079676e,sentry-org_id=4509984483246080,sentry-sampled=false,sentry-sample_rand=0.9587772993856623,sentry-sample_rate=0.1' \
  -H 'Origin: https://www.postel.app' \
  -H 'DNT: 1' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'Connection: keep-alive' \
  -H 'Priority: u=0' \
  -H 'TE: trailers' \
  --data-raw '{"input":"459846693"}'

"""

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

import lmdb # type: ignore
import numpy as np
import requests

from pony.orm import db_session, select
from pony.orm.core import Entity
from pyquery import PyQuery as pq # type: ignore
from tqdm import tqdm

from nkpylib.nkcollections.collections import Item, init_sql_db, Source, web_main, J, embeddings_main
from nkpylib.ml.nklmdb import NumpyLmdb, LmdbUpdater
from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.script_utils import cli_runner
from nkpylib.stringutils import save_json
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

class Twitter(Source):
    """A wrapper for processing twitter into a collection"""
    NAME = 'twitter'

    def __init__(self, **kw):
        """Initializes twitter source."""
        super().__init__(name=self.NAME, data_dir='db/twitter')

    @classmethod
    def assemble_post(cls, post, children) -> dict:
        """Assemble a complete Twitter post with text and images"""
        post_data = recursive_to_dict(post)

        # Find text and images
        text_items = [c for c in children if c.otype == 'text']
        image_items = [c for c in children if c.otype == 'image']
        video_items = [c for c in children if c.otype == 'video']

        # Create content blocks for consistency with other sources
        content_blocks = []
        media_blocks = []
        
        # Add text content
        for text_item in text_items:
            content_blocks.append(dict(
                type='text',
                data=recursive_to_dict(text_item)
            ))
        
        # Add media content
        for img_item in image_items:
            block = dict(
                type='image',
                data=recursive_to_dict(img_item)
            )
            content_blocks.append(block)
            media_blocks.append(block)
            
        for video_item in video_items:
            block = dict(
                type='video',
                data=recursive_to_dict(video_item)
            )
            content_blocks.append(block)
            media_blocks.append(block)

        post_data['content_blocks'] = content_blocks
        post_data['media_blocks'] = media_blocks
        return post_data

    @db_session
    def create_collection_from_archive(self, path: str, **kw) -> list[Item]:
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
        ret: list[Item] = []
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
        self.update_embeddings()
        return ret

    @db_session
    def update_embeddings(self, **kw):
        """Updates embeddings for our twitter collection."""
        # first compute basic image/text
        super().update_embeddings(**kw)
        # now compute post embeddings based on their content blocks
        limit = kw.get('limit', 0)
        if limit <= 0:
            limit = 10000000
        posts = Item.select(lambda i: i.source == self.NAME and i.otype == 'post').limit(limit)
        logger.info(f'got {len(posts)} posts to update embeddings for')
        db = NumpyLmdb.open(self.lmdb_path, flag='c')
        updater = LmdbUpdater(self.lmdb_path, n_procs=1)
        for i, post in tqdm(enumerate(posts)):
            post_key = f'{post.id}:text'
            if post.id == 51540:
                logger.info(f' Post {post.id} children: {[(f"{c.id}:text" in db) for c in Item.select(lambda it: it.parent == post)[:]]}, in db: {post_key in db}')
            if post_key in db:
                continue
            children = Item.select(lambda i: i.parent == post)[:]
            embs = []
            for c in children:
                try:
                    cur = db.get(f'{c.id}:text', None)
                except lmdb.Error:
                    db = NumpyLmdb.open(self.lmdb_path, flag='c')
                    cur = db.get(f'{c.id}:text', None)
                if cur is not None:
                    embs.append(cur)
            if not embs:
                continue
            # average the embeddings
            avg = np.mean(np.stack(embs), axis=0)
            logger.debug(f'  {post_key}: From {len(children)} children, got {len(embs)} embeddings: {embs} -> {avg}')
            md = dict(n_embs=len(embs), embedding_ts=time.time(), method='average')
            updater.add(post_key, embedding=avg, metadata=md)
            post.embed_ts = md['embedding_ts']
            #if i > 400: break
        updater.commit()



def read_archive(path: str='db/twitter/20260116-0028.json', **kw):
    """Reads a twitter archive from given path and prints summary info."""
    twitter = Twitter()
    items = twitter.create_collection_from_archive(path, **kw)

def web(**kw):
    """Runs the collections web interface."""
    print(f'got kw: {kw}')
    twitter = Twitter()
    return web_main(sqlite_path=twitter.sqlite_path, lmdb_path=twitter.lmdb_path)

def test(**kw):
    """Tests out the twitter source."""
    twitter = Twitter()
    with db_session:
        posts = Item.select(lambda i: i.source == Twitter.NAME and i.otype == 'post' and i.id > 40000)
        assembled = Twitter.assemble_posts(posts[:10])
    print(J(assembled))

def update_embeddings(**kw):
    """Updates embeddings"""
    from tumblr import Tumblr
    tw = Twitter()
    embeddings_main(**kw)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    cli_runner([test, read_archive, web, update_embeddings])
