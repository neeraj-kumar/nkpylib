"""A tumblr API that goes through the web.


"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
from os.path import exists, dirname, join, abspath
from typing import Any
from urllib.parse import urlencode, urlparse

import numpy as np
import requests

from pony.orm import db_session, flush, select, commit
from pony.orm.core import Entity
from pyquery import PyQuery as pq # type: ignore
from tqdm import tqdm

from nkpylib.nkcollections.nkcollections import (
    embeddings_main,
    init_sql_db,
    Item,
    Rel,
    Source,
    web_main,
)
from nkpylib.nkcollections.workers import LikesWorker
from nkpylib.ml.nklmdb import NumpyLmdb, LmdbUpdater
from nkpylib.ml.embeddings import Embeddings
from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.script_utils import cli_runner
from nkpylib.stringutils import save_json
from nkpylib.web_utils import make_request, make_request_async

logger = logging.getLogger(__name__)

J = lambda obj: json.dumps(obj, indent=2)

DEFAULT_CONFIG_PATH = '.tumblr_config.json'

def obj_get(path: str, obj: dict, default: Any=None) -> Any:
    """Gets a nested object value from a dict given a dot-separated path."""
    parts = path.split('.')
    cur = obj
    for part in parts:
        if part in cur:
            cur = cur[part]
        else:
            return default
    return cur

class TumblrApi:
    """Core Tumblr API functionality - authentication, requests, data parsing"""
    NAME = 'tumblr'
    MIN_DELAY = 0.3 # between requests

    COMMON_HEADERS = {
        "Accept": "application/json;format=camelcase", # Accept header used by Tumblr’s API
        #"Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-us", # Language preference
        "Content-Type": "application/json; charset=utf8", # The server expects JSON UTF‑8 payload
        "X-Version": "redpop/3/0//redpop/", # A proprietary version header used by Tumblr’s front‑end code
        # CORS‑related fetch metadata – keep them even though they are not used by `requests` directly.
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin", # or same-site for api calls?
        "Priority": "u=0", # Request priority hint (the value is not interpreted by the server)
    }

    def __init__(self, config_path: str=DEFAULT_CONFIG_PATH, **kw):
        """Initializes the Tumblr API wrapper.

        Expects a config JSON file with at least:
        {
            "api_token": "your_api_token_here",
            "cookies": {
                "cookie_name": "cookie_value",
                ...
            }
        }
        """
        self.data_dir = kw.get('data_dir', 'db/tumblr')
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logger.debug(f'Initialized Tumblr API with token {self.api_token[:10]}... and cookies: {self.cookies}')
        self.csrf = ''

    @property
    def api_token(self) -> str:
        """Returns the API token from the config."""
        return self.config['api_token']

    @property
    def cookies(self) -> dict[str, str]:
        """Returns the cookies from the config."""
        return self.config.get('cookies', {})

    def update_config(self, **kw):
        """Updates the config with new values.

        Any keys with '.' in them are treated as nested keys.
        This also updates the config file on disk.
        """
        logger.info(f'Updating our config at {self.config_path}: {kw}')
        for key, value in kw.items():
            parts = key.split('.')
            cur = self.config
            for part in parts[:-1]:
                if part not in cur:
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = value
        with open(self.config_path+'.tmp', 'w') as f:
            json.dump(self.config, f, indent=2)
        os.rename(self.config_path, self.config_path + '.bak')
        os.rename(self.config_path+'.tmp', self.config_path)

    async def make_web_req(self, endpoint: str, referer:str='', **kw) -> dict:
        """Make a request to the Tumblr web interface"""
        url = endpoint if 'tumblr.com' in endpoint else f'https://www.tumblr.com/{endpoint}'
        #print(f'Hitting url: {url}')
        headers = {
            "Authorization": f'Bearer {self.api_token}',
            **self.COMMON_HEADERS
        }
        if referer:
            headers['Referer'] = referer
        resp = await make_request_async(url, cookies=self.cookies, headers=headers, min_delay=self.MIN_DELAY, **kw)
        req = resp.request
        logger.debug(f'Made a {req.method.upper()} to {req.url} with status {resp.status_code}')
        logger.debug(f'Req Headers: {J(dict(req.headers))}\n{req._cookies}')
        if resp.cookies:
            logger.info(f'  Resp cookies: {resp.cookies}')
            # update the 'sid' cookie in our config
            if 'sid' in resp.cookies:
                self.update_config(**{'cookies.sid': resp.cookies['sid']})
        if resp.status_code != 200:
            logger.warning(f'Error {resp.status_code} from Tumblr: {resp.text[:500]}')
        resp_path = join(self.data_dir, 'last_tumblr_response.html')
        try:
            shutil.copy2(resp_path, resp_path.replace('.html', '_prev.html'))
        except Exception:
            pass
        with open(resp_path, 'w') as f:
            f.write(resp.text)
        try:
            obj = resp.json()
        except Exception:
            doc = pq(resp.text)
            # find the script tag with the initial state
            state = doc('script#___INITIAL_STATE___').text()
            try:
                obj = json.loads(state)
            except Exception as e:
                logger.warning(f'Failed to parse initial state JSON: {e}: {resp.text[:100]}')
                raise ValueError(f'Failed to parse initial state JSON: {e}')
        if 'csrfToken' in obj:
            self.csrf = obj["csrfToken"]
        return obj

    async def make_api_req(self, endpoint: str, **kw):
        """Make a request to the Tumblr API"""
        headers = {
            "Authorization": f'Bearer {self.api_token}',
            **self.COMMON_HEADERS
        }
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = f'https://api.tumblr.com/v2/{endpoint}'
        #print(f'got headers {headers}, {self.cookies}, {kw}')
        resp = await make_request_async(url, headers=headers, cookies=self.cookies, min_delay=self.MIN_DELAY, **kw)
        try:
            obj = resp.json()
        except Exception as e:
            print(resp.text)
            raise Exception(f'Failed to fetch endpoint {endpoint}: {e}')
        with open(join(self.data_dir, 'last_tumblr_api_response.json'), 'w') as f:
            json.dump(obj, f, indent=2)
        if obj['meta']['status'] != 200:
            raise Exception(f'Failed to fetch {url}: {obj["meta"]}')
        return obj['response']

    async def like_post(self, post_id: str, reblog_key: str) -> dict:
        """Like a post by ID and reblog key"""
        data = dict(
            id=post_id,
            reblog_key=reblog_key,
        )
        if self.csrf: # you need a valid csrf key from a page (doesn't matter which, as long as it's recent)
            headers = {
                "X-CSRF": self.csrf,
                **self.COMMON_HEADERS
            }
            obj = await self.make_web_req('api/v2/user/like', method='POST', headers=headers, json=data)
        else: # TODO this requires Oauth access, not just api key
            endpoint = 'user/like'
            obj = await self.make_api_req(endpoint, method='POST', json=data)
        print(f'liked post {post_id}: {obj}')
        return obj

    async def get_dashboard(self) -> list[dict]:
        """Returns our "dashboard". Useful for updating the csrf"""
        obj = await self.make_web_req('')
        logger.debug(J(obj)[:500])
        return obj['Dashboard']

    async def get_blog_content(self, blog_name: str, n_posts: int=20) -> list[dict]:
        """Get blog content from the web interface.

        You can either fetch it from blog_name.tumblr.com, in which case you can use /page/N, but
        then it doesn't have the JSON object in the source code (hence you have to parse the html),
        or you can go to tumblr.com/blog_name, which does have the json, but then pagination works
        differently.

        """
        assert blog_name.strip()
        if 'https' in blog_name:
            blog_name = urlparse(blog_name).netloc.split('.tumblr.com')[0]
        ret: list[dict] = []
        page = 1
        while len(ret) < n_posts:
            endpoint = f'{blog_name}/page/{page}'
            endpoint = f'https://{blog_name}.tumblr.com/page/{page}'
            endpoint = blog_name
            try:
                obj = await self.make_web_req(endpoint)
            except ValueError:
                break
            posts = obj['PeeprRoute']['initialTimeline']['objects']
            posts = [p for p in posts if p['objectType'] == 'post']
            ret.extend(posts)
            break # because we're using the parseable version
        return ret

    async def get_blog_archive(self, blog_name: str, n_posts: int = 20, offset: int=0) -> tuple[list[dict], int, int]:
        """Get blog archive via API.

        Note that some blogs don't allow access to the archive, so you must get it via the web
        interface.

        Returns (posts, offset, totalPosts)
        """
        blog_name = self.get_blog_name(blog_name)
        posts: list[dict] = []
        total = 0
        next_link = dict(
            npf='true',
            reblog_info='true',
            context='archive',
            offset=str(offset),
        )
        logger.info(f'Trying to fetch blog archive for {blog_name}, n_posts={n_posts}, offset={offset}')
        with db_session:
            u = self.get_blog_user(blog_name)
            if u.explored_ts and u.explored_ts < 0: # skip blogs that are marked inaccessible
                return (posts, offset, total)
            while offset < n_posts:
                #print(f'initial next link: {J(next_link)}')
                endpoint = f'blog/{blog_name}/posts'
                if next_link:
                    endpoint = f'blog/{blog_name}/posts?{urlencode(next_link)}'
                try:
                    commit()
                    obj = await self.make_api_req(endpoint)
                except Exception as e:
                    if '404' in str(e):
                        logger.warning(f'Blog archive for {blog_name} not found (404). It may be private or inaccessible.')
                        u.explored_ts = -1 # mark as inaccessible
                        commit()
                        return (posts, offset, total)
                if obj.get('links'):
                    ext_link = obj['links'].get('next', {}).get('queryParams', {})
                batch = obj['posts'] or []
                posts.extend(batch)
                total = obj['totalPosts']
                offset += 20
                logger.debug(f'Got offset {offset}, {n_posts}, {len(batch)}, {total}, {obj.get("links")}')
                next_link['offset'] = str(offset)
                if not batch or not obj.get('links', []):
                    break
        logger.info(f'For blog {blog_name}, got {len(posts)} posts (total: {total})')
        return (posts, offset, total)

    async def get_my_likes(self):
        """Returns our likes"""
        #https://www.tumblr.com/api/v2/user/likes?fields[blogs]=?advertiser_name,?avatar,?blog_view_url,?can_be_booped,?can_be_followed,?can_show_badges,?description_npf,?followed,?is_adult,?is_member,name,?primary,?theme,?title,?tumblrmart_accessories,url,?uuid&limit=21&reblog_info=true
        obj = await self.make_api_req('https://www.tumblr.com/api/v2/user/likes')
        print(J(obj)[:500])

    async def get_post_notes(self, blog_name: str, post_id: int, n_notes: int=40, mode: str='reblogs_only') -> list[dict]:
        """Returns notes related to a post.

        Reblogs look like this:
        {
          "type": "reblog",
          "timestamp": 1769703415,
          "blogName": "...", # the blog that reblogged
          "blogUuid": "t:...",
          "blogUrl": "https://....tumblr.com/",
          "followed": true, # whether we follow the reblogger
          "avatarUrl": { # reblogger's avatar
            "64": "https://64.media.tumblr.com/....pnj",
            "128": "https://64.media.tumblr.com/....pnj"
          },
          "blogTitle": "...", # reblogger's blog title
          "postId": "{post_id}", # post id on the reblogger's blog, i.e. tumblr.com/{blogName}/{postId}
          "reblogParentBlogName": "..." # only reference to who this was reblogged from
        }
        """
        dash = await self.get_dashboard()
        before_timestamp = 0
        ret = []
        referer = f'https://www.tumblr.com/{blog_name}/{post_id}'
        while len(ret) < n_notes:
            endpoint = f'api/v2/blog/{blog_name}/notes?id={post_id}&mode={mode}&sort=desc'
            if before_timestamp:
                endpoint += f'&before_timestamp={before_timestamp}'
            #print(endpoint)
            obj = await self.make_web_req(endpoint, method='get', referer=referer)
            notes = obj['response'].pop('notes', [])
            #print(J(obj)[:1500])
            if not notes:
                break
            ret.extend(notes)
            before_timestamp = obj_get('response.links.next.queryParams.beforeTimestamp', obj, 0)
            logger.debug(f'got {len(notes)} notes, total {len(ret)}, next before_timestamp: {before_timestamp}')
        return ret


class Tumblr(TumblrApi, Source):
    """Tumblr integration with the collections system"""
    NAME = 'tumblr'

    def __init__(self, config_path: str=DEFAULT_CONFIG_PATH, explore_likes:bool=True, **kw):
        """Initializes the Tumblr source.

        If you set `explore_likes` to True, then when the user likes a post, we will explore it.
        """
        data_dir = kw.pop('data_dir', 'db/tumblr')
        TumblrApi.__init__(self, config_path=config_path, data_dir=data_dir, **kw)
        Source.__init__(self, name=self.NAME, data_dir=data_dir, **kw)
        self.explore_likes = explore_likes

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return 'tumblr.com' in url

    @classmethod
    def get_blog_name(cls, url: str) -> str:
        if '/' not in url:
            return url
        # first get the blog name, which is either in the format xyz.tumblr.com or tumblr.com/xyz
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower()
        blog_name = ''
        if netloc.endswith('.tumblr.com'):
            blog_name = netloc.split('.tumblr.com')[0]
            if blog_name in ('www', 'm'):
                blog_name = ''
        if not blog_name:
            blog_name = path.split('/')[1]
        return blog_name

    async def parse(self, url: str, n_posts: int=300, **kw) -> Any:
        """Parses the given url and returns an appropriate set of GetHandler params.

        For now, this simply returns the list of posts that are children of the tumblr blog's Item.
        If there are none, we fetch one batch of posts from the archive.
        """
        blog_name = self.get_blog_name(url)
        with db_session:
            # now look for the appropriate blog user item
            u = self.get_blog_user(blog_name)
        # yield the get params to the user and do the rest of the processing later
        ret = dict(source=self.NAME, ancestor=u.id, assemble_posts=True, limit=500)
        logger.info(f'For url {url} yielding {ret}')
        yield ret
        offset = 0
        # add posts
        commit()
        posts, offset, total = await self.get_blog_archive(blog_name, n_posts=n_posts)
        self.create_collection_from_posts(posts, blog_name=blog_name, next_link=offset)

    def get_blog_user(self, blog_name: str, **kw) -> Entity:
        """Returns the blog user item for the given `blog_name`.

        Any kw are added to the metadata of the user item.
        """
        get_kw = dict(source=self.NAME, stype='blog', otype='user',
                      url=f'https://{blog_name}.tumblr.com/')
        u = Item.get(**get_kw)
        if u is None:
            u = Item(**get_kw, ts=int(time.time()), md=dict(blog_name=blog_name))
        for key, value in kw.items():
            u.md[key] = value
        return u

    async def check_logged_in(self) -> None:
        """Checks if we're logged in"""
        dash = await self.get_dashboard() # to update csrf
        logger.info('checking for logged in...')
        logger.info(J(dash)[:150])

    async def handle_me_action(self, ids: list[int], action: str, **kw) -> None:
        """Handles an action (e.g. 'like' or 'unlike') from "me" on the given list of `items`.

        This is called after generic processing, with a list of item ids and the action taken.

        For 'queue' actions, we fetch the post notes and update our database as follows:
        - each reblog note has a reblogger and rebloggee
        - we accumulate counts per user: total, as reblogger, as rebloggee
        - goals:
          1. in rels: keep track of users related to this post, with associated counts
          2. in items: for each user, increment counts

        (We also do this if self.explore_likes is True and the action is 'like'.)

        For action 'explore', it must apply to a user. We call `parse` on that user's blog url.
        """
        explore = False
        if action == 'queue':
            explore = True
        if action == 'like' and self.explore_likes:
            explore = True
        if action == 'explore':
            with db_session:
                for item_id in ids:
                    user = Item[item_id].get_closest(otype='user')
                    if not user:
                        continue
                    logger.info(f'Tumblr exploring user: {user} -> {user.url}')
                    try:
                        print('Before commit...')
                        commit()
                        print(f'After commit...')
                        #await self.parse(user.url)
                        #FIXME any awaits are not working here...
                        # async sleep for a second
                        await asyncio.sleep(1)
                        print(f'after async sleep')
                        await self.check_logged_in()
                        print(f'After check_logged_in...')
                        posts, offset, total = await self.get_blog_archive(user.url, n_posts=300)
                        print(f'Got {len(posts)} posts')
                        self.create_collection_from_posts(posts)
                    except Exception as e:
                        print(f'WTF failed to get blog archive for {user.url}: {e}')
                        print(traceback.format_exc())
                return

        if not explore:
            return
        logger.info(f'Tumblr handling action {action}: {ids}')
        with db_session:
            me = Item.get_me()
            for item_id in ids:
                post = Item[item_id].get_closest(otype='post')
                if not post:
                    continue
                if post.explored_ts: # skip already explored posts
                    continue
                logger.info(f'  {item_id} -> post: {post}: {post.to_dict()}')
                post_id = post.md['post_id']
                post.explored_ts = time.time()
                try:
                    commit()
                    await self.check_logged_in()
                    notes = await self.get_post_notes(post.md['blog_name'], post_id)
                except Exception as e:
                    logger.warning(f'  Failed to get notes for post {post.md["post_id"]}: {e}')
                    post.explored_ts = -1
                    continue
                logger.info(f'  Got {len(notes)} notes for post {post_id}')
                logger.debug(f'  Got {len(notes)} notes for post {post_id}: {J(notes)[:500]}')
                # add notes summary to our database
                counts_by_user = defaultdict(Counter)
                following = set()
                for note in notes:
                    if note['type'] != 'reblog':
                        continue
                    reblogger = note['blogName']
                    rebloggee = note['reblogParentBlogName']
                    counts_by_user[reblogger]['reblogger'] += 1
                    counts_by_user[reblogger]['total'] += 1
                    counts_by_user[rebloggee]['rebloggee'] += 1
                    counts_by_user[rebloggee]['total'] += 1
                    if note.get('followed'):
                        following.add(reblogger)
                # actually add to the database
                for user, counts in counts_by_user.items():
                    # update the user item
                    u = self.get_blog_user(user, explored_via=post.id)
                    u.md.setdefault('n_queued_reblogs', 0)
                    u.md['n_queued_reblogs'] += counts['total']
                    # update/create rels
                    r = Rel(src=post, tgt=u, rtype='queued_post_reblogs', ts=int(time.time()), md=dict(
                            n_as_reblogger=counts['reblogger'],
                            n_as_rebloggee=counts['rebloggee'],
                            n_total=counts['total'],
                        ),
                    )
                    if user in following: # make a rel for us following them
                        matching = Rel.select(lambda r: r.src == me and r.tgt == u and r.rtype == 'follows')
                        if not matching.exists():
                            Rel(src=me, tgt=u, rtype='follows', ts=int(time.time()), md=dict(
                                    via_post_id=post.id,
                                ),
                            )

    @classmethod
    def assemble_post(cls, post, children) -> dict:
        """Assemble a complete Tumblr post with all content blocks"""
        post_data = recursive_to_dict(post)
        # Find videos and their corresponding poster images to remove
        ignore = set()
        for c in children:
            if c.otype == 'video' and c.md and 'poster_media_key' in c.md:
                ignore.add(c.md['poster_media_key'])
        # Group children by type, maintaining order, but skip 'ignore' media_keys
        content_blocks = []
        media_blocks = []
        for child in sorted(children, key=lambda c: c.id):
            if child.md and 'media_key' in child.md and child.md['media_key'] in ignore:
                continue
            block = dict(
                type=child.otype,
                data=recursive_to_dict(child)
            )
            content_blocks.append(block)
            # Also add to media_blocks if it's media
            if child.otype in ['image', 'video']:
                # Skip poster images (they're already handled by their parent video)
                if child.otype == 'image' and child.md and 'poster_for' in child.md:
                    continue
                media_blocks.append(block)
        post_data['content_blocks'] = content_blocks
        post_data['media_blocks'] = media_blocks
        return post_data

    @db_session
    def create_collection_from_posts(self,
                                     posts: list[dict],
                                     next_link=None,
                                     **kw) -> list[Entity]:
        """Creates `Item` rows from tumblr posts.

        This creates separate rows (with appropriate types) for each:
        - post
        - image/video/text/link content block within the post
        - for videos: another row for the poster image of that video

        Any additional `kw` are added to the metadata of each 'post' row.
        """
        ret = []
        ts = time.time()
        ids_to_update = {}
        for post in posts:
            # get the user item
            blog_name = post['blog']['name']
            u = self.get_blog_user(
                    blog_name=blog_name,
                    title=post['blog'].get('title', ''),
                    description=post['blog'].get('description', ''),
                    uuid=post['blog'].get('uuid', ''),
            )
            # update the seen and explored time for this user
            u.seen_ts = ts
            u.explored_ts = ts
            if next_link:
                u.md['next_link'] = next_link
            ret.append(u)
            # create the main post Item
            pi: Any = Item.upsert(get_kw=dict(
                    source=self.NAME,
                    stype='blog',
                    otype='post',
                    url=post['postUrl']
                ),
                parent=u,
                ts=post.get('timestamp', int(time.time())),
                md=dict(
                    post_id=post['id'],
                    reblog_key=post['reblogKey'],
                    tags=post.get('tags', []),
                    n_notes=post.get('noteCount', 0),
                    n_likes=post.get('likeCount', 0),
                    n_reblogs=post.get('reblogCount', 0),
                    summary=post.get('summary', ''),
                    original_type=post.get('originalType', ''),
                    reblogged_from=post.get('rebloggedFromUrl', ''),
                    **kw
                )
            )
            ret.append(pi)
            # Get content from either direct content or trail
            content = post['content'] or (post['trail'][0]['content'] if post.get('trail') else [])
            # Create child collections for each content block
            for i, c in enumerate(content):
                content_type = c['type']
                match content_type:
                    case 'image':
                        media = c['media'][0] if isinstance(c['media'], list) else c['media']
                        url = media['url'].replace('.pnj', '.png')
                        md = dict(
                            w=media.get('width'),
                            h=media.get('height'),
                            media_key=media.get('mediaKey', media['url'].split('/')[3].rsplit('.', 1)[0])
                        )
                    case 'video':
                        media = c.get('media', c)
                        url = media['url']
                        md = dict(
                            w=media.get('width'),
                            h=media.get('height'),
                            media_key=media['url'].split('/')[3],
                            provider=c.get('provider', ''),
                        )
                        if 'poster' in c:
                            video_url = media['url']
                            poster = c['poster'][0]
                            poster_url = poster['url'].replace('.pnj', '.png')
                            poster_media_key=poster.get('mediaKey', poster['url'].split('/')[3].rsplit('.', 1)[0]),
                            while isinstance(poster_media_key, (list, tuple)):
                                poster_media_key = poster_media_key[0]
                            md.update(poster_url=poster_url, poster_media_key=poster_media_key)
                    case 'text':
                        url = f"{pi.url}#text_{i}"
                        md = dict(
                            text=c.get('text', ''),
                        )
                    case 'link':
                        url = c.get('url', c.get('displayUrl', ''))
                        md = dict(
                            display_url=c.get('displayUrl', ''),
                            title=c.get('title', ''),
                            description=c.get('description', '')
                        )
                    case _:
                        # For other content types
                        url = f"{pi.url}#{content_type}_{i}"
                        md = c
                # Create child collection object
                cc = Item.upsert(get_kw=dict(
                        source=pi.source,
                        stype=pi.stype,
                        otype=content_type,
                        url=url,
                    ),
                    ts=post.get('timestamp', int(time.time())),
                    md=md,
                    parent=pi
                )
                ret.append(cc)
                flush() # so that we can get cc.id
                # if it was a video, also add the poster as a separate image collection
                if content_type == 'video' and 'poster' in c:
                    poster_md = dict(
                        w=poster.get('width'),
                        h=poster.get('height'),
                        media_key=poster_media_key,
                        poster_for=cc.id, # type: ignore[attr-defined]
                        video_url=video_url,
                    )
                    pcc = Item.upsert(get_kw=dict(
                            source=pi.source,
                            stype=pi.stype,
                            otype='image',
                            url=poster_url,
                        ),
                        ts=post.get('timestamp', int(time.time())),
                        md=poster_md,
                        parent=pi,
                    )
                    ret.append(pcc)
        return ret

    async def update_blogs(self):
        """Updates all blogs in our config."""
        blogs = self.config['blogs']
        for i, name in enumerate(blogs):
            print(f'\nProcessing blog {i+1}/{len(blogs)}: {name}')
            now = time.time()
            with db_session:
                u = self.get_blog_user(name)
                if u.explored_ts and u.explored_ts < 0: # skip inaccessible blogs
                    continue
                diff = now - (u.explored_ts or 0)
                if diff < 3600*24:
                    print(f'  Last explored was {diff//3600} hours ago, skipping until 24 hours')
                    continue
            try:
                posts, next_link, total = await self.get_blog_archive(name)
                cols = self.create_collection_from_posts(posts, blog_name=name)
                print(f'Created {len(posts)} -> {len(cols)} post collections for {name}')
            except Exception as e:
                logger.warning(f'Failed to process blog {name}: {e}')
                print(traceback.format_exc())
                continue

    @db_session
    def update_embeddings(self, **kw):
        """Updates embeddings for our tumblr collection."""
        # first compute basic image/text
        counts = super().update_embeddings(**kw)
        return counts #FIXME fix the rest of this
        # now compute post embeddings based on their content blocks
        limit = kw.get('limit', 0)
        if limit <= 0:
            limit = 10000000
        posts = Item.select(lambda i: i.source == self.name and i.otype == 'post').limit(limit)
        logger.info(f'got {len(posts)} posts to update embeddings for')
        db = NumpyLmdb.open(self.lmdb_path, flag='c')
        updater = LmdbUpdater(self.lmdb_path, n_procs=1)
        n = 0
        for i, post in tqdm(enumerate(posts)):
            post_key = f'{post.id}:image'
            if post_key in db:
                continue
            children = Item.select(lambda i: i.parent == post)[:]
            embs = []
            for c in children:
                if not c.embed_ts:
                    continue
                try:
                    cur = db.get(f'{c.id}:image', None)
                    print(f'  {post.id} -> {c}, {c.embed_ts} -> {cur}, {self.lmdb_path}')
                except lmdb.Error:
                    db = NumpyLmdb.open(self.lmdb_path, flag='c')
                    cur = db.get(f'{c.id}:image', None)
                if cur is not None:
                    embs.append(cur)
            if not embs:
                continue
            print('here')
            # average the embeddings
            avg = np.mean(np.stack(embs), axis=0)
            logger.debug(f'  {post_key}: From {len(children)} children, got {len(embs)} embeddings: {embs} -> {avg}')
            md = dict(n_embs=len(embs), embed_ts=time.time(), method='average')
            updater.add(post_key, embedding=avg, metadata=md)
            post.embed_ts = md['embed_ts']
            n += 1
            #if i > 400: break
        updater.commit()
        logger.info(f'Updated embeddings for {n} tumblr posts')


def process_posts(posts):
    for i, p in enumerate(posts):
        content = p['content'] or p['trail'][0]['content']
        print(f'\nPost {i}: {p["id"]} (reblog key: {p["reblogKey"]}), tags: {p.get("tags", [])[:5]}')
        for c in content:
            match c['type']:
                case 'image':
                    m = c['media'][0]
                    if 'mediaKey' in m:
                        mk = m['mediaKey']
                    else:
                        mk = m['url'].split('/')[3]
                    print(f'  image: {mk}')
                case 'video':
                    m = c['media']
                    mk = m['url'].split('/')[3]
                    print(f'  video: {mk}')
                case 'text':
                    print(f'  text: {c["text"]}')
                case 'link':
                    print(f'  link: {c["displayUrl"]}')
                case _:
                    print(f'  unknown content type: {c["type"]}')
        if i == 500:
            print('Liking this post...')
            like_post(post_id=p['id'], reblog_key=p['reblogKey'])


def update_blogs(config_path: str, **kw):
    tumblr = Tumblr(config_path=config_path)
    tumblr.update_blogs()

def test_post(config_path: str, **kw):
    tumblr = Tumblr(config_path=config_path)
    notes = tumblr.get_post_notes('zegalba', 701480526115192832)
    print(f'Got {len(notes)} notes: {J(notes)[:3000]}')

def update_embeddings(**kw):
    """Updates embeddings"""
    t = Tumblr()
    embeddings_main(**kw)

def web(**kw):
    """Runs the collections web interface."""
    t = Tumblr()
    #print(J(t.get_dashboard())[:500])
    return web_main(sqlite_path=t.sqlite_path, lmdb_path=t.lmdb_path)

def gen_benchmark(**kw):
    """Generates a benchmark set"""
    return #FIXME remove when we want to generate a new set
    t = Tumblr()
    lw = LikesWorker(embs=Embeddings([t.lmdb_path]), classifiers_dir=t.classifiers_dir)
    lw.gen_benchmark_set()

def run_benchmark(name='like-benchmark-20260202-175744', **kw):
    """Runs the benchmark with given `name`"""
    t = Tumblr()
    lw = LikesWorker(embs=Embeddings([t.lmdb_path]), classifiers_dir=t.classifiers_dir)
    lw.run_benchmark(name=name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    cli_runner([update_blogs, test_post, update_embeddings, web, gen_benchmark, run_benchmark],
               config_path=dict(default=DEFAULT_CONFIG_PATH, help='Path to the tumblr config json file'))
