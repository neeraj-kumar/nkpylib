"""A tumblr API that goes through the web.


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
from typing import Any
from urllib.parse import urlencode, urlparse

import requests

from pony.orm import db_session, select
from pony.orm.core import Entity
from pyquery import PyQuery as pq # type: ignore

from nkpylib.ml.nkcollections import Item, init_sql_db, Source
from nkpylib.nkpony import sqlite_pragmas, GetMixin, recursive_to_dict
from nkpylib.script_utils import cli_runner
from nkpylib.stringutils import save_json
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

J = lambda obj: json.dumps(obj, indent=2)

DEFAULT_CONFIG_PATH = '.tumblr_config.json'

class Tumblr(Source):
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
        data_dir = kw.pop('data_dir', 'db/tumblr')
        super().__init__(name=self.NAME, data_dir=data_dir, **kw)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f'Initialized Tumblr API with token {self.api_token[:10]}... and cookies: {self.cookies}')
        self.csrf = ''

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return 'tumblr.com' in url

    def parse(self, url: str, n_posts: int=300, **kw) -> Any:
        """Parses the given url and returns an appropriate set of GetHandler params.

        For now, this simply returns the list of posts that are children of the tumblr blog's Item.
        If there are none, we fetch one batch of posts from the archive.
        """
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
        with db_session:
            # now look for the appropriate blog user item
            u = self.get_blog_user(blog_name)
            offset = 0
            # add posts if we don't have any yet
            if not select(p for p in Item if p.parent == u).exists(): # type: ignore[attr-defined]
                posts, offset, total = self.get_blog_archive(blog_name, n_posts=n_posts)
                self.create_collection_from_posts(posts, blog_name=blog_name, next_link=offset)
            return dict(source=self.NAME, parent=u.id, assemble_posts=True)

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
        os.rename(self.config_path, self.config_path + '.bak')
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def make_web_req(self, endpoint: str, **kw) -> dict:
        """Make a request to the Tumblr web interface"""
        url = endpoint if 'tumblr.com' in endpoint else f'https://www.tumblr.com/{endpoint}'
        print(f'Hitting url: {url}')
        headers = {
            "Authorization": f'Bearer {self.api_token}',
            **self.COMMON_HEADERS
        }
        resp = make_request(url, cookies=self.cookies, headers=headers, min_delay=self.MIN_DELAY, **kw)
        if resp.cookies:
            logger.info(f'  Resp cookies: {resp.cookies}')
            # update the 'sid' cookie in our config
            if 'sid' in resp.cookies:
                self.update_config(**{'cookies.sid': resp.cookies['sid']})
        if resp.status_code != 200:
            logger.warning(f'Error {resp.status_code} from Tumblr: {resp.text[:500]}')
            sys.exit()
        resp_path = join(self.data_dir, 'last_tumblr_response.html')
        try:
            shutil.copy2(resp_path, resp_path.replace('.html', '_prev.html'))
        except Exception:
            pass
        with open(resp_path, 'w') as f:
            f.write(resp.text)
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

    def make_api_req(self, endpoint: str, **kw):
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
        resp = make_request(url, headers=headers, cookies=self.cookies, min_delay=self.MIN_DELAY, **kw)
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



    def like_post(self, post_id: str, reblog_key: str) -> dict:
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
            obj = self.make_web_req('api/v2/user/like', method='POST', headers=headers, json=data)
        else: # TODO this requires Oauth access, not just api key
            endpoint = 'user/like'
            obj = self.make_api_req(endpoint, method='POST', json=data)
        print(f'liked post {post_id}: {obj}')
        return obj

    def get_dashboard(self) -> list[dict]:
        """Returns our "dashboard". Useful for updating the csrf"""
        obj = self.make_web_req('')
        print(J(obj)[:500])
        return obj['PeeprRoute']['initialDashboard']['posts']

    def get_blog_content(self, blog_name: str, n_posts: int=20) -> list[dict]:
        """Get blog content from the web interface.

        You can either fetch it from blog_name.tumblr.com, in which case you can use /page/N, but
        then it doesn't have the JSON object in the source code (hence you have to parse the html),
        or you can go to tumblr.com/blog_name, which does have the json, but then pagination works
        differently.

        """
        ret: list[dict] = []
        page = 1
        while len(ret) < n_posts:
            endpoint = f'{blog_name}/page/{page}'
            endpoint = f'https://{blog_name}.tumblr.com/page/{page}'
            endpoint = blog_name
            try:
                obj = self.make_web_req(endpoint)
            except ValueError:
                break
            posts = obj['PeeprRoute']['initialTimeline']['objects']
            posts = [p for p in posts if p['objectType'] == 'post']
            ret.extend(posts)
            break # because we're using the parseable version
        return ret

    def get_blog_archive(self, blog_name: str, n_posts: int = 20, offset: int=0) -> tuple[list[dict], int, int]:
        """Get blog archive via API.

        Note that some blogs don't allow access to the archive, so you must get it via the web
        interface.

        Returns (posts, offset, totalPosts)
        """
        posts: list[dict] = []
        total = 0
        next_link = dict(
            npf='true',
            reblog_info='true',
            context='archive',
            offset=str(offset),
        )
        while offset < n_posts:
            print(f'initial next link: {J(next_link)}')
            endpoint = f'blog/{blog_name}/posts'
            if next_link:
                endpoint = f'blog/{blog_name}/posts?{urlencode(next_link)}'
            obj = self.make_api_req(endpoint)
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
        return (posts, offset, total)

    def update_blogs(self):
        blogs = self.config['blogs']
        for i, name in enumerate(blogs):
            print(f'\nProcessing blog {i+1}/{len(blogs)}: {name}')
            try:
                posts, next_link, total = self.get_blog_archive(name)
                cols = self.create_collection_from_posts(posts, blog_name=name)
                print(f'Created {len(posts)} -> {len(cols)} post collections for {name}')
            except Exception as e:
                logger.warning(f'Failed to process blog {name}: {e}')
                continue
        self.update_embeddings(ids=[c.id for c in cols])

    def get_likes(self):
        """Returns our likes"""
        #https://www.tumblr.com/api/v2/user/likes?fields[blogs]=?advertiser_name,?avatar,?blog_view_url,?can_be_booped,?can_be_followed,?can_show_badges,?description_npf,?followed,?is_adult,?is_member,name,?primary,?theme,?title,?tumblrmart_accessories,url,?uuid&limit=21&reblog_info=true
        obj = self.make_api_req('https://www.tumblr.com/api/v2/user/likes')
        print(J(obj)[:500])

    @db_session
    def get_blog_user(self, blog_name: str, users_by_name: dict= {}, **kw) -> Entity:
        """Returns the blog user item for the given `blog_name`.

        Any kw are added to the metadata of the user item.
        """
        if blog_name not in users_by_name:
            u = Item.upsert(get_kw=dict(
                    source=self.NAME,
                    stype='blog',
                    otype='user',
                    url=f'https://{blog_name}.tumblr.com/'
                ),
                md=dict(
                    blog_name=blog_name,
                    **kw
                )
            )
            users_by_name[blog_name] = u
        return users_by_name[blog_name]

    @db_session
    def create_collection_from_posts(self, posts: list[dict], next_link=None, **kw) -> list[Entity]:
        """Creates `Item` rows from tumblr posts.

        This creates separate rows (with appropriate types) for each:
        - post
        - image/video/text/link content block within the post
        - for videos: another row for the poster image of that video

        Any additional `kw` are added to the metadata of each 'post' row.
        """
        print(f'creating collection with next_link: {next_link}')
        ret = []
        for post in posts:
            # get the user item
            blog_name = post['blog']['name']
            u = self.get_blog_user(
                    blog_name=blog_name,
                    title=post['blog'].get('title', ''),
                    description=post['blog'].get('description', ''),
                    uuid=post['blog'].get('uuid', ''),
            )
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
                        poster = c['poster'][0]
                        poster_url = poster['url'].replace('.pnj', '.png')
                        poster_media_key=poster.get('mediaKey', poster['url'].split('/')[3].rsplit('.', 1)[0]),
                        while isinstance(poster_media_key, (list, tuple)):
                            poster_media_key = poster_media_key[0]
                        md = dict(
                            w=media.get('width'),
                            h=media.get('height'),
                            media_key=media['url'].split('/')[3],
                            provider=c.get('provider', ''),
                            poster_url=poster_url,
                            poster_media_key=poster_media_key,
                        )
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
                # if it was a video, also add the poster as a separate image collection
                if content_type == 'video' and 'poster' in c:
                    poster_md = dict(
                        w=poster.get('width'),
                        h=poster.get('height'),
                        media_key=poster_media_key,
                        poster_for=cc.id, # type: ignore[attr-defined]
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

def simple_test(config_path: str, **kw):
    tumblr = Tumblr(config_path)
    name = tumblr.config['blogs'][0]
    while 1:
        try:
            posts, next_link, total = tumblr.get_blog_archive(name, 30)
        except Exception as e:
            print(f'got exception: {e}')
        for p in posts:
            print(p['id'])
        break
        tumblr.get_dashboard()
        tumblr.get_likes()
        #posts = tumblr.get_blog_content(name)
        print(f'{tumblr.csrf}: {len(posts)} posts: {J(posts)[:500]}...')
        sys.exit()
        sys.exit()
        break
        time.sleep(60 + random.random()*60)

@db_session
def update_blogs(config_path: str, **kw):
    tumblr = Tumblr(config_path)
    tumblr.update_blogs()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    cli_runner([simple_test, update_blogs],
               config_path=dict(default=DEFAULT_CONFIG_PATH, help='Path to the tumblr config json file'))
