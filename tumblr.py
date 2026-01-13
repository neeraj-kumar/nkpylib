"""A tumblr API that goes through the web.


"""
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
from pyquery import PyQuery as pq

from nkpylib.script_utils import cli_runner
from nkpylib.web_utils import make_request
from nkpylib.ml.nkcollections import Collection, init_sql_db

logger = logging.getLogger(__name__)

CONFIG = {}
IMAGES_DIR = 'db/tumblr/images/'
SQLITE_PATH = 'db/tumblr/tumblr_collection.sqlite'
LMDB_PATH = 'db/tumblr/tumblr_embeddings.lmdb'

J = lambda obj: json.dumps(obj, indent=2)

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

    def tumblr_req(self, endpoint: str, **kw):
        """Make a request to the Tumblr web interface"""
        url = endpoint if 'tumblr.com' in endpoint else f'https://www.tumblr.com/{endpoint}'
        headers = {
            "Authorization": f'Bearer {self.api_token}',
            **self.COMMON_HEADERS
        }
        resp = make_request(url, cookies=self.cookies, headers=headers, **kw)
        print(f'  Resp cookies: {resp.cookies}')
        if resp.status_code != 200:
            logger.warning(f'Error {resp.status_code} from Tumblr: {resp.text}')
            sys.exit()
        try:
            shutil.copy2('last_tumblr_response.html', 'last_tumblr_response_prev.html')
        except Exception:
            pass
        with open('last_tumblr_response.html', 'w') as f:
            f.write(resp.text)
        return resp

    def tumblr_api_req(self, endpoint: str, **kw):
        """Make a request to the Tumblr API"""
        headers = {
            "Authorization": f'Bearer {self.api_token}',
            **self.COMMON_HEADERS
        }
        url = f'https://api.tumblr.com/v2/{endpoint}'
        resp = make_request(url, headers=headers, **kw)
        try:
            obj = resp.json()
        except Exception as e:
            print(resp.text)
            raise Exception(f'Failed to fetch endpoint {endpoint}: {e}')
        with open('last_tumblr_api_response.json', 'w') as f:
            json.dump(obj, f, indent=2)
        return obj

    def like_post(self, post_id: str, reblog_key: str, csrf: str = '') -> dict:
        """Like a post by ID and reblog key"""
        data = dict(
            id=post_id,
            reblog_key=reblog_key,
        )
        if csrf: # you need a valid csrf key from a page (doesn't matter which, as long as it's recent)
            headers = {
                "X-CSRF": csrf, # CSRF token that must be sent with the request
                **self.COMMON_HEADERS
            }
            resp = self.tumblr_req('api/v2/user/like', method='POST', headers=headers, json=data)
            obj = resp.json()
        else: # TODO this requires Oauth access, not just api key
            endpoint = 'user/like'
            obj = self.tumblr_api_req(endpoint, method='POST', json=data)
        print(f'liked post {post_id}: {obj}')
        return obj

    def get_blog_content(self, blog_name: str) -> tuple[str, list[dict]]:
        """Get blog content from the web interface"""
        resp = self.tumblr_req(blog_name)
        if resp.status_code != 200:
            raise Exception(f'Failed to fetch blog content: {resp.status_code}')
        doc = pq(resp.text)
        # find the script tag with the initial state
        state = doc('script#___INITIAL_STATE___').text()
        try:
            obj = json.loads(state)
        except Exception as e:
            print(resp.text[:100])
            raise Exception(f'Failed to parse initial state JSON: {e}')
        csrf = obj["csrfToken"]
        posts = obj['PeeprRoute']['initialTimeline']['objects']
        posts = [p for p in posts if p['objectType'] == 'post']
        return csrf, posts

    def get_blog_archive(self, blog_name: str, n_posts: int = 20) -> tuple[list[dict], int]:
        """Get blog archive via API
        
        The archive is accessible via API, using the following curl command:
        curl 'https://api.tumblr.com/v2/blog/vsemily/posts?fields%5Bblogs%5D=%3Fadvertiser_name%2C%3Favatar%2C%3Fblog_view_url%2C%3Fcan_be_booped%2C%3Fcan_be_followed%2C%3Fcan_show_badges%2C%3Fdescription_npf%2C%3Ffollowed%2C%3Fis_adult%2C%3Fis_member%2Cname%2C%3Fprimary%2C%3Ftheme%2C%3Ftitle%2C%3Ftumblrmart_accessories%2Curl%2C%3Fuuid%2C%3Fask%2C%3Fcan_submit%2C%3Fcan_subscribe%2C%3Fis_blocked_from_primary%2C%3Fis_blogless_advertiser%2C%3Fis_password_protected%2C%3Fshare_following%2C%3Fshare_likes%2C%3Fsubscribed%2C%3Fupdated%2C%3Ffirst_post_timestamp%2C%3Fposts%2C%3Fdescription%2C%3Ftop_tags_all&npf=true&reblog_info=true&context=archive'
        """
        offset = 0
        posts = []
        total = 0
        while offset < n_posts:
            params = dict(
                npf='true',
                reblog_info='true',
                context='archive',
                offset=str(offset),
            )
            endpoint = f'blog/{blog_name}/posts?{urlencode(params)}'
            obj = self.tumblr_api_req(endpoint)
            if obj['meta']['status'] != 200:
                raise Exception(f'Failed to fetch blog archive: {obj["meta"]}')
            batch = obj['response']['posts'] or []
            posts.extend(batch)
            total = obj['response']['totalPosts']
            offset += 20
            if not batch or not obj.get('links', []):
                break
        return (posts, total)





def like_post(post_id, reblog_key, csrf=''):
    """Legacy function - use Tumblr.like_post instead"""
    tumblr = Tumblr(CONFIG)
    return tumblr.like_post(post_id, reblog_key, csrf)

def get_blog_content(blog_name):
    """Legacy function - use Tumblr.get_blog_content instead"""
    tumblr = Tumblr(CONFIG)
    return tumblr.get_blog_content(blog_name)

def get_blog_archive(blog_name, n_posts=20):
    """Legacy function - use Tumblr.get_blog_archive instead"""
    tumblr = Tumblr(CONFIG)
    return tumblr.get_blog_archive(blog_name, n_posts)

@db_session
def create_collection_from_posts(blog_name: str, posts: list[dict]) -> list[Collection]:
    """Creates `Collection` rows from tumblr posts"""
    #def upsert(cls, get_kw: dict[str, Any], **set_kw: Any) -> Entity:
    ret = []
    for post in posts:
        # Create the main post collection entry
        pc = Collection.upsert(get_kw=dict(
                source='tumblr',
                stype='blog',
                otype='post',
                url=post['postUrl']
            ),
            ts=post.get('timestamp', int(time.time())),
            md=dict(
                post_id=post['id'],
                reblog_key=post['reblogKey'],
                tags=post.get('tags', []),
                blog_name=blog_name,
                n_notes=post.get('noteCount', 0),
                n_likes=post.get('likeCount', 0),
                n_reblogs=post.get('reblogCount', 0),
                summary=post.get('summary', ''),
                original_type=post.get('originalType', ''),
                reblogged_from=post.get('rebloggedFromUrl', ''),
            )
        )
        ret.append(pc)
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
                    md = dict(
                        w=media.get('width'),
                        h=media.get('height'),
                        media_key=media['url'].split('/')[3],
                        provider=c.get('provider', ''),
                        poster_url=poster_url,
                        poster_media_key=poster_media_key,
                    )
                case 'text':
                    url = f"{pc.url}#text_{i}"
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
                    url = f"{pc.url}#{content_type}_{i}"
                    md = c
            # Create child collection object
            cc = Collection.upsert(get_kw=dict(
                    source=pc.source,
                    stype=pc.stype,
                    otype=content_type,
                    url=url,
                ),
                ts=post.get('timestamp', int(time.time())),
                md=md,
                parent=pc
            )
            ret.append(cc)
            # if it was a video, also add the poster as a separate image collection
            if content_type == 'video' and 'poster' in c:
                poster_md = dict(
                    w=poster.get('width'),
                    h=poster.get('height'),
                    media_key=poster_media_key,
                    poster_for=cc.id,
                )
                pcc = Collection.upsert(get_kw=dict(
                        source=pc.source,
                        stype=pc.stype,
                        otype='image',
                        url=poster_url,
                    ),
                    ts=post.get('timestamp', int(time.time())),
                    md=poster_md,
                    parent=pc,
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
            like_post(post_id=p['id'], reblog_key=p['reblogKey'], csrf=csrf)

def simple_test(**kw):
    tumblr = Tumblr(CONFIG)
    csrf = ''
    while 1:
        csrf, posts = tumblr.get_blog_content('virgomoon')
        print(f'{csrf}: {len(posts)} posts: {json.dumps(posts, indent=2)[:500]}...')
        break
        time.sleep(60 + random.random()*60)

@db_session
def update_blogs(**kw):
    global CONFIG
    tumblr = Tumblr(CONFIG)
    blogs = CONFIG['blogs']
    for i, name in enumerate(blogs):
        print(f'\nProcessing blog {i+1}/{len(blogs)}: {name}')
        try:
            posts, total = tumblr.get_blog_archive(name)
            cols = create_collection_from_posts(name, posts)
            print(f'Created {len(posts)} -> {len(cols)} post collections for {name}')
        except Exception as e:
            logger.warning(f'Failed to process blog {name}: {e}')
            continue
    Collection.update_embeddings(lmdb_path=LMDB_PATH, images_dir=IMAGES_DIR, use_cache=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    def parse_config(config, **kw):
        global CONFIG
        with open(config, 'r') as f:
            CONFIG = json.load(f)
        init_sql_db(SQLITE_PATH)

    cli_runner([simple_test, update_blogs],
               pre_func=parse_config,
               config=dict(default='.tumblr_config.json', help='Path to the tumblr config json file'))
    if 0:
        parser = ArgumentParser(description='Tumblr API Wrapper (web-based)')
        parser.add_argument('-c', '--config', default='.tumblr_config.json', help='Path to the tumblr config json file')
        args = parser.parse_args()
        with open(args.config, 'r') as f:
            CONFIG = json.load(f)
