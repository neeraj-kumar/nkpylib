"""A tumblr API that goes through the web.


def batch_extract_embeddings(inputs: list,
                             db_path: str,
                             embedding_type: str='text',
                             flag: str='c',
                             model: str|None=None,
                             batch_size=200,
                             md_func: Callable[[str, Any], dict]|None=None,
                             **kw) -> int:

"""


import json
import os
import sys
import time

from argparse import ArgumentParser
from os.path import exists, dirname, join, abspath
from urllib.parse import urlencode

import requests

from pony.orm import db_session, select
from pyquery import PyQuery as pq

from nkpylib.script_utils import cli_runner
from nkpylib.web_utils import make_request
from nkpylib.ml.nkcollections import Collection, init_sql_db

CONFIG = {}
IMAGES_DIR = 'cache/tumblr/'
DB_PATH = 'tumblr_collections.db'

J = lambda obj: json.dumps(obj, indent=2)

def tumblr_req(endpoint, **kw):
    url = endpoint if 'tumblr.com' in endpoint else f'https://www.tumblr.com/{endpoint}'
    resp = make_request(url, cookies=CONFIG['cookies'], **kw)
    with open('last_tumblr_response.html', 'w') as f:
        f.write(resp.text)
    print(f'  Resp cookies: {resp.cookies}')
    return resp

def tumblr_api_req(endpoint, **kw):
    headers = {
        "Authorization": f'Bearer {CONFIG["api_token"]}',
        **COMMON_HEADERS
    }
    url = f'https://api.tumblr.com/v2/{endpoint}'
    resp = make_request(url, headers=headers, **kw)
    try:
        obj = resp.json()
        #print(J(obj))
    except Exception as e:
        print(resp.text)
        raise Exception(f'Failed to fetch endpoint {endpoint}: {e}')
    with open('last_tumblr_api_response.json', 'w') as f:
        json.dump(obj, f, indent=2)
    return obj



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

def like_post(post_id, reblog_key, csrf=''):
    data = {
        'id': post_id,
        'reblog_key': reblog_key,
    }
    if csrf: # you need a valid csrf key from a page (doesn't matter which, as long as it's recent)
        headers = {
            "X-CSRF": csrf, # CSRF token that must be sent with the request
            # CORS‑related fetch metadata – keep them even though they are not used by `requests` directly.
            #"Alt-Used": "www.tumblr.com", # Tell the server we are coming from the same origin page
            #"Referrer": "https://www.tumblr.com/vsemily", # Referrer – the page that linked to the API call
            **COMMON_HEADERS
        }
        resp = tumblr_req('api/v2/user/like', method='POST', headers=headers, json=data)
        obj = resp.json()
    else: # TODO this requires Oauth access, not just api key
        endpoint = f'user/like'
        #print(endpoint)
        obj = tumblr_api_req(endpoint, method='POST', json=data)
    print(f'liked post {post_id}: {obj}')
    return obj

def get_blog_content(blog_name):
    resp = tumblr_req(blog_name)
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
    #print(json.dumps(obj, indent=2))
    posts = obj['PeeprRoute']['initialTimeline']['objects']
    posts = [p for p in posts if p['objectType'] == 'post']
    return csrf, posts

def get_blog_archive(blog_name, n_posts=20):
    """The archive is accessible via API, using the following curl command:

    curl 'https://api.tumblr.com/v2/blog/vsemily/posts?fields%5Bblogs%5D=%3Fadvertiser_name%2C%3Favatar%2C%3Fblog_view_url%2C%3Fcan_be_booped%2C%3Fcan_be_followed%2C%3Fcan_show_badges%2C%3Fdescription_npf%2C%3Ffollowed%2C%3Fis_adult%2C%3Fis_member%2Cname%2C%3Fprimary%2C%3Ftheme%2C%3Ftitle%2C%3Ftumblrmart_accessories%2Curl%2C%3Fuuid%2C%3Fask%2C%3Fcan_submit%2C%3Fcan_subscribe%2C%3Fis_blocked_from_primary%2C%3Fis_blogless_advertiser%2C%3Fis_password_protected%2C%3Fshare_following%2C%3Fshare_likes%2C%3Fsubscribed%2C%3Fupdated%2C%3Ffirst_post_timestamp%2C%3Fposts%2C%3Fdescription%2C%3Ftop_tags_all&npf=true&reblog_info=true&context=archive'
    """
    query_fields = dict(
        advertiser_name='?advertiser_name',
        avatar='?avatar',
        blog_view_url='?blog_view_url',
        can_be_booped='?can_be_booped',
        can_be_followed='?can_be_followed',
        can_show_badges='?can_show_badges',
        description_npf='?description_npf',
        followed='?followed',
        is_adult='?is_adult',
        is_member='?is_member',
        name='name',
        primary='?primary',
        theme='?theme',
        title='?title',
        tumblrmart_accessories='tumblrmart_accessories',
        url='url',
        uuid='?uuid',
        ask='?ask',
        can_submit='?can_submit',
        can_subscribe='?can_subscribe',
        is_blocked_from_primary='?is_blocked_from_primary',
        is_blogless_advertiser='?is_blogless_advertiser',
        is_password_protected='?is_password_protected',
        share_following='?share_following',
        share_likes='?share_likes',
        subscribed='?subscribed',
        updated='?updated',
        first_post_timestamp='?first_post_timestamp',
        posts='?posts',
        description='?description',
        top_tags_all='?top_tags_all',
    )
    offset = 0
    posts = []
    total = 0
    while offset < n_posts:
        params = {
            #'fields[blogs]': ','.join(query_fields.values()),
            'npf': 'true',
            'reblog_info': 'true',
            'context': 'archive',
            'offset': f'{offset}',
        }
        # url encode the params into the endpoint
        endpoint = f'blog/{blog_name}/posts?{urlencode(params)}'
        #print(endpoint)
        obj = tumblr_api_req(endpoint)
        if obj['meta']['status'] != 200:
            raise Exception(f'Failed to fetch blog archive: {obj["meta"]}')
        batch = obj['response']['posts'] or []
        posts.extend(batch)
        total = obj['response']['totalPosts']
        offset += 20
        if not batch or not obj.get('links', []):
            break
    return (posts, total)

def get_post_images(p, max_images=1, max_width=400, dir=IMAGES_DIR):
    """Downloads the post image (if we don't already have it) to `dir`.

    If the post contains multiple images, we get the first `max_images` of them.
    For each image, we select the best quality version that is less than or equal to `max_width`
    pixels wide.

    Image content blocks look like this:
    {
        "type": "image",
        "media": [
            {
                "url": "https://64.media.tumblr.com/...",
                "mediaKey": "abc123...",
                "width": 1280,
                "height": 720,
                ...
            },
            ...
        ],
        ...
    }

    Images might end with .pnj, which is a page rather than the image itself, so we should replace
    that with .png

    Also note that you have to set your accept header to images only, otherwise it will get an html
    wrapper around the image.
    """
    images = []
    content = p['content'] or p['trail'][0]['content']
    for c in content:
        if c['type'] not in ('image', 'video'):
            return
        media = [c['media']] if isinstance(c['media'], dict) else c['media']
        #TODO get filmstrip and posters for videos?
        for m in media:
            if 'mediaKey' in m:
                mk = m['mediaKey']
            else:
                mk = m['url'].split('/')[3].rsplit('.', 1)[0]
            if m['width'] <= max_width or c['type'] == 'video':
                img_url = m['url'].replace('.pnj', '.png')
                ext = img_url.split('.')[-1]
                img_path = join(dir, f'{mk}.{ext}')
                if not exists(img_path):
                    print(f'downloading image {img_url} -> {img_path}')
                    r = make_request(img_url, headers={'Accept': 'image/avif,image/webp,image/apng,image/*,video/*'})
                    try:
                        os.makedirs(dirname(img_path), exist_ok=True)
                    except Exception as e:
                        pass
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
                images.append(img_path)
                break
        if len(images) >= max_images:
            return images
    return images

@db_session
def create_collection_from_posts(blog_name: str, posts: list[dict]) -> None:
    """Creates Collection rows from tumblr posts"""
    
    for post in posts:
        # Create the main post collection entry
        post_collection = Collection(
            source='tumblr',
            stype='blog', 
            otype='post',
            url=f"https://{blog_name}.tumblr.com/post/{post['id']}",
            ts=post.get('timestamp', int(time.time())),
            md=dict(
                post_id=post['id'],
                reblog_key=post['reblogKey'],
                tags=post.get('tags', []),
                blog_name=blog_name,
                note_count=post.get('noteCount', 0),
                summary=post.get('summary', ''),
            )
        )
        
        # Get content from either direct content or trail
        content = post['content'] or (post['trail'][0]['content'] if post.get('trail') else [])
        
        # Create child collections for each content block
        for content_block in content:
            content_type = content_block['type']
            
            # Determine URL and metadata based on content type using match statement
            match content_type:
                case 'image':
                    media = content_block['media'][0] if isinstance(content_block['media'], list) else content_block['media']
                    url = media['url'].replace('.pnj', '.png')
                    md = dict(
                        width=media.get('width'),
                        height=media.get('height'),
                        media_key=media.get('mediaKey', media['url'].split('/')[3].rsplit('.', 1)[0])
                    )
                case 'video':
                    media = content_block['media']
                    url = media['url']
                    md = dict(
                        width=media.get('width'),
                        height=media.get('height'),
                        media_key=media['url'].split('/')[3]
                    )
                case 'text':
                    url = f"https://{blog_name}.tumblr.com/post/{post['id']}#text"
                    md = dict(
                        text=content_block.get('text', ''),
                        subtype=content_block.get('subtype', 'paragraph')
                    )
                case 'link':
                    url = content_block.get('url', content_block.get('displayUrl', ''))
                    md = dict(
                        display_url=content_block.get('displayUrl', ''),
                        title=content_block.get('title', ''),
                        description=content_block.get('description', '')
                    )
                case _:
                    # For other content types
                    url = f"https://{blog_name}.tumblr.com/post/{post['id']}#{content_type}"
                    md = content_block
            
            # Create child collection
            Collection(
                source='tumblr',
                stype='blog',
                otype=content_type,
                url=url,
                ts=post.get('timestamp', int(time.time())),
                md=md,
                parent=post_collection
            )

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
        if i <= 200:
            imgs = get_post_images(p, max_images=2)
            #print(f'  DLed: {imgs}')

def simple_test(**kw):
    csrf = ''
    csrf, posts = get_blog_content('virgomoon')
    #posts, total = get_blog_archive('animentality')
    #posts, total = get_blog_archive('maureen2musings')
    process_posts(posts)

@db_session
def update_blogs(**kw):
    global CONFIG
    blogs = CONFIG['blogs']
    for i, name in enumerate(blogs):
        print(f'\nProcessing blog {i}/{len(blogs)}: {name}')
        posts, total = get_blog_archive(name)
        create_collection_from_posts(name, posts)
        print(f'Created {len(posts)} post collections for {name}')
        process_posts(posts)
        #break

def extract_md(**kw):
    pass

if __name__ == '__main__':
    def parse_config(config, **kw):
        global CONFIG
        with open(config, 'r') as f:
            CONFIG = json.load(f)
        # Initialize the database
        init_sql_db(DB_PATH)

    cli_runner([simple_test, update_blogs, extract_md],
               pre_func=parse_config,
               config=dict(default='.tumblr_config.json', help='Path to the tumblr config json file'))
    if 0:
        parser = ArgumentParser(description='Tumblr API Wrapper (web-based)')
        parser.add_argument('-c', '--config', default='.tumblr_config.json', help='Path to the tumblr config json file')
        args = parser.parse_args()
        with open(args.config, 'r') as f:
            CONFIG = json.load(f)
