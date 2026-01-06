"""A tumblr API that goes through the web"""

import json
import sys

from argparse import ArgumentParser
from urllib.parse import urlencode

import requests

from pyquery import PyQuery as pq

from nkpylib.web_utils import make_request

CONFIG = {}

J = lambda obj: json.dumps(obj, indent=2)

def tumblr_req(endpoint, **kw):
    url = endpoint if 'tumblr.com' in endpoint else f'https://www.tumblr.com/{endpoint}'
    resp = make_request(url, cookies=CONFIG['cookies'], **kw)
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

def like_post(post_id, reblog_key, csrf):
    data = {
        'id': post_id,
        'reblog_key': reblog_key,
    }
    headers = {
        "X-CSRF": csrf, # CSRF token that must be sent with the request
        # CORS‑related fetch metadata – keep them even though they are not used by `requests` directly.
        #"Alt-Used": "www.tumblr.com", # Tell the server we are coming from the same origin page
        #"Referrer": "https://www.tumblr.com/vsemily", # Referrer – the page that linked to the API call
        **COMMON_HEADERS
    }
    resp = tumblr_req('api/v2/user/like', method='POST', headers=headers, json=data)
    obj = resp.json()
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

def get_blog_archive(blog_name):
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
    params = {
        #'fields[blogs]': ','.join(query_fields.values()),
        'npf': 'true',
        'reblog_info': 'true',
        'context': 'archive',
        #'offset': '50',
    }
    # url encode the params into the endpoint
    endpoint = f'blog/{blog_name}/posts?{urlencode(params)}'
    #print(endpoint)
    obj = tumblr_api_req(endpoint)
    print(J(obj))
    sys.exit()



if __name__ == '__main__':
    parser = ArgumentParser(description='Tumblr API Wrapper (web-based)')
    parser.add_argument('-c', '--config', default='.tumblr_config.json', help='Path to the tumblr config json file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        CONFIG = json.load(f)
    #csrf, posts = get_blog_content('virgomoon')
    csrf, posts = get_blog_archive('vsemily')
    for i, p in enumerate(posts):
        content = p['content'] or p['trail'][0]['content']
        print(f'\nPost {i}: {p["id"]} (reblog key: {p["reblogKey"]})')
        for c in content:
            match c['type']:
                case 'image':
                    m = c['media'][0]
                    if 'mediaKey' in m:
                        mk = m['mediaKey']
                    else:
                        mk = m['url'].split('/')[3]
                    print(f'  image: {mk}')
                case 'text':
                    print(f'  text: {c["text"]}')
                case _:
                    print(f'  unknown content type: {c["type"]}')
        if i == 3:
            print('Liking this post...')
            like_post(post_id=p['id'], reblog_key=p['reblogKey'], csrf=csrf)
