"""A tumblr API that goes through the web"""

import json

from argparse import ArgumentParser

import requests

from pyquery import PyQuery as pq

from nkpylib.web_utils import make_request

COOKIES = {}

J = lambda obj: json.dumps(obj, indent=2)

def tumblr_req(endpoint, **kw):
    resp = make_request(f'https://www.tumblr.com/{endpoint}', cookies=COOKIES, **kw)
    print(f'  Resp cookies: {resp.cookies}')
    return resp

def like_post(post_id, reblog_key):
    data = {
        'id': post_id,
        'reblog_key': reblog_key,
    }
    req = tumblr_req('api/v2/user/like', method='POST', json=data)
    obj = req.json()
    print(f'liked post {post_id}: {obj}')
    return obj


def get_blog_content(blog_name):
    req = tumblr_req(blog_name)
    if req.status_code != 200:
        raise Exception(f'Failed to fetch blog content: {req.status_code}')
    doc = pq(req.text)
    # find the script tag with the initial state
    state = doc('script#___INITIAL_STATE___').text()
    try:
        obj = json.loads(state)
    except Exception as e:
        print(req.text[:100])
        raise Exception(f'Failed to parse initial state JSON: {e}')
    #print(json.dumps(obj, indent=2))
    posts = obj['PeeprRoute']['initialTimeline']['objects']
    posts = [p for p in posts if p['objectType'] == 'post']
    return posts


if __name__ == '__main__':
    parser = ArgumentParser(description='Tumblr API Wrapper (web-based)')
    parser.add_argument('-c', '--config', default='.tumblr_config.json', help='Path to the tumblr config json file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        COOKIES = json.load(f)
    posts = get_blog_content('virgomoon')
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
            like_post(p['id'], p['reblogKey'])
