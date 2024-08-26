"""Utilities for interacting with the wikimapia API.

You should have the environment variable `WIKIMAPIA_API_KEY` set to your API key.

This is a reverse geocoding API that can be used to get information about locations. The main
functions are:

- nearby_places: get places near a given latitude and longitude
- place_info: get information about a specific place
- place_search: search for places by name
- all_categories: get all categories of places

To deal with pagination
"""

import json
import os
import urllib

from functools import partial
from typing import Any

import requests

from tqdm import tqdm

from nkpylib.cacheutils import APICache
wmcache = partial(APICache, cachedir='cache/wikimapia/%(fn)s/', mindelay=5, serializer='json')

@wmcache
def wikimapia(function, **kw):
    """Low-level function to call the wikimapia API with given `function` name.

    You should use one of the higher-level functions instead of this one directly.
    """
    wm_key = os.getenv('WIKIMAPIA_API_KEY')
    if not wm_key:
        raise ValueError('WIKIMAPIA_API_KEY not set')
    resp = requests.get(f'http://api.wikimapia.org/?function={function}&key={wm_key}&format=json&language=en&{urllib.parse.urlencode(kw)}')
    try:
        ret = resp.json()
    except Exception:
        return resp.text
    if 'debug' in ret:
        raise ValueError(f'Wikimapia API Error: {ret}')
    return ret

def wikimapia_api_gen(**kw) -> Any:
    """A generator for Wikimapia API calls.

    Useful for pagination. This will yield each page of results, one at a time.
    If you just want the first one, then use `wikimapia` directly, or call this with
    `next(wikimapia_api_gen(...))`.
    """
    page = 1
    while True:
        resp = wikimapia(page=page, **kw)
        yield resp
        if resp['found'] < page * resp['count']:
            break
        page += 1

def all_categories():
    """Returns all categories of places."""
    categories = {}
    for page in tqdm(wikimapia_api_gen(function='category.getall')):
        try:
            cats = page['categories']
        except Exception as e:
            print(f'Error: {e}')
            print(f'Page: {page}')
            raise
        if not cats:
            break
        for cat in cats:
            categories[cat['id']] = cat
    return categories

def nearby_places(lat, lon, **kw):
    """Returns places near the given latitude and longitude."""
    return wikimapia(function='place.getnearest', lat=lat, lon=lon, **kw)['places']

def place_info(place_id, **kw):
    """Returns information about a specific place."""
    return wikimapia(function='place.getbyid', id=place_id, **kw)

if __name__ == '__main__':
    lat, lon = 40.78, -73.98
    places = nearby_places(lat, lon)
    j = lambda obj: json.dumps(obj, indent=2)
    print(f'Got {len(places)} places near {lat}, {lon}: {j(places[:5])}')
    place = place_info(places[0]['id'])
    print(f'Got place info for {places[0]["id"]}:\n{j(place)}')
    cats = all_categories()
    print(f'Got {len(cats)} categories: {list(cats.items())[:5]}')
