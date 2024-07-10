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

import os

from functools import partial

import requests

from .cacheutils import APICache
wmcache = partial(APICache, cachedir='cache/wikimapia/%(fn)s/', mindelay=0.5, serializer='json')

def wikimapia(function, **kw):
    """Low-level function to call the wikimapia API with given `function` name.

    You should use one of the higher-level functions instead of this one directly.
    """
    wm_key = os.getenv('WIKIMAPIA_API_KEY')
    if not wm_key:
        raise ValueError('WIKIMAPIA_API_KEY not set')
    resp = requests.get(f'http://api.wikimapia.org/?function={function}&key={wm}&format=json&language=en&{urllib.parse.urlencode(kw)}')
    try:
        return resp.json()
    except Exception:
        return resp.text


