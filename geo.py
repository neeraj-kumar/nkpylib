"""Utilities for interacting with the wikimapia API.

You should have the environment variable `WIKIMAPIA_API_KEY` set to your API key.

This is a reverse geocoding API that can be used to get information about locations. The main
functions are:

- nearby_places: get places near a given latitude and longitude
- place_info: get information about a specific place
- place_search: search for places by name
- categories: get all categories of places

To deal with pagination
"""
#TODO normalized place info access, such as id, address, st, city, state, zip, country, etc.
#TODO multithreading
#TODO refactor common api stuff
#TODO better dealing with cache ignore and cache fmt, etc
#TODO global lock on api

import json
import logging
import os
import time
import urllib

from abc import ABC, abstractmethod
from functools import partial
from os.path import dirname, basename
from typing import Any, Iterator, Optional

import requests

from tqdm import tqdm

from nkpylib.cacheutils import APICache
from nkpylib.utils import haversinedist

wm_cache = partial(APICache, cachedir='cache/wikimapia/%(fn)s/', mindelay=5, serializer='json')
ga_cache = partial(APICache, cachedir='cache/geoapify/%(fn)s/', mindelay=5, serializer='json')

logger = logging.getLogger(__name__)

class CacheException(Exception):
    pass

@wm_cache
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


class Geocoder(ABC):
    """Abstract base class for geocoding and place-info APIs."""
    def __init__(self,
                 api_key: str='',
                 api_key_env: str='',
                 api_delay: float=1.0,
                 cache_dir: str=''):
        """Initialize the geocoder with an API key.

        If `api_key` is not provided, then it will be read from the environment variable
        `api_key_env`.
        """
        if not api_key:
            assert api_key_env, 'Either api_key or api_key_env must be provided'
            api_key = os.getenv(api_key_env)
        self.api_key = api_key
        self.last_call = 0
        self.api_delay = api_delay
        self.cache_dir = cache_dir
        self.cache_fmt = f'cache/{self.__class__.__name__}/%(fn)s/%(kw)s.json'
        self.cache_ignore_args = ['api_key']

    def get_id(self, obj: dict[str, Any]) -> Any:
        """Returns the place_id from the given object returned from the API."""
        raise NotImplementedError

    @abstractmethod
    def _api_call(self, endpoint: str, **kw: Any) -> dict[str, Any]:
        """Implementation of Low-level function to call the API at given `endpoint` and `kw`"""
        pass

    def _cache_path(self, endpoint: str, **kw: Any) -> str:
        """Returns the cache file path for the given `endpoint` and `kw`."""
        kw = {k: v for k, v in kw.items() if k not in self.cache_ignore_args}
        return os.path.join(self.cache_dir, self.cache_fmt % {'fn': endpoint, 'kw': urllib.parse.urlencode(kw)})

    def _get_cache(self, endpoint: str, **kw: Any) -> dict[str, Any]:
        """Gets the cached response for the given `endpoint` and `kw`."""
        try:
            with open(self._cache_path(endpoint, **kw)) as f:
                return json.load(f)
        except Exception as e:
            raise CacheException(f'Cache error for {endpoint} with {kw}: {e}')

    def _set_cache(self, obj, endpoint: str, **kw: Any) -> Any:
        """Sets the cache for the given `endpoint` and `kw`. Returns the object."""
        try:
            os.makedirs(dirname(self._cache_path(endpoint, **kw)), exist_ok=True)
        except Exception as e:
            pass
        with open(self._cache_path(endpoint, **kw), 'w') as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        return obj

    def api_call(self, endpoint: str, **kw: Any) -> dict[str, Any]:
        """Low-level function to call the API at given `endpoint` and `kw`"""
        try:
            return self._get_cache(endpoint, **kw)
        except CacheException:
            if time.time() - self.last_call < self.api_delay:
                time.sleep(self.api_delay)
            self.last_call = time.time()
            ret = self._api_call(endpoint, **kw)
            return self._set_cache(ret, endpoint, **kw)

    def api_gen(self, **kw) -> Iterator[dict[str, Any]]:
        """Generator for pagination of API calls."""
        raise NotImplementedError

    def categories(self, obj: Optional[Any]=None, type: str='all', **kw: Any) -> dict[str, Any]:
        """Returns categories of places.

        If `obj` is None, then possible categories. Otherwise, it returns categories of the given
        `obj`. The `type` can be set to return a specific subset of category-types, which is
        Geocoder-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def nearby_places(self, lat: float, lon: float, **kw: Any) -> list[dict[str, Any]]:
        """Returns places near the given latitude and longitude."""
        pass

    @abstractmethod
    def place_info(self, place_id: Any, **kw: Any) -> dict[str, Any]:
        """Returns information about a specific place."""
        pass

    def test(self, lat: float, lon: float) -> None:
        """Tests out the geocoder by getting nearby places and place info."""
        print(f'Testing {self.__class__.__name__} API with lat={lat}, lon={lon}')
        places = self.nearby_places(lat, lon)
        j = lambda obj: json.dumps(obj, indent=2, sort_keys=True)
        print(f'Got {len(places)} places near {lat}, {lon}: {j(places[:5])}')
        place_id = self.get_id(places[0])
        place = self.place_info(place_id)
        print(f'Got place info for {place_id}:\n{j(place)}')


class Wikimapia(Geocoder):
    """Notes:
    - Wikimapia doesn't seem maintained anymore, and the API is not very reliable.
    - There are very small limits per few mins, and per day
    - For some reason, many things in NYC get tagged with state = NJ!
    """
    def __init__(self):
        super().__init__(api_key_env='WIKIMAPIA_API_KEY')

    def _api_call(self, endpoint: str, **kw: Any) -> dict[str, Any]:
        return wikimapia(endpoint, **kw)

    def api_gen(self, **kw) -> Iterator[dict[str, Any]]:
        """A generator for Wikimapia API calls.

        Useful for pagination. This will yield each page of results, one at a time.
        If you just want the first one, then use `wikimapia` directly, or call this with
        `next(api_gen(...))`.
        """
        page = 1
        while True:
            resp = self.api_call(page=page, **kw)
            yield resp
            if resp['found'] < page * resp['count']:
                break
            page += 1

    def categories(self, obj: Optional[Any]=None, **kw: Any) -> dict[str, Any]:
        """Returns categories of places.

        We ignore the 'type' constraints, as Wikimapia doesn't have that concept.
        """
        if obj is not None:
            return [t['title'] for t in obj.get('tags', [])]
        categories = {}
        for page in tqdm(self.api_gen(function='category.getall')):
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

    def nearby_places(self, lat: float, lon:float, **kw: Any) -> list[dict[str, Any]]:
        """Returns places near the given latitude and longitude."""
        return self.api_call(function='place.getnearest', lat=lat, lon=lon, **kw)['places']

    def place_info(self, place_id: Any, **kw: Any) -> dict[str, Any]:
        """Returns information about a specific place."""
        return self.api_call(function='place.getbyid', id=place_id, **kw)

    def get_id(self, obj: dict[str, Any]) -> Any:
        return obj['id']


class Geoapify(Geocoder):
    """Notes:
    - Pricing is in credits (1 req = 1 credit generally)
      - Free tier: 3k credits/day, <5qps
      - API 10: 10k credits/day for $59/month, < 12qps
      - API 25: 25k credits/day for $109/month, < 15qps
    """
    def __init__(self):
        super().__init__(api_key_env='GEOAPIFY_API_KEY', api_delay=0.3)
        self.cache_ignore_args += ['categories', 'limit']

    def _api_call(self, endpoint: str, **kw: Any) -> dict[str, Any]:
        """Low-level function to call the Geoapify API with given `endpoint` name."""
        url = f'https://api.geoapify.com/v2/{endpoint}?apiKey={self.api_key}&{urllib.parse.urlencode(kw)}'
        logging.debug(f'Calling Geoapify API with url: {url}')
        resp = requests.get(url)
        return resp.json()

    def get_id(self, obj: dict[str, Any]) -> Any:
        """Returns the place_id from the given object returned from the API."""
        return obj['properties']['place_id']

    def categories(self, obj: Optional[Any]=None, type: str='common', **kw: Any) -> dict[str, Any]:
        """Returns categories of places.

        If `obj` is given, then returns the categories of the given object. Otherwise, returns all
        possible categories.

        The `type` can be one of:
        - 'all' - all categories, including obscure, top-level, lower-level, etc.
        - 'common' [default] - all top-level categories that are common
        - 'specific' - the most specific version of categories (this is only for specific objects)
        """
        common_cats = [
            'accommodation',
            'activity',
            'adult',
            'airport',
            'beach',
            'building',
            'camping',
            'catering',
            'childcare',
            'commercial',
            'education',
            'entertainment',
            'healthcare',
            'heritage',
            'leisure',
            'man_made',
            'national_park',
            'natural',
            'office',
            'pet',
            'populated_place',
            'production',
            'public_transport',
            'religion',
            'rental',
            'service',
            'sport',
            'tourism',
        ]
        rare_cats = [
            'administrative',
            'amenity',
            'highway',
            'low_emission_zone',
            'parking',
            'political',
            'postal_code',
            'power',
            'railway',
            'ski',
        ]
        if obj is not None:
            ocats = obj.get('properties', {}).get('categories', [])
            if type == 'all':
                return ocats
            elif type == 'common':
                return [c for c in ocats if c['key'] in common_cats]
            elif type == 'specific':
                # remove ocats which are prefixes of other ones
                filtered = []
                for c1 in ocats:
                    if not any(c2.startswith(c1+'.') for c2 in ocats if c1 != c2):
                        filtered.append(c1)
                return filtered
            else:
                raise NotImplementedError(f'Unknown type for obj categories: {type}')
        # get all categories
        if type == 'all':
            return common_cats + rare_cats
        elif type == 'common':
            return common_cats[:]

    def nearby_places(self, lat: float, lon: float, radius: float=1000, limit:int=20, include_extra_cats: bool=False, **kw: Any) -> list[dict[str, Any]]:
        """Returns places near the given `lat` and `lon`, within `radius`.

        Note that this API charges for number of places returned: 1 credit per 20 places.
        """
        cats = self.categories(type='common')
        ret = self.api_call('places',
                            categories=','.join(cats),
                            filter=f'circle:{lon},{lat},{radius}',
                            limit=limit,
                            **kw)
        return ret['features']

    def place_info(self, place_id: Any, **kw: Any) -> dict[str, Any]:
        """Returns information about a specific place.

        Note that the info returned in `nearby_places()` has all the info here except that this has
        a detailed polygon for the place, as opposed to just a point from `nearby_places()`
        """
        return self.api_call(f'place-details', id=place_id, **kw)



if __name__ == '__main__':
    # enable logging with a verbose tab-separated format including ts, func, line no, level, msg
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(funcName)s\t%(lineno)d\t%(levelname)s\t%(message)s')
    lat, lon = 40.78, -73.98
    geocoder = Geoapify()
    geocoder.test(lat, lon)
