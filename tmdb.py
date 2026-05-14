"""Module for dealing with tmdb data"""

from __future__ import annotations

import json
import logging
import os
import time

from argparse import ArgumentParser
from typing import Any

import tornado.gen

from nkpylib.web_utils import call_api, call_api_async, make_request_async

logger = logging.getLogger(__name__)

class IMDBFetcher:
    """Fetches IMDB pages and caches them.

    Note that IMDB has been rolling out more aggressive anti-bot measures, so this often results in
    an empty response with status code 202. Use the TMDB fetcher instead, if possible.
    """
    def __init__(self, cache_dir='cache/imdb', min_delay=0.3, expire_days=14):
        """Creates a new fetcher with the given cache directory and minimum delay between fetches.

        Fetched content is cached for `expire_days` days.
        """
        self.cache_dir = cache_dir
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            pass
        self.min_delay = min_delay
        self.expire_days = expire_days
        self.last_fetch = 0

    def _get_cache_file_extension(self) -> str:
        """Returns the file extension to use for cached files."""
        return '.html'

    async def _fetch(self, imdb_id: str) -> str:
        """Fetches the IMDB page for the given imdb_id, returning the content as a string."""
        r = await make_request_async(f'https://www.imdb.com/title/{imdb_id}/')
        logger.info(f'Got response {r.status_code} for {imdb_id}: {r.headers}, {r.text[:100]}, {len(r.text)} bytes')
        if len(r.text) > 100:
            return r.text
        else:
            return ''

    async def fetch(self, imdb_id: str) -> str:
        """Fetches the IMDB page for the given imdb_id, returning the content as a string.

        This will cache the content and only fetch it again if it's older than `expire_days` days.
        It's an async function so that we can wait for the fetch to complete, and also so we can
        wait between fetches if needed.
        """
        path = f'{self.cache_dir}/{imdb_id}{self._get_cache_file_extension()}'
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) > (self.expire_days * 24 * 3600):
            # delete it
            try:
                os.remove(path)
            except Exception as e:
                pass
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            # wait if we need to (async)
            if time.time() - self.last_fetch < self.min_delay:
                await tornado.gen.sleep(self.min_delay - (time.time() - self.last_fetch))
            logger.info(f'Fetching {imdb_id}')
            try:
                content = await self._fetch(imdb_id)
                if content:
                    with open(path, 'w') as f:
                        f.write(content)
                else:
                    return ''
            except Exception as e:
                logger.warning(f'Error fetching {imdb_id}: {e}')
                return ''
            self.last_fetch = time.time()
        with open(path) as f:
            return f.read()

    async def get_movie_info(self, imdb_id: str) -> dict|None:
        """Gets movie info for the given `imdb_id`, using the cache if possible."""
        page = await self.fetch(imdb_id)
        try:
            doc = pq(page)
        except Exception as e:
            logger.error(f'Error parsing page for {imdb_id}: {e}')
            return None
        cur = {}
        cur['summary'] = doc('[data-testid="plot-xl"]').text()
        ratings_div = doc('[data-testid="hero-rating-bar__aggregate-rating__score"]')
        # first span child has the rating number
        cur['rating'] = ratings_div('span').eq(0).text()
        # last div child has the number of votes
        cur['votes'] = ratings_div('div').parent().eq(-1).text().split('\n')[-1]
        return cur

    async def _get_poster_url(self, imdb_id: str, **kwargs) -> str|None:
        """Gets the poster URL for the given imdb_id. Returns None if not found."""
        doc = await self.fetch(imdb_id)
        try:
            url = pq(doc)('.ipc-poster img').attr('src')
            return url if url else None
        except Exception as e:
            return None

    async def get_poster(self, imdb_id: str, cache_dir='cache/posters', **kwargs) -> bytes|None:
        """Returns the raw bytes of the movie poster, or None on error"""
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            pass
        poster_path = f'{cache_dir}/{imdb_id}.jpg'
        if not os.path.exists(poster_path):
            # fetch the poster
            url = await self._get_poster_url(imdb_id, **kwargs)
            if url:
                with open(poster_path, 'wb') as f:
                    r = await make_request_async(url, min_delay=0.1)
                    f.write(r.content)
            else:
                return None
        with open(poster_path, 'rb') as f:
            return f.read()


class TMDBFetcher(IMDBFetcher):
    """Fetches TMDB info and caches it.

    Because tmdb ids are ints and imdb ids are strings starting with 'tt', we hardlink results
    to each other, so a lookup by either works.

    Note that TMDB has a generous API with a high rate limit, so we can fetch info in bulk without
    worrying about anti-bot measures.
    """
    def __init__(self,
                 cache_dir='cache/tmdb',
                 min_delay=0.1,
                 expire_days=4000,
                 api_key_env_var='TMDB_API_READ_ACCESS_TOKEN'):
        self.api_key_env_var = api_key_env_var
        super().__init__(cache_dir=cache_dir, min_delay=min_delay, expire_days=expire_days)

    def _get_cache_file_extension(self) -> str:
        """Returns the file extension to use for cached files."""
        return '.json'

    def _api(self, endpoint, headers=None, **kw):
        return call_api(f'https://api.themoviedb.org/3/{endpoint}',
                        api_key_env_var=self.api_key_env_var,
                        headers=headers,
                        min_delay=self.min_delay,
                        **kw)

    async def _api_async(self, endpoint, headers=None, **kw):
        return call_api_async(f'https://api.themoviedb.org/3/{endpoint}',
                              api_key_env_var=self.api_key_env_var,
                              headers=headers,
                              min_delay=self.min_delay,
                              **kw)

    async def _fetch_by_tmdb_id(self, tmdb_id: int) -> dict[str, Any]:
        """Fetches info from TMDB for the given tmdb_id, returning the content as a dict."""
        r = await self._api_async(f'movie/{tmdb_id}?append_to_response=keywords,lists')
        return r.json()

    async def _fetch(self, imdb_id: str) -> str:
        """Fetches info from TMDB for the given imdb_id, returning the content as JSON string."""
        r = await self._api_async(f'find/{imdb_id}?external_source=imdb_id')
        results = r.json()
        m = results['movie_results'][0]
        m = await self._fetch_by_tmdb_id(m['id'])
        if 'imdb_id' not in m:
            m['imdb_id'] = imdb_id
        return json.dumps(m, indent=2)

    async def fetch(self, imdb_id: str) -> dict[str, Any]:
        """Fetches info from TMDB for the given imdb_id, returning the content as an object.

        The object will be like:

            {
              "adult": false,
              "backdrop_path": "/3BTWehnJTnhxONyOcy1Bc9WQZBA.jpg",
              "genre_ids": [
                99
              ],
              "id": 869624,
              "media_type": "movie",
              "original_language": "en",
              "original_title": "Bitterbrush",
              "overview": "In remote Idaho, Colie and Hollyn embark on a long summer season working as range riders herding cattle. We follow them closely through the immensity of the landscapes and intimate moments of friendship. Emelie Mahdavian masterfully revisits the genre of the western and invites us to rethink the challenge of nomadism from the perspective of two young women.",
              "popularity": 0.7578,
              "poster_path": "/5YLUp1UXHDUlS5lwWuW98YJn1Hf.jpg",
              "release_date": "2022-06-24",
              "title": "Bitterbrush",
              "video": false,
              "vote_average": 6.0,
              "vote_count": 1
            }
        """
        content = await super().fetch(imdb_id)
        return json.loads(content) if content else {}

    async def get_movie_info(self, imdb_id: str) -> dict|None:
        """Gets movie info for the given `imdb_id`, using the cache if possible."""
        try:
            m = await self.fetch(imdb_id)
            if not m:
                return None
            cur = {}
            cur['summary'] = m.get('overview', '')
            cur['rating'] = m.get('vote_average', '')
            cur['votes'] = m.get('vote_count', '')
            return cur
        except Exception as e:
            logger.error(f'Error getting movie info for {imdb_id}: {e}')
            return None

    async def _get_poster_url(self, imdb_id: str, size: str = 'w300') -> str|None:
        """Gets the poster URL for the given imdb_id. Returns None if not found."""
        m = await self.fetch(imdb_id)
        poster_url_path = m.get('poster_path')
        if poster_url_path:
            return f'https://image.tmdb.org/t/p/{size}{poster_url_path}'
        return None



if __name__ == '__main__':
    tmdb = TMDBFetcher()
    arg_parser = ArgumentParser(description='Test TMDB API')
    arg_parser.add_argument('endpoint', type=str, help='API endpoint to call (e.g., "movie/550")')
    arg_parser.add_argument('args', nargs='*', help='Additional arguments for the API call (e.g., "language=en-US")')
    args = arg_parser.parse_args()
    resp = tmdb._api(args.endpoint, *args.args)
    print(json.dumps(resp.json(), indent=2))
