"""A module to do web searches for both normal and images.

I've built a basic bing searcher, but there are paid options that I'm considering. I've limited to
those without a monthly subscription, and that do pay-as-you-go. I'm listing prices as ($x/k) for $x
per 1000 searches. Note that prices tend not to include sales tax.

- serper.dev - $1/k, min $50, expires within 6 months
- dataforseo.com - $2/k for live (6s), $1.2/k for priority (1m), $0.6/k for std (5m), min $50, free $1 on signup
- apify - $3.5/k, free plan has $5/month credit -> 1400 searches/mo
- brightdata - $0.75/k (for 6 months), has google, bing, duckduckgo, yandex, baidu

I signed up for brightdata on Sep 4, 2025.
"""

#TODO note that using "site:" triggers a captcha
#TODO aggregate common phrases
#TODO image search
#TODO image search + embeddings
#TODO unit conversions
#TODO ingredient substitutions
#TODO web parsing with LLMs
#     - roughly rag style
#     - feed it search query
#     - define structured output types
#     - merging results?
#     - look for existing libs
#     - does bing have summarized ai results?

from __future__ import annotations

import json
import logging
import os
import re
import time

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from os.path import dirname, basename, splitext, exists
from typing import Any, Iterable
from urllib.parse import quote_plus, urlparse

import requests

from pyquery import PyQuery as pq # type: ignore

from nkpylib.constants import USER_AGENT
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    site: str = ''
    snippet_url: str = '' # snippet of the url shown in the search result
    searcher_url: str = '' # the url used by the search engine, may redirect
    tags: list[str] = None
    snippet_list: list[str] = None
    image_url: str = '' # image url for result (optional)

class Searcher(ABC):
    """An abstract class for searching the web."""
    @abstractmethod
    def search(self, query: str, site: str='', **kw) -> Iterable[SearchResult]:
        """Search the web for the given query and yield an iterator over results."""
        pass

class AuthorizationFailure(Exception):
    pass


class BrightDataSearch(Searcher):
    def __init__(self, site:str='google', api_key_env_var: str='BRIGHTDATA_API_KEY'):
        self.api_key = os.getenv(api_key_env_var)
        self.zone_name = 'serp_api1'
        self.site = site

    def _url_by_query(self, query: str, site: str='') -> str:
        if self.site == 'google':
            url = f'https://www.google.com/search?brd_json=1&q={quote_plus(query)}'
            if site:
                url += f'+site:{quote_plus(site)}'
        else:
            raise NotImplementedError(f'Unsupported site: {self.site}')
        return url


    def search(self, query: str, site: str='', **kw) -> Iterable[SearchResult]:
        url = 'https://api.brightdata.com/request'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = dict(zone=self.zone_name, format='json', url=self._url_by_query(query, site))
        logger.debug(f'Payload: {payload}')
        req = make_request(url=url, method='post', headers=headers, json=payload)
        if req.status_code == 407:
            raise AuthorizationFailure('BrightData Authorization failed, check your API key')
        #req.raise_for_status()
        try:
            data = json.loads(req.json()['body'])
        except Exception as e:
            logger.info(f'Error parsing response: {e}, full response: {req.text}')
            raise
        logger.debug(f'Searching {self.site} for: {query}, got {req.url} -> {json.dumps(data, indent=2)}')
        results = []
        sr_map = dict(title='title', url='link', description='description', image_url='image_base64')
        for r in data['organic']:
            r = {k: r.get(v, '') for k, v in sr_map.items()}
            # extract the site from the url field
            r['site'] = urlparse(r['url']).netloc
            sr = SearchResult(**r)
            yield sr


class BingWebSearch(Searcher):
    """Does bing web searches (manual parsing)"""
    def search(self, query: str, site: str='', resolve_urls: bool=True) -> Iterable[SearchResult]: # type: ignore[override]
        """Search bing for the given query"""
        url = "https://www.bing.com/search"
        params = dict(q=query, qs='n', form='QBRE', sp=-1, pq=query)
        req = make_request(url, params=params)
        req.raise_for_status()
        logger.info(f'Searching bing for: {query}, got {req.url} -> {req.text[:100]}...')
        with open('bing.html', 'w', encoding='utf-8') as f:
            f.write(req.text)
        d = pq(req.text)
        algos = d('.b_algo')
        #print(f'Got {algos} results')
        results = []
        # first check the top-section
        # we want the parent div of the .b_tpcn class within the top section (if it exists)
        top_section = d(d('#b_topw .b_tpcn').parent())
        if top_section:
            results.extend(top_section.items())
        # now add remaining results
        results.extend(d('.b_algo').items())
        #logger.info(f'Found {len(results)} results: {results[:2]}...')
        for result in results:
            r: dict[str, Any] = {}
            orig = result('a').text() # this has a bunch of stuff, that we get in other ways
            if 'Tags:' in orig: # but tags is unique here, as far as i can tell (only sometimes)
                r['tags'] = [t.strip() for t in orig.split('Tags:')[1].split('\n') if t.strip()]
            r['site'] = result('a .tptt').text()
            r['snippet_url'] = result('a cite').text().replace(' \u203a ', '/').replace('\u2026', '...')
            r['title'] = result('h2').text()
            r['description'] = result('.b_caption p').text()
            r['searcher_url'] = result('a').attr('href')
            if resolve_urls:
                try:
                    r['url'] = self.resolve_bing_url(r['searcher_url']).split('#:~:text=')[0]
                except Exception as e:
                    logger.error(f'Error resolving bing url: {e}')
                    r['url'] = ''
            else:
                r['url'] = r['searcher_url']
            r['snippet_list'] = [s.text() for s in result('.lisn_content li').items()]
            # convert to a search result
            sr = SearchResult(**r)
            yield sr

    def resolve_bing_url(self, url: str) -> str:
        """Resolves the given bing url to the final destination.

        Because it uses js, we can't just use requests to get the final url.
        Instead, we just do a regexp search.

        Returns the final url or empty string if it can't be found.
        """
        content = make_request(url).text
        pat = re.compile('var u = "([^"]*)";')
        m = pat.search(content)
        if m:
            return m.group(1)
        else:
            return ''


#DefaultWebSearch = BingWebSearch
DefaultWebSearch = BrightDataSearch

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = ArgumentParser()
    parser.add_argument('query', help='The query to search for')
    args = parser.parse_args()

    w = DefaultWebSearch()
    for result in w.search(args.query):
        print(json.dumps(asdict(result), indent=2))
