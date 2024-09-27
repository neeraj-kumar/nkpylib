"""A module to do web searches for both normal and images."""

#TODO aggregate common phrases
#TODO image search
#TODO image search + embeddings
#TODO unit conversions
#TODO ingredient substitutions

from __future__ import annotations

import json
import logging
import re

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os.path import dirname, basename, splitext, exists
from typing import Iterable, List, Tuple
from urllib.parse import quote_plus

import requests

from pyquery import PyQuery as pq

from constants import USER_AGENT

logger = logging.getLogger(__name__)

def make_request(url: str, method='get', **kwargs) -> requests.Response:
    """Makes a request to the given `url` with `method` with the given kwargs"""
    headers = {'User-Agent': USER_AGENT}
    return requests.request(method, url, headers=headers, **kwargs)

def resolve_url(url: str, method='head', **kwargs) -> str:
    """Follows the url through all redirects and returns the ultimate url"""
    r = make_request(url, method, **kwargs)
    r.raise_for_status()
    return r.url

class Searcher(ABC):
    """An abstract class for searching the web."""

    @abstractmethod
    def search(self, query: str) -> Iterable[dict]:
        """Search the web for the given query and yield an iterator over results."""
        pass


class BingWebSearch(Searcher):
    """Does bing web searches"""
    def search(self, query: str) -> Iterable[dict]:
        """Search bing for the given query"""
        url = "https://www.bing.com/search"
        params = {'q': query}
        r = make_request(url, params=params)
        r.raise_for_status()
        d = pq(r.text)
        for result in d('.b_algo').items():
            r= {}
            orig = result('a').text() # this has a bunch of stuff, that we get in other ways
            if 'Tags:' in orig: # but tags is unique here, as far as i can tell (only sometimes)
                r['tags'] = [t.strip() for t in orig.split('Tags:')[1].split('\n') if t.strip()]
            r['site'] = result('a .tptt').text()
            r['snippet_url'] = result('a cite').text().replace(' \u203a ', '/').replace('\u2026', '...')
            r['title'] = result('h2').text()
            r['description'] = result('.b_caption p').text()
            r['searcher_url'] = result('a').attr('href')
            try:
                r['url'] = self.resolve_bing_url(r['searcher_url']).split('#:~:text=')[0]
            except Exception as e:
                logger.error(f'Error resolving bing url: {e}')
                r['url'] = ''
            r['snippet_list'] = [s.text() for s in result('.lisn_content li').items()]
            yield r

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('query', help='The query to search for')
    args = parser.parse_args()

    b = BingWebSearch()
    for result in b.search(args.query):
        print(json.dumps(result, indent=2))
