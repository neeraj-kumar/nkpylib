"""A module to do web searches for both normal and images."""

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
import re
import time

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import Counter, defaultdict
from os.path import dirname, basename, splitext, exists
from typing import Any, Iterable
from urllib.parse import quote_plus, urlparse

import requests

from pyquery import PyQuery as pq # type: ignore

from nkpylib.constants import USER_AGENT
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

class Searcher(ABC):
    """An abstract class for searching the web."""
    @abstractmethod
    def search(self, query: str, **kw) -> Iterable[dict]:
        """Search the web for the given query and yield an iterator over results."""
        pass


class BingWebSearch(Searcher):
    """Does bing web searches"""
    def search(self, query: str, resolve_urls: bool=True) -> Iterable[dict]: # type: ignore[override]
        """Search bing for the given query"""
        url = "https://www.bing.com/search"
        params = {'q': query}
        req = make_request(url, params=params)
        req.raise_for_status()
        logger.info(f'Searching bing for: {query}, got {req.url} -> {req.text[:100]}...')
        d = pq(req.text)
        results = []
        # first check the top-section
        # we want the parent div of the .b_tpcn class within the top section (if it exists)
        top_section = d(d('#b_topw .b_tpcn').parent())
        if top_section:
            results.extend(top_section.items())
        # now add remaining results
        results.extend(d('.b_algo').items())
        logger.info(f'Found {len(results)} results: {results[:2]}...')
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
