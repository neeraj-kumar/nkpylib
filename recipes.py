"""Various utilities related to recipes.

This deals with ld+json recipe cards, downloading recipes, parsing recipes, generating improved
recipes, etc.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from collections import Counter, defaultdict
from subprocess import check_output
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from pyquery import PyQuery as pq # type: ignore

from nkpylib.constants import URL_REGEXP
from nkpylib.web_search import BingWebSearch
from nkpylib.web_utils import make_request

logger = logging.getLogger(__name__)

def fetch_recipe_from_url(url: str) -> Optional[dict[str, Any]]:
    """Tries to fetch a recipe from a url and returns the raw object.

    This can raise various errors.
    """
    d = pq(url=url, opener=lambda url, **kw: make_request(url, **kw).text)
    recipe = None
    # iterate through ld+json sections
    for s in d('script[type="application/ld+json"]'):
        # try to JSON load it directly
        try:
            x = json.loads(s.text)
        except Exception:
            continue
        # check for the recipe object in various places
        to_check = []
        # first in @graph (most common)
        if "@graph" in x:
            to_check.extend(x["@graph"])
        # then check the object itself
        if isinstance(x, list): # it might be a list of objects
            for obj in x:
                if isinstance(obj, dict):
                    to_check.append(obj)
        elif isinstance(x, dict): # or it might be an object itself
            to_check.append(x)
        for obj in to_check:
            types = obj.get("@type", "")
            if isinstance(types, str):
                types = [types]
            if "recipe" not in {t.lower() for t in types}:
                continue
            recipe = obj
        if recipe is not None:
            break
    return recipe

def get_urls_from_pdf(path: str) -> Iterable[tuple[str, str]]:
    """Yields (host, url) pairs extracted from a pdf.

    These are sorted by most common hostname.
    """
    urls_by_host = defaultdict(set)
    args = ["pdftotext", path, "-"]
    out = check_output(args).decode("utf-8", "replace")
    urls = [m.group(0) for m in URL_REGEXP.finditer(out)]
    # group links by hostname
    for url in urls:
        urls_by_host[urlparse(url).hostname].add(url)
    hosts = Counter(urlparse(url).hostname for url in urls)
    for host, _count in hosts.most_common():
        if host is None:
            continue
        for url in urls_by_host[host]:
            yield host, url

def get_url_from_recipe(path: str, title: str) -> str:
    """Gets the url from a recipe, or empty string on error.

    This iterates through all urls from a recipe pdf, trying to fetch the recipe card from that
    url. It also tries to search for the recipe based on the host (domain) of the url and the given
    `title`. Again, if it can successfully parse a recipe card, then it returns that url.

    It returns the first url that matches, or '' if none do.
    """
    if not os.path.exists(path):
        return ''
    checked_urls = set()
    checked_hosts = set()
    ws = BingWebSearch()
    for host, url in get_urls_from_pdf(path):
        if url in checked_urls:
            break
        checked_urls.add(url)
        # try parsing the recipe directly from this url
        try:
            r = fetch_recipe_from_url(url)
            if isinstance(r, dict):
                return url
        except Exception as e:
            logger.warning(f'Error fetching {url}: {e}')
            continue
        # now try searching for the url based on the title
        if host in checked_hosts:
            continue
        checked_hosts.add(host)
        results = ws.search(f'site:{host} {title}')
        for i, r in enumerate(results):
            logger.debug(f'    {i}: {json.dumps(r, indent=2)}\n')
            if not r['url']:
                continue
            try:
                recipe = fetch_recipe_from_url(r['url'])
                logger.debug(f'    Got recipe at {r["url"]}: {json.dumps(recipe, indent=2)[:500]}')
                if isinstance(recipe, dict):
                    return r['url']
            except Exception as e:
                logger.warning(f'Error fetching recipe from {r["url"]}: {type(e)}: {e}')
                continue
    return ''
