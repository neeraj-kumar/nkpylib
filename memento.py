"""Utilities to interact with the Memento database API.

Online reference: https://mementodatabase.docs.apiary.io/#reference/
"""

from __future__ import annotations

import json
import os
import re
import time

from typing import Any

import requests

from web_utils import make_request

MEMENTO_CACHE = {}

def memento_api(endpoint: str, method='get', **data):
    """Runs a memento API query to given `endpoint`.

    We use the `MEMENTO_ACCESS_TOKEN` env var for auth.
    If you provide `data`, then we make a POST request, else GET.
    """
    base_url = 'https://api.mementodatabase.com/v1'
    endpoint = endpoint.lstrip('/')
    conjunction = '&' if '?' in endpoint else '?'
    token = os.environ['MEMENTO_ACCESS_TOKEN']
    url = f'{base_url}/{endpoint}{conjunction}token={token}'
    url = f'{base_url}/{endpoint}'
    # there's a limit of 10 reqs/min, but we'll try pushing our luck a bit
    params = dict(**data, token=token)
    r = make_request(url, method=method, min_delay=1, params=params)
    return r.json()

def get_libraries() -> dict[str, str]:
    """Returns a dict from library name to library id in the Memento database."""
    libs = memento_api('libraries')
    return {lib['name']: lib['id'] for lib in libs['libraries']}

def get_library(library_id: str, reset_cache:bool=False) -> dict:
    """Returns the details of the library with the given `library_id`.

    The top level of the returned dict has some basic request info, and then `fields` is the list of
    fields in the library, along with their info.

    This is a cached call, but you can reset the cache by setting `reset_cache=True`.
    """
    if reset_cache or library_id not in MEMENTO_CACHE:
        MEMENTO_CACHE[library_id] = memento_api(f'libraries/{library_id}')
    return MEMENTO_CACHE[library_id]

def _get_entries(library_id: str, pageSize=10000, fields='all', **data) -> dict:
    """Returns the raw entries in the library with the given `library_id`.

    For most uses you probably want the higher level `get_entries()` function, in particular because
    the entries returned here contain field values as field ids, not field names.

    Note that this returns deleted entries as well. Check if `status` is 'deleted', in which case it
    won't have fields.

    You can pass in any additional query params as `data`.
    Note that we don't currently implement pagination, so you should set `pageSize` to a high number.
    """
    return memento_api(f'libraries/{library_id}/entries', pageSize=pageSize, fields=fields, **data)

def get_entries(library_id: str, reset_cache=False, **data) -> list[dict]:
    """Returns the entries in the library with the given `library_id`.

    This is a higher-level function that filters out deleted entries, and maps field ids to field
    names.

    We cache the call to get the library info, which you can reset by setting reset_cache=True.
    """
    # first fetch the fields to get the field names
    fields = get_library(library_id, reset_cache=reset_cache)['fields']
    field_mapping = {field['id']: field['name'] for field in fields}
    # now fetch the entries and filter out deleted ones
    entries = _get_entries(library_id, **data)
    entries['entries'] = [e for e in entries['entries'] if e['status'] != 'deleted']
    # remap fields
    for entry in entries['entries']:
        cur = {}
        for f in entry['fields']:
            cur[field_mapping[f['id']]] = f['value']
        entry['fields'] = cur
    return entries

def search_entries(q: str, library_id: str, **data) -> list[dict]:
    """Searches the entries in the library with the given `library_id` for the given query `q`.

    You can pass in any additional query params as `data`.
    """
    # first url-encode the query
    q = requests.utils.quote(q)
    return memento_api(f'libraries/{library_id}/search', q=q, fields='all', **data)


class MementoDB:
    """Class to interact with a single library in the Memento database.

    Initialize with the name (case-insensitive) of the library you want to work with.
    We support the following key operations:
    - iterate through entries
    - add entries
    - update entries
    - lookup entries by text search

    On first query, this will fetch all the entries and cache them. The cache can be updated
    periodically, based on a provided value.

    If you specify a `key_field` in the init, then you can use `__getitem__` to lookup entries and
    `__contains__` to check if an entry exists.
    """

    def __init__(self, name: str, key_field: str='', update_interval_s=60*60):
        libraries = get_libraries()
        for lib_name, lib_id in libraries.items():
            if lib_name.lower() == name.lower():
                self.library_id = lib_id
                break
        else:
            raise ValueError(f"Library '{name}' not found.")
        self.info = get_library(self.library_id)
        self.key_field = key_field
        self.update_interval_s = update_interval_s
        self.last_update = 0
        self.entries = []

    def refresh(self):
        """Refreshes the entries in the library if needed (based on last refresh time)."""
        if time.time() - self.last_update > self.update_interval_s:
            self.entries = get_entries(self.library_id)['entries']
        self.last_update = time.time()

    def __len__(self) -> int:
        """Returns the number of entries in the library."""
        self.refresh()
        return len(self.entries)

    @classmethod
    def key_compare(cls, key1: Any, key2: Any) -> bool:
        """Compares two keys for equality. By default this is just a simple equality check."""
        return key1 == key2

    def __iter__(self):
        """Iterates through the entries in the library.

        Note that this first calls `refresh()` to update the entries if needed.
        """
        self.refresh()
        if self.last_update > 0:
            return iter(self.entries)

    def __getitem__(self, key: Any) -> dict:
        """Returns the first entry with the given key (based on the `key_field`).

        If no matching entry is found, this raises a `KeyError`.
        Note that this first calls `refresh()` to update the entries if needed.
        """
        #TODO cache the key lookup
        if not self.key_field:
            raise ValueError("You must set a `key_field` in the constructor to use this method.")
        self.refresh()
        for entry in self.entries:
            if self.key_compare(entry.get(self.key_field), key):
                return entry
        raise KeyError(f"Key '{key}' not found in library.")

    def __contains__(self, key: str) -> bool:
        """Returns True if the given key is in the library (based on the `key_field`)."""
        try:
            _ = self[key]
            return True
        except (KeyError, ValueError):
            return False

    def search_entries(self, q: str, **data) -> list[dict]:
        """Searches the entries in the library for the given query `q`.

        This version uses memento's own search functionality
        """
        return search_entries(q, self.library_id, **data)


class MovieDB(MementoDB):
    """Subclass of MementoDB specialized for handling my movie list."""

    @classmethod
    def key_compare(cls, key1: Any, key2: Any) -> bool:
        """Compares two keys by extracting and comparing IMDb IDs."""
        imdb_id_pattern = re.compile(r'tt\d+')
        id1 = imdb_id_pattern.search(key1)
        id2 = imdb_id_pattern.search(key2)
        return id1.group() == id2.group() if id1 and id2 else False


if __name__ == '__main__':
    p = lambda x: print(json.dumps(x, indent=2))
    if 0:
        r = get_libraries()
        p(r)
        info = get_library(r['movies'])
        p(info)
        #entries = get_entries(r['movies'])
        entries = search_entries('tt0140352', r['movies'])
        p(entries)
        #p(entries['entries'][:10])
    else:
        movies = MementoDB('movies', key_field='imdb_link')
        print(movies.info)
        movies.refresh()
        for i, m in enumerate(movies.entries[::-1]):
            print(i, m)
            if i > 5:
                break

