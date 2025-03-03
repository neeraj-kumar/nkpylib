"""Utilities to interact with the Memento database API.

Online reference: https://mementodatabase.docs.apiary.io/#reference/
"""

from __future__ import annotations

import json
import os

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
    # there's a limit of 10 reqs/min -> 6s delay
    params = dict(**data, token=token)
    r = make_request(url, method=method, min_delay=6, params=params)
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
    """Class to interact with the Memento database API."""

    def __init__(self, name: str):
        self.library_id = self._find_library_id(name)
        self.library_info = get_library(self.library_id)

    def _find_library_id(self, name: str) -> str:
        """Finds the library ID by name (case-insensitive)."""
        libraries = get_libraries()
        for lib_name, lib_id in libraries.items():
            if lib_name.lower() == name.lower():
                return lib_id
        raise ValueError(f"Library '{name}' not found.")


    def get_entries(self, **data) -> list[dict]:
        """Returns the entries in the library."""
        return get_entries(self.library_id, **data)

    def search_entries(self, q: str, **data) -> list[dict]:
        """Searches the entries in the library for the given query `q`."""
        return search_entries(q, self.library_id, **data)


if __name__ == '__main__':
    p = lambda x: print(json.dumps(x, indent=2))
    r = get_libraries()
    p(r)
    info = get_library(r['movies'])
    p(info)
    #entries = get_entries(r['movies'])
    entries = search_entries('tt0140352', r['movies'])
    p(entries)
    #p(entries['entries'][:10])
