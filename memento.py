"""Utilities to interact with the Memento database API.

Online reference: https://mementodatabase.docs.apiary.io/#reference/
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from collections import Counter
from typing import Any, Iterator, TypedDict

import requests

from nkpylib.web_utils import make_request
from nkpylib.ml.client import call_llm
from nkpylib.ml.llm_utils import load_llm_json


logger = logging.getLogger(__name__)

class Entry(TypedDict, total=False):
    """A base class for an entry in a Memento database."""
    id: str
    status: str
    fields: dict


class EntriesWrapper(TypedDict, total=False):
    """A wrapper for a list of entries in a Memento database."""
    entries: list[Entry]


def memento_api(endpoint: str, method="get", retries=3, **data):
    """Runs a memento API query to given `endpoint`.

    We use the `MEMENTO_ACCESS_TOKEN` env var for auth.
    If the request fails due to rate limits, we retry up to `retries` times, pausing 6 seconds
    (since the API has a limit of 10/minute).
    """
    base_url = "https://api.mementodatabase.com/v1"
    endpoint = endpoint.lstrip("/")
    token = os.environ["MEMENTO_ACCESS_TOKEN"]
    url = f"{base_url}/{endpoint}"
    # there's a limit of 10 reqs/min, but we'll try pushing our luck a bit
    params = dict(**data, token=token)
    if method == "get":
        r = make_request(url, min_delay=6, params=params)
    else:
        r = make_request(f'{url}?token={token}', method=method, min_delay=6, json=params)
    logger.debug(f"Request: {r.request} to {r.request.url} with params {params}")
    ret = r.json()
    if retries > 0 and 'code' in ret and ret['code'] == 403 and 'rate limit' in ret['description']:
        logger.info(f'Rate limit hit on endpoint {endpoint}, retrying...')
        return memento_api(endpoint, method=method, retries=retries - 1, **data)
    return ret


def get_libraries() -> dict[str, str]:
    """Returns a dict from library name to library id in the Memento database."""
    libs = memento_api("libraries")
    try:
        return {lib["name"]: lib["id"] for lib in libs["libraries"]}
    except KeyError:
        logger.error(f"Failed to parse libraries response: {libs}")
        raise


def get_library_info(library_id: str) -> dict[str, Any]:
    """Returns the details of the library with the given `library_id`.

    The top level of the returned dict has some basic request info, and then `fields` is the list of
    fields in the library, along with their info.
    """
    return memento_api(f"libraries/{library_id}")


def _get_entries(library_id: str, pageSize=10000, fields="all", **data) -> dict[str, Any]:
    """Returns the raw entries in the library with the given `library_id`.

    For most uses you probably want the higher level `get_entries()` function, in particular because
    the entries returned here contain field values as field ids, not field names.

    Note that this returns deleted entries as well. Check if `status` is 'deleted', in which case it
    won't have fields.

    You can pass in any additional query params as `data`.
    Note that we don't currently implement pagination, so you should set `pageSize` to a high number.
    """
    return memento_api(f"libraries/{library_id}/entries", pageSize=pageSize, fields=fields, **data)


def map_fields(inputs: dict|list, library_info: dict[str, Any]) -> dict | list:
    """This maps fields either to or from memento format.

    Memento format is a list of dicts with 'id' and 'value' keys.
    Our format is a dict from field names to values. The mapping is from the `library_info`.
    """
    fields = library_info["fields"]
    field_mapping: dict[str, str]
    if isinstance(inputs, dict): # map from our format to memento format
        field_mapping = {field["name"]: field["id"] for field in fields}
        # we want an output list of tuples with id and value for each field
        lst = []
        for key, val in inputs.items():
            lst.append(dict(id=field_mapping[key], value=val))
        return lst
    else: # map from memento format to our format
        field_mapping = {field["id"]: field["name"] for field in fields}
        # we want an output dict with field names as keys
        d = {}
        for f in inputs:
            d[field_mapping[f["id"]]] = f["value"]
        return d


def get_entries(library_id: str, library_info=None, **data) -> EntriesWrapper:
    """Returns the entries in the library with the given `library_id`.

    This is a higher-level function that filters out deleted entries, and maps field ids to field
    names. It returns a top-level object which itself contains 'entries', which is the list.

    You can either provide the `library_info` yourself, or we will fetch it for you if `None`.
    """
    # first fetch the fields to get the field names
    if library_info is None:
        library_info = get_library_info(library_id)
    # now fetch the entries and filter out deleted ones
    entries = _get_entries(library_id, **data)
    entries["entries"] = [e for e in entries["entries"] if e["status"] != "deleted"]
    # remap fields
    for entry in entries["entries"]:
        entry["fields"] = map_fields(entry["fields"], library_info)
    return entries # type: ignore


def create_entry(library_id: str, library_info=None, **kw) -> Entry:
    """Creates an entry with given kw, which are mapped using the `library_info`."""
    if library_info is None:
        library_info = get_library_info(library_id)
    fields = map_fields(kw, library_info)
    return memento_api(f"libraries/{library_id}/entries", method="post", fields=fields)


def update_entry(library_id: str, entry_id: str, library_info=None, **kw) -> Entry:
    """Updates an entry with given kw, which are mapped using the `library_info`.

    If you don't provide a `library_info`, we will fetch it for you.
    """
    if library_info is None:
        library_info = get_library_info(library_id)
    fields = map_fields(kw, library_info)
    return memento_api(f"libraries/{library_id}/entries/{entry_id}", method="patch", fields=fields)


def search_entries(q: str, library_id: str, fields="all", **data) -> EntriesWrapper:
    """Searches the entries in the library with the given `library_id` for the given query `q`.

    You can pass in any additional query params as `data`.
    """
    # first url-encode the query
    q = requests.utils.quote(q) # type: ignore
    return memento_api(f"libraries/{library_id}/search", q=q, fields=fields, **data)


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

    def __init__(self, name: str, key_field: str = "", update_interval_s=60 * 60):
        libraries = get_libraries()
        for lib_name, lib_id in libraries.items():
            if lib_name.lower() == name.lower():
                self.library_id = lib_id
                break
        else:
            raise ValueError(f"Library '{name}' not found.")
        self.info = get_library_info(self.library_id)
        self.key_field = key_field
        self.update_interval_s = update_interval_s
        self.last_update = 0
        self.entries: dict[str, Any] = {}

    def refresh(self):
        """Refreshes the entries in the library if needed (based on last refresh time)."""
        if time.time() - self.last_update > self.update_interval_s:
            self.entries = {
                e["id"]: e for e in get_entries(self.library_id, library_info=self.info)["entries"]
            }
            self.last_update = time.time()

    def __len__(self) -> int:
        """Returns the number of entries in the library."""
        self.refresh()
        return len(self.entries)

    def _key_compare(self, key1: Any, key2: Any) -> bool:
        """Compares two keys for equality. By default this is just a simple equality check."""
        return key1 == key2

    def __iter__(self) -> Iterator[Entry]:
        """Iterates through the entries in the library.

        Note that this first calls `refresh()` to update the entries if needed.
        This also first creates a list over the entries, so that the iterator is stable.
        """
        self.refresh()
        return iter(list(self.entries.values()))

    def __getitem__(self, key: Any) -> Entry:
        """Returns the first entry with the given key (based on the `key_field`).

        If no matching entry is found, this raises a `KeyError`.
        Note that this first calls `refresh()` to update the entries if needed.
        """
        if not self.key_field:
            raise ValueError("You must set a `key_field` in the constructor to use this method.")
        self.refresh()
        for entry in self.entries.values():
            if self._key_compare(entry["fields"].get(self.key_field), key):
                return entry
        raise KeyError(f"Key '{key}' not found in library.")

    def __contains__(self, key: str) -> bool:
        """Returns True if the given key is in the library (based on the `key_field`)."""
        try:
            _ = self[key]
            return True
        except (KeyError, ValueError):
            return False

    def search(self, q: str, **data) -> list[Entry]:
        """Searches the entries in the library for the given query `q`.

        This version uses memento's own search functionality
        """
        entries = search_entries(q, self.library_id, fields="", **data)["entries"]
        return [self.entries[e["id"]] for e in entries]

    def key_to_id(self, key: str) -> str:
        """Converts a key to an entry id in the library."""
        return self[key]["id"]

    def update(self, entry_id: str, **kw) -> None:
        """Updates an entry with given fields and values."""
        resp = update_entry(self.library_id, entry_id, library_info=self.info, **kw)
        print(resp)
        # update our local version
        self.entries[entry_id]['fields'].update(**kw)

    def create(self, **kw) -> Entry:
        """Creates an entry with given fields and values, returning the new entry."""
        r = create_entry(self.library_id, library_info=self.info, **kw)
        # map the fields back to our format
        r["fields"] = map_fields(r["fields"], self.info) # type: ignore
        self.entries[r["id"]] = r
        return r


class MovieDB(MementoDB):
    """Subclass of MementoDB specialized for handling my movie list."""

    def __init__(self, update_interval_s=60 * 60):
        super().__init__(name="movies", key_field="imdb link", update_interval_s=update_interval_s)

    def _key_compare(self, key1: str, key2: str) -> bool:
        """Compares two keys by extracting and comparing IMDb IDs."""
        if key1 is None or key2 is None:
            return False
        imdb_id_pattern = re.compile(r"tt\d+")
        id1 = imdb_id_pattern.search(key1)
        id2 = imdb_id_pattern.search(key2)
        if id1 is None or id2 is None:
            return False
        # compare the first match on each
        # print(f'Comparing {key1} and {key2} -> {id1.group()} == {id2.group()}')
        return id1.group() == id2.group()

    def stats(self) -> Counter[str]:
        """Returns a Counter with various stats"""
        counts = Counter(total=len(self))
        for m in self:
            f = m["fields"]
            seen = "seen" if f["status"] else "unseen"
            pri = "low pri" if f["low priority"] else "high pri"
            counts[f"{seen} {pri}"] += 1
            counts[seen] += 1
            counts[pri] += 1
        return counts

    @staticmethod
    def test():
        """Tests out the movie database class."""
        movies = MovieDB()
        for i, m in enumerate(movies):
            print(i, m)
            if i > 2:
                break
        print(f"Movie stats: {movies.stats()}")
        #print(movies.create(**{'title': 'Blah blah blah'}))
        return
        movies.update(movies.key_to_id('tt0140352'), **{"imdb score": 8.5})
        print(movies["https://www.imdb.com/title/tt0140352/?ref_=fn_all_ttl_1"])
        print(movies["https://www.imdb.com/title/tt0140352/"])
        print(movies["https://www.imdb.com/title/tt0140352"])
        print(movies["https://m.imdb.com/title/tt0140352"])
        print(movies["tt0140352"])
        matches = movies.search("4alicia")
        print(f'Got {len(matches)} matches for "4alicia": {matches}')


class LifeLog(MementoDB):
    """Subclass of MementoDB specialized for handling my life log entries."""

    def __init__(self, update_interval_s=60 * 60):
        super().__init__(name="life log", key_field="Title", update_interval_s=1e9)


class Restaurants(MementoDB):
    """Subclass of MementoDB specialized for handling the restaurant list."""

    def __init__(self):
        super().__init__(name="Restaurants", key_field="Title")


class FoodReviews(MementoDB):
    """Subclass of MementoDB specialized for handling the food reviews list."""

    def __init__(self):
        super().__init__(name="food reviews", key_field="Title")

def split_reviews(r, revs):
    """Given a single review object `r`, split it into multiple reviews if it contains multiple dates."""
    desc = r['fields']['description']
    prompt = f'''The following is a review/notes of a restaurant. It might contain multiple reviews
    on different dates. If so, output a list of reviews as a JSON object, where each individual
    review is a [date in YYYY-MM-DD format, review text] pair. If the review starts without a date,
    then include the first part of the review (without a date) as the first review. If the whole
    thing is a single review, output an empty list.

    Output only a JSON object, no other text, errors, or explanation.

    Review:
    {desc}
    '''
    outs = load_llm_json(call_llm.single(prompt, max_tokens=128000))
    print(f'Converted {r} -> {outs}')
    if not isinstance(outs, list):
        raise ValueError(f"Expected a list of reviews, got {outs}")
    if len(outs) <= 1:
        return
    # more than one review, so duplicate with the different reviews
    f = r['fields']
    x = None
    for date, review in outs:
        if not date:
            print(f'  Updating review to: {review}')
            x = revs.update(r['id'], description=review)
        else:
            obj = dict(
                description=review,
                date=date+'T15:00:00Z', # middle of the day, since we don't know
                Restaurants=f['Restaurants'],
                Rating=f['Rating'],
                new=True,
            )
            print(f'  Creating new review: {obj}')
            x = revs.create(**obj)
        #print(f'   Got resp: {x}')



def migrate_food():
    raise NotImplementedError("Already done the migration, so don't call this again")
    rests = Restaurants()
    revs = FoodReviews()
    print(f'Migrating {len(revs)} food reviews to {len(rests)} restaurants')
    # add a restaurant to all reviews that don't have one
    for i, r in enumerate(revs):
        if 'Restaurants' not in r['fields'] and r['fields'].get('Title'):
            rest = rests[r['fields']['Title']]
            print(f'{i} Adding restaurant {rest["fields"]["Title"]} to review {r["fields"]["Title"]}')
            revs.update(r["id"], Restaurants=[rest["id"]])
    # split multiple reviews into separate entries
    for r in revs:
        if '20' in r['fields']['description']: # look for reviews with dates
            try:
                split_reviews(r, revs)
            except Exception:
                pass


if __name__ == "__main__":
    p = lambda x: print(json.dumps(x, indent=2))
    task = 'test-movies'
    if task == 'simple-test':
        r = get_libraries()
        p(r)
        info = get_library_info(r["movies"])
        p(info)
        entries = get_entries(r["movies"], library_info=info)
        p(entries["entries"][:10])
        serp = search_entries("tt0140352", r["movies"])
        p(serp)
    elif task == 'test-movies':
        MovieDB.test()
    elif task == 'migrate-food':
        migrate_food()
