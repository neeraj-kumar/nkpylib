"""Various utilities for interacting with Airtable.

The lowest-level function is `airtable_api_call`, which makes a call to the Airtable API to
arbitrary endpoints with arbitrary keyword args. It also logs all calls to `AIRTABLE_LOG_FILE`.

You can either pass in an `api_key` and `base_id` as keyword args, else it will try to read
`AIRTABLE_API_KEY` and `AIRTABLE_BASE_ID` from the environment.

For more sustained interactions, there's the `AirtableUpdater` class, which allows you to read and
write to a specific table. You can also specify a list of tables that are referenced in this one,
and we will preload them.
"""

from __future__ import annotations

import json
import logging
import os
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime
from pprint import pprint
from typing import Any, Iterator, Optional

import requests

logger = logging.getLogger(__name__)

AIRTABLE_LOG_FILE = 'airtable_log.jsonl'


def airtable_api_call(endpoint: str, method: str='get', api_key: str='', base_id: str='', **kwargs) -> Any:
    """Makes an API call to airtable and returns the JSON response.

    Uses the given `endpoint` and `method` (GET, POST, PATCH, DELETE). Any additional keyword args
    are passed as query parameters or json data, depending on the method.

    If you don't pass in `api_key` or `base_id`, then this reads the environment variables
    `AIRTABLE_API_KEY` or `AIRTABLE_BASE_ID`, respectively.
    """
    api_key = api_key or os.environ.get('AIRTABLE_API_KEY', '')
    base_id = base_id or os.environ.get('AIRTABLE_BASE_ID', '')
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    url = f"https://api.airtable.com/v0/{base_id}/{endpoint}"
    method = method.lower()
    logger.debug(f'Calling {method} on {url} with {kwargs} and headers {headers}')
    if method == "get":
        url += "?" + "&".join(f"{key}={value}" for key, value in kwargs.items())
        r = requests.get(url, headers=headers)
    else:
        # add to log file
        with open(AIRTABLE_LOG_FILE, 'a') as f:
            to_write = dict(url=url,
                            endpoint=endpoint,
                            method=method,
                            timestamp=time.time(),
                            human_time=str(datetime.now()),
                            **kwargs)
            f.write(json.dumps(to_write, sort_keys=True) + '\n')
        if method == "delete":
            # deletes are not json, but a url-encoded list of 'ids'
            # in curl, it would be records[]=rec123&records[]=rec456
            assert 'ids' in kwargs, "You must pass in a list of 'ids' to delete"
            cur_headers = dict(headers, **{'Content-Type': 'application/json'})
            # note that we send as 'params', not 'json'
            r = requests.delete(url, headers=cur_headers, params={'records[]': list(kwargs['ids'])})
        else:
            func = getattr(requests, method)
            #print(f'trying to call {func} with url={url} and json={kwargs} and headers {headers}')
            r = func(url, json=kwargs, headers=headers)
    return r.json()

def airtable_api_gen(**kw) -> Any:
    """A generator for Airtable API calls.

    Useful for pagination. This will yield each page of results, one at a time.
    If you just want the first one, then use `airtable_api_call` directly, or call this with
    `next(airtable_api_gen(...))`.
    """
    offset = ''
    while True:
        resp = airtable_api_call(offset=offset, **kw)
        if 'error' in resp:
            raise Exception(f'Error in response: {resp}, api call with kw={kw}')
        yield resp
        offset = resp.get('offset', None)
        if not offset:
            break

def airtable_all_rows(table_name: str, **kw) -> list[dict]:
    """Returns all rows from `table_name`"""
    logger.info(f'Getting all rows from {table_name}')
    return [row for resp in airtable_api_gen(endpoint=table_name, **kw) for row in resp['records']]


class AirtableUpdater:
    """A way to read/write updates to Airtable.

    You initialize it with a specific table to update as well as the field name to use as the key.
    Then you can simply use it like a dict to read/write updates to the table, keyed by the given
    field name.
    """
    def __init__(self, table_name: str, key_name: str, map_tables: Optional[list[str]]=None, needs_review:bool = True) -> None:
        """This is an updater for given table and key name.

        You can optionally specify a list of tables that are referenced in this one, and we will
        preload them.
        
        If you set `needs_review` to True (default), then we will automatically set the 'needs
        review' field on all updates.
        """
        self.table_name = table_name
        self.key_name = key_name
        self.needs_review = needs_review
        # load any mappers
        self.mappers: defaultdict[str, dict] = defaultdict(dict)
        pool = ThreadPoolExecutor()
        futures = []
        f = pool.submit(lambda: list(airtable_all_rows(table_name)))
        futures.append(('all', f))
        if map_tables is not None:
            for mt_name in map_tables:
                f = pool.submit(lambda mt_name: list(airtable_all_rows(mt_name)), mt_name)
                futures.append((mt_name, f))
            for mt_name, f in futures[1:]:
                logger.info(f'Reading mapping rows for {mt_name}')
                for row in f.result():
                    self.add_map_value(mt_name, row)
            futures = futures[:1]
        # load the entire table
        self.data = {}
        logger.info(f'Reading main data for {table_name}')
        for row in futures[0][1].result():
            self._setrow(row)
            if key_name == 'id':
                value = row['id']
            else:
                value = row['fields'].get(key_name, None)
            if value:
                self.data[value] = row['fields']
                self.data[value]['id'] = row['id']
        logger.info(f'Read {len(self.data)} rows from {table_name}, key={key_name}')

    def _setrow(self, row: dict) -> None:
        """Sets a given row in the data"""
        if self.key_name == 'id':
            key = row['id']
        else:
            key = row['fields'].get(self.key_name, None)
        if key:
            self.data[key] = row['fields']
            self.data[key]['id'] = row['id']

    def __len__(self) -> int:
        return len(self.data)

    def add_map_value(self, mt_name: str, row: dict) -> None:
        """Adds a row to the mapping table.

        The row is assumed to be from the response to a POST call.
        """
        to_add = row['fields'].copy()
        to_add['id'] = row['id']
        self.mappers[mt_name][row['id']] = to_add
        if 'Name' in row['fields']:
            self.mappers[mt_name][row['fields']['Name']] = to_add

    def create(self, error_on_mapping:bool = True, **kw) -> Any:
        """Creates a new row with given `kw`, performing mapping"""
        for mt_name, mt_data in self.mappers.items():
            if mt_name in kw:
                mt_value = kw[mt_name]
                try:
                    if isinstance(mt_value, str):
                        kw[mt_name] = mt_data[mt_value]['id']
                    else: # must be a list
                        kw[mt_name] = [mt_data[v]['id'] for v in mt_value]
                except KeyError:
                    if error_on_mapping:
                        raise Exception(f'Could not find {mt_name}={mt_value} in mappers: {sorted(mt_data)}')
                    else:
                        del kw[mt_name]
        resp = airtable_api_call(endpoint=self.table_name, method='post', records=[dict(fields=dict(**kw))])
        new_id = resp['records'][0]['id']
        self._setrow(resp['records'][0])
        return resp

    def update(self, key: str, typecast=False, **kw) -> Any:
        """Updates the row with the given key"""
        row = self.data[key]
        if self.needs_review and 'needs review' not in kw:
            kw['needs review'] = True
        row.update(**kw)
        # do mapping
        for field, value in kw.items():
            if field in self.mappers:
                if isinstance(value, str):
                    row[field] = kw[field] = self.mappers[field][value]['id']
                else:
                    row[field] = kw[field] = [self.mappers[field][v]['id'] for v in value]
        resp = airtable_api_call(endpoint=self.table_name, method='patch', records=[dict(id=row['id'], fields=kw)], typecast=typecast)
        if 'error' in resp:
            raise Exception(f'Error in airtable patch with key {key}: {resp}, api call with kw={kw}')
        self._setrow(resp['records'][0])
        return resp

    def delete(self, key: str) -> Any:
        """Deletes the row with the given key"""
        if key not in self.data:
            return
        resp = airtable_api_call(endpoint=self.table_name, method='delete', ids=[self.data[key]['id']])
        del self.data[key]
        return resp

    def upsert(self, key: str, typecast=False, error_on_mapping:bool=True, **kw) -> Any:
        """Updates or inserts"""
        if key in self:
            return self.update(key, typecast=typecast, **kw)
        else:
            newkw = {self.key_name: key, **kw}
            return self.create(error_on_mapping=error_on_mapping, **newkw)

    def get(self, key: str) -> dict[str, Any]:
        """Returns the row with the given key, mapping tables that we have."""
        row = deepcopy(self.data[key])
        for field, value in row.items():
            if field in self.mappers:
                if isinstance(value, list):
                    row[field] = [self.mappers[field].get(v, v) for v in value]
                else:
                    row[field] = self.mappers[field].get(value, value)
        return row

    def __iter__(self):
        """Iterates through our keys"""
        return iter(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def items(self) -> Iterator[tuple[str, dict]]:
        """Yields key, row pairs, where the row is the output from `get`"""
        for key in self:
            yield key, self.get(key)


if __name__ == '__main__':
    # filter by last modified time is today
    now = '2024-10-15T01:59'
    ret = airtable_api_call('Dishes', filterByFormula="{last_modified} >= '%s'" % now)
    recs = [r['fields']['last_modified'] for r in ret['records']]
    print(f'Got {len(recs)} records for dishes modified since {now}')
    pprint(recs)
