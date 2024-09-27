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
from datetime import date, datetime
from typing import Any

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
    def __init__(self, table_name: str, key_name: str, map_tables: Optional[list[str]]=None) -> None:
        """This is an updater for given table and key name.

        You can optionally specify a list of tables that are referenced in this one, and we will
        preload them.
        """
        self.table_name = table_name
        self.key_name = key_name
        # load any mappers
        self.mappers = defaultdict(dict)
        if map_tables is not None:
            for mt_name in map_tables:
                for row in airtable_all_rows(mt_name):
                    self.add_map_value(mt_name, row)
        # load the entire table
        self.data = {}
        for row in airtable_all_rows(table_name):
            if key_name == 'id':
                value = row['id']
            else:
                value = row['fields'].get(key_name, None)
            if value:
                self.data[value] = row['fields']
                self.data[value]['id'] = row['id']
        logger.info(f'Read {len(self.data)} rows from {table_name}, key={key_name}')

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
        if 'needs review' not in kw:
            kw['needs review'] = True
        return airtable_api_call(endpoint=self.table_name, method='post', records=[dict(fields=dict(**kw))])

    def update(self, key: str, typecast=False, **kw) -> Any:
        """Updates the row with the given key"""
        row = self.data[key]
        if 'needs review' not in kw:
            kw['needs review'] = True
        row.update(**kw)
        # do mapping
        for field, value in kw.items():
            if field in self.mappers:
                if isinstance(value, str):
                    row[field] = kw[field] = self.mappers[field][value]['id']
                else:
                    row[field] = kw[field] = [self.mappers[field][v]['id'] for v in value]
        return airtable_api_call(endpoint=self.table_name, method='patch', records=[dict(id=row['id'], fields=kw)], typecast=typecast)

    def upsert(self, key: str, typecast=False, error_on_mapping:bool=True, **kw) -> Any:
        """Updates or inserts"""
        if key in self:
            return self.update(key, typecast=typecast, **kw)
        else:
            newkw = {self.key_name: key, **kw}
            return self.create(error_on_mapping=error_on_mapping, **newkw)

    def get(self, key: str) -> dict[str, Any]:
        """Returns the row with the given key"""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data
