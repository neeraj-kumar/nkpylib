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


def airtable_api_call(endpoint: str, method: str='get', api_key: str='', base_id: str='', is_meta:bool=False, **kwargs) -> Any:
    """Makes an API call to airtable and returns the JSON response.

    Uses the given `endpoint` and `method` (GET, POST, PATCH, DELETE). Any additional keyword args
    are passed as query parameters or json data, depending on the method.

    If you don't pass in `api_key` or `base_id`, then this reads the environment variables
    `AIRTABLE_API_KEY` or `AIRTABLE_BASE_ID`, respectively.

    If `is_meta` is True, then this will call the meta endpoint instead of the normal one.
    In that case, we don't use `base_id`.
    """
    api_key = api_key or os.environ.get('AIRTABLE_API_KEY', '')
    base_id = base_id or os.environ.get('AIRTABLE_BASE_ID', '')
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    url = f"https://api.airtable.com/v0/{base_id}/{endpoint}"
    if is_meta:
        url = f"https://api.airtable.com/v0/meta/{endpoint}"
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

def get_base_schema(base_id: str='', **kw):
    """Returns the metadata for the schema of all tables in a particular base"""
    base_id = base_id or os.environ.get('AIRTABLE_BASE_ID', '')
    return airtable_api_call(f'bases/{base_id}/tables', is_meta=True, **kw)

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
    t0 = time.time()
    ret = [row for resp in airtable_api_gen(endpoint=table_name, **kw) for row in resp['records']]
    t1 = time.time()
    logger.info(f'Took {t1-t0:.2f} seconds to get {len(ret)} rows from {table_name}')
    return ret


class AirtableUpdater:
    """A way to read/write updates to Airtable.

    You initialize it with a specific table to update as well as the field name to use as the key.
    Then you can simply use it like a dict to read/write updates to the table, keyed by the given
    field name.
    """
    def __init__(self,
                 table_name: str,
                 key_name: str,
                 map_tables: Optional[list[str]]=None,
                 needs_review:bool = True) -> None:
        """This is an updater for given table and key name.

        You can optionally specify a list of tables that are referenced in this one, and we will
        preload them.

        If you set `needs_review` to True (default), then we will automatically set the 'needs
        review' field on all updates.
        """
        self.table_name = table_name
        self.key_name = key_name
        self.needs_review = needs_review
        # kick off fetching the main table, using futures
        pool = ThreadPoolExecutor()
        main_futures = []
        if table_name == 'Dishes': # special case since this is so big
            #TODO generalize this somehow
            # fetch the main dishes table using multiple subsets based on length of the name
            fmt = 'and(len(%7BName%7D)%3EXXX%2Clen(%7BName%7D)%3CYYY)'
            breakpoints = [0, 30, 40, 50, 60, 1000]
            for b0, b1 in zip(breakpoints[:-1], breakpoints[1:]):
                formula = fmt.replace('XXX', str(b0)).replace('YYY', str(b1))
                f = pool.submit(lambda: list(airtable_all_rows(table_name, filterByFormula=formula)))
                main_futures.append(f)
        else:
            main_futures.append(pool.submit(lambda: list(airtable_all_rows(table_name))))
        # load any mappers
        self.mappers: defaultdict[str, dict] = defaultdict(dict)
        futures = []
        if map_tables is not None:
            for mt_name in map_tables:
                f = pool.submit(lambda mt_name: list(airtable_all_rows(mt_name)), mt_name)
                futures.append((mt_name, f))
            for mt_name, f in futures:
                logger.info(f'Reading mapping rows for {mt_name}')
                for row in f.result():
                    self.add_map_value(mt_name, row)
        # placeholder for the schema
        self._schema: dict = {}
        # load the entire table
        self.data = {}
        logger.info(f'Reading main data for {table_name}')
        for f in main_futures:
            rows = f.result()
            logger.info(f'  Dish future got {len(rows)} rows')
            for row in rows:
                self._setrow(row)
                if key_name == 'id':
                    value = row['id']
                else:
                    value = row['fields'].get(key_name, None)
                if value:
                    self.data[value] = row['fields']
                    self.data[value]['id'] = row['id']
        logger.info(f'Read {len(self.data)} rows from {table_name}, key={key_name}')

    @property
    def schema(self) -> dict:
        """Returns the schema for the table"""
        if not self._schema:
            schema = get_base_schema()
            for table in schema['tables']:
                if table['name'] == self.table_name:
                    self._schema = table
                    break
        return self._schema

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


class AirtableTree(Tree):
    def __init__(self,
                 table_name: str,
                 key_field: str,
                 hash_field: str,
                 base_id: str='',
                 api_key: str='',
                 row_filter_func: Optional[Callable]=None,
                 debug: bool=False):
        """Initialize this airtable tree with the given table and base id.

        If you don't specify a base or api key, we read them from environment variables
        `AIRTABLE_BASE_ID` or `AIRTABLE_API_KEY`, respectively.

        We check for the main keys (e.g., paths) in the given `key_field` and hashes in the given
        `hash_field`. We store a mapping `key_to_row` from keys to the full row data. We pull down
        all rows that have a non-empty `key_field` and that return True-ish from
        `row_filter_func(row)`. (If no `row_filter_func` is given, then we skip that second check.)

        If you set `debug` to True, then no changes will be made.
        """
        self.table_name = table_name
        self.key_field = key_field
        self.hash_field = hash_field
        self.airtable_kw = dict(base_id=base_id, api_key=api_key)
        self.debug = debug
        self.incr = 10 # this is an airtable limit for adding/deleting/etc
        rows = airtable_all_rows(table_name=self.table_name, **self.airtable_kw)
        #print(f'Got {len(rows)} rows: {json.dumps([rows[:2]], indent=2)}')
        self.key_to_row, keys = {}, []
        for row in rows:
            if key_field not in row['fields']:
                continue
            if row_filter_func is not None and not row_filter_func(row):
                continue
            keys.append(row['fields'][key_field])
            self.key_to_row[keys[-1]] = row
        print(f'Got {len(keys)} valid rows: {keys[:5]}')
        super().__init__(keys)

    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Returns the hash values for the given `keys`"""
        return [self.key_to_row[key]['fields'].get(self.hash_field, '') for key in keys]

    def execute_add(self, diffs: list[Diff], other: Tree) -> None:
        """Executes ADD operations from given `diffs` going from `self` to `other`.

        This adds the given keys and their hashes to airtable.
        """
        logger.info(f'Adding {len(diffs)} ids: {[d.b for d in diffs[:5]]}')
        def do_add(to_add):
            if self.debug:
                logger.info(f'Adding {len(to_add)}: {json.dumps(to_add, indent=2)}')
            else:
                resp = airtable_api_call(method='post', endpoint=self.table_name, records=to_add, **self.airtable_kw)
                #print(resp)

        hashes = other.hash_function([d.b for d in diffs if d.b])
        rows = [dict(fields={self.key_field: d.b, self.hash_field: hashes[i]}) for i, d in # type: ignore[index]
                enumerate(diffs)]
        list(incr_iter(tqdm(rows), func=do_add, incr=self.incr))

    def execute_delete(self, diffs: list[Diff], other: Tree) -> None:
        """Executes DELETE operations from given `diffs` going from `self` to `other`

        This does a straightforward airtable delete, by mapping from the keys to the row ids.
        """
        logger.info(f'Deleting {len(diffs)} ids: {[d.a for d in diffs[:5]]}')
        def do_del(to_del):
            if self.debug:
                logger.info(f'Deleting {len(to_del)}: {to_del}')
            else:
                # note that this is a url-encoded array of record ids, not json
                resp = airtable_api_call(method='delete', endpoint=f'{self.table_name}', ids=to_del, **self.airtable_kw)
                #print(resp)

        to_del = [self.key_to_row[d.a]['id'] for d in diffs]
        list(incr_iter(tqdm(to_del), func=do_del, incr=self.incr))

    def execute_move(self, diffs: list[Diff], other: Tree) -> None:
        """Executes MOVE operations from given `diffs` going from `self` to `other`

        In airtable, we can do a PATCH request with the row id and new key to update the key field.
        """
        logger.info(f'Moving {len(diffs)} ids: {diffs[:5]}')
        def do_move(to_move):
            if self.debug:
                logger.info(f'Moving {len(to_move)}: {json.dumps(to_move, indent=2)}')
            else:
                resp = airtable_api_call(method='patch', endpoint=self.table_name, records=to_move, **self.airtable_kw)
                #print(resp)

        to_move = []
        for d in diffs:
            row = self.key_to_row[d.a]
            to_move.append(dict(id=row['id'], fields={self.key_field: d.b}))
        list(incr_iter(tqdm(to_move), func=do_move, incr=self.incr))

    def execute_copy(self, diffs: list[Diff], other: Tree) -> None:
        """Executes COPY operations from given `diffs` going from `self` to `other`

        We don't implement this because we generally don't want duplicate rows in airtable.
        """
        raise NotImplementedError("execute COPY not implemented for Airtable")



if __name__ == '__main__':
    # filter by last modified time is today
    now = '2024-10-15T01:59'
    ret = airtable_api_call('Dishes', filterByFormula="{last_modified} >= '%s'" % now)
    recs = [r['fields']['last_modified'] for r in ret['records']]
    print(f'Got {len(recs)} records for dishes modified since {now}')
    pprint(recs)
