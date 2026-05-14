"""Some utilities to help with PonyORM"""

from __future__ import annotations

import logging
import os

from os.path import abspath, dirname

from pony.orm import * # type: ignore
from pony.orm.core import Entity, EntityMeta, SetInstance # type: ignore
from tqdm import tqdm # type: ignore

logger = logging.getLogger(__name__)

class GetMixin():
    """Helper mixin for getting or creating an object"""
    @classmethod
    def get_or_create(cls, flush=False, **params):
        o = cls.get(**params)
        if o:
            return o
        ret = cls(**params)
        if flush:
            ret.flush()
        return ret

    @classmethod
    def create_or_get(cls, get_kw: dict[str, Any], **create_kw) -> Entity:
        """Creates a new object or gets existing.

        If an object exists with the `get_kw` dictionary, returns it. Otherwise, creates a new
        object with the `get_kw` and `create_kw` dictionaries and returns it.

        Note that this is different from `get_or_create` in that it allows you to specify different
        parameters for getting and creating. It is also different from `upsert` in that it does not
        update existing objects, it only gets or creates.
        """
        assert isinstance(cls, EntityMeta), f"{cls} is not a database entity"
        if cls.exists(**get_kw):
            return cls.get(**get_kw)
        else:
            logger.debug(f'Creating new {cls.__name__} with {get_kw} {create_kw}')
            return cls(**create_kw, **get_kw)

    @classmethod
    def upsert(cls, get_kw: dict[str, Any], **set_kw: Any) -> Entity:
        """Upserts into the database.

        First tries to lookup the object using the `get_kw` dictionary. If it exists, it updates the
        object with the `set_kw` dictionary. If it doesn't exist, it creates a new object with the
        `get_kw` and `set_kw` dictionaries.

        Returns the object.
        """
        assert isinstance(cls, EntityMeta), f"{cls} is not a database entity"
        if not cls.exists(**get_kw):
            # make new object
            logger.debug(f'Creating new {cls.__name__} with {get_kw} {set_kw}')
            return cls(**set_kw, **get_kw)
        else:
            # get the existing object
            logger.debug(f'Updating existing {cls.__name__} with {get_kw} {set_kw}')
            obj = cls.get(**get_kw)
            for key, value in set_kw.items():
                obj.__setattr__(key, value)
            return obj

    @classmethod
    def count(cls):
        """Returns the total count of this table"""
        return count(o for o in cls)


# Various SQLite pragmas to improve performance
# You can import the `sqlite_pragmas` list and apply them to your db:
#    for func in sqlite_pragmas:
#        db.on_connect(provider='sqlite')(func)
def sqlite_longer_timeout(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA busy_timeout = 100000')

def sqlite_journal_wal(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA journal_mode = WAL')

def sqlite_synchronous_normal(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA synchronous = NORMAL')

def sqlite_large_cache(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA cache_size = 1000000')  # 1 million pages, ~16GB

def sqlite_temp_memory(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA temp_store = MEMORY')

def sqlite_case_sensitivity(db, connection):
    cursor = connection.cursor()
    cursor.execute('PRAGMA case_sensitive_like = OFF')


sqlite_pragmas = [
    sqlite_longer_timeout,
    sqlite_journal_wal,
    sqlite_synchronous_normal,
    sqlite_large_cache,
    sqlite_temp_memory,
    sqlite_case_sensitivity,
]


@db_session
def recursive_to_dict(obj, _has_iterated=False, **kwargs):
    """Recursively convert a PonyORM Entity to a dictionary."""
    if isinstance(obj, Entity):
        obj = obj.to_dict(**kwargs)
    #print(f'obj: {obj}, {type(obj)} {obj.__class__.__bases__}')
    if isinstance(obj, SetInstance):
        obj = [recursive_to_dict(o, True, **kwargs) for o in obj]
    else:
        delete_these = []
        for key, value in obj.items():
            if _has_iterated:
                if isinstance(value, (list, tuple)):
                    for iterable in value:
                        if isinstance(iterable, Entity):
                            delete_these.append(key)
                            break
                    continue
            else:
                if isinstance(value, (list, tuple)):
                    value_list = []
                    for iterable in value:
                        if isinstance(iterable, Entity):
                            value_list.append(recursive_to_dict(iterable, True, **kwargs))
                    obj[key] = value_list

            if isinstance(value, Entity) and not _has_iterated:
               obj[key] = recursive_to_dict(value, True, **kwargs)
            elif isinstance(value, Entity) and _has_iterated:
                delete_these.append(key)

        for deletable_key in delete_these:
            del obj[deletable_key]

    return obj

def init_sqlite_db(path: str, db: Database|None=None, debug:bool=False) -> Database:
    """Initializes the sqlite database at the given `path`.

    You can optionally pass in your own `db` instance.
    If you set `debug` to True, SQL debugging will be enabled.
    """
    if db is None:
        db = Database()
    for func in sqlite_pragmas:
        db.on_connect(provider='sqlite')(func)
    path = abspath(path)
    try:
        os.makedirs(dirname(path), exist_ok=True)
    except Exception as e:
        pass
    try:
        db.bind('sqlite', path, create_db=True)
        set_sql_debug(debug)
        db.generate_mapping(create_tables=True)
    except BindingError:
        pass
    return db

def batch_query(query, batch_size=1000):
    """Fetches results from a PonyORM `query` in batches, as a generator.

    Note that this yields entire batches. If you want individual rows, use `batch_query_iter`.
    """
    page = 1
    while True:
        batch = query.page(page, batch_size)
        if not batch:
            break
        yield batch
        page += 1
        # Clear Pony's cache to prevent memory buildup
        #local_cache.clear()

def batch_query_iter(query, batch_size=1000):
    """Fetches individual rows from a PonyORM `query` (run in batches), as an iterator."""
    for batch in batch_query(query, batch_size):
        for item in batch:
            yield item

def sql_batch_generator(db, sql, params=None, batch_size=10000, desc="Loading data", max_num=0):
    """Generic raw SQL batch data generator.

    - db: database connection
    - sql: SQL query string
    - params: optional query parameters
    - batch_size: number of rows per batch
    - desc: description for progress bar
    - max_num: maximum number of rows to fetch (if > 0)

    Yields rows from the query in batches.
    """
    offset = 0
    params = params or {}
    n = 0
    with tqdm(desc=f"{desc} in batches of {batch_size}", total=max_num) as pbar:
        while True:
            batch_sql = f"{sql} LIMIT {batch_size} OFFSET {offset}"
            with db_session:
                batch_results = db.select(batch_sql, params)
            if not batch_results:
                break
            for row in batch_results:
                n += 1
                yield row
            if max_num > 0 and n >= max_num:
                break
            offset += batch_size
            pbar.update(len(batch_results))
            if len(batch_results) < batch_size:
                break
