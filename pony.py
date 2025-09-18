"""Some utilities to help with PonyORM"""

from __future__ import annotations

import logging

from pony.orm import * # type: ignore
from pony.orm.core import Entity, EntityMeta # type: ignore

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
    """Recursively convert a PonyORM Entity to a dictionary.
    """
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

