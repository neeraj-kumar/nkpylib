"""A simple tag database using Pony ORM and SQLite.

This is used throughout embeddings code for various purposes.

In general, for your project, you should create one of these and shove your data into it, as then
you can use all the various utils provided by this library to evaluate and explore your data (and
particularly features/embeddings).

We assume that a human-readable title is available for each id with key='title'.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from argparse import ArgumentParser
from os.path import abspath, join, dirname, exists

from pony.orm import * # type: ignore
from pony.orm.core import Entity, EntityMeta, SetInstance # type: ignore

from nkpylib.nkpony import GetMixin, recursive_to_dict, sqlite_pragmas

logger = logging.getLogger(__name__)

tag_db = Database() # Pony Tag database


class Tag(tag_db.Entity, GetMixin):
    tag_id = PrimaryKey(int, auto=True)
    id = Required(str, index=True)
    key = Required(str, index=True)
    value = Optional(str, default='')
    type = Optional(str, default='')
    added_ts = Required(int, default=lambda: int(time.time()))
    composite_index(id, type, key, value)
    composite_index(type, key, value)
    composite_index(key, value)

    def __repr__(self):
        return f'<{self.id} {self.type}: {self.key}={self.value}>'

def init_tag_db(path: str) -> Database:
    """Initializes our tag database at given `path`"""
    for func in sqlite_pragmas:
        tag_db.on_connect(provider='sqlite')(func)
    tag_db.bind('sqlite', abspath(path), create_db=True)
    #set_sql_debug(True)
    tag_db.generate_mapping(create_tables=True)
    return tag_db


def get_all_tags() -> dict[str, list[Tag]]:
    """Returns all tags, grouped by id and in lists, in a dictionary"""
    # group tags by id
    tags = {}
    with db_session:
        for tag in Tag.select():
            tags.setdefault(tag.id, []).append(dict(key=tag.key, value=tag.value, type=tag.type))
    return tags
