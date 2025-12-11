"""Generic filesystem tree comparison.

This module contains a generic tree comparison class that can be used to compare two filesystem-like
trees, generate a list of diffs, and (optionally) apply the diffs to bring them into sync. These
trees can be physically located in different kinds of systems, including local disk, airtable, or
chroma.

Each tree is represented as a list of keys (which are just strings). The keys are assumed to be
unique identifiers for the files (or objects) in the tree. The tree can also compute hashes for each
key, which are used to follow files that have been renamed, moved, or copied.

Generically, usage typically looks like this:

t1 = Tree(...)               # "before" state
t2 = Tree(...)               # "after" state
diffs = t1.compare(t2)       # generate diffs
t1.execute_diffs(diffs, t2)  # apply diffs onto t1 to bring it in sync with t2

I've used this for the following kinds of tasks:
- Bringing one directory tree on disk in sync with another
- Updating airtable with new files or changes to files on disk
- Updating chroma with new files or changes to files on disk
- Updating chroma with new rows or changes in airtable
"""

from __future__ import annotations

import functools
import json
import logging
import os

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256, md5
from typing import Optional, Iterable, Any, Callable

from tqdm import tqdm

from nkpylib.airtable import airtable_all_rows, airtable_api_call
from nkpylib.chroma import ChromaUpdater

logger = logging.getLogger(__name__)

class DiffType(str, Enum):
    ADD = 'ADD'
    DELETE = 'DELETE'
    MOVE = 'MOVE'
    COPY = 'COPY'


@dataclass
class Diff:
    """Represents a single diff between two tree listings.

    Has a `type` and two keys, `a` and `b`. If `a` is None, then this is an ADD. If `b` is None,
    then this is a DELETE. For MOVE or COPY and `a` is the original key and `b` is the new key.
    """
    type: DiffType
    a: Optional[str] # None for ADD
    b: Optional[str] # None for DELETE

    @classmethod
    def group_diffs(cls, diffs: Iterable[Diff]) -> list[list[Diff]]:
        """Groups sequential diffs of the same type together.

        Returns a list of lists of diffs.
        """
        ret = []
        cur: list[Diff] = []
        for d in diffs:
            if not cur:
                cur.append(d)
            else:
                if d.type == cur[-1].type:
                    cur.append(d)
                else:
                    ret.append(cur)
                    cur = []
                    cur.append(d)
        if cur:
            ret.append(cur)
        return ret

def incr_iter(it: Iterable[Any], func: Callable[[list[Any]], Any], incr: int) -> Iterable[Any]:
    """A generic function to run a batched function over individual items.

    Items generated from `it` are passed to `func` in batches of `incr`.
    The function is called one more time at the end if there are any leftover items.

    We yield the results of each call to `func`.
    """
    to_do = []
    for item in it:
        to_do.append(item)
        if len(to_do) >= incr:
            yield func(to_do)
            to_do = []
    if to_do:
        yield func(to_do)

@functools.cache
def hash_path(path: str, hash_constructor=sha256) -> str:
    """Returns the sha256 hash of a file."""
    with open(path, 'rb') as f:
        # read it in chunks to avoid memory issues
        h = hash_constructor()
        while chunk := f.read(4096):
            h.update(chunk)
    return h.hexdigest()


class Tree(ABC):
    """Represents a file listing as a tree and allows for some common operations on them.

    At base, this contains:
    - a list of `keys` (which are just strings).
    - a dict of `hash_by_key` (computed lazily as needed).

    """
    def __init__(self, keys: list[str]):
        self.keys = keys[:]
        self.hash_by_key: dict[str, str] = {}

    def __len__(self):
        return len(self.keys)

    @abstractmethod
    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Hashes the given `keys`."""
        pass

    def hash(self, keys: Iterable[str]) -> Iterable[str]:
        """Returns the hashes of the given `keys`, computing them if necessary.

        This just calls `hash_function` and caches the result.
        """
        todo = [key for key in keys if key not in self.hash_by_key]
        if todo:
            hashes = self.hash_function(todo)
            self.hash_by_key.update(zip(todo, hashes))
        return [self.hash_by_key[key] for key in keys]

    def compare(self, other: Tree, assume_keys_constant=False, skip_dupe_hashes=False) -> list[Diff]:
        """Compares this tree (before) with `other` (after) and returns an ordered list of diffs.

        If `assume_keys_constant` is True, then files with the same keys are assumed to be the same
        (without checking hashes).

        If `skip_dupe_hashes` is True, then skips source files with the same hash. Otherwise, raises
        a ValueError.
        """
        diffs = []
        self_keys_ = set(self.keys)
        other_keys_ = set(other.keys)
        if assume_keys_constant:
            # first filter out the keys that are the same
            common = self_keys_ & other_keys_
            self_keys_ -= common
            other_keys_ -= common
        # hash remaining keys
        self_keys = sorted(self_keys_)
        other_keys = sorted(other_keys_)
        self_hashes = list(self.hash(self_keys))
        other_hashes = list(other.hash(other_keys))
        # we can't handle duplicates in the original for now
        if len(set(self_hashes)) != len(self_hashes):
            # Duplicate hashes in original
            by_hash = defaultdict(list)
            for key, hash in zip(self_keys, self_hashes):
                by_hash[hash].append(key)
            for hash, keys in by_hash.items():
                if len(keys) > 1:
                    print(f"Duplicate hash in original: {hash}: {keys}")
            if skip_dupe_hashes:
                # remove all duplicates with this hash from our list of updates
                self_keys = [key for key, hash in zip(self_keys, self_hashes) if len(by_hash[hash]) == 1]
                self_hashes = [hash for key, hash in zip(self_keys, self_hashes) if len(by_hash[hash]) == 1]
                # also remove from the "other" lists
                other_keys = [key for key, hash in zip(other_keys, other_hashes) if not len(by_hash[hash]) > 1]
                other_hashes = [hash for key, hash in zip(other_keys, other_hashes) if not len(by_hash[hash]) > 1]
            else:
                raise ValueError()
        before_by_hash = {h: k for h, k in zip(self_hashes, self_keys)}
        after_by_hash: dict[str, dict[str, list[str]]] = {h: defaultdict(list) for h in other_hashes}
        for key, hash in zip(other_keys, other_hashes):
            after_by_hash[hash][key].append(key)
        # now we can do the comparison
        for hash, keys in after_by_hash.items(): # type: ignore[assignment]
            if hash in before_by_hash:
                # moves and copies
                before_key = before_by_hash[hash]
                # if we have the same key, remove it
                if before_key in keys:
                    keys.remove(before_key)
                # now if we have keys left, we have a move or copy
                for idx, after_key in enumerate(keys):
                    # mark the first one as a move, the rest as copies
                    diff_type = DiffType.MOVE if idx == 0 else DiffType.COPY
                    diffs.append(Diff(diff_type, before_key, after_key))
            else:
                # new file(s)
                for key in keys:
                    diffs.append(Diff(DiffType.ADD, None, key))
        # now we need to find the deletes
        for hash, key in before_by_hash.items():
            if hash not in after_by_hash:
                diffs.append(Diff(DiffType.DELETE, key, None))
        return diffs

    def execute_diffs(self, diffs, other):
        """Executes a list of a diffs.

        Assuming you have `diffs` from comparing `self` (before) to `other` (after), this method
        "executes" the diffs on `self`. This first groups sequential list of diffs of the same type
        together using `group_diffs()`. It then calls the various `execute_XXX` functions (which
        must be subclassed).
        """
        grouped = Diff.group_diffs(diffs)
        function_by_type = {
                DiffType.ADD: self.execute_add,
                DiffType.DELETE: self.execute_delete,
                DiffType.MOVE: self.execute_move,
                DiffType.COPY: self.execute_copy,
        }
        for group in grouped:
            func = function_by_type[group[0].type]
            func(group, other)

    def execute_add(self, diffs: list[Diff], other: Tree) -> None:
        """Executes ADD operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute ADD not implemented")

    def execute_delete(self, diffs: list[Diff], other: Tree) -> None:
        """Executes DELETE operations from given `diffs` going from `self` to `other`"""
        raise NotImplementedError("execute DELETE not implemented")

    def execute_move(self, diffs: list[Diff], other: Tree) -> None:
        """Executes MOVE operations from given `diffs` going from `self` to `other`"""
        # by default we do a copy followed by a delete
        self.execute_copy(diffs, other)
        self.execute_delete(diffs, other)

    def execute_copy(self, diffs: list[Diff], other: Tree) -> None:
        """Executes COPY operations from given `diffs` going from `self` to `other`"""
        # by default we just do an add
        self.execute_add(diffs, other)


class FileTree(Tree):
    """A tree that represents a file listing (on local disk).

    This sets a `root` directory that the keys are relative to.
    """
    def __init__(self,
                 root: str,
                 keys: list[str] | None=None,
                 hash_constructor: Callable = sha256,
                 read_dir: bool = False,
                 filter_func: Callable|None = None):
        """Initializes this file tree at given `root`.

        If you give a list of keys, then initializes with those keys.
        If you set `read_dir` to True, then gets the file listing from the directory, recursively.
        If you set `filter_func`, then only includes keys for which `filter_func(key)` is True.
        """
        if keys is None:
            keys = []
        super().__init__(keys)
        self.root = root
        self.filter_func = filter_func
        self.hash_constructor = hash_constructor
        if read_dir:
            self.keys = self.get_file_listing(root)

    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Hashes given paths using our hash function"""
        from multiprocessing.pool import Pool
        if len(keys) > 10:
            with Pool() as pool:
                func = functools.partial(hash_path, hash_constructor=self.hash_constructor)
                return pool.map(func, [os.path.join(self.root, key) for key in keys])
        else:
            return [hash_path(os.path.join(self.root, key), hash_constructor=self.hash_constructor) for key in keys]

    def get_file_listing(self, dir: str) -> list[str]:
        """Returns a list of all files (no dirs) in a directory, recursively."""
        ret = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                path = os.path.join(root, file)
                if self.filter_func is None or self.filter_func(path):
                    ret.append(os.path.relpath(path, self.root))
        ret.sort()
        return ret


class ChromaTree(Tree):
    def __init__(self, col, hash_key: str, add_func: Optional[Callable]=None, incr: int=10, debug: bool=False):
        """Initialize this chroma tree with the given chroma collection.

        It assumes the keys are the ids from the collection.

        It requires the collection to have hashes in the metadata for items (possibly not all filled
        out), stored under name `hash_key`.

        The `add_func` is called when we detect a new key in the "other" tree we compare this
        against. It is called with `key` (from the other tree) and `other` parameters. It should
        contain:
        - embedding: the embedding to use for this item
        - document: the document to use for this item
        - any other metadata you want to store

        The `incr` is the number of items updated/added at a time.
        If you set `debug` to True, then no changes will be made.
        """
        self.col = col
        self.hash_key = hash_key
        self.add_func = add_func
        self.incr = incr
        self.debug = debug
        keys = col.get(include=[])['ids']
        super().__init__(keys)

    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Returns the hash values for the given `keys`"""
        keys = list(keys)
        resp = self.col.get(ids=keys, include=['metadatas'])
        ids, mds = resp['ids'], resp['metadatas']
        # order them based on the keys
        mds = {id: md for id, md in zip(ids, mds)}
        ret = [mds[key].get(self.hash_key, '') for key in keys]
        #print(f'For keys {keys[-5:]} got hashes {ret[-5:]}')
        return ret

    def execute_add(self, diffs: list[Diff], other: Tree) -> None:
        """Executes ADD operations from given `diffs` going from `self` to `other`.

        This adds the given keys to chroma, including all relevant metadata.
        """
        assert self.add_func is not None
        updater = ChromaUpdater(col=self.col, item_incr=self.incr, debug=self.debug)
        for d in tqdm(diffs):
            try:
                md = self.add_func(key=d.b, other=other)
            except Exception:
                #raise #TODO for debugging
                continue
            updater.add(
                id=d.b,
                embedding=md.pop('embedding'),
                document=md.pop('document'),
                metadata=md,
            )

    def execute_delete(self, diffs: list[Diff], other: Tree) -> None:
        """Executes DELETE operations from given `diffs` going from `self` to `other`

        This does a straightforward chroma delete.
        """
        logger.info(f'Deleting {len(diffs)} ids: {[d.a for d in diffs]}')
        if not self.debug:
            self.col.delete(ids=[d.a for d in diffs])


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
    parser = ArgumentParser(description="Compare two states of a directory and generate a diff.")
    parser.add_argument("dir_a", help="'Before' directory")
    parser.add_argument("dir_b", help="'After' directory")
    args = parser.parse_args()
    fs1 = FileTree(args.dir_a, read_dir=True)
    fs2 = FileTree(args.dir_b, read_dir=True)
    diffs = fs1.compare(fs2)
    print(f"Got {len(diffs)} diffs: {diffs}")
