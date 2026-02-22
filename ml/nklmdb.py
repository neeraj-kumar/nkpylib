"""Various LMDB-related classes.

- PickleableLmdb: Base LMDB database class with pickling support
- JsonLmdb: LMDB database for JSON-serialized data
- MetadataLmdb: LMDB with additional metadata support
- NumpyLmdb: Efficient storage of numpy arrays in LMDB
"""

from __future__ import annotations

import json
import logging
import os
import sys

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from os.path import abspath, dirname, exists, join
from typing import Any, cast, Iterator

import numpy as np

from lmdbm import Lmdb
from tqdm import tqdm

from nkpylib.ml.client import chunked, embed_image, embed_text
from nkpylib.utils import specialize
from nkpylib.thread_utils import CollectionUpdater
from nkpylib.fs_tree import Tree

logger = logging.getLogger(__name__)

try:
    CONSOLE_WIDTH = os.get_terminal_size().columns
except Exception:
    CONSOLE_WIDTH = 120
np.set_printoptions(suppress=True, linewidth=CONSOLE_WIDTH)

class PickleableLmdb(Lmdb):
    """Adds pickling support to Lmdb databases.

    This just adds __getstate__ and __setstate__ methods that store the path and flag, and reopen
    the database on unpickling.
    """
    def fixme__getstate__(self) -> dict[str, Any]:
        """Returns state of this suitable for pickling.

        This just returns a dict with `path` and `flag`.
        """
        ret = dict(
            path=self.env.path(),
            flag=self.flag,
            map_size=self.map_size,
            mode=self.env.mode,
        )
        logger.info(f'Pickling LMDB with state: {ret}')
        sys.exit()
        return ret

    def fixme__setstate__(self, state: dict[str, Any]) -> None:
        """Sets state of this from given `state` dict.

        This simply reruns initialization with the given path and flag.
        """
        db = self.__class__.open(state['path'], state['flag'], map_size=state['map_size'])
        self.env = db.env
        self.autogrow = db.autogrow


class JsonLmdb(PickleableLmdb):
    """Keys are utf-8 encoded strings, values are JSON-encoded objects as utf-8 encoded strings."""
    def _pre_key(self, key):
        return key.encode("utf-8")

    def _post_key(self, key):
        return key.decode("utf-8")

    def _pre_value(self, value):
        return json.dumps(value).encode("utf-8")

    def _post_value(self, value):
        return json.loads(value.decode("utf-8"))


class MetadataLmdb(PickleableLmdb):
    """Subclass of LMDB database that stores JSON metadata for each key inside a 2nd database.

    In the main database, the keys are utf-8 encoded strings, and the values are arbitrary objects.
    (Override _pre_value and _post_value as needed in your subclass.)

    We add a few new methods, all prefixed with md_ (for metadata):
    - md_get(key): Get metadata for given key.
    - md_set(key, **kw): Set (replace) metadata for given key.
    - md_update(key, **kw): Update metadata for given key, keeping existing values.
    - md_batch_set(dict): Set multiple metadata entries at once (replacing existing values).
    - md_delete(key): Delete metadata for given key.
    - md_iter(): Iterate over all metadata keys and values.
    - md_iter_all(): Iterate over (key, value, metadata) triplets

    You can also access the metadata database directly via the `md_db` property, e.g. to check for
    existence of certain keys, etc.

    There's also a property `global_key` which is a special key name used for global metadata.

    Note that the metadata is not indexed, i.e. there's no way to search for keys based on metadata.
    """
    path: str
    md_path: str
    md_db: JsonLmdb

    @classmethod
    def open(cls, file: str, flag: str='r', **kw) -> MetadataLmdb: # type: ignore[override]
        """Opens the main LMDB database at given `file` path.

        Opens a 2nd metadata-only LMDB database inside the first one, with name 'metadata.lmdb',
        with the same `flag` and `kw` as the main db.

        The flag is one of:
        - 'r': read-only, existing
        - 'w': read and write, existing
        - 'c': read and write, create if not exists
        - 'n': read and write, overwrite

        By default, we set `map_size` to 2 ** 30, which is 1 GiB. LMDB only grows up to 12 factors,
        which would be 2 ** 30 * 2 ** 12 = 4 TiB, so this is a reasonable default.

        Note that as the original code says, keeping autogrow=True (the default) means that there
        could be problems with multiple writers.
        """
        if 'map_size' not in kw:
            kw['map_size'] = 2 ** 30 # lmdbm only grows up to 12 factors, and defaults to 2e20
        # make dirs if needed
        if flag != 'r':
            try:
                os.makedirs(dirname(file), exist_ok=True)
            except Exception:
                pass
        ret = cast(MetadataLmdb, super().open(file, flag, **kw))
        ret.path = file
        # now the metadata db
        ret.md_path = join(file, 'metadata.lmdb')
        ret.md_db = cast(JsonLmdb, JsonLmdb.open(ret.md_path, flag=flag, **kw))
        return ret

    def _pre_key(self, key: str) -> bytes:
        return key.encode('utf-8', 'ignore')

    def _post_key(self, key: bytes) -> str:
        return key.decode('utf-8', 'ignore')

    @property
    def global_key(self) -> str:
        """Special key name used for global metadata."""
        return '__global__'

    def md_get(self, key: str) -> dict:
        """Get metadata for given key, or empty dict if not present."""
        try:
            return self.md_db[key]
        except KeyError:
            return {}

    def md_set(self, key: str, **kw) -> None:
        """Set (replace) metadata for given key."""
        self.md_db[key] = kw

    def md_update(self, key: str, **kw) -> None:
        """Update metadata for given key, keeping existing values."""
        md = self.md_get(key)
        md.update(kw)
        self.md_db[key] = md

    def md_batch_set(self, data: dict[str, dict]) -> None:
        """Set multiple metadata entries at once (replacing existing values)."""
        self.md_db.update(data)

    def md_delete(self, key: str) -> None:
        """Delete metadata for given key."""
        try:
            del self.md_db[key]
        except KeyError:
            pass

    def md_iter(self) -> Iterator[tuple[str, dict]]:
        """Iterate over all metadata keys and values."""
        return self.md_db.items()

    def md_iter_all(self) -> Iterator[tuple[str, Any, dict]]:
        """Iterate over (key, value, metadata) triplets.

        Note that this only iterates over keys in the main database.
        If you set metadata on other keys (or the global one), this doesn't include them.
        """
        for key in self:
            yield key, self[key], self.md_get(key)

    def sync(self) -> None:
        """Sync both the main database and the metadata database."""
        super().sync()
        self.md_db.sync()

    def close(self) -> None:
        """Close both the main database and the metadata database."""
        super().close()
        self.md_db.close()

    def __repr__(self):
        return f'MetadataLmdb<{self.path}>'


class NumpyLmdb(MetadataLmdb):
    """Subclass of MetadataLmdb database that stores numpy arrays with utf-8 encoded string keys."""
    dtype: Any
    path: str

    @classmethod
    def open(cls, file: str, flag: str='r', dtype=np.float32, **kw) -> NumpyLmdb: # type: ignore[override]
        """Opens the LMDB database at given `file` path.

        The flag is one of:
        - 'r': read-only, existing
        - 'w': read and write, existing
        - 'c': read and write, create if not exists
        - 'n': read and write, overwrite

        We enforce that all np array values will be of `dtype` type.
        """
        ret = cast(NumpyLmdb, super().open(file, flag, **kw))
        ret.dtype = dtype
        ret.path = file
        return ret

    def _pre_value(self, value: np.ndarray) -> bytes:
        value = np.array(value, dtype=self.dtype)
        assert isinstance(value, np.ndarray), f'Value must be a numpy array, not {type(value)}'
        assert value.dtype == self.dtype, f'Value must be of type {self.dtype}, not {value.dtype}'
        #print(f'set: value of type {type(value)} with shape {value.shape} and dtype {value.dtype}: {value}')
        return value.tobytes()

    def _post_value(self, value: bytes) -> np.ndarray:
        a = np.frombuffer(value, dtype=self.dtype)
        #print(f'get: value of type {type(a)} with shape {a.shape} and dtype {a.dtype}: {a}')
        return a

    def __repr__(self):
        return f'NumpyLmdb<{self.path}>'

    @property
    def n_dims(self):
        """Returns the number of dimensions of the 1st array in the database, or ValueError if empty."""
        for value in self.values():
            return len(value)
        raise ValueError('Database is empty, cannot determine number of dimensions.')

    @classmethod
    def concat_multiple(cls, paths: list[str], output_path: str, dtype=np.float32) -> None:
        """Loads ands concatenates multiple lmdbs from given `paths`, writing to `output_path`.

        This writes a single lmdb with only those keys that are in all files.
        """
        vecs = {}
        for i, path in tqdm(enumerate(paths)):
            cur = NumpyLmdb.open(path, flag='r', dtype=dtype)
            if i == 0:
                vecs = dict(cur.items())
            else:
                cur_keys = set(cur.keys())
                # remove keys that are not in all veceddings
                to_del = set(vecs.keys()) - cur_keys
                for k in to_del:
                    del vecs[k]
                # now concatenate
                for k, existing in vecs.items():
                    vecs[k] = np.hstack([existing, cur[k]])
        # write to output path
        with cls.open(output_path, 'c', dtype=dtype) as db:
            #db.update({key: vec for key, vec in vecs.items()})
            db.update(vecs)

class LmdbUpdater(CollectionUpdater):
    """Helper class to update an LMDB database in parallel.

    This is a subclass of CollectionUpdater that takes an LMDB path as input.

    Use like this:

        updater = LmdbUpdater(db_path, NumpyLmdb.open, n_procs=4, batch_size=1000)
        updater.add('id1', embedding=np.array([...]))
        updater.add('id2', metadata=dict(key='value'))
        updater.add('id3', embedding=np.array([...]), metadata=dict(key='value'))

    """
    def __init__(self,
                 db_path: str,
                 init_fn: Callable=NumpyLmdb.open,
                 map_size: int=2**32,
                 n_procs: int=1,
                 **kw):
        """Initialize the updater with the given db_path and `init_fn` (to pick which flavor)

        If you specify `debug=True`, then commit messages will be printed using logger.info()

        All other `kw` are passed to the CollectionUpdater constructor.
        """
        global db, my_id
        super().__init__(add_fn=self._add_fn, **kw)
        if n_procs > 1:
            self.pool = ProcessPoolExecutor(max_workers=n_procs, initializer=self.init_worker, initargs=(db_path, init_fn, map_size))
        else:
            self.pool = None
            db = init_fn(db_path, flag='c', map_size=map_size, autogrow=True)
            my_id = 'main'
        self.futures = []

    @staticmethod
    def init_worker(db_path: str, init_fn: Callable, map_size: int=2**32) -> None:
        """Initializes a worker process with the given db_path and init_fn."""
        global db, my_id
        my_id = os.getpid()
        logger.debug(f'In child {my_id}: initializing db at {db_path} with {init_fn} and map_size {map_size}')
        db = init_fn(db_path, flag='c', map_size=map_size, autogrow=False)

    def add(self, id: str, embedding=None, metadata=None):
        """Add an item with given `id`, `embedding` (np array) and/or `metadata` (dict)."""
        obj: dict[str, Any] = {}
        if embedding is not None:
            obj['embedding'] = embedding
        if metadata is not None:
            obj['metadata'] = metadata
        #print(f'Added id {id} with md {metadata}')
        super().add(id, obj)

    def __contains__(self, key: str) -> bool:
        """Returns True if the given `key` is in the database.

        This calls key_in_db in a worker process.
        """
        if self.pool:
            future = self.pool.submit(self.key_in_db, key)
            return future.result()
        else:
            return self.key_in_db(key)

    @staticmethod
    def key_in_db(key: str) -> bool:
        return key in db

    @staticmethod
    def set_emb(id, emb):
        db.__setitem__(id, np.array(emb, dtype=db.dtype))

    @staticmethod
    def set_md(id, md):
        db.md_set(id, **md)

    @staticmethod
    def db_sync():
        db.sync()
        return my_id

    @staticmethod
    def batch_add(ids, objects):
        md_updates = {}
        emb_updates = {}
        for id, obj in zip(ids, objects):
            md = obj.get('metadata', None)
            if md is not None:
                md_updates[id] = md
            emb = obj.get('embedding', None)
            if emb is not None:
                emb_updates[id] = np.array(emb, dtype=db.dtype)
        if md_updates:
            logger.debug(f'Setting {len(md_updates)} metadata entries in process {my_id}: {list(md_updates.items())[:3]}...')
            db.md_batch_set(md_updates)
        if emb_updates:
            logger.debug(f'Setting {len(emb_updates)} embeddings in db {(db, len(db))}: {list(emb_updates.items())[:3]}...')
            try:
                db.update(emb_updates)
            except Exception as e:
                logger.error(f'Error updating embeddings in db {db}: {e}', exc_info=True)
                raise
        db.sync()

    def _add_fn(self, to_add: dict[str, list]) -> None:
        if self.pool:
            future = self.pool.submit(self.batch_add, to_add['ids'], to_add['objects'])
            self.futures.append(future)
        else:
            self.batch_add(to_add['ids'], to_add['objects'])

    def sync(self):
        """Makes sure all pending writes are done."""
        logger.info(f'Calling sync')
        if self.pool:
            for f in self.futures:
                f.result()
            self.futures = []


class LmdbTree(Tree):
    def __init__(self,
                 path: str,
                 add_func: Callable[[list[str], Tree], dict[dict[str, Any]]],
                 prefix: str='',
                 hash_key: str='hash',
                 db_cls: type[NumpyLmdb]|type[JsonLmdb]=NumpyLmdb,
                 debug: bool=False,
                 **init_kw):
        """Initialize this LMDB tree with db of type `db_cls` at the given `path`.

        We open the db using the `db_cls.open` classmethod with mode 'c'. Any extra `init_kw` are
        passed to that function. We currently allow for 2 types of db:
        - NumpyLmdb: We assume the values are numpy arrays and other fields are stored in the
          metadata
        - JsonLmdb: We assume the values are JSON-serializable objects with all relevant fields.

        The `add_func` is a function that takes a list of keys to add and the `other` Tree, and
        returns a dict of dicts, mapping each key to a dict with the following fields:
        - 'embedding': np.ndarray [only if db_cls is NumpyLmdb]
        - 'metadata': dict of other metadata fields to store

        This requires the db to have hashes in the metadata for items (possibly not all filled
        out), stored under field `hash_key` (either in the metadata db for NumpyLmdb, or in the
        main db for JsonLmdb).

        If you set `debug` to True, then no changes will be made.
        """
        self.path = path
        self.hash_key = hash_key
        self.prefix = prefix
        self.add_func = add_func
        self.db_cls = db_cls
        self.init_kw = init_kw
        db = self.load_db()
        self.debug = debug
        keys = [key.replace(prefix, '', 1) for key in db.keys() if key.startswith(prefix)]
        super().__init__(keys)

    def load_db(self) -> JsonLmdb|NumpyLmdb:
        """Returns a fresh LMDB database object for this tree."""
        return self.db_cls.open(self.path, flag='c', **self.init_kw)

    def hash_function(self, keys: Iterable[str]) -> Iterable[str]:
        """Returns the hash values for the given `keys`"""
        db = self.load_db()
        if isinstance(self.db_cls, NumpyLmdb):
            hashes = [db.md_get(self.prefix+key).get(self.hash_key) for key in keys]
        elif isinstance(self.db_cls, JsonLmdb):
            hashes = [db[self.prefix+key].get(self.hash_key, '') for key in keys]
        else:
            raise NotImplementedError(f'Unsupported db class {self.db_cls}')
        return hashes

    def execute_add(self, diffs: list[Diff], other: Tree) -> None:
        """Executes ADD operations from given `diffs` going from `self` to `other`.

        This adds the given keys to chroma, including all relevant metadata.
        """
        assert self.add_func is not None
        to_add = self.add_func([d.b for d in diffs], other)
        db = self.load_db()
        if isinstance(db, NumpyLmdb):
            main_to_add = {}
            md_to_add = {}
            for key, item in to_add.items():
                if 'embedding' in item:
                    main_to_add[self.prefix+key] = item.pop('embedding')
                if 'metadata' in item:
                    md_to_add[self.prefix+key] = item.pop('metadata')
            logger.info(f'Adding {len(main_to_add)} embeddings and {len(md_to_add)} metadata entries to {db}.')
            if not self.debug:
                if main_to_add:
                    db.update(main_to_add)
                if md_to_add:
                    db.md_batch_set(md_to_add)
                db.sync()
        elif isinstance(db, JsonLmdb):
            logger.info(f'Adding {len(to_add)} entries to {db}.')
            if not self.debug:
                db.update(to_add)
                db.sync()

    def execute_delete(self, diffs: list[Diff], other: Tree) -> None:
        """Executes DELETE operations from given `diffs` going from `self` to `other`

        This deletes the list of keys from our db. If our db is a subclass of MetadataLmdb, we also
        delete the metadata at those keys.
        """
        logger.info(f'Deleting {len(diffs)} ids: {[d.a for d in diffs]}')
        db = self.load_db()
        if not self.debug:
            for d in diffs:
                key = self.prefix + d.a
                if key in db:
                    del db[key]
                if isinstance(db, MetadataLmdb) and key in db.md_db:
                    db.md_delete(key)
            db.sync()


def quick_test(path: str, n: int=3, cols: int=-1, show_md:bool=False, q:str='', **kw):
    """Prints the first `n` embeddings from given `path` (assumed to be NumpyLmdb).

    We print information about the db and metadata to stderr, and then the first `n` entries to
    stdout, with key followed by tab followed by space-separated values, for ease of
    parsing/manipulating downstream.

    Params:
    - 'cols': if a positive integer, limit the number of columns printed per embedding.
    - 'show_md': if True, prints the metadata for each entry.
    - 'q': if given, searches for the given str within keys and only prints those that match.
    """
    if not isinstance(q, str):
        q = str(q)
    db = NumpyLmdb.open(path)
    num = 0
    MB = 1024*1024
    first = None
    for first in db.values():
        break
    if first is None:
        first = []
        mem = 0
    else:
        mem = len(db) * len(first) * first.dtype.itemsize // MB
    print(f'Opened {db}: {len(db)} x {len(first)} ({db.dtype.__name__}) = {mem}MB => {db.map_size//MB}MB map size.', file=sys.stderr)
    if n_md := len(db.md_db):
        g = db.md_db.get(db.global_key, None)
        print(f'  Got {n_md} metadata entries, global: {g}', file=sys.stderr)
    for key, value in db.items():
        if q and q not in key:
            continue
        val_s = ' '.join(f'{v:.4f}' for v in value[:cols])
        s = f'{key}\t{val_s}'
        if show_md:
            md = db.md_get(key)
            s += '\t' + ','.join(f'{k}={v}' for k, v in md.items())
        print(s)
        num += 1
        if num >= n:
            break

def batch_extract_embeddings(inputs: list,
                             db_path: str,
                             embedding_type: str='text',
                             flag: str='c',
                             model: str|None=None,
                             batch_size=200,
                             md_func: Callable[[str, Any], dict]|None=None,
                             skip_existing: bool=True,
                             use_cache: bool=False,
                             **kw) -> int:
    """Core function for batch extracting embeddings to an LMDB.

    This uses the LmdbUpdater to parallelize the extraction and writing of embeddings to the LMDB at
    `db_path`, in batches of `batch_size`. The `embedding_type` is either 'text' or 'image',
    determining which embedding to use. By default, for text we use model 'qwen_emb', and for image
    we use 'clip'. You can override this by specifying the `model` argument.

    If the `embedding_type` is text, then the inputs should have 2 elements each, the key and the
    text. If 'image', then they can either have a single string element each (the path, which will also be used as the key), or (key, path).

    You can optionally provide a `md_func` that takes a `(key, input)` pair and returns a dict of
    metadata to write alongside the embedding.

    We return the number of new embeddings written to the database.
    """
    if not inputs:
        return 0
    default_models = dict(image='clip', text='qwen_emb')
    assert embedding_type in default_models
    model = model or default_models[embedding_type]
    # convert inputs to (key, object) for images
    if embedding_type == 'image':
        inputs = [((inp, inp) if isinstance(inp, str) else inp) for inp in inputs]
    db = NumpyLmdb.open(db_path, flag=flag)
    logger.info(f'Got {len(db)} keys in db {db.path} with dtype {db.dtype} and flag {flag}, {len(inputs)} new inputs to process: {inputs[:3]}.')
    dtype = db.dtype
    # filter inputs to only those not already in the db (if skip_existing)
    def filter_func(row):
        if not skip_existing:
            return True
        return row[0] not in db

    inputs = list(filter(filter_func, inputs))
    del db
    if not inputs:
        return 0
    updater = LmdbUpdater(db_path, NumpyLmdb.open, n_procs=4)
    num = 0
    # create a progress bar we can manually move
    bar = tqdm(total=len(inputs), desc=f'Extracting {embedding_type} features using {model}', unit='input')
    # process inputs in batches
    for batch in chunked(inputs, batch_size):
        if not batch:
            continue
        if embedding_type == 'text':
            keys, texts = zip(*batch)
            futures = embed_text.batch_futures(list(texts), model=model, use_cache=use_cache)
        elif embedding_type == 'image':
            keys, paths = zip(*batch)
            futures = embed_image.batch_futures([abspath(path) for path in paths], model=model, use_cache=use_cache)
        else:
            raise ValueError(f'Unsupported embedding_type: {embedding_type}')
        for input, future in zip(batch, futures):
            key, obj = input
            try:
                emb = future.result()
                if emb is None:
                    logger.warning(f'No embedding for {input}')
                    continue
                #db[key] = np.array(emb, dtype=db.dtype)
                to_add = dict(embedding=np.array(emb, dtype=dtype))
                if md_func is not None:
                    to_add['metadata'] = md_func(key, obj)
                updater.add(key, **to_add)
                num += 1
                bar.update(1)
            except Exception as e:
                logger.error(f'Error extracting {input}: {e}', exc_info=True)
    updater.commit()
    return num

def extract_embeddings(path: str, embedding_type: str='text', **kw):
    """Does embedding feature extraction with given `model`.

    Reads inputs from stdin (one per line) and writes the embeddings to the given `path`.
    These are determined by `embedding_type` (either 'image' or 'text').

    If 'image', then we use the image path as the key. If text, then we assume it's tab-separated
    columns, with key as first one and text to embed as second.
    """
    # parse inputs from stdin
    inputs = []
    for line in sys.stdin:
        input = line.strip()
        if embedding_type == 'text':
            parts = input.split('\t', 1)
            assert len(parts) == 2
            key, text = parts
            inputs.append(parts)
        else:
            inputs.append(input)
    num = batch_extract_embeddings(inputs, db_path=path, embedding_type=embedding_type, **kw)
    print(f'Wrote {num} items to {path}.')


if __name__ == '__main__':
    funcs = {f.__name__: f for f in [quick_test, extract_embeddings]}
    parser = ArgumentParser(description='Test feature sets.')
    parser.add_argument('func', choices=funcs, help='Function to run')
    parser.add_argument('path', help='Path to the embeddings lmdb file')
    parser.add_argument('-f', '--flag', default='c', choices=['r', 'w', 'c', 'n'],
                        help='Flag to open the lmdb file (default: r)')
    parser.add_argument('keyvalue', nargs='*', help='Key=value pairs to pass to the function')
    args = parser.parse_args()
    kwargs = vars(args)
    for keyvalue in kwargs.pop('keyvalue', []):
        if '=' not in keyvalue:
            raise ValueError(f'Invalid key=value pair: {keyvalue}')
        key, value = keyvalue.split('=', 1)
        value = specialize(value)
        kwargs[key] = value
    func = funcs[kwargs.pop('func')]
    func(**kwargs) # type: ignore[operator]
