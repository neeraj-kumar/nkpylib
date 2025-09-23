"""Groups of features put together, as well as lmdb-based feature storage and retrieval.

Storage:
- JsonLmdb: LMDB database for JSON-serialized data
- MetadataLmdb: LMDB with additional metadata support
- NumpyLmdb: Efficient storage of numpy arrays in LMDB

Feature Management:
- FeatureSet: Collection of features with similarity search capabilities
- Support for multiple input sources (files or mapping objects)
- Automatic key intersection across sources

"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time

from argparse import ArgumentParser
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from os.path import abspath, dirname, exists, join
from typing import Any, cast, Sequence, Generic, TypeVar, Iterator

import numpy as np

from lmdbm import Lmdb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nkpylib.ml.client import chunked, embed_image, embed_text
from nkpylib.utils import specialize
from nkpylib.thread_utils import CollectionUpdater

nparray1d = np.ndarray
nparray2d = np.ndarray

array1d = nparray1d | Sequence[float]
array2d = nparray2d | Sequence[Sequence[float]]

logger = logging.getLogger(__name__)

try:
    CONSOLE_WIDTH = os.get_terminal_size().columns
except Exception:
    CONSOLE_WIDTH = 120
np.set_printoptions(suppress=True, linewidth=CONSOLE_WIDTH)

class JsonLmdb(Lmdb):
    """Keys are utf-8 encoded strings, values are JSON-encoded objects as utf-8 encoded strings."""
    def _pre_key(self, key):
        return key.encode("utf-8")

    def _post_key(self, key):
        return key.decode("utf-8")

    def _pre_value(self, value):
        return json.dumps(value).encode("utf-8")

    def _post_value(self, value):
        return json.loads(value.decode("utf-8"))


class MetadataLmdb(Lmdb):
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

        By default, we set `map_size` to 2 ** 25, which is 32 MiB. LMDB only grows up to 12 factors,
        which would be 2 ** 25 * 2 ** 12 = 16 TiB, so this is a reasonable default.

        Note that as the original code says, keeping autogrow=True (the default) means that there
        could be problems with multiple writers.
        """
        if 'map_size' not in kw:
            kw['map_size'] = 2 ** 25 # lmdbm only grows up to 12 factors, and defaults to 2e20
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
                 init_fn: Callable,
                 map_size: int=2e32,
                 n_procs: int=4,
                 **kw):
        """Initialize the updater with the given db_path and `init_fn` (to pick which flavor)

        If you specify `debug=True`, then commit messages will be printed using logger.info()

        All other `kw` are passed to the CollectionUpdater constructor.
        """
        self.pool = ProcessPoolExecutor(max_workers=n_procs, initializer=self.init_worker, initargs=(db_path, init_fn, map_size))
        super().__init__(add_fn=self._add_fn, **kw)
        self.futures = []

    @staticmethod
    def init_worker(db_path: str, init_fn: Callable, map_size: int=2**32) -> None:
        """Initializes a worker process with the given db_path and init_fn."""
        global db, my_id
        my_id = os.getpid()
        print(f'In child {my_id}: initializing db at {db_path} with {init_fn} and map_size {map_size}', flush=True)
        db = init_fn(db_path, flag='c', map_size=map_size, autogrow=False)

    def add(self, id: str, embedding=None, metadata=None):
        """Add an item with given `id`, `embedding` (np array) and/or `metadata` (dict)."""
        obj: dict[str, Any] = {}
        if embedding is not None:
            obj['embedding'] = embedding
        if metadata is not None:
            obj['metadata'] = metadata
        #print(f'Added id {id} with obj {obj}')
        super().add(id, obj)

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
            db.md_batch_set(md_updates)
        if emb_updates:
            db.update(emb_updates)
        db.sync()

    def _add_fn(self, to_add: dict[str, list]) -> None:
        future = self.pool.submit(self.batch_add, to_add['ids'], to_add['objects'])
        self.futures.append(future)

    def sync(self):
        """Makes sure all pending writes are done."""
        print(f'Calling sync')
        for f in self.futures:
            f.result()
        self.futures = []


KeyT = TypeVar('KeyT')


class FeatureSet(Mapping, Generic[KeyT]):
    """A set of features that you can do stuff with.

    It is accessible as a mapping of `KeyT` to `np.ndarray`.

    The inputs should be a list of mapping-like objects, or paths to numpy-encoded lmdb files.
    """
    def __init__(self, inputs: list[Any], dtype=np.float32, **kw):
        """Loads features from given list of `inputs`.

        The inputs should either be mapping-like objects, or paths to numpy-encoded lmdb files.
        We compute the intersection of the keys in all inputs, and use that as our list of _keys.
        """
        # remap any path inputs to NumpyLmdb objects
        self.inputs = [NumpyLmdb.open(inp, flag='r', dtype=dtype) if isinstance(inp, str) else inp
                       for inp in inputs]
        self._keys = self.get_keys()
        self.n_dims = 0
        for key, value in self.items():
            self.n_dims = len(value)
            break
        self.cached: dict[str, Any] = dict()

    def get_keys(self) -> list[KeyT]:
        """Gets the intersection of all keys by reading all our inputs.

        Useful if the underlying databases might change over time.
        Note that we make no guarantees on correctness due to changing databases!
        """
        keys = []
        for i, inp in enumerate(self.inputs):
            if i == 0:
                keys = list(inp.keys())
            else:
                cur_keys = set(inp.keys())
                keys = [k for k in keys if k in cur_keys]
        return keys

    def __iter__(self) -> Iterator[KeyT]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __getitem__(self, key: KeyT) -> np.ndarray:
        return np.hstack([inp[key] for inp in self.inputs])

    def get_keys_embeddings(self,
                            keys: list[KeyT]|None=None,
                            normed: bool=False,
                            scale_mean:bool=True,
                            scale_std:bool=True) -> tuple[list[KeyT], np.ndarray]:
        """Returns a list of keys and a numpy array of embeddings.

        By default we return embeddings for all our keys, but you can optionally pass in a list of
        keys to get embeddings for.

        You can optionally set the following flags:
        - `normed`: Normalize embeddings to unit length.
        - `scale_mean`: Scale embeddings to have zero mean.
        - `scale_std`: Scale embeddings to have unit variance.

        Note that the normalization is applied only to the set of keys you fetch embeddings for, so
        it might be degenerate if you request too few keys.

        The keys and embeddings are cached for future calls with the same flags (only if requesting
        all keys).
        """
        if keys is None:
            cache_kw = dict(normed=normed, scale_mean=scale_mean, scale_std=scale_std)
            if self.cached and all(self.cached[k] == v for k, v in cache_kw.items()):
                return self.cached['keys'], self.cached['embs']
            _keys, _embs = zip(*list(self.items()))
            keys = list(_keys)
            embs = np.vstack(_embs)
        else:
            embs = np.vstack([self[k] for k in keys if k in self])
        scaler: StandardScaler|None = None
        if normed:
            embs = embs / np.linalg.norm(embs, axis=1)[:, None]
        if scale_mean or scale_std:
            scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)
            embs = scaler.fit_transform(embs)
        if len(keys) == len(self):
            # cache these
            self.cached.update(keys=keys, embs=embs, scaler=scaler, **cache_kw)
        return keys, embs


def quick_test(path: str, n: int=3, **kw):
    """Prints the first `n` embeddings from given `path`"""
    db = NumpyLmdb.open(path)
    num = 0
    for key, value in db.items():
        if num == 0:
            print(f'Opened {db}: {len(db)} x {len(value)} ({db.dtype.__name__}) = {db.map_size} map size.')
            if hasattr(db, 'md_db') and n_md := len(db.md_db):
                g = db.md_db.get(db.global_key, None)
                print(f'  Got {n_md} metadata entries, global: {g}')
        print(f'{key}: {value}')
        num += 1
        if num >= n:
            break

def extract_embeddings(path: str, embedding_type: str='text', flag: str='c', model: str|None=None, batch_size=20, **kw):
    """Does embedding feature extraction with given `model`.

    Reads inputs from stdin (one per line) and writes the embeddings to the given `path`.
    These are determined by `embedding_type` (either 'image' or 'text').

    If 'image', then we use the image path as the key. If text, then we assume it's tab-separated
    columns, with key as first one and text to embed as second.
    """
    default_models = dict(image='clip', text='qwen_emb')
    assert embedding_type in default_models
    model = model or default_models[embedding_type]
    db = NumpyLmdb.open(path, flag=flag)
    inputs = []
    for line in sys.stdin:
        input = line.strip()
        if embedding_type == 'text':
            parts = input.split('\t', 1)
            assert len(parts) == 2
            key, text = parts
            if key in db:
                continue
            inputs.append(parts)
        else:
            if input in db:
                continue
            inputs.append(input)
    print(f'Got {len(db)} keys in db {db.path} with dtype {db.dtype} and flag {flag}, {len(inputs)} new inputs to process: {inputs[:3]}.')
    num = 0
    # create a progress bar we can manually move
    bar = tqdm(total=len(inputs), desc=f'Extracting {embedding_type} features using {model}', unit='input')
    # interleave reading files from stdin and processing them, in batches
    for batch in chunked(inputs, batch_size):
        if not batch:
            continue
        if embedding_type == 'text':
            keys, texts = zip(*batch)
            futures = embed_text.batch_futures(list(texts), model=model, use_cache=False)
        elif embedding_type == 'image':
            futures = embed_image.batch_futures([abspath(input) for input in batch], model=model, use_cache=False)
        for input, future in zip(batch, futures):
            try:
                emb = future.result()
                if emb is None:
                    logger.warning(f'No embedding for {input}')
                    continue
                if embedding_type == 'text':
                    key, _text = input
                else:
                    key = input
                db[key] = np.array(emb, dtype=db.dtype)
                num += 1
                bar.update(1)
            except Exception as e:
                logger.error(f'Error extracting {input}: {e}', exc_info=True)
        db.sync()
    db.sync()
    print(f'Wrote {num} items to {db.path}.')


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
