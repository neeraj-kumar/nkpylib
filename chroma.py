"""Some utilities for dealing with chroma"""

#TODO futures feature extraction

from __future__ import annotations

import logging
import re
import threading
import time

from argparse import ArgumentParser
from typing import Any, Callable, NamedTuple
from threading import Lock

import numpy as np

from chromadb import Collection, HttpClient, PersistentClient, Client, EphemeralClient
from tqdm import tqdm

from nkpylib.thread_utils import chained_producer_consumers, CollectionUpdater

logger = logging.getLogger(__name__)

CHROMA_CLIENTS = {}

CHROMA_LOCK = Lock()

def load_chroma_client(db_path: str='', port: int=0):
    """Loads chroma client (if not already loaded) and returns it.

    This function is thread-safe and will only load the client once.

    It will first try to load the client from the server at the given `port`.
    If that fails, it will try to load the client from the given `db_path`.

    If neither are specified, then it will create an Ephemeral (in-memory only) client (non-cached).
    """
    if not port and not db_path: # create an ephemeral client
        return EphemeralClient()
    global CHROMA_CLIENTS
    key = (db_path, port)
    if key not in CHROMA_CLIENTS:
        with CHROMA_LOCK:
            # first try loading from the server
            if port > 0:
                try:
                    CHROMA_CLIENTS[key] = HttpClient(port=port) # type: ignore # this is a bug in an older version of chromadb
                    logger.info(f'Loaded chromadb client from server at port {port}')
                except Exception as e:
                    logger.info(f'chromadb client not running: {e}')
                    pass
            # now try loading from disk
            if db_path:
                logger.info(f'Loading chromadb client at {db_path}')
                t0 = time.time()
                CHROMA_CLIENTS[key] = PersistentClient(path=db_path)
                logger.info(f"Loaded chromadb client in {time.time() - t0:.2f}s")
    return CHROMA_CLIENTS[key]

class ChromaUpdater(CollectionUpdater):
    """A chroma-specific version of `CollectionUpdater`.

    In particular, chroma's add function is clunky, requiring a dict with lists of ids, embeddings,
    etc. This makes sense for efficiency, but is annoying.

    With this class, you specify your update frequency (in number of items and/or elapsed time) in
    the constructor, and then call .add() for each item.
    """
    def __init__(self,
                 col: Collection,
                 item_incr: int=100,
                 time_incr: float=30.0,
                 post_commit_fn: Callable[[list[str]], None]|None=None,
                 debug: bool=False):
        """Initialize the updater with the given collection and update frequency.

        - item_incr: number of items to add before committing [default 100]. (Disabled if <= 0)
        - time_incr: elapsed time to wait before committing [default 30.0]. (Disabled if <= 0)

        Note that if both are specified, then whichever comes first triggers a commit.

        You can optionally pass in a `post_commit_fn` to be called after each commit. It is called
        with the list of ids that were just committed.

        If you specify `debug=True`, then commit messages will be printed using logger.info()
        """
        self.col = col
        super().__init__(add_fn=self._add_fn,
                         item_incr=item_incr,
                         time_incr=time_incr,
                         post_commit_fn=post_commit_fn,
                         debug=debug)

    def _reset_to_add(self) -> dict[str, list]:
        """Resets our batch of items to add."""
        return dict(ids=[], embeddings=[], documents=[], metadatas=[])

    def _add_fn(self, to_add: dict[str, list]) -> None:
        """Batch add to chroma"""
        _to_add = dict(ids=to_add['ids'])
        for field in ['embeddings', 'documents', 'metadatas']:
            if field in to_add:
                _to_add[field] = to_add[field]
        self.col.add(**to_add)

    def add(self, id: str, embedding: np.ndarray|None=None, document: str|None=None, metadata: dict[str, Any]|None=None):
        """Adds an item to the updater.

        If the update frequency is reached, it will commit the items to the collection.
        """
        assert id not in self.ids_seen, f'ID {id} already seen!'
        self.to_add['ids'].append(id)
        self.ids_seen[id] = False
        if embedding is not None:
            self.to_add['embeddings'].append(embedding)
        if document is not None:
            self.to_add['documents'].append(document)
        if metadata is not None:
            self.to_add['metadatas'].append(metadata)
        self.maybe_commit()


def remove_md_keys(md: dict, patterns: list[str | re.Pattern]) -> dict:
    """Removes keys from the metadata dict that match any of the given patterns.

    These should be regexp patterns (or re objects).
    Returns new metadata dict suitable for .update() with just the keys to set to None
    """
    pats = [re.compile(pat) if isinstance(pat, str) else pat for pat in patterns]
    ret: dict[str, None] = {}
    for key in list(md.keys()):
        for pat in pats:
            if pat.match(key):
                ret[key] = None
                break
    return ret

class FeatureLabel(NamedTuple):
    features: np.ndarray
    label: Any

def extract_features(feature_func: Callable[[Any], np.ndarray | FeatureLabel | None],
                     col: Collection,
                     incr: int=20,
                     ss: int=1,
                     include: list[str] | None=None,
                     **filter_kw: Any) -> tuple[list[str], np.ndarray, list[Any] | None]:
    """Extracts features from the given chroma `col` using the given `feature_func`.

    The feature_func should take in a single Chroma entry and return its features.
    It can also return `None` to skip that entry. The entry is passed as a dict containing:
    - idx (which idx-th entry this is)
    - id
    - embedding
    - document
    - metadata: all other metadata fields
    The last 3 elements are only included if the corresponding `include` parameter is passed (or if
    `include` is None, which is the default).

    The output of this function should be either:
    - A numpy array of the features
    - A FeatureLabel tuple containing:
      - A label for the entry (e.g. the class or category)
      - A numpy array of the features
    - It can also raise StopIteration to stop the extraction early
      - The current batch will be fully run in this case.

    You can optionally pass in `filter_kw` to filter the Chroma entries before extracting features.
    These are passed to the `where` parameter of the Chroma `get()` call.

    The output is a tuple of three items:
    - A list of the Chroma entry IDs
    - A numpy array of the extracted features
    - A list of the labels (if the feature_func returns a tuple), else None

    The extraction is done in batches of `incr` entries at a time.
    You can also pass in `ss` > 1 to subsample one batch out of every `ss` batches.
    """
    valid_includes = {'embeddings', 'metadatas', 'documents'}
    if include is None:
        include = list(valid_includes)
    assert all(inc in valid_includes for inc in include), f'Invalid include: {include}'
    def producer():
        """Produces Chroma entries in batches of `incr`"""
        idx = 0
        for i in tqdm(range(0, col.count()+incr, incr*ss), desc='Extracting features'):
            try:
                resp = col.get(offset=i, limit=incr, where=filter_kw, include=include)
            except Exception as e:
                if 'IndexError' in str(e):
                    print(f'Index error?? {e}')
                    continue
                raise
            if not resp['ids']:
                break
            n_ids = len(resp['ids'])
            for i, id in enumerate(resp['ids']):
                to_yield = dict(id=id, idx=idx)
                # sometimes there's a bug and the lengths of things don't match; return None if so
                if resp.get('embeddings') is not None:
                    if len(resp['embeddings']) != n_ids:
                        logger.error(f'Embeddings length mismatch: {len(resp["embeddings"])} vs {n_ids}')
                        yield None
                    else:
                        to_yield['embedding'] = resp['embeddings'][i]
                if resp.get('metadatas') is not None:
                    if len(resp['metadatas']) != n_ids:
                        logger.error(f'Metadatas length mismatch: {len(resp["metadatas"])} vs {n_ids}')
                        yield None
                    else:
                        to_yield['metadata'] = resp['metadatas'][i]
                if resp.get('documents') is not None:
                    if len(resp['documents']) != n_ids:
                        logger.error(f'Documents length mismatch: {len(resp["documents"])} vs {n_ids}')
                        yield None
                    else:
                        to_yield['document'] = resp['documents'][i]
                yield to_yield
                idx += 1

    def consumer(item):
        """Runs feature extraction on the output of the chroma function, returning (id, features, label) or None."""
        feat = feature_func(item)
        if feat is None:
            return None
        label = None
        if isinstance(feat, FeatureLabel):
            if feat.label is not None:
                label = feat.label
            assert feat.features is not None, f'features is None for {id}'
            feat = feat.features
        return (item['id'], feat, label)

    ids = []
    labels: list[Any] | None = []
    features = []
    for output in chained_producer_consumers([producer, consumer]):
        if output is None:
            continue
        id, feat, label = output
        ids.append(id)
        features.append(feat)
        if label is not None and labels is not None:
            labels.append(label)

    assert len(ids) == len(features), f'mismatched lengths for ids ({len(ids)}) and features ({len(features)})'
    if labels:
        assert len(labels) == len(features), f'Mismatched lengths for labels ({len(labels)}) and features ({len(features)})'
    else:
        labels = None
    return ids, np.array(features), labels

def copy_collection(src_db_port: int, dst_db_port: int, src_col_name: str, dst_col_name: str='', incr: int=100) -> None:
    """Copies `src_col_name` from chroma at `src_db_port` into `dst_col_name` at `dst_db_port`.

    If you don't give a `dst_col_name`, it will use the same name as `src_col_name`.
    """
    src_db = load_chroma_client(port=src_db_port)
    dst_db = load_chroma_client(port=dst_db_port)
    print(src_db.list_collections())
    print(dst_db.list_collections())
    src_col = src_db.get_collection(src_col_name)
    print(src_col, src_col.metadata)
    if not dst_col_name:
        dst_col_name = src_col_name
    try:
        dst_col = dst_db.get_collection(dst_col_name)
    except Exception:
        # create the collection, with the same metadata
        md = dict(**src_col.metadata)
        if 'hnsw:space' not in md:
            md['hnsw:space'] = 'cosine'
        dst_col = dst_db.create_collection(dst_col_name, metadata=md)
    done_ids = set(dst_col.get(include=[])['ids'])
    print(f'Copying to {dst_col} ({dst_col.metadata}) with {len(done_ids)} already done')
    # now copy the collection
    pbar = tqdm(desc='Copying collection')
    offset = 0
    while True:
        try:
            resp = src_col.get(offset=offset, limit=incr, include=['embeddings', 'metadatas', 'documents'])
        except Exception as e:
            if 'IndexError' in str(e):
                print(f'Index error?? {e}')
                continue
            raise
        # now add the embeddings to the new collection
        ids = resp['ids']
        if not ids:
            break
        pbar.update(incr)
        offset += incr
        # filter out done ids, then add the rest
        indices = [idx for idx, id in enumerate(ids) if id not in done_ids]
        if not indices:
            continue
        dst_col.add(
            ids=[resp['ids'][i] for i in indices],
            embeddings=[resp['embeddings'][i] for i in indices],
            metadatas=[resp['metadatas'][i] for i in indices],
            documents=[resp['documents'][i] for i in indices],
        )



if __name__ == '__main__':
    parser = ArgumentParser(description='Copy a collection from one chroma database to another')
    parser.add_argument('src_db_port', type=int, help='Port of the source chroma database')
    parser.add_argument('dst_db_port', type=int, help='Port of the destination chroma database')
    parser.add_argument('src_col_name', type=str, help='Name of the source collection')
    parser.add_argument('dst_col_name', type=str, nargs='?', default='', help='Name of the destination collection')
    args = parser.parse_args()
    copy_collection(args.src_db_port, args.dst_db_port, args.src_col_name, args.dst_col_name)
