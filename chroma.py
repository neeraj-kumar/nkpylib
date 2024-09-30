"""Some utilities for dealing with chroma"""

#TODO futures feature extraction

from __future__ import annotations

import logging
import re
import time

from threading import Lock
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np

from chromadb import Collection, HttpClient, PersistentClient
from tqdm import tqdm

from nkpylib.thread_utils import chained_producer_consumers

CHROMA_CLIENT = None

CHROMA_LOCK = Lock()

logger = logging.getLogger(__name__)


def load_chroma_client(db_path: str, port: int):
    """Loads chroma client (if not already loaded) and returns it.

    This function is thread-safe and will only load the client once.
    It will first try to load the client from the server at the given `port`.
    If that fails, it will try to load the client from the given `db_path`.
    """
    global CHROMA_CLIENT
    if CHROMA_CLIENT is None:
        with CHROMA_LOCK:
            # first try loading from the server
            try:
                CHROMA_CLIENT = HttpClient(port=port)
                logger.info(f'Loaded chromadb client from server at port {port}')
                return CHROMA_CLIENT
            except Exception as e:
                logger.info(f'chromadb client not running: {e}')
                pass
            # now try loading from disk
            logger.info(f'Loading chromadb client at {db_path}')
            t0 = time.time()
            CHROMA_CLIENT = PersistentClient(path=db_path)
            logger.info(f"Loaded chromadb client in {time.time() - t0:.2f}s")
    return CHROMA_CLIENT

def remove_md_keys(md: dict, patterns: list[Union[str, re.Pattern]]) -> dict:
    """Removes keys from the metadata dict that match any of the given patterns.

    These should be regexp patterns (or re objects).
    Returns new metadata dict suitable for .update() with just the keys to set to None
    """
    patterns = [re.compile(pat) if isinstance(pat, str) else pat for pat in patterns]
    ret = {}
    for key in list(md.keys()):
        for pat in patterns:
            if pat.match(key):
                ret[key] = None
                break
    return ret

class FeatureLabel(NamedTuple):
    features: np.ndarray
    label: Any

def extract_features(feature_func: Callable[[Any], Optional[Union[np.ndarray, FeatureLabel]]],
                     col: Collection,
                     incr: int=20,
                     ss: int=1,
                     include: Optional[list[str]]=None,
                     **filter_kw: Any) -> tuple[list[str], np.ndarray, Optional[list[Any]]]:
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
                if resp.get('embeddings'):
                    if len(resp['embeddings']) != n_ids:
                        logger.error(f'Embeddings length mismatch: {len(resp["embeddings"])} vs {n_ids}')
                        yield None
                    else:
                        to_yield['embedding'] = resp['embeddings'][i]
                if resp.get('metadatas'):
                    if len(resp['metadatas']) != n_ids:
                        logger.error(f'Metadatas length mismatch: {len(resp["metadatas"])} vs {n_ids}')
                        yield None
                    else:
                        to_yield['metadata'] = resp['metadatas'][i]
                if resp.get('documents'):
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
    labels = []
    features = []
    for output in chained_producer_consumers([producer, consumer]):
        if output is None:
            continue
        id, feat, label = output
        ids.append(id)
        features.append(feat)
        if label is not None:
            labels.append(label)

    assert len(ids) == len(features), f'mismatched lengths for ids ({len(ids)}) and features ({len(features)})'
    if labels:
        assert len(labels) == len(features), f'Mismatched lengths for labels ({len(labels)}) and features ({len(features)})'
    else:
        labels = None
    return ids, np.array(features), labels
