"""Some utilities for dealing with chroma"""

from __future__ import annotations

import logging
import time

from threading import Lock
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np

from chromadb import Collection, HttpClient, PersistentClient
from tqdm import tqdm

CHROMA_CLIENT = None

CHROMA_LOCK = Lock()

logger = logging.getLogger(__name__)


class FeatureLabel(NamedTuple):
    features: np.ndarray
    label: Any

def extract_features(feature_func: Callable[[Any], Optional[Union[np.ndarray, FeatureLabel]]],
                     col: Collection,
                     incr: int=20,
                     **filter_kw: Any) -> tuple[list[str], np.ndarray, Optional[list[Any]]]:
    """Extracts features from the given chroma `col` using the given `feature_func`.

    The feature_func should take in a single Chroma entry and return its features.
    It can also return `None` to skip that entry. The entry is passed as a dict containing:
    - id
    - embedding
    - document
    - idx (which idx-th entry this is)
    - metadata: all other metadata fields

    The output of this function should be either:
    - A numpy array of the features
    - A FeatureLabel tuple containing:
      - A label for the entry (e.g. the class or category)
      - A numpy array of the features

    You can optionally pass in `filter_kw` to filter the Chroma entries before extracting features.
    These are passed to the `where` parameter of the Chroma `get()` call.

    The output is a tuple of three items:
    - A list of the Chroma entry IDs
    - A numpy array of the extracted features
    - A list of the labels (if the feature_func returns a tuple), else None
    """
    ids = []
    labels = []
    features = []
    idx = 0
    for i in tqdm(range(0, col.count()*2, incr), desc='Extracting features'):
        try:
            resp = col.get(offset=i, limit=incr, where=filter_kw, include=['embeddings', 'metadatas', 'documents'])
        except Exception as e:
            if 'IndexError' in str(e): # we exhausted results (e.g. due to filters), break
                break
            raise
        for id, emb, md, doc in zip(resp['ids'], resp['embeddings'], resp['metadatas'], resp['documents']):
            feat = feature_func(dict(id=id, embedding=emb, metadata=md, document=doc, idx=idx))
            idx += 1
            if feat is None:
                continue
            ids.append(id)
            if isinstance(feat, FeatureLabel):
                labels.append(feat.label)
                features.append(feat.features)
            else:
                features.append(feat)
            len_last = lambda arr: f'{len(arr)}' if arr else 'None'
    assert len(ids) == len(features), f'mismatched lengths for ids ({len(ids)}) and features ({len(features)})'
    if labels:
        assert len(labels) == len(features), f'Mismatched lengths for labels ({len(labels)}) and features ({len(features)})'
    else:
        labels = None
    return ids, np.array(features), labels


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
