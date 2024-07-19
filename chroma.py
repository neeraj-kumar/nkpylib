"""Some utilities for dealing with chroma"""

from __future__ import annotations
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np

from chromadb import Collection
from tqdm import tqdm

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
    - all other metadata fields

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

    for i in tqdm(range(0, col.count(), incr), desc='Extracting features'):
        resp = col.get(offset=i, limit=incr, where=filter_kw, include=['embeddings', 'metadatas', 'documents'])
        for id, emb, md, doc in zip(resp['ids'], resp['embeddings'], resp['metadatas'], resp['documents']):
            feat = feature_func(dict(id=id, embedding=emb, metadata=md, document=doc))
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
