from __future__ import annotations

import logging

from nkpylib.search.searcher import SearchImpl, SearchResult

logger = logging.getLogger(__name__)

class ChromaSearch(SearchImpl):
    """Search implementation for ChromaDB."""
    def __init__(self, db, col):
        """Initialize with a ChromaDB database and collection."""
        super().__init__()
        self.db = db
        self.col = col

    def _search(self,
                query_embeddings: Sequence[Array1D],
                n_results: int=10,
                score_threshold: float=-1.0,
                conditions: list[dict]|None=None,
                include_vectors: bool=False,
                include_metadatas: bool=False,
                include_documents: bool=False,
                **kw) -> list[SearchResult]:
        """Search the ChromaDB collection with the given query embeddings.

        It passes all kw directly to chroma.query()

        You can also pass in either `where` as a kw (as chroma wants it), or `conditions`, a list of
        conditions that will be `$and`-ed together.

        Returns a list of lists of `SearchResult` objects, where each inner list corresponds to each
        query, in order.

        Returns upto `n_results` results per query, where scores are within `score_threshold` (for
        Chroma, this is a maximum).
        """
        assert not (kw.get('where') and conditions), 'Cannot pass both where and conditions'
        query_kw = dict(n_results=n_results, query_embeddings=query_embeddings, include=['distances'], **kw)
        for param, key in [(include_vectors, 'embeddings'),
                           (include_metadatas, 'metadatas'),
                           (include_documents, 'documents')]:
            if param:
                query_kw['include'].append(key)
        if conditions:
            query_kw['where'] = {'$and': conditions}
        with self.timed('search'):
            r = self.col.query(**query_kw)
        assert len(r['ids']) == len(query_embeddings)
        ret = []
        for i in range(len(r['ids'])):
            ids, distances = r['ids'][i], r['distances'][i]
            for j in range(len(ids)):
                if distances[j] > score_threshold:
                    break
                ret.append(SearchResult(
                    id=ids[j],
                    score=distances[j],
                    metadata=r['metadatas'][i][j] if include_metadatas else None,
                    document=r['documents'][i][j] if include_documents else None,
                    vector=r['embeddings'][i][j] if include_vectors else None,
                ))
        return ret

