from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from typing import Sequence
from nkpylib.search.searcher import (
        SearchImpl,
        SearchResult,
    )
from nkpylib.search.searcher import SearchCond, OpCond, Op, JoinCond, JoinType, Array1D

logger = logging.getLogger(__name__)

class QdrantSearch(SearchImpl):
    """Search implementation for Qdrant."""
    def __init__(self, client: QdrantClient, collection_name: str):
        """Initialize with a Qdrant client and collection name."""
        super().__init__()
        self.client = client
        self.collection_name = collection_name

    def parse_cond(self, cond: SearchCond) -> dict|models.Filter|None:
        """Recursively traverse the search condition and convert it to a Qdrant filter.

        This returns the thing that you initialize a `models.Filter` with, or None if nothing valid.
        Ignore embeddings, which we will deal with separately.
        """
        if isinstance(cond, OpCond):
            # Map OpCond to Qdrant filter condition
            if cond.op == Op.EQ:
                return models.FieldCondition(
                    key=cond.field,
                    match=models.MatchValue(value=cond.value)
                )
            elif cond.op in {Op.GT, Op.GTE, Op.LT, Op.LTE}:
                return models.FieldCondition(
                    key=cond.field,
                    range=models.Range(
                        **{cond.op.name.lower(): cond.value}
                    )
                )
            elif cond.op == Op.IN:
                return models.FieldCondition(
                    key=cond.field,
                    match=models.MatchAny(any=cond.value)
                )
            elif cond.op == Op.NOT_IN:
                return models.FieldCondition(
                    key=cond.field,
                    match=models.MatchExcept(**{
                        'except': cond.value
                    })
                )
            elif cond.op == Op.NOT_EXISTS:
                return models.IsEmptyCondition(
                    is_empty=models.PayloadField(key=cond.field)
                )
            elif cond.op == Op.EXISTS:
                return {
                    "must_not": [
                        models.IsEmptyCondition(
                            is_empty=models.PayloadField(key=cond.field)
                        )
                    ]
                }
            elif cond.op == Op.IS_NULL:
                return models.IsNullCondition(
                    is_null=models.PayloadField(key=cond.field)
                )
            elif cond.op == Op.IS_NOT_NULL:
                return {
                    "must_not": [
                        models.IsNullCondition(
                            is_null=models.PayloadField(key=cond.field)
                        )
                    ]
                }
            elif cond.op == Op.CLOSE_TO:
                pass # handle this separately
            else:
                #NEQ, LIKE, NOT_LIKE
                raise NotImplementedError(f"Unsupported operator: {cond.op}")
        elif isinstance(cond, JoinCond):
            # Recursively process JoinCond
            sub_filters = (self.parse_cond(c) for c in cond.conds if c is not None)
            sub_filters = [f for f in sub_filters if f is not None]
            if not sub_filters:
                return None
            if cond.join == JoinType.AND:
                return {"must": sub_filters}
            elif cond.join == JoinType.OR:
                return {"should": sub_filters}
            elif cond.join == JoinType.NOT:
                return {"must_not": sub_filters}
            else:
                raise ValueError(f"Unsupported join type: {cond.join}")
        else:
            raise TypeError("Unsupported condition type")

    def search(self, cond: SearchCond, n_results: int=15, **kw) -> list[SearchResult]:
        """Does a search with given search conditions `cond`.

        Qdrant can do batch searches with arbitrary combinations of query embeddings and conditions.
        We first parse the `cond` into a `models.Filter` object. We then take all the vector search
        conditions and send those + filters to qdrant to do the search.

        The _search() function returns a nested list of results, one per query embedding. We combine
        those into a single list by taking the max of the score for each id, and sort by score desc.

        Note that for now we assume all non-vector filters are combined into a single Filter object
        and used for all vectors in this cond.
        """
        filter_input = self.parse_cond(cond)
        logger.debug(f'got filter input: {filter_input}')
        if isinstance(filter_input, dict): # dict input, use **
            filters = models.Filter(**filter_input)
        elif filter_input is None: # no conditions, use None
            filters = None
        else: # single condition, just pass it in directly
            filters = models.Filter(filter_input)
        def get_vec(c: SearchCond):
            if isinstance(c, OpCond) and c.op == Op.CLOSE_TO:
                return (c.field, c.value)

        vecs = [v for v in cond.walk(get_vec) if v is not None]
        results = self._search(query_embeddings=vecs, filters=filters, n_results=n_results, **kw)
        ret_by_id = {}
        for res in results:
            for r in res:
                if r.id not in ret_by_id:
                    ret_by_id[r.id] = r
                if r.score > ret_by_id[r.id].score:
                    ret_by_id[r.id] = r
        # sort by score descending
        ret = sorted(ret_by_id.values(), key=lambda x: x.score, reverse=True)
        return ret

    def _search(self,
                query_embeddings: Sequence[Array1D]|Sequence[tuple[str, Array1D]],
                filters: models.Filter|Sequence[models.Filter]|None = None,
                n_results: int=10,
                score_threshold: float=-1.0,
                include_vectors: bool=False,
                include_metadatas: bool=False,
                include_documents: bool=False,
                **kw) -> list[list[SearchResult]]:
        assert len(query_embeddings) > 0
        if filters is None or isinstance(filters, models.Filter):
            filters = [filters] * len(query_embeddings)
        else:
            assert len(filters) == len(query_embeddings), "Filters must match number of query embeddings"
        using = [None] * len(query_embeddings)
        if isinstance(query_embeddings[0], tuple):
            # if we have a tuple, we expect (field_name, embedding)
            using = [field for field, _ in query_embeddings]
            query_embeddings = [emb for _, emb in query_embeddings]
        reqs = [models.QueryRequest(
                    query=emb, # type: ignore
                    filter=filter_,
                    using=u,
                    limit=n_results,
                    score_threshold=score_threshold,
                    with_vector=include_vectors,
                    with_payload=include_metadatas or include_documents,
                ) for emb, u, filter_ in zip(query_embeddings, using, filters)]
        with self.timed('search'):
            results = self.client.query_batch_points(collection_name=self.collection_name, requests=reqs)
        ret = []
        for r in results:
            cur = []
            for point in r.points:
                metadata = point.payload if include_metadatas or include_documents else None
                document = metadata.get('document') if (include_documents and metadata) else None
                cur.append(SearchResult(
                    id=point.id,
                    score=point.score,
                    metadata=metadata,
                    document=document,
                    vector=point.vector if include_vectors else None, # type: ignore
                ))
            ret.append(cur)
        return ret

