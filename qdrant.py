"""Utilities to deal with qdrant"""

from __future__ import annotations

from hashlib import sha256

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from nkpylib.thread_utils import CollectionUpdater

Array1D = list[float] | tuple[float, ...]  # Assuming a 1D array of floats

def get_qdrant(collection_name: str, dist_fn=Distance.COSINE, vectors: dict[str, int]|None=None) -> tuple[QdrantClient, str]:
    """Returns (qdrant client, col name).

    If you pass in `vectors`, it tries to initialize a new collection, with named vectors using the
    given names and dims. E.g. vectors={'jina': 768}
    Note that it defaults to using cosine distance, but you can change that with `dist_fn`.
    """
    client = QdrantClient(host='localhost', grpc_port=6334, prefer_grpc=True)
    if vectors:
        cfg = {name: VectorParams(size=dim, distance=dist_fn) for name, dim in vectors.items()}
        try:
            client.create_collection(collection_name=collection_name, vectors_config=cfg)
        except Exception:
            pass
    return client, collection_name


def iter_qdrant(collection_name: str,
                batch_size:int =1000,
                client: QdrantClient | None = None,
                **scroll_kwargs) -> Iterator[dict]:
    """
    Generator to iterate through all points in a Qdrant collection.

    Args:
        client: An instance of qdrant_client.QdrantClient.
        collection_name (str): Name of the collection to iterate through.
        batch_size (int): Number of points to fetch in each scroll request.
        **scroll_kwargs: Additional keyword arguments passed to the scroll method, e.g.
                         `with_vectors=True`, `with_payload=True`, `filter=...`, etc.

    Yields:
        dict: Point dicts returned by Qdrant.
    """
    if not client:
        client = QdrantClient(host='localhost', grpc_port=6334, prefer_grpc=True)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            **scroll_kwargs
        )
        if not points:
            break
        for point in points:
            yield point
        if offset is None:
            break

def make_uuid(s: str) -> str:
    """Returns a qdrant-compatible UUID for the given string.

    This is because qdrant doesn't allow arbitrary string ids.
    So we just hash the string to create a UUID, returning the first 32 chars of the hexdigest.
    """
    return sha256(s.encode('utf-8')).hexdigest()[:32]


class QdrantUpdater(CollectionUpdater):
    """A qdrant-specific version of `CollectionUpdater`."""
    def __init__(self,
                 col_name: str,
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
        self.client, self.col_name = get_qdrant(col_name)
        # see if we have a single vector
        vecs = self.client.get_collection(col_name).config.params.vectors
        self.vec_name: str|None=None
        if len(vecs) == 1:
            self.vec_name = next(iter(vecs.keys()))
        super().__init__(add_fn=self._add_fn,
                         item_incr=item_incr,
                         time_incr=time_incr,
                         post_commit_fn=post_commit_fn,
                         debug=debug)

    def _add_fn(self, to_add: dict[str, list]) -> None:
        """Batch add to qdrant"""
        self.client.upsert(collection_name=self.col_name, points=to_add['objects'])

    def add(self, id: int|str, payload: dict[str, Any]|None=None, embedding: dict|Array1D|None=None):
        """Adds an item to the updater.

        You can provide an `embedding` as a vector directly, in which case, we will try to add it to
        the presumably only vectors in this collection. Or, you can provide it as a dict mapping
        from vector name(s) to vectors.

        If the update frequency is reached, it will commit the items to the collection.
        """
        if embedding and len(embedding) > 10:
            if self.vec_name:
                # if we have a single vector, use that
                embedding = {self.vec_name: embedding}
        obj = PointStruct(
            id=id,
            vector=embedding,
            payload=payload,
        )
        super().add(id=id, obj=obj)


def update_qdrant(batch_size=1000, **kw):
    """Updates our qdrant db with embeddings for all movies"""
    pool = ThreadPoolExecutor(max_workers=8)
    num = client.count(collection_name)
    v0 = None
    v0 = client.retrieve(collection_name, ids=[titleid_to_num('tt0000575')])
    print(client, num, v0)
    def add_fn(to_add: dict[str, list]) -> None:
        """Adds a batch of movies to qdrant"""
        # replace the embedding future in each object with the embedding
        points = []
        for obj in to_add['objects']:
            f = obj['vector']
            if isinstance(f, Future):
                obj['vector'] = f.result()
            points.append(PointStruct(
                id=titleid_to_num(obj['id']),
                vector=obj['vector'],
                payload=obj['payload'],
            ))
        logger.debug(f'Upserting {len(points)} objects to qdrant: {points[:3]}...')
        # we do this in a future so the main thread can continue (particularly embedding)
        futures.append(pool.submit(
            client.upsert, collection_name=collection_name, points=points
        ))
        logger.debug(f'Futures status: {[f.done() for f in futures]}')

    updater = CollectionUpdater(add_fn=add_fn, item_incr=batch_size, time_incr=0, post_commit_fn=post_commit_fn)
    n_movies = Movie.select(lambda m: not m.in_qdrant).count()
    logger.info(f"Updating qdrant col {collection_name} with {len(done)} in qdrant and {n_movies} not done in sqlite")
    for movie in tqdm(orm.select(m for m in Movie if not m.in_qdrant)):
        if movie.title_id in done:
            movie.in_qdrant = True
            continue
        s = get_movie_embedding_str(movie)
        try:
            emb = embed_text.single_future(s, model='qwen_emb_small')
        except Exception as e:
            logger.error(f"Failed to get embedding for {movie}: {s} -> {e}")
            continue
        logger.debug(f'For movie {movie} got: {s} -> {emb}')
        if len(movie.ratings) > 0:
            [rating] = movie.ratings.random(1)
            rating, votes = rating.rating, rating.votes
        else:
            rating, votes = 0, 0
        obj = dict(
                id=movie.title_id,
                vector=emb,
                payload=dict(
                    title=movie.title,
                    year=movie.year or 0,
                    runtime=movie.runtime or 0,
                    rating=rating,
                    votes=votes,
                ))
        updater.add(id=movie.title_id, obj=obj)
        logger.debug(f'  Adding {obj}')
        done.add(movie.title_id)
        #if len(done) > 40000: break
    updater.commit()
    for f in futures:
        f.result()

