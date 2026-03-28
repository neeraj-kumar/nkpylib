from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback

from collections import Counter
from os.path import exists
from typing import Any, Callable, NamedTuple

from pony.orm import db_session
from pony.orm.core import Query, QueryResult

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import LmdbUpdater, NumpyLmdb
from nkpylib.nkcollections.model import Item, CFG, IMAGE_SUFFIX
from nkpylib.thread_utils import ProducerConsumerPipeline, run_async
from nkpylib.web_utils import make_request_async

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """Result structure for pipeline stages."""
    row: Any
    data: Any = None
    error: Exception|None = None

async def maybe_dl(url: str, path: str, fetch_delay: float=0.1, timeout: float=-1) -> bool:
    """Downloads the given url to the given dir if it doesn't already exist there (and is not empty).

    - fetch_delay: minimum delay in seconds between fetches, to avoid overwhelming servers. This
      limit is per domain.
    - timeout: if > 0, the maximum time in seconds to wait for the download. If the download doesn't
      complete in that time then raises TimeoutError. If <= 0, no timeout is applied.

    Returns if we actually downloaded the file.
    """
    if exists(path) and os.path.getsize(path) > 0:
        return False
    logger.debug(f'downloading image {url} -> {path}')
    r = await asyncio.wait_for(
            make_request_async(
                url,
                headers={'Accept': 'image/*,video/*'},
                min_delay=fetch_delay,
            ), timeout if timeout > 0 else None)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        pass
    with open(path, 'wb') as f:
        f.write(r.content)
        logger.debug(f'  Downloaded {url} to {path}')
    return True

async def _run_embedding_pipeline(
        rows: list[Any]|QueryResult[Any],
        stages: list[Callable],
        pipeline_config: dict,
        updater: LmdbUpdater,
        result_processor: Callable[[Any, LmdbUpdater], Counter]) -> Counter:
    """Generic pipeline runner for embedding tasks."""
    pipeline = ProducerConsumerPipeline(funcs=stages, **pipeline_config)
    counts: Counter = Counter()
    try:
        async for result in pipeline.run_async(rows):
            #logger.info(f'Processing result for row {result.row.id} through result processor')
            stats = result_processor(result, updater)
            counts.update(stats)
        logger.info('here we are')
    except Exception as e:
        print(f'Error processing result for row {result.row.id}: {e}\n{traceback.format_exc()}')
        raise
    logger.info(f'Finished pipeline with counts: {dict(counts)}')
    updater.commit()
    return counts

def _update_database_from_result(
        result: PipelineResult,
        updater: LmdbUpdater,
        key_suffix: str,
        ts_field: str,
        success_callback: Callable[[Any, Any, int], None]|None = None) -> Counter:
    """Generic database updater for embedding results.

    The success_callback is called with (row, data, timestamp) if the embedding was successful, and
    can be used to do custom processing like adding to lmdb or updating other fields. If not given,
    we just add the embedding to lmdb with the key and timestamp, and update the timestamp field in
    sqlite.
    """
    key = f'{result.row.id}:{key_suffix}'
    ts = int(time.time())
    counts: Counter = Counter()
    if result.data is not None: # Success case
        if success_callback:
            success_callback(result.row, result.data, ts)
        counts['success'] += 1
        counts['updated'] += 1
    elif result.error is not None: # Error case
        updater.add(key, metadata=dict(embed_ts=ts, error=str(result.error)))
        with db_session:
            setattr(result.row, ts_field, -1)
        counts['errors'] += 1
    counts['total'] += 1
    return counts

def _create_embedding_stage(
        embed_func: Callable,
        model: str,
        key_suffix: str,
        updater: LmdbUpdater):
    """Factory for creating embedding stages."""
    logger.info(f'Creating embedding stage with model={model}, key_suffix={key_suffix}')
    async def embed_stage(input_data: PipelineResult) -> PipelineResult:
        row = input_data.row
        key = f'{row.id}:{key_suffix}'
        # Skip if already in lmdb
        if key in updater:
            logger.info(f' skipping {row}, key={key} already in lmdb')
            return PipelineResult(row=row, data=None, error=None)
        try:
            data_to_embed = input_data.data
            if not data_to_embed:
                logger.info(f'Skipping {row} with no data to embed')
                return PipelineResult(row=row, data=None, error=None)
            embedding = await embed_func.single_async(data_to_embed, model=model) # type: ignore[attr-defined]
            logger.debug(f'For {key} embedded {data_to_embed}, got embedding {embedding} using {model}')
            return PipelineResult(row=row, data=embedding, error=None)
        except Exception as e:
            logger.warning(f'Error in embedding stage for {row}: {e}')
            return PipelineResult(row=row, data=None, error=e)
    return embed_stage

async def update_text_embeddings(q: Query, limit: int, lmdb_path: str, **kw) -> Counter:
    """Updates text embeddings for the given query `q`.

    This select 'text' and 'link' otypes and updates their embeddings.
    Returns a Counter with embedding statistics.
    """
    if not CFG.db.allow_embedding_updates:
        return Counter()
    return Counter() #FIXME
    with db_session:
        rows = q.filter(lambda c: c.otype in ('text', 'link') and not c.embed_ts).limit(limit)
        if not rows:
            return Counter()
        logger.info(f'Updating embeddings for upto {len(rows)} text rows: {rows[:5]}...')
    updater = LmdbUpdater(lmdb_path, n_procs=1, item_incr=5)
    # Stage 1: Extract text content from rows
    async def extract_text_stage(row):
        if not row.md:
            return PipelineResult(row=row, data='')
        text = ''
        if row.otype == 'text' and 'text' in row.md:
            text = row.md['text']
        elif row.otype == 'link' and 'title' in row.md:
            text = f"{row.md['title']}: {row.url}"
        logger.debug(f'Extracted text for {row}, key={row.id}:text, text={text[:30] if text else "empty"}')
        return PipelineResult(row=row, data=text)

    # Stage 2: Generate text embeddings using factory
    embed_stage = _create_embedding_stage(
        embed_func=embed_text,
        model='qwen_emb',
        key_suffix='text',
        updater=updater,
    )
    # Success callback for text embeddings
    def text_success_callback(row, embedding, ts):
        #logger.info(f'adding text embedding for {row}, key={row.id}:text, emb={embedding[:10] if embedding is not None else "failed"}')
        updater.add(f'{row.id}:text', embedding=embedding, metadata=dict(embed_ts=ts))
        with db_session:
            row.embed_ts = ts
        #logger.info(f'  end of text_success_callback for {row}, updated embed_ts to {ts}')

    def result_processor(result, updater):
        #logger.info(f'Processing result for row {result.row.id} in result_processor')
        ret = _update_database_from_result(
            result, updater, 'text', 'embed_ts', text_success_callback
        )
        #logger.info(f'done processing result for row {result.row.id} in result_processor, got stats {dict(ret)}')
        return ret

    # Run pipeline
    stats = await _run_embedding_pipeline(
        rows=rows,
        stages=[extract_text_stage, embed_stage],
        pipeline_config=dict(
            q_size=[20, 10],
            concurrency=[20, 10],
            exc_policy='stop'
        ),
        updater=updater,
        result_processor=result_processor,
    )
    updater.commit()
    if stats['updated'] > 0:
        logger.info(f'  Updated embeddings for {stats["updated"]} text rows')
    return stats


async def update_image_embeddings(q: Query,
                                  lmdb_path: str,
                                  limit: int,
                                  fetch_delay: float=0.1,
                                  **kw) -> Counter:
    """Updates images embeddings for the given query `q`.

    This select 'image' rows, downloads them if needed, and updates their embeddings.
    Returns a Counter with embedding statistics.
    """
    if not CFG.db.allow_embedding_updates:
        return Counter()
    with db_session:
        rows = q.filter(lambda c: c.otype == 'image' and not c.embed_ts).limit(limit)
    if not rows:
        return Counter()
    updater = LmdbUpdater(lmdb_path, n_procs=1)
    logger.info(f'Updating embeddings for upto {len(rows)} image rows: {rows[:5]}...')
    # Stage 1: Download images
    async def download_stage(row):
        with db_session:
            path = row.image_path()
            try:
                await maybe_dl(row.url, path, fetch_delay=fetch_delay, timeout=30)
                if not path or not exists(path) or os.path.getsize(path) == 0:
                    raise FileNotFoundError(f'File not found or empty: {path}')
                #TODO also check for valid image?
                return PipelineResult(row=row, data=path)
            except Exception as e:
                logger.warning(f'Error downloading image for row id={row.id}, url={row.url}, path={path}: {e}')
                return PipelineResult(row=row, data='')

    embed_stage = _create_embedding_stage(
        embed_func=embed_image,
        model='mobilenet',  #FIXME: should be 'clip'
        key_suffix=IMAGE_SUFFIX,
        updater=updater,
    )
    # Success callback for image embeddings
    def image_success_callback(row, embedding, ts):
        key = f'{row.id}:{IMAGE_SUFFIX}'
        updater.add(key, embedding=embedding, metadata=dict(embed_ts=ts))
        with db_session:
            row.embed_ts = ts
            logger.debug(f' emb for image {row}, key={key}, {embedding[:10] if embedding is not None else "failed"}')

    # Run pipeline
    stats = await _run_embedding_pipeline(
        rows=rows,
        stages=[download_stage, embed_stage],
        pipeline_config=dict(
            q_size=[20, 10],
            concurrency=[20, 10],
            exc_policy='stop'
        ),
        updater=updater,
        result_processor=lambda result, updater: _update_database_from_result(
            result, updater, IMAGE_SUFFIX, 'embed_ts', image_success_callback
        )
    )
    if stats['updated'] > 0:
        logger.info(f'  Updated embeddings for {stats["updated"]} images, {stats["success"]} successful')
    return stats


async def update_image_descriptions(
        q,
        lmdb_path: str,
        limit: int,
        vlm_prompt: str|None='Briefly describe this image. Include a list of tags at the end.',
        sys_prompt: str|None=None,
        vlm_model: str='fastvlm',
        **kw) -> Counter:
    """Updates image descriptions using VLM for images that have been explored.

    Filters to images where explored_ts is not null, generates descriptions via VLM,
    embeds the descriptions, and updates both LMDB and SQLite metadata.

    Returns a Counter with description statistics.
    """
    if not CFG.db.allow_embedding_updates:
        return Counter()
    #return Counter() #FIXME
    if not vlm_prompt or not vlm_model:
        return Counter()
    with db_session:
        q = q.filter(lambda c: c.otype == 'image' and c.embed_ts is not None and c.embed_ts > 0 and c.explored_ts is None)
        #rows = q.limit(limit)
        rows = q.random(limit) #FIXME
        if not rows:
            return Counter()
        logger.info(f'Updating descriptions for {len(rows)} image rows: {rows[:5]}...')
    updater = LmdbUpdater(lmdb_path, n_procs=1)
    #TODO deal with errors in getting desc
    # Stage 1: Generate VLM descriptions
    async def vlm_stage(row):
        if sys_prompt:
            messages = [
                dict(role='system', content=sys_prompt),
                dict(role='user', content=vlm_prompt)
            ]
        else:
            messages = vlm_prompt
        path = row.image_path()
        try:
            desc = await call_vlm.single_async((path, messages), model=vlm_model)
            logger.debug(f'For {row}, got description {desc} using {vlm_model}')
            with db_session:
                row.md['desc'] = desc
                row.md['desc_ts'] = int(time.time())
            return PipelineResult(row=row, data=desc)
        except Exception as e:
            e = str(e)[:500] # truncate to avoid logging huge errors
            logger.warning(f'Error generating desc for image {row}, path={path}: {e}')
            return PipelineResult(row=row, data='')

    # Stage 2: Generate text embeddings using factory
    embed_stage = _create_embedding_stage(
        embed_func=embed_text,
        model='qwen_emb',
        key_suffix='text',
        updater=updater,
    )
    pipeline = ProducerConsumerPipeline(
        funcs=[vlm_stage, embed_stage],
        q_size=[50, 20],
        concurrency=[50, 20],
        exc_policy='stop'
    )
    counts: Counter = Counter()
    async for result in pipeline.run_async(rows):
        # Handle the complex case where we need both description and embedding
        # We lost the description, need to get it from row.md
        desc = result.row.md.get('desc', '') if result.row.md else ''
        key = f'{result.row.id}:text'
        with db_session:
            if result.data is not None and desc:
                ts = int(time.time())
                logger.debug(f'adding desc embedding for {result.row}, key={key}, desc={desc[:30] if desc else "empty"}, emb={result.data[:10] if result.data is not None else "failed"}')
                updater.add(key, embedding=result.data, metadata=dict(desc=desc, embed_ts=ts))
                result.row.explored_ts = ts
                counts['updated'] += 1
                counts['success'] += 1
            elif result.error is not None and desc:
                updater.add(key, metadata=dict(desc=desc, embed_ts=time.time(), error='text embedding failed'))
                result.row.explored_ts = -1
                counts['errors'] += 1
            else:  # failed to get description
                result.row.explored_ts = -1
                counts['no_desc'] += 1
        counts['total'] += 1
    updater.commit()
    if counts['updated'] > 0:
        logger.info(f'Updated descriptions for {counts["updated"]} images')
    return counts

async def update_embeddings_async(lmdb_path: str,
                                  images_dir: str,
                                  ids: list[int]|None=None,
                                  vlm_prompt: str|None='briefly describe this image',
                                  sys_prompt: str|None=None,
                                  vlm_model: str='fastvlm',
                                  limit: int=-1,
                                  fetch_delay: float=0.1,
                                  source: str|None=None,
                                  **kw) -> dict[str, int]:
    """Updates the embeddings for all relevant rows in our table.

    This does embeddings of:
    - otype=text: text embeddings of the 'text' field in md
    - otype=link: text embeddings of the 'title' field + url in md
    - otype=image: image embeddings of the image at url (downloaded to images_dir if needed)
    - otype=image: text embeddings of image descriptions generated via VLM

    The embeddings are stored in a NumpyLmdb at the given `lmdb_path`.
    For images, we first download them to the given `images_dir`.

    If `ids` is given, we only update embeddings for those ids, else all ids that don't exist in
    the lmdb. If you specify a positive `limit`, we only update upto that many embeddings, per
    otype. Note that because image embeddings are done locally and are much slower, we apply a
    factor of 2x for the other two functions. In general, we skip rows that are already marked
    done in the sql database (via the `embed_ts` or `explored_ts` fields), and in the case of
    text, we also skip keys that are already in the lmdb.

    By default we use the given `vlm_prompt` to generate image descriptions for images. If you
    want, you can override this and also optionally override the `sys_prompt`. If `vlm_prompt`
    is empty or None, we don't generate descriptions.

    We run the 3 subfunctions (text+link, image embeddings, image descriptions+text embeddings)
    asynchronously in parallel.

    Any kw are passed to the subfunctions, some of which call `batch_extract_embeddings`.

    We return a dict with the number of embeddings updated for each type
    """
    if not CFG.db.allow_embedding_updates:
        return Counter()
    if limit <= 0:
        limit = 10000000
    with db_session:
        q = Item.select(lambda c: (ids is None or c.id in ids))
        if source is not None:
            q = q.filter(lambda c: c.source == source)
        q = q.order_by(Item.id.desc()) # type: ignore[attr-defined]
    common_kw = dict(q=q, lmdb_path=lmdb_path, **kw)
    # start async tasks for all 3 subfunctions
    text_task = asyncio.create_task(update_text_embeddings(limit=limit, **common_kw))
    image_task = asyncio.create_task(
        update_image_embeddings(fetch_delay=fetch_delay, limit=limit, **common_kw)
    )
    desc_task = asyncio.create_task(
        update_image_descriptions(vlm_prompt=vlm_prompt,
                                  sys_prompt=sys_prompt,
                                  vlm_model=vlm_model,
                                  limit=limit//2,
                                  **common_kw)
    )
    text_stats, image_stats, desc_stats = await asyncio.gather(text_task, image_task, desc_task)
    # Merge all counters with prefixes to avoid key conflicts
    ret: Counter = Counter()
    for k, v in text_stats.items():
        ret[f'text_{k}'] = v
    for k, v in image_stats.items():
        ret[f'image_{k}'] = v
    for k, v in desc_stats.items():
        ret[f'desc_{k}'] = v
    if ret:
        logger.info(f'Finished updating embeddings async: {dict(ret)}')
    return dict(ret)

def update_embeddings(lmdb_path: str,
                      images_dir: str,
                      ids: list[int]|None=None,
                      vlm_prompt: str|None='briefly describe this image',
                      sys_prompt: str|None=None,
                      vlm_model: str='fastvlm',
                      limit: int=-1,
                      fetch_delay: float=0.1,
                      **kw) -> dict[str, int]:
    """Calls the async version"""
    ret = run_async(update_embeddings_async(
        lmdb_path=lmdb_path,
        images_dir=images_dir,
        ids=ids,
        vlm_prompt=vlm_prompt,
        sys_prompt=sys_prompt,
        vlm_model=vlm_model,
        limit=limit,
        fetch_delay=fetch_delay,
        **kw
    ))
    if ret:
        print(f'Sync Done with update_embeddings, got {ret}')
    return ret

def cleanup_embeddings(lmdb_path: str) -> None:
    """Cleans up discrepancies between our sqlite and lmdb.

    Note that this doesn't modify the lmdb at all, only the sqlite.
    """
    if not CFG.db.allow_embedding_updates:
        return
    db = NumpyLmdb.open(lmdb_path, flag='r')
    keys_in_db = set(db.keys())
    n_missing = 0
    n_done = 0
    def fix(rows: list[Any], key_suffix: str, ts_field: str, fix_missing: bool, db) -> int:
        """Fix synchronization between sqlite and lmdb.

        - fix_missing: If True, fix items marked done in sqlite but missing in lmdb. If
          False, fix items present in lmdb but not marked done in sqlite
        """
        n = 0
        for row in rows:
            key = f'{row.id}:{key_suffix}'
            if fix_missing:
                # Fix wrongly marked as done in sqlite but missing in lmdb
                if key not in keys_in_db:
                    logger.debug(f'Cleaning up {row} with missing key {key}')
                    setattr(row, ts_field, None)
                    n += 1
            else:
                # Fix present in lmdb but not marked done in sqlite
                if key in keys_in_db:
                    logger.debug(f'Marking done for {row} with existing key {key}')
                    d = db.md_get(key)
                    ts = d.get('embed_ts', d.get('embedding_ts', int(time.time())))
                    setattr(row, ts_field, int(time.time()))
                    n += 1
        return n

    # first deal with embeddings wrongly marked as done in sqlite but missing in lmdb
    with db_session:
        rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype in ('text', 'link'))
        n_missing += fix(rows, 'text', 'embed_ts', fix_missing=True, db=db)
    with db_session:
        rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype == 'image')
        n_missing += fix(rows, IMAGE_SUFFIX, 'embed_ts', fix_missing=True, db=db)
    with db_session:
        rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is not None and c.explored_ts > 0)
        n_missing += fix(rows, 'text', 'explored_ts', fix_missing=True, db=db)
        # now deal with embeddings present in lmdb but not marked done in sqlite
    with db_session:
        rows = Item.select(lambda c: c.otype in ('text', 'link') and c.embed_ts is None)
        n_done += fix(rows, 'text', 'embed_ts', fix_missing=False, db=db)
    with db_session:
        rows = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None)
        n_done += fix(rows, IMAGE_SUFFIX, 'embed_ts', fix_missing=False, db=db)
    with db_session:
        rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is None)
        n_done += fix(rows, 'text', 'explored_ts', fix_missing=False, db=db)
    del db
    logger.info(f'Cleaned up {n_missing} missing and {n_done} done embeddings')

def find_similar(pos: list[str|int],
                 *,
                 embs: Embeddings,
                 cur_ids: list[int]|None,
                 classifier_path: str|None = None) -> dict[str, Any]:
    """Searches for similarity to `pos` amongst `cur_ids` using `embs`"""
    # Load pipeline from the last saved likes classifier
    #TODO write mapping from sqlite id to imdb_id
    #TODO create cfg.db.embedding_id_map_path
    #TODO in this function, remap sql ids to lmdb ids
    print(f'in find similar with {CFG.db}')
    pipeline = None
    if 0 and classifier_path: #FIXME broken
        try:
            saved_data = embs.load_classifier(classifier_path)
            pipeline = saved_data.get('pipeline')
            logger.info(f"Loaded pipeline from {classifier_path}: {pipeline}")
        except Exception as e:
            logger.warning(f"Could not load pipeline from classifier: {e}")
    pos = [f'{p}:{IMAGE_SUFFIX}' for p in pos]
    if cur_ids is None:
        all_keys = [k for k in embs if k.endswith(f':{IMAGE_SUFFIX}')]
    else:
        all_keys = [f'{id}:{IMAGE_SUFFIX}' for id in cur_ids]
    logger.info(f'got pos={pos}, {len(all_keys)} all keys: {all_keys[:5]}...')
    #ret = embs.similar(pos, all_keys=all_keys, method='nn', pipeline=pipeline)
    ret = embs.similar(pos, all_keys=all_keys, method='nn') #FIXME
    scores, curIds = zip(*ret)
    return dict(
        pos=pos,
        scores={int(id.split(':')[0]): score for id, score in zip(curIds, scores)},
        msg=f'Classified {len(scores)} items with pos {pos}',
    )
