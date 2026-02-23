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
from pony.orm.core import Query

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.nklmdb import LmdbUpdater, NumpyLmdb
from nkpylib.thread_utils import ProducerConsumerPipeline
from nkpylib.web_utils import make_request_async

logger = logging.getLogger(__name__)

IMAGE_SUFFIX = 'mn_image'


class PipelineResult(NamedTuple):
    """Result structure for pipeline stages."""
    row: Any
    data: Any = None
    error: Exception = None


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
        rows: list[Any],
        stages: list[Callable],
        pipeline_config: dict,
        updater: LmdbUpdater,
        result_processor: Callable[[Any, LmdbUpdater], Counter]) -> Counter:
    """Generic pipeline runner for embedding tasks."""
    pipeline = ProducerConsumerPipeline(funcs=stages, **pipeline_config)
    counts = Counter()
    async for result in pipeline.run_async(rows):
        stats = result_processor(result, updater)
        counts.update(stats)
    updater.commit()
    return counts

def _update_database_from_result(
        result: PipelineResult,
        updater: LmdbUpdater,
        key_suffix: str,
        ts_field: str,
        success_callback: Callable[[Any, Any, int], None] = None) -> Counter:
    """Generic database updater for embedding results."""
    key = f'{result.row.id}:{key_suffix}'
    ts = int(time.time())
    counts = Counter()
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
        updater: LmdbUpdater,
        data_extractor: Callable[[PipelineResult], Any] = lambda x: x.data):
    """Factory for creating embedding stages."""
    async def embed_stage(input_data: PipelineResult) -> PipelineResult:
        row = input_data.row
        key = f'{row.id}:{key_suffix}'
        
        # Skip if already in lmdb
        if key in updater:
            logger.info(f' skipping {row}, key={key} already in lmdb')
            return PipelineResult(row=row, data=None, error=None)
        try:
            data_to_embed = data_extractor(input_data)
            if not data_to_embed:
                logger.debug(f'Skipping {row} with no data to embed')
                return PipelineResult(row=row, data=None, error=None)
            embedding = await embed_func.single_async(data_to_embed, model=model)
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
    with db_session:
        rows = q.filter(lambda c: c.otype in ('text', 'link') and not c.embed_ts).limit(limit)
        if not rows:
            return Counter()
        logger.info(f'Updating embeddings for upto {len(rows)} text rows: {rows[:5]}...')
    
    updater = LmdbUpdater(lmdb_path, n_procs=1)
    
    # Stage 1: Extract text content from rows
    async def extract_text_stage(row):
        if not row.md:
            return PipelineResult(row=row, data='')
        text = ''
        if row.otype == 'text' and 'text' in row.md:
            text = row.md['text']
        elif row.otype == 'link' and 'title' in row.md:
            text = f"{row.md['title']}: {row.url}"
        return PipelineResult(row=row, data=text)

    # Stage 2: Generate text embeddings using factory
    embed_stage = _create_embedding_stage(
        embed_func=embed_text,
        model='qwen_emb',
        key_suffix='text',
        updater=updater,
        data_extractor=lambda x: x.data
    )
    
    # Success callback for text embeddings
    def text_success_callback(row, embedding, ts):
        updater.add(f'{row.id}:text', embedding=embedding, metadata=dict(embed_ts=ts))
        with db_session:
            row.embed_ts = ts
    
    # Run pipeline
    stats = await _run_embedding_pipeline(
        rows=rows,
        stages=[extract_text_stage, embed_stage],
        pipeline_config=dict(
            q_size=[20, 10],
            concurrency=[2, 3],
            exc_policy='stop'
        ),
        updater=updater,
        result_processor=lambda result, updater: _update_database_from_result(
            result, updater, 'text', 'embed_ts', text_success_callback
        )
    )
    
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
                return PipelineResult(row=row, data=path)
            except Exception as e:
                logger.warning(f'Error downloading image for row id={row.id}, url={row.url}, path={path}: {e}')
                return PipelineResult(row=row, data='')

    # Stage 2: Generate embeddings using factory
    def validate_image_path(input_data: PipelineResult) -> str:
        path = input_data.data
        if not path or not exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f'File not found or empty: {path}')
        return path

    embed_stage = _create_embedding_stage(
        embed_func=embed_image,
        model='mobilenet',  #FIXME: should be 'clip'
        key_suffix=IMAGE_SUFFIX,
        updater=updater,
        data_extractor=validate_image_path
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
            concurrency=[5, 3],
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


async def update_image_descriptions(q,
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
    if not vlm_prompt or not vlm_model:
        return Counter()
    
    with db_session:
        rows = q.filter(lambda c: c.otype == 'image' and c.embed_ts is not None and c.embed_ts > 0 and c.explored_ts is None).limit(limit)
        if not rows:
            return Counter()
        logger.info(f'Updating descriptions for {len(rows)} image rows: {rows[:5]}...')
    
    updater = LmdbUpdater(lmdb_path, n_procs=1)
    
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
            return PipelineResult(row=row, data=desc)
        except Exception as e:
            logger.warning(f'Error generating desc for image {row}, path={path}: {e}')
            return PipelineResult(row=row, data='')

    # Stage 2: Generate text embeddings using factory
    embed_stage = _create_embedding_stage(
        embed_func=embed_text,
        model='qwen_emb',
        key_suffix='text',
        updater=updater,
        data_extractor=lambda x: x.data
    )
    
    # We need a custom approach for descriptions since we need both desc and embedding
    # Let's use the original approach but with some refactored components
    pipeline = ProducerConsumerPipeline(
        funcs=[vlm_stage, embed_stage],
        q_size=[50, 20],
        concurrency=[3, 2],
        exc_policy='stop'
    )
    
    counts = Counter()
    async for result in pipeline.run_async(rows):
        # Handle the complex case where we need both description and embedding
        # We lost the description, need to get it from row.md
        desc = result.row.md.get('desc', '') if result.row.md else ''
            
        key = f'{result.row.id}:text'
        with db_session:
            if result.data is not None and desc:
                ts = int(time.time())
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
    # Import here to avoid circular imports
    from .model import Item
    
    if limit <= 0:
        limit = 10000000
    
    with db_session:
        q = Item.select(lambda c: (ids is None or c.id in ids))
        if source is not None:
            q = q.filter(lambda c: c.source == source)
        q = q.order_by(Item.id.desc())
    
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
                                  limit=limit//15,
                                  **common_kw)
    )
    
    text_stats, image_stats, desc_stats = await asyncio.gather(text_task, image_task, desc_task)
    
    # Merge all counters with prefixes to avoid key conflicts
    ret = Counter()
    for k, v in text_stats.items():
        ret[f'text_{k}'] = v
    for k, v in image_stats.items():
        ret[f'image_{k}'] = v
    for k, v in desc_stats.items():
        ret[f'desc_{k}'] = v
    
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
    from nkpylib.thread_utils import run_async
    
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
    print(f'Sync Done with update_embeddings, got {ret}')
    return ret


def cleanup_embeddings(lmdb_path: str):
    """Cleans up discrepancies between our sqlite and lmdb.

    Note that this doesn't modify the lmdb at all, only the sqlite.
    """
    # Import here to avoid circular imports
    from .model import Item
    
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
