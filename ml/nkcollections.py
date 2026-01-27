"""An abstraction over collections to make it easy to filter/sort/etc

"""
#TODO Embeddings status page
#TODO propagate likes to source sites if possible
#TODO fast scanning/detector of all images?

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time

from argparse import ArgumentParser
from collections import defaultdict
from functools import cache
from os.path import abspath, exists, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread
from typing import Any

import joblib
import termcolor
import tornado.web

from tornado.web import RequestHandler
from pony.orm import (
    composite_index,
    Database,
    db_session,
    desc,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    select,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore

from nkpylib.ml.client import call_vlm, embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkpony import init_sqlite_db, GetMixin, recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

sql_db = Database()

J = lambda obj: json.dumps(obj, indent=2)


class BackgroundWorker(abc.ABC):
    """Abstract base class for long-running background worker threads.

    Subclasses must implement the `process_task` method to define how tasks are processed.

    Communication with the worker happens via thread-safe queues:
    - `add_task(task)` to send work to the background thread
    - `get_result()` to retrieve completed results
    - `get_all_results()` to retrieve all pending results

    The worker runs in a daemon thread and will automatically stop when the main process exits.
    """
    def __init__(self, name: str = "BackgroundWorker"):
        self.name = name
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.running = False
        self.thread: Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self.running:
                logger.warning(f"{self.name} is already running")
                return
            self.running = True
            self.thread = Thread(target=self._worker_loop, daemon=True, name=self.name)
            self.thread.start()
            logger.info(f"Started {self.name}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker thread.
        - timeout: Maximum time to wait for the thread to finish
        """
        with self._lock:
            if not self.running:
                return
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(f"{self.name} did not stop within {timeout}s")
            else:
                logger.info(f"Stopped {self.name}")

    def _worker_loop(self) -> None:
        """Main worker loop that processes tasks from the input queue."""
        logger.debug(f"{self.name} worker loop started")
        while self.running:
            try:
                # Get task with timeout so we can check self.running periodically
                task = self.input_queue.get(timeout=1.0)
                try:
                    result = self.process_task(task)
                    if result is not None:
                        self.output_queue.put(result)
                except Exception as e:
                    logger.error(f"{self.name} error processing task {task}: {e}")
                    # Optionally put error result in output queue
                    error_result = self._handle_task_error(task, e)
                    if error_result is not None:
                        self.output_queue.put(error_result)
                finally:
                    self.input_queue.task_done()
            except Empty:
                # Timeout - continue loop to check self.running
                continue
            except Exception as e:
                logger.error(f"{self.name} unexpected error in worker loop: {e}")
        logger.debug(f"{self.name} worker loop finished")

    @abc.abstractmethod
    def process_task(self, task: Any) -> Any:
        """Process a single task and return the result.

        This method must be implemented by subclasses.

        - task: The task data from add_task()
        - Returns: Result to be put in output queue, or None to skip
        """
        pass

    def _handle_task_error(self, task: Any, error: Exception) -> Any:
        """Handle errors that occur during task processing.

        Override this method to customize error handling.

        - task: The task that caused the error
        - error: The exception that was raised
        - Returns: Error result to put in output queue, or None to skip
        """
        return dict(error=str(error), task=task)

    def add_task(self, task: Any) -> None:
        """Add a task to the input queue for processing.

        - task: Any data that will be passed to process_task()
        """
        if not self.running:
            raise RuntimeError(f"{self.name} is not running")

        self.input_queue.put(task)

    def get_result(self) -> Any | None:
        """Get one result from the output queue, or None if no results available."""
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None

    def get_all_results(self) -> list[Any]:
        """Get all available results from the output queue."""
        results = []
        while True:
            result = self.get_result()
            if result is None:
                break
            results.append(result)
        return results

    def queue_sizes(self) -> dict[str, int]:
        """Get current queue sizes for monitoring."""
        return dict(
            input_queue_size=self.input_queue.qsize(),
            output_queue_size=self.output_queue.qsize()
        )

    def is_running(self) -> bool:
        """Check if the worker thread is running."""
        return self.running and self.thread is not None and self.thread.is_alive()


class LikesWorker(BackgroundWorker):
    """Background worker that maintains likes-based image classifier state.

    Continuously monitors for changes in liked images and updates classifier scores.
    Maintains a dict of scores (id->score), tracks the last computed time, and
    saves classifiers to disk for persistence.
    """

    def __init__(self,
                 embs: Embeddings,
                 classifiers_dir: str,
                 name: str = "LikesWorker",
                 method: str = 'rbf',
                 neg_factor: float = 10,
                 sleep_interval: float = 30.0,
                 exclude_top_n: int = 2000):
        super().__init__(name)
        self.embs = embs
        self.classifiers_dir = classifiers_dir
        self.method = method
        self.neg_factor = neg_factor
        self.sleep_interval = sleep_interval
        self.exclude_top_n = exclude_top_n

        # State tracking
        self.scores: dict[str, float] = {}
        self.last_computed_time: float = 0
        self.last_pos_ids: frozenset[int] = frozenset()
        self.last_classifier = None
        self.last_scaler = None

        # Ensure classifiers directory exists
        os.makedirs(self.classifiers_dir, exist_ok=True)

    def process_task(self, task: Any) -> Any:
        """Process a likes classification task.

        Task can be:
        - 'update': Check for changes and update classifier if needed
        - 'force': Force update classifier regardless of changes
        """
        if task == 'update':
            return self._update_classifier()
        elif task == 'force':
            return self._update_classifier(force=True)
        else:
            logger.warning(f"Unknown task type: {task}")
            return None

    def _get_current_pos_ids(self) -> frozenset[int]:
        """Get current set of liked image IDs."""
        with db_session:
            liked_images = Rel.get_likes(valid_types=['image'])
            # Filter to only those with embeddings
            pos_ids = frozenset(img.id for img in liked_images
                               if img.embed_ts and img.embed_ts > 0)
        return pos_ids

    def _get_all_image_ids(self) -> list[int]:
        """Get all image IDs that have embeddings, excluding top N by ID."""
        with db_session:
            all_images = Item.select(lambda c: c.otype == 'image' and c.embed_ts > 0)
            all_ids = [img.id for img in all_images]
            # Sort by ID descending and exclude top N
            all_ids.sort(reverse=True)
            if len(all_ids) > self.exclude_top_n:
                all_ids = all_ids[self.exclude_top_n:]
        return all_ids

    def _update_classifier(self, force: bool = False) -> dict[str, Any]:
        """Update the classifier if needed."""
        current_pos_ids = self._get_current_pos_ids()

        # Check if we need to update
        if not force and current_pos_ids == self.last_pos_ids:
            logger.debug(f"No change in liked images, sleeping for {self.sleep_interval}s")
            time.sleep(self.sleep_interval)
            return dict(status='no_change', pos_count=len(current_pos_ids))

        if not current_pos_ids:
            logger.info("No liked images found, skipping classifier update")
            return dict(status='no_positives', pos_count=0)

        try:
            # Reload embeddings keys
            self.embs.reload_keys()

            # Get all image IDs for classification
            all_ids = self._get_all_image_ids()
            to_cls = [f'{id}:image' for id in all_ids]

            # Prepare positive samples
            pos = [f'{id}:image' for id in current_pos_ids]

            # Prepare negative samples (exclude positives and top N IDs)
            with db_session:
                neg_candidates = Item.select(lambda c:
                    c.otype == 'image' and
                    c.embed_ts > 0 and
                    c.id not in current_pos_ids
                )
                # Further exclude top N by ID
                neg_ids = [img.id for img in neg_candidates]
                neg_ids.sort(reverse=True)
                if len(neg_ids) > self.exclude_top_n:
                    neg_ids = neg_ids[self.exclude_top_n:]

                # Sample negatives
                neg_sample_size = min(len(neg_ids), int(len(pos) * self.neg_factor))
                neg_ids = random.sample(neg_ids, neg_sample_size)
                neg = [f'{id}:image' for id in neg_ids]

            logger.info(f"Training classifier with {len(pos)} pos, {len(neg)} neg, "
                       f"classifying {len(to_cls)} images")

            # Train and run classifier
            t0 = time.time()
            classifier, scores = self.embs.train_and_run_classifier(
                pos=pos, neg=neg, to_cls=to_cls, method=self.method
            )
            t1 = time.time()

            # Convert scores to int keys
            self.scores = {k.split(':')[0]: v for k, v in scores.items()}

            # Save classifier to disk
            classifier_path = os.path.join(
                self.classifiers_dir,
                f'likes_classifier_{int(time.time())}.joblib'
            )

            # Get scaler from the embeddings if available
            scaler = getattr(self.embs, '_last_scaler', None)

            save_data = self.embs.save_classifier(
                classifier_path,
                classifier,
                scaler=scaler,
                method=self.method,
                neg_factor=self.neg_factor,
                pos_count=len(pos),
                neg_count=len(neg),
                total_classified=len(to_cls),
                training_time=t1 - t0
            )

            # Update state
            self.last_computed_time = time.time()
            self.last_pos_ids = current_pos_ids
            self.last_classifier = classifier
            self.last_scaler = scaler

            logger.info(f"Updated likes classifier in {t1-t0:.2f}s, "
                       f"saved to {classifier_path}")

            return dict(
                status='updated',
                pos_count=len(pos),
                neg_count=len(neg),
                scores_count=len(self.scores),
                training_time=t1 - t0,
                classifier_path=classifier_path
            )

        except Exception as e:
            logger.error(f"Error updating likes classifier: {e}")
            return dict(status='error', error=str(e))

    def get_scores(self) -> dict[str, float]:
        """Get current classifier scores."""
        return self.scores.copy()

    def get_last_computed_time(self) -> float:
        """Get timestamp of last classifier computation."""
        return self.last_computed_time

    def force_update(self) -> None:
        """Force an update of the classifier."""
        self.add_task('force')

    def request_update(self) -> None:
        """Request an update check (will skip if no changes)."""
        self.add_task('update')


def save_classifier(classifier, scaler, path: str, **extra_params) -> dict[str, Any]:
    """Save classifier with scaler and additional parameters"""
    model_data = {
        'classifier': classifier,
        'scaler': scaler,
        'params': {
            'created_at': time.time(),
            'n_features': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None,
            **extra_params
        }
    }
    joblib.dump(model_data, path)
    return model_data


def load_classifier(path: str) -> dict[str, Any]:
    """Load classifier with all associated data"""
    return joblib.load(path)


async def maybe_dl(url: str, path: str, fetch_delay: float=0.1) -> bool:
    """Downloads the given url to the given dir if it doesn't already exist there (and is not empty).

    Returns if we actually downloaded the file.
    """
    if exists(path) and os.path.getsize(path) > 0:
        return False
    logger.debug(f'downloading image {url} -> {path}')
    r = await make_request_async(url, headers={'Accept': 'image/*,video/*'}, min_delay=fetch_delay)
    try:
        os.makedirs(dirname(path), exist_ok=True)
    except Exception as e:
        pass
    with open(path, 'wb') as f:
        f.write(r.content)
    return True

class Item(sql_db.Entity, GetMixin): # type: ignore[name-defined]
    """Each individual item, which can include users, posts, images, links, etc."""
    id = PrimaryKey(int, auto=True)
    source = Required(str)
    stype = Required(str)
    otype = Required(str, index=True)
    url = Required(str, index=True)
    composite_index(source, stype, otype, url)
    name = Optional(str, index=True)
    parent = Optional('Item', reverse='children', index=True) # type: ignore[var-annotated]
    # time of the actual item
    ts = Required(float, default=lambda: time.time(), index=True)
    # time we added this to our database
    added_ts = Required(float, default=lambda: time.time())
    # time we last explored this item
    explored_ts = Optional(float, index=True)
    # time we last saw this item
    seen_ts = Optional(float, index=True)
    # time we last extracted embeddings for this item
    embed_ts = Optional(float, index=True)
    # all other metadata
    md = Optional(Json)
    # cumulative seconds spent on this item
    dwell_time = Required(float, default=0.0)
    children = Set('Item', reverse='parent') # type: ignore[var-annotated]
    rel_srcs = Set('Rel', reverse='src') # type: ignore[var-annotated]
    rel_tgts = Set('Rel', reverse='tgt') # type: ignore[var-annotated]

    @classmethod
    def get_me(cls):
        """Returns the "me" Item row."""
        with db_session:
            me = cls.get(source='me')
            return me

    @classmethod
    def image_path(cls, row, images_dir: str='') -> str:
        """Returns the image path for a given image `row`"""
        if not row:
            return ''
        url = row.url
        ext = row.md.get('ext', url.split('.')[-1])
        mk = row.md.get('media_key', row.id)
        path = abspath(join(images_dir, f'{mk}.{ext}'))
        return path


    @classmethod
    async def update_text_embeddings(cls, q: Query, limit: int, lmdb_path: str, **kw) -> int:
        """Updates text embeddings for the given query `q`.

        This select 'text' and 'link' otypes and updates their embeddings.
        Returns the number of embeddings updated.
        """
        with db_session:
            rows = q.filter(lambda c: c.otype in ('text', 'link') and not c.embed_ts).limit(limit)
            if not rows:
                return 0
            logger.info(f'Updating embeddings for upto {len(rows)} text rows: {rows[:5]}...')
            inputs = []
            row_by_key = {}
            for r in rows:
                if not r.md:
                    continue
                key = f'{r.id}:text'
                row_by_key[key] = r
                if r.otype == 'text' and 'text' in r.md:
                    inputs.append((key, r.md['text']))
                elif r.otype == 'link' and 'title' in r.md:
                    inputs.append((key, f"{r.md['title']}: {r.url}"))

        def md_func(key, input):
            row = row_by_key[key]
            ts = int(time.time())
            with db_session:
                row.embed_ts = ts
            return dict(embed_ts=ts)

        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        n_text = await loop.run_in_executor(
            None,
            lambda: batch_extract_embeddings(inputs=inputs, embedding_type='text', md_func=md_func, db_path=lmdb_path, **kw)
        )
        if n_text > 0:
            logger.info(f'  Updated embeddings for {n_text} text rows')
        return n_text

    @classmethod
    async def update_image_embeddings(cls,
                                      q: Query,
                                      lmdb_path: str,
                                      images_dir: str,
                                      limit: int,
                                      fetch_delay: float=0.1,
                                      **kw) -> int:
        """Updates images embeddings for the given query `q`.

        This select 'image' rows, downloads them if needed, and updates their embeddings.
        Returns the number of embeddings updated.
        """
        with db_session:
            rows = q.filter(lambda c: c.otype == 'image' and not c.embed_ts).limit(limit)
        if not rows:
            return 0
        updater = LmdbUpdater(lmdb_path, n_procs=1)
        logger.info(f'Updating embeddings for upto {len(rows)} image rows: {rows[:5]}...')
        # kick off downloads
        async def dl_image(row):
            with db_session:
                path = cls.image_path(row, images_dir=images_dir)
                await maybe_dl(row.url, path, fetch_delay=fetch_delay)
            return row, path

        download_tasks = [dl_image(row) for row in rows]
        # kick off embeddings as downloads complete
        embedding_tasks = []
        async def embed_image_task(row, key, path):
            try:
                emb = await embed_image.single_async(path, model='clip', use_cache=kw.get('use_cache', True))
            except Exception as e:
                logger.warning(f'Error embedding image for row id={row.id}, path={path}: {e}')
                emb = None
            return (row, key, emb)

        done = set()
        for task in asyncio.as_completed(download_tasks):
            row, path = await task
            key = f'{row.id}:image'
            if key in updater or key in done:
                logger.info(f' skipping image {row}, key={key} already in lmdb')
                continue
            done.add(key)
            embedding_tasks.append(embed_image_task(row, key, path))
        # kick off postprocessing of embeddings as they complete
        n_images = 0
        for task in asyncio.as_completed(embedding_tasks):
            row, key, emb = await task
            ts = int(time.time())
            with db_session:
                # update the lmdb and sqlite
                logger.debug(f' emb for image {row}, key={key}, {emb[:10] if emb is not None else "failed"}')
                if emb is None: # error
                    updater.add(key, metadata=dict(embed_ts=ts, error='image embedding failed'))
                    row.embed_ts = -1 #TODO in the future decrement this
                else:
                    updater.add(key, embedding=emb, metadata=dict(embed_ts=ts))
                    row.embed_ts = ts
            n_images += 1
        updater.commit()
        if n_images > 0:
            logger.info(f'  Updated embeddings for {n_images} images')
        return n_images

    @classmethod
    async def update_image_descriptions(cls,
                                        q,
                                        lmdb_path: str,
                                        images_dir: str,
                                        limit: int,
                                        vlm_prompt: str|None='briefly describe this image',
                                        sys_prompt: str|None=None,
                                        vlm_model: str='fastvlm',
                                        **kw) -> int:
        """Updates image descriptions using VLM for images that have been explored.

        Filters to images where explored_ts is not null, generates descriptions via VLM,
        embeds the descriptions, and updates both LMDB and SQLite metadata.

        Returns the number of descriptions updated.
        """
        if not vlm_prompt or not vlm_model:
            return 0
        updater = LmdbUpdater(lmdb_path, n_procs=1)
        with db_session:
            rows = q.filter(lambda c: c.otype == 'image' and c.embed_ts is not None and c.embed_ts > 0 and c.explored_ts is None).limit(limit)
            if not rows:
                return 0
            logger.info(f'Updating descriptions for {len(rows)} image rows: {rows[:5]}...')
        # Create VLM tasks for all images
        async def vlm_task(row):
            if sys_prompt:
                messages = [
                    dict(role='system', content=sys_prompt),
                    dict(role='user', content=vlm_prompt)
                ]
            else:
                messages = vlm_prompt
            path = cls.image_path(row, images_dir=images_dir)
            try:
                desc = await call_vlm.single_async((path, messages), model=vlm_model)
            except Exception as e:
                logger.warning(f'Error generating desc for image {row}, path={path}: {e}')
                desc = ''
            return row, desc

        vlm_tasks = [vlm_task(row) for row in rows]
        # Process VLM results as they complete
        n_descs = 0
        for task in asyncio.as_completed(vlm_tasks):
            row, desc = await task
            key = f'{row.id}:text'
            with db_session:
                if desc:
                    row.md['desc'] = desc
                    try:
                        text_embedding = embed_text.single(desc, model='qwen_emb')
                        ts = int(time.time())
                        updater.add(key, embedding=text_embedding, metadata=dict(desc=desc, embed_ts=ts))
                        row.explored_ts = ts
                        n_descs += 1
                    except Exception as e:
                        logger.warning(f'Error embedding description {desc} for {row}: {e}')
                        updater.add(key, metadata=dict(desc=desc, embed_ts=time.time(), error='text embedding failed'))
                else: # failed to get description
                    row.explored_ts = -1 #TODO decrement on more errors
        updater.commit()
        if n_descs > 0:
            logger.info(f'Updated descriptions for {n_descs} images')
        return n_descs

    @classmethod
    async def update_embeddings_async(cls,
                          lmdb_path: str,
                          images_dir: str,
                          ids: list[int]|None=None,
                          vlm_prompt: str|None='briefly describe this image',
                          sys_prompt: str|None=None,
                          vlm_model: str='fastvlm',
                          limit: int=-1,
                          fetch_delay: float=0.1,
                          **kw) -> int:
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

        We return the total number of embeddings updated.
        """
        if limit <= 0:
            limit = 10000000
        q = cls.select(lambda c: (ids is None or c.id in ids))
        q = q.order_by(desc(Item.id))
        common_kw = dict(q=q, lmdb_path=lmdb_path, **kw)
        # start async tasks for all 3 subfunctions
        text_task = asyncio.create_task(cls.update_text_embeddings(limit=limit*2, **common_kw))
        image_task = asyncio.create_task(
            cls.update_image_embeddings(images_dir=images_dir, fetch_delay=fetch_delay, limit=limit, **common_kw)
        )
        desc_task = asyncio.create_task(
            cls.update_image_descriptions(vlm_prompt=vlm_prompt,
                                          sys_prompt=sys_prompt,
                                          vlm_model=vlm_model,
                                          images_dir=images_dir,
                                          limit=limit*2,
                                          **common_kw)
        )
        n_text, n_images, n_descs = await asyncio.gather(text_task, image_task, desc_task)
        return n_text + n_images + n_descs

    @classmethod
    def update_embeddings(cls,
                          lmdb_path: str,
                          images_dir: str,
                          ids: list[int]|None=None,
                          vlm_prompt: str|None='briefly describe this image',
                          sys_prompt: str|None=None,
                          vlm_model: str='fastvlm',
                          limit: int=-1,
                          fetch_delay: float=0.1,
                          **kw) -> int:
        """Calls the async version"""
        return run_async(cls.update_embeddings_async(
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


class Rel(sql_db.Entity, GetMixin): # type: ignore[name-defined]
    """Relations between items"""
    src = Required('Item', reverse='rel_srcs') # type: ignore[var-annotated]
    tgt = Required('Item', reverse='rel_tgts') # type: ignore[var-annotated]
    rtype = Required(str)
    ts = Required(int)
    PrimaryKey(src, tgt, rtype, ts)
    md = Optional(Json)

    @classmethod
    @db_session
    def get_likes(cls, valid_types: list[str]|None=None) -> list[Item]:
        """Returns Items I've liked, optionally filtered to the given `valid_types`."""
        me = Item.get_me()
        like_rels = cls.select(lambda r: r.src == me and r.rtype == 'like')[:]
        ret = set()
        def maybe_add(obj):
            if valid_types is None or obj.otype in valid_types:
                ret.add(obj)

        for r in like_rels:
            # check the item itself
            try:
                ot = r.tgt.otype
            except UnrepeatableReadError: # tgt was deleted
                continue
            maybe_add(r.tgt)
            # also check its children
            for child in r.tgt.children.select():
                maybe_add(child)
        return list(ret)

def init_sql_db(path: str) -> Database:
    """Initializes the sqlite database at the given `path`"""
    init_sqlite_db(path, db=sql_db)
    with db_session:
        Item.upsert(get_kw=dict(
            source='me',
            stype='user',
            otype='user',
            url='me'
        ))
    return sql_db


class Source(abc.ABC):
    """Base class for all sources. Subclass this.

    Implement can_parse() and parse() methods if you want to handle custom inputs.
    """
    _registry: dict[str, Source] = {}  # Class variable to maintain map from names to Source classes

    def __init__(self,
                 name: str,
                 data_dir: str,
                 sqlite_path: str='',
                 lmdb_path: str='',
                 images_dir: str='',
                 classifiers_dir: str='',
                 **kw):
        self.name = name
        self.data_dir = data_dir
        self.sqlite_path = sqlite_path or join(data_dir, 'collection.sqlite')
        self.lmdb_path = lmdb_path or join(data_dir, 'embeddings.lmdb')
        self.images_dir = images_dir or join(data_dir, 'images')
        self.classifiers_dir = classifiers_dir or join(data_dir, 'classifiers')
        init_sql_db(self.sqlite_path)
        Source._registry[name] = self

    def __repr__(self) -> str:
        return f'Source<{self.name}>'

    @classmethod
    def can_parse(cls, url: str) -> bool:
        """Returns if this source can parse the given url"""
        return False

    def parse(self, url: str, **kw) -> Any:
        """Parses the given url and does whatever it wants."""
        raise NotImplementedError()

    @staticmethod
    def handle_url(url: str, **data) -> dict[str, Any]:
        """This is the main entry point to handle a given url.

        This finds the appropriate Source subclass that can parse the given url,
        and calls its parse() method with the given url and data.
        """
        for source_cls in Source.__subclasses__():
            if source_cls.can_parse(url):
                # subclasses must be instantiable with no args
                source = source_cls() # type: ignore[call-arg]
                result = source.parse(url, **data)
                return result
        raise NotImplementedError(f'No source found to parse url {url}')

    @classmethod
    def assemble_post(cls, post, children) -> dict:
        """Assembles a post, generically.

        This method can be overridden by subclasses for custom behavior.
        In this version, we take all children and add them to a subkey called "children".
        """
        assembled_post = recursive_to_dict(post)
        assembled_post['children'] = [recursive_to_dict(child) for child in children]
        # Extract media blocks for carousel functionality
        media_blocks = []
        for child in children:
            if child.otype in ['image', 'video']:
                if child.otype == 'image' and child.md and 'poster_for' in child.md:
                    continue
                media_blocks.append(dict(
                    type=child.otype,
                    data=recursive_to_dict(child)
                ))
        assembled_post['media_blocks'] = media_blocks
        return assembled_post

    @classmethod
    @db_session
    def assemble_posts(cls, posts: list[Item]) -> list[dict]:
        """Assemble complete posts with their children content.

        Takes a list of post `Item`s and returns a list of assembled post dictionaries
        with their children content nested appropriately based on source type.
        """
        assembled_posts = []
        #print(f'Got registry: {Source._registry}, {posts}')
        # for each post, get its children and assemble based on the source type
        for post in posts:
            src = Source._registry.get(post.source, Source)
            #print(f'for post source {post.source}, using src={src}, {Source._registry}')
            assembled_posts.append(src.assemble_post(post, post.children.select()))
        return assembled_posts

    @classmethod
    def cleanup_embeddings(cls, lmdb_path: str):
        """Cleans up discrepancies between our sqlite and lmdb.

        Note that this doesn't modify the lmdb at all, only the sqlite.
        """
        db = NumpyLmdb.open(lmdb_path, flag='r')
        keys_in_db = set(db.keys())
        n_missing = 0
        n_done = 0
        with db_session:
            # first deal with embeddings wrongly marked as done in sqlite but missing in lmdb
            def process_wrongly_done(rows: list[Item], key_suffix: str, ts_field: str):
                nonlocal n_missing
                for row in rows:
                    key = f'{row.id}:{key_suffix}'
                    if key not in keys_in_db:
                        logger.debug(f'Cleaning up {row} with missing key {key}')
                        setattr(row, ts_field, None)
                        n_missing += 1

            rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype in ('text', 'link'))
            process_wrongly_done(rows, 'text', 'embed_ts')
            rows = Item.select(lambda c: c.embed_ts is not None and c.embed_ts > 0 and c.otype == 'image')
            process_wrongly_done(rows, 'image', 'embed_ts')
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is not None and c.explored_ts > 0)
            process_wrongly_done(rows, 'text', 'explored_ts')
            # now deal with embeddings present in lmdb but not marked done in sqlite
            def process_missing_done(rows: list[Item], key_suffix: str, ts_field: str):
                nonlocal n_done
                for row in rows:
                    key = f'{row.id}:{key_suffix}'
                    if key in keys_in_db:
                        logger.debug(f'Marking done for {row} with existing key {key}')
                        d = db.md_get(key)
                        ts = d.get('embed_ts', d.get('embedding_ts', int(time.time())))
                        setattr(row, ts_field, int(time.time()))
                        n_done += 1

            rows = Item.select(lambda c: c.otype in ('text', 'link') and c.embed_ts is None)
            process_missing_done(rows, 'text', 'embed_ts')
            rows = Item.select(lambda c: c.otype == 'image' and c.embed_ts is None)
            process_missing_done(rows, 'image', 'embed_ts')
            rows = Item.select(lambda c: c.otype == 'image' and c.explored_ts is None)
            process_missing_done(rows, 'text', 'explored_ts')
        del db
        logger.info(f'Cleaned up {n_missing} missing and {n_done} done embeddings')

    @db_session
    def update_embeddings(self, **kw):
        """Updates the embeddings for this Source.

        By default, this just calls Item.update_embeddings, filtered to the ids of items from
        this source. If 'ids' is given in kw, we further filter to those ids.

        We pass all `kw` to Item.update_embeddings.
        """
        source_ids = select(c.id for c in Item if c.source == self.name)[:]
        # if we had a list of input ids, filter to those
        if 'ids' in kw and kw['ids'] is not None:
            ids = [id for id in source_ids if id in kw['ids']]
        else:
            ids = source_ids
        Item.update_embeddings(lmdb_path=self.lmdb_path, images_dir=self.images_dir, ids=ids, **kw)


class MyBaseHandler(BaseHandler):
    @property
    def sql_db(self) -> Database:
        return self.application.sql_db # type: ignore[attr-defined]

    @property
    def lmdb(self) -> NumpyLmdb:
        return self.application.lmdb # type: ignore[attr-defined]

    @property
    def embs(self) -> Embeddings:
        return self.application.embs # type: ignore[attr-defined]

    @property
    @cache
    def all_otypes(self) -> list[str]:
        with db_session:
            otypes = list(select(r.otype for r in Item)) # type: ignore[attr-defined]
            return otypes

class GetHandler(MyBaseHandler):
    @db_session
    def build_query(self, data: dict[str, Any]) -> Query:
        """Builds up the database query to get items matching the given data filters.

        For string fields, the value can be a string (exact match) or a list of strings (any of).
        For numeric fields, the value can be a number (exact match) or a string with an operator
        such as '>=123', '<=456', '>789', '<1011', '!=1213'.
        """
        q = Item.select()
        # Handle ids parameter (num spec)
        if 'ids' in data:
            ids = parse_num_spec(data['ids'])
            q = q.filter(lambda c: c.id in ids)
        # Handle string fields
        string_fields = ['source', 'stype', 'otype', 'url', 'name']
        for field in string_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    q = q.filter(lambda c: getattr(c, field) in value)
                else:
                    q = q.filter(lambda c: getattr(c, field) == value)
        # Handle parent field
        if 'parent' in data:
            parent_id = int(data['parent'])
            q = q.filter(lambda c: c.parent and c.parent.id == parent_id)
        # Handle numeric fields
        numeric_fields = ['ts', 'added_ts', 'explored_ts', 'seen_ts', 'embed_ts']
        for field in numeric_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str): # Parse operator
                    value = value.replace(' ', '')
                    if value.startswith('>='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) >= threshold)
                    elif value.startswith('<='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) <= threshold)
                    elif value.startswith('!='):
                        threshold = float(value[2:])
                        q = q.filter(lambda c: getattr(c, field) != threshold)
                    elif value.startswith('>'):
                        threshold = float(value[1:])
                        q = q.filter(lambda c: getattr(c, field) > threshold)
                    elif value.startswith('<'):
                        threshold = float(value[1:])
                        q = q.filter(lambda c: getattr(c, field) < threshold)
                    else: # No operator, treat as exact match
                        q = q.filter(lambda c: getattr(c, field) == float(value))
                else: # Exact match
                    q = q.filter(lambda c: getattr(c, field) == value)
        if 'order' in data: # check for ordering info
            # order value will be a field name, optionally prefixed by - for descending
            order_field = data['order']
            if order_field.startswith('-'):
                q = q.order_by(lambda c: desc(getattr(c, order_field[1:])))
            else:
                q = q.order_by(lambda c: getattr(c, order_field))
        else: # sort by id descending
            q = q.order_by(lambda c: desc(c.id))
        # if there was a limit parameter, set it
        if 'limit' in data:
            q = q.limit(int(data['limit']))
        return q

    def post(self):
        data = json.loads(self.request.body)
        print(f'GetHandler got data={data}')
        # Build query conditions
        with db_session:
            q = self.build_query(data)
            items = q[:]
            if data.get('assemble_posts', False):
                rows = {r['id']: r for r in Source.assemble_posts(items)}
            else:
                rows = {r.id: recursive_to_dict(r) for r in q}
            cur_ids = list(rows.keys())
            # fetch all rels with source = me and tgt in ids and update the appropriate rows
            me = Item.get_me()
            rels = Rel.select(lambda r: r.src == me and r.tgt.id in cur_ids)
            for rel in rels:
                tgt_id = rel.tgt.id
                if 'rels' not in rows[tgt_id]:
                    rows[tgt_id]['rels'] = {}
                rel_md = rel.md or {}
                rel_md['ts'] = rel.ts
                rows[tgt_id]['rels'][rel.rtype] = rel_md
        self.write(dict(rows=rows, allOtypes=self.all_otypes))

class SourceHandler(MyBaseHandler):
    def post(self):
        """Set a source url to parse."""
        data = json.loads(self.request.body)
        url = data.pop('url', '')
        print(f'SourceHandler got url={url}, {data}')
        # find a source that can parse this url
        parsed = Source.handle_url(url, **data)
        print(f'parsed to {parsed}')
        if 0:
            parsed_params = '&'.join([f'{k}={v}' for k, v in parsed.items()])
            self.redirect(f"/get/0-100000?{parsed_params}")
        else:
            # send the parsed result to the client
            self.write(parsed)

class DwellHandler(MyBaseHandler):
    """Update dwell times of objects"""
    def post(self):
        pass


class ActionHandler(MyBaseHandler):
    """The user took some action, which we will store in our `rels` table"""
    def post(self):
        data = json.loads(self.request.body)
        action = data.get('action', '')
        print(f'ActionHandler got action={action}, {data}')
        self.write(dict(action=action, status='ok'))
        # create a new rel
        with db_session:
            me = Item.get_me()
            print(f'Got me={me}')
            get_kw = dict(src=me, tgt=Item[int(data['id'])], rtype=action)
            match action:
                case 'like': # create or update the rel (only 1 like possible)
                    r = Rel.upsert(get_kw=get_kw, ts=int(time.time()))
                case 'unlike': # delete the rel if it exists
                    get_kw['rtype'] = 'like'
                    r = Rel.get(**get_kw)
                    if r:
                        r.delete()
                case _:
                    print(f'Unknown action {action}')

class ClassifyHandler(MyBaseHandler):
    def _handle_pos(self, pos):
        """Simple positive only classifier"""
        # for now, we use the first pos to set the otype to search over
        with db_session:
            otype = Item[pos[0]].otype
        pos = [f'{p}:{otype}' for p in pos]
        all_keys = [k for k in self.embs if k.endswith(f':{otype}')]
        print(f'ClassifyHandler got pos={pos}, {otype}, {len(all_keys)} total keys: {all_keys[:5]}...')
         # get similar from embs
        ret = self.embs.similar(pos, all_keys=all_keys, method='nn')
        print(f'Got ret {ret}')
        scores, curIds = zip(*ret)
        curIds = [p.split(':')[0] for p in curIds]
        self.write(dict(pos=pos,
                        scores={id: score for id, score in zip(curIds, scores)},
                        curIds=curIds))

    async def _handle_likes(self,
                            cur_ids: list[int]|None=None,
                            otypes=['image'],
                            feature_types=None,
                            method: str='rbf',
                            neg_factor: float=10,
                            **kw):
        """Likes-based classifier.

        There are a few different high-level params that we care about:
        - otypes: what input types we want to process (images, text, posts, etc)
          - for now, we only do images
        - feature_types: what type of features to use for classification
          - for image items:
            - image embeddings
            - text embeddings of image descriptions
            - text description tag embeddings
          - for text items:
            - text embeddings
          - for post items:
            - average text/image embeddings of all children
            - some sort of post-specific embeddings?
        - classifier saving/loading:
          - we could train a new classifier or load an existing classifier or none
          - we run inference with the chosen classifier, or not?
          - we could save the trained classifier for future use or not

        Other minor params:
        - method: classification method (default rbf)
        - neg_factor: how many negative samples per positive sample to use
        """
        images = set()
        if cur_ids is not None:
            cur_ids = set(int(i) for i in cur_ids)
            to_cls = [f'{id}:image' for id in cur_ids]
        else:
            to_cls = [k for k in self.embs if k.endswith(':image')]
        with db_session:
            # first get pos images from likes
            images.update(Rel.get_likes(valid_types=['image']))
            # filter down to only those with embeddings
            pos = [img for img in images if img.embed_ts and img.embed_ts > 0]
            pos_ids = [p.id for p in pos]
            # get a bunch of random negative images
            neg = list(Item.select(lambda c: c.otype == 'image' and c.embed_ts > 0 and c.id not in pos_ids))
            # remove any to_cls from it
            if cur_ids is not None:
                neg = [n for n in neg if n.id not in cur_ids]
            neg = random.sample(neg, min(len(neg), len(pos)*neg_factor))
        # train and run the classifier
        pos = [f'{r.id}:image' for r in pos]
        neg = [f'{r.id}:image' for r in neg]
        # Run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        t0 = time.time()
        cls, scores = await loop.run_in_executor(
            None,
            lambda: self.embs.train_and_run_classifier(pos=pos, neg=neg, to_cls=to_cls, method=method)
        )
        t1 = time.time()
        scores = {k.split(':')[0]: v for k, v in scores.items()}
        self.write(dict(
            msg=f'Likes image classifier with {len(pos)} pos, {len(neg)} neg, {len(scores)} scores in {t1 - t0:.2f}s',
            pos=pos, neg=neg, scores=scores))

    async def post(self):
        self.embs.reload_keys()
        # figure out what kind of classification we're doing
        data = json.loads(self.request.body)
        pos = data.get('pos', [])
        if pos:
            return self._handle_pos(pos)
        cls_type = data.get('type', '')
        if cls_type == 'likes':
            return await self._handle_likes(**data)


class ClusterHandler(MyBaseHandler):
    """Cluster objects semi-automatically.

    Call with manually labeled clusters: {id: cluster_num, ...} and ids (list of all ids to
    consider), and returns suggested clusters for all ids with scores, as:
        clusters={id: {num=cluster_num, score=score}, ...}

    """
    def post(self):
        data = json.loads(self.request.body)
        print(f'In clustering, got manual clusters {data["clusters"]}')
        # randomly assign cluster nums and scores for now, making sure that the manually labeled
        # clusters are preserved
        self.embs.reload_keys()
        manual_clusters = data.get('clusters', {})
        labels = {f'{id}:text': num for id, num in manual_clusters.items()}
        keys = {f'{id}:text' for id in data.get('ids', [])}
        method = data.get('method', 'rbf')
        # number of clusters is the max of the manual cluster num, unless the method is random
        n_clusters = max(manual_clusters.values())
        if method == 'random':
            n_clusters = 5
        clusters = self.embs.guided_clustering(labels=labels,
                                               keys=keys,
                                               method=method,
                                               n_clusters=n_clusters)
        clusters = {key.split(':')[0]: v for key, v in clusters.items()}
        ret = dict(msg=f'method: {method}', clusters=clusters)
        self.write(ret)


def web_main(port: int=12555, sqlite_path:str='', lmdb_path:str='', **kw):
    # load the data file from first arg
    parser = ArgumentParser(description="NK collections main")
    if sqlite_path:
        parser.add_argument('--sqlite_path', default=sqlite_path, help="The path to the sqlite database")
    else:
        parser.add_argument('sqlite_path', help="The path to the sqlite database")
    if lmdb_path:
        parser.add_argument('--lmdb_path', default=lmdb_path, help="The path to the lmdb database")
    else:
        parser.add_argument('lmdb_path', help="The path to the lmdb database")
    parser.add_argument('ignore', nargs='*', help="Ignore extra args")
    #FIXME add images dir and make it accessible via a static path
    kw = {}
    def post_parse_fn(args):
        print(f'Got args {args}')

    def on_start(app, args):
        app.sql_db = init_sql_db(args.sqlite_path)
        temp = NumpyLmdb.open(args.lmdb_path, flag='c')
        del temp
        app.embs = Embeddings([args.lmdb_path])

    more_handlers = [
        (r'/get', GetHandler),
        (r'/source', SourceHandler),
        (r'/action', ActionHandler),
        (r'/dwell', DwellHandler),
        (r'/classify', ClassifyHandler),
        (r'/cluster', ClusterHandler),
    ]

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/nkcollections.jsx',
                                port=port,
                                more_handlers=more_handlers,
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw,
                                on_start=on_start)

def embeddings_main(batch_size: int=20, loop_delay: float=10, **kw):
    """Runs embedding updates from the command line in an infinite loop.

    You probably want to call this from your subclass, after having initialized your Source.
    """
    sources = list(Source._registry.values())
    logger.info(f'Initialized embeddings main with {len(sources)} sources: {sources}')
    for s in sources:
        s.cleanup_embeddings(s.lmdb_path)
    while 1:
        t0 = time.time()
        try:
            for s in sources:
                s.update_embeddings(limit=batch_size, **kw)
        except Exception as e:
            logger.warning(f'Error in embeddings main loop: {e}')
        elapsed = time.time() - t0
        diff = loop_delay - elapsed
        time.sleep(max(0, diff))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
    web_main()
