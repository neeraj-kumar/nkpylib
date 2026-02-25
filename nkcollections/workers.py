#TODO migrate likes to sql?

from __future__ import annotations

import abc
import asyncio
import csv
import glob
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import traceback

from argparse import ArgumentParser
from collections import defaultdict, Counter
from functools import cache
from os.path import abspath, exists, join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread
from typing import Any, Callable

import joblib
import termcolor
import tornado.web

from pony.orm import (
    composite_index,
    commit,
    Database,
    db_session,
    desc,
    Json,
    Optional,
    PrimaryKey,
    Required,
    rollback,
    Set,
    select,
    set_sql_debug,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore
from tornado.web import RequestHandler
from tqdm import tqdm

from nkpylib.ml.client import embed_image, embed_text
from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings, compute_binary_classifier_stats
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import Item, Rel, Score, Source, ret_immediate, ACTIONS, J
from nkpylib.nkpony import init_sqlite_db, GetMixin, recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async, background_task
from nkpylib.web_utils import BaseHandler, simple_react_tornado_server, make_request, make_request_async

logger = logging.getLogger(__name__)

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

    def run(self) -> None:
        """Start the processing in the main thread (blocking)."""
        with self._lock:
            if self.running:
                logger.warning(f"{self.name} is already running")
                return
            self.running = True
            logger.info(f"Starting {self.name} in the main thread")
            self._worker_loop()

    def start(self) -> None:
        """Start the processing in a background worker thread."""
        with self._lock:
            if self.running:
                logger.warning(f"{self.name} is already running")
                return
            self.running = True
            self.thread = Thread(target=self._worker_loop, daemon=True, name=self.name)
            self.thread.start()
            logger.info(f"Started {self.name} in the background")

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

    def _compute_user_stats(self, user_id: int, item_ids: list[int]) -> dict[str, Any]:
        """Compute statistics for a specific `user_id`.

        This expects a list of item IDs that belong to the user to be passed in.

        Returns dict with user statistics including content counts, engagement, and scores.
        """
        timing = Counter()
        t0 = time.time()
        with db_session:
            timing['db_session_start'] = time.time() - t0
            t1 = time.time()
            # Get all items from this user
            user_items = Item.select(lambda i: i.id in item_ids)[:]
            timing['user_items_query'] = time.time() - t1
            t2 = time.time()
            t3 = time.time()
            # Initialize counters
            counts = Counter(ts=time.time())
            timing['init_counters'] = time.time() - t3
            t4 = time.time()
            like_scores = [score for id, score in self.like_scores.items() if int(id) in item_ids]
            timing['extract_like_scores'] = time.time() - t4
            t5 = time.time()
            for item in user_items:
                t_item_start = time.time()
                # otype counts
                counts[f'n_{item.otype}s'] += 1
                timing['otype_counts'] += time.time() - t_item_start
                if item.otype == 'image': # set as a url to use for the user
                    counts['image_url'] = item.url
                t_otype = time.time()
                # liked counts
                liked_rel = Rel.get(src=Item.get(source='me'), tgt=item, rtype='like')
                timing['liked_rel_query'] += time.time() - t_otype
                t_liked = time.time()
                if liked_rel:
                    counts['n_liked_items'] += 1
                timing['liked_check'] += time.time() - t_liked
                t_ts = time.time()
                # track most recent item
                if item.ts and item.ts > counts['last_item_ts']:
                    counts['last_item_ts'] = item.ts
                timing['timestamp_check'] += time.time() - t_ts
            timing['item_loop_total'] = time.time() - t5
            t6 = time.time()
            # average like score
            counts['n_scored'] = len(like_scores)
            counts['avg_like_score'] = sum(like_scores) / len(like_scores) if like_scores else 0.0
            counts['n_pos_like_score'] = sum(1 for s in like_scores if s > 0)
            timing['avg_score_calc'] = time.time() - t6
            timing['total_function'] = time.time() - t0
            # Print top timing items
            formatted_timings = [(name, f"{time:.4f}") for name, time in timing.most_common(5)]
            logger.debug(f"User {user_id} stats timing (top 5): {formatted_timings}")
            return dict(counts)

    async def _handle_user_actions(self, max_items:int= 10) -> None:
        """Handles user actions that are in the table but have not been done."""
        with db_session:
            actions = Rel.select(lambda r: r.rtype in ACTIONS and r.md['processed_ts'] is None)[:max_items]
            for a in actions:
                s = a.tgt.get_source()
                logger.info(f'handling action {a} with source {s}')
                try:
                    await s.handle_me_action([a.tgt.id], a.rtype)
                    a.md['processed_ts'] = time.time()
                except Exception as e:
                    logger.error(f'Error handling action {a}: {e}')
                    a.md['processed_ts'] = -1


    async def _explore_users(self, min_count=60) -> None:
        """Explores users who have more than `min_count` reblogs queued."""
        with db_session:
            users = Item.select(lambda u: u.otype == 'user' and u.explored_ts is None and u.md['n_queued_reblogs'] is not None)
            to_explore = []
            for user in users:
                n_qbr = user.md.get('n_queued_reblogs', 0)
                if n_qbr > min_count:
                    to_explore.append(user.id)
        if not to_explore:
            return
        logger.info(f'Exploring {len(to_explore)} users with at least {min_count} queued reblogs')
        await Rel.handle_me_action(to_explore, 'explore')
        logger.info(f'Done exploring users')

    def _update_user_stats(self, max_users:int=200) -> None:
        """Update statistics for upto `max_users` in the database, sorted by oldest stats time.

        I'm tuning `max_users` to be under 30 secs.
        """
        n_updated = 0
        t0 = time.time()
        with db_session:
            users = Item.select(lambda u: u.otype == 'user')
            # sort by last stats time
            last_times = []
            for u in users:
                if u.md.setdefault('stats', {}) is None:
                    u.md['stats'] = {}
                last_times.append(u.md['stats'].get('ts', 0))
            users = [u for _, u in sorted(zip(last_times, users), key=lambda x: x[0])][:max_users]
            user_ids = {u.id for u in users}
        # first build a dict from user id to list of their item ids
        items_by_user = defaultdict(list)
        n_items = 0
        with db_session:
            items = Item.select(lambda i: i.otype in ('post', 'image', 'video'))
            items = items.filter(lambda i: i.parent and (i.parent.id in user_ids or (i.parent.parent and i.parent.parent.id in user_ids)))
            for item in items:
                user = item.get_closest(otype='user')
                if not user or user.id not in user_ids:
                    continue
                items_by_user[user.id].append(item.id)
                n_items += 1
        logger.info(f'Starting to get stats for {len(user_ids)} users with {n_items} items')
        for user_id in user_ids:
            try:
                stats = self._compute_user_stats(user_id, items_by_user[user_id])
                with db_session:
                    user = Item[user_id]
                    # Update user's metadata with computed stats
                    if user.md is None:
                        user.md = {}
                    # add the queued reblogs
                    stats['n_queued_reblogs'] = user.md.get('n_queued_reblogs', 0)
                    user.md['stats'] = stats
                    n_updated += 1
            except Exception as e:
                logger.error(f"Error computing stats for user {user.id}: {e}")
                print(traceback.format_exc())
                continue
        t1 = time.time()
        logger.info(f"Updated stats for {n_updated} users in {t1 - t0:.2f}s")

    def queue_sizes(self) -> dict[str, int]:
        """Get current queue sizes for monitoring."""
        return dict(
            input_queue_size=self.input_queue.qsize(),
            output_queue_size=self.output_queue.qsize()
        )

    def is_running(self) -> bool:
        """Check if the worker thread is running."""
        return self.running and self.thread is not None and self.thread.is_alive()


class CollectionsWorker(BackgroundWorker):
    """Background worker for collections.

    This does a few different things in a loop:
    - Updates like scores:
      - If the set of liked items has changed, it retrains the classifier and updates scores.
      - If not, but there are new items with embeddings that haven't been scored, it runs inference
        on those items to update scores.
      - In both cases, the updated scores are written out with the current classifier to a joblib
        file for use by other parts of the system.
    - Updates user stats: periodically updates user statistics in the database, such as counts of
      different content types, average like scores, etc. This is used for exploration and other
      features.
    - User exploration: periodically checks for users who have a large number of queued reblogs and
      automatically triggers them for exploration.
    """

    def __init__(self,
                 embs: Embeddings,
                 classifiers_dir: str,
                 name: str = "CollectionsWorker",
                 method: str = 'sgd',
                 max_pos: int = 10000,
                 neg_factor: float = 10,
                 min_new_liked: int = 50,
                 image_suffix: str = 'mn_image',
                 sleep_interval: float = 10.0,
                 valid_tags: list[str]|None = None,
                 exclude_top_n: int = 2000):
        super().__init__(name)
        self.embs = embs
        self.classifiers_dir = classifiers_dir
        self.method = method
        self.max_pos = max_pos
        self.neg_factor = neg_factor
        self.min_new_liked = min_new_liked
        self.image_suffix = image_suffix
        self.sleep_interval = sleep_interval
        self.exclude_top_n = exclude_top_n
        self.valid_tags = valid_tags
        # Set classifier path -- no need to create dir since save_classifier does that
        self.likes_classifier_path = join(self.classifiers_dir, f'likes-{image_suffix}.joblib')

        # State tracking
        self.like_scores: dict[str, float] = {}
        self.last: dict[str, Any] = {
            'pos_ids': frozenset(),
            'saved_classifier': None,
            'classifier_version': 0.0,
            'pos_count': 0,
        }
        self._load_and_run_initial_inference()

    def process_task(self, task: Any) -> Any:
        """Process a likes classification task.

        Task is a string and can be:
        - 'update': Check for changes and update classifier if needed
        - 'force': Force update classifier regardless of changes
        """
        if task == 'update':
            while 1:
                t0 = time.time()
                try:
                    run_async(self._explore_users())
                    run_async(self._handle_user_actions())
                    self._update_user_stats()
                    self._update_classifier()
                except Exception as e:
                    logger.error(f"Error in process_task: {e}")
                    print(traceback.format_exc())
                elapsed = time.time() - t0
                diff = self.sleep_interval - elapsed
                if diff > 0:
                    time.sleep(diff)
        else:
            logger.warning(f"Unknown task type: {task}")
            return None

    def _get_current_pos_ids(self) -> frozenset[int]:
        """Get current set of liked image IDs."""
        with db_session:
            liked_images = Rel.get_likes(valid_types=['image', 'post'])
            # Filter to only those with embeddings
            pos_ids = frozenset(img.id for img in liked_images
                               if img.embed_ts and img.embed_ts > 0)
        return pos_ids

    @db_session
    def _get_all_image_ids(self) -> list[int]:
        """Get all image IDs that have embeddings."""
        all_ids = select(c.id for c in Item
                         if c.otype in ('image', 'post')
                         and c.embed_ts and c.embed_ts > 0)[:]
        return all_ids

    def _get_negative_candidate_ids(self, pos_ids: frozenset[int]) -> list[int]:
        """Get image IDs suitable for negative sampling, excluding positives and most recent."""
        with db_session:
            neg_candidates = Item.select(lambda c:
                c.otype == 'image' and
                c.embed_ts > 0 and
                c.id not in pos_ids
            )
            neg_ids = [img.id for img in neg_candidates]
            neg_ids.sort(reverse=True)
            if len(neg_ids) > self.exclude_top_n:
                neg_ids = neg_ids[self.exclude_top_n:]
        return neg_ids

    def _get_disliked_ids(self) -> list[int]:
        """Get image IDs that have been disliked."""
        with db_session:
            disliked_images = Rel.select(lambda r:
                r.src.source == 'me' and
                r.rtype == 'dislike' and
                r.tgt.otype in ('image', 'post') and
                r.tgt.embed_ts and
                r.tgt.embed_ts > 0
            )
            disliked_ids = [rel.tgt.id for rel in disliked_images]
        return disliked_ids

    def _get_training_set(self, current_pos_ids: list[int]|None=None) -> tuple[list[str], list[str]]:
        """Generates a training set based on current likes and dislikes.

        Returns `(pos, neg)`, where each are lists of keys like `['123:{image_suffix}', ...]`.
        """
        self.embs.reload_keys()
        if current_pos_ids is None:
            current_pos_ids = self._get_current_pos_ids()
        pos = [f'{id}:{self.image_suffix}' for id in sorted(current_pos_ids)]
        # randomly sample max_pos of these
        if len(pos) > self.max_pos: #TODO recency bias?
            pos = random.sample(pos, self.max_pos)
        # Start with disliked items as negatives
        disliked_ids = self._get_disliked_ids()
        # Remove any overlap with positives (shouldn't happen but be safe)
        disliked_ids = [id for id in disliked_ids if id not in current_pos_ids]
        # Calculate how many more negatives we need
        neg_sample_size = min(len(pos) * self.neg_factor, len(pos) * self.neg_factor)
        remaining_neg_needed = max(0, int(neg_sample_size) - len(disliked_ids))
        # Fill remainder with random candidates if needed
        if remaining_neg_needed > 0:
            neg_candidates = self._get_negative_candidate_ids(current_pos_ids)
            # Remove disliked IDs from candidates to avoid duplicates
            neg_candidates = [id for id in neg_candidates if id not in disliked_ids]
            # Sample the remaining negatives
            if len(neg_candidates) > remaining_neg_needed:
                additional_neg_ids = random.sample(neg_candidates, remaining_neg_needed)
            else:
                additional_neg_ids = neg_candidates
            neg_ids = disliked_ids + additional_neg_ids
        else:
            neg_ids = disliked_ids[:int(neg_sample_size)]
        neg = [f'{id}:{self.image_suffix}' for id in neg_ids]
        logger.info(f'Sampled {len(pos)} pos and {len(neg)} neg ({len(disliked_ids)} disliked)')
        return pos, neg

    def _update_classifier(self) -> None:
        """Update the classifier if needed."""
        current_pos_ids = self._get_current_pos_ids()
        if not current_pos_ids:
            logger.info("No liked images found, skipping classifier update")
            return
        current_pos_count = len(current_pos_ids)
        last_pos_count = self.last.get('pos_count', 0)
        # Check if we need to update
        if current_pos_ids == self.last['pos_ids']: # no training data change, just run inference
            return self.run_inference()
        # Only retrain if we have enough new likes since last training
        new_likes_count = current_pos_count - last_pos_count
        if new_likes_count <= self.min_new_liked:
            logger.info(f"Only {new_likes_count} new likes (need >{self.min_new_liked}), running inference only")
            return self.run_inference()
        logger.info(f"Found {new_likes_count} new likes, retraining classifier")
        try:
            pos, neg = self._get_training_set(current_pos_ids=current_pos_ids)
            all_ids = self._get_all_image_ids()
            to_cls = [f'{id}:{self.image_suffix}' for id in all_ids]
            logger.info(f'Training likes: {len(pos)} pos, {len(neg)} neg, {len(to_cls)} to_cls')
            # Reload keys before training
            self.embs.reload_keys()
            # Train and run classifier
            t0 = time.time()
            if 1:
                classifier, scores, other_stuff = self.embs.train_and_run_classifier(
                    pos=pos, neg=neg, to_cls=to_cls, method=self.method, cv=5
                )
                joblib.dump(dict(classifier=classifier, scores=scores, other_stuff=other_stuff), 'blah.joblib')
            else:
                l = joblib.load('blah.joblib')
                classifier, scores, other_stuff = l['classifier'], l['scores'], l['other_stuff']
            t1 = time.time()
            new_scores = self.rescore(scores=scores, pos=pos)
            self.like_scores = {k.split(':')[0]: v for k, v in new_scores.items()}
            saved_classifier = self.embs.save_classifier(
                self.likes_classifier_path,
                classifier,
                method=self.method,
                neg_factor=self.neg_factor,
                pos_ids=sorted(current_pos_ids),
                pos_count=len(pos),
                neg_count=len(neg),
                total_classified=len(to_cls),
                scores=self.like_scores,
                **other_stuff,
            )
            # Store all scores in Score table
            with db_session:
                ttype = f'like:{self.image_suffix}'
                for item_id, score in self.like_scores.items():
                    Score.upsert(
                        get_kw=dict(id=Item[int(item_id)], ttype=ttype, tag='like'),
                        score=float(score),md=dict(classifier_version=saved_classifier['created_at'])
                    )
            logger.info(f"  Stored {len(self.like_scores)} like scores in Score table")
            # Update state
            self.last.update({
                'pos_ids': current_pos_ids,
                'saved_classifier': saved_classifier,
                'classifier_version': saved_classifier['created_at'],
                'pos_count': current_pos_count,
            })
            t2 = time.time()
            logger.info(f"Updated likes classifier in {t1-t0:.2f}s+{t2-t1:.2f}s, v{self.last['classifier_version']}")
        except Exception as e:
            logger.error(f"Error updating likes classifier: {e}")
            print(traceback.format_exc())

    def get_like_scores(self) -> dict[str, float]:
        """Get current classifier scores."""
        return self.like_scores.copy()

    def rescore(self, scores: dict[str, float], pos: list[int]) -> dict[str, float]:
        """Rescores `scores` using nearest neighbors from positive IDs."""
        if not scores:
            return {}
        #return scores
        fix = lambda k: k if (isinstance(k, int) or ':' in k) else f'{k}:{self.image_suffix}'
        scores = {fix(k): v for k, v in scores.items()}
        pos = [fix(id) for id in pos]
        logger.debug(f'running rescore on {len(scores)} scores with {len(pos)} positives')
        logger.debug(f'  First scores: {list(scores.items())[:5]}, first pos: {pos[:5]}')
        new_scores = self.embs.rescore_by_nn(scores=scores, pos=pos, min_score=1.0, metric='l2')
        logger.debug(f'  Output scores: {list(new_scores.items())[:5]}')
        max_score = 2.0
        assert frozenset(new_scores) == frozenset(scores)
        for k, v in new_scores.items():
            assert v <= max_score, f"(a) Rescored value {v} for {k} exceeds {max_score}"
        ret = {k.split(':')[0]: v for k, v in new_scores.items()}
        for k, v in ret.items():
            assert v <= max_score, f"(b) Rescored value {v} for {k} exceeds {max_score}"
        #logger.info('EXITING!!!'); sys.exit()
        return ret

    def _load_and_run_initial_inference(self) -> None:
        """Load existing classifiers on initialization and populate scores from saved data.

        This first loads the likes classifier and loads scores from it if available.
        Then loads all tags classifiers and checks if they're in lmdb already or not, and if not,
        adds them.
        """
        # load the likes classifier
        try:
            saved_data = self.embs.load_classifier(self.likes_classifier_path)
            pos_ids = [int(id) for id in saved_data.get('pos_ids', [])]
            self.last.update({
                'saved_classifier': saved_data,
                'classifier_version': saved_data.get('created_at', 0),
                'pos_ids': frozenset(pos_ids),
            })
            # Load scores from saved classifier data
            self.like_scores = {}
            for k, v in saved_data.get('scores', {}).items():
                self.like_scores[int(k)] = float(v)
            # note that these have already been rescored, so we don't redo it.
            # Also update pos_count from saved data
            self.last['pos_count'] = saved_data.get('pos_count', len(pos_ids))
            logger.info(f"Loaded existing likes classifier v{self.last['classifier_version']} with {len(self.like_scores)} scores, {self.last['pos_count']} pos examples")
        except Exception as e:
            logger.warning(f"Failed to load existing likes classifier: {e}\n{traceback.format_exc()}")
        # deal with tags classifiers
        cls_paths = glob.glob(join(self.classifiers_dir, f'tags-{self.image_suffix}/*.joblib'))
        logger.debug(f'Got {len(cls_paths)} tag clses: {cls_paths[:5]}')
        todo = []
        db = self.load_lmdb()
        self.tag_keys = {}
        # build up list of tags to add to lmdb
        for path in cls_paths:
            data = self.embs.load_classifier(path)
            tag = data['tag']
            # delete some fields we don't want to save in lmdb
            for field in 'classifier scaler scores total_classified'.split():
                if field in data:
                    del data[field]
            version = data['version'] = str(data.get('created_at', os.path.getmtime(path)))
            logger.debug(f'For tag {tag} read {data}')
            # now fetch lmdb data and compare versions
            key = f'tags:{tag}'
            self.tag_keys[tag] = key
            if key not in db:
                todo.append((tag, data, key, version))
                continue
            md = db.md_get(key)
            if md.get('version', '') != version:
                todo.append((tag, data, key, version))
        if not todo:
            return
        # actually add tags to lmdb
        model = 'qwen_emb'
        emb_futures = {row[0]: embed_text.single_future(row[0], model=model) for row in todo}
        for tag, data, key, version in todo:
            try:
                emb = emb_futures[tag].result()
            except Exception:
                emb = []
            logger.info(f'  Updating lmdb for tag {tag} with version {version}: {emb[:5]}')
            db[key] = emb
            db.md_set(key, **data)
        db.sync()

    def load_lmdb(self, flag='c', **kw) -> NumpyLmdb:
        """Loads the lmdb for our embeddings for direct manipulation.

        Pass `flag` and `kw` to `NumpyLmdb.open()`
        """
        assert len(self.embs.inputs) == 1 and isinstance(self.embs.inputs[0], NumpyLmdb), "Expecting exactly one NumpyLmdb input for tag embeddings"
        db = NumpyLmdb.open(self.embs.inputs[0].path, flag=flag, **kw)
        return db

    def run_inference(self) -> None:
        """Run inference on unclassified items.
        """
        all_ids = self._get_all_image_ids()
        self.run_likes_inference(all_ids=all_ids)
        self.run_tags_inference(all_ids=all_ids)

    def run_tags_inference(self,
                           all_ids: list[int]|None=None,
                           score_threshold: float=0.5,
                           min_to_cls: int=20,
                           max_tags: int=100) -> None:
        """Runs inference for the tags classifiers on items that don't have scores yet.

        If `self.valid_tags` is not `None`, we only do tags that are in that list.

        We only add a tag to the Score table if the score is above `score_threshold`.

        If `max_tags` > 0, then only runs inference for upto that many tags (skipping those that
        have no updates). This is not so much for memory usage (although that also if you're running
        for the first time on lots of items * lots of tags), but because we don't actually update
        the databases until we run inference on all tags (upto the max).
        """
        tag_keys = self.tag_keys
        logger.info(f'Got {len(tag_keys)} tag keys, {len(self.valid_tags)} valid: {self.valid_tags[:5]}')
        if all_ids is None:
            all_ids = self._get_all_image_ids()
        logger.debug(f'Got {len(all_ids)} ids with embeddings for inference: {all_ids[:5]}')
        all_ids = set(all_ids)
        ttype = f'tag:{self.image_suffix}'
        current_ts = time.time()
        db = self.load_lmdb(flag='c')
        # create tqdm progress bar that we can update
        bar = tqdm(total=len(tag_keys), desc='tag cls')
        n_tags_done = 0
        for i, (tag, key) in enumerate(sorted(tag_keys.items())):
            bar.update(1)
            if self.valid_tags is not None and tag not in self.valid_tags:
                continue
            # get done ids for this tag from lmdb (Score table only has positives)
            done = set(db.md_get(key+':done').get('ids', []))
            #logger.info(f'  For tag "{tag}": {len(done)} done')
            to_cls = all_ids - done
            s = f'{tag} [{i+1}/{len(tag_keys)}]: {len(done)} done, {len(to_cls)} new'
            bar.set_postfix_str(s)
            if len(to_cls) < min_to_cls:
                continue
            # run classifier
            cls_path = join(self.classifiers_dir, f'tags-{self.image_suffix}/{tag}.joblib')
            result = self.load_and_run_classifier(ids=to_cls, path=cls_path)
            if result['items_classified'] == 0:
                continue
            n_tags_done += 1
            scores = result.pop('new_scores')
            logger.debug(f'  Got tag result {result}')
            # update scores and done by item id
            with db_session:
                for id, v in scores.items():
                    if v < score_threshold:
                        continue
                    Score(id=Item[int(id)], ttype=ttype, tag=tag, score=float(v), ts=current_ts)
            # update lmdb with done ids
            db.md_set(key+':done', ids=sorted(done | set(scores.keys())), version=result['classifier_version'])
            db.sync()
            if max_tags > 0 and n_tags_done >= max_tags:
                break

    def run_likes_inference(self, all_ids: list[int]|None=None) -> None:
        """Runs inference for the likes classifier

        - Find items that are currently unclassified (not in `self.like_scores`) but have embeddings.
        - Load the likes classifier from disk and run on those items
        - Since we run classification on all items when we train the likes classifier, we never have
          inconsistent classifications from different versions of the classifier.
        """
        logger.debug(f'Running inference with likes classifier v{self.last["classifier_version"]}')
        if not self.last['saved_classifier'] or self.last['classifier_version'] == 0:
            logger.info("No classifier available for inference")
            return
        try:
            # Get all image IDs that have embeddings
            if all_ids is None:
                all_ids = self._get_all_image_ids()
            classified_ids = set(int(id) for id in self.like_scores.keys())
            unclassified_ids = [id for id in all_ids if id not in classified_ids]
            if not unclassified_ids:
                return
            # run classification and rescoring
            result = self.load_and_run_classifier(ids=unclassified_ids, path=self.likes_classifier_path)
            result['new_scores'] = self.rescore(result['new_scores'], list(self.last['pos_ids']))
            # Update scores if successful
            if result['status'] == 'inference_completed':
                self.like_scores.update(result['new_scores'])
                logger.info(f"Inference completed in {result['inference_time']:.2f}s for {len(result['new_scores'])} items, {len(self.like_scores)} total scores")
                # Store scores in Score table
                with db_session:
                    ttype = f'like:{self.image_suffix}'
                    for item_id, score in result['new_scores'].items():
                        Score.upsert(
                            get_kw=dict(id=Item[int(item_id)], ttype=ttype, tag='like'),
                            score=float(score),
                        )
                logger.info(f"  Stored {len(result['new_scores'])} like scores in Score table")
                # Save the classifier with updated scores
                try:
                    if self.last['saved_classifier'] and result['new_scores']:
                        # Load the existing classifier data and update scores
                        saved_data = self.last['saved_classifier'].copy()
                        saved_data['scores'] = self.like_scores

                        # Save the updated classifier data
                        self.embs.save_classifier(
                            self.likes_classifier_path,
                            saved_data.pop('classifier'),
                            **saved_data
                        )
                        logger.info(f"  Updated classifier saved with {len(self.like_scores)} scores")
                except Exception as e:
                    logger.warning(f"Failed to save updated classifier: {e}, {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            traceback.print_exc()

    def load_and_run_classifier(self, ids: list[int], path: str) -> dict[str, Any]:
        """Loads and run classifier at `path`"""
        saved_data = self.embs.load_classifier(path)
        to_cls = [f'{id}:{self.image_suffix}' for id in ids]
        self.embs.reload_keys()
        t0 = time.time()
        new_scores = self.embs.run_classifier(to_cls=to_cls,
                                              classifier=saved_data['classifier'],
                                              scaler=saved_data.get('scaler', None),
                                              sampler=saved_data.get('sampler', None))
        new_scores = {int(key.split(':')[0]): score for key, score in new_scores.items()}
        t1 = time.time()
        return dict(
            status='inference_completed',
            classifier_version=saved_data.get('classifier_version', saved_data['created_at']),
            items_classified=len(new_scores),
            inference_time=t1-t0,
            new_scores=new_scores
        )

    def gen_benchmark_set(self, name: str='', **kw):
        """Generate a benchmark set based on likes.

        Get training data (exactly like _update_classifier), then for those Items in the sqlite
        database, set `md[name] = {label}`, where label is either +1 or -1 for pos/neg examples,
        respectively.

        We also further restrict the training set to only items that exist locally.
        """
        if not name:
            name = 'like-benchmark-' + time.strftime('%Y%m%d')
        logger.info(f'Generating benchmark dataset: {name}')
        pos, neg = self._get_training_set()
        print(f'Initial pos: {len(pos)}, {len(neg)}: {pos[:5]}, {neg[:5]}')
        with db_session:
            def item_exists(key: str) -> bool:
                item_id = int(key.split(':')[0])
                item = Item[item_id]
                return item is not None and item.image_path() is not None and exists(item.image_path())

            pos = [p for p in pos if item_exists(p)]
            neg = [n for n in neg if item_exists(n)]
        logger.info(f'Benchmark {name}: {len(pos)} positives, {len(neg)} negatives')
        label_by_id = {(key.split(':')[0]): 1 for key in pos}
        label_by_id.update({int(key.split(':')[0]): -1 for key in neg})
        # Update database with benchmark labels
        with db_session:
            n_updated = 0
            for item_id, label in label_by_id.items():
                item = Item[item_id]
                if item:
                    if item.md is None:
                        item.md = {}
                    item.md[name] = label
                    n_updated += 1
        logger.info(f'Updated {n_updated} items with benchmark labels for {name}')

    def run_benchmark(self, name: str, item_suffix: str='image') -> dict[str, float]:
        """Runs benchmark evaluation for the given benchmark set `name`.

        Computes accuracy, balanced accuracy, precision, recall, F1, etc.

        Returns dict with evaluation metrics.
        """
        logger.info(f'Running benchmark evaluation for: {name} with suffix {item_suffix}')
        # Get all items with the benchmark label
        with db_session:
            benchmark_items = Item.select(lambda c: c.md and name in c.md)
            labels = {}
            for item in benchmark_items:
                labels[item.id] = item.md[name]
            pos_ids = [id for id, label in labels.items() if label == 1]
            neg_ids = [id for id, label in labels.items() if label == -1]
            logger.info(f'Benchmark {name}: {len(pos_ids)} positives, {len(neg_ids)} negatives')
        if not labels:
            logger.warning(f'Benchmark {name} needs both positive and negative examples')
            return {}
        # run training and inference
        self.embs.reload_keys()
        pos = [f'{id}:{item_suffix}' for id in pos_ids]
        neg = [f'{id}:{item_suffix}' for id in neg_ids]
        cls, scores, other_stuff = self.embs.train_and_run_classifier(
            pos=pos,
            neg=neg,
            to_cls=pos+neg,
            method='sgd',
            cv=5
        )
        #TODO check if scores are based on cv folds or not
        scores = {int(k.split(':')[0]): v for k, v in scores.items()}
        logger.info(f'Other stuff: {other_stuff}')
        print(f'Got {len(scores)} scores for {len(labels)} benchmark items: {list(scores.items())[:5]}')
        # some bookkeeping
        benchmark_scores = {}
        benchmark_labels = {}
        for item_id in labels:
            if item_id in scores:
                benchmark_scores[item_id] = scores[item_id]
                benchmark_labels[item_id] = labels[item_id]
        if len(benchmark_scores) < len(labels) * 0.8:  # Less than 80% coverage
            logger.warning(f'Only {len(benchmark_scores)}/{len(labels)} benchmark items have scores')
        if not benchmark_scores:
            logger.error(f'No benchmark items have classifier scores')
            return {}
        y_true = [benchmark_labels[id] for id in benchmark_scores.keys()]
        y_scores = [benchmark_scores[id] for id in benchmark_scores.keys()]
        # compute stats
        results = compute_binary_classifier_stats(y_true, y_scores)
        logger.info(f'Benchmark {name} results:')
        for metric, value in results.items():
            if isinstance(value, float):
                logger.info(f'  {metric}: {value:.4f}')
            else:
                logger.info(f'  {metric}: {value}')
        return results

    def run_embedder_on_benchmark(self,
                                  benchmark_name: str,
                                  batch_size=100,
                                  embedder_name='mobilenet',
                                  id_suffix=':mn_image') -> None:
        """Runs specified embedder on the benchmark set `benchmark_name`.

        This does the following:
        - gets all items with benchmark labels
        - generates their lmdb keys by adding the `id_suffix`
        - looks for which ones don't already exist in the lmdb database
        - divides these up into batches of size `batch_size`
        - creates a `LmdbUpdater` to handle writing to the lmdb
        - extracts embeddings for these batches using embed_image.batch(model=`embedder_name`)
        - writes embeddings to lmdb via the `LmdbUpdater`
        """
        logger.info(f'Running embedder {embedder_name} on benchmark {benchmark_name}')
        images_dir = Source.first_source().images_dir
        # Get all items with benchmark labels
        with db_session:
            benchmark_items = Item.select(lambda c: c.md and benchmark_name in c.md)
            item_data = [(item.id, item.url, item.image_path()) for item in benchmark_items]
        logger.info(f'Found {len(item_data)} items in benchmark {benchmark_name}: {item_data[:5]}')
        if not item_data:
            return
        # Generate LMDB keys and check which ones don't exist
        lmdb_keys = [f'{item_id}{id_suffix}' for item_id, _, _ in item_data]
        self.embs.reload_keys()
        existing_keys = set(self.embs.get_keys())
        todo = []
        for (item_id, url, local_path), lmdb_key in zip(item_data, lmdb_keys):
            if lmdb_key not in existing_keys:
                todo.append((item_id, url, local_path, lmdb_key))
        logger.info(f'Need to generate embeddings for {len(todo)} items')
        if not todo:
            return
        # Create LmdbUpdater for writing embeddings
        lmdb_path = self.embs.inputs[0].path  # Use first LMDB path
        logger.info(f'Using LMDB path: {lmdb_path}, {embed_image}')
        updater = LmdbUpdater(lmdb_path)
        # Process in batches
        n_processed = 0
        n_batches = (len(todo) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(todo), batch_size)):
            batch = todo[start:start + batch_size]
            batch_num = start // batch_size + 1
            logger.debug(f'Processing batch {batch_num}/{n_batches} ({len(batch)} items)')
            image_paths = []
            batch_keys = []
            for item_id, url, local_path, lmdb_key in batch:
                # Use local_path if available, otherwise use URL
                image_path = local_path or url
                image_paths.append(image_path)
                batch_keys.append(lmdb_key)
            try:
                # Extract embeddings using batch processing
                futures = embed_image.batch_futures(image_paths, model=embedder_name)
                for lmdb_key, future in zip(batch_keys, futures):
                    try:
                        embedding = future.result(timeout=5)
                    except Exception as e:
                        logger.error(f'Error extracting embedding for {lmdb_key} (path: {image_paths[batch_keys.index(lmdb_key)]}): {e}')
                        embedding = None
                        continue
                    if embedding is not None:
                        md = dict(ts=time.time(), benchmark=benchmark_name, embedder=embedder_name)
                        updater.add(lmdb_key, embedding=embedding, metadata=md)
                n_processed += len(batch)
                logger.debug(f'Processed {n_processed}/{len(todo)} items')
            except Exception as e:
                logger.error(f'Error processing batch {batch_num}: {e}')
                print(traceback.format_exc())
                raise
        updater.commit()
        logger.info(f'Completed embedding generation for benchmark {benchmark_name}: {n_processed} items processed')

    def train_tag_classifier(self,
                             tags_path: str,
                             id_suffix: str='mn_image',
                             tag_dlm: str=';',
                             min_pos: int=5,
                             neg_factor: float=10,
                             **cls_kw) -> None:
        """Trains a classifier to predict tags based on embeddings.

        The input `tags_path` should be a tsv file with Item id in the first column and tags in the
        last column tags, delimited by `tag_dlm` [default ;]

        We generate training data by looking up the embeddings for these items. For each tag with at
        least `min_pos` valid embeddings, we sample upto num_pos*neg_factor negatives by looking at
        items that share none of the tags that co-occur with any positive. E.g. if we're looking at
        tag A and its positives include {A, B, C} and {A, D}, then we look for items that have none
        of {A,B,C,D} for the negatives.

        We train the classifier using `train_and_run_classifier` (with `cls_kw` passed through) and
        save the resulting classifier and scores to a joblib file with path
        `{self.classifiers_dir}/tags-{id_suffix}/{tag_name}.joblib`.
        """
        logger.info(f'Training tag classifiers from {tags_path} with suffix {id_suffix}')
        item_tags: dict[int, set[str]] = {}
        tag_counts: dict[str, int] = Counter()
        with open(tags_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    item_id = int(row[0])
                    tags_str = row[-1].strip()
                    tags = {tag.strip() for tag in tags_str.split(tag_dlm) if tag.strip()}
                    if tags:
                        item_tags[item_id] = tags
                        for tag in tags:
                            tag_counts[tag] += 1
                except (ValueError, IndexError) as e:
                    logger.warning(f'Skipping invalid row: {row}, error: {e}')
        logger.info(f'Loaded {len(item_tags)} items with tags, {len(tag_counts)} unique tags')
        # Filter tags that have enough positive examples
        valid_tags = [tag for tag, count in tag_counts.items() if count >= min_pos]
        logger.info(f'Found {len(valid_tags)} tags with at least {min_pos} examples')
        if not valid_tags:
            logger.warning('No tags meet minimum positive example threshold')
            return
        # Reload embeddings keys
        self.embs.reload_keys()
        available_keys = set(self.embs.get_keys())
        # For each valid tag, train a classifier
        for i, tag in enumerate(valid_tags):
            logger.info(f'Training classifier {i+1}/{len(valid_tags)} for tag: {tag}')
            # Get positive examples (items with this tag that have embeddings)
            pos_item_ids = [item_id for item_id, tags in item_tags.items() if tag in tags]
            pos_keys = [f'{item_id}:{id_suffix}' for item_id in pos_item_ids]
            pos_keys = [key for key in pos_keys if key in available_keys]
            if len(pos_keys) < min_pos:
                logger.warning(f'Tag {tag}: only {len(pos_keys)} items have embeddings, skipping')
                continue
            # Find co-occurring tags for this tag
            cooccurring_tags = set()
            for item_id in pos_item_ids:
                if item_id in item_tags:
                    cooccurring_tags.update(item_tags[item_id])
            logger.debug(f'Tag {tag}: {len(pos_keys)} positives, co-occurring with {len(cooccurring_tags)} tags')
            # Get negative candidates (items that don't have any co-occurring tags)
            neg_candidates = []
            for item_id, tags in item_tags.items():
                if not tags.intersection(cooccurring_tags):  # No overlap with co-occurring tags
                    neg_key = f'{item_id}:{id_suffix}'
                    if neg_key in available_keys:
                        neg_candidates.append(neg_key)
            # Sample negatives
            max_neg = int(len(pos_keys) * neg_factor)
            neg_keys = random.sample(neg_candidates, min(max_neg, len(neg_candidates)))
            logger.info(f'Tag {tag}: {len(pos_keys)} pos, {len(neg_keys)} neg (from {len(neg_candidates)} candidates)')
            if not neg_keys:
                logger.warning(f'Tag {tag}: no negative examples found, skipping')
                continue
            # Train classifier
            try:
                to_cls = pos_keys + neg_keys
                classifier, scores, other_stuff = self.embs.train_and_run_classifier(
                    pos=pos_keys,
                    neg=neg_keys,
                    to_cls=to_cls,
                    cv=5,
                    **cls_kw
                )
                # Save classifier
                classifier_path = join(self.classifiers_dir, f'tags-{id_suffix}/{tag}.joblib')
                saved_classifier = self.embs.save_classifier(
                    classifier_path,
                    classifier,
                    tag=tag,
                    pos_count=len(pos_keys),
                    neg_count=len(neg_keys),
                    total_classified=len(to_cls),
                    scores=scores,
                    cooccurring_tags=list(cooccurring_tags),
                    **other_stuff,
                    **cls_kw
                )
                logger.info(f'Saved likes classifier for tag {tag} to {classifier_path}')
            except Exception as e:
                logger.error(f'Error training classifier for tag {tag}: {e}')
                print(traceback.format_exc())
                continue
        logger.info(f'Completed training tag classifiers for {len(valid_tags)} tags')
