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
    Set,
    select,
) # type: ignore
from pony.orm.core import BindingError, Query, UnrepeatableReadError # type: ignore
from tornado.web import RequestHandler
from tqdm import tqdm

from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings, compute_binary_classifier_stats
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import Item, Rel, Source, ret_immediate
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
            like_scores = [score for id, score in self.scores.items() if int(id) in item_ids]
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

    def _explore_users(self, min_count=80) -> None:
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
        run_async(ret_immediate(Rel.handle_me_action(to_explore, 'explore')))

    @db_session
    def _update_user_stats(self, max_users:int=1000) -> None:
        """Update statistics for upto `max_users` in the database, sorted by oldest stats time.

        I'm tuning `max_users` to be under 30 secs.
        """
        n_updated = 0
        t0 = time.time()
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
        items = Item.select(lambda i: i.otype in ('post', 'image', 'video'))
        items = items.filter(lambda i: i.parent and (i.parent.id in user_ids or (i.parent.parent and i.parent.parent.id in user_ids)))
        for item in items:
            user = item.get_closest(otype='user')
            if not user or user.id not in user_ids:
                continue
            items_by_user[user.id].append(item.id)
            n_items += 1
        logger.info(f'Starting to get stats for {len(users)} users with {n_items} items')
        for user in users:
            try:
                stats = self._compute_user_stats(user.id, items_by_user[user.id])
                # Update user's metadata with computed stats
                if user.md is None:
                    user.md = {}
                # add the queued reblogs
                stats['n_queued_reblogs'] = user.md.get('n_queued_reblogs', 0)
                user.md['stats'] = stats
                n_updated += 1
            except Exception as e:
                logger.error(f"Error computing stats for user {user.id}: {e}")
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
                 method: str = 'sgd',
                 max_pos: int = 10000,
                 neg_factor: float = 10,
                 sleep_interval: float = 10.0,
                 exclude_top_n: int = 2000):
        super().__init__(name)
        self.embs = embs
        self.classifiers_dir = classifiers_dir
        self.method = method
        self.max_pos = max_pos
        self.neg_factor = neg_factor
        self.sleep_interval = sleep_interval
        self.exclude_top_n = exclude_top_n
        # Set classifier path -- no need to create dir since save_classifier does that
        self.classifier_path = join(self.classifiers_dir, 'likes.joblib')

        # State tracking
        self.scores: dict[str, float] = {}
        self.last: dict[str, Any] = {
            'pos_ids': frozenset(),
            'saved_classifier': None,
            'classifier_version': 0.0,
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
                    self._explore_users()
                    self._update_user_stats()
                    self._update_classifier()
                except Exception as e:
                    logger.error(f"Error in process_task: {e}")
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

    def _get_all_image_ids(self) -> list[int]:
        """Get all image IDs that have embeddings."""
        with db_session:
            all_images = Item.select(lambda c: c.otype in ('image', 'post') and c.embed_ts > 0)
            all_ids = [img.id for img in all_images]
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

        Returns `(pos, neg)`, where each are lists of keys like `['123:image', ...]`.
        """
        self.embs.reload_keys()
        if current_pos_ids is None:
            current_pos_ids = self._get_current_pos_ids()
        pos = [f'{id}:image' for id in sorted(current_pos_ids)]
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
        neg = [f'{id}:image' for id in neg_ids]
        logger.info(f'Sampled {len(pos)} pos and {len(neg)} neg ({len(disliked_ids)} disliked)')
        return pos, neg

    def _update_classifier(self) -> None:
        """Update the classifier if needed."""
        current_pos_ids = self._get_current_pos_ids()
        if not current_pos_ids:
            logger.info("No liked images found, skipping classifier update")
            return
        # Check if we need to update
        if current_pos_ids == self.last['pos_ids']: # no training data change, just run inference
            # Run inference
            r = self.run_inference()
            logging.debug(f'Running inference result: {r}')
            return dict(status='no_change', pos_count=len(current_pos_ids))
        try:
            pos, neg = self._get_training_set(current_pos_ids=current_pos_ids)
            all_ids = self._get_all_image_ids()
            to_cls = [f'{id}:image' for id in all_ids]
            logger.info(f'Training likes: {len(pos)} pos, {len(neg)} neg, {len(to_cls)} to_cls')
            # Train and run classifier
            t0 = time.time()
            if 1:
                classifier, scores, other_stuff = self.embs.train_and_run_classifier(
                    pos=pos, neg=neg, to_cls=to_cls, method=self.method
                )
                joblib.dump(dict(classifier=classifier, scores=scores, other_stuff=other_stuff), 'blah.joblib')
            else:
                l = joblib.load('blah.joblib')
                classifier, scores, other_stuff = l['classifier'], l['scores'], l['other_stuff']
            t1 = time.time()
            new_scores = self.rescore(scores=scores, pos=pos)
            self.scores = {k.split(':')[0]: v for k, v in new_scores.items()}
            saved_classifier = self.embs.save_classifier(
                self.classifier_path,
                classifier,
                method=self.method,
                neg_factor=self.neg_factor,
                pos_ids=sorted(current_pos_ids),
                pos_count=len(pos),
                neg_count=len(neg),
                total_classified=len(to_cls),
                scores=self.scores,
                **other_stuff,
            )
            # Update state
            self.last.update({
                'pos_ids': current_pos_ids,
                'saved_classifier': saved_classifier,
                'classifier_version': saved_classifier['created_at'],
            })
            t2 = time.time()
            logger.info(f"Updated likes classifier in {t1-t0:.2f}s+{t2-t1:.2f}s, v{self.last['classifier_version']}")
            return dict(
                status='updated',
                pos_count=len(pos),
                neg_count=len(neg),
                scores_count=len(self.scores),
                classifier_path=self.classifier_path,
                **other_stuff
            )
        except Exception as e:
            logger.error(f"Error updating likes classifier: {e}")
            print(traceback.format_exc())
            return dict(status='error', error=str(e))

    def get_scores(self) -> dict[str, float]:
        """Get current classifier scores."""
        return self.scores.copy()

    def rescore(self, scores: dict[str, float], pos: list[int]) -> dict[str, float]:
        """Rescores `scores` using nearest neighbors from positive IDs."""
        if not scores:
            return {}
        #return scores
        fix = lambda k: k if (isinstance(k, int) or ':' in k) else f'{k}:image'
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
        """Load existing classifier on initialization and populate scores from saved data."""
        try:
            saved_data = self.embs.load_and_setup_classifier(self.classifier_path)
            pos_ids = [int(id) for id in saved_data.get('pos_ids', [])]
            self.last.update({
                'saved_classifier': saved_data,
                'classifier_version': saved_data.get('created_at', 0),
                'pos_ids': frozenset(pos_ids),
            })
            # Load scores from saved classifier data
            self.scores = {}
            for k, v in saved_data.get('scores', {}).items():
                self.scores[int(k)] = float(v)
            # note that these have already been rescored, so we don't redo it.
            logger.info(f"Loaded existing classifier v{self.last['classifier_version']} with {len(self.scores)} scores")
        except Exception as e:
            logger.warning(f"Failed to load existing classifier: {e}\n{traceback.format_exc()}")

    def run_inference(self) -> dict[str, Any]:
        """Run inference using the last classifier on items that haven't been classified by it.

        Assumes all items in self.scores have been classified with the current classifier version.
        Only runs inference on items not already in self.scores.

        Returns dict with status and inference results.
        """
        logger.debug(f'Running inference with classifier v{self.last["classifier_version"]}')
        if not self.last['saved_classifier'] or self.last['classifier_version'] == 0:
            logger.info("No classifier available for inference")
            return dict(status='no_classifier')
        try:
            # Get all image IDs that have embeddings
            all_ids = self._get_all_image_ids()
            classified_ids = set(int(id) for id in self.scores.keys())
            unclassified_ids = [id for id in all_ids if id not in classified_ids]
            #unclassified_ids = [id for id in all_ids]
            if not unclassified_ids:
                return dict(status='all_classified', classifier_version=self.last['classifier_version'])
            result = self._run_inference_blocking(unclassified_ids)
            # Update scores if successful
            if result['status'] == 'inference_completed':
                self.scores.update(result['new_scores'])
                logger.info(f"Inference completed in {result['inference_time']:.2f}s for {len(result['new_scores'])} items, {len(self.scores)} total scores")
                # Save the classifier with updated scores
                try:
                    if self.last['saved_classifier'] and result['new_scores']:
                        # Load the existing classifier data and update scores
                        saved_data = self.last['saved_classifier'].copy()
                        saved_data['scores'] = self.scores

                        # Save the updated classifier data
                        self.embs.save_classifier(
                            self.classifier_path,
                            saved_data.pop('classifier'),
                            **saved_data
                        )
                        logger.info(f"  Updated classifier saved with {len(self.scores)} scores")
                except Exception as e:
                    logger.warning(f"Failed to save updated classifier: {e}, {traceback.format_exc()}")
            return result

        except Exception as e:
            logger.error(f"Error running inference: {e}")
            traceback.print_exc()
            return dict(status='error', error=str(e))

    def _run_inference_blocking(self, unclassified_ids: list[int]) -> dict[str, Any]:
        """The blocking part of inference that runs in a thread pool."""
        try:
            # Load the classifier
            saved_data = self.embs.load_and_setup_classifier(self.classifier_path)
            classifier = saved_data['classifier']
            to_cls = [f'{id}:image' for id in unclassified_ids]
            # Get embeddings and run inference+rescoring
            self.embs.reload_keys()
            t0 = time.time()
            new_scores = self.embs.run_classifier(to_cls=to_cls,
                                                  classifier=classifier,
                                                  scaler=saved_data.get('scaler', None))
            new_scores = {key.split(':')[0]: score for key, score in new_scores.items()}
            t1 = time.time()
            new_scores = self.rescore(new_scores, list(self.last['pos_ids']))
            return dict(
                status='inference_completed',
                classifier_version=self.last['classifier_version'],
                items_classified=len(new_scores),
                inference_time=t1-t0,
                new_scores=new_scores
            )

        except Exception as e:
            logger.error(f"Error in blocking inference: {e}")
            traceback.print_exc()
            return dict(status='error', error=str(e))

    def gen_benchmark_set(self, name: str='', **kw):
        """Generate a benchmark set based on likes.

        Get training data (exactly like _update_classifier), then for those Items in the sqlite
        database, set `md[name] = {label}`, where label is either +1 or -1 for pos/neg examples,
        respectively.
        """
        if not name:
            name = 'like-benchmark-' + time.strftime('%Y%m%d')
        logger.info(f'Generating benchmark dataset: {name}')
        pos, neg = self._get_training_set()
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

    def run_benchmark(self, name: str) -> dict[str, float]:
        """Runs benchmark evaluation for the given benchmark set `name`.

        Computes accuracy, balanced accuracy, precision, recall, F1, etc.

        Returns dict with evaluation metrics.
        """
        logger.info(f'Running benchmark evaluation for: {name}')
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
        # run inference
        scores = self._run_inference_blocking(list(labels))['new_scores']
        scores = {int(k): v for k, v in scores.items()}
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
