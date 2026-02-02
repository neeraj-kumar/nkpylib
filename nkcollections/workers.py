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

from nkpylib.ml.constants import data_url_from_file
from nkpylib.ml.embeddings import Embeddings
from nkpylib.ml.nklmdb import NumpyLmdb, batch_extract_embeddings, LmdbUpdater
from nkpylib.nkcollections.model import Item, Rel, Source
from nkpylib.nkpony import init_sqlite_db, GetMixin, recursive_to_dict
from nkpylib.stringutils import parse_num_spec
from nkpylib.thread_utils import run_async
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
                    self._update_classifier()
                except Exception as e:
                    pass
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

    def _update_classifier(self) -> dict[str, Any]:
        """Update the classifier if needed."""
        current_pos_ids = self._get_current_pos_ids()
        # Check if we need to update
        if current_pos_ids == self.last['pos_ids']: # no training data change, just run inference
            # Run inference
            r = self.run_inference()
            logging.debug(f'Running inference result: {r}')
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
            pos = [f'{id}:image' for id in sorted(current_pos_ids)]
            # randomly sample max_pos of these
            if len(pos) > self.max_pos:
                #pos = random.sample(pos, self.max_pos)
                pos = pos[-self.max_pos:]
            # Prepare negative samples (exclude positives and most recent IDs)
            neg_ids = self._get_negative_candidate_ids(current_pos_ids)
            # Sample negatives
            neg_sample_size = min(len(neg_ids), int(len(pos) * self.neg_factor))
            neg_ids = random.sample(neg_ids, neg_sample_size)
            neg = [f'{id}:image' for id in neg_ids]
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


