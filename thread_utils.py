"""Various thread-related utilities.

Note that I'm starting over in 2024, as the old threadutils.py (no underscore) was getting too messy
and most of it is outdated.
"""
#TODO sql progress updater

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import typing

from functools import wraps
from inspect import iscoroutinefunction
from os.path import abspath, basename, dirname, exists, join, relpath
from queue import Queue, Empty
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

def chained_producer_consumers(functions: list[function],
                               sleep_interval: float=0.1,
                               get_timeout: float=0.5) -> Iterable[Any]:
    """
    Start a chain of producer-consumer operations.

    Args:
    - functions: A list of functions each with one output, where the first function takes no input,
      and each subsequent function takes an item from the previous function.
    - sleep_interval: The interval between producing items in the producer threads.
    - get_timeout: The timeout for getting items from the input queues.

    Yields:
    - Items produced by the last function in the chain.

    The producer-consumer chain is implemented using a series of threads and queues, where each
    thread consumes items from the previous thread's output queue, processes them with the function,
    and puts the results into the next thread's input queue. The final output queue's items are
    yielded by the main processing thread of this function.

    Once the first function exhausts its input, it puts a Sentinel into the output queue to signal
    the end. Each subsequent function stops when it receives a Sentinel from the input queue, and
    also puts a Sentinel into the output queue to keep signalling forward.

    If a function raises a StopIteration, we need to stop all upstream functions as well. We do this
    by setting a stop_idx to the index of the function that raised the StopIteration, and each
    function checks its own index against this stop_idx before processing an item.
    """
    lock = threading.Lock()
    stop_idx = -1
    sentinel = object()

    def producer_consumer_wrapper(func, in_q, out_q, idx):
        """Wrapper for each function in the chain.

        This consumes items from the input queue, processes them with the function,
        and puts the results into the output queue.

        If `in_q` is None, then the function is assumed to be the initial producer.

        If the function raises a StopIteration, we set the stop_idx to our index.
        If the stop_idx is set to a value at our index or higher, we finish.
        """
        nonlocal stop_idx
        logger.debug(f'producer_consumer_wrapper: {func} {in_q} {out_q} {idx} {stop_idx}')
        try:
            if in_q is None:
                for item in func():
                    out_q.put(item)
                    with lock:
                        if stop_idx >= idx:
                            break
                # put the sentinel to signal the end
                out_q.put(sentinel)
            else:
                while True:
                    try:
                        item = in_q.get(timeout=get_timeout)
                        in_q.task_done()
                        if item is sentinel:
                            out_q.put(sentinel)
                            break
                        result = func(item)
                        out_q.put(result)
                    except Empty:
                        continue
                    with lock:
                        if stop_idx >= idx:
                            break
        except StopIteration:
            with lock:
                stop_idx = max(stop_idx, idx)
            out_q.put(sentinel)
            logger.debug(f'stopping thread for {func} {in_q} {out_q} {idx} {stop_idx}')
        logger.debug(f'exiting producer_consumer_wrapper for {func} {in_q} {out_q} {idx} {stop_idx}')

    queues: list[Queue] = [Queue() for _ in range(len(functions))]

    # Start all but the last function in separate threads
    threads = []
    for i in range(len(functions)):
        kwargs = dict(func=functions[i], in_q=None, out_q=queues[i], idx=i)
        if i > 0:
            kwargs['in_q'] = queues[i-1]
        logger.debug(f'starting thread {i} with kwargs {kwargs}')
        thread = threading.Thread(target=producer_consumer_wrapper, kwargs=kwargs)
        thread.start()
        threads.append(thread)

    # Yield results from the last queue
    while threads or queues[-1].qsize() > 0:
        try:
            result = queues[-1].get(timeout=get_timeout)
            queues[-1].task_done()
            if result is sentinel:
                break
            yield result
            # check threads for completion
            threads = [t for t in threads if t.is_alive()]
            logger.debug(f'yielded result, threads left: {len(threads)}')
        except Empty:
            continue
        # stop on control-c and set stop_idx to the last index
        except KeyboardInterrupt:
            with lock:
                stop_idx = len(functions) - 1
            break
    for thread in threads:
        thread.join()

def ensure_singleton(max_delay):
    """Decorator to ensure a given function is run as a singleton.

    This creates a file in the same directory as the function with filename '.{function_name}'
    which contains the output of time.time(). If it detects that the file exists, it checks the
    timestamp. If the timestamp is older than current time - max_delay, then it assumes it was too
    old, and deletes it.

    Once the underlying function finishes (or errors), we delete the file.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kw):
            path = join(dirname(__file__), f'.{basename(fn.__name__)}.lock')
            now = time.time()
            if exists(path):
                # read first line
                with open(path) as f:
                    try:
                        ts = float(f.readline().strip())
                        if now - ts > max_delay:
                            os.remove(path)
                        else:
                            # it's running, so abort
                            logger.info(f'Function {fn.__name__} already running for {now-ts:0.1f}s (< {max_delay}s), aborting')
                            sys.exit(99)
                    except Exception:
                        # not a valid lock file, delete
                        os.remove(path)
            # at this point, it's safe to run the function, so create lock file
            with open(path, 'w') as f:
                f.write(str(now))
            # run the function
            try:
                return fn(*args, **kw)
            finally:
                # delete the lock file on success or exception
                try:
                    os.remove(path)
                except Exception:
                    pass

        return inner
    return decorator

def is_async_callable(obj: typing.Any) -> bool:
    """Checks if the given `obj` is an async callable."""
    while isinstance(obj, functools.partial):
        obj = obj.func

    return asyncio.iscoroutinefunction(obj) or (
        callable(obj) and asyncio.iscoroutinefunction(obj.__call__)
    )

def run_async(coroutine):
    """Runs given async `coroutine` synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to create one
        return asyncio.run(coroutine)
    else:
        # Already running, create a new task and wait for result
        future = asyncio.ensure_future(coroutine)
        done, _ = loop.run_until_complete(asyncio.wait([future]))
        return future.result()

def sync_or_async(async_func):
    """Decorator that turns an async function into either sync or async depending on the caller.

    In particular, you can call the wrapped function either with `await` in an async context, or
    just directly in a normal sync context.
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        if is_async_callable(async_func):
            try:
                loop = asyncio.get_running_loop()
                # We're already inside an event loop: return coroutine for the caller to await
                return async_func(*args, **kwargs)
            except RuntimeError:
                # No running loop (i.e., from synchronous code): run and block
                return asyncio.run(async_func(*args, **kwargs))
        else:
            # If someone passes a sync func (rare case), just call it
            return async_func(*args, **kwargs)

    return wrapper

def background_task(func_or_coroutine) -> None:
    """Runs a task in the background, ignore the result and errors.

    Can handle both sync functions/callables and async coroutines.
    """
    async def bg_task(func_or_coroutine):
        try:
            if inspect.iscoroutine(func_or_coroutine):
                await func_or_coroutine
            elif callable(func_or_coroutine):
                # It's a sync function, call it
                func_or_coroutine()
            else:
                # Assume it's already a result, nothing to do
                pass
        except Exception as e:
            logger.warning(f'Error in background task: {e}')
            logger.info(traceback.format_exc())
    asyncio.create_task(bg_task(func_or_coroutine))

def classify_func_output(output):
    """Return `(is_async, is_generator)` tuple from function `output`"""
    is_async = inspect.iscoroutine(output) or inspect.isasyncgen(output)
    is_generator = inspect.isgenerator(output) or inspect.isasyncgen(output)
    return (is_async, is_generator)

class CollectionUpdater:
    """A simple class that makes it easy to update a collection in more natural ways.

    Particularly when working with databases of various kinds, it's much more efficient to add stuff
    in batches, but in code, it's often more natural to just iterate through items and add them one
    by one. This class lets you have the best of both worlds.

    Each item you add must have a string 'id', and then any other fields you want to use. You
    specify your update frequency (in number of items and/or elapsed time) in the constructor, and
    then call .add() for each item.

    You can also manually call commit() to force a commit at any time.

    Note that when the updater is deleted, it will automatically commit any remaining items, so you
    don't have to worry about the pesky "last commit" that is always annoying to deal with -- as
    soon as this goes out-of-scope (and garbage collected), it will commit. You can also just call
    commit() at the end of your add loop.

    The class also keeps track of all ids ever seen and whether they have been committed or not via
    `ids_seen`.

    It will also call the optional `post_commit_fn(list_of_ids_committed)` after each commit,
    passing in the list of ids in that batch.
    """
    def __init__(self,
                 # add function can take arbitrary args and return anything
                 add_fn: Callable[[Any], Any],
                 item_incr: int=100,
                 time_incr: float=30.0,
                 post_commit_fn: Callable[[list[str]], None]|None=None,
                 debug: bool=False):
        """Initialize the updater with the given underlying `add_fn` and update frequency.

        - add_fn: a function that takes a batch of items to add. It will be called directly with
          `self.to_add`
        - item_incr: number of items to add before committing [default 100]. (Disabled if <= 0)
        - time_incr: elapsed time to wait before committing [default 30.0]. (Disabled if <= 0)

        Note that if both are specified, then whichever comes first triggers a commit.

        You can optionally pass in a `post_commit_fn` to be called after each commit. It is called
        with the list of ids that were just committed.

        If you specify `debug=True`, then commit messages will be printed using logger.info()
        """
        self.add_fn = add_fn
        self.item_incr = item_incr
        self.time_incr = time_incr
        self.last_update = time.time()
        self.to_add = self._reset_to_add()
        self.timer = None
        self.ids_seen: dict[str, bool] = {}
        self.post_commit_fn = post_commit_fn
        self.debug = debug

    def _reset_to_add(self) -> dict[str, list]:
        """Resets our batch of items to add."""
        return dict(ids=[], objects=[])

    def commit(self):
        """Commits the current items to the collection and resets the updater."""
        if not self.to_add['ids']:
            return
        log_func = logger.info if self.debug else logger.debug
        log_func(f'Committing {len(self.to_add["ids"])} items')
        if 'objects' in self.to_add:
            assert len(self.to_add['ids']) == len(self.to_add['objects'])
        self.add_fn(self.to_add)
        for id in self.to_add['ids']:
            self.ids_seen[id] = True
        if self.post_commit_fn:
            self.post_commit_fn(self.to_add['ids'])
        self.to_add = self._reset_to_add()
        self.last_update = time.time()
        if self.timer:
            self.timer.cancel()
        self.timer = None

    def __del__(self):
        """Commit any remaining items before deleting the updater."""
        self.commit()

    def maybe_commit(self):
        """Called to check if we should commit based on the update frequencies."""
        if not self.to_add['ids']:
            return
        if self.item_incr > 0 and len(self.to_add['ids']) >= self.item_incr:
            self.commit()
        if self.time_incr > 0 and time.time() - self.last_update >= self.time_incr:
            self.commit()
        if self.time_incr <= 0:
            return
        # we also want to set a timer to make sure we commit even if we don't add any more items
        if self.timer and self.timer.is_alive(): # timer is already running, we're fine
            return
        # at this point, we need to set a timer
        self.timer = threading.Timer(1.0, self.maybe_commit)
        self.timer.start()

    def add(self, id: str, obj: Any):
        """Adds an `obj` to the updater.

        Note that if the object needs the id inside it, you should include it there as well.
        The param `id` is only for bookkeeping for this class.

        If the update frequency is reached, it will commit the items to the collection.
        """
        assert id not in self.ids_seen, f'ID {id} already seen!'
        self.to_add['ids'].append(id)
        self.to_add['objects'].append(obj)
        self.ids_seen[id] = False
        self.maybe_commit()


class JSONUpdater(CollectionUpdater):
    """A CollectionUpdater specialized for JSON files.

    This takes a JSON path as input and on first use, reads existing items from it. It then appends
    new items to it as they are added via the `add()` method. If `id` is None-like, then it assumes
    the underlying JSON object is a list, otherwise a dict keyed by id. Note that in the former case,
    we create an id based on the length of the list at the time of addition.

    When you commit, it writes out the entire JSON file again with the new items appended. This is
    done to a tempfile (in the same dir as the original) and then renamed to avoid corruption.
    """
    def __init__(self,
                 path: str,
                 item_incr: int=100,
                 time_incr: float=30.0,
                 post_commit_fn: Callable[[list[str]], None]|None=None,
                 debug: bool=False):
        """Initializes the JSONUpdater with the given `path` and update frequencies.

        - path: path to the JSON file to read/write
        - item_incr: number of items to add before committing [default 100]. (Disabled if <= 0)
        - time_incr: elapsed time to wait before committing [default 30.0]. (Disabled if <= 0)

        Note that if both are specified, then whichever comes first triggers a commit.

        You can optionally pass in a `post_commit_fn` to be called after each commit. It is called
        with the list of ids that were just committed.

        If you specify `debug=True`, then commit messages will be printed using logger.info()
        """
        self.path = path
        self.data: list[Any]|dict[str, Any]|None = None
        super().__init__(
            add_fn=self._add_items,
            item_incr=item_incr,
            time_incr=time_incr,
            post_commit_fn=post_commit_fn,
            debug=debug,
        )

    def load_existing(self, as_dict: bool) -> None:
        """Loads our existing data into self.data if it exists; else creates an empty list or dict.

        You must specify whether the data is a dict (if as_dict=True) or a list.
        """
        try:
            with open(self.path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            self.data = {} if as_dict else []
        if isinstance(self.data, list):
            existing_ids = [str(i) for i in range(len(self.data))]
        elif isinstance(self.data, dict):
            existing_ids = list(self.data.keys())
        else:
            raise ValueError(f'JSON file {self.path} must contain a list or dict at top level')
        for id in existing_ids:
            self.ids_seen[id] = True

    def _add_items(self, batch: dict[str, list[Any]]) -> None:
        """Adds the given batch of items to the JSON file."""
        for id, obj in zip(batch['ids'], batch['objects']):
            if isinstance(self.data, list):
                self.data.append(obj)
            elif isinstance(self.data, dict):
                self.data[id] = obj
            else:
                raise ValueError(f'Data must be a list or dict, not {type(self.data)}')
        # write out to a temp file and rename (overwriting existing)
        dir = dirname(abspath(self.path))
        with tempfile.NamedTemporaryFile('w', dir=dir, delete=False) as tf:
            json.dump(self.data, tf, indent=2)
            tempname = tf.name
        shutil.move(tempname, self.path)

    def add(self, id: str|None, obj: Any):
        """Adds an `obj` to the updater.

        If `id` is None-like, we assume the underlying JSON object is a list, otherwise a dict
        keyed by id. Note that in the former case, we create an id based on the length of the list
        at the time of addition.

        If the update frequency is reached, it will commit the items to the collection.
        """
        as_dict = id is not None
        self.load_existing(as_dict=as_dict)
        if id is None:
            assert isinstance(self.data, list), 'Data must be a list if id is None'
            id = str(len(self.data))
        super().add(id, obj)


class Singleton:
    """A singleton implementation that hashes the args and kwargs to the init of a class"""
    def __init__(self, cls):
        self._cls = cls
        self._instances = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _make_key(self, args, kwargs):
        # Note: assumes all args and kwargs are hashable
        return (args, frozenset(kwargs.items()))

    def __call__(self, *args, **kwargs):
        key = self._make_key(args, kwargs)

        # Try fast path first
        instance = self._instances.get(key)
        logger.debug(f'Singleton __call__ key={key} instance={instance}')
        if instance is not None:
            return instance

        # Thread-safe path
        with self._lock:
            instance = self._instances.get(key)
            if instance is None:
                instance = self._cls(*args, **kwargs)
                self._instances[key] = instance
        logger.debug(f'  loaded new instance={instance}')
        return instance

    async def __acall__(self, *args, **kwargs):
        key = self._make_key(args, kwargs)

        # Try fast path first
        instance = self._instances.get(key)
        if instance is not None:
            return instance

        async with self._async_lock:
            instance = self._instances.get(key)
            if instance is None:
                if iscoroutinefunction(self._cls.__init__):
                    instance = await self._cls(*args, **kwargs)
                else:
                    instance = self._cls(*args, **kwargs)
                self._instances[key] = instance
        return instance

    def __get__(self, obj, objtype=None):
        # In case someone applies this to a method accidentally
        return self


if __name__ == "__main__":
    # example usage
    logging.basicConfig(level=logging.DEBUG)
    def example_producer():
        for i in range(20):
            logger.debug(f"Starting item {i}")
            yield f"item {i}"

    def example_processor(item):
        logger.debug(f"Processing {item}")
        return f"processed {item}"

    def example_final_processor(item):
        if item == "processed item 10": raise StopIteration
        return item.upper()

    # Function chain
    functions = [example_producer, example_processor, example_final_processor]

    # Start the chained producer-consumer processing
    for final_result in chained_producer_consumers(functions, sleep_interval=0.1):
        logger.debug(f"Final output: {final_result}")

