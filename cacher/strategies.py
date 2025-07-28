"""Cache strategies: ways to modify caching behavior."""

from __future__ import annotations

import queue
import threading
import time

from abc import ABC
from typing import Any, Generic

from nkpylib.cacher.constants import KeyT

class CacheStrategy(Generic[KeyT]):
    """Base class for cache strategies.

    Strategies can hook into different stages of cache operations:
    - pre_get: Before retrieving a value
    - post_get: After retrieving a value
    - pre_set: Before setting a value
    - post_set: After setting a value
    - pre_delete: Before deleting a value
    - post_delete: After deleting a value
    """
    def pre_get(self, key: KeyT) -> bool:
        """Called before retrieving a value.

        Returns:
            False to skip cache lookup, True to proceed
        """
        return True

    def post_get(self, key: KeyT, value: Any) -> Any:
        """Called after retrieving a value.

        Args:
            key: The cache key
            value: The retrieved value

        Returns:
            Potentially modified value
        """
        return value

    def pre_set(self, key: KeyT, value: Any) -> bool:
        """Called before setting a value.

        Returns:
            False to skip caching, True to proceed
        """
        return True

    def post_set(self, key: KeyT, value: Any) -> None:
        """Called after setting a value."""
        pass

    def pre_delete(self, key: KeyT) -> bool:
        """Called before deleting a value. Returns False to skip deletion, True to proceed"""
        return True

    def post_delete(self, key: KeyT) -> None:
        """Called after deleting a value."""
        pass

    def pre_clear(self) -> bool:
        """Called before clearing all entries. Returns False to skip clearing, True to proceed"""
        return True

    def post_clear(self) -> None:
        """Called after clearing all entries."""
        pass


class RateLimiter(CacheStrategy[KeyT]):
    """Strategy that enforces a global minimum interval between operations.

    This is useful for rate-limiting API calls or other resources that need
    to be accessed with some delay between requests, regardless of which key
    is being accessed.
    """
    def __init__(self, min_interval: float):
        """Initialize with minimum interval between requests.

        Args:
            min_interval: Minimum time (in seconds) between any operations
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0

    def pre_get(self, key: KeyT) -> bool:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
        return True


class TTLPolicy(CacheStrategy[KeyT]):
    """Strategy that enforces a time-to-live (TTL) on cached items.

    Items older than the TTL are considered invalid and will be re-fetched.
    """
    def __init__(self, ttl_seconds: float):
        """Initialize with TTL duration.

        Args:
            ttl_seconds: Time-to-live in seconds for cached items
        """
        self.ttl = ttl_seconds
        self.timestamps: dict[Any, float] = {}

    def pre_get(self, key: KeyT) -> bool:
        if key in self.timestamps:
            age = time.time() - self.timestamps[key]
            if age > self.ttl:
                return False  # Skip cache lookup, forcing a miss
        return True

    def post_set(self, key: KeyT, value: Any) -> None:
        self.timestamps[key] = time.time()

    def post_delete(self, key: KeyT) -> None:
        self.timestamps.pop(key, None)

    def post_clear(self) -> None:
        self.timestamps.clear()


class BackgroundWriteStrategy(CacheStrategy[KeyT]):
    """Strategy that performs cache writes in a background thread.

    This is useful when:
    - Cache writes are slow (e.g., to disk or network)
    - You want to return to the caller quickly
    - Write order doesn't matter
    - It's ok if the most recent writes are lost on crash
    """
    def __init__(self, queue_size: int = 1000):
        """Initialize with maximum queue size."""
        self.write_queue: queue.Queue[tuple[str, KeyT, Any]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        """Background thread that processes writes."""
        while not self.stop_event.is_set():
            try:
                # Wait for work with timeout to check stop_event periodically
                op, key, value = self.write_queue.get(timeout=0.1)
                if op == 'set':
                    self._backend._set_value(key, value)
                elif op == 'delete':
                    self._backend._delete_value(key)
                elif op == 'clear':
                    self._backend._clear()
                self.write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in bg write worker: {e}")

    def pre_set(self, key: KeyT, value: Any) -> bool:
        """Queue the write operation."""
        try:
            self.write_queue.put(('set', key, value), block=False)
            return False  # Skip the normal write
        except queue.Full:
            return True  # Queue full, do normal write

    def pre_delete(self, key: KeyT) -> bool:
        """Queue the delete operation."""
        try:
            self.write_queue.put(('delete', key, None), block=False)
            return False  # Skip the normal delete
        except queue.Full:
            return True  # Queue full, do normal delete

    def pre_clear(self) -> bool:
        """Queue the clear operation."""
        try:
            self.write_queue.put(('clear', None, None), block=False)
            return False  # Skip the normal clear
        except queue.Full:
            return True  # Queue full, do normal clear

    def wait(self, timeout: float|None = None) -> bool:
        """Wait for all queued operations to complete.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait forever

        Returns:
            True if all operations completed, False if timeout occurred
        """
        try:
            self.write_queue.join()
            return True
        except TimeoutError:
            return False

    def stop(self, timeout: float|None = None):
        """Stop the background thread and wait for it to finish.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait forever
        """
        self.stop_event.set()
        if timeout is not None:
            self.worker.join(timeout)
        else:
            self.worker.join()


