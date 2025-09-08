"""Cache strategies: ways to modify caching behavior."""

from __future__ import annotations

import queue
import threading
import time

import atexit
from abc import ABC
from typing import Any, Callable, Generic

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
    def __init__(self, backend=None):
        self._backend = backend

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
    def __init__(self, min_interval: float|Callable[[KeyT], float]):
        """Initialize with minimum interval between requests.

        Args:
            min_interval: Minimum time (in seconds) between any operations,
                          or a function that takes in a key and returns the interval to use.
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0

    def pre_get(self, key: KeyT) -> bool:
        now = time.time()
        elapsed = now - self.last_request_time
        min_interval = self.min_interval(key) if callable(self.min_interval) else self.min_interval
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
        return True


class TTLPolicy(CacheStrategy[KeyT]):
    """Strategy that enforces a time-to-live (TTL) on cached items.

    Items older than the TTL are considered invalid and will be re-fetched.
    """
    def __init__(self, ttl_seconds: float|Callable[[KeyT], float]):
        """Initialize with TTL duration.

        Args:
            ttl_seconds: Time-to-live in seconds for cached items
                         or a function that takes in a key and returns the TTL to use.
        """
        self.ttl = ttl_seconds
        self.timestamps: dict[Any, float] = {}

    def pre_get(self, key: KeyT) -> bool:
        if key in self.timestamps:
            age = time.time() - self.timestamps[key]
            ttl = self.ttl(key) if callable(self.ttl) else self.ttl
            if age > ttl:
                return False  # Skip cache lookup, forcing a miss
        return True

    def post_set(self, key: KeyT, value: Any) -> None:
        self.timestamps[key] = time.time()

    def post_delete(self, key: KeyT) -> None:
        self.timestamps.pop(key, None)

    def post_clear(self) -> None:
        self.timestamps.clear()


class DelayedWriteStrategy(CacheStrategy[KeyT]):
    """Strategy that delays cache writes for better performance.

    Two modes are supported:
    - "background": Writes happen in a background thread (good for slow backends)
    - "memory": Writes are cached in memory and flushed periodically (good for fast access)

    You can also configure:
    - batch_size: How many writes to batch together (<=1 means one at a time)
    - flush_interval: How often to flush to backend (<=0 means immediate)
    """
    def __init__(self, 
                 mode: str = "background",
                 batch_size: int = 1,
                 flush_interval: float = 0.0):
        """Initialize with write mode and timing settings.
        
        Args:
            mode: Either "background" (thread) or "memory" (cache)
            batch_size: Number of writes to batch (<=1 means one at a time)
            flush_interval: Seconds between flushes (<=0 means immediate)
        """
        if mode not in ("background", "memory"):
            raise ValueError("mode must be 'background' or 'memory'")
        self.mode = mode
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.0, flush_interval)
        
        # For background mode
        if mode == "background":
            self.write_queue: queue.Queue[tuple[str, KeyT, Any]] = queue.Queue(maxsize=batch_size)
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.worker.start()
            atexit.register(self.shutdown)
            self._batch: list[tuple[str, KeyT, Any]] = []
            
        # For memory mode
        else:
            self._cache: dict[KeyT, Any] = {}
            self._dirty = set()
            self._last_flush = time.time()

    def _worker(self):
        """Background thread that processes writes."""
        batch: list[tuple[str, KeyT, Any]] = []
        last_flush = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Get item with timeout to check stop_event
                try:
                    op, key, value = self.write_queue.get(timeout=0.1)
                    batch.append((op, key, value))
                    self.write_queue.task_done()
                except queue.Empty:
                    pass

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size or
                    (self.flush_interval > 0 and 
                     time.time() - last_flush >= self.flush_interval)
                )
                
                if should_flush and batch:
                    # Group by operation type
                    sets = [(k,v) for op,k,v in batch if op == 'set']
                    deletes = [k for op,k,v in batch if op == 'delete']
                    clears = any(op == 'clear' for op,k,v in batch)
                    
                    # Process in order: clear, deletes, sets
                    if clears:
                        self._backend._clear()
                    for key in deletes:
                        self._backend._delete_value(key)
                    for key, value in sets:
                        self._backend._set_value(key, value)
                        
                    batch.clear()
                    last_flush = time.time()
                    
            except Exception as e:
                print(f"Error in delayed write worker: {e}")
                batch.clear()

    def pre_get(self, key: KeyT) -> bool:
        """Check memory cache first in memory mode."""
        if self.mode == "memory" and key in self._cache:
            return False
        return True

    def post_get(self, key: KeyT, value: Any) -> Any:
        """Return from memory cache in memory mode."""
        if self.mode == "memory" and key in self._cache:
            return self._cache[key]
        return value

    def pre_set(self, key: KeyT, value: Any) -> bool:
        """Handle write based on mode."""
        if self.mode == "background":
            try:
                self.write_queue.put(('set', key, value), block=False)
                return False
            except queue.Full:
                return True
        else:  # memory mode
            self._cache[key] = value
            self._dirty.add(key)
            self._maybe_flush()
            return False

    def pre_delete(self, key: KeyT) -> bool:
        """Handle delete based on mode."""
        if self.mode == "background":
            try:
                self.write_queue.put(('delete', key, None), block=False)
                return False
            except queue.Full:
                return True
        else:  # memory mode
            if key in self._cache:
                del self._cache[key]
            self._dirty.add(key)
            self._maybe_flush()
            return False

    def pre_clear(self) -> bool:
        """Handle clear based on mode."""
        if self.mode == "background":
            # Clear queue
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    self.write_queue.task_done()
                except queue.Empty:
                    break
            self.write_queue.put(('clear', None, None), block=False)
            return False
        else:  # memory mode
            self._cache.clear()
            self._dirty.clear()
            self._backend._clear()
            self._last_flush = time.time()
            return False

    def _maybe_flush(self):
        """Check if we should flush memory cache."""
        if not self.flush_interval:
            self.flush()
        elif time.time() - self._last_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush pending writes to backend."""
        if self.mode == "memory" and self._dirty:
            for key in self._dirty:
                if key in self._cache:
                    self._backend._set_value(key, self._cache[key])
            self._dirty.clear()
            self._last_flush = time.time()

    def shutdown(self, timeout: float|None = None):
        """Clean shutdown of background thread or memory cache."""
        if self.mode == "background":
            # Process remaining items and stop thread
            self.write_queue.join()
            self.stop_event.set()
            if timeout is not None:
                self.worker.join(timeout)
            else:
                self.worker.join()
        else:  # memory mode
            self.flush()
