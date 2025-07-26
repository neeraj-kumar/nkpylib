"""Cache strategies: ways to modify caching behavior.

"""

class CacheStrategy(ABC, Generic[KeyT]):
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


