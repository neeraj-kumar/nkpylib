import random
import time
from urllib.request import urlopen
from urllib.error import URLError

class ExpensiveClass:
    """A class with some expensive methods to test method caching."""
    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier
    
    def expensive_method(self, x: int, y: int) -> int:
        """An expensive method that we'll want to cache."""
        time.sleep(0.1)  # Simulate expensive work
        return (x * y) * self.multiplier

def fibonacci(n: int) -> int:
    """Compute nth fibonacci number recursively (intentionally inefficient)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def fetch_url_size(url: str) -> int:
    """Fetch URL and return content length."""
    try:
        with urlopen(url, timeout=10) as response:
            return len(response.read())
    except URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

def random_choice(items: list) -> str:
    """Return random item from list (to test cache consistency)."""
    return random.choice(items)
