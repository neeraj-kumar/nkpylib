Full-Featured Cacher
--------------------
Written by Neeraj Kumar <me@neerajkumar.org>


Implementation consists of:
- Cache: Main class managing multiple backends and policies
- CacheBackend: Base class for storage+formatter combinations
- CachePolicy: Base class for cache policies (TTL, limits, etc)
- CacheFormatter: Base class for serialization formats


# Examples

Let's say we want to cache the results of a function that fetches some data from a remote API:

```python

from cacher import Cache, CacheBackend, CachePolicy, CacheFormatter


def fetch_from_api(endpoint: str, **kwargs) -> dict:
    print(f'fetching from api at {endpoint} with kw {kwargs}')
    ...


user1_posts = fetch_from_api('/posts', userId=1, sort='asc')
#> fetching from api at /posts with kw {'userId': 1, 'sort': 'asc'}

# Define a simple cache backend that stores data in memory
cacher = MemoryBackend(fn=fetch_from_api)
user1_posts_cached = cacher('/posts', userId=1, sort='asc')
#> fetching from api at /posts with kw {'userId': 1, 'sort': 'asc'}
assert user1_posts == user1_posts_cached
user1_posts_cached = cacher('/posts', userId=1, sort='asc')
#> (no output, cached result returned)
assert user1_posts == user1_posts_cached
```

Or if you always want to cache outputs from that function, you can apply it as a decorator:

```python
cacher = MemoryBackend()

@cacher
def fetch_from_api(endpoint: str, **kwargs) -> dict:
    ...

user1_posts_cached = fetch_from_api('/posts', userId=1, sort='asc')

```


# OLD STUFF, ignore

                 fn: Callable|None=None,
                 *,
                 formatter: CacheFormatter,
                 keyer: Keyer|None = None,
                 strategies: list[CacheStrategy]|None = None,
                 error_on_missing: bool = True,
                 **kwargs):


Design for new version:
- different key_funcs
- runnable in background/async/futures/threads
- batchable
- decorable
- ignore certain args
- cache a list or dict:
  - figure out which are already cached and which aren't
  - where underlying function takes a batch
- something for imdb data dump updates -> either run function or read from db/cache?
- expiration criteria
  - time (either relative from now, or absolute time)
  - count
  - memory
  - other?
- single-value cache with different keys
  - e.g. the embeddings cache which checks for current normed, scale_mean, scale_std
- TTL
- ignore cache for individual calls
- archival
- delay + variance
- different formats:
  - pickle
- different backing stores - mem, fs, lmdb, numpylmdb
- one file per key, or one file overall, or ...?
- stats/timing
- prefetch?
- caching binary files (e.g. web fetch request)
- per-host timers (like in make_request)?
- works on class methods (how to check for other instance var dependencies?)
- store revisions?
- named revisions?
- external dependencies:
    external_counter = 0
    @cache(depends_on=lambda:[external_counter])
    def things_with_external(a,b,c):
        global external_counter
        from time import sleep; sleep(1) # <- simulating a long-running process
        return external_counter + a + b + c
