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

from cacher import MemoryBackend


def fetch_from_api(endpoint: str, **kwargs) -> dict:
    print(f'fetching from api at {endpoint} with kw {kwargs}')
    ...


user1_posts = fetch_from_api('/posts', userId=1, sort='asc')
# output: fetching from api at /posts with kw {'userId': 1, 'sort': 'asc'}

# Define a simple cache backend that stores data in memory
cacher = MemoryBackend(fn=fetch_from_api)
user1_posts_cached = cacher('/posts', userId=1, sort='asc')
# output: fetching from api at /posts with kw {'userId': 1, 'sort': 'asc'}
assert user1_posts == user1_posts_cached
user1_posts_cached = cacher('/posts', userId=1, sort='asc')
# [no output, cached result returned]
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

Now let's say you want to store the cache to disk as JSON files, one per separate set of request
parameters:
```python
from cacher import SeparateFileBackend, JsonFormatter

cacher = SeparateFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
)

cacher('/posts', userId=1, sort='asc')
# the results of that call are stored in '/cache/dir/posts_userId=1_sort=asc.json'
cacher('/posts', userId=2)
# the results of that call are stored in '/cache/dir/posts_userId=2'
```
If you need to send in some parameters that you don't want to be part of the cache key, such as an
api key, you can cache a lambda like this:
```python
from cacher import SeparateFileBackend, JsonFormatter

API_KEY = 'some_api_key'
cacher = SeparateFileBackend(
    fn=lambda endpoint, **kwargs: fetch_from_api(endpoint, api_key=API_KEY, **kwargs),
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
)
```

Cachers also work fine if you're using methods on a class, such as this:
```python
class ApiClient:
    def call_api(self, endpoint: str, **kwargs) -> dict:
        print(f'fetching from api at {endpoint} with kw {kwargs}')
        ...

client = ApiClient()

cacher = SeparateFileBackend(
    fn=client.call_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
)
```
You can also apply the decorator version instead:

```python
from cacher import SeparateFileBackend, JsonFormatter

cacher = SeparateFileBackend(
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
)

class ApiClient:
    @cacher
    def call_api(self, endpoint: str, **kwargs) -> dict:
        print(f'fetching from api at {endpoint} with kw {kwargs}')
        ...
```
So far, we've been using the default `Keyer`, which generates a filename based on the input args and
kwargs to the function. This works fine for many cases, and has the advantage that the filenames are
human-readable, which is useful for debugging and manual inspection of the cache. However,
these filenames can get very long and unwieldy if you have lots of parameters.

So instead, you can use a `HashStringKeyer` that generates a hash of the parameters instead:

```python
from cacher import SeparateFileBackend, JsonFormatter, HashStringKeyer

cacher = SeparateFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
    keyer=HashStringKeyer(),
)

cacher('/posts', userId=1, sort='asc')
# Now the results of that call are stored in '/cache/dir/5f4dcc3b5aa765d61d8327deb882cf99.json'
```
If you run this at large scale, you're going to end up with a directory with thousands of files,
which can be slow to read from and write to. So you can also specify a `filename_fn` that takes the
input key and outputs a filename (including subdirs). Let's use the first 4 chars of the hash as a
subdir:

```python
from cacher import SeparateFileBackend, JsonFormatter, HashStringKeyer

cacher = SeparateFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
    keyer=HashStringKeyer(),
    filename_fn=lambda key: f'{key[:4]}/{key}.json',
)

cacher('/posts', userId=1, sort='asc')
# Now the results of that call are stored in '/cache/dir/5f4d/5f4dcc3b5aa765d61d8327deb882cf99.json'
```

Of course, one problem with using a disk cache is that it stores stuff forever, which can lead to
results being out of date. You can use a `TTLPolicy` (which is a type of `Strategy`) to automatically
invalidate cached entries after a day:

```python
from cacher import SeparateFileBackend, JsonFormatter, HashStringKeyer, TTLPolicy

cacher = SeparateFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
    keyer=HashStringKeyer(),
    filename_fn=lambda key: f'{key[:4]}/{key}.json',
    strategies=[TTLPolicy(ttl_seconds=60*60*24)],
)
```

Another issue with APIs is that they can return errors, and you probably don't want to cache those
(e.g. overloaded or rate-limited errors). The cachers automatically raise the underlying error and
don't cache if the function raises an exception.

For rate-limiting specifically, you can also add a `RateLimiter` (another `Strategy`) to enforce a
minimum interval between calls:
```python
from cacher import SeparateFileBackend, JsonFormatter, HashStringKeyer, TTLPolicy, RateLimiter

cacher = SeparateFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_dir='/cache/dir',
    keyer=HashStringKeyer(),
    filename_fn=lambda key: f'{key[:4]}/{key}.json',
    strategies=[
        TTLPolicy(ttl_seconds=60*60*24),
        RateLimiter(min_interval=1.0),
    ],
)
```
An alternate way to store the cached data is in a single JSON file, where each top-level key is a
hash of the input parameters. You can do that by using a `JointFileBackend` instead of the
`SeparateFileBackend`:

```python
from cacher import JointFileBackend, JsonFormatter, HashStringKeyer, TTLPolicy, RateLimiter

cacher = JointFileBackend(
    fn=fetch_from_api,
    formatter=JsonFormatter(),
    cache_path='/cache/dir/api_cache.json',
    keyer=HashStringKeyer(),
    strategies=[
        TTLPolicy(ttl_seconds=60*60*24),
        RateLimiter(min_interval=1.0),
    ],
)

cacher('/posts', userId=1, sort='asc')
# Now the results of that call are stored in '/cache/dir/api_cache.json', under the key '5f4dcc3b5aa765d61d8327deb882cf99'
```


# OLD STUFF, ignore

                 fn: Callable|None=None,
                 *,
                 formatter: CacheFormatter,
                 keyer: Keyer|None = None,
                 strategies: list[CacheStrategy]|None = None,
                 error_on_missing: bool = True,
                 **kwargs):
