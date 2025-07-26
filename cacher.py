"""A fully-functional cacher with all the bells-and-whistles.


Design for new version:
- different key_funcs
- runnable in background/async/futures/threads
- batchable
- decorable
- ignore certain args
- cache a list or dict:
  - figure out which are already cached and which aren't
  - where underlying function takes a batch
- use tempfile + rename when writing files
- JSON encoder/decoder class
- something for imdb data dump updates -> either run function or read from db/cache?
- expiration criteria
  - time
  - count
  - memory
  - other?
- single-value cache with different keys
  - e.g. the embeddings cache which checks for current normed, scale_mean, scale_std
- TTL
- ignore cache for individual calls
- force invalidate, either single key or all
- archival
- expiration
- delay + variance
- different formats - json, pickle, ...
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
"""
