"""Caching-related python utilities, written by Neeraj Kumar.

An easy way to use this in your code is to create a partial function like this:

from functools import partial
fbcache = partial(APICache, cachedir='db/freebase/cache/%(fn)s/', mindelay=0.2, nthreads=5, serializer='json')

Then you can apply it as a decorator to any function:

@fbcache
def myfunc():
    ...
"""

import json
import os
import pickle
import random
import sys
import time
import urllib

from hashlib import md5
from queue import Queue

def utf(s):
    """Converts the given string to utf-8, if it isn't already."""
    if isinstance(s, unicode):
        return s
    try:
        return s.decode('utf-8', 'ignore')
    except Exception:
        return s

def spawnWorkers(num, target, name=None, args=(), kwargs={}, daemon=1, interval=0):
    """Spawns the given number of workers, by default daemon, and returns a list of them.
    'interval' determines the time delay between each launching"""
    from threading import Thread
    threads = []
    for i in range(num):
        t = Thread(target=target, name=name, args=args, kwargs=kwargs)
        t.setDaemon(daemon)
        t.start()
        threads.append(t)
        time.sleep(interval)
    return threads

def memoize(fn):
    """Decorator to cache a function.
    Make sure it's a functional method (i.e., no side effects).
    (copied from utils.py for completeness here)
    """
    cache = {}
    def newfn(*args, **kw):
        key = (tuple(args), tuple(sorted(kw.items())))
        if key in cache:
            return cache[key]
        else:
            cache[key] = val = fn(*args, **kw)
            return val
    newfn.__name__ = fn.__name__ + ' (MEMOIZED)'
    newfn.__module__ = fn.__module__
    return newfn

def cleanstr(s):
    """A simple function to generate a clean string from the given input object or string."""
    try:
        s = str(s)
    except Exception:
        s = utf(s).encode('utf-8')
    s = s.replace('/', '.')
    return s

def seq2str(seq):
    """Converts a sequence of elements to a string in a standard, filesystem-safe way"""
    return ','.join(map(cleanstr, seq))


def dict2str(d):
    """Converts a dict to a string in a standard, filesystem-safe way"""
    els = ['%s=%s' % (k,v) for k,v in sorted(d.items())]
    return seq2str(els)

def defaultcachefunc(*args, **kw):
    """Default cachefunc"""
    ret = seq2str(args)
    if kw:
        ret += '@'+dict2str(kw)
    return ret

def hashed_cache_func(*args, **kw):
    """Generates a hashed cache filename"""
    return md5(defaultcachefunc(*args, **kw).encode('utf-8')).hexdigest()


class APICache(object):
    """A wrapper for an API that handles caching, rate-limiting, and multi-threading."""
    def __init__(self,
                 apifunc,
                 cachefunc=None,
                 cachedir='cache/',
                 mindelay=0.5,
                 delay_variance=0.1,
                 nthreads=1,
                 expiration=0,
                 expiration_policy='overwrite',
                 serializer='json',
                 defaultkwargs=None):
        """Creates a new cache for the given apifunc.
        Params:
            cachefunc - Converts apifunc requests (*args, **kwargs) into a filename (MINUS extension!)
                        If not given, then uses all args and kwargs to generate a unique filename.
            cachedir - The cache paths given by cachefunc() are appended to this cachedir.
                       This can include format specifiers, because it is rendered using an object
                       containing:
                           fn: function name
            mindelay - Minimum delay (in secs) between multiple requests to the API. Set to 0 to disable.
            delay_variance - The variance in the delay (as a multiple of the delay) between multiple requests to the API.
            nthreads - Number of simultaneous threads to use.
            expiration - If > 0, then expires cache after given number of seconds.
            expiration_policy - What to do when a cached page expires.
                            If 'overwrite' (default), then just overwrite new version.
                            If 'archive', then appends modification timestamp to fname.
            serializer - One of 'pickle' or 'json' -- the file format to save the cache in.
            defaultkwargs - Default kwargs to add to every API request. This is useful for API keys, etc.
                            (These are not included in the default cachefunc filenames.)
        """
        # save params
        self.apifunc = apifunc
        if not cachefunc:
            cachefunc = defaultcachefunc
        self.cachefunc = cachefunc
        self.cachedir = cachedir % dict(fn=apifunc.__name__)
        self.mindelay = mindelay
        self.delay_variance = delay_variance
        self.nthreads = nthreads
        self.serializer = serializer
        self.expiration, self.expiration_policy = expiration, expiration_policy
        self.defaultkwargs = defaultkwargs
        # instance vars
        self.lastcall = 0
        self.inq, self.outq = Queue(), Queue()
        # spawn threads
        self.threads = spawnWorkers(self.nthreads, self._qprocess)

    def _sleep(self):
        """Sleeps until we're ready to make a call again"""
        cur_delay = self.mindelay * (1 + self.delay_variance * (random.random()-0.5))
        while 1:
            diff = cur_delay - (time.time()-self.lastcall)
            if diff <= 0:
                return
            time.sleep(max(diff, 0.01))

    def cachepath(self, *args, **kw):
        """Returns the path for the cached filename"""
        cachename = self.cachefunc(*args, **kw)
        ret = os.path.join(self.cachedir, cachename)+'.'+self.serializer
        return ret

    def archivepath(self, cachefname):
        """Returns the archival path for a given cache filename.
        Appends the last modified timestamp for it.
        """
        modtime = os.stat(cachefname).st_mtime
        ext = '.'+self.serializer
        base = cachefname.rsplit(ext, 1)[0]
        ret = '%s-%f%s' % (base, modtime, ext)
        return ret

    def load_cache(self, cachepath):
        """Loads the cache from the given cachepath.
        If not found, raises IOError."""
        loadfunc = json.load if self.serializer == 'json' else pickle.load
        try:
            # check for recency
            if self.expiration > 0:
                elapsed = time.time() - os.stat(cachepath).st_mtime
                #print >>sys.stderr, '%s exp, %s elapsed' % (self.expiration, elapsed)
                if elapsed > self.expiration:
                    if self.expiration_policy == 'archive':
                        os.rename(cachepath, self.archivepath(cachepath))
                    raise IOError
            return loadfunc(open(cachepath))
        except Exception as e:
            #print >>sys.stderr, 'Could not load cache file %s: %s' % (cachepath, e)
            raise IOError('Could not load cache file %s: %s' % (cachepath, e))

    def save_cache(self, obj, cachepath):
        """Saves the given obj to the given cachepath.
        Returns the file size of the cache file."""
        try:
            os.makedirs(os.path.dirname(cachepath))
        except Exception:
            pass
        tmpfname = cachepath+'.tmp-%d' % (int(time.time()*1000))
        if self.serializer == 'json':
            with open(tmpfname, 'w') as f:
                json.dump(obj, f, indent=2, sort_keys=1)
        elif self.serializer == 'pickle':
            with open(tmpfname, 'wb') as f:
                pickle.dump(obj, f, -1)
        try:
            os.rename(tmpfname, cachepath)
            size = os.stat(cachepath).st_size
        except Exception:
            print('save_cache rename failed from %s to %s' % (tmpfname, cachepath))
            raise
        return size

    def __call__(self, *args, **kw):
        """Calls the api function synchronously"""
        cachepath = self.cachepath(*args, **kw)
        try:
            # try returning from cache first
            return self.load_cache(cachepath)
        except IOError:
            # not found, so run api query
            self._sleep()
            self.lastcall = time.time()
            ret = self.apifunc(*args, **kw)
            self.save_cache(ret, cachepath)
            return ret

    def call_many(self, allargs):
        """Calls the api function many times.
        Put tuples of (*args, **kw) into allargs.
        Yields results in the same order as inputs, as we compute them.
        """
        seqs = []
        # add all inputs to queue
        for args, kw in allargs:
            t = time.time()
            seqs.append(t)
            self.inq.put((t, args, kw))
        # read all outputs
        outs = {}
        while len(seqs) > 0:
            t, ret = self.outq.get()
            # if we don't know this seq number, put it back on the queue
            if t not in seqs:
                self.outq.put((t, ret))
                time.sleep(0.01)
                continue
            # if this is the first item, yield it
            if not seqs: break
            if t == seqs[0]:
                seqs.pop(0)
                yield ret
                # also check if we have the next item(s) done
                while seqs and seqs[0] in outs:
                    t = seqs.pop(0)
                    ret = outs.pop(t)
                    yield ret
                continue
            # else, save it for future use
            outs[t] = ret

    def _qprocess(self):
        """Processes our input queue.
        Call this multiple times from threads.
        """
        while 1:
            t, args, kw = self.inq.get()
            ret = self.__call__(*args, **kw)
            self.outq.put((t, ret))

