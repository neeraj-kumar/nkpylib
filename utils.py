#!/usr/bin/env python
"""Lots of small python utilities, written by Neeraj Kumar.

Table of Contents
-----------------

Because sphinx-autodoc doesn't allow me to put section headings, here's a table
of contents, each of which points to the first function in each section


- Decorators: :func:`tracefunc`
- Timing Utils: :func:`getTimestamp`
- Memory Utils: :data:`MEMORY_UNITS`
- Logging Utils: :func:`log`
- Itertools and Sequences Utils: :func:`arange`
- Dict Utils: :func:`getDictValues`
- Math Utils: :func:`clamp`
- Geometry Utils: :func:`getDistance`
- Probability and Sampling Utils: :func:`minsample`
- Other/misc: :func:`spawnWorkers`


Code Starts Now
-----------------
"""

from __future__ import annotations

import code
import functools
import inspect
import math
import os
import traceback
import json
import pickle
import re
import random
import string
import smtplib
import sys
import tempfile
import threading
import time
import readline
import rlcompleter

from calendar import timegm
from datetime import date, datetime
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.request import urlopen
from itertools import *
from math import pi, exp, sqrt, sin, cos, radians, atan2, acos
from threading import Thread
from typing import Any, Literal, Type, TypeVar, get_args, get_origin
from random import choice, shuffle
from subprocess import Popen, PIPE

## DECORATORS
def tracefunc(fn):
    """Decorator that traces function exits, arguments, and return values"""
    def newfn(*args, **kw):
        ret = fn(*args, **kw)
        log('[%s Trace] args: %s, kw: %s, ret: %s' % (fn.__name__, args, kw, ret))
        return ret
    newfn.__name__ = fn.__name__ + ' (TRACED)'
    newfn.__module__ = fn.__module__
    return newfn

def timed(fn):
    """Decorator that times a function, log()-ing the time spent."""
    def newfn(*args, **kw):
        start = time.time()
        r = fn(*args, **kw)
        elapsed = time.time() - start
        log('-----  Function %s.%s took %s secs' % (fn.__module__, fn.__name__, elapsed))
        return r
    newfn.__name__ = fn.__name__ + ' (TIMED)'
    newfn.__module__ = fn.__module__
    return newfn

def memuse(fn):
    """Decorator that log()s the memory usage of a function.

    .. note::
       if there's no /proc filesystem, this does nothing
    """
    if not os.path.exists('/proc/'): return fn
    def newfn(*args, **kw):
        m1 = procmem()
        r = fn(*args, **kw)
        used = procmem() - m1
        log('*****  Function %s.%s took %d bytes' % (fn.__module__, fn.__name__, used))
        return r
    newfn.__name__ = fn.__name__ + ' (MEMUSE)'
    newfn.__module__ = fn.__module__
    return newfn

def memoizelist(fn):
    """Decorator to cache a function which takes a list of values.
    This differs from the basic memoize in that the first arg takes a list of values.
    The return value is a list of outputs, one corresponding to each input. Only inputs
    in the list that weren't previously computed are computed.

    .. warning::
       Make sure it's a functional method (i.e., no side effects)!
    """
    cache = {}
    def newfn(*args, **kw):
        key = tuple(args[1:])+tuple(sorted(kw.items()))
        cur = cache.setdefault(key, {})
        # create a list of ids which need to be done
        todo = []
        #print >>sys.stderr, 'Got args %s' % (args,)
        for i in args[0]:
            #print >>sys.stderr, '  For %s, in cur = %s' % (i, i in cur)
            # if this index is not in the cache or expired, we need to do it
            if i not in cur:
                todo.append(i)
        #print >>sys.stderr, '  Todo now contains %s' % (todo,)
        # call the function (if needed) with this reduced set of entries
        if todo:
            newargs = (todo,) + tuple(args[1:])
            #print >>sys.stderr, '  newargs are %s' % (newargs,)
            vals = fn(*newargs, **kw)
            assert len(vals) == len(todo), "We should have had %d outputs, but instead only had %d" % (len(todo), len(vals))
            #print >>sys.stderr, '  got back vals from func: %s' % (vals,)
            for i, val in zip(todo, vals):
                cur[i] = val
            #print >>sys.stderr, '  cur now contains: %s' % (cur.keys(),)
        # now build the final output
        #print >>sys.stderr, '  Finally, args[0] should still contain the old things: %s' % (args[0],)
        output = [cur[i] for i in args[0]]
        return output
    newfn.__name__ = fn.__name__ + ' (MEMOIZED LIST)'
    newfn.__module__ = fn.__module__
    return newfn

def threadedmemoize(fn):
    """Decorator to cache a function, in a thread-safe way.
    This means that different threads computing the same value get stored separately.

    .. warning::
       Make sure it's a functional method (i.e., no side effects)!

    .. warning::
       Not tested very much.
    """
    cache = {}
    def newfn(*args):
        now = time.time()
        t = threading.currentThread().getName()
        key = (t, args)
        if key in cache:
            return cache[key]
        else:
            #logger.debug('Memoizing %s with key=%s (%d entries in cache)' % (fn.__name__, key, len(cache)))
            val = fn(*args)
            cache[key] = val
            return val
    newfn.__name__ = fn.__name__ + ' (THREAD MEMOIZED)'
    newfn.__module__ = fn.__module__
    return newfn

def picklecache(name, incr=0, protocol=-1):
    """Decorator to pickle the function outputs to the given name, as a cache.
    Useful to apply to functions that load in a bunch of data from various inputs.
    If `incr` is true, then name is actually a prefix, and each set of inputs
    is turned into a new file based on the function's arguments.
    The `protocol` is passed to :mod:`pickle` and defaults to highest.

    .. note::
       If `incr` is false, then the arguments are completely ignored!

    .. seealso::
        :func:`incrpicklecache`
            Something similar, except it saves different args to the same pickle
    """
    def actualret(fn):
        def retfunc(*args, **kw):
            if incr:
                key = (tuple(args), tuple(sorted(kw.items())))
                pickname = name+str(key)+'.pickle'
            else:
                pickname = name
            try:
                return pickle.load(open(pickname))
            except Exception:
                ret = fn(*args, **kw)
                # save to temp file and atomically rename
                dirname, basename = os.path.split(pickname)
                try:
                    os.makedirs(dirname)
                except OSError:
                    pass
                f = tempfile.NamedTemporaryFile(prefix='.'+basename, dir=dirname, delete=0)
                pickle.dump(ret, f, protocol)
                tempname = f.name
                f.close()
                os.rename(tempname, pickname)
                # return result
                return ret
        retfunc.__name__ = fn.__name__ + ' (%sPICKLECACHED)' % ('DIR' if incr else '')
        retfunc.__module__ = fn.__module__
        return retfunc
    return actualret

def incrpicklecache(pickname, protocol=-1, interval=-1):
    """Decorator to pickle, incrementally, the function outputs to the given name, as a cache.
    If `interval` > 0, then it will only write to disk if it hasn't written to
    disk in that many seconds.

    .. note::
        This has to load the entire pickle file from disk before returning the
        results, so it can become really slow if the pickle grows large.

    .. seealso::
        :func:`picklecache`
            Something similar, except it saves different args to different pickles.
    """
    def actualret(fn):
        cache = [None]
        lasttime = [time.time()]
        def retfunc(*args, **kw):
            key = (tuple(args), tuple(sorted(kw.items())))
            if cache[0] is None:
                try:
                    cache[0] = pickle.load(open(pickname))
                except Exception:
                    cache[0] = {}
            if key not in cache[0]:
                ret = fn(*args, **kw)
                # reopen cache
                if cache[0]:
                    cache[0] = pickle.load(open(pickname))
                # save to cache
                cache[0][key] = ret
                now = time.time()
                if now - lasttime[0] > interval:
                    # save to temp file and atomically rename
                    f = tempfile.NamedTemporaryFile(prefix='.'+os.path.basename(pickname), dir=os.path.dirname(pickname), delete=0)
                    pickle.dump(cache[0], f, protocol)
                    tempname = f.name
                    f.close()
                    os.rename(tempname, pickname)
                    lasttime[0] = now
            return cache[0][key]
        retfunc.__name__ = fn.__name__ + ' (INCRPICKLECACHED)'
        retfunc.__module__ = fn.__module__
        return retfunc
    return actualret

def timed_cache(seconds: int):
    """Decorator to cache the result of a function for a given number of seconds."""
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(kwargs.items()))  # Create a cache key based on arguments
            current_time = time.time()

            # Check if result is in the cache and if it's still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < seconds:
                    return result

            # Call the function and store the result in the cache
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)  # Store result with the current time
            return result

        return wrapper
    return decorator

def bgrun(nthreads=1, bgthreads=[]):
    """Decorator to run an operation in the background, upto `nthreads` instances.
    Adds as threads as needed to bring the total number of alive threads in `bgthreads`
    upto `nthreads`, and each thread is spawned using the exact same args and kw.

    .. note::
        Because we use an array as a default initializer, you may get unexpected behavior if you're not aware of that.

    .. note::
       This changes the behavior of the function to return the thread, not the original return.
    """
    def actualret(fn):
        def retfunc(*args, **kw):
            # figure out whether to run or not
            bgthreads[:] = [t for t in bgthreads if t and t.is_alive()]
            t = None
            if len(bgthreads) < nthreads: # run
                t = spawnWorkers(1, fn, args=args, kwargs=kw)[0]
                bgthreads.append(t)
            return t
        retfunc.__name__ = fn.__name__ + ' (backgrounded)'
        retfunc.__module__ = fn.__module__
        return retfunc
    return actualret

def genericBind(thefunction, **kwtobind):
    """Decorator to bind some keywords in a function and returns the new bound function"""
    def ret(*args, **kw):
        kw.update(kwtobind)
        return thefunction(*args, **kw)
    return ret

def listifyFunc(func):
    """Decorator to make a function which accepts a list of arguments instead of just one set.
    So if the original function was something like this::
        va = func(a1, a2)
        vb = func(b1, b2)


    Then the new function will work like this::

        lfunc = listify(func)
        vals = lfunc([(a1, a2), (b1, b2)])
        vals[0] == va
        vals[1] == vb

    """
    def ret(args, myfunc=func):
        out = []
        for a in args:
            out.append(myfunc(*a))
        return out
    ret.__name__ = func.__name__ + ' (LISTIFIED'
    return ret

def inSameDir(f):
    """Decorator to make sure a function is executed in the same directory as `__file__`.
    This changes the current working directory using :func:`os.chdir()` and then reverts back afterwards.
    """
    def ret(*args, **kw):
        """Decorated function"""
        oldwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        r = f(*args, **kw)
        os.chdir(oldwd)
        return r
    return ret

def autoRestarter(exceptions, timeout=1.0):
    """A decorator that wraps a function in a giant while 1: try/except loop.
    This basically detects if any of the given exception occurs and restarts the function.
    The `timeout` parameter can be used to control how long we :func:`sleep()` for."""
    exceptions = tuple(exceptions)
    def decorator(f, exceptions=exceptions, timeout=timeout):
        def ret(*args, **kw):
            while 1:
                try:
                    return f(*args, **kw)
                except exceptions as e:
                    log('Restarting function %s in %0.2f secs due to %s: %s' % (f, timeout, type(e), e))
                    time.sleep(timeout)
        return ret
    return decorator

def queueize(inq, outq, func, endfunc=None):
    """Decorator that queueize's the given function.
    Basically reads args from `inq` and calls `outq.put(func(*args))` in an
    infinite loop.

    All exceptions are caught and printed to stderr.
    """
    while 1:
        try:
            #print >>sys.stderr, '    At top of queueize loop, inq size is %d' % (inq.qsize())
            args = inq.get()
            outq.put(func(*args))
            #print >>sys.stderr, '    At bottom of queueize loop'
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(' ** Hit an exception of type %s: %s\n' % (type(e), e))
    #print >>sys.stderr, 'Finished queueize, this is a problem!'
    #sys.stderr.flush()


## TIMING UTILS
def getTimestamp(t=None, fmt='%Y-%m-%d %H:%M:%S'):
    """Returns the timestamp for the given time (defaults to current time).
    The time should be in secs since epoch.

    .. note::
        The default `fmt` includes spaces and colons.
    """
    if not t:
        t = time.time()
    return time.strftime(fmt, time.localtime(t))

def getCleanTimestamp(t=None, fmt='%Y-%m-%d %H:%M:%S', reps=None):
    """Returns a "clean" version of a timestamp, suitable for filenames.
    This uses :func:`strftime()` with the given `fmt` on the given time
    (which defaults to `None` -> current time) to generate the timestamp. Then
    it uses the dict reps to replace strings.  If reps is not given, it defaults
    to::

        ' ' -> '_'
        ':' -> '.'
    """
    s = getTimestamp(t=t, fmt=fmt)
    if not reps:
        reps = {' ': '_', ':': '.'}
    for old, new in reps.iteritems():
        s = s.replace(old, new)
    return s

def getDate(fmt='%Y-%m-%d'):
    """Returns the current date"""
    return getTimestamp(fmt=fmt)

def timeleft(i, num, elapsed):
    """Returns the time left, in secs, of a given operation.
    Useful for loops, where `i` is the current iteration,
    `num` is the total number of iterations, and `elapsed`
    is the time elapsed since starting.
    """
    try:
        rate = i/float(elapsed)
        left = (num-i)/rate
    except ZeroDivisionError: return 0
    return left

def collapseSecs(s, collapse=()):
    """Collapses number of seconds to (years, days, hrs, mins, secs), where all but secs are ints.
    If collapse contains any of 'years', 'days', 'hours', 'mins', then those are collapsed to lower units"""
    y = d = h = m = 0
    # secs per X
    spm = 60
    sph = spm * 60
    spd = sph * 24
    spy = spd * 365
    # compute vals
    while 'years' not in collapse and s >= spy:
        y += 1
        s -= spy
    while 'days' not in collapse and s >= spd:
        d += 1
        s -= spd
    while 'hours' not in collapse and s >= sph:
        h += 1
        s -= sph
    while 'minutes' not in collapse and s >= spm:
        m += 1
        s -= spm
    return (y, d, h, m, s)

def getTimeDiffs(times, timenames=None, fmt='%0.4f', percs=0):
    """Returns a string of time diffs between the given list of times, as well as the total time.
    If a list of `timenames` is given, then those are used as the names for each time diff.
    If `percs` is true, then also shows the percentage of time for each piece.

    Example::

        times = [time.time()]
        foo()
        times.append(time.time())
        bar()
        times.append(time.time())
        baz()
        times.append(time.time())
        print getTimeDiffs(times)
    """
    if len(times) < 2: return ''
    total = times[-1]-times[0]
    if percs:
        pfmt = fmt+' (%0.1f%%)'
        difs = [pfmt % (next-prev, 100.0*(next-prev)/total) for next, prev in zip(times[1:], times)]
    else:
        difs = [fmt % (next-prev) for next, prev in zip(times[1:], times)]
    if timenames:
        difs = ['%s=%s' % (n, d) for n, d in zip(timenames, difs)]
        difs = ', '.join(difs)
    else:
        difs = '+'.join(difs)
    return '%s=%ss' % (difs, fmt % (total))

def getSqlTimeStr(t=None):
    """Returns a sqlite-compatible time string for the given time value
    (in secs since epoch), or :func:`now()` if it's `None` or negative"""
    if not t or t < 0:
        t = time.time()
    d = datetime.fromtimestamp(t).replace(microsecond=0) # since sql cannot handle this
    return d.isoformat(' ')

def iso2secs(t, retdatetime=0):
    """Converts an ISO time (RFC3339_) to seconds since epoch.
    If the given time is not a string, returns it unchanged.
    If `retdatetime` is true, then returns a :class:`datetime.datetime` instance.
    Can deal with only dates, and times without seconds, and fractional seconds.

    .. _RFC3339: http://www.ietf.org/rfc/rfc3339.txt
    """
    from calendar import timegm
    if not isinstance(t, basestring): return t
    fmt = '%Y'
    ndashes = t.count('-')
    if ndashes >= 1:
        fmt += '-%m'
    if ndashes >= 2:
        fmt += '-%d'
    if 'T' in t:
        fmt += 'T'
        if t.find('T') < len(t)-1:
            fmt += '%H'
            ncolons = t.count(':')
            if ncolons >= 1:
                fmt += ':%M'
            if ncolons >= 2:
                fmt += ':%S'
            if '.' in t: fmt += '.%f'
    if t.endswith('Z'): fmt += 'Z'
    ts = datetime.strptime(t, fmt)
    if retdatetime: return ts
    #print 'From %s, got fmt %s and ts %s' % (t, fmt, ts)
    #return time.mktime(ts.timetuple())
    return timegm(ts.timetuple())

def secs2iso(t, fmt='%Y-%m-%dT%H:%M:%SZ'):
    """Converts time as seconds-from-epoch to ISO (RFC3339_) format

    .. _RFC3339: http://www.ietf.org/rfc/rfc3339.txt
    """
    return time.strftime(fmt, time.localtime(t))

def cleanisotime(t, fmt='%a %Y-%m-%d %H:%M (UTC)'):
    """Converts time as seconds-from-epoch or ISO to a clean (user displayable) format"""
    if isinstance(t, (float, int)):
        t = secs2iso(t)
    # also add day-of-week
    ts = iso2secs(t)
    t = time.strftime(fmt, time.localtime(ts))
    #t = t.replace('T', ' ').replace('Z','')
    return t

def flat2iso(s):
    """Converts a flat timestamp, like 20070221033032 to iso time-format.
    The format should be `YYYYMMDDhhmmss`
    """
    ts = '%s-%s-%sT%s:%s:%sZ' % (s[:4], s[4:6], s[6:8], s[8:10], s[10:12], s[12:14])
    return ts

def makesecs(t):
    """Converts a time as a `string` or :class:`datetime` obj to secs, assuming sqlite format:
        2010-01-01 23:37:37
    `None` or floats are sent back unchanged.
    Anything else is sent back as `None`.

    Can also deal with full times with fractional seconds and timezones like this:
        2010-08-18 19:33:41.383751+00:00
    """
    if isinstance(t, datetime):
        return timegm(t.timetuple())
    if not isinstance(t, (str, unicode)): return t
    els = t.split('.', 1)
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        st = time.strptime(els[0], fmt)
    except ValueError as e:
        #logger.info('Time %s (els %s) did not match fmt %s' % (t, els[0], fmt))
        return None
    #ret = time.mktime(st) # localtime
    ret = timegm(st) # localtime
    # now deal with the 'extra' part, which is possibly fractional second and possibly timezone
    if len(els) > 1:
        # normalize spaces, etc.
        extra = els[1].replace(' ', '').replace(':','')
        # split into fractional second and timezone
        if '+' in extra:
            frac, tz = extra.split('+', 1)
            tz = '+'+tz
        elif '-' in extra:
            frac, tz = extra.split('-', 1)
            tz = '-'+tz
        else:
            frac = extra
            tz = ''
        # parse fractional second and add to return value
        try:
            ret += float('.'+frac.strip())
        except ValueError: pass
        # parse timezone and add/subtract to return value
        # we're conservative and don't mess with the return value if there's any problem parsing this.
        if tz and tz[0] in '-+' and len(tz)==5:
            try:
                hours = int(tz[1:3])
                mins = int(tz[3:5])
                secs = 60*secs + 60*60*hours
                if tz[0] == '+': # if a timezone is ahead of utc, we subtract the seconds
                    secs *= -1
                # now add this offset to the return value
                ret += secs
            except Exception: pass
    return ret

def fmttime(t=None, withsecs=1):
    """Formats a time.
    If `t` is the empty string (''), then returns it unchanged"""
    if t == '': return t
    t = makesecs(t)
    fmt = '%a %b %d, %Y, %H:%M'
    if withsecs:
        fmt += ':%S'
    ret = time.strftime(fmt, time.localtime(t))
    return ret

def fmtunits(t):
    """Converts a number of seconds to a string with appropriate units:
        HH:MM:SS
    """
    t = makesecs(t)
    y, d, h, m, s = collapseSecs(t, collapse='years days'.split())
    return '%02d:%02d:%02d' % (h, m, s)

def utcnow():
    """Returns the current time as a :class:`datetime` obj, with ordinary precision, and in GMT"""
    try:
        import pytz # type: ignore
        d = datetime.now(pytz.utc)
    except ImportError:
        d = datetime.utcnow()
    #d.microsecond = 0 #FIXME this is not writable...is it needed?
    return d

def now():
    """Returns the current time as a :class:`datetime` obj, with ordinary precision, in localtime"""
    d = datetime.now()
    #d.microsecond = 0 #FIXME this is not writable...is it needed?
    return d

def getTZLookup(tzfname='cities15000.txt'):
    """Returns a mapping from gps locations to time-zone names.
    The `tzfname` file is read to map gps locations to timezone names.
    This is from: http://download.geonames.org/export/dump/cities15000.zip
    Returns a list of `((lat, lon), timezone)` pairs.
    """
    ret = [l.rstrip('\n').split('\t') for l in open(tzfname) if l.strip()]
    ret = [((float(l[4]), float(l[5])), l[17]) for l in ret]
    return ret

def localizeTime(t, loc, tzlookup=None):
    """Localizes time using gps info.
    The given time should be utc time, either in secs, or as a :class:`datetime` object.
    The loc should be a `(latitude, longitude)` pair, in decimal degrees.
    The tzlookup should be a list of `((lat, lon), timezone)` pairs.
    If not given, it's looked up from :func:`getTZLookupDict()`.

    The :mod:`pytz` module is used to map time using the computed timezone.
    Returns a localized :class:`datetime` object.

    If `loc` is not given or is invalid, returns an un-normalized :class:`datetime` object.
    """
    import pytz
    from nkpylib.geo import haversine_dist
    # convert to datetime
    if not isinstance(t, datetime):
        t = datetime.fromtimestamp(t)
    # check for invalid
    if not loc or len(loc) != 2 or None in loc: return t
    if not (-90 <= loc[0] <= 90) or not (-180 <= loc[1] <= 180): return t
    # get the lookup
    if not tzlookup:
        tzlookup = getTZLookup()
    # compute dists and find closest point
    dists = [(haversine_dist(loc, l), l, tz) for l, tz in tzlookup]
    dists.sort(key=lambda pair: pair[0])
    # get the right timezone
    tzname = dists[0][-1]
    try:
        tz = pytz.timezone(tzname)
        # normalize
        ret = tz.fromutc(t)
    except Exception:
        ret = t
    #print dists[:5], tzname, tz, t, ret
    return ret

def time2names(t, thresh=0, gps=None):
    """Converts a float time into a set of names.
    This includes year, month, day of week, date, as well as holidays.

    If thresh is greater than 0, then also includes a holiday if the date
    is within 'thresh' days of the holiday.

    If gps is true, then first converts time to localtime. Also adds time-of-day info.

    Returns a list of (tag, tagtype) pairs, where tag is the string tag, and
    tagtype is the type of tag (not-necessarily unique):
        year
        month
        date
        day
        holiday
        daytime (usually one of: morning, afternoon, evening, night)
        isweekend (usually one of: weekday, weekend)
    """
    # normalize
    if not isinstance(t, datetime) and t > 10000000000:
        t /= 1000.0
    d = localizeTime(t, gps)
    #print d
    # get basic date info
    ret = []
    #ret.append((str(d.year), 'year'))
    #fmtstrs = [('%Y', 'year'), ('%b',''), ('%B',''), ('%b %Y','month'), ('%B %Y', 'month'), ('%b %d %Y', 'day'), ('%B %d %Y', 'day')]
    # no months with years at end
    #fmtstrs = [('%Y', 'year'), ('%b',''), ('%B',''), ('%b %d', 'day'), ('%B %d', 'day')]
    # no short month names
    fmtstrs = [('%Y', 'year'), ('%B', 'month'), ('%B %d', 'date'), ('%A', 'day')]
    for f, name in fmtstrs:
        ret.append((d.strftime(f), name))
        #print f, name, ret[-1]
    day = d.strftime('%A').lower()
    if day in 'saturday sunday'.split():
        ret.append(('weekend', 'isweekend'))
    else:
        ret.append(('weekday', 'isweekend'))
    # add time of day
    daytime = ''
    if 6 <= d.hour <= 12:
        daytime = 'morning'
    elif 12 <= d.hour <= 17:
        daytime = 'afternoon'
    elif 17 <= d.hour <= 21:
        daytime = 'evening'
    else:
        daytime = 'night'
    ret.append((daytime, 'daytime'))
    # add holidays
    holidays = json.load(open('holidays.json'))['holidays']
    for hname, (m, day, range) in holidays.items():
        hdate = date(d.year, m, day)
        dif = hdate - d.date()
        dist = (thresh+range) - abs(dif.days)
        if dist >= 0:
            ret.append((hname, 'holiday'))
    return ret


## MEMORY UTILS
#: mapping strings to multipliers on bytes
MEMORY_UNITS = {'B': 1, 'kB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}

def memstr2bytes(s):
    """Converts a memory string like '1249 MB' to number of bytes.
    If it can't be converted, raises a :class:`ValueError`."""
    try:
        g = re.search(r'(\d+)\s*(\S+)', s).groups()
        num, units = int(g[0]), g[1]
        ret = num * MEMORY_UNITS[units]
        return ret
    except Exception:
        raise ValueError("Can't convert %s to bytes" % (s))

def getmem(obj):
    """Returns some very rough estimate of the memory usage (in bytes) of the given object.
    Works with ints/floats/strings and lists/tuples/dicts of the previous.
    Does not count pointer memory (which can be substantial with nested
    lists/tuples/dictionaries)!"""
    INT_MEM = 4
    FLOAT_MEM = 4
    CHAR_MEM = 1
    #log('Obj %s has type %s' % (repr(obj), type(obj)))
    if type(obj) == type(123): return INT_MEM
    elif type(obj) == type(1.23): return FLOAT_MEM
    elif isinstance(obj, str): return len(obj)*CHAR_MEM
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sum((getmem(o) for o in obj))
    elif isinstance(obj, dict):
        return sum((getmem(k)+getmem(v) for k, v in obj.items()))
    return 0

def dictMemUsage(d):
    """Returns a dictionary with various memusage stats for a dict.
    Works for simple keys (ints/floats/strings) and simple values (ints/floats/strings)
    or lists of simple values. These stats include:
        * `nkeys`: number of keys in `d`
        * `keymem`: the memory usage of all keys in `d`
        * `valmem`: the memory usage of all values in `d`
        * `totalmem`: the sum of the above
        * `nvals`: if the values of `d` are lists, then their total length, else just `nkeys`
    """
    ret = {'nkeys': len(d)}
    ret['keymem'] = sum((getmem(k) for k in d))
    ret['valmem'] = sum((getmem(v) for v in d.itervalues()))
    ret['totalmem'] = ret['keymem'] + ret['valmem']
    try:
        ret['nvals'] = sum((len(v) for v in d.itervalues()))
    except TypeError: ret['nvals'] = len(d)
    return ret

def procdict(fname):
    """Returns a dictionary of key-values from the given file.
    These are in `/proc` format::
        key:[\t]*value
    """
    d = dict(l.strip().split(':', 1) for l in open(fname))
    for k in d:
        d[k] = d[k].strip()
    return d

def procmem(include_children=0):
    """Returns memory usage for the running process, in bytes.
    If `include_children` is true, then adds up the memory usage of our children as well.
    If the `/proc` filesystem is not available, raises a :class:`NotImplementedError`"""
    try:
        mem = procdict('/proc/%d/status' % os.getpid())['VmRSS']
        return memstr2bytes(mem)
    except Exception:
        raise NotImplementedError

def totalmem():
    """Returns the total number of bytes of memory available on this machine.
    If the `/proc` filesystem is not available, raises a :class:`NotImplementedError`"""
    try:
        mem = procdict('/proc/meminfo')['MemTotal']
        return memstr2bytes(mem)
    except Exception:
        raise NotImplementedError

class MemUsage(object):
    """A simple memory usage profiler for use over a program.
    On initialization, it stores the current memory use.
    You can then call :func:`add()` to add a checkpoint, optionally with
    a name, and any kwargs you want. These are stored in a list,
    which you can iter over as usual.

    There are also various convenience functions.
    """
    def __init__(self):
        """Creates the object and stores current memory usage"""
        self.data = []
        self.add('start')

    def add(self, name='', **kwargs):
        """Adds a checkpoint with the given `name`, and any `kwargs`"""
        d = {'name': name, 'mem': procmem(), 'time': time.time()}
        d.update(kwargs)
        self.data.append(d)

    def usage(self, key=None):
        """Memory usage for the given `key`, in bytes. See :func:`__getitem__` for details on keys"""
        d = self[key]
        return d['mem']

    def delta(self, key=None):
        """Delta in bytes to given `key`. See :func:`__getitem__` for details on keys"""
        d = self[key]
        i = self.data.index(d)
        if i == 0: return d['mem'] # first point is from 0
        return d['mem'] - self.data[i-1]['mem']

    def deltas(self):
        """Returns all deltas as a list of bytes"""
        all = list(self)
        ret = [m1['mem']-m0['mem'] for m0, m1 in zip(all, all[1:])]
        return ret

    def vals(self, fields, diffs=0):
        """Returns a list of our values, for the given `fields`.
        If `fields` is a :class:`string`, then simply returns a list of that field.
        If `fields` is a :class:`seq`, then returns a list of tuples.
        Fields are:
            * `name`
            * `index`
            * `mem`
            * `time`
        If `diffs` is true, then computes diffs between fields instead.
        """
        def sub(a, b):
            """subtract which returns 2nd field if there's an error (e.g., strings)"""
            try:
                return a-b
            except Exception:
                return b

        # check for single field
        if isinstance(fields, basestring):
            ret = [cur[fields] for cur in self]
            if diffs:
                ret = [sub(r1,r0) for r0, r1 in zip(ret, ret[1:])]
            return ret
        # else assume it's a list
        ret = [getDictValues(cur, fields) for cur in self]
        if diffs:
            trans = zip(*ret)
            trans = [[sub(r1, r0) for r0, r1 in zip(series, series[1:])] for series in trans]
            ret = zip(*trans)
        return ret

    def namedtimes(self, **kw):
        """Returns a string with named time deltas.
        All kwargs are passed to :func:`getTimeDiffs()`"""
        names, times = zip(*self.vals('name time'.split()))
        if 'percs' not in kw:
            kw['percs'] = 1
        return getTimeDiffs(times, timenames=names[1:], **kw)

    def __len__(self):
        """Returns the number of checkpoints we have"""
        return len(self.data)

    def __iter__(self):
        """Iterates through our list of checkpoints."""
        return iter(self.data)

    def __getitem__(self, key):
        """Returns the last item matching the given `key`.
        The `key` can be a named string, or the given index if integer, or last item if :class:`None`"""
        # last item if None
        if key is None: return self.data[-1]
        # index if integral
        try:
            return self.data[key]
        except (IndexError, TypeError):
            pass
        # otherwise name
        for d in reversed(self.data):
            if d['name'] == key:
                return d
        # keyerror
        raise KeyError("'%s' not found in items" % (key,))


## LOGGING UTILS
def log(s, f=sys.stderr, funcindent=-1):
    """Logs the given string to the given file (:class:`sys.stderr` by default).
    Unless the string contains '\r', an endline is printed.
    if `funcindent` is >= 0, then indents the line with spaces according to the
    function depth, subtracting `funcindent` from the stack length to get the
    number of spaces to use in the indentation."""
    if not s:
        s = ' '
    if not isinstance(s, basestring):
        s = str(s)
    if isinstance(s, unicode):
        s = s.encode('utf-8', 'ignore')
    while s[0] == '\n':
        f.write('\n')
        s = s[1:]
    if funcindent >= 0:
        s = '  ' * max(len(inspect.stack())-funcindent, 0) + s
    if '\r' in s:
        f.write(s)
    else:
        f.write(s+'\n')
    f.flush()

def makeProgress(out=sys.stderr, length=None):
    """Makes a progress function, with the given line-width (or unlimited if `None`).
    This function lets you print a string, and '\r' is added at the end to
    prevent a newline. However, the function keeps track of the number of
    characters printed last time, so that when you call it repeatedly with
    strings of different lengths, it appropriately pads with spaces to prevent
    residual characters from staying on screen.

    This function prints to the given `out` file (:class:`sys.stderr` by default).

    The function takes params:
        * `msg`: The message to print
        * `i`: Ignored for now
        * `total`: Ignored for now
    """
    last = [0]
    def progress(msg, i=-1, total=-1, last=last, out=out, length=length):
        """Prints a message with progress"""
        # first print blanks for each character in the previous
        blanks = ' ' * (last[0]+5)
        out.write('\r%s' % (blanks))
        # now print the message
        # TODO deal with i/total somehow
        out.write('\r%s' % (msg))
        out.flush()
        # now save the length
        last[0] = len(msg)
    return progress

class repl(Thread):
    """A simple way to add a REPL to a program.
    Just create a new repl, and do :func:`repl.start()`, before you launch your program."""
    def __init__(self, locals={}, *args, **kw):
        Thread.__init__(self, *args, **kw)
        self.kw = {}
        self.kw.update(locals)

    def run(self):
        """Starts the repl, with readline and rlcompleter activated.
        Because this class inherits from :class:`Thread`, you should call
        :func:`start()` instead of :func:`run()`
        """
        readline.parse_and_bind('tab: complete')
        readline.parse_and_bind('"\e[A": history-search-backward')
        readline.parse_and_bind('"\e[B": history-search-forward')
        readline.parse_and_bind('set completion-ignore-case on')
        readline.parse_and_bind('set show-all-if-ambiguous on')
        self.kw.update(locals())
        self.kw.update(globals())
        code.InteractiveConsole(self.kw).interact()

def spark(vals, wrap=0, scale=None, f=sys.stdout):
    """Prints a spark graph of the given values to the given output stream.
    If you provide a `wrap` value > 0, then groups inputs into that length.
    If you provide a `scale`, then multiplies all values by that scale.
    Note that the `spark` executable can't handle float values,
    so if your values are e.g. between 0-1, then you will want to set the scale.
    The default output stream is :class:`sys.stdout`

    Right now, this needs a `spark` executable to run through :func:`subprocess.Popen`
    """
    from subprocess import PIPE, Popen
    from io import StringIO
    if wrap > 0:
        groups = nkgrouper(wrap, vals)
    else:
        groups = [vals]
    for g in groups:
        args = ['spark']
        if scale:
            g = [v*scale for v in g]
        args.extend(map(str, g))
        if isinstance(f, StringIO):
            sout, serr = Popen(args, stdout=PIPE).communicate()
            f.write(sout+'\n')
        else:
            Popen(args, stdout=f).communicate()


## ITERTOOLS AND SEQUENCES UTILS
def arange(from_, to, step):
    """Returns samples generated in the range ``[from_, to]``, with given `step` size.
    If `step` is 0, then just returns ``[from_]``
    """
    if step == 0: return [from_]
    nsteps = int((to-from_)/float(step)) + 1
    ret = [from_ + i*step for i in xrange(nsteps)]
    return ret

def grange(from_, to, nsteps):
    """Gets the geometric range (exponential) in the range ``[from_, to]``, using
    the given number of `nsteps`.

    .. note::
        The parameter `nsteps` here is the number of steps, as opposed to in
        :func:`arange`, where it's the *size* of each `step`.
    """
    from math import log, exp
    base = exp(log(to/from_)/(nsteps-1))
    x1 = log(from_)/log(base)
    x2 = log(to)/log(base)
    xvals = arange(x1, x2, (x2-x1)/(nsteps-1))
    ret = [base**x for x in xvals]
    return ret

def grouper(n, iterable, padvalue=None):
    """Groups items from an iterable into tuples of size `n`, with padding.
    Taken from the :mod:`itertools` recipes. Example::
        >>> grouper(3, 'abcdefg', 'x')
        ('a','b','c'), ('d','e','f'), ('g','x','x')
    """
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

def nkgrouper(n, iterable):
    """Like :func:`grouper()`, but without padding"""
    UNIQ = 'ads0f9jasd09fj0sjff09d8jfa8sjcc38j' #FIXME hacky!
    groups = grouper(n, iterable, padvalue=UNIQ)
    for g in groups:
        els = [e for e in g if e != UNIQ]
        yield els

def cumsum(seq, start=0):
    """Returns the cumulative sum of the given elements.
    Uses ``+=`` to add, so you can even use lists.
    Adds to the given `start` value."""
    ret = []
    for n in seq:
        start += n
        ret.append(start)
    return ret

def uniqueize(lst, hashable=1):
    """Makes a list unique, maintaining order.
    If `hashable` is true (default), then assumes things are hashable and uses a
    set. This makes the algorithm run in linear time.
    Otherwise uses a list, which makes the algorithm O(N^2).
    """
    t = type(lst)
    ret = []
    done = set() if hashable else []
    for x in lst:
        if x in done: continue
        if hashable:
            done.add(x)
        else:
            done.append(x)
        ret.append(x)
    return t(ret)

def argsort(seq, key=None, cmp=None, reverse=False):
    """Returns the indices corresponding to a sort of the given `seq`.
    Can optionally pass in `cmp` and `reverse` just as you would to :func:`sorted()`.
    """
    if not seq: return ()
    ukey = key
    iseq = sorted([(v, i) for i, v in enumerate(seq)], key=lambda v, i: ukey(v), cmp=cmp, reverse=reverse)
    vals, idxs = zip(*iseq)
    return idxs

def median(seq):
    """Returns the median of a sequence.
    Note that if the list is even-length, then just returns the item to the left
    of the median, not the average of the median elements, as is strictly correct.
    """
    seq = sorted(seq)
    return seq[len(seq)//2]

def lists2dict(keys, vals):
    """Creates a dictionary from `keys` and `vals`, creating lists for each key,
    and appending values to those lists. This is useful if you have many values
    per key and need to convert to a dict."""
    ret = {}
    for k, v in zip(keys, vals):
        ret.setdefault(k, []).append(v)
    return ret

def getMatchingIndices(func, seq):
    """Returns indices of a sequence where `func` evaluated to True."""
    return [i for i, v in enumerate(seq) if func(v)]

def pruneListByIndices(lst, indices):
    """Prunes a `lst` to only keep elements at the given `indices`."""
    return [l for i, l in enumerate(lst) if i in indices]

def flatten(x):
    """Returns a single, flat list which contains all elements retrieved from
    the sequence and all recursively contained sub-sequences (iterables).

    Examples::
        >>> [1, 2, [3,4], (5,6)]
        [1, 2, [3, 4], (5, 6)]
        >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, (8,9,10)])
        [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]

    Uses the :data:`__iter__` attribute to check for whether it's a list.
    """
    ret = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            ret.extend(flatten(el))
        else:
            ret.append(el)
    return ret

def xcombine(*seqin):
    """Returns a generator which returns combinations of argument sequences.
    For example, ``xcombine((1,2),(3,4))`` returns a generator; calling the
    `next()` method on the generator will return (sequentially):
    ``[1,3], [1,4], [2,3], [2,4]`` and then a :class:`StopIteration` exception.

    This will not create the whole list of combinations in memory at once.
    """
    def rloop(seqin,comb):
        """recursive looping function"""
        if seqin:                  # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]  # add next item to current combination
                # call rloop wth remaining seqs, newcomb
                for item in rloop(seqin[1:],newcomb):
                    yield item        # seqs and newcomb
        else:                          # processing last sequence
            yield comb                # comb finished, add to list
    return rloop(seqin,[])

def genPowerSet(seq):
    """Returns the powerset of a sequence (i.e. all combinations)."""
    # by Tim Peters
    pairs = [(2**i, x) for i, x in enumerate(seq)]
    for i in xrange(2**len(pairs)):
        yield [x for (mask, x) in pairs if i & mask]

def lazy(s):
    """A way to lazy evaluate a string in the parent frame.
    From Guido"""
    f = sys._getframe(1)
    return eval(s, f.f_globals, f.f_locals)

def partitionByFunc(origseq, partfunc):
    """Partitions a sequence into a number of sequences, based on the `partfunc`.
    Returns ``(allseqs, indices)``, where:
        - `allseqs` is a dictionary of output sequences, based on output values
          of `partfunc(el)`.
        - `indices` is a dictionary of ``(outval, i) -> orig_i``, which allows mapping results back.
    So if your `partfunc` returns 'foo' and 'bar', `allseqs` will have ``{'foo': [...], 'bar': [...]}``.
    You access `indices` using ``(partout, seq_i)``, where `partout` is 'foo' or 'bar' in this case, and
    `seq_i` is the index number from the ``allseqs[partout]`` sequence.

    This function is very useful for categorizing a list's entries based on some
    function. If your function was binary, you would normally do it using 2 list
    comprehensions::
        a = [el for el in seq if partfunc(el)]
        b = [el for el in seq if not partfunc(el)]

    But that quickly gets inefficient and bloated if you have more complicated
    partition functions, which is where this function becomes useful.
    """
    allseqs = {}
    indices = {}
    for i, el in enumerate(origseq):
        partout = partfunc(el)
        seq = allseqs.setdefault(partout, [])
        indices[(partout, len(seq))] = i
        seq.append(el)
    return allseqs, indices

def getFirstValid(opts, default):
    """Returns the first valid entry from `opts`, or `default` if none found.
    Valid is defined as ``if o`` returns true."""
    for o in opts:
        if o: return o
    return default

def linearweighting(vals, valfunc, start, end, ndivs=100, smoothing=1.0):
    """Returns linearly weighted values, within the given interval.
    This is useful for converting arbitrarily sampled values (with timestamps)
    into a regularly-sampled one, e.g., for plotting.
    Parameters:
        - `vals` should contain ``(value, time)`` tuples.
        - `valfunc` should take the list of `vals` and a period and return a single value.
        - `start` is the first time at which you want a value.
        - `end` is the last time at which you want a value.
        - `ndivs` is the number of equal-sized divisions between `start` and `end` (default 100).
        - `smoothing` is how many divisions back and forward to average (default 1).
    The result is a dictionary of ``{time: val}`` of length `ndivs+1` (to include both boundaries).
    """
    from bisect import bisect_left, bisect_right
    if not vals: return {}
    start = makesecs(start)
    end = makesecs(end)
    incr = (end-start)/ndivs
    ret = {}
    cur = start
    junk, times = zip(*vals)
    for i in range(ndivs+1):
        i1 = bisect_left(times, cur-(smoothing*incr))
        i2 = bisect_right(times, cur+(smoothing*incr))
        #curvals = [(i, v,t) for i, (v, t) in enumerate(vals) if cur-(smoothing*incr) <= t <= cur+(smoothing*incr)]
        #curvals = [(v,t) for v, t in vals if cur-(smoothing*incr) <= t <= cur+(smoothing*incr)]
        #log('Got t %s, T %s, i1 %s, i2 %s, %d times, curvals from %s to %s' % (mint, maxt, i1, i2, len(times), curvals[0][0], curvals[-1][0]))
        curvals = vals[i1:i2]
        v = valfunc(curvals, 2*smoothing*incr)
        ret[cur] = v
        cur += incr
    return ret

def exponentialweighting(vals, valfunc, period, ndivs=5, curtime=None):
    """Returns an exponentially weighted value, upto the current time.
    This is useful for figuring out current rates based on historical values,
    giving greater weight to more recent values, even when your inputs are not
    necessarily evenly sampled. Parameters:
        - `vals` should contain ``(value, time)`` tuples.
        - `valfunc` should take a list of vals and a period and return a single value.
        - `period` is the initial total period.
        - `ndivs` is the number of times the period gets halved.
    The way this works is to take an average over the full `period`, then an
    average of half the period (around the `curtime`, which defaults to now),
    then half that, and so on, `ndivs` times.

    The result is a single average of all of these vals (a float).
    """
    from bisect import bisect_left, bisect_right
    if not vals: return 0.0
    if not curtime:
        curtime = time.time()
    curtime = makesecs(curtime)
    start = curtime-period
    interval = period
    ret = []
    # compute values over progressively smaller intervals and then average them all
    # this applies an exponential weighting, emphasizing the most recent values
    junk, times = zip(*vals)
    for i in range(ndivs):
        i1 = bisect_left(times, start)
        i2 = bisect_right(times, curtime)
        #curvals = [(v,t) for v, t in vals if start <= t <= curtime]
        curvals = vals[i1:i2]
        ret.append(valfunc(curvals, interval))
        #print '  %s' % ((period, interval, start, len(vals), len(curvals), ret[-1]),)
        interval /= 2.0
        start = curtime-interval
    ret = sum(ret)/len(ret)
    return ret

def makeWindowingFunc(name, incr):
    """Makes a windowing function of the given type and increment.
    The windowing function takes a single parameter `tdif`, which is the
    difference between two values. This gets divided by the `incr` to get a
    percentage `p`, which gets fed into the windowing function to get a final
    output value.

    Types of windowing functions:
        - `linear`: returns `1.0 - p`
        - `constant`: returns `1.0`
        - `exp`: returns ``exp(1-p)-exp(0)``
        - `sqrt`: returns ``sqrt(1-p)``

    Typical usages are for figuring out how much to weight a particular sample
    given its distance (often in time) from a "known" value.
    """
    funcmap = dict(linear=lambda p: 1.0-p, constant=lambda p: 1.0, exp=lambda p: math.exp(1-p)-math.exp(0), sqrt=lambda p: math.sqrt(1-p))
    func = funcmap[name]
    def winfunc(tdif, incr=incr, func=func):
        """Takes the given time difference and returns a float weight."""
        tdif = abs(tdif)
        if tdif > incr: return 0.0
        perc = tdif/float(incr)
        return func(perc)

    return winfunc

def fieldize(f):
    """Makes a function that takes a list of fields and runs the given func
    with those fields extracted from the underlying objects.
    """
    def retfunc(*fields):
        def sub(a):
            """Substitutes the given arg using the list of fields"""
            ret = [a[field] for field in fields]
            ret = flatten(ret)
            return ret

        def ret(*args, **kw):
            newargs = [sub(a) for a in args]
            return f(*newargs, **kw)
        return ret

    retfunc.__name__ == f.__name__
    retfunc.__doc__ == f.__doc__
    return retfunc

def roundrobin(seqOfSeqs, num, dopad=0, pad=None):
    """Selects `num` elements in round-robin fashion from the given sequence-of-sequences.
    If there are less total elements than `num`, then:
        - If `dopad` is 0 (default): does nothing (returned list has ``len() < num``)
        - If `dopad` is 1: uses `pad` (default=None) to pad list
    """
    ret = []
    if not seqOfSeqs: return ret
    cur = [0 for s in seqOfSeqs]
    while len(ret) < num and max(cur) >= 0:
        for i, seq in enumerate(seqOfSeqs):
            if cur[i] < 0: continue
            try:
                ret.append(seq[cur[i]])
                if len(ret) == num: break
                cur[i] += 1
            except IndexError:
                cur[i] = -1
    # if we don't have enough items and we want to pad, do it
    if len(ret) < num and dopad:
        ret.extend([pad] * range(num-len(ret)))
    return ret


## DICT UTILS
def specialize(v):
    """Specializes a value into a more specific type, if possible."""
    constants = dict(true=True, false=False, none=None)
    if isinstance(v, str):
        if v.lower() in constants:
            return constants[v.lower()]
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass
    return v

def get_nested(field, obj):
    """Accesses a nested field value in given dict or list.

    E.g. get_nested('a.0.3', obj) -> obj['a'][0][3]
    """
    fields = field.split('.')
    for f in fields:
        try:
            obj = obj[int(f)]
        except:
            pass
        obj = obj[f]
    return obj

def getDictValues(d, fields, defaults=None):
    """Returns a list of values from a dictionary using the given seq of `fields`.
    If `defaults` is given, it should be same length as `fields`."""
    if defaults is None:
        return [d[f] for f in fields]
    assert len(defaults) == len(fields)
    return [d.get(f, de) for f, de in zip(fields, defaults)]

def whitelist(d, fields):
    """Whitelists a dictionary by keeping ONLY the selected `fields`.
    Non-destructive (creates and returns a new dict)."""
    ret = type(d)()
    for f in fields:
        if f in d:
            ret[f] = d[f]
    return ret

def blacklist(d, fields):
    """Blacklists a dictionary by keeping all EXCEPT the selected `fields`.
    Non-destructive (creates and returns a new dict)."""
    ret = type(d)()
    fields = set(fields)
    for k, v in d.iteritems():
        if k not in fields:
            ret[k] = v
    return ret

def kvdict2str(d, sep='@', dlm='::'):
    """Turns a key-value dict into a string.
    Keys and values are separated using `sep` [default '@'].
    Each key-value pair is delimited using `dlm` [default '::'].
    """
    ret = '::'.join('%s@%s' % (k, v) for k, v in d.iteritems())
    return ret

def str2kvdict(s, sep='@', dlm='::'):
    """Returns a key-value dict from the given string.
    Keys and values are assumed to be separated using `sep` [default '@'].
    Each key-value pair is delimited using `dlm` [default '::'].

    .. warning::
        Silently skips any elements that don't have the separator in them or are blank.
    """
    ret = dict([pair.split(sep,1) for pair in s.split(dlm) if pair and sep in pair])
    return ret

def renamefields(d, *args):
    """Renames fields in a dict (IN-PLACE), using the ``(from, to)`` tuples provided in `args`.

    .. warning::
        If one of the `from` fields is also a `to` field, then behavior might be undefined.
    """
    for from_, to in args:
        d[to] = d[from_]
        del d[from_]
    return d

def makedefdict(d, default_factory=int):
    """Converts a normal dictionary into a :class:`defaultdict`, using the given `default_factory`"""
    from collections import defaultdict
    ret = defaultdict(default_factory)
    ret.update(d)
    return ret

class HashDict(dict):
    """A simple extension to a dictionary that can be hashed (i.e., used as a key).
    This uses a sorted tuple on ``self.items()`` as the hasher.

    .. warning::
        This is very brittle and so should only be used when you are sure it will be safe.
    """
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class LazyDict(dict):
    """A dictionary which calls a given function to initialize values when a key doesn't exist.
    It's sort-of a generalization of :class:`defaultdict`, except instead of the defaultfunc's
    initializer taking no arguments, it takes the key itself as an arg to initialize."""
    def __init__(self, func, *args, **kw):
        """Initializes this lazy dict with the given function"""
        dict.__init__(self, *args, **kw)
        self.func = func

    def __getitem__(self, key):
        """If we have it, then just return it. Otherwise run the function"""
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            #print 'Loading dict because of key %s' % (key)
            ret = self[key] = self.func(key)
            #print '  Dict is: %s' % (self.items())
            return ret

def mapping2dict(d):
    """Converts a mapping (any subclass of `dict`) back to a real dict.
    Return non-mappings as-is."""
    if not isinstance(d, dict): return d
    ret = dict(d)
    for k, v in ret.iteritems():
        if isinstance(v, dict):
            d[k] = mapping2dict(v)
    return ret

def dictequal(a, b):
    """Compares two mappings to see if they're equal.

    .. note::
        This maps both to real dicts using :func:`mapping2dict()` and then runs ``==`` to compare them.
    """
    a = mapping2dict(a)
    b = mapping2dict(b)
    return a == b

def summarizekeys(d, counts={}, base=''):
    """Summarizes keys in the given dict, recursively.
    This means counting how many fields exist at each level.
    Returns keys of the form ``key0.key1`` and values of ints.
    Checks if `d` is instance of dict before doing anything.
    """
    if not isinstance(d, dict): return counts
    for k, v in d.items():
        k = '.'.join((base, k)) if base else k
        if k not in counts:
            counts[k] = 0
        counts[k] += 1
        summarizekeys(v, counts=counts, base=k)
    return counts


## MATH UTILS
def clamp(val, minval, maxval):
    """Clamps the given value to lie between the given `minval` and `maxval`"""
    return min(max(val, minval), maxval)

def triclamp(val, thresh=0.8):
    """Trinary clamps a value around 0.
    The return value is::

        (-inf, -thresh) -> -1.0
        [-thresh, thresh] -> 0.0
        (thresh, inf) -> 1.0
    """
    if val < -thresh: return -1.0
    elif val > thresh: return 1.0
    return 0.0

def remap(x, min=-pi, max=pi):
    """Remaps a value from the range ``[min, max]`` to the range ``[0, 255]``"""
    assert max > min
    return int((x-min)*255/(max-min))

def getMean(seq):
    """Returns the mean of the given sequence"""
    return sum(seq)/float(len(seq))

def getVariance(seq, mean=None):
    """Returns the variance of the given sequence.
    If `mean` is `None` (default), then it is computed"""
    if mean is None: mean = getMean(seq)
    var = sum([(x-mean)**2 for x in seq])/float(len(seq))
    return var

def getStdDev(seq, mean=None):
    """Returns the standard deviation of the given sequence.
    If `mean` is `None`, then it is computed"""
    from math import sqrt
    return sqrt(getVariance(seq, mean))

def linscale(seq, minval=0.0, maxval=1.0):
    """Linearly scales all the values in the sequence to lie between the given values.
    Shifts up to minval and scales by the difference ``maxval-minval``
    If all values are identical, then sets them to `minval`."""
    m, M = min(seq), max(seq)
    def sc(s, m=m, M=M):
        if m==M: return minval
        return minval + ((s-m)/float(M-m))*(maxval-minval)
    seq = [sc(s) for s in seq]
    return seq

def lerp(x, from_, to):
    """Linear interpolates a value using the `x` given and ``(x,y)`` pairs `from_` and `to`.
    All x values must be numbers (have `-` and `/` defined).
    The y values can either be single numbers, or sequences of the same length.
    If the latter case, then each dimension is interpolated linearly and the
    output is a sequence of the same type."""
    x0, x1, y0, y1 = from_[0], to[0], from_[1], to[1]
    if x0 == x1: return y0 # degenerate case
    perc = (x-x0)/float(x1-x0)
    # see if they're sequences
    try:
        y = [(t-f)*perc + f for f, t in zip(y0, y1)]
        # cast the output to the type of the input
        return type(y0)(y)
    except TypeError:
        y = (to[1]-from_[1])*perc + from_[1]
        return y

def linearsample(val, seq):
    """Samples a discrete sequence continuously using linear interpolation.
    `seq` should contain pairs of numbers, sorted by first element (`x`).
    The `val` at which to sample is simply an x-value."""
    from bisect import bisect
    xs, ys = zip(*seq)
    i = bisect(xs, val)
    #print 'Got xs %s, ys %s, val %s, i %s' % (xs, ys, val, i)
    # edge cases -- return the edge val
    if i == 0: return seq[0][1]
    if i == len(seq): return seq[-1][1]
    # middle cases -- lerp
    return lerp(val, seq[i-1], seq[i])

def gaussiankernel(variance, width=None):
    """Creates a 1-d gaussian kernel of given `variance` and `width`.
    If no `width` is given, then ``6*variance+1`` is used"""
    if width is None:
        width = 6*int(variance + 0.99) + 1
    ret = [gaussian(x-width//2, 0, variance) for x in range(width)]
    s = sum(ret)
    ret = [r/s for r in ret]
    #print 'For v=%s and w=%s, returning %s with sum %s' % (variance, width, ret, sum(ret))
    return ret

def smoothpoints(pts, kernel=None):
    """Smooths the given set of 1-dimensional points using the given `kernel`.
    If no kernel is given, then a gaussian kernel of variance 1 is used"""
    import numpy
    if kernel is None or len(kernel) == 0: kernel = gaussiankernel(1.0)
    w = len(kernel)
    data = numpy.array([pts[0]]*w + list(pts) + [pts[-1]]*w)
    kernel = numpy.array(kernel)
    assert kernel.ndim == 1 == data.ndim
    out = numpy.convolve(data, kernel, 'same')
    out = out[w:w+len(pts)]
    #print 'got points %s and returning %s' % (pts, out)
    return out

def getCenter(coords):
    """Returns the center of the given set of 2d coords as a 2-ple."""
    xs = [c for i, c in enumerate(coords) if i % 2 == 0]
    ys = [c for i, c in enumerate(coords) if i % 2 == 1]
    cx = sum(xs)/float(len(xs))
    cy = sum(ys)/float(len(ys))
    return (cx, cy)

def approx(f, eps=0.001):
    """Returns an "approximate" value of `f`, within `eps`.
    This is useful if you want to quantize floats into bins.
    """
    return int(f/eps + 0.5) * eps

def sigmoid(x):
    """Returns the value mapped using a sigmoid.
    This is the logistic function: ``1/(1+exp(-x))``
    The input can be any number.
    Results are in the range 0 to 1, with ``x=0 -> y=0.5``
    """
    return 1.0/(1+exp(-x))

def lpdist(x, y, p=2):
    """Returns the :math:`L_p` distance between the two vectors.
    Works for p=0, 1, 2 or any higher number (but not infinity).

    .. seealso::
        :func:`linfdist`
            The function for computing :math:`L_\infty` distances.
    """
    if p == 0:
        return sum(a!=b for a, b in zip(x, y))
    elif p == 1:
        return sum(abs(a-b) for a, b in zip(x, y))
    elif p == 2:
        return sqrt(sum((a-b)**p for a, b in zip(x,y)))
    else:
        return sum((a-b)**p for a, b in zip(x, y)) ** 1/float(p)

def l2dist(a, b):
    """Shortcut for ``lpdist(a, b, 2)``"""
    return lpdist(a, b, 2)

def linfdist(a, b):
    """Returns the :math:`L_\infty` distance between the two seqs.
    This is the sum of ``abs(i-j)`` for each element.
    """
    return max((abs(i-j) for i, j in zip(a,b)))

def intersectiondist(x,y):
    """Returns the histogram intersection distance between two vectors.
    This is the sum of ``min(a,b)`` for each element.
    This is usually the most effective distance measure when comparing histograms,
    such as SIFT vectors in computer vision.
    """
    return sum(min(a,b) for a,b in zip(x,y))

def normalize(seq, val=1.0, power=2.0, epsilon=0.00000001):
    """Normalizes this vector to the given power.
    For example, ``power=2.0`` normalizes the vector using Euclidean norm.
    The given `epsilon` is added to the denominator (to prevent divide-by-zero).
    """
    fac = (val/(sum(s**power for s in seq)+epsilon)) ** (1/power)
    return [s*fac for s in seq]

def normalizesdevs(lst, nstddevs=2):
    """Returns a normalized version of a list using the mean and standard deviation.
    This means subtracting the mean, and dividing by ``nstddevs*stdev``"""
    from math import sqrt
    mean = sum(lst)/float(len(lst))
    sdev = sqrt(sum([(x-mean)**2 for x in lst])/float(len(lst)))
    if sdev == 0: return [0.0] * len(lst)
    ret = array('f', [(x-mean)/(nstddevs*sdev) for x in lst])
    return ret

def derivatives(seq):
    """Returns the derivative of this sequence by differencing.
    The output has the same length (`0` is added to the end)."""
    ret = [b-a for a,b in zip(seq, seq[1:])]
    ret.append(0)
    return ret

def extrema(seq):
    """Returns the (locs, vals) of the extrema of the given `seq`.
    An extrema is defined as a point which is greater or smaller than both of
    its neighbors. Both endpoints of the `seq` are always extrema."""
    locs = []
    vals = []
    for i, v in enumerate(seq):
        if i == 0 or i == len(seq)-1:
            locs.append(i)
            vals.append(v)
            continue
        prev = seq[i-1]
        next = seq[i+1]
        if (v < prev and v < next) or (v > prev and v > next):
            locs.append(i)
            vals.append(v)
            continue
    return locs, vals

def rankedpeaks(a, minorder, fac=0.9):
    """Returns indices of peaks in `a`, ranked by `order` of peak.
    This repeatedly calls :func:`scipy.signal.argrelmax` with decreasing
    `order` parameter down to the given `minorder`, by factors of `fac`.
    What this means is that the peaks that are furthest from other peaks are
    returned first.
    """
    from scipy.signal import find_peaks_cwt, argrelmax # type: ignore
    import numpy as np
    a = np.array(a)
    #peaki = find_peaks_cwt(np.array(vals), np.arange(1,100))
    maxorders = np.zeros(a.shape)
    done = set()
    cur = len(a)
    ret = []
    while 1:
        peaki = argrelmax(a, order=cur)[0]
        for i in peaki:
            if i not in done:
                ret.append(i)
            done.add(i)
        cur = int(cur*fac)
        if cur < minorder: break
    return ret

def selectivity(weights, thresh=0.1):
    """Computes normalized selectivity of a set of `weights`.
    This is something like "peakiness" of the distribution.
    Currently, this is computed by looking at how many weights
    are above the given `thresh`.
    The result is normalized by the length of `weights`.
    """
    return len([w for w in weights if w > thresh])/float(len(weights))

def entropy(h, normalize=1.0):
    """Returns the entropy of a given histogram (just a seq of values).
    This is just ``-sum(v*log(v))``.
    If `normalize` is > 0 (default: 1.0), then first normalizes vals to have given sum."""
    from math import log
    from nkpylib.utils import normalize as norm
    if normalize > 0:
        h = norm(h, power=1.0)
    ret = -sum(v*log(v) for v in h if v != 0)
    return ret

def infogain(h1, h2):
    """Computes the information gain going from `h1` to `h2`.
    These are histograms (actually just a sequence of values).
    The gain is calculated as ``entropy(h2)-entropy(h1)``.
    """
    return entropy(h2)-entropy(h1)

def histsimilarity(h1, h2):
    """Computes the similarity between two histograms (seq of floats).
    Right now, this is just inverse :math:`L_1` distance (or 1 if equal)"""
    d = lpdist(h1, h2, 1.0)
    if d == 0: return 1.0
    return 1.0/d

def simplenn(data, fvec, metric='l2', normalize=None, withsum=1):
    """Simple nearest neighbors classification.
    Computes distances from the given feature vector `fvec` to each row of `data`.
    Returns a vector of distances, in same order as data.
    You can specify one of the following metrics:
        - 'l2': L2 (euclidean) [default]
        - 'l1': L1 (manhattan)
        - 'chisq': Symmetric chi-squared [:math:`\\frac{(fvec-datarow)^2}{2*(fvec+datarow)}`]
        - 'int': Histogram intersection [sum of minimum of values]
        - 'bhatt': Bhattacharya distance [sum of sqrt of products of values]
    If `normalize` is `None` (default), no normalization is done.
    Else, it should be a pair of ``(means, sdevs)``, which is used for normalization.
    If `withsum` is 1 (default), then computes the final sum for each row.
    Else, does not compute final sum, so output is a full matrix.
    """
    import numpy
    METRICS = 'l1 l2 chisq int bhatt'.split()
    assert metric in METRICS
    # normalize if needed
    fvec = numpy.array(fvec)
    if normalize:
        means, sdevs = normalize
        fvec = (fvec - means) / sdevs
    #print fvec.shape, data.shape
    # get distances
    if metric == 'l2':
        dists = (data - fvec) ** 2
        if withsum:
            dists = numpy.sum(dists, 1)
    elif metric == 'chisq':
        top = ((fvec - data) ** 2)
        dists = top/(2*(data+fvec+0.0001))
        if withsum:
            dists = numpy.sum(dists, 1)
        #print 'Sizes: %s' % ([fvec.shape, data.shape, top.shape, dists.shape],)
    elif metric == 'l1':
        dists = numpy.abs((data - fvec))
        if withsum:
            dists = numpy.sum(dists, 1)
    elif metric == 'int':
        dists = numpy.minimum(data, fvec)
        #print >>sys.stderr, 'here we are!!', dists[0, :], dists.shape, dists[0, :].shape, sum(dists[10, :])
        if withsum:
            dists = MAX_VALUE - numpy.sum(dists, 1)
        #print >>sys.stderr, 'here we are!!', dists[0], dists.shape
    elif metric == 'bhatt':
        dists = numpy.sqrt(data * fvec)
        if withsum:
            dists = MAX_VALUE - numpy.sum(dists, 1)
    return dists

def bulkNNl2(test, data):
    """Bulk nearest neighbor calculation using :math:`L_2` distance.
    If you have a set of feature vectors (`test`), and for each one,
    you want to compute distances to all feature vectors in `data`,
    then this is what you want to use. That is::
        m = data = [M x D]
        n = test = [N x D]
        dists = [N x M]

    where `M` is the number of vectors in the dataset,
    `N` is the number of vectors in the test set,
    and `D` is the dimensionality of each feature vector.
    Returns the `dists` matrix. You can use :func:`filternnresults()` to actually get NN.

    This function uses the following expansion for speed:
    :math:`\\sum{(m_i - n_j)^2} = \\sum{m_i^2} + \\sum{n_j^2} - 2\\sum{m_i*n_j}`

    The benefit here is that no loops or indexing are needed, and if `m` stays the same,
    part of the computation can be cached. However, we do not currently support caching.
    """
    import numpy as np
    m, n = data, test
    times = [time.time()]
    m2 = m**2
    m2 = m2.sum(1)
    m2 = np.tile(m2, (len(n),1))
    mt = m.transpose()
    times.append(time.time())
    n2 = n**2
    n2 = n2.sum(1)
    n2 = np.tile(n2, (len(m),1)).transpose()
    times.append(time.time())
    mn = np.dot(n, mt)
    times.append(time.time())
    ret = m2+n2-2*mn
    times.append(time.time())
    #print getTimeDiffs(times)
    return ret

def filternnresults(dists, k=None, r=None, sort=1):
    """Filters nearest neighbor results based on the given `k`, `r`, `sort`.
    Takes a list of distances as input (e.g., from :func:`simplenn`) and
    returns a list of ``(distance, index)`` pairs.

    If ``sort == 1`` (default), then sorts results.
    If ``r >= 0``, then only keeps results which are within that radius.
    If ``k > 0 and sort==1``, then only keeps the top `k` results.
    You can specify both `k` and `r` if you want.

    .. note::
        If you specify `k` but turn off sorting, then the `k` is ignored.
        This is not a requirement for `r`.
    """
    import numpy
    assert r is None or isinstance(r, (int,long,float))
    assert k is None or isinstance(k, (int,long))
    # filter and sort results
    t1 = t2 = t3 = t4 = t5 = time.time()
    rfilt = r is not None and r >= 0
    dists = numpy.array(dists)
    #print 'Got r %s' % (r,)
    if sort:
        if rfilt:
            # if we're sorting and filtering by distance, do the distance first
            #print 'd', len(dists), min(dists), max(dists), dists
            #origdists = dists
            # keep track of indices of valid dists
            tokeep = (dists <= r)
            #print 'tk', len(tokeep), tokeep
            t2 = time.time()
            nz = tokeep.nonzero()[0]
            #print 'nz', len(nz), nz
            t3 = time.time()
            # filter list of distances down to this valid list
            dists = dists[tokeep]
            #print 'd,o', len(dists), len(origdists), dists
            # get indices of sorted dists
            inds = numpy.argsort(dists)
            #print 'i', len(inds), inds
            # sort dists by these indices
            dists = dists[inds]
            t4 = time.time()
            # map the indices back to the original ones
            inds = nz[inds]
            #print 'i', len(inds), inds
            if 0: # for checking only
                # map the original distances using these indices
                check = origdists[inds]
                print('c', len(check), check)
                # check that the original distances map to the same list of sorted distances
                for idx, i in enumerate(inds):
                    if idx % 1000 == 0:
                        print(i, origdists[i], dists[idx])
                    if idx > 20000: break
        else:
            # we're sorting, but not filtering by distance
            #log('got %s, %s' % (type(dists), dists[:5]))
            inds = numpy.argsort(dists)
            dists = dists[inds]
    else:
        # not sorting
        if rfilt:
            # distance filtering
            # keep track of indices of valid dists
            tokeep = (dists <= r)
            t2 = time.time()
            inds = tokeep.nonzero()[0]
            t3 = time.time()
            # filter list of distances down to this valid list
            dists = dists[tokeep]
            t4 = time.time()
        else:
            # doing nothing
            # indices are simply from range()
            inds = range(len(dists))

    # zip distances and indices together
    ret = zip(dists, inds)
    # filter by k if wanted
    if k is not None and k > 0:
        ret = ret[:k]
    t5 = time.time()
    #log('In filter, got %s' % (getTimeDiffs([t1,t2,t3,t4,t5])))
    return ret


## PROBABILITY AND SAMPLING UTILS
def minsample(population, k, randomize=1):
    """Samples upto `k` elements from `population`, without replacement.
    Equivalent to :func:`random.sample`, but works even if ``k >= len(population)``.
    In the latter case, it samples all elements (in random order)."""
    if randomize:
        from random import sample
        return sample(population, min(k, len(population)))
    return population[:k]

def freqs2probs(freqs):
    """Converts the given frequencies (list of numeric values) into probabilities.
    This just normalizes them to have sum = 1"""
    freqs = list(freqs)
    total = float(sum(freqs))
    return [f/total for f in freqs]

def choiceWithProbs(seq, probs):
    """Chooses an item from `seq` randomly, but using probabilities given in `probs`.
    Both sequences should have the same length. `probs` is normalized first to unit sum.
    Runs in linear time, by converting to CDFs first.
    """
    cdfs = cumsum(freqs2probs(probs))
    r = random.random()
    for el, cdf in zip(seq, cdfs):
        if r < cdf: return el
    assert 'Invalid probabilities!'

def propsample(freqs, num):
    """Proportionally samples from the given frequencies.
    Returns a list of same length with the number of times each index should be
    sampled such that the total number of elements sampled is `num`.
    """
    lens = [int(f*num)+1 for f in freqs]
    total = 0
    for i, l in enumerate(lens):
        if l+total > num:
            lens[i] = num-total
        total += lens[i]
    return lens

def sampleWithReplacement(population, k):
    """Samples `k` elements, with replacement, from the `population`.
    Just calls :func:`random.choice` `k` times.
    """
    return [choice(population) for i in xrange(k)]

def estimateGaussian(data, unbiased=1):
    """Estimates a 1-d gaussian from `data` (a list of values) and returns ``(mean, variance)``."""
    l = float(len(data))
    mean = sum(data)/l
    var = sum([(d-mean)**2.0 for d in data])
    if unbiased:
        var /= l-1
    else:
        var /= l
    return mean, var

def unitstep(t):
    """Returns the unit step function: ``u(t) = 1.0 if t>=0 else 0``"""
    return 1.0 if t >= 0 else 0.0

def gaussian(x, mean, var):
    """Given the mean and variance of a 1-d gaussian, return the y value for a given `x` value.

    .. math:: \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}
    """
    denom = sqrt(2*pi*var)
    num = exp(-((x-mean)**2)/(2*var))
    ret = num/float(denom)
    #print "Gaussian of x=%s (m=%s, var=%s) is %s" % (x, mean, var, ret)
    return ret

def gaussian2d(x, y, sigma):
    """The symmetric 2d unit gaussian function.

    .. math:: \\frac{1}{2\\pi\\sigma^2}e^{\\frac{x^2 + y^2}{2\\sigma^2}}
    """
    s2 = sigma * sigma
    ret = exp(-0.5 * (x*x + y*y)/s2) / (2 * pi * s2)
    return ret

def randomizedPartition(data, probs, randomize=1):
    """Partitions a dataset (list of values/rows) into sets using the probabilities given.
    Useful for generating training/test sets.
    If `randomize` is set to 0, then simply assigns data sequentially to output sets.
    Otherwise (default) randomizes the outputs.

    Returns a list of same length as `probs`, with each entry as a list of
    non-overlapping elements from `data`.
    """
    probs = freqs2probs(probs)
    indices = []
    for i, p in enumerate(probs):
        num = int(p*len(data))
        if i == len(data)-1:
            num = len(data)-len(indices)
        indices.extend([i]*num)
    assert len(indices) == len(data)
    ret = [[] for p in probs]
    if randomize:
        shuffle(indices)
    for i, d in zip(indices, data):
        ret[i].append(d)
    return ret

def expweight(lst, fromt=None, fac=1.0):
    """Weights values in list of ``(value, time)`` pairs exponentially by time and returns the sum.
    If `fromt` is not `None`, then subtracts it from each time first.
    Each time is multiplied by the given factor, prior to exponentiation (default 1.0).
    """
    ret = 0.0
    tot = 0.0
    for v, t in lst:
        if fromt is not None:
            t -= fromt
        t *= fac
        ret += exp(t)*v
        tot += exp(t)
    ret /= tot
    return ret


# OTHER/MISC
def spawnWorkers(num, target, name=None, args=(), kwargs={}, daemon=1, interval=0):
    """Spawns the given number of workers, by default daemon, and returns a list of them.
    'interval' determines the time delay between each launching"""
    threads = []
    for i in range(num):
        if name and '%d' in name:
            name = name % i
        t = Thread(target=target, name=name, args=args, kwargs=kwargs)
        t.setDaemon(daemon)
        t.start()
        threads.append(t)
        time.sleep(interval)
    return threads

def gpsdeg2dec(lat, lon):
    """Converts GPS coordinates given as (deg, min, sec, dir) pairs (dir = N/S/E/W) to decimal degrees tuple"""
    assert len(lat) == len(lon) == 4
    vals = [lat, lon]
    dirs = [lat[-1], lon[-1]]
    vals = [(v[0], int(v[1])/60.0, v[2]/3600.0) for v in vals]
    vals = [sum(v) for v in vals]
    vals[0] *= 1 if dirs[0].lower() == 'n' else -1
    vals[1] *= 1 if dirs[1].lower() == 'e' else -1
    return vals

def getGeoName(gpsloc):
    """Returns the closest city/neighborhood name for a given latitude, longitude pair, or '' on error"""
    url = 'http://ws.geonames.org/findNearbyPlaceNameJSON?lat=%s&lng=%s' % (gpsloc[0], gpsloc[1])
    try:
        s = urlopen(url).read()
    except IOError: return ''
    locs = json.loads(s)['geonames']
    if not locs: return ''
    l = locs[0]
    name = l['name'] + ', ' + l['countryName']
    return name

def sendemail(toaddress, subject, body, images=[], username=None, password=None, fromaddress=None, server='localhost', ssl=0, replytoaddress=None):
    """Sends email with information given.
    If fromaddress is not given, it is set as the toaddress.
    If replaytoaddress is not given, it is set as the fromaddress.
    If images is not None, then sends in HTML format with embedded images."""
    conn = smtplib.SMTP(server)
    if ssl:
        #from ssmtplib import SMTP_SSL
        #conn = SMTP_SSL(server)
        conn.starttls()
        conn.login(username, password)

    #print "Sending email to %s with subject %s and body %s" % (toaddress, subject, body)
    # note that the from address in the header can be whatever we want...
    if not fromaddress:
        fromaddress = toaddress
    if not replytoaddress:
        replytoaddress = fromaddress

    # This part from http://docs.python.org/library/email-examples.html
    # Here are the email package modules we'll need

    COMMASPACE = ', '

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = subject
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = fromaddress
    msg['To'] = toaddress
    msg.add_header('Reply-to', replytoaddress)
    #msg.preamble = body # TODO I think this is not needed
    msg.attach(MIMEText(body))

    # add images at the end
    for im in images:
        # Open the files in binary mode.  Let the MIMEImage class automatically
        # guess the specific image type.
        img = MIMEImage(open(im, 'rb').read())
        img.add_header('Content-Disposition', 'attachment', filename=im.replace('/', '-'))
        msg.attach(img)

    #msg = 'From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n%s' % (fromaddress, toaddress, subject, body)
    # ...but here, the from address is actually checked by the smtp server, so we have to use something real
    conn.sendmail(fromaddress, toaddress, msg.as_string())
    conn.quit()

def _memtest():
    """Tests the various mem utils"""
    print(procmem())
    print(totalmem())
    m = MemUsage()
    print('Created m')
    print(m.usage())
    print(m.delta())
    a = range(1000000)
    m.add('after a')
    print(m.usage())
    print(m.delta())
    b = range(2000000)
    m.add('after b')
    print(m.usage())
    print(m.delta())
    del b
    m.add('after del')
    print(m.usage())
    print(m.delta())
    print(m.usage('start'))
    print(m.delta('after b'))
    for i in m:
        print(i)
    print(m['after a'])

def getConsoleSize():
    """Returns the (width, height) of the current console window.
    If there is some error, returns (-1, -1).
    Only tested on linux.
    """
    try:
        if 0:
            # Taken from http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python/943921#943921
            rows, cols = Popen(['stty', 'size'], stdout=PIPE).communicate()[0].strip().split()
        else:
            # Taken from https://bbs.archlinux.org/viewtopic.php?pid=1091712
            rows = Popen(['tput', 'lines'], stdout=PIPE).communicate()[0]
            cols = Popen(['tput', 'cols'], stdout=PIPE).communicate()[0]
        return (int(cols), int(rows))
    except Exception:
        return (-1, -1)

def getNumCPUs():
    """Returns the number of cpus (independent cores) on this machine.
    If any error, returns 1"""
    n = [l.strip() for l in open('/proc/cpuinfo') if l.startswith('processor')]
    return len(n)

def parseNProcs(num):
    """Parses a given number of procs into an actual number:
        if num is an int:
            > 0: that many procs
            <= 0: getNumCPUs() - num
        elif num is a float:
            that percentage of cpus of this sys
        Num is guaranteed to be at least 1
    """
    # convert num to concrete nprocs
    if isinstance(num, (int, long)):
        if num > 0:
            nprocs = num
        else:
            nprocs = getNumCPUs()+num
    elif isinstance(num, float):
        nprocs = int(getNumCPUs()*num)
    else:
        raise ValueError('num must be an int or float! Got: %s' % (num,))
    # make sure we have at least one proc
    if nprocs <= 0:
        nprocs = 1
    return nprocs


def stdmainloop(callback):
    """A "standard" main loop that reads a line at a time from stdin and does something with it.
    The loop exits when we hit EOF or when the callback returns anything false
    value other than None. This allows you to have a callback that doesn't
    return anything, but keeps the loop going. We flush the stdout after every call.
    """
    while 1:
        line = sys.stdin.readline()
        if not line: break
        line = line.rstrip('\n')
        ret = callback(line)
        if ret is not None and not ret: break
        try:
            sys.stdout.flush()
            pass
        except IOError: pass
    try:
        sys.stdout.flush()
    except IOError: pass

def genericWorkerLoop(funcgetter='eval', globals=None, locals=None):
    """Runs a loop that calls arbitrary functions per input line from stdin.
    THIS IS EXTREMELY DANGEROUS, SO USE WITH CAUTION!

    'funcgetter' determines how we get a runnable function from a name:
        'eval' (default): eval(funcname, globals, locals)
        'global': globals[funcname]
        'local': locals[funcname]
        'method': getattr(locals['self'], funcname)

    If globals or locals are not given, then they are created as usual.

    Each input line should contain the following (tab-separated):
        a unique id for this call
        function name - 'funcgetter' determines how a function is gotten from this.
        args - json-encoded list of arguments, or empty string for no args
        kw - json-encoded list of kwargs, or empty string for none
    This results in the following computation:
        ret = eval(function_name)(*args, **kw)
        out = json.dumps(ret)
        print '%s\t%s' % (id, out)
    The result is printed to stdout, as a single line, and then the stream is flushed.
    The loop exits if the input is empty or closed.
    If the input was invalid, an error string is printed to stderr, and just "error" to stdout.
    """
    if not globals:
        globals = globals()
    if not locals:
        locals = locals()
    while 1:
        line = sys.stdin.readline()
        if not line: break
        line = line.rstrip('\n')
        out = 'error'
        try:
            lineid, funcname, args, kw = line.split('\t')
            out = '%s\terror' % (lineid)
            if funcgetter == 'eval':
                func = eval(funcname, globals, locals)
            elif funcgetter == 'global':
                func = globals[funcname]
            elif funcgetter == 'local':
                func = locals[funcname]
            elif funcgetter == 'method':
                func = getattr(locals['self'], funcname)
            args = json.loads(args) if args else ()
            kw = json.loads(kw) if kw else {}
            #print >>sys.stderr, 'Got: |%s| |%s| |%s|' % (func, args, kw)
            ret = func(*args, **kw)
            out = '%s\t%s' % (lineid, json.dumps(ret))
        except Exception as e:
            print('Ran into error of type %s: %s' % (type(e), e), file=sys.stderr)
            traceback.print_exc()
        try:
            print(out)
            sys.stdout.flush()
        except IOError: pass
    try:
        sys.stdout.flush()
    except IOError: pass

def printprofile(profile, stream=sys.stderr):
    """Given a LineProfile() object from the line_profiler class,
    prints it if it has something to print, to the given stream.
    http://packages.python.org/line_profiler/

    To use it, import it and create a new instance at the top of your module:
        from line_profiler import LineProfiler
        profile = LineProfiler()

    Then decorate any functions you want with the instance:
        @profile
        def myfunc():
            ...

    Finally, call this function:
        printprofile(profile)
    """
    printprof = 0
    for k, v in profile.code_map.items():
        if v:
            printprof = 1
            break
    if printprof:
        profile.print_stats(stream=stream)


def directmain(taskfuncs):
    """A main() function that is just a simple wrapper around direct function calls."""
    tasks = dict([(f.__name__, f) for f in taskfuncs])
    if len(sys.argv) < 2:
        print('Usage: python %s <%s> [<args>...]' % (sys.argv[0], '|'.join(tasks)))
        sys.exit()
    task = sys.argv[1]
    assert task in tasks
    func = tasks[task]
    return func(*sys.argv[2:])

def testurl2fname():
    """Driver to test out url2fname()"""
    for line in sys.stdin:
        url = line.strip()
        hostdir = host2dirname(url, collapse=1, delport=1)
        basedir = os.path.join('Angie Rocks!@()#$%^&*~~.33.jpg', hostdir)
        dir = url2fname(url, basedir=basedir, maxlen=128)
        print(dir)

T = TypeVar("T")
def is_instance_of_type(value: Any, expected_type: Type[T]) -> bool:
    """Recursively checks if `value` matches the structure of `expected_type`.

    Supports generic types like list[tuple[float, str]].
    """
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is None:  # Base case: non-generic type
        if expected_type is Any:
            return True
        return isinstance(value, expected_type)
    elif origin is list:
        if not isinstance(value, list):
            return False
        return all(is_instance_of_type(item, args[0]) for item in value)
    elif origin is tuple:
        if not isinstance(value, tuple) or len(value) != len(args):
            return False
        return all(is_instance_of_type(item, arg) for item, arg in zip(value, args))
    elif origin is Literal:
        return value in args
    else: # If the type isn't handled explicitly
        raise NotImplementedError(f"Unsupported type: {expected_type}")

def is_mapping(obj: Any) -> bool:
    """Returns True if the given `obj` is a mapping (dict-like).

    This checks for various methods, including __getitem__, __iter__, and __len__, keys(), items(),
    values(), etc.
    """
    to_check = ['__getitem__', '__iter__', '__len__', 'keys', 'items', 'values']
    for method in to_check:
        if not hasattr(obj, method):
            return False
    return True


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    try:
        cmd = sys.argv[1]
        if cmd == 'mail':
            to, subject = sys.argv[2:4]
            body = sys.stdin.read()
            sendemail(to, subject, body)
    except Exception: pass
    #sendemail('neeraj@cs.columbia.edu', 'hello4', 'Here is the body of the image. Image should be attached.', fromaddress="API Upload <neeraj@cs.columbia.edu>", images=['blah.jpg'])
