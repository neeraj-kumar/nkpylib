#!/usr/bin/env python
"""Lots of small python utilities, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2010-2014, Neeraj Kumar (http://neerajkumar.org)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NEERAJ KUMAR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
- String and Path Utils: :data:`IMAGE_EXTENSIONS`
- Other/misc: :func:`spawnWorkers`


Code Starts Now
-----------------
"""

import os, sys, random, math, time
from math import pi
from itertools import *
from threading import Thread
import string

# DECORATORS
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

def memoize(fn):
    """Decorator to cache a function.

    .. warning::
       Make sure it's a functional method (i.e., no side effects)!
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
    import threading
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
            import cPickle as pickle
            import tempfile
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
        import cPickle as pickle
        cache = [None]
        lasttime = [time.time()]
        def retfunc(*args, **kw):
            import cPickle as pickle
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
                    import tempfile
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
            import time
            while 1:
                try:
                    return f(*args, **kw)
                except exceptions, e:
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
        except Exception, e:
            print >>sys.stderr, ' ** Hit an exception of type %s: %s' % (type(e), e)
    #print >>sys.stderr, 'Finished queueize, this is a problem!'
    #sys.stderr.flush()


# TIMING UTILS
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
    from datetime import datetime
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
    from datetime import datetime
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
    from calendar import timegm
    from datetime import datetime
    if isinstance(t, datetime):
        return timegm(t.timetuple())
    if not isinstance(t, (str, unicode)): return t
    els = t.split('.', 1)
    fmt = '%Y-%m-%d %H:%M:%S'
    try:
        st = time.strptime(els[0], fmt)
    except ValueError, e:
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
    from datetime import datetime
    try:
        import pytz
        d = datetime.now(pytz.utc)
    except ImportError:
        d = datetime.utcnow()
    #d.microsecond = 0 #FIXME this is not writable...is it needed?
    return d

def now():
    """Returns the current time as a :class:`datetime` obj, with ordinary precision, in localtime"""
    from datetime import datetime
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
    from datetime import datetime
    import pytz
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
    dists = [(haversinedist(loc, l), l, tz) for l, tz in tzlookup]
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
    from datetime import date, datetime
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


# MEMORY UTILS
#: mapping strings to multipliers on bytes
MEMORY_UNITS = {'B': 1, 'kB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}

def memstr2bytes(s):
    """Converts a memory string like '1249 MB' to number of bytes.
    If it can't be converted, raises a :class:`ValueError`."""
    try:
        import re
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
    UNICODE_MEM = 4
    #log('Obj %s has type %s' % (repr(obj), type(obj)))
    if type(obj) == type(123): return INT_MEM
    elif type(obj) == type(1.23): return FLOAT_MEM
    elif isinstance(obj, str): return len(obj)*CHAR_MEM
    elif isinstance(obj, unicode): return len(obj)*UNICODE_MEM
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sum((getmem(o) for o in obj))
    elif isinstance(obj, dict):
        return sum((getmem(k)+getmem(v) for k, v in obj.iteritems()))
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
        except (IndexError, TypeError): pass
        # otherwise name
        for d in reversed(self.data):
            if d['name'] == key:
                return d
        # keyerror
        raise KeyError("'%s' not found in items" % (key,))


# LOGGING UTILS
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
        print >>f
        s = s[1:]
    if funcindent >= 0:
        from inspect import stack
        s = '  ' * max(len(stack())-funcindent, 0) + s
    if '\r' in s:
        print >>f, s,
    else:
        print >>f, s
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
        print >> out, '\r%s' % (blanks),
        # now print the message
        # TODO deal with i/total somehow
        print >> out, '\r%s' % (msg),
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
        import code
        import readline, rlcompleter
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
    from StringIO import StringIO
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
            print >>f, sout
        else:
            Popen(args, stdout=f).communicate()


# ITERTOOLS AND SEQUENCES UTILS
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

def any(seq, pred=None):
    """Returns `True` if `pred(x)` is true for at least one element in the iterable.
    Taken from the :mod:`itertools` recipes.
    """
    for elem in ifilter(pred, seq):
        return True
    return False

def all(seq, pred=None):
    """Returns `True` if `pred(x)` is true for all elements in the iterable.
    Taken from the :mod:`itertools` recipes.
    """
    for elem in ifilterfalse(pred, seq):
        return False
    return True

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
    iseq = sorted([(v, i) for i, v in enumerate(seq)], key=lambda (v,i): ukey(v), cmp=cmp, reverse=reverse)
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


# DICT UTILS
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


# MATH UTILS
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
    from math import exp
    return 1.0/(1+exp(-x))

def lpdist(x, y, p=2):
    """Returns the :math:`L_p` distance between the two vectors.
    Works for p=0, 1, 2 or any higher number (but not infinity).

    .. seealso::
        :func:`linfdist`
            The function for computing :math:`L_\infty` distances.
    """
    from math import sqrt
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
    from scipy.signal import find_peaks_cwt, argrelmax
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
                print 'c', len(check), check
                # check that the original distances map to the same list of sorted distances
                for idx, i in enumerate(inds):
                    if idx % 1000 == 0:
                        print i, origdists[i], dists[idx]
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


# GEOMETRY UTILS
# All triangle functions take (x,y) pairs as inputs for points
def getDistance(pt1, pt2):
    """Returns euclidean distance between two points"""
    return lpdist(pt1, pt2, 2)

def ptLineDist(pt, line):
    """Returns distance between `pt` ``(x,y)`` to `line` ``((x0,y0), (x1,y1))``, and the closest point on the line.

    Adapted from http://paulbourke.net/geometry/pointlineplane/

    Example::
        >>> ptLineDist((0.5, 1.0), [(0,0), (1, 0)])
        (1.0, (0.5, 0.0))
        >>> ptLineDist((0.0, 0.0), [(0,0), (1, 0)])
        (0.0, (0.0, 0.0))
        >>> ptLineDist((1.0, 0.0), [(0,0), (1, 1)])
        (0.70710678118654757, (0.5, 0.5))
        >>> ptLineDist((-5, 0.0), [(0,0), (1, 0)])
        (5.0, (0.0, 0.0))
    """
    x, y = pt
    (x0, y0), (x1, y1) = line
    dx, dy = x1-x0, y1-y0
    t = ((x-x0)*dx + (y-y0)*dy)/(dx**2 + dy**2)
    t = clamp(t, 0.0, 1.0)
    intersection = intx, inty = (x0+t*dx, y0+t*dy)
    d = getDistance(pt, intersection)
    return (d, intersection)

def distAlong(d, pt1, pt2):
    """Returns the coordinate going distance `d` from `pt1` to `pt2`.
    Works for any dimensionalities.
    """
    dist = getDistance(pt1, pt2)
    ret = [(d/dist * (pt2[dim]-pt1[dim])) + pt1[dim] for dim in range(len(pt1))]
    return ret

def expandBox(box, facs):
    """Expands a `box` about its center by the factors ``(x-factor, y-factor)``.
    The box is given as ``(x0, y0, x1, y1)``"""
    w, h = box[2]-box[0], box[3]-box[1]
    cen = cx, cy = (box[2]+box[0])/2.0, (box[1]+box[3])/2.0
    nw2 = w*facs[0]/2.0
    nh2 = h*facs[1]/2.0
    box = [cx-nw2, cy-nh2, cx+nw2, cy+nh2]
    return box

def rectarea(r, incborder=1):
    """Returns the area of the given ``(x0, y0, x1, y1)`` rect.
    If `incborder` is true (default) then includes that in calc. Otherwise doesn't.
    If either width or height is not positive, returns 0."""
    w = r[2]-r[0] + incborder
    h = r[3]-r[1] + incborder
    if w <= 0 or h <= 0: return 0
    return w * h

def rectcenter(rect, cast=float):
    """Returns the center ``[x,y]`` of the given `rect`.
    Applies the given `cast` function to each coordinate."""
    return [cast((rect[0]+rect[2]-1)/2.0), cast((rect[1]+rect[3]-1)/2.0)]

def rectintersection(r1, r2):
    """Returns the rect corresponding to the intersection between two rects.
    Returns `None` if non-overlapping.
    """
    if r1[0] > r2[2] or r1[2] < r2[0] or r1[1] > r2[3] or r1[3] < r2[1]: return None
    ret = [max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3])]
    return ret

def rectoverlap(r1, r2, meth='min'):
    """Returns how much the two rects overlap, using different criteria:

        - 'min': ``intersection/min(a1, a2)``
        - 'max': ``intersection/max(a1, a2)``
    """
    a1 = rectarea(r1)
    a2 = rectarea(r2)
    i = rectintersection(r1, r2)
    if not i: return 0
    ai = float(rectarea(i))
    if meth == 'min':
        return ai/min(a1, a2)
    if meth == 'max':
        return ai/max(a1, a2)

def rectAt(cen, size):
    """Returns a rectangle of the given `size` centered at the given location.
    The coordinates are inclusive of borders."""
    x, y = cen[:2]
    w, h = size[:2]
    return [x-w//2, y-h//2, x-w//2+w-1, y-h//2+h-1]

def trilengths(pt1, pt2, pt3):
    """Returns the lengths of the sides opposite each corner"""
    d1 = getDistance(pt2, pt3)
    d2 = getDistance(pt1, pt3)
    d3 = getDistance(pt1, pt2)
    ret = [d1, d2, d3]
    return ret

def triarea(pt1, pt2, pt3):
    """Returns the area of the triangle.
    Uses `Heron's formula <http://en.wikipedia.org/wiki/Heron%27s_formula>`_
    """
    a, b, c = trilengths(pt1, pt2, pt3)
    s = (a+b+c)/2.0
    return math.sqrt(s*(s-a)*(s-b)*(s-c))

def getTriAngles(pt1, pt2, pt3):
    """Returns the angles (in rads) of each corner"""
    from math import acos
    lens = l1, l2, l3 = trilengths(pt1, pt2, pt3)
    a1 = acos((l2**2 + l3**2 - l1**2)/(2 * l2 * l3))
    a2 = acos((l1**2 + l3**2 - l2**2)/(2 * l1 * l3))
    a3 = acos((l1**2 + l2**2 - l3**2)/(2 * l1 * l2))
    angles = [a1, a2, a3]
    return angles

def trialtitude(pt1, pt2, pt3):
    """Returns the coordinates of the other end of the altitude starting at `p1`."""
    from math import cos
    lens = l1, l2, l3 = trilengths(pt1, pt2, pt3)
    angles = a1, a2, a3 = getTriAngles(pt1, pt2, pt3)
    dfrom2 = cos(a2)*l3
    return distAlong(dfrom2, pt2, pt3)

def haversinedist(loc1, loc2):
    """Returns the haversine great circle distance (in meters) between two locations.
    The input locations must be given as ``(lat, long)`` pairs (decimal values).

    See http://en.wikipedia.org/wiki/Haversine_formula
    """
    from math import sin, cos, radians, atan2, sqrt
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    R = 6378100.0 # mean radius of earth, in meters
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    sdlat2 = sin(dlat/2)
    sdlon2 = sin(dlon/2)
    a = sdlat2*sdlat2 + cos(radians(lat1))*cos(radians(lat2))*sdlon2*sdlon2
    d = R * 2 * atan2(sqrt(a), sqrt(1-a))
    return d

def polyarea(poly):
    """Returns the signed area of the given polygon.
    The polygon is given as a list of ``(x, y)`` pairs.
    Counter-clockwise polys have positive area, and vice-versa.
    """
    area = 0.0
    p = poly[:]
    # close the polygon
    if p[0] != p[-1]:
        p.append(p[0])
    for (x1, y1), (x2, y2) in zip(p, p[1:]):
        area += x1*y2 - y1*x2
    area /= 2.0
    return area

def pointInPolygon(pt, poly, bbox=None):
    """Returns `True` if the point is inside the polygon.
    If `bbox` is passed in (as ``(x0,y0,x1,y1)``), that's used for a quick check first.
    Main code adapted from http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
    """
    x, y = pt
    if bbox:
        x0, y0, x1, y1 = bbox
        if not (x0 <= x <= x1) or not (y0 <= y <= y1): return 0
    c = 0
    i = 0
    nvert = len(poly)
    j = nvert-1
    while i < nvert:
        if (((poly[i][1]>y) != (poly[j][1]>y)) and (x < (poly[j][0]-poly[i][0]) * (y-poly[i][1]) / (poly[j][1]-poly[i][1]) + poly[i][0])):
            c = not c
        j = i
        i += 1
    return c

def pointPolygonDist(pt, poly, bbox=None):
    """Returns the distance from a given point to a polygon, and the closest point.
    If the point is inside the polygon, returns a distance of 0.0, and the point itself.
    The point should be ``(x,y)``, and the poly should be a series of ``(x,y)`` pairs.
    You can optionally pass-in a bounding box ``[x0,y0,x1,y1]`` to run a quick check first.
    (If you don't, it's computed and checked.)

    Returns ``(distance, (x,y))`` of the closest point on the polygon (if outside), else `pt` itself.
    If the polygon is degenerate, then returns ``(0.0, pt)``

    .. note::
        This is not the most efficient function (linear in number of edges of the `poly`).
    """
    if not bbox:
        xs, ys = zip(*poly)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    x, y = pt
    inside = pointInPolygon(pt, poly, bbox=bbox)
    if inside: return (0.0, pt)
    # else, it's outside, so compute distance
    lines = zip(poly, poly[1:]+[poly[0]])
    lines = [(p1, p2) for p1, p2 in lines if p1 != p2]
    dists = [ptLineDist(pt, l) for l in lines]
    if not dists: return (0.0, pt)
    return min(dists)

def distInMeters(dist):
    """Converts distances to a numeric distance in meters.
    If the input is a string, then it can have the following suffixes:
        - 'm': meters
        - 'meter': meters
        - 'meters': meters
        - 'metre': meters
        - 'metres': meters
        - 'km': kilometers
        - 'kilometer': kilometers
        - 'kilometers': kilometers
        - 'kilometre': kilometers
        - 'kilometres': kilometers
        - 'mi': miles
        - 'mile': miles
        - 'miles': miles
        - 'ft': feet
        - 'feet': feet
        - 'foot': feet

    Assumes the string is in the form of a number, optional spaces (of any sort), then the suffix.
    Else, assumes it's numeric and returns it as is.
    """
    import re
    if not isinstance(dist, basestring): return dist
    # else, it's a string, so map it
    mPerMile = 1609.34
    mPerFoot = 0.3048
    UNITS = dict(m=1.0, meter=1.0, meters=1.0, metre=1.0, metres=1.0,
        km=1000.0, kilometer=1000.0, kilometers=1000.0, kilometre=1000.0, kilometres=1000.0,
        mi=mPerMile, mile=mPerMile, miles=mPerMile,
        ft=mPerFoot, feet=mPerFoot, foot=mPerFoot,
    )
    # has units, so parse
    match = re.match(r'([-+]?\d*\.\d+|\d+)\s*([a-zA-Z]*)', dist.lower().strip())
    val, unit = match.group(1, 2)
    val = float(val)*UNITS[unit]
    return val

def boxAroundGPS(loc, dist):
    """Returns a bounding box around the given GPS location, within the given distance.
    The location is ``(latitude, longitude)`` and the distance is either a
    single value, or a pair of values ``(lat_dist, lon_dist)``.
    These can be floats (i.e., degrees), or strings, which are assumed to be
    degrees if there is no suffix, or mapped to meters using
    :func:`distInMeters()` if there is a suffix.

    .. note::
        If you give no units, then the returned bbox will be symmetrical in
        degrees around the center, but this is NOT symmetrical in terms of
        distance, since longitudinal distance varies with latitude.

    In contrast, giving units should give symmetric (in terms of distance) bounds.

    For reference:
        - 1 degree latitude = 111.319 km = 69.170 miles.
        - 1 degree longitude = 69.170 miles * cos(`lat`)

    Returns ``[lat0, lon0, lat1, lon1]``
    """
    import re
    assert len(loc) == 2
    try:
        xdist, ydist = dist
    except (ValueError, TypeError):
        xdist = ydist = dist
    ret = []
    mPerDeg = 111318.845 # meters per degree
    for i, (cen, d) in enumerate(zip(loc, [xdist, ydist])):
        try:
            d = float(d)
            # no units -- is degrees
            # easy to calculate ret
            ret.extend([cen-d, cen+d])
        except ValueError:
            # has units, so parse
            val = distInMeters(d)/mPerDeg
            #print 'd %s: Val %s, unit %s' % (d.lower().strip(), val, unit)
            if i == 0:
                # latitude just needs equal increments
                ret.extend([cen-val, cen+val])
            else:
                # longitude needs special computation
                minlat, maxlat = ret # get min and max latitudes
                minlon = val/math.cos(math.radians(minlat))
                maxlon = val/math.cos(math.radians(maxlat))
                #print minlat, maxlat, minlon, maxlon
                ret.extend([cen-minlon, cen+maxlon])
    # permute into right order
    ret = [ret[0], ret[2], ret[1], ret[3]]
    return ret

def getBoxProjection(loc, dist, imsize):
    """Creates a box around the given location and projects points to it.
    The loc is (latitude, longitude).
    The dist is a string that is interpretable by boxAroundGPS().
    The imsize is the size of the images created.

    Returns (project, polyim), which are both functions:
        project(loc): takes a (lat, lon) pair and returns an image (x,y) pair.
        polyim(coords): takes a project()-ed set of coordinates and returns a
                        1-channel image with the polygon drawn in it.
    """
    from PIL import Image, ImageDraw
    from nkpylib.utils import boxAroundGPS, lerp, timed, polyarea, uniqueize
    lat, lon = loc
    box = boxAroundGPS(loc, dist)
    w, h = imsize
    lon2x = lambda lon: int(lerp(lon, (box[1], 0), (box[3], w)))
    lat2y = lambda lat: int(lerp(lat, (box[0], 0), (box[2], h)))
    project = lambda loc: (lon2x(loc[1]), lat2y(loc[0]))
    def polyim(coords):
        """Returns a single channel image for this polygon (already projected)"""
        im = Image.new('L', (w, h), 0)
        if coords:
            draw = ImageDraw.Draw(im)
            draw.polygon(coords, outline=255, fill=255)
        return im

    return (project, polyim)

def createNearMask(imsize):
    """Cached and memoized "near" mask generation.
    This is simply a wrapper on createRadialMask().
    Note that we invert the mask, so that later on we can simply paste(),
    rather than have to composite() with a black image.
    """
    from nkpylib.imageutils import createRadialMask
    from PIL import Image, ImageChops
    fname = 'mask-%d-%d.png' % (imsize[0], imsize[1])
    try:
        return Image.open(fname)
    except Exception:
        mask = createRadialMask(imsize)
        mask = ImageChops.invert(mask)
        mask.save(fname)
    return mask

def projectAndGetExtrema(p, project, polyim, fname=None, mask=None):
    """Takes a polygon and projects it and gets extrema.
    Uses project() to project the coordinates,
    polyim() to get the polygon image.
    If mask is given, then composites the image with the mask.
    If fname is given, then saves the (possibly composited) image to that name.
    Finally, computes the extrema.
    Returns (max value, polygon image, projected coordinates).
    """
    from PIL import Image
    coords = map(project, p)
    pim = polyim(coords)
    if mask:
        pim.paste(0, (0,0), mask)
    if fname:
        pass #pim.save(fname) #FIXME this takes too long...
    m, M = pim.getextrema()
    return (M, pim, coords)

def locateGPS(loc, objs, imsize=(1000,1000), indist='50 meters', neardist='1 km', imdir=None):
    """Figures out what objects this location is "in" and "near".
    'loc' is a (latitude, longitude) pair.
    'objs' is a list of (objkey, polygon) tuples.
    For both "in" and "near", projects a box around the given location to an image.
    This image has size 'imsize'. Also projects all given object polygons to this image.

    For "in", checks for any objects that intersect a box within distance
    "indist" from the given location.

    For "near", computes distance from loc to any objects within 'neardist'
    (that were not 'in').

    Returns (objsin, objsnear), where each is a sorted list of (objkey, score) pairs.
    For "in", the score is 1.0. [Should it be (area of intersection)/(area of obj)?]
    The objects are sorted from least area to greatest area.
    For "near", the score is minimum distance between location and obj
    boundaries as a fraction of 'indist', squared to get a faster fall-off.

    If imdir is given, then saves debugging images within that directory.
    """
    #TODO check if done?
    from PIL import Image
    from nkpylib.utils import polyarea, uniqueize
    from nkpylib.imageutils import combineImages
    #log('Trying to locate %s with %d objs, imsize %s, dists %s, %s, imdir %s: %s' % (loc, len(objs), imsize, indist, neardist, imdir, objs[:2]))
    # init
    # create imdir if needed
    if imdir:
        try:
            os.makedirs(imdir)
        except OSError:
            pass
    # setup projection for "in" and run on all objects
    project, polyim = getBoxProjection(loc, indist, imsize)
    objsin = []
    for objkey, p in objs:
        fname = os.path.join(imdir, 'in-%s.png' % (objkey.rsplit(':', 1)[-1])) if imdir else ''
        M, pim, coords = projectAndGetExtrema(p, project, polyim, fname=fname)
        if M == 0: continue # ignore things that don't match at all
        objsin.append([objkey, abs(polyarea(coords)), pim])
    # sort "in" objects by area
    objsin.sort(key=lambda o: o[1])
    if imdir:
        comb = combineImages([o[2] for o in objsin])
        if comb:
            comb.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(imdir, 'in-poly.png'))
    # remap to get scores instead of areas and pims
    objsin = [(o[0], 1.0) for o in objsin]
    log('    Got %d objects "in": %s' % (len(objsin), objsin[:5]))
    # now do "near"
    project, polyim = getBoxProjection(loc, neardist, imsize)
    mask = createNearMask(imsize)
    doneobjs = set([o for o, s in objsin])
    objsnear = []
    for objkey, p in objs:
        if objkey in doneobjs: continue # skip objects we're in
        fname = os.path.join(imdir, 'near-%s.png' % (objkey.rsplit(':', 1)[-1])) if imdir else ''
        M, pim, coords = projectAndGetExtrema(p, project, polyim, fname=fname, mask=mask)
        if M == 0: continue # ignore things that weren't close enough
        objsnear.append([objkey, M/255.0, pim])
    # sort "near" objects by closevalue
    objsnear.sort(key=lambda o: o[1], reverse=1)
    if imdir:
        comb = combineImages([o[2] for o in objsnear])
        if comb:
            comb.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(imdir, 'near-poly.png'))
    # remap to get final scores
    objsnear = [(o[0], o[1]*o[1]) for o in objsnear] # we square the score to get a steeper falloff
    log('    Got %d objects "near": %s' % (len(objsnear), objsnear[:5]))
    return objsin, objsnear


# PROBABILITY AND SAMPLING UTILS
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
    from random import choice
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
    from math import sqrt, exp, pi
    denom = sqrt(2*pi*var)
    num = exp(-((x-mean)**2)/(2*var))
    ret = num/float(denom)
    #print "Gaussian of x=%s (m=%s, var=%s) is %s" % (x, mean, var, ret)
    return ret

def gaussian2d(x, y, sigma):
    """The symmetric 2d unit gaussian function.

    .. math:: \\frac{1}{2\\pi\\sigma^2}e^{\\frac{x^2 + y^2}{2\\sigma^2}}
    """
    from math import exp, pi
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
    from random import shuffle
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
    from math import exp
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


# STRING AND PATH UTILS
#: Mapping from filename extensions to canonical 3-letter form
IMAGE_EXTENSIONS = {
                    'jpg': 'jpg', 'jpeg': 'jpg', 'jp2': 'jpg',
                    'gif': 'gif',
                    'png': 'png',
                    'bmp': 'bmp',
                    'tiff': 'tif', 'tif': 'tif',
                    'ppm': 'ppm',
                    'pgm': 'pgm',
                   }

def getListAsStr(lst, sep=',', fmt='%s'):
    """Returns a list of values as a string using the given separator and formatter"""
    return sep.join([fmt % (x,) for x in lst])

def utf(obj):
    """Converts the given object to a unicode string.
    Handles strings, unicodes, and other things as well."""
    if isinstance(obj, unicode):
        s = obj
    elif isinstance(obj, str):
        s = unicode(obj, 'utf-8')
    else:
        s = unicode(str(obj), 'utf-8')
    return s

def numseqstr(seq, sep=',', fmt='%0.2f'):
    """Takes a number sequence or single number and prints it using the given separator and format."""
    def gets(s):
        """Returns a number formatting string using fmt if a number, else str() version"""
        try: # number
            return fmt % s
        except TypeError: # non-number
            return str(s)
    try:
        # sequence
        seq = [s for s in seq]
        return sep.join(gets(s) for s in seq)
    except TypeError:
        return gets(seq)

def urlquote(url):
    """Quotes a url for submission through forms, etc."""
    from urllib import quote
    if isinstance(url, unicode):
        url = url.encode('utf-8', 'replace')
    if '//' in url:
        prot, rest = url.strip().split('//',1)
        return prot + '//' + quote(rest)
    else:
        return quote(url.strip())

def urlunquote(url):
    """Unquotes a url for submission through forms, etc."""
    from urllib import unquote
    if '//' in url:
        prot, rest = url.strip().split('//',1)
        return prot + '//' + unquote(rest)
    else:
        return unquote(url.strip())

def blockindent(s, indent='\t', initial='\t'):
    """Block-indents a string by adding the given indent to the start of every line.
    The default indent is 1 tab.  You can also provide an initial indent to
    apply at the beginning of the string (defaults to 1 tab).
    """
    if initial:
        s = initial + s
    s = s.replace('\n', '\n'+indent)
    return s

def replaceTill(s, anchor, base):
    """Replaces the beginning of this string (until anchor) with base"""
    n = s.index(anchor)
    return base + s[n:]

def convertBasePath(objs, anchor, base):
    """Converts the path of this set of objects so that everything before 'anchor' gets converted to 'base'"""
    if not objs: return
    oldpaths = [o.path for o in objs]
    #print "Basename was %s and the first object originally had path %s" % (base, objs[0].path)
    for obj in objs:
        obj.path = replaceTill(obj.path, anchor, base)
    print >>sys.stderr, "Converted base paths to '%s'" % (base)
    return oldpaths

def getKWArgsFromArgs(args=None):
    """Returns a dictionary of keys and values extracted from strings.
    This splits each arg using '=', and then evals the second part.
    WARNING: This is very unsafe, so use at your own-risk!"""
    kw = {}
    if not args: return kw
    for a in args:
        k, v = a.split('=', 1)
        try:
            kw[k] = eval(v)
        except (NameError, SyntaxError): kw[k] = v
    return kw

def numformat(num, fmt='%d'):
    """Formats a number nicely, with commas.
    You can optionally give a custom format, e.g. for floats"""
    if isinstance(num, basestring):
        if '%d' in num:
            try:
                num = int(num)
            except ValueError: return num
        else:
            try:
                num = float(num)
            except ValueError: return num
    import locale
    locale.setlocale(locale.LC_ALL, '')
    return locale.format(fmt, num, grouping=1)

def intOrNone(v, default=0, exctype=Exception):
    """Returns the int value of the given value, or default (which is normally 0) on error.
    Catches exceptions of the given exctype (Exception by default)"""
    try:
        return int(v)
    except exctype:
        return default

def floatOrNone(v, default=0.0, exctype=Exception):
    """Returns the float value of the given value, or default (which is normally 0.0) on error.
    Catches exceptions of the given exctype (Exception by default)"""
    try:
        return float(v)
    except exctype:
        return default

def makesafe(s):
    """Makes the given string "safe" by replacing spaces and lower-casing (less aggressive)"""
    def rep(c):
        if c.isalnum(): return c.lower()
        else: return '_'
    ret = ''.join([rep(c) for c in s])
    return ret

def safestr(s, validchars=string.ascii_letters + string.digits+'-.', rep='_', collapse=1):
    """Makes the given string very safe (super aggressive).
    Limits characters to be in the given set of validchars, and lowercases them.
    Any illegal character is replaced by the given 'rep'.
    If collapse is true, then also goes through and collapses all instances of
    the rep character so there's only 1 at most."""
    # make a dictionary mapping which lowercases
    d = dict([(c,c.lower()) for c in validchars])
    out = []
    if type(s) == str:
        s = unicode(s, 'utf-8')
    for c in s.encode('utf-8', 'replace'):
        # get the appropriate map, or the replacement character
        c = d.get(c, rep)
        # if we want to collapse, don't put copies of the rep character on the output
        if collapse and out and c == rep and out[-1] == rep: continue
        out.append(c)
    outs = ''.join(out)
    return outs

def stringize(opts, safefunc=safestr, delim='_'):
    """Stringizes a set of opts by join()-ing safe versions of keys and values in opts.
    opts must be a list of pairs, NOT a dict!
    Both the key and value are run through the given safefunc, defaulting to safestr()."""
    s = delim.join('%s-%s' % (safefunc(k),safefunc(v)) for k, v in opts)
    return s

def randstr(nchars=10):
    """Creates a random string identifier with the given number of chars"""
    import string, random
    chars = string.ascii_letters + string.digits
    ret = ''.join([random.choice(chars) for i in range(nchars)])
    return ret

def host2dirname(url, safefunc=safestr, collapse=1, delport=1):
    """Converts the hostname from a url to a single directory name.
    This gets the hostname from the url, optionally collapses some
    elements from the beginning, and then makes it safe using the given function.
    If delport is true, then also removes port information from the host.

    The collapsing includes:
        - removing www from the beginning
        - keeping the last 2 elements
        - if the 2nd-to-last element is special (.com., .ac., etc.), then keeps last 3 elements.

    This function assumes you're giving it a URL. However, it'll
    check for a // somewhere in the url, and if not found, then assumes it's a host
    """
    from urlparse import urlparse
    h = urlparse(url).netloc if '//' in url else url
    if delport and ':' in h:
        h = h.split(':')[0]
    if collapse:
        # only collapse if it's not a numeric ip
        if [c for c in h if c not in '1234567890.']:
            # first get rid of initial www
            if h.startswith('www.'):
                h = h.replace('www.', '', 1)
            # now figure out how many elements to shorten down to
            minlen = 2
            # these are special cases for other countries, like .com.br, or .ac.uk, etc.
            prefixes = 'com co gov edu org ac net ne go gob unam govt sapo academic'.split()
            if len(h.split('.')) > 2:
                if h.split('.')[-2] in prefixes:
                    minlen = 3
            # chop off all beginning elements, maintaining minlen
            h = '.'.join(h.split('.')[-minlen:])
    h = '.'.join(safefunc(u) for u in h.split('.'))
    return h

def shortenurl(url, maxwidth=50):
    """Shortens the url reasonably"""
    if len(url) <= maxwidth: return url
    url = url.replace('http://','')
    if len(url) <= maxwidth: return url
    if url.startswith('www'):
        url = url[3:]
    if len(url) <= maxwidth: return url
    els = url.split('/')
    els[1] = '...'
    u = '/'.join(els)
    if len(u) <= maxwidth: return u
    while len(els) > 3:
        del els[2]
        u = '/'.join(els)
        if len(u) <= maxwidth: break
    return u

def strsim(a, b, weights={'difflib':1.0, 'longest_ratio':5.0, 'matching_ratio':6.0, 'exact':5.0}):
    """Returns the similarity between two strings as a float from 0-1 (higher=similar).
    Computes a few different measures and combines them, for optimal matching.
    The methods are linearly weighted using the dictionary 'weights'. It contains:
        'difflib': The ratio given by difflib's SequenceMatcher.ratio()
        'longest_ratio': Ratio of longest match over min(len)
        'matching_ratio': Ratio of sum(len(block)/minlen for block in get_matching_blocks())
        'exact': Scores for exact matches at beginning or end (very high), or in the middle (less high)
    Some default weights are defined for optimal file-renaming performance.
    You can tweak weights as needed.
    """
    from difflib import SequenceMatcher
    minlen = float(min(len(a), len(b)))
    maxlen = float(max(len(a), len(b)))
    sm = SequenceMatcher(None, a, b)
    # compute the ratio given by difflib
    r0 = sm.ratio()
    # compute the ratio of the longest match over the minlen
    lm = sm.find_longest_match(0, len(a), 0, len(b))
    r1 = lm[2] / minlen
    # compute ratio of sum of all matching block lengths over the minlen
    totmatch = sum(m[2] for m in sm.get_matching_blocks())
    r2 = totmatch / minlen
    # add a score if we have an exact substring match
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        r3 = 1.0
    elif a in b or b in a:
        r3 = minlen/maxlen
    else:
        r3 = 0.0
    # add weighted combinations of the different vars
    vars = {'difflib': r0, 'longest_ratio': r1, 'matching_ratio': r2, 'exact': r3}
    total = 0.0
    ret = 0.0
    for k, w in weights.items():
        ret += vars[k] * w
        total += w
    ret /= total
    DEBUG = 0
    if DEBUG:
        print >>sys.stderr, a, b, '\n\t', r0, r1, r2, r3, ret
    return ret

def matrixprint(m, width=None, fillchar=' ', sep=' ', rowsep='\n', fmt='%0.2f'):
    """Returns a string representation of a matrix of strings using the given separator.
    Each string is center()-ed using the given width and fillchar.
    If width is None (default), then uses the maximum width of the strings+2"""
    if not width:
        width = max(len(numseqstr(s, fmt=fmt)) for s in flatten(m))
    ret = []
    for row in m:
        # convert Nones to empty strings
        row = [(numseqstr(s, fmt=fmt) if s else '') for s in row]
        row = sep.join(s.center(width, fillchar) for s in row)
        ret.append(row)
    return rowsep.join(ret)

def tempfname(**kw):
    """Creates a temporary filename using mkstemp with nice names.
    You can pass in:
        prefix: [default: int(time.time()*1000)]
        suffix: [default: random 10-char string.ascii_letters]

    Note that this does introduce a race condition, but it's usually okay.
    """
    from random import choice
    import tempfile
    kw.setdefault('prefix', '%d_' % (int(time.time()*1000)))
    kw.setdefault('suffix', '%s' % (''.join(choice(string.ascii_letters) for i in range(10))))
    tempf, fname = tempfile.mkstemp(**kw)
    os.close(tempf)
    return fname

def cleanimgext(fname):
    """Returns the image filename with the extension fixed.
    If it doesn't seem like a valid image name, then only lowercases"""
    try:
        fname, ext = fname.rsplit('.', 1)
    except ValueError: return fname
    ext = ext.lower().split('?')[0].split('&')[0]
    return fname + '.' + IMAGE_EXTENSIONS.get(ext, ext)

def url2fnamefmtdict(url):
    """Converts a url into a dictionary of strings to use for creating a filename.
    This includes:
        scheme, netloc, path, params, query, fragment, username, password, hostname, port - from urlparse
        basefname - the last part of the url, with extension
        basename - the last part of the url, without extension
        ext - the extension
        q-%(query param)s - the value of the given query param
        path%d - the d'th element in the path (stripped and split by '/')
        pathn%d - the -d'th element in the path (stripped and split by '/')
        hel%d - the d'th element in the hostname (split by '.')
        heln%d - the -d'th element in the hostname (split by '.')
        rand%d - random safe string of length d, upto 32
        time - current time, in milliseconds
        md5url - md5 of the url
        md5path - md5 of the path
        md5basefname - md5 of the basefname
    Returns a dict of strings.
    """
    from urlparse import urlparse, parse_qs
    import hashlib
    p = urlparse(url)
    ret = dict(url=url)
    for k in 'scheme netloc path params query fragment username password hostname port'.split():
        ret[k] = getattr(p, k)
    ret['basefname'] = os.path.basename(p.path)
    els = ret['basefname'].rsplit('.', 1)
    ret['basename'] = els[0]
    ret['ext'] = els[1] if len(els) > 1 else ''
    pels = p.path.strip('/').split('/')
    for i, pel in enumerate(pels):
        ret['path%d' % i] = ret['pathn%d' % (len(pels)-i)] = pel
    if p.hostname:
        hels = p.hostname.split('.')
        for i, hel in enumerate(hels):
            ret['hel%d' % i] = ret['heln%d' % (len(hels)-i)] = hel
    rs = randstr(nchars=32)
    for i in range(len(rs)):
        ret['rand%d' % (i)] = rs[:i+1]
    ret['time'] = int(time.time()*1000)
    for k in 'url path basefname'.split():
        ret['md5'+k] = hashlib.md5(utf(ret[k]).encode('utf-8')).hexdigest()
    qels = parse_qs(ret['query'], keep_blank_values=1)
    for k, v in qels.items():
        ret['q-'+k] = v
    return ret

def url2fname(url, basedir='', maxlen=250, safefunc=safestr):
    """Returns a SAFE fname to use for downloading the given url. This means:
        Length considerations:
            - Filenames/dirnames are not too long (set by maxlen)
                - Note that names are cut at the end, so you may end up with dupes
                - Also, the final fname len might be up to maxlen + len('.') + len(ext)

        Illegal characters (handled using safefunc, which is safestr by default):
            - No unicode
            - No weird characters like | , @, :, etc.

        Path-specific character handling:
            - No pathname elements start with '.'
            - Filenames end with .ext (lowercase)
            - Image extensions are normalized by type
            - If the fname had a valid extension with the period, we fix it
            - If any path element is empty (including fname without ext), then 'temp_%06d' % (rand) is used.
    """
    from urllib import url2pathname
    import re
    from random import randint
    fname = url2pathname(os.path.basename(url))
    try:
        fname, ext = fname.rsplit('.', 1)
    except ValueError:
        # see if the fname happens to end with an extension but without the period
        ext = ''
        for test, realext in IMAGE_EXTENSIONS.iteritems():
            if fname.lower().endswith(test):
                fname = fname.rsplit(test, 1)[0]
                ext = realext
                break

    path = os.path.join(basedir.encode('utf-8', 'replace'), fname.encode('utf-8', 'replace'))
    def fix(s):
        """Fixes a given string"""
        #print '  Got input: %s' % (s,),
        s = safefunc(s)[:maxlen]
        while s.startswith('.'):
            s = s[1:]
        if not s:
            s = 'temp_%06d' % (randint(0, 999999))
        #print ' and returning: %s' % (s,)
        return s

    path = u'/'.join(fix(el) for el in path.split('/'))
    if ext:
        path += safefunc(cleanimgext('.'+ext))
    path = re.sub(r'\.+', '.', path)
    return path

def dlFileIfNeeded(f, repfunc=lambda f:replaceTill(f, '/db/', 'http://leaf.cs.columbia.edu')):
    """Downloads a file if needed"""
    if os.path.exists(f):
        return f
    url = repfunc(urllib.quote(f))
    # create the parent directories if needed
    try: os.makedirs(os.path.dirname(f))
    except OSError: pass
    # download the file
    try:
        fname, headers = urllib.urlretrieve(url, f)
        assert fname == f
        return f
    except IOError, e:
        #print "Error: %s, %s, %s" % (url, f, e)
        return None

def downloadfile(url, outdir='.', outf=sys.stderr, delay=1):
    """Downloads a file from the given url.
    Returns the local file path, or None if doesn't exist/couldn't download.
    This function tries to be smart about things, especially multi-threading issues.
    If the url is a local path, then simply returns that"""
    from urllib import urlretrieve
    from urlparse import urlparse
    if url.startswith('http'):
        path = urlparse(url).path[1:] # strip the leading / from the path
        outpath = os.path.join(outdir, path)
        if not os.path.exists(outpath):
            # check for temp file existence, so multiple threads don't all try to download at once
            temp = outpath + '_dl_temp_%d' % (int(time.time())//100) # the temp file is accurate to the 100s of secs
            print >>outf, 'Trying to download path %s to %s via temp name %s' % (url, outpath, os.path.basename(temp))
            while os.path.exists(temp) and not os.path.exists(outpath):
                print >>outf, '  Detected temp file %s, just sleeping for %s' % (temp, delay)
                time.sleep(delay)
                temp = outpath + '_dl_temp_%d' % (int(time.time())//100) # the temp file is accurate to the 100s of secs
            # another check for outpath existence
            if not os.path.exists(outpath):
                # if we're here, then we need to download the file
                try:
                    os.makedirs(os.path.dirname(temp))
                except OSError: pass
                t1 = time.time()
                temp, headers = urlretrieve(url, temp)
                elapsed = time.time()-t1
                # rename it atomically to the right name
                try:
                    os.rename(temp, outpath)
                except Exception: pass
                s = os.stat(outpath).st_size
                print >>outf, 'Downloaded %s to %s in %0.3fs (%s bytes, %0.1f bytes/sec)' % (url, outpath, elapsed, s, s/elapsed)
                try:
                    os.remove(outtemp)
                except Exception: pass
    else:
        outpath = url
    # at this point, we've downloaded the file if we needed to
    if os.path.exists(outpath):
        return outpath
    return None

def checkForFiles(fnames, progress=None):
    """Checks to see if the given files exist, otherwise downloads them"""
    ret = []
    for i, f in enumerate(fnames):
        if progress: progress('  Downloading %d of %d: %s...' % (i, len(fnames), f))
        f = dlFileIfNeeded(f)
        if f: ret.append(f)
    if progress: progress("Done downloading files\n")
    return ret

def getArg(seq, index, default=None, func=lambda x: x):
    """Returns func(seq[index]), or 'default' if it's an invalid index"""
    try:
        return func(seq[index])
    except IndexError:
        return default

def cleanDirTree(p, ntimes=-1):
    """Cleans the given directory tree by deleting all directory trees with no files"""
    done = 0
    iters = 0
    while not done:
        ndel = 0
        for root, dirs, files in os.walk(p, topdown=0):
            if not files and not dirs:
                os.rmdir(root)
                ndel +=1
        iters += 1
        if iters == ntimes: return # we've done this many repetitions
        if ntimes < 0 and ndel == 0: return # we've done all the deletions we can

def specialize(v):
    """Specializes a string value into an int or float or bool"""
    if not isinstance(v, basestring): return v
    if v.strip() == 'True': return True
    if v.strip() == 'False': return False
    try:
        # see it's an int...
        v = int(v)
    except (ValueError,TypeError):
        # maybe it's a float...
        try:
            v = float(v)
        except (ValueError,TypeError): pass
    return v

def specializeDict(d):
    """Takes a dictionary and for each value, sees if it can be cast to an int or a float"""
    for k, v in d.iteritems():
        d[k] = specialize(v)
    return d

def readNLines(f):
    """Reads a line which contains the number of future lines to read, followed by that many lines.
    Returns as a list of stripped strings"""
    try:
        n = int(f.readline().strip())
    except ValueError: return []
    ret = [f.readline().strip() for i in xrange(n)]
    ret = [l for l in ret if l]
    return ret

def detectdelimiter(fname):
    """Detects the delimiter of the given datafile.
    Returns one of ',', '\t', ' ', or None on error."""
    dlms = ['\t', ',', ' ']
    f = open(fname)
    header = f.readline().strip()
    # keep reading lines until we have a non-header line
    while 1:
        curline = f.readline().strip()
        if not curline.startswith('#'): break
        header = curline
    # figure out the format by comparing number of delimiters
    #print 'Got header line %s' % (header)
    for dlm in dlms:
        hnum = len(header.split(dlm))
        cnum = len(curline.split(dlm))
        #print 'For dlm "%s" got hnum %s, cnum %s' % (dlm, hnum, cnum)
        if 1 < cnum <= hnum <= cnum+1: # the +1 because there's an extra delimiter due to the '#'
            return dlm
    return None

def opts2dict(opts):
    """Converts options returned from an OptionParser into a dict"""
    ret = {}
    for k in dir(opts):
        if callable(getattr(opts, k)): continue
        if k.startswith('_'): continue
        ret[k] = getattr(opts, k)
    return ret

def openVersionedFile(fname, mode='wb'):
    """Opens a file for writing with the given name.
    If the file already exists, it is renamed to 'fname-%Y%m%d-%H%M%S'"""
    if os.path.exists(fname):
        newname = '%s-%s' % (fname, getTimestamp(fmt='%Y%m%d-%H%M%S'))
        os.rename(fname, newname)
    return open(fname, mode)

def saveandrename(fname, func, retfile=1, infork=0, mode='wb'):
    """Opens a new file with a tempfilename, runs the given func, then renames it to the given fname.
    The func is run with the file if retfile=1, else with the tmpfname.
    Creates parent dirs."""
    if infork:
        pid = os.fork()
        if pid != 0: return # the parent process just returns out
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    tmpfname = fname+'.tmp-%d' % (int(time.time()*1000))
    if retfile:
        f = open(tmpfname, mode)
        func(f)
        f.close()
    else:
        func(tmpfname)
    os.rename(tmpfname, fname)
    if infork:
        os._exit(0)


def savejson(data, fname, delay=60, lastsave=[0], savebackups=1, jsonmodule=None):
    """Safe save function, which optionally makes backups if elapsed time has exceed delay.
    This is "safe" in that it writes to a temp file, then atomically renames to target fname when done.
    It uses a closure on lastsave[0] to determine when the last time we made a backup was.
    """
    if not jsonmodule:
        try:
            import simplejson as json
        except Exception:
            import json
    else:
        json = jsonmodule
    from shutil import copy2
    t1 = time.time()
    if time.time()-lastsave[0] > delay:
        try:
            if savebackups:
                copy2(fname, '%s.bak-%d' % (fname, time.time()))
            lastsave[0] = time.time()
        except Exception:
            pass
    tmpname = fname + '.tmp-%d' % (time.time())
    kw = {}
    if json.__name__ == 'ujson':
        kw = dict(encode_html_chars=False, ensure_ascii=False)
    else:
        kw = dict(sort_keys=1, indent=2)
    try:
        os.makedirs(os.path.dirname(tmpname))
    except OSError: pass
    json.dump(data, open(tmpname, 'wb'), **kw)
    os.rename(tmpname, fname)

def readLinesOfVals(fname, convfunc=lambda vals, fields: vals, prefunc=lambda l: l, func=lambda d: d, dlm=' ', offset=0, maxlines=-1, onlyheaders=0):
    """Reads data values from the given fname.
    Implementation for readListOfVals and readDictOfVals.

    Input parameters:
           fname - the filename to read data from
         prefunc - if given, then used to filter and remap values (prior to convfunc). i.e.:
                       lines = (prefunc(l) for l in lines if prefunc(l))
                   but note that it doesn't actually call the prefunc twice
        convfunc - function which takes list of values from a line and returns a val (list or dict)
            func - if given, then used to filter and remap values (after convfunc). i.e.:
                        f = func(f)
                        if not f: continue
             dlm - IGNORED. we do automatic delimiter checking now
          offset - the data row offset number
        maxlines - if positive, then only that many datalines are read
     onlyheaders - if true, then we only read the headers and return (fields, dlm)

    Returns (faces, fields), where faces is a list of data items (each corresponding to a row),
    and fields is a list of the data fields.

    The datafile can have any number of comments at the top (lines starting with '#'),
    but none once the data starts. At the top of the file, the last non-data line should have the
    fields in it. The possible delimiters for data (and in the fields row) are '\t', ' ', or ','.
    Note that in the last case, the fields line should not start with #, but in all others, it should.

    This function tries to be efficient about memory by using generators,
    but returns a list in the end, to prevent confusion.
    """
    # figure out the delimiter
    dlm = detectdelimiter(fname)
    assert dlm is not None
    # figure out how many lines of headers there are (the last one is assumed to have the fields)
    skipheaders = -1 # -1 because we expect at least one for the fields
    for i, line in enumerate(open(fname)):
        if not (dlm == ',' and i < 1) and not line.strip().startswith('#'): break
        skipheaders += 1
    # now actually read the file
    f = open(fname)
    firstline = f.readline().strip()
    while skipheaders > 0:
        firstline = f.readline().strip()
        skipheaders -= 1
    if ',' not in dlm: # , delimiters means a CSV file, which doesn't use the #
        firstline = firstline.split('#'+dlm, 1)[1]
    fields = firstline.split(dlm)
    if onlyheaders: return (fields, dlm)
    # read lines
    lines = (l.strip() for i, l in enumerate(f) if l.strip() and (maxlines < 0 or i < maxlines) and (i >= offset))
    # apply prefunc
    lines = (prefunc(l) for l in lines)
    # filter by prefunc
    lines = (l for l in lines if l)
    # convert to list, specialize
    faces = (convfunc(l.strip().split(dlm), fields) for l in lines)
    # apply function
    faces = (func(f) for f in faces)
    # filter, and start computations
    faces = [f for f in faces if f]
    return (faces, fields)

def readListOfVals(fname, dospecialize=1, **kw):
    """Reads a list of values from the given fname.
    If prefunc is given, then it's used as a prefilter on lines (before specialize()).
    If func is given, then it's used as both a filtering and slicing function:
        f = func(f)
        if not f: continue
    This function tries to be efficient about memory by using generators, but
    returns a list in the end, to prevent confusion.
    Returns (faces, fields)"""
    if dospecialize:
        convfunc = lambda vals, fields: map(specialize, vals)
    else:
        convfunc = lambda vals, fields: vals
    return readLinesOfVals(fname, convfunc=convfunc, **kw)

def readDictOfVals(fname, specialize=1, **kw):
    """Reads a dictionary of values from the given fname.
    If prefunc is given, then it's used as a prefilter on lines (before dict()).
    If func is given, then it's used as both a filtering and slicing function:
        f = func(f)
        if not f: continue
    This function tries to be efficient about memory by using generators, but
    returns a list in the end, to prevent confusion.
    Returns (faces, fields)"""
    if specialize:
        convfunc = lambda vals,fields: specializeDict(dict(zip(fields, vals)))
    else:
        convfunc = lambda vals,fields: dict(zip(fields,vals))
    return readLinesOfVals(fname, convfunc=convfunc, **kw)

def writeLinesOfVals(linevals, fields, fname, dlm=' ', **kw):
    """Implementation function for writeDictOfVals and writeListOfVals"""
    if fname == '-':
        outf = sys.stdout
    else:
        outf = open(fname, 'w')
    if dlm != ',':
        print >>outf, '#' + dlm,
    print >>outf, dlm.join(fields)
    for vals in linevals:
        print >>outf, dlm.join(vals)

def writeListOfVals(faces, fields, fname, **kw):
    """Prints data in 'faces' (assumed to be lists) using the fields given"""
    linevals = ((str(v) for v in f) for f in faces)
    return writeLinesOfVals(linevals, fields, fname, **kw)

def writeDictOfVals(faces, fields, fname, errfunc=lambda field: 'ERR', **kw):
    """Prints data in 'faces' using the fields given"""
    linevals = ((str(f.get(field, errfunc(field))) for field in fields) for f in faces)
    return writeLinesOfVals(linevals, fields, fname, **kw)

import csv
class CSVUnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    Taken from:
    http://stackoverflow.com/questions/15960044/python-write-unicode-to-csv-using-unicodewriter
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        import cStringIO
        import codecs
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def pprintJson(inf, outf=sys.stdout):
    """Pretty-prints a json file"""
    try:
        import simplejson as json
    except ImportError:
        import json
    s = inf.read()
    n1 = s.index('{')
    n2 = s.index(';')-1
    f = json.loads(s[n1:n2])
    from pprint import pprint
    pprint(f, outf)

def extractToDir(fname, dir):
    """Extracts the given file (zip, tgz, gz, tar.gz, tar) to the given dir.
    Returns the retcode"""
    from subprocess import call, PIPE
    ends = '.zip .tgz .tar.gz .tar'.split()
    try:
        type = [e for e in ends if fname.lower().endswith(e)][0]
    except IndexError:
        raise TypeError

    try:
        os.makedirs(dir)
    except OSError: pass

    if type == '.zip': # unzip
        args = ['unzip', '-qq', '-d', dir, fname]
    elif type in '.tgz .tar.gz .tar'.split(): # tar xf
        args = ['tar', '-C', dir, '-xf', fname]
    ret = call(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    return ret

def filechecksum(f, blocksize=2**20, hashfunc='md5'):
    """Fixed-memory checksum on file, by incrementally updating hash over file.
    The given file can be a string (filename) or a file-like object itself.
    Reads file in chunks of the given size, to handle large files.
    The hashfunc is either given as a string (looked up in hashlib) or should be a
    constructor that initializes with no parameters, and returns an object
    with update() and hexdigest() functions in it.
    """
    import hashlib
    h = hashlib.new(hashfunc) if isinstance(hashfunc, basestring) else hashfunc()
    if isinstance(f, basestring):
        f = open(f)
    while 1:
        data = f.read(blocksize)
        if not data: break
        h.update(data)
    return h.hexdigest()

class FileLockException(Exception): pass

class FileLock(object):
    """A context-manager wrapper on a fcntl.flock()"""
    def __init__(self, f, shared=0):
        """Initializes a lock on the given file or file descriptor or filename.
        If shared is 0 [default], it's an exclusive lock (LOCK_EX).
        Else it's a shared lock (LOCK_SH).
        """
        if isinstance(f, basestring):
            f = open(f)
        self.f = f
        self.shared = shared
        self.locked = 0

    def acquire(self):
        """Acquire this lock, if possible.
        If not, raise a FileLockException."""
        import fcntl
        if self.locked: return
        try:
            locktype = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
            fcntl.flock(self.f, locktype|fcntl.LOCK_NB)
            self.locked = 1
        except IOError:
            raise FileLockException

    def release(self):
        """Releases our lock"""
        import fcntl
        if self.locked:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.locked = 0

    def __enter__(self):
        """Acquire a lock in a 'with' statement"""
        self.acquire()
        return self

    def __exit__(self, type, value, trackback):
        """Release a lock at the end of the 'with' statement"""
        self.release()


# OTHER/MISC
def spawnWorkers(num, target, name=None, args=(), kwargs={}, daemon=1, interval=0):
    """Spawns the given number of workers, by default daemon, and returns a list of them.
    'interval' determines the time delay between each launching"""
    from threading import Thread
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
    from urllib import urlopen
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
    import smtplib
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
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

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

def httpresponse(url):
    """Returns the http response code (code, reason) associated with the given url"""
    import httplib
    from urlparse import urlparse
    p = urlparse(url)
    conn = httplib.HTTPConnection(p.netloc)
    rest = url.split(p.netloc, 1)[-1]
    conn.request('GET', rest)
    r = conn.getresponse()
    return r.status, r.reason

def _memtest():
    """Tests the various mem utils"""
    print procmem()
    print totalmem()
    m = MemUsage()
    print 'Created m'
    print m.usage()
    print m.delta()
    a = range(1000000)
    m.add('after a')
    print m.usage()
    print m.delta()
    b = range(2000000)
    m.add('after b')
    print m.usage()
    print m.delta()
    del b
    m.add('after del')
    print m.usage()
    print m.delta()
    print m.usage('start')
    print m.delta('after b')
    for i in m:
        print i
    print m['after a']

def getConsoleSize():
    """Returns the (width, height) of the current console window.
    If there is some error, returns (-1, -1).
    Only tested on linux.
    """
    from subprocess import Popen, PIPE
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
    import traceback
    try:
        import simplejson as json
    except Exception:
        import json
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
        except Exception, e:
            print >>sys.stderr, 'Ran into error of type %s: %s' % (type(e), e)
            traceback.print_exc()
        try:
            print out
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
        print 'Usage: python %s <%s> [<args>...]' % (sys.argv[0], '|'.join(tasks))
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
        print dir

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
