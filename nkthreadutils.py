#!/usr/bin/env python
"""Lots of small python thread-related utilities, written by Neeraj Kumar.

This module contains various utilities for multithreaded programs.
This also includes various classes and functions to deal with queues,
events, and synchronized collections.

Licensed under the 3-clause BSD License:

Copyright (c) 2010, Neeraj Kumar (neerajkumar.org)
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
"""

import os, sys, time, Queue
from threading import Thread, Event, Lock
from Queue import Empty
from StringIO import StringIO
import urllib

class CustomURLopener(urllib.FancyURLopener):
    """Custom url opener that defines a new user-agent.
    Needed so that sites don't block us as a crawler."""
    version = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5"

    def prompt_user_passwd(host, realm):
        """Custom user-password func for downloading, to make sure that we don't block"""
        return ('', '')

urllib._urlopener = CustomURLopener()


# THREAD UTILITIES
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

def parmap(func, args, **kw):
    """Parallel map. Runs multiple threads and gathers results.
    You can give the following keywords (with [defaults]):
        nworkers: Number of worker threads [5]
        interval: Delay between spawning each worker [0]

    """
    from Queue import Queue
    from threading import Lock
    inq = Queue()
    retlock = Lock()
    todo = tuple(args)
    ret = [None] * len(todo)
    # create the callback and spawn the workers
    def fn():
        while 1:
            i, el = inq.get()
            #print 'Going to call with %d, %s' % (i, el,)
            out = func(*el)
            with retlock:
                ret[i] = out
            inq.task_done()

    threads = spawnWorkers(kw.get('nworkers', 5), fn, interval=kw.get('interval', 0))

    # put arguments into the input queue
    for i, el in enumerate(todo):
        inq.put((i, el))
    inq.join()
    return ret

def dlmany(urls, fnames, nprocs=10, callback=None, validfunc=os.path.exists, checkexists=0, timeout=None):
    """Downloads many images simultaneously, `nprocs` at a time.
    Handles many error cases, eg, downloads to a tempfname and atomically
    renames on completion, or skips invalid urls or paths.

    You can optionally supply a callback function, which is called for each
    output with (index, url, fname), where the index corresponds to the input.
    If there was a problem downloading the image, the fname will be None.
    Returns a list of (url, fname) pairs, in order.

    This also checks for validity on the output by calling `validfunc(fname)`.
    That should return true for valid files. All exceptions are caught and
    treated as invalid files. Invalid outputs are deleted, and return None for
    the fname.

    If you set `checkexists` to 1, then first checks each input to see if it
    already exists.  It does this by running the `validfunc` on each input
    filename. If it exists, it is not re-downloaded. However, it is still
    included in the outputs, and the callback is also called on it.

    You can also change the user-agent by redefining `urllib._urlopener`.
    By default, this module uses an older firefox.

    Note that websites that take a long time to return might hold up this
    function for a while, because python's default web timeout is pretty large.
    So, you can optionally pass in a `timeout` in seconds. However, note that
    python sets this GLOBALLY for all sockets!
    """
    from urllib import urlretrieve
    from Queue import Queue
    import socket
    assert len(urls) == len(fnames)
    if not urls: return []
    if timeout:
        socket.setdefaulttimeout(timeout)
    # init vars
    q = Queue()
    outq = Queue()
    if not validfunc:
        validfunc = os.path.exists
    def dlproc():
        """This is the main loop for the downloader, which reads and writes from queues."""
        while 1:
            i, u, f = q.get()
            if not u or not f: # invalid input
                outq.put((i, u, None))
                continue
            try:
                os.makedirs(os.path.dirname(f))
            except OSError: pass
            try:
                # download to tempfile and atomically rename
                tmpfname = f+'.tmp-%d' % (int(time.time()*1000))
                tmpfname, junk = urlretrieve(u, tmpfname)
                os.rename(tmpfname, f)
                # check validity
                if validfunc(f):
                    outq.put((i, u, f))
                else:
                    #os.remove(f)
                    raise ValueError
            except Exception, e:
                #print >>sys.stderr, 'Exception on %d: %s -> %s: %s' % (i, u, f, e)
                outq.put((i,u,None))

    # spawn download threads
    threads = spawnWorkers(nprocs, dlproc, interval=0)
    # add to download queue after optionally checking for existence
    ret = [None] * len(urls)
    todo = len(urls)
    for i, (u, f) in enumerate(zip(urls, fnames)):
        if checkexists:
            try:
                if validfunc(f):
                    callback(i, u, f)
                    ret[i] = (u,f)
                    todo -= 1
                else:
                    raise ValueError
            except Exception:
                q.put((i, u, f))
        else:
            q.put((i, u, f))
    # collect outputs
    while todo > 0:
        i, u, f = outq.get()
        if callback:
            callback(i, u, f)
        ret[i] = (u, f)
        todo -= 1
    return ret

def inthread(target, args=(), kwargs={}, daemon=1, procs=[]):
    """Runs the given function in the background and adds it to the given list 'procs'.
    Returns the spawn process as well.
    Uses spawnWorkers as underlying implementation."""
    threads = spawnWorkers(1, target=target, args=args, kwargs=kwargs, daemon=daemon, interval=0)
    procs.append(threads[0])
    return procs[-1]

def urlopenInThread(url, callback=None, *args, **kw):
    """Does urllib.urlopen in a thread, with a callback function called once it completes.
    The callback is called with (url, urlopen(url, *args, **kw))
    This is useful for checking timeouts.
    This function returns the thread running the urlopen() function."""
    def func(url=url, callback=callback, *args, **kw):
        from urllib import urlopen
        try:
            ret = urlopen(url, *args, **kw)
            callback(url, ret)
            return ret
        except Exception: pass

    t = spawnWorkers(1, func, interval=0)[0]
    return t

def backgroundFunc(target, args=(), kwargs={}, callback=None, num=1, daemon=1):
    """Calls a function in the background with given args and kwargs.
    If callback is given, then it's called with the result.

    By default, this is a single daemon thread.
    You can set num > 1 to run the function multiple times (with same args).
    The same callback will be called each time.
    You can also set daemon to 0 to make this a main thread.
    Returns the started threads."""
    def bgfunc(target=target, args=args, kwargs=kwargs, callback=callback):
        """Wrapper for background function execution"""
        ret = target(*args, **kwargs)
        if callback:
            callback(ret)
    workers = spawnWorkers(num, bgfunc, daemon=daemon, interval=0)
    return workers

class DecoratedLock(object):
    """A simple wrapper on a lock that performs some callbacks on acquire() and release()"""
    def __init__(self, acquire_callback=None, release_callback=None, lock_class=Lock):
        """Create this decorated lock with the given callbacks.
        By default, the lock class is threading.Lock, but you can pass something else."""
        self._lock = lock_class()
        self.acquire_callback = acquire_callback
        self.release_callback = release_callback

    def acquire(self, *args, **kw):
        """Acquire this lock, and then call the callback"""
        self._lock.acquire(*args, **kw)
        if self.acquire_callback:
            self.acquire_callback()

    def release(self, *args, **kw):
        """Release this lock, and then call the callback"""
        self._lock.release(*args, **kw)
        if self.release_callback:
            self.release_callback()

    def __enter__(self):
        """Acquire this lock, with callback"""
        self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        """Release this lock, with callback"""
        self.release()




# NETWORK UTILITIES
def checkForValidURL(url, timeout=1):
    """Checks the given url to see if it's valid, within the given timeout"""
    status = []
    try:
        def callback(url, urlf, status=status):
            status.append(url)
        urlopenInThread(url, callback)
        start = time.time()
        while time.time() - start < timeout:
            if status: break
            time.sleep(0.1)
    except Exception: pass
    return len(status)

def getDefaultHost():
    """Returns a default host for this machine"""
    import socket
    return socket.gethostbyname(socket.gethostname())

def getHostIP(url='www.google.com'):
    """Returns the (external) host ip for this machine"""
    import socket
    s = socket.socket()
    try:
        s.connect((url, 80))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())

class SimpleWebServer(Thread):
    """A simple webserver with a user-specified root directory.
    It's multithreaded and you can also spawn it from a thread if needed."""
    def __init__(self, rootdir, host='', port=39090, logf=StringIO()):
        #FIXME since the logf is a StringIO(), it will eat up lots of memory for long-running processes as more log messages are printed and stored
        import SocketServer, BaseHTTPServer, SimpleHTTPServer
        Thread.__init__(self)
        self.setDaemon(1)
        self.rootdir = rootdir = os.path.abspath(rootdir)
        self.lastrequests = lastrequests = []
        self.logf = logf

        class HTTPServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
            pass

        class Handler(SimpleHTTPServer.SimpleHTTPRequestHandler):
            """A custom handler which simply does stuff relative to the given rootdir"""
            def translate_path(self, path, rootdir=rootdir, lastrequests=lastrequests):
                """Translate a /-separated PATH to the local filename syntax.

                Copied from SimpleHTTPServer, except that the path is relative to rootdir
                """
                import urlparse, posixpath, urllib
                # keep track of the last places people went to
                lastrequests.append(path)
                lastrequests = lastrequests[-20:]
                # abandon query parameters
                path = urlparse.urlparse(path)[2]
                path = posixpath.normpath(urllib.unquote(path))
                words = path.split('/')
                words = filter(None, words)
                path = rootdir #os.getcwd()
                for word in words:
                    drive, word = os.path.splitdrive(word)
                    head, word = os.path.split(word)
                    if word in (os.curdir, os.pardir): continue
                    path = os.path.join(path, word)
                return path

            def log_message(self, fmt, *args, **kw):
                """Logs to our logfile, rather than sys.stderr"""
                print >>logf, fmt % args

        self.httpd = HTTPServer((host, port), Handler)
        self.address = self.httpd.socket.getsockname()
        self.port = self.address[1]
        self.validhosts = set()
        # create our initial list of valid hostnames
        from socket import gethostbyname_ex, gethostbyaddr
        for func in [gethostbyname_ex, gethostbyaddr]:
            hostname, aliaslist, ipaddrlist = func(self.address[0])
            self.validhosts.add(hostname)
            self.validhosts.update(aliaslist)
            self.validhosts.update(ipaddrlist)
        self.validhosts.discard(self.address[0])
        self.validhosts = list(self.validhosts) + [self.address[0]]
        print >>logf, 'Valid hosts are: %s' % (self.validhosts,)

    def gethost(self): return self.validhosts[-1]
    host = property(gethost, None)

    def run(self):
        """Starts the daemon. Call using start() to run in a new thread."""
        print >>self.logf, 'Serving HTTP on %s:%s' % (self.host, self.port)
        self.httpd.serve_forever()

    def localToNet(self, path):
        """Converts a local path to a url.
        If an invalid path is given (i.e., not in our location), None is returned"""
        from urllib import pathname2url
        path = os.path.abspath(path)
        #print 'path is %s and root is %s' % (path, self.rootdir)
        if not path.startswith(self.rootdir): return None
        rel = path.replace(self.rootdir, '')
        url = 'http://%s:%s' % (self.host, self.port) + pathname2url(rel)
        return url

    def netToLocal(self, url):
        """Converts a url to a local path.
        If an invalid url is given (i.e., not us), None is returned"""
        from urlparse import urlparse
        from urllib import url2pathname
        if not self.isUs(url): return None
        u = urlparse(url)
        return self.rootdir + url2pathname(u.path) # don't replace this with os.path.join!

    def isUs(self, url, timeout=1):
        """Checks a url to see if points to us. Timeout determines how long to wait"""
        from urlparse import urlparse
        from random import random
        from urllib2 import urlopen
        u = urlparse(url)
        if u.hostname in self.validhosts: return 1
        id = '/%s' % (random(),)
        # we run this twice, one with the given port, and once with our port.
        # if either one matches, we accept it.
        # this handles the case where we have multiple servers running on our local machine, on diff ports.
        for port in [u.port, self.port]:
            if not port: port = 80
            check = 'http://%s:%s%s' % (u.hostname, port, id)
            checkForValidURL(check, timeout=timeout)
        if id in self.lastrequests:
            self.validhosts.append(u.hostname)
        return id in self.lastrequests

class ExistingWebServer(Thread):
    """A wrapper on an existing webserver with the same interface as
    SimpleWebServer."""
    def __init__(self, rootdir, rooturl, logf=StringIO()):
        from urlparse import urlparse
        Thread.__init__(self)
        self.setDaemon(1)
        self.rootdir = rootdir = os.path.abspath(rootdir)
        rooturl = self.fixurl(rooturl)
        self.rooturl = rooturl
        self.logf = logf
        els = urlparse(rooturl)
        self.scheme = els.scheme
        self.host = els.hostname
        self.port = els.port if els.port else 80
        self.address = (self.host, self.port)
        self.basepath = els.path
        assert self.basepath.startswith('/')

    def fixurl(self, url):
        """Fixes a url for easy comparison.
        If it doesn't have a scheme, then http:// is added.
        If it doesn't end with /, that's added.
        """
        if '://' not in url:
            url = 'http://'+url
        if not url.endswith('/'):
            url += '/'
        return url

    def run(self):
        """Does nothing, since we're not actually running anything."""
        pass

    def localToNet(self, path):
        """Converts a local path to a url.
        If an invalid path is given (i.e., not in our location), None is returned"""
        from urllib import pathname2url
        path = os.path.abspath(path)
        #print 'path is %s and root is %s' % (path, self.rootdir)
        if not path.startswith(self.rootdir): return None
        rel = pathname2url(path.replace(self.rootdir, '')).lstrip('/')
        return self.rooturl + rel

    def netToLocal(self, url):
        """Converts a url to a local path.
        If an invalid url is given (i.e., not us), None is returned"""
        from urlparse import urlparse
        from urllib import url2pathname
        url = fix(url)
        if not self.isUs(url): return None
        u = urlparse(url)
        return self.rootdir + url2pathname(u.path) # don't replace this with os.path.join!

    def isUs(self, url, timeout=1):
        """Checks a url to see if points to us. Timeout determines how long to wait"""
        from urlparse import urlparse
        # first, quick checks
        if url.startswith(self.rooturl): return 1
        url = self.fixurl(url)
        if url.startswith(self.rooturl): return 1
        # match all relevant elements of the given url to our rooturl
        u = urlparse(url)
        if u.scheme != self.scheme: return 0
        if u.hostname != self.host: return 0
        if u.port != self.port or (not u.port and self.port != 80): return 0
        if not u.path.startswith(self.basepath): return 0
        # if we passed all checks, then it should be us
        return 1


def testSimpleWebServer(rootdir):
    """Runs a series of tests on the simple web server"""
    w = SimpleWebServer(rootdir)
    baseurl = 'http://%s:%s/' % (w.host, w.port)
    print 'Created web server at %s' % (baseurl)
    w.start()
    urls = [baseurl, baseurl+'odijf/dofij/df%20dfi/', 'http://127.0.0.1:%s/' % (w.port), 'http://aphex.cs.columbia.edu:%s/' % (w.port), 'http://apu1nahasa.dyndns.org:%s/' % (w.port)]
    for url in urls:
        print 'NetToLocal for %s yielded %s' % (url, w.netToLocal(url))
    paths = ['/', '/home/neeraj/', rootdir, rootdir +'/odijf/dfoij asdoij~dofij/']
    for path in paths:
        print 'LocalToNet for %s yielded %s' % (path, w.localToNet(path))
    print 'Webserver now reports adddress as: %s:%s' % (w.host, w.port)

    while 1:
        try:
            time.sleep(1)
        except KeyboardInterrupt: break


# QUEUE UTILITIES
def feastOnQueue(q, timeout):
    """Feasts on a queue for upto 'timeout' secs, or until it's empty (whichever comes first).
    Setting a timeout to less than 0 causes it to feast on queue until empty (using current size).
    This always uses non-blocking calls."""
    start = time.time()
    ret = []
    try:
        if timeout < 0:
            for i in xrange(q.qsize()):
                try:
                    ret.append(q.get_nowait())
                except Empty: break
        else:
            while time.time()-start < timeout:
                try:
                    ret.append(q.get_nowait())
                except Empty: break
    except TypeError: pass
    return ret

def findFromQueue(q, matchfunc):
    """Goes through a queue and returns any elements which matchfunc(el) returns true for.
    The other elements are put back on the queue"""
    from Queue import Empty
    num = q.qsize()
    ret = []
    for i in range(num):
        try:
            el = q.get_nowait()
            if matchfunc(el):
                ret.append(el)
            else:
                q.put(el)
        except Empty: pass
    return ret

def getFirstNFromQueue(q, n):
    """Returns the first n items from the given queue (non-blocking) as a list.
    The returned list could have less than n items (or empty if queue is empty)"""
    from Queue import Empty
    ret = []
    for i in range(n):
        try:
            ret.append(q.get_nowait())
        except Empty: break
    return ret

def qprocthread(func, inq, outq, timeout=0.1, *args, **kw):
    """Reads from an input queue, runs a function with them, and puts results on outq.
    The func is called with a list of inputs as func(inputs, *args, **kw)
    The output is assumed to be a list, in the same order as the input.
    Then pairs of (input, output) are put on the outq, one at a time.
    If the outq is None, then it just throws away the computations.
    This is useful if you only care about some side-effect."""
    while 1:
        els = feastOnQueue(inq, timeout=timeout)
        outels = func(els, *args, **kw)
        if not outq: continue
        for input, output in zip(els, outels):
            outq.put((input, output))

class HiLoQueue(object):
    """A queue-like object that wraps two queues.
    One is a hi-priority queue and the other a lo-priority one.
    The hi-priority queue is always taken first"""
    def __init__(self, num=-1, qhi=None, qlo=None, cons=lambda: Queue.Queue(-1)):
        """Initializes this with the given queues, or creates new ones if None.
        The cons() function is used to construct the queue objects."""
        from Queue import Queue
        if qhi is None:
            qhi = cons()
        if qlo is None:
            qlo = cons()
        self.qhi, self.qlo = qhi, qlo
        self.cond = Event()
        if not self.qhi.empty() or not self.qlo.empty():
            self.cond.set()

    def qsize(self):
        """Returns the combined size of the two queues"""
        return self.qhi.qsize() + self.qlo.qsize()

    def empty(self):
        """Returns true only if both queues are empty"""
        return self.qhi.empty() and self.qhi.empty()

    def full(self):
        """Returns true only if both queues are full"""
        return self.qhi.full() and self.qhi.full()

    def put(self, item, block=1, timeout=None, hi=0):
        """Puts item on the queue (default low-priority).
        block and timeout behave just like Queue.Queue.
        'hi' determines whether to use the high-priority one or not (default not)"""
        q = self.qhi if hi else self.qlo
        q.put(item, block=block, timeout=timeout)
        self.cond.set()

    def put_nowait(item, hi=0):
        """Just like Queue.put_nowait, except for addition of 'hi'"""
        q = self.qhi if hi else self.qlo
        q.put_nowait(item)
        self.cond.set()

    def get(self, block=1, timeout=None):
        """Just like Queue.get, except that it first checks the hi-priority queue,
        then falls back to the lo-priority one"""
        def log(s):
            print >>sys.stderr, s

        #log('\nNew get request with block=%s and timeout=%s and hisize=%s, losize=%s' % (block, timeout, self.qhi.qsize(), self.qlo.qsize()))

        try:
            # first check the hi-queue to see if there's an object
            #log('  trying to get from hi')
            return self.qhi.get_nowait()
        except Empty:
            # it failed, so now see if the lo-queue has one
            try:
                #log('  trying to get from lo')
                return self.qlo.get_nowait()
            except Empty:
                # it's also empty
                self.cond.clear()
                # if we're non-blocking, simply send this exception to our caller
                #log('  checking for block (%s)' % (block,))
                if not block: raise
                # else if we're blocking, wait for upto 'timeout' secs for our cond variable
                #log('  waiting for cond for %s' % (timeout,))
                self.cond.wait(timeout=timeout)
                self.cond.clear()
                # at this point, we've either got an object or timeout has expired
                # so again check both queues, this time definitely letting the exception go to caller
                try:
                    #log('  trying hi again')
                    return self.qhi.get_nowait()
                except Empty:
                    #log('  trying lo again')
                    return self.qlo.get_nowait()

    def get_nowait(self):
        """Just like Queue.get_nowait() except it checks both queues in order"""
        return self.get(block=0)

def testHiLoQueue():
    """Test method for hiloqueues.
    >>> testHiLoQueue()
    50
    0
    1
    90
    2
    """
    hlq = HiLoQueue()
    hi, lo = hlq.qhi, hlq.qlo
    hi.put(50)
    for i in range(10):
        lo.put(i)
    print hlq.get() # 50
    print hlq.get() # 0
    print hlq.get() # 1
    hi.put(90)
    print hlq.get() # 90
    print hlq.get() # 2


class QueueService(object):
    """A service based around a queue"""
    def __init__(self, inq=None, outq=None, cons=lambda:Queue.Queue(-1), identfunc=lambda x: x):
        """Initializes this service with the given input and output queues.
        If either queue is not given (defaults), then creates new queues using the given constructor.
        The identfunc returns an identifier from a given input object. This is used to store
        results in a dictionary. Note that this MUST be something which can be hashed.

        NOTE that no computation is done in this class. You can do it yourself by reading elements from
        the inq and then putting results on the output queue, as tuples (input, output).
        """
        # init vars
        if inq is None:
            inq = cons()
        if outq is None:
            outq = cons()
        self.inq, self.outq = inq, outq
        self.identfunc = identfunc
        self.outdict = {} # TODO see if this needs to be a syncdict
        self.allnotify = Event()
        self.notifier = SpecificEvent()
        self.callbacks = Queue.Queue(-1)
        # init daemons
        self.endmon = spawnWorkers(1, self._monitorEnd, interval=0)[0]
        self.callmon = spawnWorkers(1, self._monitorCallbacks, interval=0)[0]

    def _monitorEnd(self):
        """Monitors the output queue to make sure we grab elements from it
        and add to our dict of results"""
        time.sleep(1.0) # TODO just for making sure we don't start too early
        while 1:
            try:
                input, output = self.outq.get()
                id = self.identfunc(input)
                self.outdict[id] = (input, output)
                self.notifier.set(id)
                self.allnotify.set()
            except Exception, e:
                print >>sys.stderr, 'Caught an exception in monitor end!!! %s' % (e,)

    def _monitorCallbacks(self):
        """Monitors our list of callbacks to see if any are done"""
        time.sleep(1.0) # TODO just for making sure we don't start too early
        while 1:
            # wait until something's been added to the output
            self.allnotify.wait()
            self.allnotify.clear()
            # create our matchfunc, which checks to see if any of the output ids have associated callbacks
            doneids = set(self.outdict.keys())
            def matchfunc(el, doneids=doneids):
                # the queue elements are (set(ids), callback)
                ids, callback = el
                # only match if all ids are done
                return ids.issubset(doneids)
            els = findFromQueue(self.callbacks, matchfunc)
            # call these callbacks and remove from our dict
            for ids, callback in els:
                # get the results and clear from our list
                ret = [self.outdict[id] for id in ids if id in self.outdict]
                for id in ids:
                    if id:
                        self._clear(id)
                callback(ret)

    def _clear(self, id):
        """Clears the given id"""
        try:
            del self.outdict[id]
        except KeyError: pass
        self.notifier.clear(id)

    def doOneNow(self, input, hi=1):
        """Does one element and blocks until result is ready. Returns just the output.
        If hi==1 and our input queue is a hi-lo queue, puts things on the high-priority queue."""
        # remove existing elements with this id
        id = self.identfunc(input)
        self._clear(id)
        # put the item on the queue and wait for completion
        try:
            self.inq.put(input, hi=hi)
        except TypeError:
            self.inq.put(input)
        self.notifier.wait(id)
        # now delete this thing from the results and return it
        newin, out = self.outdict[id]
        assert newin == input
        self._clear(id)
        return out

    def doOneSoon(self, input, callback, hi=0):
        """Does one element and calls the given callback with (input, output) after it's done.
        This method is non-blocking (returns right away).
        If hi==1 and our input queue is a hi-lo queue, puts things on the high-priority queue."""
        id = self.identfunc(input)
        self._clear(id)
        idset = set()
        idset.add(id)
        def callbackwrapper(results, callback=callback):
            """Simple wrapper to return results[0]"""
            if results:
                callback(results[0])
        self.callbacks.put((idset, callbackwrapper))
        try:
            self.inq.put(input, hi=hi)
        except TypeError:
            self.inq.put(input)

    def doManyNow(self, inputs, hi=1):
        """Does many elements and blocks until all the results are ready.
        Returns just the outputs.
        If hi==1 and our input queue is a hi-lo queue, puts things on the high-priority queue."""
        # put all these elements on the input queue
        ids = [self.identfunc(el) for el in inputs]
        for id, el in zip(ids, inputs):
            self._clear(id)
            try:
                self.inq.put(input, hi=hi)
            except TypeError:
                self.inq.put(input)
        # wait for all of them to get done
        # Note that since this whole function is blocking, we don't need to do fancy tricks with
        # regards to picking up the outputs. We can just block for each one, until they're all done.
        ret = []
        for id in ids:
            self.notifier.wait(id)
            self._clear(id)
            ret.append(self.outdict[id][1])
        return ret

    def doManySoon(self, inputs, callback, hi=0):
        """Does all elements and calls the given callback only when all are ready.
        Returns a list with (input, output) tuples (not necessarily in same order).
        If hi==1 and our input queue is a hi-lo queue, puts things on the high-priority queue."""
        # clear ids from results and add callback
        ids = [self.identfunc(el) for el in inputs]
        for id in ids:
            self._clear(id)
        self.callbacks.put((set(ids), callback))
        # add elements to queue
        for id, input in zip(ids, inputs):
            try:
                self.inq.put(input, hi=hi)
            except TypeError:
                self.inq.put(input)

    def doManyOneByOne(self, inputs, callback, hi=0):
        """Does elements and calls the given callback each time any one is ready.
        The callback calls with (input, output) tuples, and of course not necessarily in order submitted.
        If hi==1 and our input queue is a hi-lo queue, puts things on the high-priority queue."""
        for input in inputs:
            self.doOneSoon(input, callback, hi=hi)


# EVENT UTILITIES
class SpecificEvent(object):
    """Similar to the Event class, except that this contains an identifier.
    This allows for efficient notification with only the relevant objects wakened."""
    def __init__(self):
        """Creates a new specific event."""
        self.d = {}

    def wait(self, id, timeout=None):
        """Waits until the specified id is notified.
        If timeout is None (the default), this blocks.
        Otherwise, waits for the number of seconds given"""
        if id not in self.d:
            self.d[id] = Event()
        return self.d[id].wait(timeout=timeout)

    def notify(self, id):
        """Notifies the given id that the event is ready"""
        if id not in self.d:
            self.d[id] = Event()
        self.d[id].set()

    def set(self, id):
        """Synonym for notify()"""
        return self.notify(id)

    def clear(self, id):
        """Clears the event at the given id"""
        if id not in self.d:
            self.d[id] = Event()
        self.d[id].clear()

    def isSet(self, id):
        """Checks if the event for the given id is set"""
        if id not in self.d:
            self.d[id] = Event()
        return self.d[id].isSet()

    def is_set(self, id):
        """Checks if the event for the given id is set"""
        if id not in self.d:
            self.d[id] = Event()
        return self.d[id].is_set()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from random import random
    testSimpleWebServer('/home/neeraj/temp/'); sys.exit()
    def proc(lst):
        return [str(x) for x in lst]

    qs = QueueService(proc)
    def producer(q=qs.inq):
        while 1:
            time.sleep(5)
            q.put([random() for i in range(5)])

