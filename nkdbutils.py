"""A set of utilities to use for web.py's db class.

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

from __future__ import with_statement
import os, sys, time
from nkutils import now, utcnow

def selectone(db, table, **kw):
    """Selects a single element from the database if exists, or None"""
    kw['limit'] = 1
    r = [r for r in db.select(table, **kw)]
    if not r:
        return None
    return r[0]

def selectdict(db, table, **kw):
    """Creates a dict from a select query.
    Uses the 'what' part of the kw to make the dictionary.
    If the length of each row is 1, then actually creates a set.
    If the length of each row is 2, then uses the entries as key/value, resp.
    If the length of each row is >2, then uses the first entry as key, and the rest as value"""
    fields = kw['what']
    if isinstance(fields, (str, unicode)):
        fields = [f.strip().split(' ')[-1] for f in fields.split(',')]
    results = db.select(table, **kw)
    ret = {}
    retset = set()
    for row in results:
        assert len(row) >= 1
        key = row[fields[0]]
        if len(row) == 1:
            retset.add(key)
            continue
        elif len(row) == 2:
            val = row[fields[1]]
        else:
            val = [row[f] for f in fields[1:]]
        ret[key] = val
    if len(fields) == 1: return retset
    return ret

def count(db, query, typefunc=int, **kw):
    """Returns the count of the given query.
    Query can be either a query object or the database name to query from.
    If the latter, then runs db.select(query, **kw) first.
    if kw doesn't have 'what', then sets it to 'count(*)'"""
    import web
    import web.utils
    if not isinstance(query, web.utils.IterBetter):
        if 'what' not in kw:
            kw['what'] = 'count(*)'
        query = db.select(query, **kw)
    return typefunc(query[0].values()[0])

def sqllist(lst, strquote=0):
    """Real sqllist function, which handles ints and floats as well"""
    import web
    if not lst: return ''
    lst = list(lst)
    if not lst: return ''
    if isinstance(lst[0], (int, long, float)):
        return ','.join(str(v) for v in lst)
    if strquote and isinstance(lst[0], basestring):
        return ','.join("'%s'" % (s.replace("'", "\'")) for s in lst)
    return web.db.sqllist(lst)

def addcheckfield(db, existingtable, newtable, combinetable, cols, opts, combexistkey=None, combnewkey=None, debug=0):
    """Adds a "checkbox" field to an existing table.
    This is where you have multiple non-exclusive options available
    for a given table type, where it can be many-to-many.
    The general SQL solution to this is to create a table of all the options,
    and then another table which just contains pairs of ids from the
    original table and the new one.

    This function automates this task, to some degree. You pass in:
        - existingtable: the name of the existing table
        - newtable: the name of the new table of options
        - combinetable: the name of the table which contains combinations
        - combexistkey: the name of the column for the existing table in the combine table.
            This defaults to None, which means it uses the name of the existingtable.
        - combnewkey: the name of the column for the new table in the combine table.
            This defaults to None, which means it uses the name of the newtable.
        - cols: the full string used in the create table() call for newtable
        - opts: the list of options to add to the new table, given as items to add directly using db.insert()
    """
    with db.transaction():
        print db.query('CREATE TABLE %s (%s);' % (newtable, cols), _test=debug)
        if not combexistkey:
            combexistkey = existingtable
        if not combnewkey:
            combnewkey = newtable
        print db.query('CREATE TABLE %s (id integer primary key autoincrement, %s integer not null, %s integer not null);' % (combinetable, combexistkey, combnewkey), _test=debug)
        print db.query('CREATE UNIQUE INDEX %s_%s%s ON %s (%s, %s);' % (combinetable, existingtable, newtable, combinetable, combexistkey, combnewkey), _test=debug)
        for o in opts:
            print db.insert(newtable, _test=debug, **o)

def retrysql(meth, retries, *args, **kw):
    """Retries the given method several times, checking for sqlite errors.
    Raises the last error if we didn't succeed even after all the retries.
    Uses a transaction.
    If you give it a kwarg of multiple_entries=[(args, kw),...], then it will put all
    of those into a single transaction.
    """
    import sqlite3
    lasterr = None
    for i in range(retries):
        try:
            if 'multiple_entries' in kw and kw['multiple_entries']:
                with meth.im_self.transaction():
                    for curargs, curkw in kw['multiple_entries']:
                        ret = meth(*curargs, **curkw)
                    return ret
            else:
                with meth.im_self.transaction():
                    return meth(*args, **kw)
        except sqlite3.OperationalError, e:
            #TODO see about other database errors as well!
            lasterr = e
            time.sleep(0.2)
    raise e


class TransactionWrapper(object):
    """Database transaction wrapper.
    No support for nested transactions right now."""
    def __init__(self, dbwrap):
        self.dbwrap = dbwrap
        self._trans = self.dbwrap._db.transaction()

    def __enter__(self):
        import logging
        logger = logging.getLogger('db.transaction')
        logger.info('Entering db transaction')
        return self._trans

    def __exit__(self, exctype, excvalue, traceback):
        import logging
        logger = logging.getLogger('db.transaction')
        if exctype is not None:
            logger.error('Hit a transaction snag, rolling back: %s, %s, %s' % (exctype, excvalue, traceback))
        else:
            logger.info('Successfully completed transaction')


class DBWrapper(object):
    """A wrapper for a web.py db so that we can log all calls.
    Set db = DBWrapper(db) at the top of your code
    """
    @classmethod
    def parselog(cls, fname, func=lambda x: x):
        """Parses the given logfile and returns a list of queries, optionally filtered by func"""
        import datetime
        from sqlite3 import OperationalError
        ret = []
        for i, line in enumerate(open(fname)):
            #if i > 100000: break
            try:
                timestamp, name, level, s = line.strip().split(' :: ', 3)
                x = eval(s)
                x['level'] = level
                ret.append(func(x))
            except (SyntaxError, NameError):
                print >>sys.stderr, 'Could not parse line %d: %s' % (i+1, line.strip())
            except ValueError:
                print >>sys.stderr, 'Line %d was in a weird format: %s' % (i+1, line.strip())
        return ret

    def __init__(self, db, logfname=None):
        """Initializes this wrapper, optionally with a logname"""
        import logging
        self._db = db
        self.queries = []
        # setup formatters
        logfmt = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
        # setup basic logging
        self.logger = logging.getLogger('db')
        self.logger.setLevel(logging.DEBUG)
        if logfname: 
            log = logging.FileHandler(logfname)
            log.setFormatter(logfmt)
            self.logger.addHandler(log)

    def _log(self, meth, *args, **kw):
        """Logs a the given method"""
        import logging
        from web.utils import Storage
        import sqlite3
        if '_test' in kw and kw['_test']: return meth(*args, **kw) # don't measure test methods
        x = Storage()
        self.queries.append(x)
        x.type = meth.__name__
        try:
            x.table = args[0]
        except IndexError:
            x.table = kw.get('table', '')
        x.args = args[:]
        x.kw = dict(kw)
        # first run a test version to get the query
        kw['_test'] = 1
        x.query = str(meth(*args, **kw)) + ';' # for ease in copy-pasting
        kw['_test'] = 0
        # strip irrelevant vars, which can be huge
        if 'vars' in x.kw:
            x.kw['vars'] = dict(x.kw['vars']) # create our own copy
            for key in x.kw['vars'].keys():
                k = '$%s' % (key)
                if k not in x.kw.get('where', '') and k not in x.args and k not in x.kw.values():
                    del x.kw['vars'][key]
        x.t1 = time.time()
        # now run the actual query
        try:
            ret = meth(*args, **kw)
        except sqlite3.OperationalError, e:
            x.t2 = time.time()
            x.elapsed = x.t2-x.t1
            x.exception = e
            logger = logging.getLogger('db.%s' % (x.type))
            logger.exception('%s' % (dict(x,)))
            raise
        x.t2 = time.time()
        x.elapsed = x.t2-x.t1
        def getlogmeth(x):
            """Returns the appropriate log meth"""
            logger = logging.getLogger('db.%s' % (x.type))
            if 'exception' in x: return logger.error
            e = x.elapsed
            if e < 0.1: return logger.info
            if e < 0.5: return logger.warning
            return logger.error
        getlogmeth(x)('%s' % (dict(x,)))
        #print >>sys.stderr, '  Got x: %s' % (x,)
        return ret
    
    def query(self, *args, **kw): 
        return self._log(self._db.query, *args, **kw)
    
    def select(self, *args, **kw):
        return self._log(self._db.select, *args, **kw)
    
    def where(self, *args, **kw):
        return self._log(self._db.where, *args, **kw)
    
    def insert(self, *args, **kw):
        return self._log(self._db.insert, *args, **kw)
        
    def multiple_insert(self, *args, **kw):
        return self._log(self._db.multiple_insert, *args, **kw)
    
    def update(self, *args, **kw):
        raise NotImplementedError # FIXME there's a bug here that causes the thing to not actually be supported
        return self._log(self._db.update, *args, **kw)
    
    def delete(self, *args, **kw):
        return self._log(self._db.delete, *args, **kw)

    def transaction(self):
        """Wrapper for a transaction"""
        return TransactionWrapper(self)

def analyzeDBLog(logfname):
    """Analyzes a logfile from DBWrapper"""
    validkeys = 'level query table elapsed type'.split()
    def func(x):
        """Does some basic stuff with a single query x"""
        for key in x.keys():
            if key not in validkeys:
                del x[key]
        return x
    ret = DBWrapper.parselog(sys.argv[1], func=func)
    print 'Got %d lines in log %s' % (len(ret), sys.argv[1])
    print ret[0]
    def timestr(seq):
        """Returns a time string from a given list of elapsed times"""
        l = len(seq)
        tot = sum(seq)
        m = min(seq)
        M = max(seq)
        avg = tot/float(l)
        median = sorted(seq)[l//2]
        ret = '%d times for total of %0.3fs, min %0.3fs, max %0.3fs, avg %0.3fs, med %0.3fs' % (l, tot, m, M, avg, median)
        return ret

    # group by each and get elapsed stats
    for group in validkeys:
        if group == 'elapsed': continue
        d = {}
        for r in ret:
            d.setdefault(r[group], []).append(r['elapsed'])
        print 'For grouping %s got %d unique keys:' % (group, len(d))
        for k in sorted(d, key=lambda x: sum(d[x]), reverse=1):
            print '  %s: %s' % (timestr(d[k]), k)


def createDBDelayQ(maxsize=-1, callback=None, nretries=30, debug=0, nworkers=1, penalty=0.1):
    """Creates a queue to use for delayed sql queries.
    Params:
        maxsize: the maximum size of the queue (default = -1 = infinite)
        callback: a function called with (func, args, kw, timestamp) prior to sql (default=None)
        nretries: the sql call is made using retrysql with this many retries (default=30)
        debug: if 1, then all delayed calls are made immediate (default=0)
        nworkers: number of workers that will add stuff from queue (default=1)
    Returns (q, addDelayedSQL function)"""
    from Queue import Queue
    from nkthreadutils import spawnWorkers
    dbdelayq = Queue(maxsize)
    delays = [0]
    MAX_DELAYS = 10
    def dbdelayexec(callback=callback, nretries=nretries, delays=delays):
        """Gets statements to execute from the dbdelay q and executes them.
        The q should contain elements like (func, args, kw, timestamp).
        If a callback is given, then it's executed with the same args when we're about to add."""
        while 1:
            try:
                el = dbdelayq.get()
                func, args, kw, timestamp = el[:4]
                delay = time.time()-timestamp
                delays.append(delay)
                if len(delays) > MAX_DELAYS:
                    delays.pop(0)
                if callback:
                    callback(func, args, kw, timestamp)
                retrysql(func, nretries, *args, **kw)
            except Exception, e:
                pass
    spawnWorkers(nworkers, dbdelayexec)

    def addDelayedSQL(meth, *args, **kw):
        """Adds a sql statement to execute delayed.
        If you want it to execute immediately (synchronously), you can send immediate=1.
        Note that for debugging, we can easily turn all delayed entries into immediate ones."""
        #TODO perhaps add a backoff sleep() if the lagtime becomes too long
        if debug or ('immediate' in kw and kw['immediate']):
            if 'immediate' in kw:
                del kw['immediate']
            retrysql(meth, nretries, *args, **kw)
        else:
            dbdelayq.put((meth, args, kw, time.time()))
            if penalty > 0:
                d = delays[-1]
                p = penalty * d
                #print 'Delaying %0.4fs due to penalty %s and last delay %0.3fs' % (p, penalty, d)
                time.sleep(p)

    return (dbdelayq, addDelayedSQL)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <logfname>' % (sys.argv[0])
        sys.exit()
    analyzeDBLog(sys.argv[1])
