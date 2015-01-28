"""Utilities to help with constructing distance matrices"""

import os, sys, time
import numpy as np

INF = 1e9

class Distancer(object):
    """A class to keep track of distances computed so far and compute more"""
    def __init__(self, distfunc, datafname, defaultval=INF, sortasc=1, feats=None, savefreq=30):
        """Initializes with the distfunc to use"""
        self.distfunc = distfunc
        self.datafname = datafname
        self.defaultval = defaultval
        self.sortasc = sortasc
        self.savefreq = savefreq
        self.lastsave = 0
        self.feats = []
        self.dists = None
        self.load()
        if feats:
            for f in feats:
                if f not in self.feats:
                    self.feats.append(f)
        self.extenddists()

    def load(self):
        """Loads data from disk"""
        from cPickle import load
        try:
            self.dists, self.feats = load(open(self.datafname))
        except Exception:
            pass

    def save(self, infork=1):
        """Saves our data to disk"""
        from cPickle import dump
        from nkutils import saveandrename
        if time.time() - self.lastsave < self.savefreq: return
        self.lastsave = time.time()
        try:
            os.makedirs(os.path.dirname(self.datafname))
        except OSError:
            pass
        print 'Saving data to %s with %d feats' % (self.datafname, len(self.feats))
        saveandrename(self.datafname, lambda f: dump((self.dists, self.feats), f), retfile=1, infork=infork)
        print '  Finished saving data with %d feats' % (len(self.feats))

    def extenddists(self, saveinfork=0):
        """Extends distance matrix by the needed number of rows/cols"""
        s = self.dists
        n = len(self.feats)
        # first check if we have distances at all
        if s is None:
            self.dists = np.ones((n,n)) * self.defaultval
            self.save()
            return
        coldiff = n - s.shape[1]
        changed = 0
        if coldiff > 0:
            s = np.hstack((s, np.ones((s.shape[0], coldiff))* self.defaultval))
            changed = 1
        rowdiff = n - s.shape[0]
        if rowdiff > 0:
            s = np.vstack((s, np.ones((rowdiff, s.shape[1]))* self.defaultval))
            changed = 1
        self.dists = s
        if changed:
            #self.save(infork=saveinfork)
            pass

    def valid(self, v):
        """Checks if a valid is valid"""
        return abs(self.defaultval-v) > 1e-4

    def dist(self, a, b):
        """Returns the distance between the given elements"""
        # extend matrices if we've never seen these before
        if a not in self.feats:
            self.feats.append(a)
        if b not in self.feats:
            self.feats.append(b)
        self.extenddists()
        # find the indices and precomputed distance
        i = self.feats.index(a) #FIXME these are linear scans for now...
        j = self.feats.index(b) #FIXME these are linear scans for now...
        d = self.dists[i,j]
        # if it's not a default val, it's already computed
        #print 'here... with %s, %s => %s, %s => %s' % (a,b, i,j,d)
        if self.valid(d):
            return d
        try:
            d = self.dists[i,j] = self.dists[j,i] = self.distfunc(a, b)
        except Exception, e:
            print '%s Error with %s, %s: %s, setting to default val of %s' % (type(e), a, b, e, self.defaultval)
            d = self.defaultval
        return d

    def computemissing(self, useprocs=1):
        """Computes missing distances"""
        s = self.dists
        n = len(self.feats)
        nums = dict(missing=0, examined=0)
        todo = [(i, j) for i in range(n) for j in range(i+1, n)]
        print 'Got %d todo' % (len(todo))
        def callback(result):
            """Takes the result of dfunc() and applies it"""
            #print 'Got call back with %s' % (result,)
            d, i, j = result
            s[j,i] = s[i,j] = d
            self.save(infork=1)

        self.computemany(todo, callback=callback, useprocs=useprocs, indices=1)
        print 'All done'
        self.lastsave = 0
        self.save(infork=0)

    def computemany(self, todo, callback=None, useprocs=1, indices=0):
        """Computes many distances using procs.
        Todo should consist of (i,j) pairs (if indices=1), or feat names (if indices=0).
        If you supply a callback it is called with a single tuple: (distance, i, j).
        If you don't, the default callback simply sets dists[i,j] = dists[j,i] = d.
        """
        import multiprocessing as mp
        newtodo = []
        print 'Adding indices if necessary'
        if not indices:
            # add any missing elements
            for a, b in todo:
                if a not in self.feats:
                    self.feats.append(a)
                if b not in self.feats:
                    self.feats.append(b)
            self.extenddists()
            # find the indices
            for a, b in todo:
                i = self.feats.index(a) #FIXME these are linear scans for now...
                j = self.feats.index(b) #FIXME these are linear scans for now...
                newtodo.append((i,j))
            todo = newtodo

        def dfunc(i, j):
            """Returns the distance between the given rows of the data.
            Returns (d, i, j) for convenience."""
            #print 'Got dfunc with %s, %s' % (i, j)
            a, b = self.feats[i], self.feats[j]
            try:
                d = self.distfunc(a, b)
            except Exception, e:
                print '%s Error with %s (%s), %s (%s): %s, setting to default val of %s' % (type(e), a, i, b, j, e, self.defaultval)
                d = self.defaultval
            return (d, i, j)

        if not callback:
            def callback(result):
                """Default callback just sets the distances"""
                d, i, j = result
                self.dists[j,i] = self.dists[i,j] = d

        # only do those that aren't valid
        newtodo = []
        for i, j in todo:
            d = self.dists[i,j]
            if not self.valid(d):
                newtodo.append((i,j))
            else:
                #callback((d,i,j)) #FIXME see if we want to actually call the callback here
                pass
        print 'Filtered down from %d to %d' % (len(todo), len(newtodo))
        todo = newtodo

        if useprocs:
            # setup the queues and processing function
            inq, outq = mp.Queue(), mp.Queue()
            def procfunc(inq, outq):
                while 1:
                    cur = inq.get()
                    if cur is None: break
                    ret = dfunc(*cur)
                    outq.put(ret)

            # start the processes
            #procs = [mp.Process(target=procfunc, args=(inq,outq)) for i in range(mp.cpu_count()//2)]
            procs = [mp.Process(target=procfunc, args=(inq,outq)) for i in range(12)]
            for p in procs:
                p.daemon = True
                p.start()
            # add data to the input queue
            for cur in todo:
                inq.put(cur)
            # add sentinels at end to quit
            for n in range(100):
                inq.put(None)
            # read results from the output queue until we're all done
            ntodo = len(todo)
            while ntodo > 0:
                if ntodo % 100 == 0:
                    print 'ntodo is %s, %d in inq, %d in outq' % (ntodo, inq.qsize(), outq.qsize())
                ret = outq.get()
                callback(ret)
                ntodo -= 1
        else:
            for i, j in todo:
                callback(dfunc(i, j))
        print 'Finished with computemany'

    def computerow(self, a):
        """Computes and returns distances for a full row vs. the given element.
        Returns sorted (dist, name) pairs."""
        if a not in self.feats:
            self.feats.append(a)
        self.extenddists()
        ret = [(self.dist(a, b),b) for b in self.feats]
        ret.sort(reverse=not self.sortasc)
        return ret

    def getmatrix(self, rows, cols, indices=0):
        """Returns a matrix consisting of the values at the given row and cols.
        Does no computation; only returns existing values."""
        ret = np.ones((len(rows), len(cols))) * self.defaultval
        for i, r in enumerate(rows):
            print i, r
            if not indices:
                try:
                    r = self.feats.index(r)
                except:
                    continue
            for j, c in enumerate(cols):
                if not indices:
                    try:
                        c = self.feats.index(c)
                    except:
                        continue
                ret[i,j] = self.dists[r,c]
        return ret
