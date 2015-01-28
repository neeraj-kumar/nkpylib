#!/usr/bin/env python
"""A dedicated nearest-neighbors computation process.
Originally written by Neeraj Kumar <me@neerajkumar.org>.

Licensed under the 3-clause BSD License:

Copyright (c) 2011, Neeraj Kumar (neerajkumar.org)
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


#TODO
"""This thing is really slow with multiple processes.
I've tried various optimizations with using nkprocrunner.Proc, and spawning procs myself.
I think the problem is in the massive amount of work to stringify queries and de-stringify them.
(Relative to the amount of work of running a single NN query without any of that.)
So, the right solution is to somehow not require this.

There are two parts:
    1. Getting the queries to the subprocs
    2. Getting the results back

Solution 1: Communicate via files.
- Write queries to file, and have children read them and parse them in one go. This might not actually save much

Solution 2: Investigate threading with np
- It seems that np actually releases the GIL for many operations, so this might actually just work!

Solution 3: Multiprocessing
- This is the next thing to investigate. It lets you share variables using shared memory somehow (I think via ctypes, but also possibly np directly)

Solution 4: Shared memory myself
- Last resort, but if all else fails, this is the only other way i can think of doing it...
"""


import os, sys, random, math, time
from nkpylib.nkutils import log, getListAsStr
import numpy as np

METRICS = 'l1 l2 chisq int bhatt'.split()
SEARCH_TYPES = 'k radius radius-k sorted unsorted'.split()

def readData(f=sys.stdin, dtype=float):
    """Reads the data matrix from the given file.
    The first line should contain the # rows (height) and # cols (width) of the matrix"""
    nrows, ncols = map(int, f.readline().strip().split())
    data = []
    for i in range(nrows):
        row = np.array(map(dtype, f.readline().strip().split()))
        assert len(row) == ncols
        data.append(row)
    data = np.vstack(data)
    return data

class DistribNN(object):
    """A class to do easy distributed Nearest Neighbor (NN) calculations."""
    def __init__(self, data, nprocs=1, metric='l2', stype='k', k=None, r=None):
        """Initializes this manager with the given data and number of processes to spawn.
        The data can be one of:
            list of values -> assumed to be the data directly. Written out to temp mmap.
            string -> assumed to be filename for plain text data. Read it in.
            (string, (rows,cols), dtype) -> assumed to mmap fname, to be read with given params.
        Also sets default values for:
            metric: 'l2'
            stype: 'k'
            k: 5 (if no other params given), None otherwise
            r: None
        """
        from subprocess import Popen, PIPE
        import tempfile
        # set params
        usingmmap = 0
        if isinstance(data, basestring):
            # raw data file
            self.datafname = data
            self.data = readData(open(self.datafname))
            usingmmap = 0
        else:
            if len(data) == 3 and isinstance(data[0], basestring) and len(data[1]) == 2:
                # mmap
                self.datafname, shape, dtype = data
                usingmmap = 1
                self.data = np.memmap(self.datafname, dtype=dtype, mode='r', shape=shape)
            else:
                # must be data itself
                dtype, shape = data.dtype, data.shape
                f = tempfile.NamedTemporaryFile(delete=0)
                self.datafname = f.name
                f.close()
                self.data = np.memmap(self.datafname, dtype=dtype, mode='w+', shape=shape)
                self.data[:] = data
        self.nprocs = nprocs
        self.metric, self.stype = metric, stype
        if stype == 'k' and k is None:
            k = 5
        self.k, self.r = k, r
        # spawn procs
        if usingmmap:
            self.args = ['python', 'nknnproc.py', self.datafname, self.data.shape[0], self.data.shape[1], self.data.dtype]
        else:
            self.args = ['python', 'nknnproc.py', self.datafname]
        self.procs = None

    def fvec2inputstr(self, fvec):
        """Converts a fvec and our params to make the input string for the procs"""
        s = [self.metric, self.stype]
        if self.stype == 'k':
            s.append(str(self.k))
        elif self.stype == 'radius':
            s.append(str(self.r))
        elif self.stype == 'radius-k':
            s.append(str(self.r))
            s.append(str(self.k))
        s.extend(fvec)
        return getListAsStr(s, sep='\t')

    def searchUsingProcs(self, fvecs):
        from nkpylib.nkutils import simplenn, nkgrouper
        from nkpylib.nkthreadutils import spawnWorkers
        if not self.procs:
            self.procs = [Popen(self.args, stdin=PIPE, stdout=PIPE, bufsize=0, close_fds=1) for i in range(self.nprocs)]
        # group into inputs for each process
        inputs = map(self.fvec2inputstr, fvecs)
        numeach = len(inputs)//len(self.procs) + 1
        groups = list(nkgrouper(numeach, inputs))
        outputs = [[] for p in self.procs]
        def readouts(p, out):
            while 1:
                s = p.stdout.readline().rstrip()
                out.append(s)

        def sendins(p, inputs):
            for input in inputs:
                p.stdin.write(input+'\n')
            p.stdin.flush()

        tempprocs = []
        for p, g, out in zip(self.procs, groups, outputs):
            tempprocs.extend(spawnWorkers(1, sendins, args=(p, g)))
            tempprocs.extend(spawnWorkers(1, readouts, args=(p, out)))

        for i, group in enumerate(groups):
            while len(outputs[i]) < len(group):
                time.sleep(0.1)
        outputs = sum(outputs, [])
        return outputs

    def search(self, fvecs):
        """Runs a nn search for the given feature vectors.
        Uses the existing parameters for metric, stype, k, and r.
        Any of those can be changed at any time.
        """
        from nkpylib.nkutils import simplenn, filternnresults, nkgrouper, getTimeDiffs
        from nkpylib.nkthreadutils import spawnWorkers
        from Queue import Queue
        #return self.searchUsingProcs(fvecs)
        self.sort = 0 if self.stype == 'unsorted' else 1
        start = time.time()
        if self.nprocs>1:
            inq, outq = Queue(), Queue()
            def inproc():
                while 1:
                    idx, fvec = inq.get()
                    t1 = time.time()
                    dists = simplenn(self.data, fvec, metric=self.metric, normalize=None)
                    t2 = time.time()
                    out = filternnresults(dists, k=self.k, r=self.r, sort=self.sort)
                    t3 = time.time()
                    #log('Got times: %s' % (getTimeDiffs([t1,t2,t3])))
                    #log('Got outs: %s' % (out,))
                    outq.put((idx, out))

            # spawn procs
            procs = spawnWorkers(self.nprocs, inproc, interval=0)
            # add to inq
            for i, fvec in enumerate(fvecs):
                inq.put((i, fvec))
            #log('Added %d fvecs to inq' % (len(fvecs)))
            # read from outq
            outputs = [0]*len(fvecs)
            todo = set(range(len(fvecs)))
            while todo:
                if len(todo) % 10 == 0:
                    log('%d left in todo, %0.3fs elapsed' % (len(todo), time.time()-start))
                    #log('Outputs: %s' % (outputs,))
                    pass
                idx, out = outq.get()
                todo.remove(idx)
                outputs[idx] = out
        else:
            alldists = (simplenn(self.data, fvec, metric=self.metric, normalize=None) for fvec in fvecs)
            outputs = [filternnresults(dists, k=self.k, r=self.r, sort=self.sort) for dists in alldists]
        return outputs


def nnmainloop(data, l):
    """Runs a single query with given input line."""
    from nkutils import simplenn
    try:
        # parse line
        els = l.strip().split()
        metric = els.pop(0)
        assert metric in METRICS
        stype = els.pop(0)
        assert stype in SEARCH_TYPES
        # set params
        k = -1
        r = -1
        sort = 1
        if stype == 'k':
            k = int(els.pop(0))
        elif stype == 'radius':
            r = float(els.pop(0))
        elif stype == 'radius-k':
            r = float(els.pop(0))
            k = int(els.pop(0))
        elif stype == 'unsorted':
            sort = 0
        # parse data
        fvec = np.array(map(float, els))
        assert len(fvec) == data.shape[1]
        # run the actual search
        dists = simplenn(data, fvec, metric=metric, normalize=None)
        ret = filternnresults(dists, k=k, r=r, sort=sort)
        print ret
    except Exception, e:
        log('Exception of type %s: %s' % (type(e), e))
        print 0


import numpy
from nkutils import MemUsage, getTimeDiffs
from nkutils import simplenn, bulkNNl2, filternnresults

def nn1(m,n,ret):
    """Iterating over each row and running simplenn"""
    for i, row in enumerate(n):
        ret[i,:] = simplenn(m, row, metric='l2', normalize=None, withsum=1)
        #b = simplenn(m, row, metric='l2', normalize=None, withsum=1)
        #dists = (m - row) ** 2
        #ret1[i,:] = numpy.sum(dists, 1)
        #assert np.array_equal(b,ret1[i,:])
    return ret

def benchmark():
    """Runs a benchmark to see what's faster"""
    import numpy.random as ra
    np.set_printoptions(precision=7, linewidth=150, suppress=1)
    times = [time.time()]
    r = ra.random
    M = 200000
    N = 100
    D = 200
    mem = MemUsage()
    m = r((M,D)).astype(np.float32)
    mem.add()
    n = r((N,D)).astype(np.float32)
    mem.add()
    #ret1 = np.zeros((N,M), dtype=np.float32)
    times.append(time.time())
    #ret1 = nn1(m,n,ret1)
    times.append(time.time())
    mem.add()
    times.append(time.time())
    ret2 = bulkNNl2(n,m)/D
    times.append(time.time())
    mem.add()
    #print ret1, ret1.shape
    print ret2, ret2.shape
    for row in ret2:
        pairs = filternnresults(row, k=None, r=0.12, sort=1)
    print pairs[:10], len(pairs)
    times.append(time.time())
    mem.add()
    #assert np.allclose(ret1,ret2)
    for i in range(len(mem)):
        print '% 10.4f' % (mem.delta(i) / 1024.0/1024.0)
    print '% 10.4f' % (mem.usage()/1024.0/1024.0)
    print getTimeDiffs(times)
    #print list(mem)

if __name__ == '__main__':
    benchmark(); sys.exit()
    # do appropriate main loop
    if sys.argv[1] == 'distrib':
        # distrib
        d = DistribNN(sys.argv[2], nprocs=4)
        # read query data from stdin
        queries = [map(float, l.strip().split()[3:]) for l in sys.stdin]*1
        print 'Read %d queries' % (len(queries))
        t1 = time.time()
        ret = d.search(queries)
        t2 = time.time()
        print 'Took %0.3fs to run %d queries: %s' % (t2-t1, len(ret), ret[:2])
        #print ret
    else:
        # NN proc main loop
        from nkutils import stdmainloop
        # to read raw data, args are just (fname,)
        # to read mmap, args are (fname, nrows, ncols, [dtype='float64'])
        # read data
        t1 = time.time()
        if len(sys.argv) == 2:
            # raw data
            fname = sys.argv[1]
            if fname == '-':
                f = sys.stdin
            else:
                f = open(fname)
            data = readData(f=f)
        elif len(sys.argv) >= 4:
            fname, nrows, ncols = sys.argv[1:4]
            mmapshape = (int(nrows), int(ncols))
            try:
                dtype = sys.argv[4]
            except Exception:
                dtype = 'float64'
            data = np.memmap(fname, dtype=dtype, mode='r', shape=mmapshape)
        t2 = time.time()
        log('Read %d data vecs with %d dims each in %0.3fs' % (data.shape[0], data.shape[1], t2-t1))
        stdmainloop(lambda l: nnmainloop(data, l))

