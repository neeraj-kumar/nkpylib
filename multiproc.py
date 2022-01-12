#!/usr/bin/env python
"""A utility to run multiple processes and stripe inputs across them, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2013, Neeraj Kumar (neerajkumar.org)
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

import os, sys, time
from utils import log
from asyncproc import Process

class MultiProc(object):
    """Run multiple async processes and stripe inputs across all of them."""
    def __init__(self, args, num=0, **kw):
        """Starts a set of processes, defined by num:
            if num is an int:
                > 0: that many procs
                <= 0: getNumCPUs() - num
            elif num is a float:
                that percentage of cpus of this sys
        Any additional kw args are passed to the initializer for Process().
        (These are the same as the inputs to subprocess.Popen())
        """
        from threading import Lock
        from copy import deepcopy
        from Queue import Queue, LifoQueue
        from threadutils import spawnWorkers
        from utils import parseNProcs
        self.nprocs = nprocs = parseNProcs(num)
        self.args = deepcopy(args) # in case we alter them
        # spawn processes and associated locks and working queues
        self.procs = [Process(args, **kw) for i in range(nprocs)]
        self.proclocks = [Lock() for p in self.procs]
        self.working = [LifoQueue() for p in self.procs]
        # spawn instance vars to track inputs and results
        self.inq = Queue()
        self.results = {}
        self.resultlock = Lock()
        # spawn worker threads
        self.inloop = spawnWorkers(1, self.inputloop)[0]
        self.outloop = spawnWorkers(1, self.outputloop)[0]

    def getbestproc(self):
        """Returns the index of the best proc to write to"""
        bestsize, besti = min((q.qsize(), i) for i, q in enumerate(self.working))
        #log(' ** returning best proc %s' % (besti,))
        return besti

    def inputloop(self, delay=0.01):
        """Runs an infinite loop taking things from our inq and sending to processes."""
        from threadutils import feastOnQueue
        while 1:
            t = time.time()
            todo = feastOnQueue(self.inq, -1)
            if not todo:
                wait = delay - (time.time()-t)
                if wait > 0:
                    time.sleep(wait)
                continue
            #log(' ** processing %d inputs' % (len(todo)))
            for id, input, callback in todo:
                p = self.getbestproc()
                if not input.endswith('\n'):
                    input += '\n'
                with self.proclocks[p]:
                    self.procs[p].write(input)
                    self.working[p].put((id, callback))

    def outputloop(self, delay=0.01):
        """Runs an infinite loop which reads completed outputs and adds them to results."""
        while 1:
            t = time.time()
            for p in range(len(self.procs)):
                # if there is nothing scheduled for this proc, continue
                if self.working[p].empty(): continue
                # check to see if the proc has completed a job
                with self.proclocks[p]:
                    out = self.procs[p].readline(blocking=-1)
                #log(' ** checking output for proc %d got %s' % (p, out))
                if out is not None:
                    out = out.rstrip('\n')
                    # we got something, so add to results
                    id, callback = self.working[p].get()
                    with self.resultlock:
                        self.results[id] = out
                    # also call callback
                    if callback:
                        callback(id, out)
            wait = delay - (time.time()-t)
            if wait > 0:
                time.sleep(wait)

    def write(self, inputs, delay=0.01):
        """Synchronously processes given inputs (a list of strings).
        If they don't have '\n' at the end, it's added.
        Yields an iterator with results, in the same order as inputs.
        Each result has the '\n' rstripped().
        You can specify a custom delay to use for the busy-wait.
        """
        ids = self.submit(inputs)
        #log('submitted %d inputs: %s' % (len(ids), ids))
        for id in ids:
            yield self.getresult(id, blocking=delay)

    def submit(self, inputs, ids=None, callback=None):
        """Asynchronously processes given inputs (a list of strings).
        If they don't have '\n' at the end, it's added.
        If ids is given, it should be of the same length as inputs.
        If not, then random ids will be created.
        If a callback is given, then it is called each time one item finishes:
            callback(id, output)
        Note that items could be finished out-of-order, hence the need for ids.
        Returns a list of ids, corresponding to the inputs given.
        """
        import uuid
        if not inputs: return []
        if not ids:
            ids = [uuid.uuid1().hex for i in inputs]
        for id, input in zip(ids, inputs):
            self.inq.put((id, input, callback))
        return ids

    def getresult(self, id, blocking=0.01):
        """Returns the result for the given id, with '\n' rstripped().
        If it's not done yet, then what we do depends on 'blocking':
            > 0: busy-waits, until it's ready, with given sleeptime
            <= 0: returns None
        Note that once we return the result, it's no longer cached here.
        """
        while 1:
            with self.resultlock:
                if id in self.results:
                    return self.results.pop(id)
            # if we're here, then it was not done yet
            if blocking > 0:
                time.sleep(blocking)
            else:
                return None


if __name__ == '__main__':
    args = sys.argv[1:]
    assert args
    p = MultiProc(args)
    t = time.time()
    todo = list(sys.stdin)
    ret = p.write(todo)
    for r in ret:
        print r
    elapsed = time.time()-t
    print >>sys.stderr, 'Total time with %d procs and %d inputs: %0.5f (%0.4f s/proc/input)' % (p.nprocs, len(todo), elapsed, elapsed*p.nprocs/len(todo))

