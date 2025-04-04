#!/usr/bin/env python
"""Neeraj's feature extractor server.
Starts an rqs-based server for extracting features across many machines.
Assumes images are already on all workers, with same basepath.


Licensed under the 3-clause BSD License:

Copyright (c) 2012, Neeraj Kumar (neerajkumar.org)
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
try:
    import simplejson as json
except ImportError:
    import json
import threading
from threading import Thread
from nkpylib.utils import log
from nkpylib.redisutils import *

GROUPSIZE = 20

def keyize(dlst):
    """Converts a list of dictionaries into a key"""
    return tuple(sorted([tuple(sorted(d.items())) for d in dlst]))

def createjob_featx(inputs, **job):
    """Populates a featx job using the given inputs and job dict"""
    job['cmdline'] = ['nice', 'python', 'pylib/features.py']
    job.update(inputs=inputs[:], ninputs=len(inputs), stdin='')
    for fname, rownum in inputs:
        s = '\t'.join([fname, job['mstr'], job['fstr'], job['fiducials'], 'outfmt@%(path)s\t%(fvec)s::errfmt@%(path)s\t%(errormsg)s'])
        job['stdin'] += s+'\n'
    return job

def createjob_gist(inputs, **job):
    """Populates a gist job using the given inputs and job dict"""
    job['cmdline'] = ['nice', 'python', 'pylib/nkgist.py', job['w'], job['h']]
    job.update(inputs=inputs[:], ninputs=len(inputs), stdin='')
    for fname, rownum in inputs:
        job['cmdline'].append(fname)
    return job

class RQSServer(Thread):
    """Class to maintain a 'server' for an rqs proc.
    """
    def __init__(self, jobid, qbase, rqs_cfg, inputs, jobkw, jobfunc, **c):
        #RQS_CONFIG = dict(host='arnold.cs.washington.edu', port=10001, password='clothingrqsftw')
        """Initializes this with various params"""
        # initialize vars
        Thread.__init__(self)
        self.jobid = jobid
        self.qname = makeqname(qbase)
        self.rqs = RedisQueueService(**rqs_cfg)
        self.db = self.rqs.redis
        self.inputs = inputs
        self.jobkw = jobkw
        self.jobfunc = jobfunc

    def submitjobs(self):
        """Submits jobs"""
        inq = self.qname('inq', self.jobid)
        allinq = self.qname('inq')
        def inqcallback(id, item, qname, rqs):
            """Sets the status to inq"""
            rqs.setstatusmsg(qname.replace(':inq', ':status'), id, 'inq')

        groups = list(nkgrouper(GROUPSIZE, self.inputs))
        print '    Got %d groups of groupsize %d' % (len(groups), GROUPSIZE)
        for inputs in groups:
            realid = time.time()
            # create the job structure
            job = self.jobfunc(inputs, id=realid, jobid=self.jobid, submitted=time.time(), **self.jobkw)
            # submit it
            log('Adding job with %d inputs' % (len(inputs)))
            item = (realid, job)
            self.rqs.put(realid, item, inq, callback=inqcallback)
            # make sure the overall list of featx inputs is valid for this jobid
            toadd = {self.jobid: 1000}
            self.rqs.redis.zadd(allinq, **toadd)
            # add to list of todos
            p['todo'].append(dict(params=params, svmstrs=svmstrs, submitted=time.time(), realid=realid))

    def readdone(self):
        """Reads any that are done"""
        outq = self.qname('outq', self.jobid)
        status = self.qname('status', self.jobid)
        while 1:
            obj = self.db.rpop(outq)
            if not obj: break
            idpair, item = json.loads(obj)
            realid, out = item
            self.parseresult(out, {})
            self.db.hdel(status, realid)

    def run(self):
        """Runs feature selection, picking up where we left off.
        Note that since we inherit from Thread, you can call
        start() to run this in a new Thread."""
        #TODO what's the advantage to running this in a thread?
        import Queue

        # figure out what's been already submitted from our todos and submit the rest
        inq = self.qname('inq', self.jobid)
        outq = self.qname('outq', self.jobid)
        all = self.db.lrange(inq, 0, -1) + self.db.lrange(outq, 0, -1)
        inprog = {}
        for el in all:
            try:
                j = json.loads(el)[1][1]
                params = j['params']
                inprog.setdefault(keyize(params), set()).update(svmstrs)
            except Exception: pass
        tosub = []
        for t in todo:
            sub = inprog.get(keyize(t['params']), set())
            tosub.append(t['params'])
        log('Had %d in progress, %d todo, got %d to submit' % (len(inprog), len(todo), len(tosub)))
        self.submitjobs(tosub)




        p = self.progress
        inq = self.qname('inq', self.jobid)
        outq = self.qname('outq', self.jobid)
        status = self.qname('status', self.jobid)
        last = time.time()
        while len(p['done']) < self.config['maxcombos'] and len(p['todo']) > 0:
            if time.time() - last > self.config['status_interval']:
                log('STATUS (%s): %d done (best score %s), %d todo' % (self.config['name'], len(p['done']), p['best'].get('score', 0), len(p['todo'])))
                last = time.time()
            #print 'At top of loop'
            self.readdone()
            # see if we need to add more to the input q
            while self.db.llen(inq) < self.config['min_inq_size']:
                c = self.newcombo()
                if c:
                    self.submitjobs([c])
                else:
                    break
                self.readdone()
            time.sleep(0.5)
        # at this point, we've ended
        p['end'] = time.time()
        log('All done, with best combo %s!' % (pprint.pformat(p['best']),))

    def parseresult(self, out, done):
        """Parses the result in 'out', updating 'done'."""
        for l in out['stdout'].split('\n'):
            l = l.rstrip('\n')
            if not l: continue
            score, ndims, svmstr = l.split('\t')
            #log('Read a line from stdout: %s: %s, %s, %s' % (l, score, ndims, svmstr))
            score = float(score)
            ndims = int(ndims)

    def paramstr(self, params):
        """Returns a compact param string"""
        return '; '.join('%s:%s' % (p['mstr'], p['fstr']) for p in params)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <dmname> [<dmname> ...]' % (sys.argv[0])
        sys.exit()
    NKDataMatrix('testdm', mode='readdata').mmap.toimage(valrange='data').save('testdm.png'); sys.exit()
