#!/usr/bin/env python
"""
Somewhat-greedy, iterative, distributed feature selection for SVM-based classifiers.

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
from nkpylib.nkutils import log
from nkpylib.nkredisutils import *
from nkpylib.nktrainutils import getSVMStrs

# default config parameters
DEFAULT_CFG = dict(
    maxsvmstrs = 8, # maximum number of svm strs per process
    qbase = 'rqs:featsel', # basename for rqs queues
    min_inq_size = 10, # min number of items to maintain in queue
    cmdline = ['python', 'pylib/nksvmtrain.py', '-n5'], # command to call
    default_job_kw = dict(ttl=1800), # job kw
    status_interval = 10, # how frequently to print status
    rqs_cfg = dict(socket_timeout=60, encoder='json'), # default rqs params
    maxcombos = 10000, # maximum number of combos to do
    svmstrs = getSVMStrs([''], ['']), # by default, just a single default RBF svmstr
)

# these are required fields in the config
REQUIRED = 'rqs_cfg name jobid pos neg'.split()

# SMALL UTILITIES
def keyize(dlst):
    """Converts a list of dictionaries into a key"""
    return tuple(sorted([tuple(sorted(d.items())) for d in dlst]))

def chooselabelsubset(lst, n, method):
    """Chooses a subset of the given list, using the given method:
          start: from the beginning
          end: from the end
          random: randomly
    """
    from random import shuffle
    # pick the the trainpos and trainneg sets
    def randindexes(lst):
        """Returns a random list of indexes for the given lst"""
        ret = range(len(lst))
        shuffle(ret)
        return ret

    methdict = dict(start = lambda lst: [i for i, x in enumerate(lst)],
                      end = lambda lst: [i for i, x in enumerate(reversed(lst))],
                   random = randindexes)
    func = methdict[method]
    ret = func(lst)[:n]
    return ret

class FeatureSelector(Thread):
    """Class to perform feature selection.
    This tries to do things in an easily restartable way.
    The configuration is defined in a JSON file, for portability.
    Since it might be a bit complex, it's recommended that you
    write a script to generate the config.
    This class will read it and then maintain a "progress"
    structure keeping track of feature selection progress so far.
    It writes this to a progress json file. This file also
    becomes the results file at the end.
    """
    def __init__(self, configfname, progressfname):
        #RQS_CONFIG = dict(host='arnold.cs.washington.edu', port=10001, password='clothingrqsftw')
        """Initializes this with the config and progress filenames"""
        Thread.__init__(self)
        # set vars
        self.configfname = configfname
        self.progressfname = progressfname
        # parse data files
        c = self.config = json.load(open(configfname))
        for k in REQUIRED:
            assert k in c, 'Field %s must be in config!' % (k)
        # fill in defaults if missing
        for k, v in DEFAULT_CFG.items():
            if k == 'rqs_cfg':
                # for rqs, go down into the corresponding dict
                crqs = c[k]
                for rk, rv in v.iteritems():
                    if rk not in crqs:
                        crqs[rk] = rv
            else:
                # for everything else, just add it if not given
                if k not in c:
                    c[k] = v
        self.jobid = c['jobid']
        # initialize rqs
        self.rqs = RedisQueueService(**c['rqs_cfg'])
        self.db = self.rqs.redis
        self.qname = makeqname(c['qbase'])
        init = 0
        try:
            self.progress = json.load(open(progressfname))
        except IOError:
            init = 1
            self.progress = self.newprogress()
        # set our pos and neg
        self.pos = [c['pos'][p] for p in self.progress['trainpos']]
        self.neg = [c['neg'][n] for n in self.progress['trainneg']]
        if init:
            # initialize all single params, and submit them all
            combos = self.createcombos()
            self.submitcombos(combos)
        else:
            # figure out what's been already submitted from our todos and submit the rest
            #TODO for now, we just go through the inq and outq, rather than anything fancier
            inq = self.qname('inq', self.jobid)
            outq = self.qname('outq', self.jobid)
            all = self.db.lrange(inq, 0, -1) + self.db.lrange(outq, 0, -1)
            inprog = {}
            for el in all:
                try:
                    j = json.loads(el)[1][1]
                    params = j['params']
                    svmstrs = j['svmstrs']
                    inprog.setdefault(keyize(params), set()).update(svmstrs)
                except Exception: pass
            #print 'Got %d inprog: %s' % (len(inprog), pprint.pformat(inprog))
            todo = self.progress['todo']
            #print 'Had %d todo: %s' % (len(todo), pprint.pformat(todo))
            tosub = []
            for t in todo:
                sub = inprog.get(keyize(t['params']), set())
                left = set(t['svmstrs']) - sub
                if not left: continue # there was nothing left to submit for this job
                # if we're here, then there is something to submit, so add this
                tosub.append(t['params'])
            log('Had %d in progress, %d todo, got %d to submit' % (len(inprog), len(todo), len(tosub)))
            self.submitcombos(tosub)
        # write our progress to disk, in case we had a new one
        self.writeprogress()

    def cleanup(self):
        """Runs various cleanup on ourselves:
            - Removes things that are done from todo
            - Computes new best
            - Sets some instance variables
        """
        c = self.config
        p = self.progress
        existing = {}
        # one pass to update bests and figure out all the things we've done
        for d in p['done']:
            # add the list of svmstrs we've done for this set of params
            existing.setdefault(keyize(d['params']), set()).update(d['scores'])
            # update the global best
            b = d['best']
            if not p['best'] or p['best']['score'] < b['score']:
                p['best'] = dict(params=d['params'], ndims=d['ndims'], **b)
                p['best']['updated'] = time.time()
        # one pass to prune todo's svmstrs
        for t in p['todo']:
            kp = keyize(t['params'])
            if kp not in existing: continue
            #print 'Got kp %s, ekp %s' % (kp, existing[kp])
            # if we're here, then we've done at least a few of the svmstrs, so filter them out
            t['svmstrs'] = [s for s in t['svmstrs'] if s not in existing[kp]]
        # one pass to kill todos with no svmstrs
        p['todo'] = [t for t in p['todo'] if t['svmstrs']]
        # set some instance vars
        p['ntodo'] = len(p['todo'])
        p['ndone'] = len(p['done'])
        p['updated'] = time.time()

    def writeprogress(self):
        """Writes our progress to file"""
        #TODO include mutex here
        import tempfile
        outf, tmpfname = tempfile.mkstemp(dir=os.path.dirname(self.progressfname))
        self.cleanup()
        json.dump(self.progress, os.fdopen(outf, 'wb'), sort_keys=1, indent=2)
        try:
            os.rename(tmpfname, self.progressfname)
        except OSError:
            pass

    def newprogress(self):
        """Creates a new progress structure, filling in values from our config."""
        ret = dict(start=time.time(), end=None, elapsed=0, done=[], todo=[])
        ret.update(dict(configfname=self.configfname, trainpos=[], trainneg=[], best={}))
        # pick the the trainpos and trainneg sets
        ret['trainpos'], ret['trainneg'] = self.createtrainset()
        ret['ntrainpos'] = len(ret['trainpos'])
        ret['ntrainneg'] = len(ret['trainneg'])
        return ret

    def submitcombos(self, combos):
        """Adds a new combination of params to work on, using an rqs q"""
        c = self.config
        p = self.progress
        inq = self.qname('inq', self.jobid)
        allinq = self.qname('inq')
        def inqcallback(id, item, qname, rqs):
            """Sets the status to inq"""
            rqs.setstatusmsg(qname.replace(':inq', ':status'), id, 'inq')

        for params in combos:
            # break into subjobs by sets of svmstrs
            for svmstrs in nkgrouper(c['maxsvmstrs'], c['svmstrs']):
                realid = time.time()
                # create the job structure
                job = self.createjob(params, svmstrs=svmstrs[:], id=realid, jobid=self.jobid, submitted=time.time(), **c['default_job_kw'])
                # submit it
                log('Adding job with %d params %s, with %d svmstrs, %d pos, %d neg' % (len(params), self.paramstr(params), len(svmstrs), len(self.pos), len(self.neg)))
                item = (realid, job)
                self.rqs.put(realid, item, inq, callback=inqcallback)
                #TODO figure this out
                if 1: # this seems bad if one set of jobs are much slower than the other
                    toadd = {self.jobid: 1000}
                    self.rqs.redis.zadd(allinq, **toadd)
                else:
                    self.rqs.redis.zincrby(allinq, self.jobid, 1)
                # add to list of todos
                p['todo'].append(dict(params=params, svmstrs=svmstrs, submitted=time.time(), realid=realid))
        # write our progress to disk
        self.writeprogress()

    def readresults(self, out):
        """Parses a complete set of results"""
        c = self.config
        p = self.progress
        params = out['params']
        # first find the relevant "done" struct, if it exists, else add one
        found = 0
        for d in p['done']:
            if d['params'] == params:
                found = 1
                break
        if not found:
            d = dict(params=params, scores={}, elapsed=0, best={})
            p['done'].append(d)
        # now copy info from it
        d['elapsed'] += out['elapsed']
        p['elapsed'] += out['elapsed']
        try:
            self.parseresult(out, d)
            d['end'] = time.time()
            log('Read results for %d params %s, with best %s, and global best %s' % (len(d['params']), self.paramstr(d['params']), d['best']['score'], p['best'].get('score', 0)))
        except Exception, e:
            log('Had an error trying to read results for %d params %s: %s' % (len(d['params']), self.paramstr(d['params']), e))
        self.writeprogress()

    def joinparams(self, p1, p2):
        """Joins two parameters lists to make a new parameter list"""
        nc = p1 + p2
        return nc

    def newcombo(self):
        """Creates a new feature combination and returns (realid, (realid, combo))."""
        c = self.config
        p = self.progress
        # precompute some lists
        existing = set(keyize(t['params']) for t in p['todo'])
        existing.update(keyize(d['params']) for d in p['done'])
        singles = [d for d in p['done'] if len(d['params']) ==1]
        combos = []
        # build up a list of combos with priorities
        for d in p['done']:
            for s in singles:
                if d == s: continue # skip itself
                if s['params'][0] in d['params']: continue # skip dones which contain this single's params
                nc = self.joinparams(d['params'], s['params'])
                if keyize(nc) in existing: continue # skip things we've done
                existing.add(keyize(nc))
                # if we're here, then this is a legal new combo, so prioritize it
                pri = self.getpriority(s, nc, d)
                combos.append((pri, nc))
        # choose the best combo
        if 0:
            print 'Came up with %d combos:' % (len(combos))
            for c in sorted(combos):
                print '  ', c
        if not combos: return None
        pri, nc = max(combos)
        return nc

    def readdone(self):
        """Reads any that are done"""
        outq = self.qname('outq', self.jobid)
        status = self.qname('status', self.jobid)
        while 1:
            obj = self.db.rpop(outq)
            if not obj: break
            idpair, item = json.loads(obj)
            realid, out = item
            self.readresults(out)
            self.db.hdel(status, realid)

    def run(self):
        """Runs feature selection, picking up where we left off.
        Note that since we inherit from Thread, you can call
        start() to run this in a new Thread."""
        #TODO what's the advantage to running this in a thread?
        import Queue
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
                    self.submitcombos([c])
                else:
                    break
                self.readdone()
            time.sleep(0.5)
        # at this point, we've ended
        p['end'] = time.time()
        log('All done, with best combo %s!' % (pprint.pformat(p['best']),))

    def createtrainset(self):
        """Creates a training set for ourselves"""
        c = self.config
        pos = chooselabelsubset(c['pos'], c['nfselpos'], c['trainpickmeth'])
        neg = chooselabelsubset(c['neg'], c['nfselneg'], c['trainpickmeth'])
        return (pos, neg)

    def createcombos(self):
        """Creates a list of parameter combinations."""
        raise NotImplementedError

    def createjob(self, params, **kw):
        """Creates a job structure to submit, using the given parameters and other kw."""
        raise NotImplementedError

    def parseresult(self, out, done):
        """Parses the result in 'out', updating 'done'."""
        raise NotImplementedError

    def getpriority(self, single, combo, done):
        """Returns a priority for the given single added to the given combo, given the done"""
        raise NotImplementedError

    def paramstr(self, params):
        """Returns a compact param string"""
        raise NotImplementedError


class ClothingFeatureSelector(FeatureSelector):
    """Feature selector for clothing project"""
    def createcombos(self):
        """Creates a list of parameter combinations."""
        c = self.config
        combos = [[dict(mstr=mstr, fstr=fstr)] for mstr in c['mstrs'] for fstr in c['fstrs']]
        return combos

    def createjob(self, params, **kw):
        """Creates a job structure to submit, using the given parameters and other kw."""
        job = dict(**kw)
        job['params'] = params[:]
        c = self.config
        # fill out the command line
        job['cmdline'] = c['cmdline'][:]
        featfiles = ['feats50/%(mstr)s_%(fstr)s.txt' % f for f in params]
        featfiles = sum([['-f', f] for f in featfiles], [])
        job['cmdline'].extend(featfiles)
        job['cmdline'].extend(svmstrs)
        # fill out the stdin
        s = '%d %d\n' % (len(self.pos), len(self.neg))
        for id in self.pos+self.neg:
            s += '%s\n' % (id)
        job['stdin'] = s
        return job

    def parseresult(self, out, done):
        """Parses the result in 'out', updating 'done'."""
        for l in out['stdout'].split('\n'):
            l = l.rstrip('\n')
            if not l: continue
            score, ndims, svmstr = l.split('\t')
            #log('Read a line from stdout: %s: %s, %s, %s' % (l, score, ndims, svmstr))
            score = float(score)
            ndims = int(ndims)
            done['ndims'] = ndims
            done['scores'][svmstr] = score
            if not done['best'] or done['best']['score'] < score:
                done['best'] = dict(score=score, svmstr=svmstr, updated=time.time())

    def getpriority(self, single, combo, done):
        """Returns a priority for the given single added to the given combo, given the done"""
        c = self.config
        pri = done['best']['score'] + single['best']['score'] # sum of their scores
        #TODO generalize/formalize this priority code
        pri -= c['ncombosfac'] * len(combo) # subtract by the length of this feature set
        pri -= (done['ndims'] + single['ndims']) ** c['ndimsfac']  # subtract by (sum of ndims) ** fac
        return pri

    def paramstr(self, params):
        """Returns a compact param string"""
        return '; '.join('%s:%s' % (p['mstr'], p['fstr']) for p in params)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python %s <config file> <progress file>' % (sys.argv[0])
        sys.exit()
    FeatureSelector(sys.argv[1], sys.argv[2]).run()
