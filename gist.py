#!/usr/bin/env python
"""Gist-related feature-extraction related utilities, written by Neeraj Kumar.
Note that many of the generic (non-gist-related) functions might be broken here.

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

import os, sys, time
from PIL import Image
from threadutils import spawnWorkers

def featxloop(func, inq, outq, callback=None):
    """A feature extraction loop.
    Reads fnames from inq, runs them through func, and puts (fname, feats) on outq."""
    while 1:
        fname = inq.get()
        try:
            im = Image.open(fname)
            feats = func(im)
        except IOError:
            feats = None
            #TODO see if this is the right thing to do
        if callback:
            callback(fname, feats)
        outq.put((fname, feats))


def q2dict(q, d):
    """Read (fname, obj) from queue and add to dict: d[fname] = obj"""
    while 1:
        fname, obj = q.get()
        d[fname] = obj

class FeatureExtractor(object):
    """A wrapper on feature extraction, that does it in parallel."""
    def __init__(self, func, makedict=1, nthreads=5, inq=None, outq=None, callback=None):
        """Set the function you want to use for extraction and the number of threads.
        The function should take a PIL Image as input and return an array of feature values.
        You can then directly add filenames to be feature extracted to this.inq.

        If 'makedict' is true (the default), then there will be an instance variable
        called 'featdict' which will be populated with the feature values as they are computed.
        This is a dict mapping fnames (from the inq) to the feature values.
        If you want to later use the 'getmultiple()' function, 'makedict' needs to be true.

        If this is false, then you can directly access the output queue yourself
        by calling this.outq.get(). Note that if using multiple threads, then you're not
        guaranteed that outputs will be in the same order as the inputs.

        You can optionally pass in the input and output queues.
        If you don't, then they are created.

        You can optionally pass in a callback called when a feature is extracted for an image.
        It is called with (fname, feats)
        """
        from Queue import Queue
        # set params
        self.func = func
        self.nthreads = nthreads
        if not inq: inq = Queue()
        if not outq: outq = Queue()
        self.inq, self.outq = inq, outq
        self.callback = callback
        # start threads
        self.featxs = spawnWorkers(self.nthreads, featxloop, args=(self.func, self.inq, self.outq, self.callback), interval=0.1)
        if makedict:
            self.featdict = {}
            self.q2dictthread = spawnWorkers(1, q2dict, args=(self.outq, self.featdict))
        else:
            self.featdict = None

    def getmultiple(self, fnames, blocking=1, defaultval=None, delay=0.1):
        """Returns the feature values for the given fnames.
        Returns a sequence of fvals, not a dict.
        If blocking is true (default), then blocks until everything is ready.
        This is implemented as a sleep() with the given delay.
        Note that this could take forever if there's some problem!
        If not, then fills in missing values with 'defaultval'.

        In either case, you're guaranteed to get the same number of elements
        in the output as in fnames.
        """
        assert self.featdict is not None
        ret = []
        for f in fnames:
            gotit = 0
            while not gotit:
                try:
                    ret.append(self.featdict[f])
                    gotit = 1
                except KeyError:
                    #print '  Waiting for %s' % (f)
                    if blocking:
                        time.sleep(delay)
                    else:
                        ret.append(defaultval)
                        gotit = 1
        return ret

def gistfeatures(im, maxsize=None):
    """Extracts gist features (dims=960) from the given image.
    Optionally resizes the image using the thumbnail() function to maxsize.

    Uses the pyleargist library:
        http://pypi.python.org/pypi/pyleargist/
         or
        sudo easy_install pyleargist

    GIST extraction seems to scale about linearly with number of pixels,
    so resizing is often essential for fast speed.

    On error, returns None
    """
    import leargist
    if maxsize:
        #im.thumbnail(maxsize, Image.ANTIALIAS)
        im = im.resize(maxsize, Image.ANTIALIAS)
    try:
        ret = leargist.color_gist(im)
    except Exception:
        # some problem with leargist
        return None
    return ret

def gisttrain(outfname, maxsize=None):
    """Runs gist training on each of the given training filenames"""
    import cPickle as pickle
    from trainutils import trainSingleSVM, saveSVMModelAndParams, runGridSearch, getSVMStrs
    def callback(fname, feats):
        if feats is not None:
            print '  For %s, got %d-dim featvec: %s' % (fname, len(feats), feats[:3])

    fx = FeatureExtractor(lambda im: gistfeatures(im, maxsize=maxsize), nthreads=21, callback=callback)
    npos = int(sys.stdin.readline().strip())
    nneg = int(sys.stdin.readline().strip())
    posfnames = [sys.stdin.readline().rstrip('\n') for i in range(npos)]
    negfnames = [sys.stdin.readline().rstrip('\n') for i in range(nneg)]
    for fname in posfnames+negfnames:
        fx.inq.put(fname)
    posfvals = fx.getmultiple(posfnames)
    negfvals = fx.getmultiple(negfnames)
    posfnames, posfvals = zip(*[(fname, fvals) for fname, fvals in zip(posfnames, posfvals) if fvals is not None])
    negfnames, negfvals = zip(*[(fname, fvals) for fname, fvals in zip(negfnames, negfvals) if fvals is not None])
    print len(posfvals), len(negfvals)
    labels = [1] * len(posfvals) + [-1] * len(negfvals)
    print labels
    features = posfvals + negfvals
    svmstr = 'svm_type=C_SVC, kernel_type=RBF'
    if 1:
        results = model, score, svmstr = runGridSearch(labels, features, getSVMStrs([1,10,100,1000], [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))
    else:
        results = model, score = trainSingleSVM(labels, features, svmstr)
    print results
    print len(model.scales)
    saveSVMModelAndParams(model, outfname)

def gisttest(modelfname, maxsize=None):
    """Runs gist classification on all fnames passed in through stdin"""
    from trainutils import readSVMModelAndParams, bulkclassify
    from utils import stdmainloop
    # read the model
    model = readSVMModelAndParams(modelfname)
    assert model.scales
    fx = FeatureExtractor(lambda im: gistfeatures(im, maxsize=maxsize), makedict=0, nthreads=11)

    # setup the classification, in other threads
    def classifyfunc(fx=fx, model=model, timeout=1):
        from threadutils import feastOnQueue
        while 1:
            t = time.time()
            els = [(fname, feats) for fname, feats in feastOnQueue(fx.outq, timeout) if feats is not None]
            if els:
                fnames, feats = zip(*els)
                scores = [l*v for l, v in bulkclassify(model, feats)]
                for score, fname in zip(scores, fnames):
                    print '%s\t%s' % (score, fname)
            elapsed = time.time()-t
            if els:
                print >>sys.stderr, 'Classified %d els in %0.3fs, each of len %d' % (len(feats), elapsed, len(feats[0]))
            time.sleep(max(0, timeout-elapsed))

    clsworkers = spawnWorkers(5, classifyfunc)

    # start reading from stdin
    stdmainloop(lambda fname: fx.inq.put(fname))

    time.sleep(10)

def fullmain():
    maxsize = ((400,400))
    tasks = 'train test'.split()
    if len(sys.argv) < 3:
        print 'Usage: python %s <%s> <model name>' % (sys.argv[0], '|'.join(tasks))
        sys.exit()
    task, fname = sys.argv[1:3]
    assert task in tasks
    if task == 'train':
        t1 = time.time()
        gisttrain(outfname=fname, maxsize=maxsize)
        print time.time()-t1
    elif task == 'test':
        gisttest(fname, maxsize=maxsize)

def testmain():
    """A simple main to process gist features streaming"""
    from utils import getListAsStr
    w, h = map(int, sys.argv[1:3])
    def do(fname):
        try:
            im = Image.open(fname)
            fvec = gistfeatures(im, maxsize=(w,h))
            print '%s\t%s' % (fname, getListAsStr(fvec, ' '))
            sys.stdout.flush()
        except Exception:
            print '%s\t' % (fname)

    if len(sys.argv) > 3:
        for fname in sys.argv[3:]:
            do(fname)
    else:
        while 1:
            fname = sys.stdin.readline().rstrip('\n')
            if not fname: break
            do(fname)


if __name__ == '__main__':
    testmain()
