#!/usr/bin/env python
"""SVM Training Script, written by Neeraj Kumar.

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

from nktrainutils import *
from optparse import OptionParser
from itertools import *

VERSION = '1.0'

DEFAULT_CROSS_VALIDATION = 5
DEFAULT_CACHE_DIR = 'cache/'

#TODO see if we can factor common optparse options into nktrainutils

USAGE = '''

Trains one or more SVMs from one set of training data.
The training data is read from stdin, and can be given as raw feature vectors,
or indices to lookup into a set of feature dictionaries.  Multiple SVM
parameter strings can be given in the arguments, and it will train SVM models
for each string. In the end, prints out cross-validation scores, and
(optionally) saves the models to disk.
'''

class SVMTrainer(FeatureParser):
    """A class encapsulating the functionality of SVM Training.
    This is here for convenience in using this module from other
    python scripts, although the module is most optimized for
    running on the command line.
    """
    def __init__(self, featfiles=None, indexmode='', columns='', urlbase='', cachedir=DEFAULT_CACHE_DIR, rediscfgfname=None, errval=None, prescaled=0):
        """Simply all parameters on to base class."""
        FeatureParser.__init__(self, featfiles=featfiles, indexmode=indexmode, columns=columns, urlbase=urlbase, cachedir=cachedir, rediscfgfname=rediscfgfname, errval=errval)
        self.prescaled = prescaled

    def readdata(self, f):
        """Reads data from the given input file.
        Returns (pos, neg)
        The input file can be in one of two formats:
            numbered: The first line contains "npos nneg".
                      The following 'npos' lines are read as positive samples.
                      The following 'nneg' lines are read as negative samples.
            labeled:  Each line contains a label and the fvec, separated by a tab.
        This function figures out which, based on the first line.
        """
        pos, neg = [], []
        npos, nneg = 0, 0
        for i, l in enumerate(f):
            l = l.rstrip('\n')
            #print i, l, npos, nneg, len(pos), len(neg)
            if i == 0:
                # figure out format
                try:
                    npos, nneg = map(int, l.split()) # numbered
                    continue
                except Exception:
                    pass # labeled
            # read row
            if npos and nneg:
                # if numbered, append to appropriate list
                if npos >= len(pos):
                    pos.append(l)
                elif nneg >= len(neg):
                    neg.append(l)
                else:
                    break
            else:
                # if labeled, split into label and fvec
                label, fvec = l.split('\t', 1)
                label = float(label)
                if label > 0:
                    pos.append(fvec)
                else:
                    neg.append(fvec)
        #log('Read %d pos, %d neg: %s, %s' % (len(pos), len(neg), pos[:2], neg[:2]))
        return (pos, neg)

    def getlabelsfeatures(self, pos, neg):
        """Returns (labels, features) from (pos, neg).
        Filters out feature extraction failures."""
        t1 = time.time()
        allinputs = list(pos) + list(neg)
        features = self.getfeatures(allinputs)
        labels = [1]*len(pos) + [-1]*len(neg)
        all = [(id, fvec, l) for id, fvec, l in zip(allinputs, features, labels) if fvec is not None]
        if not all: return ([], [])
        ids, features, labels = zip(*all)
        t2 = time.time()
        return (labels, features)

    @classmethod
    def splitTrainEval(c, pos, neg, ncv):
        """Given the cross-validation value, splits data into training and eval sets.
        Returns ((trainpos, trainneg), (evalpos, evalneg))
        If ncv >= 0, then everything is simply in trainpos and trainneg.
        If ncv < 0, then assumes ncv is negative percentage to use for eval.
        Splits up pos and neg accordingly (proportionally for each).
        """
        if ncv >= 0: return ((pos, neg), ([], []))
        perc = -ncv/100.0
        spos = set(minsample(range(len(pos)), int(perc*len(pos))))
        sneg = set(minsample(range(len(neg)), int(perc*len(neg))))
        trainpos = [p for i, p in enumerate(pos) if i not in spos]
        trainneg = [n for i, n in enumerate(neg) if i not in sneg]
        evalpos = [p for i, p in enumerate(pos) if i in spos]
        evalneg = [n for i, n in enumerate(neg) if i in sneg]
        args = (ncv, perc)+tuple(map(len, (pos, trainpos, evalpos, neg, trainneg, evalneg)))
        log('For ncv %d, got perc %0.2f, and split %d pos into %d, %d and %d neg into %d, %d' % args)
        return ((trainpos, trainneg), (evalpos, evalneg))

    @classmethod
    def evaluate(c, SVMCls, model, labels, features, applyscales=0):
        """Evaluates the given model using the given pos and neg.
        Returns a weighted score (i.e., suitable for unbalanced datasets)"""
        from pprint import pformat
        if features is None: return 0.0
        times = [time.time()]
        results = SVMCls.classify(model, features, applyscales=applyscales)
        times.append(time.time())
        npos = nneg = tpos = tneg = 0
        for v, truth in zip(results, labels):
            if truth >= 0:
                npos += 1
                if v > 0:
                    tpos += 1
            else:
                nneg += 1
                if v <= 0:
                    tneg += 1
        scores = tpos/float(npos), tneg/float(nneg)
        score = sum(scores)/2.0
        times.append(time.time())
        #log('Got tpos %d, npos %d, tneg %d, nneg %d' % (tpos, npos, tneg, nneg))
        log('In eval with %d pos, %d neg (ratio %0.3f), got scores %0.3f, %0.3f = %0.3f, times %s' % (npos, nneg, nneg/float(npos), scores[0], scores[1], score, getTimeDiffs(times)))
        #90/100 + 90/100 = .9
        #90/100 + 50/100 = .7
        #90/100 + 9/10 = 0.9
        #90/100 + 5/10 = 0.7 vs 95/110 = 0.864
        return score

    def train(self, pos, neg, svmstr, ncv=DEFAULT_CROSS_VALIDATION, **kw):
        """Trains a model using the given positive and negative examples.
        Returns (model, score). If ncv is 0, then score is 0.
        You also need to give it a svm parameter string.
        Other params:
            ncv: Number of cross-validation folds, or 0 for no cross-validation.

        This function is a wrapper on nktrainutils.SVM.train()
        """
        from cStringIO import StringIO
        (trainpos, trainneg), (evalpos, evalneg) = self.splitTrainEval(pos, neg, ncv)
        assert len(trainpos) > 0 and len(trainneg) > 0
        t1 = time.time()
        labels, features = self.getlabelsfeatures(trainpos, trainneg)
        t2 = time.time()
        SVMCls, params = parsesvmstr(svmstr)
        SVM = SVMCls()
        scales = [] if self.prescaled else None
        model, score = SVM.train(features, labels, scales=scales, ncv=ncv, **params)
        model.trainkw.update(ntrainpos=len(trainpos), ntrainneg=len(trainneg), nevalpos=len(evalpos), nevalneg=len(evalneg), getlabeltime=t2-t1)
        if evalpos and evalneg:
            # we have an eval set, so evaluate to get real score
            t1 = time.time()
            labels, features = self.getlabelsfeatures(evalpos, evalneg)
            score = self.evaluate(SVM, model, labels, features, applyscales=not self.prescaled)
            t2 = time.time()
            model.trainkw.update(score=score, evaltime=t2-t1)
        return (model, score)

    def trainmany(self, pos, neg, svmstrs, ncv=DEFAULT_CROSS_VALIDATION, **kw):
        """Trains several models.
        Returns (model, score) for each given svmstring, as a list.
        """
        from cStringIO import StringIO
        times = [time.time()]
        (trainpos, trainneg), (evalpos, evalneg) = self.splitTrainEval(pos, neg, ncv)
        labels, features = self.getlabelsfeatures(trainpos, trainneg)
        times.append(time.time())
        log('Got %d pos and %d neg = %d total, vs %d features and %d labels' % (len(trainpos), len(trainneg), len(trainpos)+len(trainneg), len(features), len(labels)))
        if self.prescaled:
            scales = []
        else:
            # also pre-scale, to save time
            scales, features = self.computeScales(features)
            times.append(time.time())
        log('In trainmany, got times %s' % (getTimeDiffs(times)))
        if evalpos and evalneg:
            evallabels, evalfeatures = self.getlabelsfeatures(evalpos, evalneg)
            times.append(time.time())
        for svmstr in svmstrs:
            SVMCls, params = parsesvmstr(svmstr)
            SVM = SVMCls()
            times.append(time.time())
            model, score = SVM.train(features, labels, scales=scales, ncv=ncv, **params)
            times.append(time.time())
            if evalpos and evalneg:
                # we have an eval set, so evaluate to get real score
                score = self.evaluate(SVM, model, evallabels, evalfeatures, applyscales=not self.prescaled)
                times.append(time.time())
                model.trainkw.update(score=score, evaltime=times[-1]-times[-2])
            model.trainkw.update(ntrainpos=len(trainpos), ntrainneg=len(trainneg), nevalpos=len(evalpos), nevalneg=len(evalneg), trainmanytimes=times, svmstr=svmstr, svmstrs=svmstrs)
            #log('Yielding %s' % (score,))
            yield (model, score)

    def save(self, model, outfname):
        """Saves the given model to the given outfname"""
        SVMCls = pickSVMClass(model)
        SVMCls().save(model, outfname)

def main():
    """Main method"""
    # setup command parser
    usage = 'Usage: python %s [opts] <svm string> [<svm string> ...]' % (sys.argv[0]) + USAGE
    parser = OptionParser(usage=usage, version=VERSION)
    parser.add_option('-f', '--featfiles', dest='featfiles', action='append', help='feature dictionary filenames, or assume input contains raw feature vectors if not given')
    parser.add_option('-i', '--indexmode', dest='indexmode', default='', help='how to interpret input indices [default: '']')
    parser.add_option('-C', '--columns', dest='columns', help='0-indexed columns to extract from input (inclusive ranges and/or comma-separated)')
    parser.add_option('-u', '--urlbase', dest='urlbase', default='http://s3.amazonaws.com/neerajkumar-pub/projects/clothing/', help='base url from where to download feature dictionaries')
    parser.add_option('-c', '--cachedir', dest='cachedir', default=DEFAULT_CACHE_DIR, help='cachedir to store downloaded feature dictionaries to [default: %s]' % (DEFAULT_CACHE_DIR))
    parser.add_option('-e', '--errval', dest='errval', type='float', help='If input feature vector consists of entirely this value, then it counts as an error')
    parser.add_option('-o', '--outfmt', dest='outfmt', help='output model filename format (python string style), or no saving of models if not given')
    parser.add_option('-n', '--num-cross-validation', dest='ncv', type='int', default=DEFAULT_CROSS_VALIDATION, help='# of cross-validation folds, or no cross-validation if set to 0, or percentage to use for eval if negative [default: %d]' % DEFAULT_CROSS_VALIDATION)
    parser.add_option('-b', '--best-only', dest='best', action='store_true', default=0, help='only print and save the best-performing svm model, by cross-validation score [default: do all]')
    parser.add_option('-p', '--prescaled', dest='prescaled', action='store_true', default=0, help='features are prescaled [default: compute scaling]')
    # parse arguments
    opts, args = parser.parse_args()
    if not args:
        parser.error('Need at least one SVM string to operate on!')
    svmstrs = args[:]
    #log('Got %d svmstrs: %s' % (len(svmstrs), '::'.join(svmstrs)))
    # create trainer and process
    M = MemUsage()
    t = SVMTrainer(opts.featfiles, indexmode=opts.indexmode, columns=opts.columns, cachedir=opts.cachedir, urlbase=opts.urlbase, errval=opts.errval, prescaled=opts.prescaled)
    M.add('create')
    pos, neg = t.readdata(sys.stdin)
    M.add('readdata')
    #log('Got %d pos, %d neg: %s, %s' % (len(pos), len(neg), pos[:2], neg[:2]))
    def saveandprint(i, model, score, svmstr):
        """Saves the model if necessary, and then prints out the output line."""
        try:
            ndims = len(model.scales)
        except TypeError:
            ndims = len(model.scales.mean_)
        fmtdict = dict(ndims=ndims, score=score, i=i, svmstr=svmstr)
        toprint = [str(score), str(ndims), svmstr]
        # save if needed
        if opts.outfmt:
            outfname = opts.outfmt % fmtdict
            t.save(model, outfname)
            toprint.append(outfname)
        # print the output
        print '\t'.join(toprint)
        sys.stdout.flush()

    if len(svmstrs) == 1:
        # single svm string
        svmstr = svmstrs[0]
        model, score = t.train(pos, neg, svmstr, ncv=opts.ncv)
        saveandprint(0, model, score, svmstr)
    else:
        # multiple svm strings
        results = t.trainmany(pos, neg, svmstrs, ncv=opts.ncv)
        M.add('trained')
        iter = ((score, svmstr, model) for (model, score), svmstr in izip(results, svmstrs))
        if opts.best:
            # get the best one only
            bestscore = 0
            best = None
            for score, svmstr, model in iter:
                M.add()
                if score > bestscore:
                    best = (score, svmstr, model)
                    bestscore = score
                else:
                    del model
            score, svmstr, model = best
            M.add('about to save')
            saveandprint(0, model, score, svmstr)
        else:
            # print all
            for inum, (score, svmstr, model) in enumerate(iter):
                #log('Here in main loop with score %s, svmstr %s' % (score, svmstr))
                saveandprint(inum, model, score, svmstr)
        M.add('done')
        for d in M:
            log('Got d %s' % (d,))


if __name__ == '__main__':
    main()
