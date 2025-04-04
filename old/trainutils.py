#!/usr/bin/env python
"""Lots of training- and classification-related utilities, written (mostly) by Neeraj Kumar.

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

import os, sys, random, math, time
from utils import *
from math import pi
from itertools import *

USE_SCALER = 1

# SMALL UTILITIES
def getSVMStrs(Cs, gammas=None):
    """Returns svm classification strings for the given cs and gammas.
    If gammas is None, then assumes we want a linear svm.
    If gamms is a list with 0 or more elements, then assumes RBF."""
    ret = []
    if gammas is None:
        for c in Cs:
            svmstr = 'svm_type=C_SVC, kernel_type=LINEAR'
            if c:
                svmstr += ', C=%s' % (c,)
            ret.append(svmstr)
    else:
        for c, g in xcombine(Cs, gammas):
            svmstr = 'svm_type=C_SVC, kernel_type=RBF'
            if c: svmstr += ', C=%s' % (c,)
            if g: svmstr += ', gamma=%s' % (g,)
            ret.append(svmstr)
    return ret

def getLinearSVMStrs(Cs, s=1):
    """Returns linear svm strings.
    Cs is the set of slack variables to try.
    s is the solver to use:
        1 -- L2-regularized L2-loss support vector classification (dual) [default]
        2 -- L2-regularized L2-loss support vector classification (primal)
        3 -- L2-regularized L1-loss support vector classification (dual)
        5 -- L1-regularized L2-loss support vector classification
    """
    assert int(s) in [1,2,3,5]
    ret = []
    for c in Cs:
        svmstr = '-q -s %s' % (s,)
        if c:
            svmstr += ' -c %s' % (c,)
        ret.append(svmstr)
    return ret

def parseSVMType(params):
    """Parses a given set of params and returns one of:
        'libsvm': if params is a string and it's for libsvm
        'liblinear': if params is a string for liblinear
        'sgd': if params is a mapping
    """
    if isinstance(params, basestring):
        if 'svm_type' in params or 'kernel_type' in params: return 'libsvm'
        if '-s' in params: return 'liblinear'
    elif isinstance(params, dict):
        return 'sgd'
    raise ValueError('Could not figure out svm type from params %s' % (params,))

def getRegressionSVMStrs(Cs, gammas, epsilons):
    """Returns svm regression strings for the given Cs, gammas and epsilons"""
    ret = []
    for c, g, e in xcombine(Cs, gammas, epsilons):
        svmstr = 'svm_type=EPSILON_SVR, kernel_type=RBF'
        if c: svmstr += ', C=%s' % (c,)
        if g: svmstr += ', gamma=%s' % (g,)
        if e: svmstr += ', p=%s' % (e,)
        ret.append(svmstr)
    return ret

def applyScales(fvals, scales, EPSILON=0.000001):
    """Applies the given normalization scales to the set of feature values and returns a normalized set.
    The scales sequence should contain (mean, stdev) pairs.
    The normalization applied is norm = (value-mean)/(2*(stdev+EPSILON)).
    """
    from array import array
    if not scales: return fvals
    assert len(fvals) == len(scales), 'Scales had length %d but fvals had length %d' % (len(scales), len(fvals))
    if 0:
        # Simple code, but slow
        fvals = array('f', [(f-m)/(2*(s+EPSILON)) for f, (m,s) in zip(fvals, scales)])
    else:
        # more involved, but faster
        import numpy
        fvals = numpy.array(fvals)
        x = numpy.array(scales)
        m, s = x[:,0], x[:,1]
        fvals = (fvals-m)/(2*(s+EPSILON))
        fvals = array('f', fvals)
        #TODO see if this code below is faster
        if 0:
            means = numpy.array([m for (m,s) in scales])
            stds = numpy.array([s for (m,s) in scales])
            fvals = (fvals-means)/(2*(stds+0.000001))
            return numpy.cast['float32'](fvals)
    return fvals


# DATA MANIPULATION
def createCrossValidationSets(pos, neg, ncv, negscore=-1):
    """Takes positive and negative examples and splits them into ncv partitions.
    Returns a (training, testing) pair for each fold.
    Within each training and testing is (data, labels), where labels are 1 for pos, -1 for neg (unless otherwise specified)"""
    from random import randint
    poslabels = [randint(0, ncv-1) for p in pos]
    neglabels = [randint(0, ncv-1) for n in neg]
    ret = []
    sizes = []
    # go through and create testing/training sets
    for i in range(ncv):
        postraining = [(p, 1) for p, l in zip(pos, poslabels) if l != i]
        postesting = [(p, 1) for p, l in zip(pos, poslabels) if l == i]
        negtraining = [(n, negscore) for n, l in zip(neg, neglabels) if l != i]
        negtesting = [(n, negscore) for n, l in zip(neg, neglabels) if l == i]
        train = postraining + negtraining
        test = postesting + negtesting
        ret.append((zip(*train), zip(*test)))
        sizes.append((len(train), len(test)))
    print >>sys.stderr, "For %d-fold validation, got %d sets, of sizes: %s" % (ncv, len(ret), sizes)
    return ret

def getTrainingData(data, featdict, featfunc):
    """Given the data and a feature function, return the training data as (features, labels, inputs).
    The data should be a dictionary of {1: pos examples, -1: neg examples}.
    The featfunc should take an example from the data and return a list of featvals"""
    features, labels, inputs = [], [], []
    for label in [1, -1]:
        for input in data[label]:
            fvals = featfunc(input, featdict)
            if not fvals: continue
            inputs.append(input)
            features.append(fvals)
            labels.append(label)
    return (features, labels, inputs)

def getSimFeaturesFromFvals(fvals1, fvals2, meths):
    """Returns similarity features computed from the features for two objects.
    'meths' is a list of (meth, weight), where meth is one of the following
    methods for combining values 'a' and 'b' from the two input feature vectors:
        absdiff: |a-b|
         diffsq: (a-b)^2
           prod: a*b
            avg: (a+b)/2
         concat: concatenate fvals1 and fvals2 into a long vector
    If the weight val is > 0, then weights differences using a gaussian of the given variance.
    Returns the computed feature vector.
    """
    from array import array
    allmeths = {'absdiff': lambda a,b: abs(a-b), 'diffsq': lambda a,b: (a-b)**2, 'prod': lambda a,b: a*b, 'avg': lambda a,b: (a+b)/2.0, 'concat': None}
    ret = array('d')
    for meth, weighted in meths:
        assert meth in allmeths
        variance = weighted
        try:
            if meth == 'concat':
                ret.extend(fvals1+fvals2) # concat
            else:
                func = allmeths[meth]
                if weighted > 0:
                    func = lambda a,b: allmeths[meth](a, b) * gaussian((a+b)/2.0, 0.0, weighted)
                ret.extend([func(f1,f2) for f1, f2 in zip(fvals1, fvals2)])
        except KeyError: pass
    return ret


# COMPLETE PIPELINES
def computeSimilarityScores(inputs, fvecfunc, combfunc, clsfunc):
    """Computes similarity scores for all pairs in the given set of inputs.
    Exhaustively goes through all pairs of inputs, and looks up feature vectors
    using the given fvecfunc. Pairs of feature vectors are combined using the
    combfunc. These are fed into clsfunc (as a list) to get a score.
    Returns a dict mapping pairs (input1, input2) -> similarity score.
    """
    ret = []
    for i, i1 in enumerate(inputs):
        f1 = fvecfunc(i1)
        for i2 in inputs[i+1:]:
            f2 = fvecfunc(i2)
            fvec = combfunc(f1, f2)
            ret.append((i1,i2,fvec))
    allin1, allin2, allfvecs = zip(*ret)
    scores = clsfunc(allfvecs)
    ret = dict(((i1,i2), s) for (i1,i2,fvec), s in zip(ret, scores))
    return ret


# SVM DATA-FILE FUNCTIONS
def testmodel(path):
    """Tests the file at the given path to see if the svm model file is complete"""
    try:
        lines = [l.strip() for l in open(path)]
    except IOError: return 0
    num = 0
    # find the line where the number of support vectors is defined
    for i, l in enumerate(lines):
        if l.startswith('nr_feature'): return 1 # assume liblinear files to always be fine
        if l.startswith('nr_sv'):
            els = l.split()
            try:
                num = int(els[1]) + int(els[2])
            except IndexError: return 0 # there weren't enough elements here
            break
    if not num: return 0 # we didn't even get to the nr_sv line
    if len(lines) < (i+num+2): return 0 # 2 for the nr_sv line and the SV lines
    return 1

def parseSVMModel(fname, ndimsonly=0):
    """Parses an svm model file.
    If ndimsonly is false (default), then returns (vals, labels, ndims), where:
        vals is a dict mapping names to values
        For libsvm:
            labels is a dict mapping labels to [alpha, list_of_weights] for each item.
        For liblinear:
            labels is a dict mapping labels to lists of weights.
        ndims is the number of feature dimensions
    Else if ndimsonly is true, then returns (0, 0, ndims).
    """
    assert testmodel(fname)
    lines = [l.strip() for l in open(fname) if l.strip()]
    vals = {}
    init = 1
    # read the initial data
    for i, l in enumerate(lines):
        if l == 'SV' or l == 'w': break # we're done
        name, val = l.split(' ', 1)
        if name in 'label nr_sv'.split():
            val = map(int,val.split())
        vals[name] = val
    # now read the support vectors
    labels = {}
    lines = lines[i+1:]
    if 'solver_type' in vals: #liblinear
        ndims = int(vals['nr_feature'])
        if ndimsonly: return (0, 0, ndims)
        weights = [map(float, l.strip().split()) for l in lines]
        assert len(weights) == int(vals['nr_feature'])
        weights = zip(*weights)
        nlabels = len(vals['label'])
        assert nlabels == 2 or len(weights) == nlabels
        labels = dict((l, w) for l, w in zip(vals['label'], weights))
    elif 'svm_type' in vals: # libsvm
        def parseline(l):
            """Parses a line from a libsvm file"""
            els = l.strip().split(' ')
            alpha = float(els[0])
            weights = [e.split(':') for e in els[1:]]
            weights = dict((int(w[0]), float(w[1])) for w in weights)
            ret = [0]*(max(weights)+1)
            for i, w in weights.iteritems():
                ret[i] = w
            return [alpha, ret]

        ndims = len(parseline(lines[0])[1])
        if ndimsonly: return (0, 0, ndims)
        for num, label in zip(vals['nr_sv'], vals['label']):
            labels[label] = [parseline(l) for l in lines[:num]]
            lines = lines[num:]
    return (vals, labels, ndims)

def svmModelFileType(modelfname):
    """Returns either 'libsvm' or 'liblinear' depending on type of model file"""
    d, vals, ndims = parseSVMModel(modelfname)
    if 'solver_type' in d: return 'liblinear'
    return 'libsvm'

def getSVMDims(fname):
    """Returns the number of dimensions data should be for a given svm.
    This is just a convenience wrapper on parseSVMModel(fname, ndimsonly=1)"""
    junk1, junk2, ndims = parseSVMModel(fname, ndimsonly=1)
    return ndims

def getLinearSVMWeights(d, labels, ndims):
    """Using the output from parseSVMModel(), this gets the weights of each dim for a linear svm."""
    assert set(d['label']) == set((-1, 1))
    if 'solver_type' in d:
        # liblinear
        assert len(labels) == 1
        return [-w for w in labels.values()[0]]
    # libsvm
    assert d['svm_type'] == 'c_svc'
    assert d['kernel_type'] == 'linear'
    b = float(d['rho'])
    weights = [0] * ndims
    for l, lst in labels.iteritems():
        for i, (alpha, vals) in enumerate(lst):
            weights = [c+(alpha*v) for c, v in zip(weights, vals)]
    weights = [-w for w in weights]
    return weights

def getSVMCoefficients(fname):
    """Gets the weight coefficients from an svm model file.
    This is useful to get the equation of the separating hyperplane.
    """
    #TODO not tested recently. Was taken from conjtrain.py
    import operator as op
    found = 0
    rho = 0
    labels = []
    weights = []
    for line in open(fname):
        l = line.strip()
        if not found:
            if l == 'SV': found = 1
            if l.startswith('rho'): rho = float(l.split()[1])
            if l.startswith('label'): labels = map(int, l.split()[1:])
            continue
        els = l.split()
        alpha = float(els[0])
        sv = [alpha*float(v.split(':')[1]) for v in els[1:]]
        if not weights: weights = [0]*len(sv)
        weights = map(op.add, sv, weights)
    return weights, rho

def getModelAndParamsFnames(fname, modelext='.model', paramext='.param'):
    """Returns the filenames to use for the model and params files.
    If the fname ends with modelext:
        model  -> fname
        params -> fname.replace(modelext, paramext)
    Else:
        model  -> fname+modelext
        params -> fname+paramext
    Returns (modelfname, paramfname)
    """
    if fname.endswith(modelext):
        fname = fname.rsplit(modelext,1)[0]
    modelfname = fname + modelext
    paramfname = fname + paramext
    return (modelfname, paramfname)

def saveSVMModelAndParams(model, fname, modelext='.model', paramext='.param'):
    """Saves the svm model and the 'scales' variable to separate files.
    The outputs depend on if the fname ends with modelext
    If yes:
        model  -> fname
        params -> fname.replace(modelext, paramext)
    Else:
        model  -> fname+modelext
        params -> fname+paramext
    """
    try:
        import liblinear.liblinearutil as ll
    except ImportError:
        ll = None
    modelfname, paramfname = getModelAndParamsFnames(fname, modelext, paramext)
    try:
        os.makedirs(os.path.dirname(modelfname))
    except OSError: pass
    try:
        # in case they're not in the same dir
        os.makedirs(os.path.dirname(paramfname))
    except OSError: pass
    if ll and isinstance(model, ll.model):
        ll.save_model(modelfname, model)
    else:
        try:
            model.save(modelfname)
        except Exception:
            pass #FIXME
    if 'scales' not in dir(model):
        model.scales = []
    open(paramfname, 'w').write(str(model.scales))

def readSVMModelAndParams(fname, modelext='.model', paramext='.param'):
    """Reads svm model and its normalizing scales.
    The filenames are generated exactly as in saveSVMModelAndParams:
    If the fname ends with modelext:
        model  -> fname
        params -> fname.replace(modelext, paramext)
    Else:
        model  -> fname+modelext
        params -> fname+paramext
    Returns the model, with the scales in it as model.scales
    """
    modelfname, paramfname = getModelAndParamsFnames(fname, modelext, paramext)
    svmtype = svmModelFileType(modelfname)
    if svmtype == 'libsvm':
        from svm import svm_model
        model = svm_model(modelfname)
    elif svmtype == 'liblinear':
        import liblinear.liblinearutil as ll
        model = ll.load_model(modelfname)
    model.scales = eval(open(paramfname).readline().rstrip('\n'))
    return model


# CLASSIFIERS AND ERROR RATES, ETC.
def makeVerificationPairsFromClsDict(data, npos, nneg):
    """Makes npos positive and nneg negative pairs from the given data.
    The data is given a dictionary of cls->data.
    Returns (pos, neg), which are both sets.
    Pairs are sorted, to test for equality.
    Note that this can get stuck in an infinite loop if too many pairs are requested."""
    import random
    ckeys = data.keys()
    pos, neg = set(), set()
    # make positive pairs by choosing a random class and 2 els within that class
    while len(pos) < npos:
        c = random.choice(ckeys)
        pair = tuple(sorted(random.sample(data[c], 2)))
        pos.add(pair)
    # make negative pairs by choosing 2 random classes and 1 el within each class
    while len(neg) < nneg:
        c1, c2 = random.sample(ckeys, 2)
        pair = tuple(sorted((random.choice(data[c1]), random.choice(data[c2]))))
        neg.add(pair)
    return pos, neg

def makeVerificationPairs(data, npos, nneg, clsfunc):
    """Makes npos positive and nneg negative pairs from the given data.
    Returns (pos, neg), which are both sets.
    The output of clsfunc(data[i]) is used to determine classes,
    and == is used to compare classes.
    Pairs are sorted, to test for equality.
    Note that this can get stuck in an infinite loop if too many pairs are requested."""
    import random
    clses, indices = partitionByFunc(data, clsfunc)
    return makeVerificationPairsFromClsDict(clses, npos, nneg)

def getPrecisionRecall(data):
    """Returns (recall, precision) pairs"""
    recall = 0

def getErrorRates(data, thresh):
    """Returns the true positive (detection or recall rate), false positive,
    true negative, false negative, and precision rates, given some data and a threshold.
    Precision is pos/(pos+fpos).
    The data should be pairs of (value, label) pairs. For sweeping a threshold over a set of values,
    use the getAllErrorRates function instead."""
    # compute counts
    pos = sum(1.0 for c, l in data if c > thresh and l > 0)
    fpos = sum(1.0 for c, l in data if c > thresh and l <= 0)
    neg = sum(1.0 for c, l in data if c <= thresh and l <= 0)
    fneg = sum(1.0 for c, l in data if c <= thresh and l > 0)
    npos = sum(1.0 for c, l in data if l > 0)
    nneg = sum(1.0 for c, l in data if l <= 0)
    # normalize values
    try:
        precision = pos/(pos+fpos)
    except ZeroDivisionError:
        precision = 0.0
    pos /= npos
    fpos /= nneg
    neg /= nneg
    fneg /= npos
    return (pos, fpos, neg, fneg, precision)

def getROCValues(results, incr=0.5, roctype='roc'):
    """Returns ROC values for the given results, by changing the threshold.
    Results should be pairs of (cls output, labels).
    If an increment is given and it's a float, it's used to sweep through values.
    If an increment is given and it's an int, it's used as a subsampling through results.
    If incr is None, then all possible thresholds are tried.
    The roctype determines which types of values to return:
        'roc' [default] : true positive/detection/recall rate, false positive rate
        'precision'     : true positive/detection/recall rate, precision rate
    """
    cls, labels = zip(*results)
    scls = sorted(cls)
    if incr and incr > 0 and type(incr) == type(1.3):
        threshes = arange(min(cls), max(cls), incr)
    else:
        threshes = sorted(set([min(cls)-1.0] + [(a+b)/2.0 for a,b in zip(scls, scls[1:])] + [max(cls)+1.0]))
        #print >>sys.stderr, 'Got %d threshes, which we will subsample by %d' % (len(threshes), incr)
        if incr and incr > 0 and type(incr) == type(10):
            threshes = threshes[::incr]
    #TODO do the fast trick by doing a linear scan and updating counts
    if 0:
        rates = []
        pos, fpos, neg, fneg, prec = getErrorRates(results, threshes[0])[:5]
        npos = sum(1.0 for c, l in data if l > 0)
        nneg = sum(1.0 for c, l in data if l <= 0)
        pos *= npos
        fpos *= nneg
        neg *= nneg
        fneg *= npos
        print pos, fpos, neg, fneg, npos, nneg
        i = -1
        sres = sorted(results)
        for t in threshes:
            pass #TODO HERE!
    else:
        def retfunc(t):
            pos, fpos, neg, fneg, prec = getErrorRates(results, t)[:5]
            if roctype == 'roc': return (pos, fpos)
            elif roctype == 'precision': return (pos, prec)
        rates = [retfunc(t) for t in threshes]
    return rates

def computeEER(rates, withthresh=0):
    """Computes equal error rates from the list of (true pos, false pos) values.
    If withthresh is true (not the default), then returns (eer, index at which eer occurs)."""
    det, fpos = zip(*rates)
    fpos = map(float, fpos)
    npos = [1.0-f for f in fpos]
    difs = [(abs(d-n), i) for i, (d, n) in enumerate(zip(det, npos))]
    mval, mindex = min(difs)
    if withthresh:
        return det[mindex], mindex
    else:
        return det[mindex]

def computeAUC(rates):
    """Computes the Area Under Curve (AUC) measure from (recall, precision) values."""
    import numpy as np
    recall, precision = zip(*sorted(rates))
    return np.trapz(precision, x=recall)

def getAllErrorRates(data):
    """Sweeps a threshold through some data (as (value, label) pairs) to get error rates at different values.
    Returns (thresh, (pos, fpos, neg, fneg)) tuples."""
    # generate a list of threshold values to try, between each pair of values, and after
    vals, labels = zip(*data)
    vals = sorted(set(vals))
    threshes = [(a+b)/2.0 for a, b in zip(vals, vals[1:])]
    threshes = threshes + [vals[-1]+1]
    npos = sum(1.0 for v, l in data if l > 0)
    nneg = sum(1.0 for v, l in data if l <= 0)
    # the threshold is initially below all the points, so everything is positive
    cur = 0
    pos, fpos, neg, fneg = npos, nneg, 0, 0
    ret = [(vals[0]-1.0, (pos, fpos, neg, fneg))]
    for t in threshes:
        if cur >= len(data): break
        while data[cur][0] < t:
            v, l = data[cur]
            if l > 0: # actually positive
                if v <= t: # score below thresh
                    fneg += 1.0
                    pos -= 1.0
            else: # actually negative
                if v <= t: # score below thresh
                    neg += 1.0
                    fpos -= 1.0
            cur += 1
            if cur >= len(data): break
        ret.append((t, (pos, fpos, neg, fneg)))
    # normalize values
    ret = [(t, (pos/npos, fpos/nneg, neg/nneg, fneg/npos)) for (t, (pos, fpos, neg, fneg)) in ret]
    if 0: # for checking only
        for t, args in ret:
            print '%s: %s' % (t, args)
            print '%s: %s' % (t, getErrorRates(data, t))
    return ret


# MAIN SVM FUNCTIONS
def bulkclassify(model, seqOfFvals, DUMP_FVALS=0, applyscales=1):
    """Classify multiple items in bulk. Returns (label, value) pairs for each input."""
    try:
        import liblinear.liblinearutil as ll
    except Exception:
        ll = None
    try:
        from sklearn import linear_model as linear_model
    except Exception:
        linear_model = None
    #print 'for %d seqs, applyscales %s, scales %s' % (len(seqOfFvals), applyscales, model.scales)
    if applyscales:
        # No exception handling here because we want to get an error if no scales!
        if USE_SCALER:
            seqOfFvals = model.scales.transform(seqOfFvals)
            #print 'new fvecs (%d), [0] was %s' % (len(seqOfFvals), seqOfFvals[0][:20],)
        else:
            seqOfFvals = [applyScales(fvals, model.scales) for fvals in seqOfFvals]
    if DUMP_FVALS:
        f = open('benchmark_scaled_feats_cls_%d.txt' % (int(time.time())), 'w')
        for fvals in seqOfFvals:
            f.write('%s\n' % (' '.join('%d:%s' % (i+1, f) for i, f in enumerate(fvals))))
        f.close()
    #import numpy
    #seqOfFvals = [numpy.array(f, dtype=numpy.float32) for f in seqOfFvals]
    if ll and isinstance(model, ll.model):
        # liblinear
        seqOfFvals = map(list, seqOfFvals)
        labels, acc, vals = ll.predict([], seqOfFvals, model)
        return [(l, abs(v[0])) for l, v in zip(labels, vals)]
    elif linear_model and isinstance(model, linear_model.SGDClassifier):
        # SGD classifier from scikit-learn
        scores = model.decision_function(seqOfFvals)
        labels = model.predict(seqOfFvals)
        return zip(labels, map(abs, scores))
        #return [(-1 if v <= 0 else 1, abs(v)) for v in results]
    else:
        # libsvm
        return model.predict_many(seqOfFvals)

def bulkclassifyscores(model, seqOfFvals, DUMP_FVALS=0, applyscales=1):
    """Classify multiple items in bulk. Returns score=label*value pairs for each input.
    Assumes positive labels are > 0 and negative labels are <= 0
    Just a wrapper on bulkclassify."""
    ret = []
    for label, value in bulkclassify(model, seqOfFvals, DUMP_FVALS=DUMP_FVALS, applyscales=applyscales):
        s = (-1.0 if label <= 0 else 1.0) * value
        ret.append(s)
    return ret

def classify(model, fvals, applyscales=1):
    """Runs a single classification.
    Only use this for doing a couple of classifications at a time. For many, use bulkclassify().
    In fact, the current implementation for this uses bulkclassify"""
    if not fvals: return (None, None)
    return bulkclassify(model, [fvals], applyscales=applyscales)[0]

def bulkpredictregression(model, seqOfFvals):
    """Use a regressor to predict values in bulk.  Returns a list of the predicted value for
    each element of seqOfFvals."""
    #TODO not tested recently
    try:
        seqOfFvals = [applyScales(fvals, model.scales) for fvals in seqOfFvals]
    except AttributeError: pass
    predictions = model.predict_many_regression(seqOfFvals, True)
    # Not sure why, but this seems to make things worse.  Perhaps the "adjustment" is just a
    # form of overfitting.
    # if model.adjustment:
    #   print 'adjusting predictions predictions...'
    #   adjOffset, adjScale = model.adjustment['offset'], model.adjustment['scale']
    #   predictions = [adjScale * (p + adjOffset) for p in predictions]
    return predictions

def do_cross_validation(prob_x, prob_y, param, nr_fold):
    "Do cross validation for a given SVM problem."
    import math
    from svm import svm_problem, cross_validation, EPSILON_SVR, NU_SVR
    prob_l = len(prob_y)
    total_correct = 0
    total_error = sumv = sumy = sumvv = sumyy = sumvy = 0.
    prob = svm_problem(prob_y, prob_x)
    target = cross_validation(prob, param, nr_fold)
    for i in range(prob_l):
        if param.svm_type == EPSILON_SVR or param.svm_type == NU_SVR:
            v = target[i]
            y = prob_y[i]
            sumv += v
            sumy += y
            sumvv += v * v
            sumyy += y * y
            sumvy += v * y
            total_error += (v-y) * (v-y)
        else:
            v = target[i]
            if v == prob_y[i]:
                total_correct += 1
    if param.svm_type == EPSILON_SVR or param.svm_type == NU_SVR:
        rmse = math.sqrt(total_error / prob_l)
        score = 1 / rmse # something that increases when things are better
        scc = (((prob_l * sumvy - sumv * sumy) * (prob_l * sumvy - sumv * sumy))
               / ((prob_l * sumvv - sumv * sumv) * (prob_l * sumyy - sumy * sumy)))
        print 'Cross validation result:\n  params: %s\n  rmse: %f\n  score:  %f' % (param, rmse, score)
        #print "Cross Validation Mean squared error = %g" % rmse
        #print "Cross Validation Squared correlation coefficient = %g" % scc
        return score
    else:
        try:
            acc = (100.0 * total_correct / prob_l)
        except ZeroDivisionError: acc = 0
        #print "  Cross Validation Accuracy = %g%%" % acc
        return acc

def handleSVMweightsAndIO(labels, outf=sys.stderr, fvalsoutfile=None):
    """Creates """
    from cStringIO import StringIO
    if not outf:
        outf = StringIO()
    if fvalsoutfile:
        f = fvalsoutfile
        for l, fvals in zip(labels, features):
            f.write('%+d %s\n' % (l, ' '.join('%d:%s' % (i+1, f) for i, f in enumerate(fvals))))
    #print 'First few features: %s' % (features[0][:10],)
    # compute the negweight
    npos = sum(1.0 for l in labels if l == 1)
    nneg = sum(1.0 for l in labels if l <= 0)
    negweight = npos/nneg
    print >>outf, '  With %d positive examples and %d negatives, our negweight is %f' % (npos, nneg, negweight)
    return outf, negweight

def trainSingleSVMPrescaled(labels, features, svmstr, scales, niters=5, propweight=1, outf=sys.stderr, fvalsoutfile=None, svmprob=None):
    """Trains a single svm, with scaling already applied"""
    import svm as cursvm
    times = [time.time()]
    outf, negweight = handleSVMweightsAndIO(labels, outf=outf, fvalsoutfile=fvalsoutfile)
    # now create the svm params and problem
    if propweight:
        svmstr += ', nr_weight=2, weight_label=[1, -1], weight=[%f,%f]' % (1.0, negweight)
    C_SVC, LINEAR, RBF, svm_problem, svm_model, svm_parameter = cursvm.C_SVC, cursvm.LINEAR, cursvm.RBF, cursvm.svm_problem, cursvm.svm_model, cursvm.svm_parameter
    print >>outf, 'svmstr: %s' % svmstr
    params = eval('svm_parameter(%s)' % svmstr) #TODO why does this need an eval?
    try:
        prob = svmprob[0]
    except (TypeError, IndexError):
        prob = svm_problem(labels, features)
        if svmprob is not None:
            svmprob.append(prob)
    times.append(time.time())
    #print 'Training model with params %s' % (params,)
    #print 'First few featvals are: %s' % (features[:5],)
    #print 'labels are: %s' % (labels[:],)
    t1 = time.time()
    model = svm_model(prob, params)
    times.append(time.time())
    t2 = time.time()
    print >>outf, 'Took %0.3fs to train model' % (t2-t1)
    model.scales = scales
    if niters > 0:
        score = do_cross_validation(features, labels, params, niters)
        #print >>outf, 'Got score %s' % (score,)
    else:
        score = 0.0
    times.append(time.time())
    print >>outf, 'In trainsinglesvm, got score %f and times %s' % (score, getTimeDiffs(times))
    return (model, score)

def trainLinearSVMPrescaled(labels, features, svmstr, scales, niters=5, propweight=1, outf=sys.stderr, fvalsoutfile=None, svmprob=None):
    """Trains a single linear svm, with scaling already applied"""
    import liblinear.liblinearutil as ll
    times = [time.time()]
    outf, negweight = handleSVMweightsAndIO(labels, outf=outf, fvalsoutfile=fvalsoutfile)
    # now create the svm params and problem
    if propweight:
        svmstr += ' -w1 %f' % (1.0/negweight)
    print >>outf, 'svmstr: %s' % svmstr
    features = map(list, features)
    try:
        prob = svmprob[0]
    except (TypeError, IndexError):
        prob = ll.problem(labels, features)
        if svmprob is not None:
            svmprob.append(prob)
    times.append(time.time())
    #print 'First few featvals are: %s' % (features[:5],)
    #print 'labels are: %s' % (labels[:],)
    t1 = time.time()
    model = ll.train(prob, svmstr)
    times.append(time.time())
    t2 = time.time()
    print >>outf, 'Took %0.3fs to train model' % (t2-t1)
    model.scales = scales
    if niters > 0:
        score = ll.train(prob, svmstr+' -v %d' % (niters))
        #print >>outf, 'Got score %s' % (score,)
    else:
        score = 0.0
    times.append(time.time())
    print >>outf, 'In trainsinglelinear, got score %f and times %s' % (score, getTimeDiffs(times))
    # also add a save function to simplify things
    model.save = lambda fname, m=model: ll.save_model(fname, m)
    return (model, score)

def trainSGDPrescaled(labels, features, svmstr, scales, niters=5, propweight=1, outf=sys.stderr, fvalsoutfile=None, svmprob=None):
    """Trains a single SGD-based svm, with scaling already applied"""
    import numpy as np
    from sklearn import linear_model
    times = [time.time()]
    outf = sys.stderr
    outf, negweight = handleSVMweightsAndIO(labels, outf=outf, fvalsoutfile=fvalsoutfile)
    kw = dict(shuffle=True, n_iter=20)
    # now create the svm params and problem
    if propweight:
        kw['class_weight'] = {1: 1.0, -1: negweight}
        pass
    print >>outf, 'svmstr: %s' % svmstr
    try:
        prob = svmprob[0]
    except (TypeError, IndexError):
        prob = (features, labels)
        if svmprob is not None:
            svmprob.append(prob)
    times.append(time.time())
    features = np.array(features)
    log('First few featvals are %s: %s' % (features.shape, features[:5],))
    #log('labels are: %s' % (labels[:],))
    t1 = time.time()
    model = linear_model.SGDClassifier(**kw)
    model.fit(features, labels)
    times.append(time.time())
    t2 = time.time()
    print >>outf, 'Took %0.3fs to train model' % (t2-t1)
    model.scales = scales
    if niters > 0:
        raise NotImplementedError()
        #score = ll.train(prob, svmstr+' -v %d' % (niters))
        #print >>outf, 'Got score %s' % (score,)
    else:
        score = 0.0
    times.append(time.time())
    print >>outf, 'In trainsinglesgd, got score %f and times %s' % (score, getTimeDiffs(times))
    # also add a save function to simplify things
    #TODO model.save = lambda fname, m=model: ll.save_model(fname, m)
    return (model, score)

def trainSinglePrescaled(labels, features, svmstr, scales, niters=5, propweight=1, outf=sys.stderr, fvalsoutfile=None, svmprob=None):
    """Trains a single svm, with scaling already applied.
    Parses svmstr to figure out whether to call the function for libsvm or liblinear.
    The way svmprob is handled is as follows:
        If it's None (default), then doesn't affect anything.
        If it's an empty array, then the created 'problem' structure is append()ed to it.
        If it's an array with at least 1 element, then the first element is
           assumed to be a valid svm problem. Thus, it is not re-computed. This
           can save an immense amount of memory, and some time as well.
    """
    kw = dict(niters=niters, propweight=propweight, outf=outf, fvalsoutfile=fvalsoutfile, svmprob=svmprob)
    svmtype = parseSVMType(svmstr)
    if svmtype == 'libsvm':
        func = trainSingleSVMPrescaled
    elif svmtype == 'liblinear':
        func = trainLinearSVMPrescaled
    return func(labels, features, svmstr, scales=scales, **kw)

def trainSingleSVM(labels, features, svmstr, niters=5, means=None, stddevs=None, propweight=1, outf=sys.stderr, fvalsoutfile=None, svmprob=None):
    """Trains a new classifier and returns the model, as well as the cross-validation score.
    Also returns a set of scales as model.scales, which are (mean, stddev) pairs used for scaling.
    This is a pre-processing wrapper on trainSinglePrescaled, which does the actual training.
    This function:
        * computes scalings if not given
        * applies scalings
    niters is the number of iterations of cross-validation to run.
    If 0, then all training samples are used to train the classifier, but then a cross-validation score of 0.0 is returned.
    If propweight is true, then a weighting is applied according to the proportion of positive and negative examples.
    If outf is given, then debug messages are printed there.
    If fvalsoutfile is given, then the scaled feature values are written there.
    """
    import svm as cursvm
    from cStringIO import StringIO
    if not outf:
        outf = StringIO()
    # apply scalings
    times = [time.time()]
    if not means or not stddevs:
        if not means:
            means = [getMean(dim) for dim in izip(*features)]
        if not stddevs:
            stddevs = [getStdDev(dim, m) for dim, m in izip(izip(*features), means)]
    times.append(time.time())
    scales = zip(means, stddevs)
    assert len(scales) == len(means) == len(stddevs)
    #print 'First few features: %s' % (features[0][:10],)
    #print 'Scales: %s' % (scales[:10],)
    features = [applyScales(fvals, scales) for fvals in features]
    times.append(time.time())
    print >>outf, 'In trainsingle first part, got times %s' % (getTimeDiffs(times))
    kw = dict(niters=niters, propweight=propweight, outf=outf, fvalsoutfile=fvalsoutfile, svmprob=svmprob)
    return trainSinglePrescaled(labels, features, svmstr, scales=scales, **kw)

def runGridSearch(labels, features, svmstrs, niters=5, callback=None, outf=sys.stderr, trainfunc=trainSingleSVM):
    """Run a grid search over different svm parameters.
    The training function used is specified in trainfunc, and is trainSingleSVM by default.
    The function should return (model, score) pairs.
    If there's a callback, it's called with (svmstr, score) pairs each time one is calculated.
    Debugging out goes to outf.
    Returns (bestmodel, bestscore, bestsvmstr)."""
    from random import shuffle
    start = time.time()
    best = (0, None, '')
    score = 0
    print >>outf, '  Starting grid search...\r',
    sys.stdout.flush()
    for i, svmstr in enumerate(svmstrs):
        print >>outf, ('    [%s] Grid search on %d examples, svmstr %d of %d (best: %0.2f, last: %0.2f): %s...   ' %
                       (time.strftime('%Y-%m-%d %H:%M:%S'), len(features), i+1, len(svmstrs), best[0], score, svmstr))
        outf.flush()
        model, score = trainfunc(labels, features, svmstr, niters=niters, outf=outf)
        if callback:
            callback(svmstr, score)
        if score > best[0]:
            best = (score, model, svmstr)
    bestscore, bestmodel, bestsvmstr = best
    elapsed = time.time() - start
    print >>outf, ('  Ran crossvalidation in %s secs on %s sets of parameters, %d examples, for best score of %s' %
                   (elapsed, len(svmstrs), len(features), bestscore))
    return (bestmodel, bestscore, bestsvmstr)

def splitTrainEval(pos, neg, ncv):
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

def evalSVM(SVMCls, model, features, labels, applyscales=1):
    """Evaluates the given model using the given labels and features.
    Returns a weighted score (i.e., suitable for unbalanced datasets)"""
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
    log('In eval with %d pos, %d neg (ratio %0.3f), got scores %0.3f, %0.3f = %0.3f, times %s' % (npos, nneg, nneg/float(npos), scores[0], scores[1], score, getTimeDiffs(times)))
    return score


class SVM(object):
    """Light-weight wrapper on Support Vector Machines.
    Mainly used to keep track of various parameters and
    have different subclasses with different implementations.
    In particular, this class deals with:
        - proportional weighting of positive/negative classes
        - feature scaling and saving/loading scales to/from models
        - optional output logging
        - optional saving of transformed features
        - w-score normalization
        - saving training parameters to models
    """
    def __init__(self, propweight=1, outf=sys.stderr, fvalsoutfile=None):
        """Initializes with various parameters:
            propweight: if 1 [default], then weights classes by number of examples
                  outf: logging output gets sent to this file-like object [sys.stderr]
          fvalsoutfile: saves computed features to the given filename [None]
        You can also change any of these parameters at any time.
        """
        self.propweight = propweight
        self.outf = outf
        self.fvalsoutfile = fvalsoutfile

    @classmethod
    def load(c, fname, **kw):
        """Loads a saved SVM model from file.
        Implement in subclasses."""
        raise NotImplementedError

    def save(self, model, fname, **kw):
        """Saves this SVM model to file.
        Implement in subclasses."""
        raise NotImplementedError

    @property
    def outf(self):
        return self._outf

    @outf.setter
    def outf(self, value):
        """Sets the output file. If None, then creates a StringIO()."""
        from cStringIO import StringIO
        if not value:
            value = StringIO()
        self._outf = value

    def computeScales(self, features):
        """Computes scales on the given features.
        Returns (scales, transformed features)
        """
        if USE_SCALER:
            from sklearn.preprocessing import StandardScaler
            scales = StandardScaler()
            try:
                features = scales.fit_transform(features)
            except Exception:
                print 'Had features: %s, %s' % (type(features), len(features))
                print set(map(lambda x: x.shape, features))
                import pdb
                pdb.set_trace()
                raise
        else:
            means = [getMean(dim) for dim in izip(*features)]
            stddevs = [getStdDev(dim, m) for dim, m in izip(izip(*features), means)]
            scales = zip(means, stddevs)
            assert len(scales) == len(means) == len(stddevs)
            features = self.scaleFeatures(features, scales)
        return (scales, features)

    def scaleFeatures(self, features, scales):
        """Scales features using the given scales.
        Scales are assumed to be zip(means, stddevs).
        Returns scaled features."""
        #print 'First few features: %s' % (features[0][:10],)
        #print 'Scales: %s' % (scales[:10],)
        if USE_SCALER:
            features = scales.transform(features)
        else:
            features = [applyScales(fvals, scales) for fvals in features]
        return features

    def writeFeaturesToFile(self, features, labels=None, sparse=0):
        """Writes features and optionally labels to self.fvalsoutfile, if not None.
        If labels are given, the format is:
            label\tfeatures
        where label is '+1' or '-1', and features depends on 'sparse':
            sparse=0 [default]: space-separated values
            sparse=1: space-separated 'index:value' (like libsvm)
        """
        if self.fvalsoutfile:
            f = self.fvalsoutfile
            if sparse:
                fvecfmt = lambda fvec: ' '.join('%d:%s' % (i+1, f) for i, f in enumerate(fvec))
            else:
                fvecfmt = lambda fvec: ' '.join('%s' % (f,) for f in fvec)
            if labels:
                for l, fvec in zip(labels, features):
                    f.write('%+d\t%s\n' % (l, fvecfmt(fvec)))
            else:
                for fvec in features:
                    f.write('%s\n' % fvecfmt(fvec))

    def computeNegWeight(self, labels):
        """Computes the positive-to-negative weight ratio."""
        npos = sum(1.0 for l in labels if l == 1)
        nneg = sum(1.0 for l in labels if l <= 0)
        negweight = npos/nneg
        print >>self.outf, '  With %d positive examples and %d negatives, our negweight is %f' % (npos, nneg, negweight)
        return negweight

    def train(self, features, labels, scales=None, ncv=5, **kw):
        """Trains the classifier.
        Inputs:
            features: a 2d-matrix-like object, with N rows and F features per row [the 'X' in math terms]
              labels: a list of len N, generally with labels +1 or -1 [the 'Y' in math terms]
              scales: if None, then computes them. If [], then assumes prescaled. Else, should be given.
                 ncv: the number of cross-validation iterations to run.
                    if ncv > 0: divides data into ncv folds and iterates through each one,
                                training using all other folds and testing on that one.
                                scores are averaged. In the end, a final classifier is trained
                                using all of the data.
        This should return (model, score).
        """
        times = [time.time()]
        # compute negative weight
        if self.propweight:
            kw['negweight'] = self.computeNegWeight(labels)
        else:
            kw['negweight'] = 1.0
        times.append(time.time())
        # handle scaling
        if scales is None: # compute them
            scales, features = self.computeScales(features)
        elif not scales: # precomputed
            scales = []
        else: # given, so apply them
            features = self.scaleFeatures(features, scales)
        times.append(time.time())
        # deal with cross-validation
        if ncv > 0:
            score = self.crossvalidate(features, labels, ncv=ncv, **kw)
        else:
            score = 0.0
        times.append(time.time())
        # train the final classifier
        model = self._train(features, labels, **kw)
        times.append(time.time())
        # save the scales in the model itself
        model.scales = scales
        times.append(time.time())
        # save other key params
        model.trainkw = dict(traintime=times[-1], trainelapsed=times[-1]-times[0], score=score, nfeats=len(features), ndims=len(features[0]), negweight=kw['negweight'], ncv=ncv)
        print >>self.outf, 'In SVM.train, got score %f and times %s' % (score, getTimeDiffs(times))
        return (model, score)

    def classify(self, model, features, wfit=1, **kw):
        """Classifies the given set of features using our model.
        Returns a list of signed distances.
        If wfit is true (the default) and the wfit parameters exist in the model,
        applies the w-score normalizations to the outputs before returning them.
        For more details, please see the following paper:

            Walter Scheirer, Neeraj Kumar, Peter N. Belhumeur, Terrance E. Boult,
            "Multi-Attribute Spaces: Calibration for Attribute Fusion and Similarity Search,"
            Proceedings of the 25th IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
            June 2012.

        """
        from wfit import easynormsvm
        #print 'in classify, with features %s' % (str(features)[:200])
        ret = self._classify(model, features, **kw)
        if wfit and hasattr(model, 'trainkw') and 'wfitparams' in model.trainkw:
            ret = easynormsvm(ret, model.trainkw['wfitparams'])
        return ret

    def crossvalidate(self, features, labels, ncv, **kw):
        """Runs crossvalidation with the given features and labels and number of cv-folds.
        Returns the average score"""
        from random import choice
        if ncv <= 0: return 0.0
        if not labels: return 0
        # generate foldids
        foldids = [choice(range(ncv)) for l in labels]
        # iterate over each fold
        scores = 0.0
        for fold in range(ncv):
            trainfeats, trainlabels = zip(*[(f, l) for f, l, fid in zip(features, labels, foldids) if fid != fold])
            evalfeats, evallabels = zip(*[(f, l) for f, l, fid in zip(features, labels, foldids) if fid == fold])
            model = self._train(trainfeats, trainlabels, **kw)
            score = evalSVM(self, model, evalfeats, evallabels, applyscales=0)
            scores += score
        scores /= ncv
        return scores


    def wfit(self, clsoutputs, model=None):
        """Fits a w-score model to the given outputs and optionally attaches them to the given model.
        This is a normalization procedure based on Extreme-Value Theory.
        In contrast to the more commonly used Platt Scaling for SVMs, this does not need class labels.
        Just give it a list of outputs from classify() and it will return suitable normalization parameters.
        If you give it a model, it will put them inside the model's 'trainkw', as 'wfitparams'.

        For more details, please see the following paper:

            Walter Scheirer, Neeraj Kumar, Peter N. Belhumeur, Terrance E. Boult,
            "Multi-Attribute Spaces: Calibration for Attribute Fusion and Similarity Search,"
            Proceedings of the 25th IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
            June 2012.

        Returns [positive params, negative params].
        """
        from wfit import easyfitsvm
        params = easyfitsvm(clsoutputs)
        if model:
            model.trainkw['wfitparams'] = params
        return params

    def _train(self, features, labels, **kw):
        """Actual training call. Implement in subclass."""
        raise NotImplementedError

    def _classify(self, model, features, **kw):
        """Actual classification call. Implement in subclass."""
        raise NotImplementedError

class LibSVM(SVM):
    """SVMs implemented using LibSVM.
    This is a Libsvm wrapper wrapper =)"""
    def _train(self, features, labels, **kw):
        """Actual training call"""
        import svm
        times = [time.time()]
        svmstr = kw.get('svmstr', kw.get('params', 'svm_type=C_SVC, kernel_type=RBF'))
        # now create the svm params and problem
        if self.propweight:
            svmstr += ', nr_weight=2, weight_label=[1, -1], weight=[%f,%f]' % (1.0, kw['negweight'])
        C_SVC, LINEAR, RBF, svm_problem, svm_model, svm_parameter = svm.C_SVC, svm.LINEAR, svm.RBF, svm.svm_problem, svm.svm_model, svm.svm_parameter
        params = eval('svm_parameter(%s)' % svmstr) #TODO why does this need an eval?
        times.append(time.time())
        # use cached svmprob if given
        if 'svmprob' in kw:
            prob = kw['svmprob']
            # if it's None, then we need to compute it
            if not prob:
                kw['prob'] = prob = svm_problem(labels, features)
        else:
            prob = svm_problem(labels, features)
        times.append(time.time())
        model = svm_model(prob, params)
        times.append(time.time())
        print >>self.outf, 'Computed LibSVM model with svmstr %s, times: %s' % (svmstr, getTimeDiffs(times))
        return model

    def _classify(self, model, features, **kw):
        """Classifies the given list of features and returns signed distance values."""
        return bulkclassifyscores(model, features, **kw)

    @classmethod
    def load(c, fname, **kw):
        """Loads a model file from disk"""
        return readSVMModelAndParams(fname, **kw)

    def save(self, model, fname, **kw):
        """Saves the given model to the given filename.
        Parameters (scales) are stored in separate file.
        kw args are passed directly to saveSVMModelAndParams()"""
        saveSVMModelAndParams(model, fname, **kw)

class LinearSVM(LibSVM):
    """Linear SVMs implemented using LibLinear.
    This is a Liblinear wrapper wrapper =)"""
    def _train(self, features, labels, **kw):
        """Actual training call"""
        import liblinear.liblinearutil as ll
        times = [time.time()]
        svmstr = kw.get('svmstr', kw.get('params', '-q -s 2'))
        if self.propweight:
            svmstr += ' -w1 %f' % (1.0/kw['negweight'])
        features = map(list, features)
        if 'svmprob' in kw:
            prob = kw['svmprob']
            # if it's None, then we need to compute it
            if not prob:
                kw['prob'] = prob = ll.problem(labels, features)
        else:
            prob = ll.problem(labels, features)
        times.append(time.time())
        model = ll.train(prob, svmstr)
        times.append(time.time())
        print >>self.outf, 'Computed LibLinear model with svmstr %s, times: %s' % (svmstr, getTimeDiffs(times))
        return model

class SGDSVM(SVM):
    """Stochastic Gradient Descent SVMs, implemented using scikit-learn."""
    def _train(self, features, labels, **kw):
        """Actual training call"""
        import numpy as np
        from sklearn import linear_model
        times = [time.time()]
        if self.propweight:
            kw['class_weight'] = {1: 1.0, -1: kw['negweight']}
        features = np.array(features)
        times.append(time.time())
        del kw['negweight']
        if 'shuffle' not in kw:
            kw['shuffle'] = True
        fitkw = {}
        if 'sample_weight' in kw:
            fitkw['sample_weight'] = kw['sample_weight']
            del kw['sample_weight']
        model = linear_model.SGDClassifier(**kw)
        times.append(time.time())
        model.fit(features, labels, **fitkw)
        times.append(time.time())
        print >>self.outf, 'Computed SGD SVM model with kw %s, times: %s' % (kw, getTimeDiffs(times))
        return model

    def _classify(self, model, features, applyscales=1, **kw):
        """Classifies the given list of features and returns signed distance values."""
        from sklearn import linear_model
        #print 'in _classify, with features %s' % (str(features)[:200])
        if applyscales:
            features = model.scales.transform(features)
        #print 'in _classify after scaling, with features %s' % (str(features)[:200])
        s = model.decision_function(features)
        return s


    @classmethod
    def load(c, fname, **kw):
        """Loads a model file from disk"""
        from cPickle import load
        return load(open(fname), **kw)

    def save(self, model, fname, **kw):
        """Saves the given model to the given filename.
        Simple pickles it using cPickle. kw are passed to cPickle.dump()"""
        from cPickle import dump
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass
        if 'protocol' not in kw:
            kw['protocol'] = -1
        dump(model, open(fname, 'wb'), **kw)

def pickSVMClass(params):
    """Picks the appropriate SVM class based on a given set of parameters or model instance.
    If it's a mapping, then returns SGDSVM.
    If it's a string, then parses the string into either LibSVM or LinearSVM.
    If it's an instance, then looks at the class type."""
    classmap = dict(libsvm=LibSVM, liblinear=LinearSVM, sgd=SGDSVM)
    if isinstance(params, basestring) or isinstance(params, dict):
        svmtype = parseSVMType(params)
    else:
        clsname = str(params.__class__).lower()
        if 'stochastic_gradient' in clsname:
            svmtype = 'sgd'
        elif 'svm_model' in clsname:
            svmtype = 'libsvm'
        else:
            svmtype = 'liblinear'
    return classmap[svmtype]

def parsesvmstr(svmstr):
    """Parses an svm string into the appropriate class and parameters.
    Returns (SVMCls, params), where:
        - SVMCls is a a subclass of trainutils.SVM
        - params is dict mapping 'svmstr' to the given svmstr
        - params is a dict of kw for SGD otherwise.
    For SGD, the svmstr should be in the format: 'sgd,k=v,k=v,...'
    SGD default params are:
        shuffle=1
    """
    from utils import specialize
    if svmstr.startswith('sgd'):
        cls = SGDSVM
        params = [s.strip().split('=', 1) for s in svmstr.split(',') if '=' in s]
        params = dict((k.strip(), specialize(v.strip())) for k, v in params)
    else:
        cls = pickSVMClass(svmstr)
        params = dict(svmstr=svmstr)
    log('From svmstr %s, parsed cls %s, params %s' % (svmstr, cls, params))
    return (cls, params)

def trainmany(dataiter, outdir, skipexisting=1, svmstr='sgd', ncv=-15):
    """Trains many one-vs-all classifiers from a single set of features.
    You provide the data in the form of an iterator 'dataiter'
    that returns list of (class name, feature vector(s)) pairs.
    All the feature vector(s) are then vstack()ed into a single matrix.
    This matrix is then prescaled so that we're not rescaling everytime.
    For each unique class name, we get the current set of labels
    (curclass = 1, otherclasses = -1) and current set of features.
    These are fed to SVMTrainer.train() with the provided svmstr and ncv.
    We save the scales to the model as well.
    The classifier is saved in os.path.join(outdir, clsname+'.model')
    If 'skipexisting' is true (default), then doesn't retrain any model that already exists.
    The classes are done in shuffled order, so when combined with skipexisting,
    makes this safe to run in multiple procs with identical arguments.
    Returns the list of trained classifier fnames.
    """
    from svmtrain import SVMTrainer
    import numpy as np
    from random import shuffle
    feats = []
    names = []
    labels = []
    for inum, (n, fvecs) in enumerate(dataiter):
        t1 = time.time()
        if n not in names:
            names.append(n)
        labels.extend([n]*len(fvecs))
        feats.append(fvecs)
        elapsed = time.time() - t1
        print '%d: Read %d vecs for %s to get %d feats, %d names, %d labels in %0.2fs' % (inum+1, len(fvecs), n, len(feats), len(names), len(labels), elapsed)
        #if len(names) > 2: break
    feats = np.vstack(feats)
    scales, feats = SGDSVM().computeScales(feats)
    svmt = SVMTrainer(prescaled=1)
    shuffle(names)
    ret = []
    for inum, n in enumerate(names):
        t1 = time.time()
        outfname = os.path.join(outdir, n+'.model')
        if skipexisting and os.path.exists(outfname):
            print '%s existed, so skipping' % (outfname)
            continue
        curlabels = [1 if n==l else -1 for l in labels]
        #TODO look into setting sample weights (e.g., to 0) in SGDSVM so that we can keep the feature matrix fixed across the whole process
        pos = np.vstack((feats[i] for i, l in enumerate(curlabels) if l == 1))
        neg = np.vstack((feats[i] for i, l in enumerate(curlabels) if l != 1))
        model, score = svmt.train(pos, neg, svmstr=svmstr, ncv=ncv)
        model.scales = scales
        svmt.save(model, outfname)
        ret.append(outfname)
        elapsed = time.time() - t1
        print '%d: %s, for pos %s, neg %s, got score %s and saved to %s in %0.2fs' % (inum+1, n, pos.shape, neg.shape, score, outfname, elapsed)
    return ret

def addscales(dataiter, modelfnames, suffix='.new'):
    """One-off function to add scales to pretrained models"""
    from svmtrain import SVMTrainer
    import numpy as np
    feats = []
    t1 = time.time()
    for inum, (n, fvecs) in enumerate(dataiter):
        feats.append(fvecs)
        elapsed = time.time() - t1
        print '%d: Read %d vecs for %s to get %d feats in %0.2fs' % (inum+1, len(fvecs), n, len(feats), elapsed)
        #if len(feats) > 15: break
    feats = np.vstack(feats)
    scales, feats = SGDSVM().computeScales(feats)
    svmt = SVMTrainer()
    for modelfname in modelfnames:
        model = SGDSVM.load(modelfname)
        model.scales = scales
        outfname = modelfname+suffix
        svmt.save(model, outfname)
        print 'Saved %s to %s' % (modelfname, outfname)


INDEX_MODES = ['tab-columns', 'datamatrix-fnames', 'datamatrix-rownums', 'redis']

class FeatureParser(object):
    """A base class for parsing features from inputs.
    This allows for flexible ways of reading feature vectors.
    There are two main types of input:
        1. Raw feature vectors from inputs
        2. Indexed feature vectors, which map input indices to some sort of feature dictionary.
    For very simple cases, option 1 is probably fine, but for data-intensive tasks,
    you probably want some form of indexing.

    The indexing can take many forms, and is quite flexible in a number of ways.
    """
    def __init__(self, featfiles=None, indexmode='', columns='', usemmap=0,
            urlbase=None, cachedir='.', rediscfgfname=None, errval=None, debug=0):
        """Initialize this parser with various configuration options.
        The major options are 'featfiles' and 'indexmode'.

        If featfiles is None (default), then assumes feature vectors are given
        directly as inputs (as space-separated strings, one feature vector per
        line (ended by '\n'), or as sequences of values convertable to
        array('f')).

        Else, featfiles is a list of feature dictionaries. In this case, Each
        input determines an index into the dictionaries, from which the feature
        vectors are copied. The feature vectors are concatenated from each
        feature dictionary, in the order given. The data is read one dictionary
        at a time, and the indices into the dictionaries don't have to be in
        order -- the outputs are automatically sorted into the same order as the
        inputs. Of course, having the inputs in order may speed things up.

        indexmode determines the format of the dictionary and of the inputs:
            tab-columns: the dictionary is in 'id\tf1 f2 f3...\n' format, and
            the inputs are strings corresponding to the 'id' column.

            datamatrix-fnames: the dictionary is a DataMatrix, and the inputs
            correspond to the first dimension of the DataMatrix.

            datamatrix-rownums: the dictionary is a DataMatrix, and the inputs
            correspond to row numbers within the Matrix.

            redis: the dictionary is in redis as array('f').tostring(), and the
            inputs map to individual redis keys.

        Regardless of whether the feature vectors were gotten via direct mode or
        indexed mode, various postprocessing is applied:
            'columns': a specification of which columns of the output to keep.
            This includes comma-separated elements, where each element can be one of:
                0-indexed column number (e.g., 5)
                [,) range of 0-indexed column number (e.g., 0-5)
            By default, this is empty and all columns are kept.

            'errval': if all elements of a row are equal to this, then the row
            is discarded. This is None by default, but if the indexmode is a
            datamatrix, then uses dm.errval instead.

        Finally, if there was any problem with any input row, it is set to None
        in the output. Problems include having less dimensions than the row with
        the highest number of dimensions, or not existing for any other reason.

        Other parameters are used for a subset of the reading methods:
            'urlbase' and 'cachedir': If a featfile isn't found on disk and you set
            a urlbase, downloads the featfile from urlbase+featfile to the given
            cachedir. The cachedir is the current directory by default. This is
            used by all indexed modes, except redis.

            'rediscfgfname': The file containing the redis database parameters,
            in json format. This is only used by the 'redis' indexmode.

        The output is a list of the same length as the number of inputs, and
        each element will either be a valid feature vector of array('f'), all
        the same length, or None on any sort of error.  If the list of inputs is
        None or empty, then the output is [].
        """
        self.featfiles = featfiles[:] if featfiles else []
        self.indexmode = indexmode
        self.columns = self.parsecols(columns)
        self.usemmap = usemmap
        self.urlbase = urlbase
        self.cachedir = cachedir
        self.rediscfgfname = rediscfgfname
        self.errval = errval
        self.debug = debug

    def parsecols(self, s):
        """Parses a columns string into a list of [start, end) range pairs"""
        if not s: return None
        els = [e.strip() for e in s.split(',')]
        ret = []
        def intOrNone(v):
            try:
                return int(v)
            except Exception:
                return None

        for e in els:
            if '-' in e:
                start, end = e.split('-')
                ret.append((intOrNone(start), intOrNone(end)))
            else:
                e = int(e)
                end = None if e == -1 else e+1
                ret.append((e, end))
        return ret

    def columnize(self, row):
        """Extracts the right columns from the given row"""
        from array import array
        if not self.columns or not row: return row
        ret = array('f')
        for start, end in self.columns:
            ret.extend(row[start:end])
        return ret

    def parserow(self, row):
        """Parses a single row"""
        from array import array
        #log(row)
        return array('f', map(float, row.split()))

    def readDirect(self, inputs):
        """Returns features from the inputs directly.
        Called by getfeatures() as appropriate.
        Uses the first input to figure out the format.
        If it's a string, uses self.parserow().
        Else, tries to map it directly to an array('f').
        Always returns array('f').
        Doesn't do columnizing or filtering of results.
        Iterator-friendly.
        """
        from array import array
        fmtfunc = None
        ret = []
        for row in inputs:
            if not fmtfunc:
                # use first input to figure out the format
                if isinstance(row, basestring):
                    fmtfunc = self.parserow
                else:
                    fmtfunc = lambda row: array('f', row)
            ret.append(fmtfunc(row))
        return ret

    def readTabColumns(self, fname, inputs, outdict):
        """Reads tabbed-column inputs from the file.
        For each row, splits into id and fvec using a tab.
        Checks if id is in 'inputs', and if so, parses the fvec using self.parserow().
        Note that this handles header rows transparently, since the header row
        will not have a matching 'id'.
        Extends outdict[id] with the given fvec.
        Returns outdict.
        """
        from array import array
        import mmap
        #log('Reading featfile %s' % (fname))
        if not os.path.exists(fname) and self.urlbase:
            # download the file if we don't have it
            url = self.urlbase+fname
            fname = downloadfile(url, outdir=self.cachedir, outf=sys.stderr, delay=0)
        # at this point, we've downloaded the file, so open it
        f = open(fname, 'r+b')
        # use a different iterator depending on if we want to use a mmap or not
        if self.usemmap:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            iterator = iter(mm.readline, '')
        else:
            iterator = f
        # iterate through the featfile
        for i, l in enumerate(iterator):
            # for now, don't parse the fvec yet
            try:
                id, fvec = l.rstrip('\n').split('\t', 1)
            except Exception, e:
                log('Got exception "%s" in reading tab-columns on line %d: %s' % (e, i, l.rstrip()))
                continue
            # if this id is not part of our input, continue on
            if id not in inputs: continue
            # NOW parse the fvec
            fvec = self.parserow(fvec)
            if id not in outdict:
                outdict[id] = array('f')
            outdict[id].extend(fvec)
        return outdict

    def readDataMatrix(self, dmname, inputs, outdict, indexmode, inset=None):
        """Reads data from the given DataMatrix (DM).
        'indexmode' determines what inputs are:
            datamatrix-fnames: elements in the first dimension of the DM.
            datamatrix-rownums: row numbers in the first dimension of the DM.
        The DM is downloaded using self.urlbase and self.cachedir if needed.
        Results are extend()ed into outdict.
        You can optionally pass in an 'inset', which is just a set() of the inputs.
        Returns outdict.
        """
        from array import array
        from nkpylib.datamatrix import DataMatrix
        try:
            import simplejson as json
        except ImportError:
            import json
        assert indexmode in ['datamatrix-fnames', 'datamatrix-rownums']
        times = [time.time()]
        # download files if they don't exist
        if not os.path.exists(dmname+'.json') or not os.path.exists(dmname+'.mmap'):
            urls = [self.urlbase+dmname+'.json', self.urlbase+dmname+'.mmap']
            for u in urls:
                downloadfile(url, outdir=self.cachedir, outf=sys.stderr, delay=0)
        times.append(time.time())
        # at this point, we've downloaded the files
        dm = DataMatrix(dmname, mode='readdata', loadfields=0)
        self.errval = dm.errval
        times.append(time.time())
        # create a lookup of the appropriate type
        if self.debug > 0:
            log('Inputs were: %s' % (inputs[:5],))
        if indexmode == 'datamatrix-fnames':
            inset = set(inset) if inset else set(inputs)
            lookupdict = dm.getlookup(0, keys=inset, keytype='exact')
            lookupfunc = lambda input: lookupdict[unicode(input, 'utf-8', 'ignore')]
        elif indexmode == 'datamatrix-rownums':
            lookupfunc = lambda rownum: int(rownum)
        times.append(time.time())
        #log('About to go through inputs, and got times %s' % (getTimeDiffs(times),))
        # Iterate through inputs
        M = dm.mmap
        for input in inputs:
            try:
                rownum = lookupfunc(input)
                fvec = M.getrow(rownum)
                outdict[input].extend(fvec)
            except Exception: continue
        times.append(time.time())
        if self.debug > 0:
            log('Created outdict with %d els in readDataMatrix, times: %s' % (len(outdict), getTimeDiffs(times)))
        del dm
        #sys.exit()
        return outdict

    def readRedis(self, name, inputs, outdict, db, ids=None):
        """Reads data from redis.
        The keys for the data are at 'f:<name>:<id>'
        The redis database should already be initialized at 'db'.
        The list of ids can optionally be initialized using
        db.hmget('fnameids', inputs), or if not, then it's done here.
        Extends outdict using the returned fvecs.
        Returns outdict.
        """
        from nkpylib.redisutils import redis, pipefunc
        if not ids:
            ids = db.hmget('fnameids', inputs)
        keys = ['f:%s:%s' % (name, id) for id in ids if id]
        fvecs = [array('f', fvec) for fvec in pipefunc(db, keys, 'get')]
        i = 0
        for input, id in zip(inputs, ids):
            if not id: continue
            fvec = fvecs[i]
            i += 1
            outdict[input].extend(fvec)
        return outdict

    def postprocess(self, ret):
        """Does various postprocessing to feature outputs.
        - Columnizes, based on self.columns.
        - Figures out the max dimensionality of any row, and sets any row with less elements to None.
        - If the entire row matches errval, then sets it to None.
        Returns (ret, ndims, nvalid)
        """
        from collections import defaultdict
        from array import array
        times = [time.time()]
        # columnize outputs
        #TODO see about storing ret as a numpy 2d matrix, and then using faster submatrix operations
        #TODO because this next call takes ~85% of this function!
        #ret = [self.columnize(row) for row in ret]
        #FIXME we get 50% speedup by inlining the columnize here
        if self.columns:
            for i, row in enumerate(ret):
                if not row: continue
                ret[i] = new = array('f')
                for start, end in self.columns:
                    new.extend(row[start:end])
        times.append(time.time())
        # get row dimensionalities
        allndims = defaultdict(int)
        times.append(time.time())
        nvalid = 0
        for i, row in enumerate(ret):
            if row:
                # Remove rows marked 'errval'
                for r in row:
                    # if any el is not the errval, then we know it's a valid row
                    if r != self.errval:
                        break
                else:
                    # we only come here if all els were errval -- it's an error
                    ret[i] = None
                    continue
                allndims[len(row)] += 1
                nvalid += 1
        times.append(time.time())
        ndims = 0
        if allndims:
            ndims = max(allndims)
            if self.debug > 0:
                log('Got all ndims: %s, and max ndims %d' % (allndims, ndims))
            # eliminate short rows
            if ndims != min(allndims):
                for i, row in enumerate(ret):
                    if row and len(row) < ndims:
                        ret[i] = None
                        nvalid -= 1
        times.append(time.time())
        # optionally, write outputs to file
        if 0:
            outf = open('blah', 'w')
            for row in ret:
                print >>outf, row.tolist() if row else row
            outf.close()
            sys.exit()
        if self.debug > 0:
            log('Got postprocess times: %s' % (getTimeDiffs(times)))
        return (ret, ndims, nvalid)

    def getfeatures(self, inputs):
        """Returns a list of feature vectors from the given inputs.
        This does all the necessary work, whether it's parsing from strings,
        reading from feature dictionary files, etc."""
        from array import array
        from collections import defaultdict
        ret = []
        if not inputs: return ret
        times = [time.time()]
        outdict = None
        if self.featfiles:
            # if we have feature dictionaries, then inputs are lookups.
            # we're going to build up a dictionary of results, and then put it all into order.
            outdict = defaultdict(lambda: array('f'))
            inputs = list(inputs) # cache these in case we have a generator input
            # do other init
            if self.rediscfgfname:
                # if we have redis, then create the connection
                db = redis.Redis(**json.load(open(self.rediscfgfname)))
                if self.debug > 0:
                    log('Loaded redis db from %s: %s' % (self.rediscfgfname, db))
                    log('For %d inputs, getting ids' % (len(inputs)))
                ids = db.hmget('fnameids', inputs)
            else:
                # for all other indexed input types, we generally need this 'inset'
                inset = set(inputs)
            times.append(time.time())
            # go through each feature file and extend feature vectors for each id
            for fname in self.featfiles:
                if self.indexmode.startswith('datamatrix-'):
                    self.readDataMatrix(fname, inputs=inputs, inset=inset, outdict=outdict, indexmode=self.indexmode)
                elif self.indexmode == 'redis':
                    self.readRedis(os.path.basename(fname), inputs=inputs, outdict=outdict, db=db, ids=ids)
                elif self.indexmode == 'tab-columns':
                    self.readTabColumns(fname, inputs=inset, outdict=outdict)
                else:
                    raise NotImplementedError('indexmode must be one of %s' % (','.join(INDEX_MODES)))
            # now build the output in order
            ret = [outdict.get(id, None) for id in inputs]
        else:
            # feature vectors are directly given
            ret = self.readDirect(inputs)
        times.append(time.time())
        ret, ndims, nvalid = self.postprocess(ret)
        times.append(time.time())
        del outdict
        nfeatfiles = len(self.featfiles) if self.featfiles else 0
        if self.debug > 0:
            log('Read %d valid fvecs (%d dims) from %d inputs and %d featfiles in %s' % (nvalid, ndims, len(ret), nfeatfiles, getTimeDiffs(times)))
        #sys.exit()
        return ret

