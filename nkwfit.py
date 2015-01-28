"""Simple access to libMR for weibull fitting and normalization of svm outputs.
You most likely want easyfitsvm() and easynormsvm().

If you use anything from here, please cite our work:

    Walter Scheirer, Neeraj Kumar, Peter N. Belhumeur, Terrance E. Boult,
    "Multi-Attribute Spaces: Calibration for Attribute Fusion and Similarity Search,"
    Proceedings of the 25th IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    June 2012.


Licensed under the 3-clause BSD License:

Copyright (c) 2012-2014, Neeraj Kumar (neerajkumar.org)
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
from subprocess import PIPE, Popen
from ctypes import *

curdir = os.path.abspath(os.path.dirname(__file__))
EXE = 'simple-mr' # must be somewhere on your path!
LIB = 'libsimple-mrlib.so'
try:
    libsimple = cdll.LoadLibrary(LIB)
except Exception:
    libsimple = None


def fitParamsFromSVMOutputs(vals, cls, nfit=9):
    """Fits parameters from the given signed svm output values.
    This is a lower-level function. Use the easy fit function in general.
    cls must be either 1 (positive) or -1 (negative).
    You can optionally pass in the number of values to use for fitting (default 9).
    Returns a string with the params."""
    #TODO there seems to be some issue if we give it negative values and cls==-1, so better to manually make everything positive
    # sort and filter data
    assert cls in [1, -1]
    vals = sorted([v for v in vals if v*cls > 0], reverse=cls>0)
    sign = lambda v: 1 if v > 0 else -1
    inputs = '\n'.join(['%s\t%s' % (sign(v), v) for v in vals])
    # fit the params
    args = [EXE, 'fitsvmpos' if cls > 0 else 'fitsvmneg', '-', str(nfit)]
    sout, serr = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate(inputs)
    params = sout.strip()
    if not params:
        raise Exception(serr.strip())
    return params

def normalizeSVMOutputs(vals, params, stable=1):
    """Normalizes one side of svm outputs using the given params.
    To normalize all outputs from both sides, use the easy norm function.
    The svm outputs should be signed output values.
    The params can be gotten from fitParamsFromSVMOutputs().
    If stable is true, then decrements a small amount from each output in order
    to maintain the relative order of values which would have become identical in the output.
    Returns a list of normalized float values."""
    #TODO see if there is an easy way to not keep writing tempfiles with the params (which are probably not changing too much)
    import tempfile
    from nkutils import getTimeDiffs
    if not vals: return []
    # write params to a temp file
    f = tempfile.NamedTemporaryFile()
    print >>f, params
    f.flush()
    # normalize
    if libsimple:
        times = [time.time()]
        #print LIB, libsimple, libsimple.normalize_raw
        datatype = c_double * len(vals)
        data = datatype(*vals)
        out = datatype()
        times.append(time.time())
        libsimple.normalize_raw(f.name, data, len(vals), out)
        times.append(time.time())
        #void normalize_raw(char *paramsfname, double *data, int ndata, double *ret){
        normvals = map(float, out)
    else:
        inputs = '\n'.join(map(str, vals))
        times.append(time.time())
        sout, serr = Popen([EXE, 'normalize', f.name], stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate(inputs)
        times.append(time.time())
        normvals = map(float, sout.split())
    times.append(time.time())
    #print zip(normvals, normvals1)
    #print getTimeDiffs(times, percs=1)
    assert len(normvals) == len(vals)
    if stable:
        # get an ordered list
        ordered = sorted([(v, i) for i, v in enumerate(vals)], reverse=1)
        for rownum, (v, i) in enumerate(ordered):
            # decrement small amount from each item
            decr = rownum*1e-8
            #print rownum, v, i, decr, normvals[i], normvals[i]-decr
            normvals[i] -= decr
    return normvals

def easyfitsvm(vals, nfitmin=9, nfitmax=25, nfitfac=0.5):
    """Does "easy" w-fitting of both sides of an SVM from a given set of classification outputs.
    This tries to compute the nfit param based on:
        nfitmin - At least this many points are used
        nfitfac - number of valid vals * this factor are used
    It also deals with exceptions (i.e., too few points).
    It returns [posparams, negparams], either of which might be None if there was an issue.
    """
    ret = []
    for cls in [1, -1]:
        params = None
        if cls == 1:
            cur = [v for v in vals if v > 0]
        else:
            cur = [-v for v in vals if v <= 0]
        nfit = min(nfitmax, max(nfitmin, int(len(cur)*nfitfac)))
        try:
            #print len(cur), nfit
            params = fitParamsFromSVMOutputs(cur, 1, nfit=nfit)
        except Exception, e:
            print >>sys.stderr, 'Exception when trying to fit params using %d inputs and nfit %d: %s' % (len(cur), nfit, e)
        ret.append(params)
    return ret

def easynormsvm(vals, bothparams, stable=1):
    """Does "easy" normalization of SVM outputs from both sides of an svm.
    Give it a list of values to normalize, and (posparams, negparams).
    The latter is the output from easyfitsvm().
    Returns normalized versions of vals, in the same order as inputs.
    It's more efficient to call this function fewer times with more vals,
    because we will eventually be spawning a new process and writing to it.
    We also set the output to be stable by default, meaning we slightly
    tweak probabilities so that things remain in sorted order as they were.
    """
    # first sort into pos and neg vals, and keep track of which element went where
    posvals, negvals = [], []
    indices = []
    for v in vals:
        if v > 0:
            posvals.append(v)
            indices.append(1)
        elif v < 0:
            negvals.append(-v)
            indices.append(-1)
        else: # v == 0
            indices.append(0)
    # do the actual normalization
    posparams, negparams = bothparams
    #print 'entered easynorm and got %s and %s' % (bothparams, vals)
    if posparams and posvals:
        posvals = normalizeSVMOutputs(posvals, posparams, stable=stable)
        #print 'Ran pos: %s' % (posvals,)
    if negparams and negvals:
        negvals = normalizeSVMOutputs(negvals, negparams, stable=stable)
        #print 'Ran neg: %s' % (negvals,)
    # assemble the outputs
    ret = []
    posi, negi = 0, 0
    for i in indices:
        if i == 1:
            ret.append(posvals[posi])
            posi += 1
        elif i == -1:
            ret.append(-negvals[negi])
            negi += 1
        elif i == 0:
            ret.append(0.0)
    return ret

# testing main, uses SGDSVM from nktrainutils (which uses scikit-learn)
if __name__ == '__main__':
    from nktrainutils import SGDSVM
    import numpy as np
    import numpy.random as npr
    from pprint import pprint
    svm = SGDSVM()
    size = (1000,100)
    posm = 0.8
    negm = -0.9
    # train
    trainpos = npr.normal(posm, size=size)
    trainneg = npr.normal(negm, size=size)
    model, _ = svm.train(np.vstack((trainpos, trainneg)), [1]*len(trainpos)+[-1]*len(trainneg), ncv=0)
    # eval
    evalpos = npr.normal(posm, size=size)
    evalneg = npr.normal(negm, size=size)
    outs = svm.classify(model, np.vstack((evalpos, evalneg)))
    # fit
    params = easyfitsvm(outs)
    # norm
    outs = sorted(outs[::40])
    nouts = easynormsvm(outs, params)
    oldsort = np.argsort(outs)
    newsort = np.argsort(nouts)
    pprint(zip(outs, nouts, oldsort, newsort))
