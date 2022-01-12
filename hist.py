"""A histoprogramming utility to go from some input lines to something plottable.

Licensed under the 3-clause BSD License:

Copyright (c) 2011-2014, Neeraj Kumar (neerajkumar.org)
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
import os, sys
from utils import specialize

def optimalBinSize(vals):
    """Returns "optimal" bin size for a histogram using Scott's Rule:
        b = 3.49 * stdev*N^(-1/3),
    stdev = standard deviation of the N descriptor values.
    From D. Scott - On optimal and data-based histograms.
    """
    from utils import getStdDev
    n = len(vals)
    sd = getStdDev(vals)
    return 3.49 * sd*(n**(-1/3))

def makehist(vals, incr=1, normalize=0):
    """Makes a histogram for the given vals and returns a dict.
    Incr is how much each item counts for
    If normalize is true, then the sum of return values is normalize.
    """
    ret = {}
    sum = 0
    for v in vals:
        if v not in ret:
            ret[v] = 0
        ret[v] += incr
        sum += incr
    if normalize:
        for k in ret:
            ret[k] *= (normalize/sum)
    return ret

def centers2edges(centers):
    """Converts a set of bin centers into edges"""
    centers = sorted(set(centers))
    ret = [-1e99]
    ret.extend((c1+c2)/2.0 for c1, c2 in zip(centers, centers[1:]))
    ret.append(1e99)
    return ret

def histfromcenters(vals, centers, incr=1, normalize=0):
    """Makes a histogram from a set of numbers, with bin centers given.
    incr is how much to increment each bin per item.
    if normalize is true, then output has sum=normalize"""
    from bisect import bisect
    edges = centers2edges(centers)
    hist = []
    for v in vals:
        i = bisect(edges, v)
        assert i > 0
        hist.append(centers[i-1])
    if type(vals[0]) == type(1.3) or normalize:
        incr = float(incr)
    hist = makehist(hist, incr, normalize)
    return hist

def histogram(vals, binwidth=1, incr=1, normalize=0):
    """Makes a histogram from a set of values of arbitrary type.
    binwidth determines which values all map to the same value.
    incr is how much to increment each bin per item.
    if normalize is true, then output has sum=normalize"""
    try:
        hist = [(v//binwidth)*binwidth for v in vals]
    except TypeError:
        hist = vals
    if type(vals[0]) == type(1.3) or normalize:
        incr = float(incr)
    hist = makehist(hist, incr, normalize)
    return hist

def cumhist(hist):
    """Takes a histogram and makes a cumulative histogram out of it"""
    ret = {}
    cur = 0
    for k in sorted(hist):
        ret[k] = hist[k]+cur
        cur = ret[k]
    return ret

def multhist(hists, asone=1):
    """Takes a set of histograms and combines them.
    If asone is true, then returns one histogram of key->[val1, val2, ...].
    Otherwise, returns one histogram per input"""
    ret = {}
    num = len(hists)
    for i, h in enumerate(hists):
        for k in sorted(h):
            if k not in ret:
                ret[k] = [0]*num
            ret[k][i] = h[k]
    if asone: return ret
    # otherwise, convert to separate ones
    toret = []
    for i in hists:
        toret.append({})
    for k, vals in ret.iteritems():
        for i, v in enumerate(vals):
            toret[i][k] = v
    return toret

def collapsebins(hist, bin, func=lambda b: b>bin):
    """Collapses bins of a histogram into one, based on the given function.
    The function is given all bins (keys) and for every bin that tests positive, 
    it will collapse it to the chosen bin.

    This function copies the given histogram, rather than modifying it directly.
    """
    hist = dict(**hist)
    todel = []
    for b in hist:
        if func(b):
            hist[bin] += hist[b]
            todel.append(b)
    for b in todel:
        del hist[b]
    return hist

if __name__ == '__main__':
    lines = [specialize(l.strip()) for l in sys.stdin]
    if not lines: sys.exit()
    args = [specialize(a) for a in sys.argv[1:]]
    hist = histogram(lines, *args)
    for k in sorted(hist):
        print '%s %s' % (k, hist[k])
