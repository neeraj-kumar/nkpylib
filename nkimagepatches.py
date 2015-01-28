"""A set of utilities to deal with image patches.

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

import os, sys, time
from itertools import *
from array import array
from PIL import Image, ImageFilter
from nkutils import *

def applyFunc(func, num, input):
    """Recursively applies a given function 'num' times.
    Assumes func takes input as its sole argument, and returns something equivalent."""
    cur = input
    for i in range(num):
        cur = func(cur)
    return cur

def getHistogram(lst, nbins, minval=None, maxval=None):
    """Returns a histogram of values in the given histogram.
    If no minval and maxval are given, then are extracted from the data."""
    ret = array('f', [0] * nbins)
    if minval is None:
        minval = min(lst)
    if maxval is None:
        maxval = max(lst)
    try:
        factor = nbins/float(maxval-minval)
    except ZeroDivisionError: factor = 1.0
    bins = [clamp(int((x-minval)*factor), 0, nbins-1) for x in lst]
    for b in bins:
        ret[b] += 1
    return ret

def getRectsFromArgs():
    """Helper method to get a list of patches from command line args"""
    if len(sys.argv[0]) < 2:
        print 'Usage: python %s <image> [<num levels=5>]' % (sys.argv[0])
        sys.exit()
    fname = sys.argv[1]
    im = Image.open(fname).convert('L')
    rects = getPatchRects(im, (7,7), 1)
    try:
        num = int(sys.argv[2])
    except IndexError: num = 5
    return im, rects, num

def dumpPatches():
    """A main method for extracting normalized patches from images and dump them to stdout"""
    im, rects, num = getRectsFromArgs()
    ret = []
    for i in xrange(num+1):
        #print 'Image size is', im.size
        print '  On iteration %d of %d...' % (i, num)
        patches = [getPatchAsLst(im, r) for r in rects]
        npatches = [normalize(p, 2.5) for p in patches]
        f = open(fname+'.patches_%d' % i, 'w')
        for np in npatches:
            print >> f, getListAsStr(np, sep=' ')
        f.close()
        im = im.filter(ImageFilter.BLUR)

def normalizesum(seq, val=1.0):
    """Normalizes the given sequence"""
    return normalize(seq, val=val, power=1.0)

def smooth(seq, sigma=1.0):
    """Smooths a given sequence by the specified amount"""
    import operator as op
    facs = [0.25, 0.5, 0.25]
    facs = [0.1, 0.8, 0.1]
    avgs = [sum(map(op.mul, facs, seq[i-1:i+2])) for i in range(1, len(seq)-1)]
    ret = [seq[0]] + avgs + [seq[-1]]
    return ret


def testDistances(npairs=1000):
    """A main method for testing distance relations between patches and histograms"""
    from random import sample
    im, rects, num = getRectsFromArgs()
    rects = [r for i, r in enumerate(rects) if i % 15 == 0]
    testset = [sample(xrange(len(rects)), 2) for i in xrange(npairs)]
    print testset[0]
    print 'Got %d rects and num = %s' % (len(rects), num)
    allpatches = []
    allhists = []
    for i in xrange(num+1):
        print '  On iteration %d of %d...' % (i, num)
        patches = [getPatchAsLst(im, r) for r in rects]
        #patches = [normalize(p, 2.5) for p in patches]
        patches = [normalizesum(p) for p in patches]
        hists = [getHistogram(p, 128, minval=0, maxval=1.0) for p in patches]
        if i == 0:
            allpatches = patches[:]
            allhists = hists[:]
        else:
            allhists = [h1+h2 for h1, h2 in zip(allhists, hists)]
    allhists = [array('f', smooth(normalizesum(h))) for h in allhists]
    dists = []
    for i, j in testset:
        p1, p2 = allpatches[i], allpatches[j]
        h1, h2 = allhists[i], allhists[j]
        dp = l2dist(p1, p2)
        hp = linfdist(h1, h2)
        if dp+hp < 0.0001: continue
        dists.append((hp, dp))
    print getListAsStr([p1, p2, h1, h2, dp, hp], sep='\n')
    from plot import plotfunc
    plotfunc(dists, {'title': 'Patch vs Histogram Distances', 'y': 'Patch L2 distances', 'x': 'Histogram L_inf distances', 'plotstrs': ['ro']})

if __name__ == '__main__':
    #dumpPatches()
    testDistances()

