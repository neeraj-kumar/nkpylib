"""Wrappers and utils for PatchMatch
Originally written by Neeraj Kumar <me@neerajkumar.org>

Licensed under the 3-clause BSD License:

Copyright (c) 2014, Neeraj Kumar (neerajkumar.org)
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
from subprocess import Popen, PIPE, call
from nkpylib.utils import *
from nkpylib.imageutils import imageiter, imagelociter

def pm(src, tgt, annf, annd):
    """Wrapper on patchmatch"""
    for f in [annf, annd]:
        try:
            os.makedirs(os.path.dirname(f))
        except OSError: pass
    call(['pm', src, tgt, annf, annd])

def seg(src, tgt, k, min=100, sigma=0.8):
    """Wrapper on segmentation"""
    try:
        os.makedirs(os.path.dirname(tgt))
    except OSError: pass
    args = 'segment %(sigma)s %(k)s %(min)s <(convert "%(src)s" ppm:- ) >(convert - "%(tgt)s")' % (locals())
    #print args
    #call(args, shell=True, executable='/bin/bash') #FIXME doesn't work for some reason
    call(['/bin/bash', '-c', args])
    time.sleep(0.5)

def annf2data(annf):
    """Converts a raw ann field to a list of (tx, ty) pairs.
    This is in same order and size as getdata().
    Returns (data, maxval)
    """
    im = Image.open(annf) if isinstance(annf, basestring) else annf
    data = []
    maxval = 0
    for val in im.getdata():
        r, g, b = val[:3]
        n = (b << 16) + (g << 8) + r
        ty = (n >> 12)
        tx = n % (1 << 12)
        data.append((tx,ty))
        maxval = max(maxval, tx, ty)
    return data, maxval

def saveandretim(im, fname=None):
    """Saves the given image if fname is not None.
    Returns it as well"""
    if fname:
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError: pass
        im.save(fname)
    return im

def annd2data(annd):
    """Converts raw ANN distance map to list of distances.
    This is in same order and size as getdata().
    """
    im = Image.open(annd) if isinstance(annd, basestring) else annd
    data = []
    for val in im.getdata():
        r, g, b = val[:3]
        n = (b << 16) + (g << 8) + r
        data.append(n)
    return data

def drawmatches(annf, outfname, im1fname, im2fname, scale=5, ss=111, orientation='vertical'):
    """Draws the computed nearest neighbor field using a pair of images"""
    from PIL import ImageDraw
    from imageutils import ImagePair, randcolor
    scale = float(scale)
    ss = int(ss)
    data, maxval = annf2data(annf)
    im1 = Image.open(im1fname).convert('RGB')
    im2 = Image.open(im2fname).convert('RGB')
    dims = im1.size
    im1 = im1.resize((int(im1.size[0]*scale), int(im1.size[1]*scale)), Image.ANTIALIAS)
    im2 = im2.resize((int(im2.size[0]*scale), int(im2.size[1]*scale)), Image.ANTIALIAS)
    ip = ImagePair(im1, im2, orientation=orientation, background=(255,255,255))
    draw = ImageDraw.Draw(ip.outim)
    pairs = []
    i = 0
    assert len(data) == dims[0]*dims[1]
    cs = lambda x: int((x+0.5)*scale)
    for y in range(dims[1]):
        for x in range(dims[0]):
            tx, ty = data[i]
            i += 1
            if i % ss != 0: continue
            c1, c2 = (0, cs(x), cs(y)), (1, cs(tx), cs(ty))
            color = randcolor('RGB')
            if (tx,ty) != (0,0):
                #print x, y, tx, ty, i, c1, c2
                pass
                ip.drawline([c1, c2], fill=color)
    try:
        os.makedirs(os.path.dirname(outfname))
    except OSError: pass
    ip.outim.save(outfname)

def annf2im(annf, outfname=None):
    """Converts an ANN field image (output from PatchMatch) to a readable image.
    If outfname is given, then saves to disk.
    Returns the output image regardless."""
    im = Image.open(annf) if isinstance(annf, basestring) else annf
    data, maxval = annf2data(im)
    fac = 255.0/maxval
    data = [(int(tx*fac), 0, int(ty*fac)) for tx, ty in data]
    outim = Image.new('RGB', im.size)
    outim.putdata(data)
    return saveandretim(outim, outfname)

def segannf(annfim, segfname):
    """Segments an ANN field image (output of annf2im) to get regions."""
    seg(annfim, segfname, k=300)

def scorefuncE(locs, offsets, dists):
    """Given offsets and distances within a region, computes a score.
    This version uses entropy on distances"""
    from nkpylib.hist import histogram
    #print offsets[:5], dists[:5]
    dh = histogram(dists, binwidth=10000, incr=1, normalize=1000.0)
    vals = [v for k, v in sorted(dh.items())]
    if 1:
        e = 30.0*entropy(vals)
    else:
        e = 100.0/entropy(vals) # this is definitely wrong
    #spark(vals)
    return e

def consistency(locs, trans):
    """Returns the consistency between the given pair of locations and transformations"""
    num = l2dist(*trans)
    den = l2dist(*locs)
    ret = (num+1.0)/(den+1.0)
    #print locs, trans, num, den, ret
    return ret

def scorefuncC(locs, offsets, dists, thresh=0.8):
    """Given offsets and distances within a region, computes a score.
    This version uses the coherence as defined in the NRDC paper."""
    from math import sqrt
    #print locs[:5], offsets[:5], dists[:5], len(offsets)
    # sample sqrt() of the pairs
    if len(offsets) < 2: return 0
    n = int(sqrt(len(offsets)))
    pairs = set()
    while len(pairs) < n:
        i, j = minsample(xrange(len(offsets)), 2)
        pairs.add((i,j))
    pairs = sorted(pairs)
    #pairs = [(i, j) for i, u in enumerate(offsets) for j, v in enumerate(offsets[i+1:])]
    #pairs = minsample(pairs, n)
    #print len(pairs), pairs[:5]
    cons = [consistency((locs[i], locs[j]), (offsets[i], offsets[j])) for i, j in pairs]
    ncons = sum(1.0 for c in cons if c > thresh)
    #print cons, ncons, ncons/len(cons)
    error = ncons/len(cons) # 0-1, 1=totally incoherent
    ret = int(255*(error))
    #print len(cons), ret
    return ret

scorefunc = scorefuncC

def score(annseg, annf, annd, outfname=None):
    """Scores regions from the given segmented ANN field.
    'annf' should be the raw field, not the processed one.
    Outputs an 'L' image with scores.
    If outfname is given, writes to that fname.
    Returns the image regardless."""
    from collections import defaultdict
    # read annf and annd and convert to lists
    offs, maxval = annf2data(annf)
    dists = annd2data(annd)
    # aggregate into lists for each region
    locs = defaultdict(list)
    offh = defaultdict(list)
    disth = defaultdict(list)
    seg = Image.open(annseg) if isinstance(annseg, basestring) else annseg
    w, h = seg.size
    for col, off, d, (x,y) in zip(seg.getdata(), offs, dists, imagelociter(seg)):
        if x>=w-7 or y>=h-7: # bottom-right border, with no vals, so ignore
            #print x,w,y,h,col,off,d
            continue
        locs[col].append((x,y))
        offh[col].append(off)
        disth[col].append(d)
    # compute scores per region
    scores = {}
    for col in offh:
        scores[col] = int(min(scorefunc(locs[col], offh[col], disth[col]), 255))
    print 'Got %d regions, with min score %s and max score %s' % (len(scores), min(scores.values()), max(scores.values()))
    # create output
    outim = Image.new('L', seg.size)
    data = [scores.get(col, 0) for col in seg.getdata()]
    outim.putdata(data)
    return saveandretim(outim, outfname)

def offsetimg(fname, off=(3,3)):
    """Offsets the given image by the given amount"""
    from PIL import ImageChops
    im = Image.open(fname)
    im = ImageChops.offset(im, off[0], off[1])
    im.save(fname)

def match(src, dst, dir, *tgts):
    """Matches pixels from src to all tgts and outputs match likelihood to dst.
    The output is the same size as the source, and is an 'L' image."""
    times = [time.time()]
    scoreims = []
    Image.open(src).save('%s-orig.png' % (dst))
    # operate on each image
    for i, tgt in enumerate(tgts):
        def makefname(suffix):
            tgtbase = os.path.basename(tgt).rsplit('.',1)[0]
            fname = os.path.join(dir, '%s-%s.png' % (tgtbase, suffix))
            return fname

        annffname, anndfname = 'annf.png', 'annd.png'
        pm(src, tgt, annffname, anndfname)
        times.append(time.time())
        annfimfname = makefname('annif')
        annfim = annf2im(annffname, annfimfname)
        times.append(time.time())
        anndimfname = makefname('annid')
        dists = annd2data(anndfname)
        dim = Image.new('L', annfim.size)
        fac = 255.0/1000000
        dim.putdata([int(fac*d) for d in dists])
        dim.save(anndimfname)
        times.append(time.time())
        annsegfname = makefname('anns')
        segannf(annfimfname, annsegfname)
        times.append(time.time())
        scoreimfname = makefname('annsc')
        scoreim = score(annsegfname, annffname, anndfname, outfname=scoreimfname)
        times.append(time.time())
        scoreims.append(scoreim)
        for fname in [annfimfname, anndimfname, annsegfname, scoreimfname]:
            offsetimg(fname)
    # add all together
    outim = Image.new('L', scoreims[0].size, 0)
    data = outim.getdata()
    for sim in scoreims:
        data = [d+s for d, s in zip(data, sim.getdata())]
    data = [int(float(d)/len(scoreims)) for d in data]
    outim.putdata(data)
    saveandretim(outim, dst)
    offsetimg(dst)
    print 'Finished matching (%s, %s) -> %s in times %s' % (src, tgts, dst, getTimeDiffs(times, percs=1))

def matchmain(placedir, num=10):
    """Runs matches from the given match directory"""
    datadir = '/'.join(placedir.split('/')[:2])
    j = lambda d, p: os.path.join(d, p)
    tgts = [os.path.join(placedir, 'gt-%03d.jpg' % (i)) for i in range(int(num))]
    matchdir = placedir.replace('/google/', '/matches/')
    match(j(datadir, 'thumb.jpg'), j(matchdir, 'annscored.png'), matchdir, *tgts)


if __name__ == '__main__':
    TASKS = 'matchmain match annf2im drawmatches seg segannf score'.split(' ')
    if len(sys.argv) < 3:
        print 'Usage: python %s <%s> [args]' % (sys.argv[0], '|'.join(TASKS))
        print '       python %s match <src> <dst> <tgt> [<tgt> ...]' % (sys.argv[0])
        print '       python %s annf2im <annf> <dst>' % (sys.argv[0])
        print '       python %s drawmatches <annf> <dst> <im1> <im2> [<scale=5> <ss=111> <orientation=vertical>]' % (sys.argv[0])
