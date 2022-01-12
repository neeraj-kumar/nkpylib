"""Some simple utilities to align subtitles.
By Neeraj Kumar <me@neerajkumar.org>

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
from pprint import pprint

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def parsetime(s):
    """Parses a time value into a float representing number of seconds.
    Examples:
    >>> parsetime('00:01:10,070')
    70.069999999999993
    >>> parsetime('00:00:00,000')
    0.0
    >>> parsetime('00:00:01,000')
    1.0
    >>> parsetime('00:04:00,000')
    240.0
    >>> parsetime('12:34:56,789')
    45296.788999999997
    """
    import re
    m = re.search(r'(?P<hr>\d*):(?P<min>\d*):(?P<sec>\d*),(?P<ms>\d*)', s)
    hr, min, sec, ms = [int(t) for t in m.group('hr', 'min', 'sec', 'ms')]
    return sec + 60*min + 60*60*hr + ms/1000.0

def parseinterval(s, delim=' --> '):
    """Parses a time interval.
    Examples:
    >>> parseinterval('00:01:10,070 --> 00:01:15,030')
    (70.069999999999993, 75.030000000000001)
    >>> parseinterval('01:26:41,362 --> 01:26:43,853')
    (5201.3620000000001, 5203.8530000000001)
    """
    t1, t2 = [parsetime(t.strip()) for t in s.split(delim, 1)]
    return (t1, t2)

def secs2time(t):
    """Converts number of seconds into string time value"""
    from utils import collapseSecs
    units = y, d, h, m, s = collapseSecs(t)
    ms = 1000.0*(s-int(s))
    return '%02d:%02d:%02d,%03d' % (h, m, int(s), ms)


def getWindow(n, type='rect', order=0):
    """Returns a window of the given length, type and order.
    Types are:
        'rect' or 'dirichlet': rectangular window
        'tri' or 'triangle' or 'bartlett': triangle window with 0-endpoints
        'hamming': hamming window
        'han' or 'hanning': hanning window
        'lanczos' or 'sinc': lanczos window
    Order refers to derivatives. It can be either 0 (no deriv) or 1 (1st deriv).
    Examples:
    >>> getWindow(8)
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> getWindow(8, 'tri')
    [0.0, 0.2857142857142857, 0.5714285714285714, 0.8571428571428571, 0.8571428571428571, 0.5714285714285714, 0.2857142857142857, 0.0]
    >>> getWindow(8, 'hamming')
    [0.076719999999999955, 0.25053216786993415, 0.64108456395159286, 0.95428326817847264, 0.95428326817847275, 0.64108456395159297, 0.25053216786993426, 0.076719999999999955]
    >>> getWindow(8, 'hann')
    [0.0, 0.1882550990706332, 0.61126046697815717, 0.95048443395120952, 0.95048443395120952, 0.61126046697815728, 0.18825509907063331, 0.0]
    >>> getWindow(8, 'sinc')
    [1.0, 0.87102641569756023, 0.54307608733699464, 0.16112773088475874, -0.120845798163569, -0.21723043493479788, -0.14517106928292672, -3.8980430910514779e-017]
    >>> getWindow(8, 'rect', order=1)
    [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    >>> getWindow(8, 'bartlett', order=1)
    [0.0, -0.2857142857142857, -0.5714285714285714, -0.8571428571428571, 0.8571428571428571, 0.5714285714285714, 0.2857142857142857, 0.0]
    >>> getWindow(8, 'hamming', order=1)
    [-0.076719999999999955, -0.25053216786993415, -0.64108456395159286, -0.95428326817847264, 0.95428326817847275, 0.64108456395159297, 0.25053216786993426, 0.076719999999999955]
    >>> getWindow(8, 'hanning', order=1)
    [0.0, -0.1882550990706332, -0.61126046697815717, -0.95048443395120952, 0.95048443395120952, 0.61126046697815728, 0.18825509907063331, 0.0]
    >>> getWindow(8, 'lanczos', order=1)
    [-1.0, -0.87102641569756023, -0.54307608733699464, -0.16112773088475874, -0.120845798163569, -0.21723043493479788, -0.14517106928292672, -3.8980430910514779e-017]
    """
    from math import pi, cos, sin
    assert order in [0, 1]
    type = type.lower()
    valid = 'rect dirichlet tri triangle bartlett hamming hann hanning lanczos sinc'.split()
    assert type in valid
    # first get the window for the 0th order
    n = int(n)
    n1 = float(n-1.0)
    if type in 'rect dirichlet'.split():
        ret = [1.0] * n
    elif type in 'tri triangle bartlett'.split():
        ret = [(2.0/n1) * ((n1/2.0) - abs(i - (n1/2.0))) for i in range(n)]
    elif type == 'hamming':
        ret = [0.53836 - 0.46164*cos(2*pi*i/n1) for i in range(n)]
    elif type in 'hanning hann'.split():
        ret = [0.5 * (1-cos(2*pi*i/n1)) for i in range(n)]
    elif type in 'lanczos sinc'.split():
        def sinc(x):
            try:
                return sin(pi*x)/(pi*x)
            except ZeroDivisionError: return 1.0
        ret = [sinc(2*i/n1) for i in range(n)]
    # now if the order is 1, then negate the first half
    if order == 1:
        facs = [-1.0]*(n//2) + [1.0]*(n - (n//2))
        ret = [r*f for r, f in zip(ret, facs)]
    return ret

def normalize(arr, total=1.0):
    """Normalizes an array to have given total sum"""
    try:
        fac = total/float(sum([abs(v) for v in arr]))
    except ZeroDivisionError: fac = 1.0
    return [v*fac for v in arr]

def getTimes(from_, to, incr=1.0, frommid=1):
    """Returns a list of "times" in the given range and incr.
    If frommid=1, then returns in increasing distance from midpoint.
    Examples:
    >>> getTimes(-5, 5, 1)
    [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
    >>> getTimes(-5, 5, 1, 0)
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    >>> getTimes(-120.0, 100.0, 25.5)
    [-18.0, 7.5, -43.5, 33.0, -69.0, 58.5, -94.5, 84.0, -120.0]
    """
    ret = []
    i = from_
    while i <= to:
        ret.append(i)
        i += incr
    if frommid:
        mid = (from_ + to)/2
        ret = sorted(ret, key=lambda t: abs(t-mid))
    return ret

def getdialogs(lines):
    """Returns a list of (time, dialogs) from the given lines"""
    ret = []
    times, txt = None, ''
    for l in lines:
        try: # see if it's an index line
            num = int(l)
            if times and txt:
                txt = txt.strip().replace('  ', ' ')
                ret.append((times, txt))
                times = None
                txt = ''
            continue
        except ValueError: pass
        if '-->' in l: # see if it's a time line
            times = parseinterval(l)
        else: # assume it's text
            txt += ' ' + l
    return ret

def shiftdialogs(dialogs, offset):
    """Shifts dialogs ((from, to), txt) by the given amount of offset"""
    ret = [((a+offset,b+offset), txt) for (a, b), txt in dialogs]
    return ret

def getSubtitleStarts(f=sys.stdin):
    """Returns a list of start times for dialogs in the given file"""
    lines = [l.strip() for l in f if l.strip()]
    dialogs = getdialogs(lines)
    times, txts = zip(*dialogs)
    starts, ends = zip(*times)
    return starts

class AudioFile(object):
    """A simple container for an audio file"""
    def __init__(self, fname, newrate=0):
        """Initializes an audio file from an uncompressed wavefile on disk.
        The file is converted to mono, and if newrate is positive, then the rate is converted"""
        import wave, audioop
        try: # see if we have numpy
            from numpy import array
            self.numpy = 1
        except ImportError: self.numpy = 0
        # read data
        f = wave.open(fname, 'rb')
        nchans, w, fps, nframes, comptype, compname = f.getparams()
        print "Read audio file %s with %d chans, %d width, %d fps and %d frames" % (fname, nchans, w, fps, nframes)
        self.width, self.fps = w, fps
        self.dat = f.readframes(nframes)
        print "  Original data length was %d" % (len(self.dat))
        # convert to mono and (optionally) convert the rate
        self.dat = audioop.tomono(self.dat, w, 0.5, 0.5)
        print "  After mono, data length is %d" % (len(self.dat))
        if newrate > 0:
            self.dat, junk = audioop.ratecv(self.dat, w, 1, fps, newrate, None)
            self.fps = newrate
            print "  Converted to new rate %s, and data length is now %d" % (self.fps, len(self.dat))
        # now extract the data into a simple array
        from audioop import getsample
        self.dat = [abs(getsample(self.dat, w, i)) for i in range(len(self.dat)//w)]
        print "  Final data length is now of length %s" % (len(self.dat),)
        if self.numpy:
            self.dat = array(self.dat)

    def t2i(self, t):
        """Converts a time (in secs) to an index number"""
        return int(self.fps * t)

    def i2t(self, i):
        """Converts an index number to a time (in secs)"""
        return i/float(self.fps)

    def _get(self, i):
        """Returns a value at the given index, or 0 on error"""
        if i < 0 or i >= len(self.dat): return 0
        return self.dat[i]

    def __getitem__(self, i):
        """Returns the data at the given index number (NOT time) or slice.
        Use t2i to get an index number from a time"""
        try: # slice
            return [self._get(i) for i in range(*i.indices(len(self.dat)))]
        except AttributeError: # int/long
            return self._get(i)

    def getEnergy(self, t, win):
        """Returns the "energy" at the given time, using the given windowing func"""
        starti = self.t2i(t) - len(win)//2
        t1 = time.time()
        if self.numpy and starti >= 0 and starti < len(self.dat)-len(win):
            ret = sum(self.dat[starti:starti+len(win)] * win)
        else:
            ret = sum((v*w for v, w in izip(self[starti:], win)))
        elapsed = time.time()-t1
        #print '    Energy at time %s (i=%s) is %s (computed in %0.2f secs)' % (t, self.t2i(t), ret, elapsed)
        return ret


def getScore(off, starts, enfunc):
    """Returns the score for a given offset.
    The enfunc is the function that returns energy for a given time"""
    i = 0
    cur = starts[i]+off
    while cur < 0:
        i += 1
        cur = starts[i]+off
    s = sum((max(0, enfunc(t+off)) for t in starts[i:]))
    print '  For offset %s, started at %s and got a sum of %s' % (off, i, s)
    return s

def getSyncOffset(st, au, from_=-50, to=50, resolutions=[(1,5),(0.1,1),(0.01,1)]):
    """Returns the sync offset from the given subtitle start times and audio file,
    within the given "from_" and "to" params and using the given list of resolutions"""
    ret = []
    res, ntop = resolutions[0]
    win = normalize(getWindow(2*res*au.fps, type='rect', order=1))
    try:
        from numpy import array
        win = array(win)
    except ImportError: pass
    times = getTimes(from_, to, res, frommid=0)
    print "Doing resolution of %s secs, and got a window of length %s and times from %s to %s with length %s" % (res, len(win), from_, to, len(times))
    enfunc = lambda t: au.getEnergy(t, win)
    start = time.time()
    offs = sorted([(off, getScore(off, st, enfunc)) for off in times], reverse=1, key=lambda o: o[1])
    elapsed = time.time() - start
    print "  Computed %d scores in %0.2f secs (%0.2f scores/sec). Exploring top %d values" % (len(offs), elapsed, len(offs)/elapsed, ntop)
    for i in range(ntop):
        best = offs[i]
        print "    Top offset %d is at %s with score %s" % (i, best[0], best[1])
        from_, to = best[0]-res, best[0]+res
        if len(resolutions) > 1:
            ret.extend(getSyncOffset(st, au, from_, to, resolutions[1:]))
        else:
            return [((from_+to)/2.0, best[1])]
    return ret

def syncmain():
    base = sys.argv[1]
    stfname = '%s.srt' % (base)
    aufname = '%s.wav' % (base)
    starts = getSubtitleStarts(f=open(stfname))
    starts = starts[:100]
    print 'There are %d start times, the first few are: %s' % (len(starts), starts[:5])
    au = AudioFile(aufname, newrate=1200)
    t1 = time.time()
    offsets = getSyncOffset(starts, au)
    print 'Got final offsets of %s in %0.2f secs' % (offsets, time.time()-t1)

def simplemain():
    """Simply adds a given offset to the file"""
    if len(sys.argv) < 3:
        print 'Usage: python %s <srt filename> <offset in ms>' % (sys.argv[0])
        sys.exit()
    fname = sys.argv[1]
    offset = int(sys.argv[2])

def chunks2str(chunks):
    """Takes a list of chunks: (i,j) pairs, and makes a string"""
    s = ''
    lastj = 0
    for i, j in chunks:
        if i > lastj:
            s += ' '
        s += '-'*(j-i)
        s += '|'
        lastj = j
    return s

def shiftchunk(chunks, c, which, incr):
    """Shifts the 'which' endpoint of chunk 'c' by 'incr'.
    """
    ret = [ch[:] for ch in chunks]
    ret[c][which] += incr
    last = ret[c][which]
    if which == 1:
        for w in range(c+1, len(ret)):
            oldi, oldj = i, j = ret[w]
            if i < last:
                i = last
            if j < i:
                j = i
            #print '%s (%s,%s) -> (%s,%s)' % (w, oldi, oldj, i, j)
            last = j
            if (i, j) == (oldi, oldj): break
            ret[w] = [i,j]
    else:
        for w in range(c-1, -1, -1):
            oldi, oldj = i, j = ret[w]
            if j > last:
                j = last
            if i > j:
                i = j
            #print '%s (%s,%s) -> (%s,%s)' % (w, oldi, oldj, i, j)
            last = i
            if (i, j) == (oldi, oldj): break
            ret[w] = [i,j]
    return ret


def textaudiomainauto(txtfname, labelsfname, subfname):
    """A driver that takes a text and label file and creates subtitles.
    This tries to do it automatically, but doesn't work too well.
    The txt file should contain blank lines for major parts with no dialog.
    Lines starting with '(' are for signs in the video (no speech).
    The labels are as output from audacity's labeling feature:
        start time in seconds \t end time in seconds \t optional label
    (The labels are ignored.)
    """
    # Read script and tokenize into chunks
    import re
    from utils import memoize, spark, partitionByFunc
    import numpy as np
    if 0:
        DLM = '([".,;:?!\n][\n]?)'
        DLMSPACE = '([ ".,;:?!\n][\n]?)'
    else:
        DLM = '([".,;:?!\n]+)'
        DLMSPACE = '([ ".,;:?!\n]+)'
    lines = [l.strip() for l in open(txtfname)]
    full = ' '.join([l.strip() for l in open(txtfname) if l.strip()])
    ntotallines = len(lines)
    #script = [l.strip() for l in open(txtfname) if not l.startswith('(')]
    allseqs, indices = partitionByFunc(lines, lambda s: 'comment' if s.startswith('(') else 'script')
    #indices is a dictionary of (outval, i) -> orig_i, which allows mapping results back.
    comments, script = allseqs['comment'], allseqs['script']
    script = '\n'.join(script)
    while '\n\n' in script:
        script = script.replace('\n\n', '\n')
    nlines = len(script.split('\n'))
    nchars = len(script)
    nwords = len(list(re.finditer(DLMSPACE, script)))
    tokens = list(re.finditer(DLM, script))
    locs = set([0, len(script)-1])
    for t in tokens:
        locs.add(t.end())
    locs = sorted(locs)
    toks = ['%s (%s)' % (t.group(), t.span()) for t in tokens]
    print 'Read %d non-comment script lines (%d words, %d tokens, %d chars, %d locs): %s %s' % (nlines, nwords, len(tokens), nchars, len(locs), toks[:4], locs[:4])
    # Read labels and compute speaking rates
    labels = [map(float, l.strip().split('\t')[:2]) for l in open(labelsfname)]
    llens = [b-a for a, b in labels]
    totalsecs = sum(llens)
    print 'Read %d labels, %0.2f secs: %s' % (len(labels), totalsecs, zip(labels, llens)[:2])
    wpm = nwords/(totalsecs/60.0)
    spc = totalsecs/nchars
    print 'Got %0.1f wpm, %0.4f secs per char' % (wpm, spc)

    # Define cost function and memoize it
    def costfunc(labelnum, start, end, zerocost=0.2, spc=spc):
        """Computes the cost (in secs) of assigning the given start and end locs to the label.
        The locs are specified w.r.t. to the 'locs' array. They can be identical.
        If the length is 0, the cost is 'zerocost'.
        Else, the cost is (length of label) - (length of chunk)*spc
        Notice that's signed: positive means label is longer than chunk, and vice versa.
        """
        if start == end: return zerocost
        t = llens[labelnum]
        try:
            i, j = locs[start], locs[end]
            nchars = j-i
            nsecs = spc*nchars
            #print t, i, j, nchars, nsecs
            return t - nsecs
        except:
            return zerocost

    C = memoize(costfunc)
    #print C(0, 0, 0)
    #print C(0, 0, 1)
    #print C(0, 0, 2)
    #print C(0, 1, 2)

    # Initialize chunks
    M = len(locs)-1
    fac = M/float(len(llens))
    chunks = [[min(int(i*fac),M),min(int((i+1)*fac),M)] for i in range(len(llens))]
    print len(llens), len(chunks), llens[:5], chunks[:5]+chunks[-5:]
    if 0:
        print locs
        for a,b in zip(locs, locs[1:]):
            print '<%s>' % (script[a:b].strip())
        sys.exit()
    costs = [C(i, a,b) for i, (a,b) in enumerate(chunks)]
    acosts = np.abs(np.array(costs))
    best = [sum(acosts), chunks]
    iter = 0
    from random import randint
    while iter < 10:
        iter += 1
        n = np.argmax(acosts)
        mc = costs[n]
        which = randint(0,1)
        print 'On iter %d, total cost %0.3f, maxcost %0.3f at %d, shifting %d' % (iter, sum(acosts), mc, n, which)
        print '  %s' % (chunks2str(chunks))
        if mc < 0: # label shorter than chunk
            incr = 1 if which == 0 else -1
        else: # label longer than chunk
            incr = 1 if which == 1 else -1
        newchunks = shiftchunk(chunks, n, which, incr)
        costs = [C(i, a,b) for i, (a,b) in enumerate(newchunks)]
        acosts = np.abs(np.array(costs))
        if sum(acosts) < best[0]:
            chunks = newchunks
    print chunks
    # now write output
    sf = srtfile(subfname)
    last = 0
    #print full
    for idx, ((i, j), (t0, t1)) in enumerate(zip(chunks, labels)):
        if i == j: continue
        if i < 0 or j >= len(locs): continue
        s = script[locs[i]:locs[j]].strip()
        try:
            n = full.index(s.replace('\n', ' '))
        except Exception, e:
            print '  ERROR: |%s|: %s' % (s, full[:200])
            #TODO this is because of comments inside the s
            n = 1
            #raise

        if n > 0:
            # we have some skipped stuff, so dump it all in a single line
            dif = 0.05*(t0-last) # so we're not touching boundaries
            sf(full[:n].strip(), last+dif, t0-dif)
        #print '%d: %s' % ((full.index(s) if s in full else -1), s)
        full = full[n+len(s):].strip()
        # see if we have any skipped things todo
        sf(s, t0, t1)
        last = t1
    t0 = time.time()
    x = playpreview(videofname, subfname, 35, 45)
    print 'Got out %s in %0.3fs' % (x, time.time()-t0)
    print 'hello continuing on'

def srtfile(fname, els=None):
    """Creates an iterator for writing subtitles to the given filename.
    If you give no els (default), then returns a function that you call with
    (s, t0, t1) to add to the file.
    If you give a list of elements, then they are assumed to be args to pass in.
    The args can be either (txt, t0, t1), or ((t0, t1), txt)
    Then file is then closed. Nothing is returned.
    """
    f = open(fname, 'wb')
    num = [1]
    def writeToSrt(s, t0, t1, f=f, num=num):
        """Writes the given string from t0 to t1.
        Deals with newlines and numbering"""
        s = s.rstrip('\n')
        print >>f, num[0]
        print >>f, '%s --> %s' % (secs2time(t0), secs2time(t1))
        print >>f, s + '\n'
        f.flush()
        num[0] += 1

    if els is None: return writeToSrt
    for args in els:
        if len(args) == 3:
            writeToSrt(*args)
        elif len(args) == 2 and len(args[0]) == 2:
            (t0, t1), s = args
            writeToSrt(s, t0, t1)
        else:
            raise ValueError()
    f.close()

def tokenize(s, DLM='([".,;:?!\n]+)'):
    """Tokenizes the given string into a list of strings."""
    import re
    tokens = list(re.finditer(DLM, s))
    locs = set([0, len(s)-1])
    for t in tokens:
        locs.add(t.end())
    locs = sorted(locs)
    tokens = [s[i:j].strip() for i, j in zip(locs, locs[1:])]
    return tokens

def readlabels(labelsfname, spacelen=-1):
    """Reads and returns (labels, llens) from labelsfname.
    If spacelen < 0 (default), then only does the listed labels.
    Otherwise, also includes spaces between labels, if they are >= spacelen.
    """
    labels = [map(float, l.strip().split('\t')[:2]) for l in open(labelsfname)]
    last = 0
    all = []
    for i, j in labels:
        if spacelen >= 0 and i-last >= spacelen:
            all.append([last, i])
        all.append([i, j])
        last = j
    labels = all
    llens = [b-a for a, b in labels]
    print 'Read %d labels from %s: %s' % (len(labels), labelsfname, zip(labels, llens)[:2])
    return (labels, llens)

def textaudiomain(txtfname, labelsfname, videofname, subfname, chunksfname='chunks.json'):
    """A driver that takes text, label, and video files and creates subtitles.
    This is run on an interactive loop.
    The txt file should contain blank lines for major parts with no dialog.
    Lines starting with '(' are for signs in the video (no speech).
    The labels are as output from audacity's labeling feature:
        start time in seconds \t end time in seconds \t optional label
    (The labels are ignored.)
    The video file is used to preview the subtitles.
    """
    import json
    # Read script and tokenize
    from utils import memoize, spark, partitionByFunc
    import numpy as np
    lines = [l.strip() for l in open(txtfname) if l.strip()]
    script = '\n'.join(lines)
    while '\n\n' in script:
        script = script.replace('\n\n', '\n')
    #print script
    tokens = tokenize(script)
    print 'Read %d lines from %s, and got %d tokens' % (len(lines), txtfname, len(tokens))
    # Read labels
    labels, llens = readlabels(labelsfname, 1)
    try:
        chunks = json.load(open(chunksfname))
    except:
        chunks = [[] for l in labels]
    def makesubels():
        """Makes list of subels"""
        els = []
        for chunk, (t0, t1) in zip(chunks, labels):
            if not chunk: continue
            s = ''.join(tokens[c] for c in chunk)
            els.append((s, t0, t1))
        return els

    # run main loop
    L = 0
    T = 0
    incr = 5
    exit = 0
    while not exit:
        if chunks[L]:
            T = chunks[L][-1]
        print '\nOn label %d of %d (%d done), tokens around %d:' % (L, len(labels), sum(1 for c in chunks if c), T)
        m, M = max(T-incr, 0), min(T+incr+1, len(tokens))
        print
        for i in range(m, M):
            print '  %d: %s' % (i, tokens[i])
        t0, t1 = labels[L]
        print '\n%s - %s (%0.3f secs): %s' % (secs2time(t0), secs2time(t1), t1-t0, chunks[L])
        print 'Options: (v/V)ideo, (p)rev/(n)ext label, (P)rev/(N)ext tokens, (q)uit, #, #-#, (e)mpty'
        opts = raw_input('> ').split(',')
        if not opts: continue
        for opt in opts:
            if opt == 'q':
                exit = 1
                break
            if opt[0] in 'VvpnPN':
                # get parameter
                if ':' in opt:
                    opt, num = opt.split(':')
                    num = int(num)
                else:
                    num = 1
                if opt == 'V':
                    playpreview(videofname, makesubels(), t0, t1, pad=1.5*num)
                elif opt == 'v':
                    playpreview(videofname, makesubels(), t0, t1, pad=0.2*num)
                elif opt == 'p':
                    L = max(0, L-num)
                    t0, t1 = labels[L]
                elif opt == 'n':
                    L = min(L+num, len(labels)-1)
                    t0, t1 = labels[L]
                elif opt == 'P':
                    T = max(0, T-(incr*num))
                elif opt == 'N':
                    T = min(len(tokens)-1, T+(incr*num))
            elif opt[0] in '0123456789':
                if '-' in opt:
                    i, j = map(int, opt.split('-'))
                    chunks[L] = range(i,j+1)
                else:
                    chunks[L] = [int(opt)]
            elif opt == 'e':
                chunks[L] = []
        json.dump(chunks, open(chunksfname, 'wb'), indent=2)
    # now write out full files
    els = makesubels()
    srtfile(subfname, els)
    print 'Exited loop and wrote %d els to %s' % (len(els), subfname)


def playpreview(videofname, subels, start, stop, pad=1.5):
    """Plays a quick preview, writing the subtitles to a tempfile."""
    #vlc $VIDEOFILE :start-time=$SECONDS :sub-file=$SUBFILE :subsdec-encoding=UTF-8
    tempfname = '.temp-%f.srt' % (time.time())
    if subels:
        srtfile(tempfname, subels)
    start = max(start-pad, 0)
    stop = stop+pad
    sub = ':sub-file=%s :subsdec-encoding=UTF-8 ' % tempfname if subels else ''
    cmd = 'vlc "%s" :start-time=%s :stop-time=%s %s --play-and-exit --no-osd --verbose=0 2>/dev/null >/dev/null' % (videofname, start, stop, sub)
    x = os.system(cmd)
    try:
        os.remove(tempfname)
    except Exception: pass
    return x

def extractAudio(fname, outfname):
    """Extracts audio from the given movie in wav format to the output file.
    Uses vlc's command line mode"""
    from subprocess import call
    try:
        os.makedirs(os.path.dirname(outfname))
    except OSError: pass
    outarg = '#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst="%s"}' % (outfname)
    retcode = call(['vlc', '-I', 'dummy', fname, '--no-sout-video', '--sout', outarg, 'vlc://quit'])

def extractAudioFeatures(data, rate, ftype='mfcc', incr=5, start=0, stop=-1, normalize=1):
    """Extracts audio features from an audio buffer.
    The audio data and sampling rate can be gotten using:
        import scipy.io.wavfile as wav
        rate, data = wav.read('blah.wav')

    Specify the feature type as either 'mfcc', 'ssc', 'fbank' (which is logfbank)
    Reads the audio in increments of the given number of seconds.
    First subsamples data from the given start and stop times (in secs).
    If stop < 0, goes to end.
    If normalize is true (default), then normalizes the segment first
    If there's an error, returns None

    Uses python_speech_features library:
        https://github.com/jameslyons/python_speech_features

    For reference, it looks like 1 second of audio returns:
        200 x 13 mfcc features
        200 x 20 ssc features
        200 x 26 fbank features

    As of July 12, 2014, all feats are roughly 40x input time (48khz),
    """
    import numpy as np
    from features import mfcc, logfbank, ssc
    #print '%s %s' % (start, stop)
    if stop < 0:
        stop = len(data)
    data = data[int(start*rate):int(stop*rate)]
    #print len(data), start*rate, stop*rate, data[:10]
    #sys.exit()
    if len(data) == 0 or data.max() == 0: return None
    if normalize:
        data = normaudio(data)
        pass
    cur = 0
    ret = []
    FEATS = dict(mfcc=mfcc, fbank=logfbank, ssc=ssc)
    try:
        featfunc = FEATS[ftype]
    except KeyError: raise NotImplementedError()
    while cur < len(data):
        #print 'On frame %d of %d (%0.1f%%)...   \r' % (cur, len(data), 100.0*cur/len(data)),
        sys.stdout.flush()
        next = cur+int(incr*rate)
        chunk = data[cur:next]
        feats = featfunc(chunk, rate)
        if feats.shape != (49,13):
            print 'hello', len(chunk), feats.shape
        ret.append(feats)
        cur = next
    #print
    if not ret: return None
    ret = np.vstack(ret)
    return ret

def normaudio(data):
    """Normalizes the given audio segment"""
    import numpy as np
    MAX = 16384
    try:
        ratio = MAX/(np.fabs(data).max()+1)
    except Exception:
        print 'Error in norm'
        print data
        print data.shape
        raise

    data *= ratio
    return data

def readwav(fname):
    """Reads a wavefile and returns (data, sampling rate).
    Normalizes if wanted (default: yes)"""
    import scipy.io.wavfile as wav
    import numpy as np
    (rate, data) = wav.read(fname)
    try: # convert to mono
        data = np.mean(data, axis=1)
    except IndexError: pass# already mono
    print 'Read %s with rate %s and %s frames (%0.2f s)' % (fname, rate, data.shape, len(data)/float(rate))
    return (data, rate)

def oldresyncmain():
    """Main driver for subtitle resyncing"""
    from trainutils import SGDSVM, splitTrainEval, evalSVM
    import librosa
    from cPickle import dump, load
    rate = 22050
    if len(sys.argv) < 1:
        print 'Usage: python %s <video or audio file> <subtitle file> <vad model>' % (sys.argv[0])
        sys.exit()
    fname, subfname, modelfname = sys.argv[1:4]
    model = SGDSVM.load(modelfname)
    allfeats = getmel(fname)
    def featfunc(a,b):
        """concats feats from the given times"""
        a, b = int(a*10), int(b*10)
        ret = allfeats[:,a:b].transpose().flatten()
        return ret
    seglen = 0.5
    subels = []
    A, B = 0, 300
    for start in range(A, B):
        feats = featfunc(start, start+seglen)
        cls = SGDSVM().classify(model, [feats])[0]
        subels.append(('Cls: %0.4f' % (cls), start, start+1))
        print start, cls
    sys.stdout.flush()
    #print subels
    playpreview(fname.replace('.mel',''), subels, A, B)


def in_interval(seg, intervals):
    """Checks if the given (start, end) segment overlaps the list of intervals"""
    intervals = sorted(intervals)
    a, b = seg
    for s, t in intervals:
        if s > b: break
        if t < a: continue
        return 1
    return 0

def randseg(start, stop, seglen):
    """Returns a random segment of length seglen between start and stop.
    Raises ValueError if the segment is not long enough.
    """
    from random import uniform
    if stop-start < seglen: raise ValueError
    a = uniform(start, stop-seglen)
    b = a+seglen
    return (a, b)

def getTrainingSegments(dialogs, npos, nneg, seglen=2, negpad=2):
    """Returns training segments of the given length.
    dialogs is the output of getdialogs().
    Returns (pos, neg), where each is a sorted list of (start, end) pairs,
    each of which will be seglen seconds long.
    Returns npos positive and nneg negative segments.
    For negative locations, finds segments which are at least negpad secs
    away from any positive dialog
    """
    from random import choice, uniform
    # functions to randomly sample a positive/negative segment
    def randpos():
        while 1:
            times, txt = choice(dialogs)
            if '[' not in txt: return times # skip non-verbal lines

    def randneg():
        # get the space between two adjacent dialogs
        i = choice(xrange(len(dialogs)-1))
        start = dialogs[i][0][1]
        end = dialogs[i+1][0][0]
        return (start+negpad, end-negpad)

    # accumulate segments
    pos, neg = [], []
    for (lst, func, limit) in [(pos, randpos, npos), (neg, randneg, nneg)]:
        while len(lst) < limit:
            (start, end) = func()
            try:
                a, b = randseg(start, end, seglen)
            except ValueError: continue
            #print start, end, a, b
            if not in_interval((a, b), lst):
                lst.append((a,b))
                lst.sort()
            #print lst
    return pos, neg

def trainvad():
    """Driver to train a Voice Activation Detection (VAD) classifier.
    """
    from trainutils import SGDSVM, splitTrainEval, evalSVM
    import numpy as np
    import librosa
    from cPickle import dump, load
    if len(sys.argv) < 3:
        print 'Usage: python %s <input movie names> <output classifier>' % (sys.argv[0])
        sys.exit()
    posfeats, negfeats = [], []
    rate = 22050
    LogAmp = librosa.util.FeatureExtractor(librosa.logamplitude, ref_power=np.max)
    for i, fname in enumerate(open(sys.argv[1])):
        fname = fname.rstrip('\n')
        subfname = fname.rsplit('.',2)[0] + '.srt'
        try:
            # read subtitles and get training segments
            lines = [l.strip() for l in open(subfname) if l.strip()]
            dialogs = getdialogs(lines)
            pos, neg = getTrainingSegments(dialogs, 100, 600, seglen=0.5)
            print fname, len(dialogs), len(pos), len(neg)
            # read features
            feats = load(open(fname))
            def featfunc(a,b):
                """concats feats from the given times"""
                a, b = int(a*10), int(b*10)
                ret = feats[:,a:b].transpose().flatten()
                return ret

            curpos = [featfunc(a,b) for a, b in pos]
            curneg = [featfunc(a,b) for a, b in neg]
            posfeats.extend(f for f in curpos if f is not None and len(f) > 0 and not np.isnan(f).any())
            negfeats.extend(f for f in curneg if f is not None and len(f) > 0 and not np.isnan(f).any())
            print len(posfeats), len(negfeats)
            if len(posfeats) >= 5000: break
        except IOError:
            continue
    # split into train and eval subsets and then train and run svm
    (trainpos, trainneg), (evalpos, evalneg) = splitTrainEval(posfeats, negfeats, -20)
    svm = SGDSVM()
    t1 = time.time()
    model, score = svm.train(trainpos+trainneg, [1]*len(trainpos)+[-1]*len(trainneg), ncv=0, n_iter=50)
    try:
        score = evalSVM(svm, model, evalpos+evalneg, [1]*len(evalpos)+[-1]*len(evalneg))
    except Exception, e:
        print set(map(lambda x: x.shape, trainpos+trainneg))
        raise
    t2 = time.time()
    print 'Trained model with %d pos, %d neg feats and score %s in %0.2fs. Saving to %s' % (len(posfeats), len(negfeats), score, t2-t1, sys.argv[2])
    print model.scales
    svm.save(model, sys.argv[2])


# To debug, play vlc clips around pos/neg
# Then also save wavs before/after normalizations and play

def getmel(vidfname, rate=22050):
    """Returns the melfeats for the given video filename, doing all necessary preprocessing"""
    from cPickle import dump, load
    fname = vidfname.rstrip('\n')
    melfname = fname+'.mel'
    try:
        feats = load(open(melfname))
    except Exception:
        import librosa
        MS = librosa.util.FeatureExtractor(librosa.feature.melspectrogram, sr=rate, n_fft=2048, n_mels=128, hop_length=rate/10)
        # read audio and extract features
        if fname.endswith('.wav'):
            audiofname = fname
        else:
            dir, base = os.path.split(fname)
            audiofname = os.path.join(dir, '.'+base+'.wav')
        if not os.path.exists(audiofname):
            extractAudio(fname, audiofname+'.tmp')
            os.rename(audiofname+'.tmp', audiofname)
        data, sr = librosa.load(audiofname, sr=rate)
        print fname, rate, data.shape
        feats = MS.transform([data])[0]
        print feats.shape
        dump(feats, open(fname+'.mel', 'wb'), protocol=-1)
    return feats

def extractmels():
    """Extracts mels and saves them to disk with suffix '.mel'"""
    import librosa
    from cPickle import dump, load
    if len(sys.argv) < 2:
        print 'Usage: python %s <input movie names>' % (sys.argv[0])
        sys.exit()

    rate = 22050
    MS = librosa.util.FeatureExtractor(librosa.feature.melspectrogram, sr=rate, n_fft=2048, n_mels=128, hop_length=rate/10)
    for i, fname in enumerate(open(sys.argv[1])):
        fname = fname.rstrip('\n')
        subfname = fname.rsplit('.',1)[0] + '.srt'
        try:
            # read subtitles and get training segments
            lines = [l.strip() for l in open(subfname) if l.strip()]
            feats = getmel(fname)
        except Exception: continue

def hypresyncmain():
    """Main driver for subtitle resyncing, hypothesis-based"""
    import numpy as np
    import matplotlib.pyplot as plt
    from plot import plotfunc
    rate = 22050
    if len(sys.argv) < 2:
        print 'Usage: python %s <video file> <subtitle file> <output offsets>' % (sys.argv[0])
        sys.exit()
    fname, subfname, outfname = sys.argv[1:4]
    featres = 0.1 # number of seconds per feature slice
    # read features and transform them
    feats = getmel(fname).transpose()
    #feats /= np.linalg.norm(feats, ord=1)
    comb = feats.mean(axis=1).transpose()
    #print comb.shape, comb.max(), comb.mean(), comb.min()
    #comb = (comb > 1.5).astype(int)
    #print comb.shape, comb.max(), comb.mean(), comb.min()
    #plt.imshow(np.vstack([comb]*10))
    #plt.show()
    # read dialogs
    dialogs = getdialogs([l.strip() for l in open(subfname) if l.strip()])
    def intimes(t):
        """Returns 1 if the given time is within the times else 0"""
        for (start, end), txt in dialogs:
            if end < t: continue
            if start > t: break
            return 1
        return 0

    labels = np.array([intimes(t*featres) for t in range(len(feats))])
    scorefunc = lambda o: (o*featres, np.dot(comb, np.roll(labels, o)))
    scores = [scorefunc(i) for i in range(-1000, 1000)]
    f = plotfunc(scores, x='Offset (secs)', y='Score', title='Offset-scores for "%s"' % (fname), plotstrs=['b-'], figsize=(8,8))
    plt.savefig(outfname)
    print 'Wrote plot for "%s" to %s' % (fname, outfname)
    f = open(fname+'.scores', 'wb')
    for t, s in scores:
        print >>f, '%s\t%s' % (t, s)
    f.close()

def resyncmain():
    """Resyncs based on precomputed scores.
    Runs a repl to try out various options and then write out file.
    """
    from random import choice, sample
    import matplotlib.pyplot as plt
    from nkpylib.utils import rankedpeaks
    if len(sys.argv) < 2:
        print 'Usage: python %s <video file> <subtitle file>' % (sys.argv[0])
        sys.exit()
    fname, subfname = sys.argv[1:3]
    # read dialogs
    dialogs = getdialogs([l.strip() for l in open(subfname) if l.strip()])
    scores = [tuple(map(float, l.strip().split())) for l in open(fname+'.scores')]
    offs, vals = zip(*scores)
    peaki = rankedpeaks(vals, minorder=10)
    peaks = [scores[i] for i in peaki]
    if 0: # debugging
        print peaki
        print peaks
        plt.plot(offs, vals, 'b-x')
        plt.hold(1)
        peakx, peaky = zip(*peaks)
        plt.plot(peakx, peaky, 'ro')
        plt.show()
        sys.exit()
    exit = 0
    offset = 0.0
    while not exit:
        print '\nCur offset: %0.3f\n' % (offset)
        for i, (off, score) in enumerate(peaks[:20]):
            print '% 2d. Offset %s, score %s' % (i, off, score)
        print 'Options: (c)ustom:, (v)ideo:, (w)rite, (q)uit, #'
        opt = raw_input('> ')
        if not opt: continue
        if opt == 'q':
            exit = 1
            break
        elif opt == 'w':
            outfname = subfname+'.offset_%0.3f' % (offset)
            srtfile(outfname, shiftdialogs(dialogs, offset))
            print 'Wrote to %s' % (outfname)
        elif opt[0] == 'v':
            try:
                reps = int(opt[2:])
            except Exception:
                reps = 1
            toshow = shiftdialogs(sorted(sample(dialogs, reps)), offset)
            for (t0, t1), txt in toshow:
                print t0, t1, txt
                playpreview(fname, toshow, t0, t1, pad=1.0)
        elif opt[0] == 'c':
            offset = float(opt[2:])
        elif opt[0] in '0123456789':
            num = int(opt)
            offset = peaks[num][0]

if __name__ == '__main__':
    #simplemain()
    #textaudiomain(*sys.argv[1:])
    #trainvad()
    resyncmain()
    #hypresyncmain()
    #extractmels()
