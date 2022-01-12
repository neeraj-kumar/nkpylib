"""Various sift utilities

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

import os, sys, time, math, os.path, random
from PIL import Image, ImageDraw, ImageColor
from utils import log, getListAsStr
try:
    import simplejson as json
except:
    import json

#TODO add nice command line options

COLORS = ImageColor.colormap.values()

def existsnonzero(fname):
    """Checks if the given file exists and is non-zero"""
    try:
        if os.stat(fname).st_size > 0: return 1
    except Exception:
        pass
    return 0

def siftfname(imfname, ext=None, dir=None):
    """Returns the sift filename for the given image filename.
    Assumes it's in the same directory, unless you specify a dir.
    Tries all formats, in this order:
        .projected.gz - gzipped projected output
        .projected - projected output
        .sift.gz - gzipped vlfeat output
        .sift - vlfeat output
        .key - Lowe's binary output
    Or you can specify the extension yourself, either as a string, or a list of strings to try.
    Returns a filename, or the empty string if no suitable file found.
    Note that we're not actually checking that the file is actually in the right format.
    """
    siftdir = dir if dir else os.path.dirname(imfname)
    base = os.path.join(siftdir, os.path.basename(imfname).rsplit('.', 1)[0])
    # if they didn't specify an extension, check them all from most projected to least
    if not ext:
        ext = '.projected.gz .projected .sift.gz .sift .key'.split()
    # make list of extensions to check
    exts = [ext] if isinstance(ext, basestring) else ext
    # check each extension
    for ext in exts:
        if ext[0] != '.':
            ext = '.'+ext
        fname = base+ext
        if existsnonzero(fname):
            return fname
    # if we're here, then no valid file was found
    return ''

def grouper(n, iterable, padvalue=None):
    """Taken from Python's itertools recipes.
    >>> list(grouper(3, 'abcdefg', 'x'))
    [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'x', 'x')]"""
    from itertools import izip, chain, repeat
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

class SiftFeat:
    """Keeps track of a single sift descriptor"""
    def __init__(self, x, y, scale, orientation, data):
        """Creates a new sift descriptor from all the relevant information"""
        # x, y, and scale are all to sub-pixel accuracy
        self.x, self.y, self.scale = x, y, scale
        # orientation is in radians from -PI to +PI
        self.orientation = orientation
        # the actual descriptor should all be bytes (0-255)
        self.data = data

    @classmethod
    def fakefeat(cls, val, cache={}, **kw):
        """Makes a fake sift feature for the given val.
        This function is memoized in cache, so you'll always get the same feat for the same input val.
        You can optionally pass in any of x, y, scale, orientation, data.
        Otherwise, they are initialized to:
            x: uniform(0, 100)
            y: uniform(0, 100)
            scale: uniform(0, 10)
            orientation: uniform(0, pi)
            data: randint(0,256)*128
        """
        from random import uniform, randint
        from math import pi
        if val not in cache:
            for varname, maxval in zip('x y scale orientation'.split(), [100, 100, 10, math.pi]):
                if varname not in kw:
                    kw[varname] = uniform(0, maxval)
            if 'data' not in kw:
                kw['data'] = [randint(0,256) for i in range(128)]
            ret = cache[val] = cls(**kw)
        return cache[val]

    @classmethod
    def siftFromFile(cls, f, fmt=None):
        """Creates a list of sift features from a given file or filename or vectors.
        This tries to do the appropriate format detection.
        If f is a string, then we assume it's a filename. We handle:
            '.key' files, as dumped by Lowe's binary
            '.sift' files, as dumped by VLFeat's sift binary
            '.sift.gz' files, compressed versions of .sift files.
            '.projected' files, as dumped by projectindivmain()
            '.projected.gz' files, compressed versions of .projected files
        You can optionally specify the fmt if you know it:
            'lowe'
            'vlfeat'
            'projected'
        If f is a file, assumes it's in Lowe's format.
        Else, assumes it's a pair of (locs, fvecs).
        Returns an empty string or list on error.
        """
        if fmt == 'lowe': return cls.siftFromLowe(f)
        if fmt == 'vlfeat': return cls.siftFromVLFeat(f)
        if fmt == 'projected': return cls.siftFromProjected(f)
        if isinstance(f, basestring):
            # it's a filename
            if f.endswith('.key'): return cls.siftFromLowe(open(f))
            if f.endswith('.sift') or f.endswith('.sift.gz'): return cls.siftFromVLFeat(f)
            if f.endswith('.projected') or f.endswith('.projected.gz'): return cls.siftFromProjected(f)
        elif isinstance(f, file):
            # it's a file itself, so assume it's in lowe's format
            return cls.siftFromLowe(f)
        else:
            # it's a list
            try:
                # see if it's a pair of (locs, fvecs)
                ret = [cls(x,y,s,o,fvec) for (x,y,s,o), fvec in zip(*f)]
            except Exception:
                ret = []
                for el in f:
                    if isinstance(el, cls):
                        # check if it's already a siftfeat
                        ret.append(el)
            return ret
        return []

    @classmethod
    def siftFromLowe(cls, f):
        """Creates a list of sift features from text output from Lowe's original sift binary"""
        if isinstance(f, basestring):
            f = open(f)
        # read the number of points and the length of each descriptor
        num, length = [int(i) for i in f.readline().split()]
        # read the rest of it and transform it appropriately
        all = ''.join(f.readlines()).replace('\n', ' ').split()
        items = grouper(length+4, all)
        # now read each feature
        feats = []
        for item in items:
            # the first four correspond to the metadata for that feature
            y, x, scale, orientation = [float(i) for i in item[:4]]
            # the rest of it corresponds to the actual data for the descriptor
            data = [int(i) for i in item[4:]]
            feats.append(cls(x, y, scale, orientation, data))
        return feats

    @classmethod
    def siftFromVLFeat(cls, f):
        """Creates a list of sift features from text output from VLFeat's sift binary"""
        import gzip
        import numpy as np
        # if the file is actually a filename, open it first
        if isinstance(f, basestring):
            # check for gzipped files (since we often gzip the sift files)
            f = gzip.open(f) if f.endswith('.gz') else open(f)
        # descrips are one-per-line, making it easy to parse
        feats = []
        for l in f:
            els = l.rstrip().split()
            x, y, scale, ori = map(float, els[:4])
            data = np.array(map(int, els[4:]), dtype=np.float32)
            feats.append(cls(x, y, scale, ori, data))
        return feats

    @classmethod
    def siftFromProjected(cls, f):
        """Creates a list of sift features from projected features output.
        This is for "extended mode" output, which includes the locations as well.
        """
        import gzip
        import numpy as np
        # if the file is actually a filename, open it first
        if isinstance(f, basestring):
            # check for gzipped files (since we often gzip the sift files)
            f = gzip.open(f) if f.endswith('.gz') else open(f)
        # descrips are one-per-line, making it easy to parse
        feats = []
        for l in f:
            els = l.rstrip().split()
            x, y, scale, ori = map(float, els[:4])
            fnum = int(els[4])
            #data = np.array(map(int, els[4:]), dtype=np.float32)
            feats.append(cls(x, y, scale, ori, [fnum]))
        return feats

    @classmethod
    def siftsFromVocab(cls, f, fnames=None):
        """Loads projected sift (or other) features from the given file.
        The file can be a filename (and optionally compressed).
        If 'fnames' is None (default), loads all lines of the file.
        Else, fnames can contain a list of:
            strings - assumes each is a filename (first 1st col of file)
            ints - line numbers to read (prunes out 1st col of file)
        Returns a dict mapping filenames to lists of ints.
        """
        import gzip
        if isinstance(f, basestring):
            f = gzip.open(f) if f.endswith('.gz') else open(f)
        tomatch = set(fnames) if fnames else None
        ret = {}
        for i, l in enumerate(f):
            fname, fvec = l.rstrip('\n').split('\t', 1)
            if tomatch and i not in tomatch and fname not in tomatch: continue
            ret[fname] = map(int, fvec.split())
        return ret

    def __repr__(self):
        """Reproducible description (vlsift format)"""
        from utils import getListAsStr
        return '%s %s %s %s %s' % (self.x, self.y, self.scale, self.orientation, getListAsStr(map(int,self.data), ' '))

    def __str_disabled__(self):
        """Returns simple descrip"""
        return "x: %f, y: %f, scale: %f, orientation: %f, start of data: %d %d, length of data: %d" % (self.x, self.y, self.scale, self.orientation, self.data[0], self.data[1], len(self.data))

    def dist(self, other):
        """Returns some measure of distance between the descriptors of self and other.
        Currently this is square of simple euclidean (L2) distance.
        Simple test example with length-3 descriptors:
        >>> a = SiftFeat(1, 2, 3, 4, [0, 40, 0])
        >>> b = SiftFeat(1, 2, 3, 4, [30, 0, 0])
        >>> a.dist(b)
        2500.0
        >>> b.dist(a)
        2500.0
        """
        from math import sqrt
        ret = 0.0
        for s, o in zip(self.data, other.data):
            ret += (s-o)*(s-o)
        return ret

    def findClose(self, seq, **kw):
        """Finds the closest descriptors from self to the given seq.
        The kw are passed to filternnresults (prominently, k and r)
        """
        from utils import simplenn, filternnresults
        import numpy as np
        if not isinstance(seq, np.ndarray):
            seq = np.array([o.data for o in seq])
        #log('  Got seq of type %s, shape %s, and data shape %s and data %s' % (type(seq), seq.shape, type(self.data), self.data))
        # deal with degenerate data (happens with projected inputs)
        if isinstance(self.data, list) or len(self.data.shape) == 0:
            self.data = np.array([self.data])
        if len(seq.shape) == 1:
            seq = seq.reshape((len(seq), 1))
        #log('  Got seq of type %s, shape %s, and data shape %s' % (type(seq), seq.shape, self.data.shape))
        #raise NotImplementedError
        dists = simplenn(seq, self.data)
        kw.setdefault('sort', 1)
        dists = filternnresults(dists, **kw)
        return dists

    def drawOnImage(self, im, color="white", minsize=1, drawline=1, fill=0):
        """Draws this descriptor onto the given image (destructive)"""
        draw = ImageDraw.Draw(im)
        # draw a circle to represent this descriptor
        s = max(self.scale*1, 1) # *6 for equivalent to sift binary
        if s < minsize: return
        bbox = (self.x-s, self.y-s, self.x+s, self.y+s)
        bbox = [int(i) for i in bbox]
        if fill:
            draw.ellipse(bbox, fill=color)
        else:
            draw.arc(bbox, 0, 360, fill=color)
        # draw a line to show what direction it's in
        if not drawline: return
        s = max(s, 6)
        dir = [s*i for i in [math.cos(self.orientation), -math.sin(self.orientation)]]
        draw.line([self.x, self.y, self.x+dir[0], self.y+dir[1]], fill=color)
        return im

    def getImage(self):
        """Returns an image which represents the descriptor"""
        im = Image.new('RGB', (19, 11), (0, 0, 0))
        # each group of 8 in the descriptor corresponds to an orientation histogram
        vals = list(grouper(8, self.data))
        cur = 0

        # go through each cell and use its histogram to draw the image
        for yloc in range(4):
            y = yloc * 3
            for xloc in range(4):
                x = xloc * 5
                # the first row of this cell contains the first 4 orientation bins
                for i, v in enumerate(vals[cur][:4]):
                    color = (v, 0, 255-v)
                    #print "Putting bin %d at %d, %d" % (i+cur*8, x+i, y)
                    im.putpixel((x+i, y), color)
                # the second row contains the latter 4 orientation bins
                for i, v in enumerate(vals[cur][4:]):
                    color = (v, 0, 255-v)
                    #print "Putting bin %d at %d, %d" % (4+i+cur*8, x+i, y)
                    im.putpixel((x+i, y+1), color)
                cur += 1
        return im

    def getDiffImage(self, other):
        """Returns an image which represents the difference between two descriptors"""
        # get the two images and then take their difference
        from PIL import ImageChops
        im1, im2 = self.getImage(), other.getImage()
        ret = ImageChops.difference(im1, im2)
        return ret

    def getDiffGraph(self, other):
        """Returns a graph which is the difference between two descriptors"""
        w, h = 512, 64
        im = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        vals1 = [int(i) for i in self.data]
        vals2 = [int(i) for i in other.data]
        difs = [i-j for i, j in zip(vals1, vals2)]
        # draw the baseline
        draw.line((0, h/2, w-1, h/2), fill="black")
        lines = []
        for i, v in enumerate(difs):
            x = i*4
            y = h - 1 - int(v*h/512) - h/2
            lines.append((x, y))
        # draw them all
        #print "Drawing %s for %s" % (lines, difs)
        draw.line(lines, fill="black")
        return im

    def getGraph(self):
        """Returns a graph which represents the descriptor"""
        w, h = 24, 16
        im = Image.new('RGB', (w*4, (h+1)*4), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        # each group of 8 in the descriptor corresponds to an orientation histogram
        vals = list(grouper(8, self.data))
        cur = 0

        # go through each cell and draw its histogram
        for yloc in range(4):
            y = (yloc+1) * (h+1) - 1
            for xloc in range(4):
                x = xloc * w
                # collect the points in this histogram
                lines = []
                for i, v in enumerate(vals[cur]):
                    curx = 3*i + 1 + x
                    cury = y - int(v*h/256)
                    lines.append((curx, cury))
                # draw them all
                #print "Drawing %s for %s" % (lines, vals[cur])
                draw.line(lines, fill="black")
                cur += 1
        return im

    def getOverallGraph(self):
        """Returns a single graph which represents the descriptor"""
        w, h = 512, 32
        im = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        vals = [int(i) for i in self.data]
        lines = []
        for i, v in enumerate(vals):
            x = i*4
            y = h - 1 - int(v*h/256)
            lines.append((x, y))
        # draw them all
        #print "Drawing %s for %s" % (lines, vals)
        draw.line(lines, fill="black")
        return im


class Path:
    """A path is a simple struct that takes into account a point appearing and moving"""
    def __init__(self, imagenum, descrip):
        """Creates a path that starts with the given image number and index within that image"""
        self.imagenums = [imagenum]
        self.descrips = [descrip]
        self.active = 1
        self.color = random.choice(COLORS)

    def findNext(self, descrips, imagenum):
        """Finds the closest descriptor from the given ones"""
        # if we're inactive, return
        if not self.active: return -1
        last = self.descrips[-1]
        close = last.findClose(descrips)
        # if the first one is very far away, assume it's gone
        n = close[0][1]
        dist = math.sqrt((last.x-descrips[n].x)**2 + (last.y-descrips[n].y)**2)
        if dist > 10:
            self.active = 0
            return -1
        # otherwise, assume it's a good match
        self.imagenums.append(imagenum)
        self.descrips.append(descrips[n])
        return n

    def __repr__(self):
        """Returns the list of imagenums for this path"""
        return "%s" % (self.imagenums)

def mainloop(path):
    """Runs the mainloop"""
    # read all the sift files
    fnames = [f for f in os.listdir(path) if f.endswith('.key')]
    fnames.sort()
    sifts = []
    print "Reading sift descriptors"
    for fname in fnames:
        sifts.append(SiftFeat.siftFromFile(open("%s%s" % (path, fname))))

    print "Read %d key files. Initializing paths" % (len(sifts))
    paths = []
    for n, descrip in enumerate(sifts[0]):
        paths.append(Path(0, descrip))

    # now go through and trace paths by following points and seeing where similar descriptors move to
    for imagenum, s in enumerate(sifts[1:]):
        print "On image %d with %d descrips" % (imagenum+1, len(s))
        cur = s[:]
        # first check all paths
        found = 0
        for n, p in enumerate(paths):
            index = p.findNext(cur, imagenum+1)
            if index >= 0:
                # delete this index from cur
                cur.pop(index)
                found += 1
                print "  Found %d out of %d paths searched\r" % (found, n),
        # now that we have removed all points that correspond to existing paths, add the rest as new paths
        for n, descrip in enumerate(cur):
            paths.append(Path(imagenum+1, descrip))

    # now print out the paths
    print "Paths:"
    for p in paths:
        print p

def findPaths():
    """Finds coherent paths"""
    path = './'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    mainloop(path)

def testDraw(siftPath, imgPath, outPath="./"):
    """Tests out the drawing of descriptors"""
    siftFnames = [f for f in os.listdir(siftPath) if f.endswith('.key')]
    basenames = [f.rsplit('.')[-2] for f in siftFnames]
    imgFnames = ["%s%s.png" % (imgPath, f) for f in basenames]
    for siftname, imgname, basename in zip(siftFnames, imgFnames, basenames):
        descrips = SiftFeat.siftFromFile(open("%s%s" % (siftPath, siftname)))
        print "Opened %s with %d descriptors" % (siftname, len(descrips))
        im = Image.open(imgname).convert('RGB')
        for i, d in enumerate(descrips):
            c = "white" #COLORS[i%len(COLORS)]
            d.drawOnImage(im, c, minsize=3, drawline=0)
        im.save('%s%s_out.png' % (outPath, basename))

def extract(fnames, checkexist=1, compress=1):
    """Extracts sift for the given filenames.
    The outputs are stored to {filename without ext}.sift if compress is 0,
    or with .gz appended if compress is 1.
    Yields list of output filenames, in same order as input.
    If checkexist is true (default), then first sees if the output exists and is non-zero size."""
    from subprocess import Popen, PIPE
    vlsiftexe = ['vlsift', '-o', '/dev/stdout', '/dev/stdin']
    gzipexe = ['gzip', '-c']
    for i, f in enumerate(fnames):
        log('On input %d of %d: %s' % (i+1, len(fnames), f))
        t1 = time.time()
        outfname = f.rsplit('.',1)[0]+'.sift'
        if compress:
            outfname += '.gz'
        done = 0
        if checkexist:
            if existsnonzero(outfname):
                done = 1
                log('  Outfile %s already existed...' % (outfname))
        if not done:
            outf = open(outfname, 'wb')
            p1 = Popen(['convert', f, 'pgm:-'], stdout=PIPE)
            if compress:
                p2 = Popen(vlsiftexe, stdin=p1.stdout, stdout=PIPE)
                p3 = Popen(gzipexe, stdin=p2.stdout, stdout=outf)
                p1.stdout.close()
                p2.stdout.close()
                p3.communicate()
            else:
                p2 = Popen(vlsiftexe, stdin=p1.stdout, stdout=outf)
                p1.stdout.close()
                p2.communicate()
            outf.close()
            log('  Took %0.3fs to extract sift feats to %s...' % (time.time()-t1, outfname))
        yield outfname

def extractmain(*fnames):
    """A driver that extracts sift matches for filenames given in args.
    This calls extract(fnames, checkexist=1, compress=1)
    Prints sift output filenames on separate lines, one per input
    """
    for outfname in extract(fnames, checkexist=1, compress=1):
        print outfname
        sys.stdout.flush()

def oldmain():
    #findPaths()
    if len(sys.argv) < 3:
        print "Usage: python %s <sift files path> <image files path>" % (sys.argv[0])
        sys.exit()
    if sys.argv[1][-1] != '/': sys.argv[1] += '/'
    if sys.argv[2][-1] != '/': sys.argv[2] += '/'
    testDraw(sys.argv[1], sys.argv[2])

def drawmain(*args):
    """A driver that draws sift points onto images"""
    minsize = 2
    if len(args) < 3:
        print 'Draw args: <vlsift output filename> <img> <out imgfname> [<minsize=%d>]' % (minsize)
        sys.exit()
    featfname, imgfname, outfname= args[0:3]
    try:
        minsize = int(args[3])
    except Exception: pass
    im = Image.open(imgfname)
    feats = SiftFeat.siftFromVLFeat(featfname)
    # generate color
    from imageutils import colormap, indexedcolor
    for i, f in enumerate(feats):
        c = 'white'
        #c = colormap(i/float(len(feats)))
        c = indexedcolor(i%100, 100)
        f.drawOnImage(im, minsize=minsize, color=c)
    im.save(outfname)

def oldmatchmain(*args):
    """A driver that finds matches between two sift files"""
    import numpy as np
    maxratio = 0.8
    if len(args) < 4:
        print 'Match args: <vlsift outfile 1> <vlsift outfile 2> <outsift1> <outsift2> [<maxratio=%f>]' % (maxratio)
        sys.exit()
    func = SiftFeat.siftFromVLFeat
    f1, f2 = map(func, args[0:2])
    print >>sys.stderr,  'Read %d feats in file %s, %d feats in file %s' % (len(f1), args[0], len(f2), args[1])
    matches = []
    f2data = np.array([f.data for f in f2])
    outf1, outf2 = open(args[2], 'wb'), open(args[3], 'wb')
    for i, f in enumerate(f1):
        if i % 1 == 0:
            print >>sys.stderr, '  On point %d of %d, %d matches so far %s...  \r' % (i+1, len(f1), len(matches), matches[-1] if matches else []),
        d = f.findClose(f2data, k=2)
        #print d
        if len(d) < 2: continue
        ratio = d[0][0]/d[1][0]
        if ratio > maxratio: continue
        j = d[0][1]
        matches.append((i, j))
        print >>outf1, f
        print >>outf2, f2[j]
        outf1.flush()
        outf2.flush()
    print >>sys.stderr

def match(f1, f2, outfname=None, maxratio=0.8, maxk=5, maxr=None, checkexist=1, *args, **kw):
    """Matches features between two lists of feature vectors and optionally saves to outfname.
    The inputs can be in any parseable format.
    For each pt in f1, finds closest points in f2.
    The outputs are written in tab-separated lines with the following fields:
        distance - normalized distance between fvecs
        feat number of match in 1st input
        feat number of match in 2nd input
        full line of 1st input (tabs converted to spaces)
        full line of 2nd input (tabs converted to spaces)
    Parameters:
        maxk: maximum number of neighbors to retrieve
        maxr: maximum distance to allow (set to 0 if matching projected points)
        maxratio: if >0, then applies the ratio test
        checkexist: if 1, then if the output file exists, doesn't do anything.
    """
    import numpy as np
    if isinstance(f1, basestring) and isinstance(f2, basestring):
        log('Matching features in %s vs %s to %s with maxratio %s, maxk %s, maxr %s' % (f1, f2, outfname, maxratio,  maxk, maxr))
    if checkexist and existsnonzero(outfname):
        log('  Outfile %s existed already, so returning' % (outfname))
        return list(readmatches(outfname))
    feats1 = SiftFeat.siftFromFile(f1)
    feats2 = SiftFeat.siftFromFile(f2)
    #log('%s (%d) vs %s (%d)' % (f1, len(feats1), f2, len(feats2)))
    other = np.array([o.data for o in feats2])
    matches = []
    outf = open(outfname, 'wb') if outfname else None
    for i, f1 in enumerate(feats1):
        dists = f1.findClose(other, k=maxk, r=maxr)
        if not dists: continue
        best = dists[0][0]
        matchratio = 0 if maxratio and maxratio > 0 else 1
        if best == 0:
            matchratio = 1
        # for ratio test, see if we have any
        nonzero = [d for d, j in dists[1:] if d > 0]
        if not matchratio and maxratio and maxratio > 0 and nonzero:
            ratio = best/float(nonzero[0])
            if ratio < maxratio:
                matchratio = 1
        toadd = []
        for mnum, (d, j) in enumerate(dists):
            if d == 0 or (matchratio and d == best):
                # if it has 0 distance, just add it
                # if we're matching ratio, then we add all results with the best score.
                toadd.append((d, i, j, f1, feats2[j]))
            else:
                break
        matches.extend(toadd)
        if outf:
            for m in toadd:
                print >>outf, getListAsStr(m, '\t')
        #log('  On %d of %d, with data %s, %s best, %d dists, %d toadd, %d matches: %s' % (i+1, len(feats1), f1.data[:2], best, len(dists), len(toadd), len(matches), [(d, j, f2.data[:2]) for d, i, j, f1, f2 in toadd[:3]]))
    return matches

def matchmain(f1, f2, outfname, *args):
    """Driver for match().
    Checks filenames. Asserts both are of same type.
    If projected, then runs with k=5 (or int(args[0])), maxr=0, maxratio=None.
    Else, then runs with maxratio=0.8 (or float(args[0])), k=5.
    """
    p = '.projected'
    proj = p in f1
    if proj:
        assert p in f2
    else:
        assert p not in f2
    if proj:
        try:
            k = int(args[0])
        except:
            k = 5
        return match(f1, f2, outfname, maxratio=None, maxk=k, maxr=0)
    else:
        try:
            maxratio = float(args[0])
        except:
            maxratio = 0.8
        return match(f1, f2, outfname, maxratio=maxratio, maxk=5, maxr=None)

def readmatches(f):
    """Parses the given match file and yields (dist, i, j, f1, f2)"""
    from gzip import GzipFile
    f = GzipFile(f) if f.endswith('.gz') else open(f)
    def makefeat(feat):
        els = feat.split()
        x, y, s, o = map(float, els[:4])
        data = map(int, els[4:])
        return SiftFeat(x, y, s, o, data)

    for l in f:
        d, i, j, f1, f2 = l.rstrip('\n').split('\t')
        d = float(d)
        i, j = int(i), int(j)
        f1 = makefeat(f1)
        f2 = makefeat(f2)
        yield (d, i, j, f1, f2)

def drawmatchesmain(matchesfname, outfname, im1fname, im2fname, minscale=0, orientation='vertical', pointsize=3):
    """Draws matches between images.
    Parameters:
        matchesfname - text file with matches (as output from match())
        outfname - output image filename
        im1fname - 1st image of matchpair
        im2fname - 2nd image of matchpair
        minscale - the minimum "scale" of points to draw matches for
        orientation - 'horizontal' or 'vertical' (default)
        pointsize - if > 0, then draws circles on each sift point at this radius.
    """
    from PIL import Image, ImageDraw
    from imageutils import ImagePair, randcolor
    from utils import rectat
    im1 = Image.open(im1fname)
    im2 = Image.open(im2fname)
    ip = ImagePair(im1, im2, orientation=orientation, background=(255,255,255))
    minscale = float(minscale)
    draw = ImageDraw.Draw(ip.outim)
    pairs = []
    # first collect pairs, and optionally draw points
    for d, i, j, f1, f2 in readmatches(matchesfname):
        if f1.scale < minscale and f2.scale < minscale: continue
        loc1 = (0, f1.x, f1.y)
        loc2 = (1, f2.x, f2.y)
        color = randcolor('RGB')
        pairs.append((loc1, loc2, color))
        if pointsize > 0:
            p1 = ip.globalcoords(0, (f1.x, f1.y))
            p2 = ip.globalcoords(1, (f2.x, f2.y))
            draw.ellipse(rectat(p1[0], p1[1], pointsize*2, pointsize*2), outline=color, fill=color)
            draw.ellipse(rectat(p2[0], p2[1], pointsize*2, pointsize*2), outline=color, fill=color)
    # draw lines
    for loc1, loc2, color in pairs:
        ip.drawline([loc1, loc2], fill=color)
    ip.outim.save(outfname)

def ransacmain(matchesfname, outfname, minerror=0.5, maxiters=5000):
    """Runs ransac on the given set of matches.
    Writes the filtered matches out to the given outfname.
    Also prints (to stdout) the homography.
    Returns [list of inlier matches, dict of best model].
    The best model dict has fields:
        inliers: list of inliers, as simple indices
        model: the best HomographyModel
        error: the error (measured as percentage of inliers)
        finalerror: the final error (as reprojection error on only the inliers)
    """
    from ransac import Ransac, HomographyModel
    minerror = float(minerror)
    matches = list(readmatches(matchesfname))
    data = [((f1.x, f1.y), (f2.x, f2.y)) for d, i, j, f1, f2 in matches]
    dataidx = dict((d, i) for i, d in enumerate(data))
    def callback(niters, elapsed, best, ransac, **kw):
        """Generic callback"""
        perc = len(best['inliers'])/float(len(dataidx))
        perc = min(perc, 0.8)
        ransac.maxiters = min(maxiters, Ransac.estimateNiters(4, perc))
        print >>sys.stderr, 'On iter %d (%0.3fs elapsed), best model has %d inliers (%0.3f%%), error %0.2f, new maxiters %d' % (niters, elapsed, len(best['inliers']), perc, best['error'], ransac.maxiters)

    r = Ransac(4, HomographyModel, callback=callback, strongthresh=10, weakthresh=30, mininliers=4, minerror=minerror)
    try:
        ret = r.run(data)
    except KeyboardInterrupt:
        ret = r.best
    # also compute final error
    ret['finalerror'] = ret['model'].geterror(ret['inliers'])
    outf = open(outfname, 'wb')
    toret = [[], ret]
    for pair in ret['inliers']:
        idx = dataidx[pair]
        m = matches[idx]
        toret[0].append(m)
        print >>outf, getListAsStr(m, '\t')
    print 'Error (1-inlier perc): %s, reprojection error: %s, model:\n%s' % (ret['error'], ret['finalerror'], ret['model'])
    return toret

def fullsiftmain(im1, im2, vocabname=None, minscale=0, orientation='vertical', minerror=0.5):
    """Does the full sift pipeline for the given two images.
    - Extracts sift features for both
    - Computes matches, storing the output to '%(im1)s-%(im2)s-matches.txt'
    - Draws matches to same fname, but as .png (using minscale)
    - Runs ransac on the matches, storing to ...-rmatches.txt
    - Draws ransac matches, to same fname but as .png (using minscale)
    """
    minscale = float(minscale)
    minerror = float(minerror)
    siftfnames = list(extract([im1, im2]))
    if vocabname:
        # project using given vocab
        outfnames = projectindiv(vocabname, siftfnames)
        siftfnames = outfnames
    base = lambda f: os.path.basename(f).rsplit('.',1)[0]
    matchbase = '%s-%s-matches' % (base(im1), base(im2))
    matchmain(siftfnames[0], siftfnames[1], matchbase+'.txt')
    drawmatchesmain(matchbase+'.txt', matchbase+'.png', im1, im2, minscale=minscale, orientation=orientation)
    rbase = matchbase.replace('-matches', '-rmatches')
    inliermatches, best = ransacmain(matchbase+'.txt', rbase+'.txt', minerror=minerror)
    print 'Best error is %s, %d inliers, finalerror %s, model is\n%s' % (best['error'], len(best['inliers']), best['finalerror'], best['model'])
    drawmatchesmain(rbase+'.txt', rbase+'.png', im1, im2, minscale=minscale, orientation=orientation)


def match1stmain(userims, googims):
    """Matches userims to google images and returns scores for each userim"""
    user = [(f, locs, fvecs) for i, f, locs, fvecs in readfvecs(userims)]
    ret = []
    for i, f, locs, fvecs in readfvecs(googims):
        if fvecs is None or len(fvecs) == 0: continue
        for j, (userfname, userlocs, userfvecs) in enumerate(user):
            matches = match((locs, fvecs), (userlocs, userfvecs), 'testmatches', maxratio=None, maxk=5, maxr=0)
            filtmatches, best = ransacmain('testmatches', 'outransac')
            print 'Went from %d matches to %d filtered' % (len(matches), len(filtmatches))
            sys.exit()


def readfvecs(fnames, ss=0, nprocs=1, minsize=0):
    """Reads filenames from the given input.
    If it's a string, then assumes it's a filename which lists descriptor filenames.
    Else, should be a list of descriptor filenames.
    The descriptors are read, and can be:
        filename - read using GzipFile if ends with .gz, else read directly
        list of fvecs
    These are converted to an np.array, with dtype=uint8.
    Does subsampling of feature vectors based on ss:
        <= 0: no subsampling
        int: maximum number of fvecs per input
    You can also pass in:
        minsize: if > 0, descriptors with 'scale' less than this are discarded (only if we know scales)

    The method yields (i, f, locs, fvecs) for each input descriptor file.
    Locs is a list of (x, y, scale, orientation) float tuples.
    This corresponds to the fvecs, which are np.array(dtype=np.uint8)
    """
    from gzip import GzipFile
    from Queue import Queue
    import numpy as np
    from threadutils import spawnWorkers
    from utils import minsample
    import cPickle as pickle
    if isinstance(fnames, basestring):
        fnames = [f.rstrip('\n') for f in open(fnames) if f.rstrip('\n')]
    inq, outq = Queue(), Queue(5)
    ss = int(ss)

    def inqloop():
        last = None
        while 1:
            i, f = inq.get()
            if isinstance(f, basestring):
                # it's a filename, so read it
                ident = f
                try:
                    # pickle files are already fully processed, so just continue after reading it
                    if f.endswith('.pickle'):
                        locs, fvecs, allfnames = pickle.load(open(f))
                        # filter by size
                        indices = [i for i, (x, y, s, o) in enumerate(locs) if not (0 < s < minsize)]
                        # subsample as well
                        if 0 < ss < len(indices):
                            indices = minsample(indices, ss)
                        fvecs = fvecs[indices]
                        locs = locs[indices]
                        outq.put((i, ident, locs, fvecs))
                        continue
                    curf = GzipFile(f) if f.endswith('.gz') else open(f)
                    fvecs = []
                    try:
                        for l in curf:
                            try:
                                row = l.strip().split()
                                x,y,s,o = loc = map(float, row[:4])
                                if 0 < s < minsize: continue # skip small points
                                fvec = map(int, row[4:])
                                fvecs.append((loc,fvec))
                            except Exception:
                                pass
                    except Exception: # this is needed for gzip iteration, which sometimes causes a problem here
                        pass
                    log('  Loaded file %d: %s with len %d (ss %s, minsize %s)' % (i, f, len(fvecs), ss, minsize))
                except Exception, e:
                    raise
                    fvecs = last
            else:
                # it must a list of fvecs already -- see if it has locs
                rows = []
                for row in fvecs:
                    if len(row) == 2: # (loc, fvec)
                        (x,y,s,o), fvec = row
                        if 0 < s < minsize: continue # skip small points
                        rows.append(row)
                    else: # fvec only
                        rows.append((0,0,0,0), row)
                fvecs = rows
                ident = i
            last = fvecs
            # subsample
            if ss > 0:
                fvecs = minsample(fvecs, ss)
            # make nparray
            fvecs = [(loc, f) for loc, f in fvecs if len(f) == 128 or len(f) == 1]
            if fvecs:
                locs, fvecs = zip(*fvecs)
                if len(fvecs[0]) == 1:
                    fvecs = np.array(fvecs, dtype=np.uint32)
                else:
                    fvecs = np.array(fvecs, dtype=np.uint8)
            else:
                locs = []
                fvecs = np.array((), dtype=np.uint8)
            # put it on the output queue
            outq.put((i, ident, locs, fvecs))

    # spawn workers
    inworkers = spawnWorkers(nprocs, inqloop)
    # send inputs to inqloop, in a separate thread, so that we don't block outputs
    def sendinputs():
        for i, f in enumerate(fnames):
            inq.put((i, f))

    sendinputworker = spawnWorkers(1, sendinputs)
    # read outputs
    t1 = time.time()
    ntotal = 0
    ndone = 0
    while 1:
        i, f, locs, fvecs = outq.get()
        ntotal += len(fvecs)
        ndone += 1
        log('  Loaded %d fvecs (%d total) from input %d of %d (%0.2fs elapsed)' % (len(fvecs), ntotal, ndone, len(fnames), time.time()-t1))
        yield (i, f, locs, fvecs)
        del f, locs, fvecs
        if ndone >= len(fnames): break

def aggregatemain(outfname, ss, minsize, *fnames):
    """Aggregates sift features from many files into one pickled file.
    This file contains (alllocs, allfvecs, allfnames), which are:
        alllocs: a Nx4 numpy array of x,y,scale,orientation values for each point
        allfvecs: a Nx128 numpy array of descriptors for each point
        allfnames: a list of len N with the filename for each point
    """
    import cPickle as pickle
    import numpy as np
    alllocs = []
    allfvecs = []
    t1 = time.time()
    ss = int(ss)
    minsize = float(minsize)
    allfnames = []
    for i, f, locs, fvecs in readfvecs(fnames, ss=ss, minsize=minsize):
        alllocs.append(np.array(locs))
        allfvecs.append(np.array(fvecs))
        allfnames.extend([f]*len(fvecs))
    t2 = time.time()
    alllocs = np.vstack(alllocs)
    allfvecs = np.vstack(allfvecs)
    t3 = time.time()
    log('Got final matrices of shape %s (%s), %s (%s) in %0.3fs' % (alllocs.shape, alllocs.dtype, allfvecs.shape, allfvecs.dtype, t3-t1))
    pickle.dump((alllocs,allfvecs,allfnames), open(outfname, 'wb'), -1)

def applyrootsift(fvecs):
    """Applies the root sift transformation to a given set of feature vectors.
    This is L1 normalization followed sqrt.
    """
    import numpy as np
    fvecs = fvecs.astype(np.float32)
    norms = np.sum(fvecs, axis=1).reshape((1, len(fvecs)))
    fvecs = np.sqrt(fvecs/norms.transpose())
    return fvecs

def kmcombosmain(siftdir=''):
    """Generates commandlines for k-means combinations"""
    from math import ceil
    def num2str(k):
        """Converts a k value into a k-string"""
        units = [(1000000, 'm'), (1000, 'k')]
        for val, unit in units:
            if k >= val:
                return '%d%s' % (k//val, unit)
        return str(k)

    print '#!/bin/bash\n'

    for k in [10000, 100000, 500000, 1000000]:
        for npickles in [1, 10, 25, 50]:
            for ss in [100000, 500000]:
                if npickles*ss/1.1 < k: continue # skip combos with nsamples ~ k
                for rootsift in [1, 0]:
                    kstr = num2str(k)
                    mem = ss*128*4*npickles*8
                    memgb = mem/1024.0/1024/1024
                    if memgb > 60: continue # skip combinations which are very high memory
                    memgbstr = '%0.2f GB' % (memgb)
                    outname = 'siftvocab-%(kstr)s-%(npickles)s-%(ss)s-%(rootsift)s.pickle' % (locals())
                    exe = 'python'
                    memneeded = int(ceil(memgb))
                    exe = 'qsub -N %(outname)s -cwd -o outs/ -e errs/ -l mem_free=%(memneeded)sG ~/pysub.sh' % (locals())
                    cmdline = '%(exe)s ~/pylib/sift.py makevocab ~/db/siftvecs/siftvecs-%(npickles)s.lst %(outname)s %(k)s %(ss)s %(rootsift)s' % (locals())
                    print cmdline


def makevocab(fnames, outfname, k=10000, ss=0, rootsift=0, n_jobs=4, max_iter=100, normalkmeans=0, **kw):
    """Runs kmeans using the list of filenames given, and given k.
    Parameters:
        k - number of clusters
        ss - maximum number of descriptors to take from each input file
        rootsift - if 1, then computes centers on RootSift descriptors
                   this is done by L1-norming the vector, then taking the sqrt of each element.
        n_jobs - number of simultaneous cpus to use (only for normal k-means)
        max_iter - number of iterations to run within k-means
        normalkmeans - if true, uses normal k-means; else uses mini-batch k-means
    All other kw are passed to the KMeans initializer.
        batch_size is an important parameter for minibatch k-means
    Vocab created April 24, 2013:
        python sift.py kmeans <(shuf -n 5000 allsift.txt) siftvocab-10k-5000-500.pickle 10000 500
    Vocabs created May 7, 2013:
        Combinations (using sift fvec pickles, created from 1000 files * 500 vecs each):
            K = 10k,100k,500k,1m
            ss = 0 (use entire pickle), 100000 (~100 fvecs per image)
            npickles = 1, 10, 25, 50
            rootsift = 0, 1
        Fname fmt:
            siftvocab-%(K)-%(npickles)-%(ss)-%(rootsift).pickle
        Cmd line usage:
            python sift.py kmeans ~/db/picasa/shufsift-%(nfiles).txt) %(outfile) %(K) %(ss) %(rootsift)
    Rough guide to memory usage: x million descriptors ~ x gigs of ram
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    import cPickle as pickle
    import numpy as np
    from utils import specializeDict
    m = []
    rootsift = int(rootsift)
    t1 = time.time()
    for i, f, locs, fvecs in readfvecs(fnames, ss=ss):
        if rootsift:
            fvecs = applyrootsift(fvecs)
        m.append(np.array(fvecs))
    t2 = time.time()
    m = np.vstack(m)
    t3 = time.time()
    log('Got final matrix of shape %s, dtype %s in %0.3fs' % (m.shape, m.dtype, t3-t1))
    # now run kmeans
    if normalkmeans:
        # normal k-means
        defkw = dict(precompute_distances=0, verbose=3, n_jobs=n_jobs, max_iter=max_iter, copy_x=0)
    else:
        # mini-batch
        defkw = dict(compute_labels=0, verbose=3, n_jobs=n_jobs, max_iter=max_iter)
        defkw['batch_size'] = max(1000, int(0.002*len(m)))
    defkw.update(kw)
    defkw = specializeDict(defkw)
    if not normalkmeans:
        try:
            del defkw['n_jobs']
        except Exception: pass
    log('Running kmeans with k %s and kw %s' % (k, defkw))
    kmfunc = KMeans if normalkmeans else MiniBatchKMeans
    km = kmfunc(int(k), **defkw)
    km.fit(m)
    km.rootsift = rootsift
    t4 = time.time()
    log('KMeans finished, taking %0.3fs (%0.3fs total)' % (t4-t3, t4-t1))
    pickle.dump(km, open(outfname, 'wb'), -1)
    t5 = time.time()
    log('Dumped KMeans to file %s. Total time %0.3fs' % (outfname, t5-t1))

makevocabmain = makevocab

def project(model, fnames, **kw):
    """Projects sift descriptors in 'fnames' using the 'model'.
    The model can either be the model itself, or a pickled filename.
    KW args are passed onto readfvecs.
    If the model contains a field 'rootsift' which is true, then
    applies the rootsift operator to all feature vectors.
    Yields (i, fname, locs, projected values)
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    import cPickle as pickle
    if isinstance(model, basestring):
        model = pickle.load(open(model))
    rootsift = ('rootsift' in dir(model) and model.rootsift)
    transfunc = applyrootsift if rootsift else lambda fvecs: fvecs
    incr = 1000
    for i, f, locs, fvecs in readfvecs(fnames, **kw):
        log('  In main loop with %s, %s, %d' % (i, f,  len(fvecs)))
        try:
            # generate outputs incrementally (to avoid soaking up all memory)
            y = []
            t1 = time.time()
            for start in range(0, len(fvecs), incr):
                if start >= len(fvecs): break
                # get the current batch of feature vectors, applying any transformations (such as rootsift)
                curfvecs = transfunc(fvecs[start:start+incr])
                cur = model.predict(curfvecs)
                y.extend(cur)
                log('    Projected %d vecs from %d, %d total pts so far in %0.3fs' % (len(cur), start, len(y), time.time()-t1))
            del fvecs
            yield (i, f, locs, y)
        except Exception, e:
            log('  Error %s on %d, %s (%s): %s' % (type(e), i, f, type(fvecs), e))

def projectindiv(model, fnames, checkexist=1, compress=5, minscale=0, cachedir=None, returnfvecs=0, printout=1, **kw):
    """Projects sift descriptors in 'fnames' using the model from 'model'.
    Writes individual output files with extension '.projected', including pos/scale/orientation.
    If printout, Prints the outfname for each processed input (not all inputs).
    If checkexist is true, skips existing outfiles.
    If compress is >0, then compresses using gzip with given compression level.
    In this case, outfilenames have '.gz' appended to them.
    Else, no compression is applied.
    If returnfvecs is true, then returns list of (i, f, locs, projectedvals).
    Else, returns a list of all output names.
    """
    from utils import getListAsStr
    from gzip import GzipFile
    from itertools import izip
    import numpy as np
    todo = []
    ret = []
    for fname in fnames:
        outdir = cachedir if cachedir else os.path.dirname(fname)
        outfname = os.path.join(outdir, os.path.basename(fname).rsplit('.sift', 1)[0]+'.projected')
        if compress > 0:
            outfname += '.gz'
        try:
            os.makedirs(os.path.dirname(outfname))
        except OSError: pass
        if checkexist and existsnonzero(outfname):
            log('  Outfile %s already existed...' % (outfname))
            if returnfvecs:
                ret.append(list(readfvecs([outfname]))[0])
            else:
                ret.append(outfname)
            continue
        todo.append((fname, outfname))
    if not todo: return ret
    fnames, outfnames = zip(*todo)
    for (i, f, locs, y), outfname in izip(project(model, fnames, **kw), outfnames):
        outf = GzipFile(outfname, 'wb', compresslevel=compress) if compress > 0 else open(outfname, 'wb')
        if locs:
            for loc, val in zip(locs, y):
                xloc, yloc, scale, ori = loc
                if scale < minscale: continue
                print >>outf, '%s\t%s' % (getListAsStr(loc, sep='\t'), val)
        outf.close()
        if printout:
            print outfname
            sys.stdout.flush()
        if returnfvecs:
            # convert y into a 2d numpy array, which is what we expect to use
            if isinstance(y, list):
                y = np.array(y, dtype=np.uint32).reshape((len(y), 1))
            ret.append((i, f, locs, y))
        else:
            ret.append(outfname)
    return ret

def projectindivmain(model, minsize, *fnames):
    """Driver for projectindiv().
    This calls projectindiv(model, fnames, checkexist=1, compress=3)
    Writes individual output files with extension '.projected.gz', including pos/scale/orientation.
    """
    minsize = float(minsize)
    return projectindiv(model, fnames, checkexist=1, compress=3, minsize=minsize)

def projectallmain(model, *fnames):
    """Projects sift descriptors in 'fnames' using the model from 'model'.
    Prints <input filename> <list of cluster ids>, all separated by tabs, to stdout.
    Note that outputs are not guaranteed to be in same order as inputs, if you use multithreading.
    """
    from utils import getListAsStr
    for i, fname, locs, y in project(model, fnames):
        print '%s\t%s' % (fname, getListAsStr(y, sep='\t'))
        sys.stdout.flush()

def evalvocabmain(model, outfname, impairs, cachedir='./', maxk=50):
    """Evaluates a vocabulary on a set of image pairs.
    Steps:
    - extracts sift from each pair of images
    - computes matches between each pair
    - projects each sift point using the given vocab
    - computes matches using projected points
    - computes score: # common matches/# original matches
    - outputs scores for each pair as [score, image 1, image 2], tab-separated to 'outfname'
    Parameters:
        model - either the model itself, or the pickle filename
        outfname - filename where the output scores are written to
        impairs - either a filename which lists image pairs (tab-separated, one pair per line)
                  or a list of fname pairs directly
        cachedir - optionally, gives the directory to store projected sift vectors to.
                   defaults to current directory
    """
    import cPickle as pickle
    maxk = int(maxk)
    base = lambda f: os.path.basename(f).rsplit('.',1)[0]
    if isinstance(model, basestring):
        model = pickle.load(open(model))
    # get list of image pairs
    if isinstance(impairs, basestring):
        impairs = [l.rstrip('\n').split('\t')[:2] for l in open(impairs)]
    # operate on each pair
    outf = open(outfname, 'wb')
    for im1, im2 in impairs:
        # extract sift
        sift1, sift2 = list(extract([im1,im2]))
        # compute matches and matchpairs
        matchfname = '%s-%s-matches.txt' % (base(im1), base(im2))
        matches = match(sift1, sift2, matchfname)
        matchpairs = set((i,j) for d, i, j, _, _ in matches)
        # project
        pf1, pf2 = projectindiv(model, [sift1,sift2], cachedir=cachedir, returnfvecs=1)
        pf1, pf2 = pf1[2:4], pf2[2:4]
        # match projections
        pmatchfname = os.path.join(cachedir, matchfname.replace('-matches', '-pmatches'))
        pmatches = match(pf1, pf2, pmatchfname, maxr=0, maxk=maxk)
        pmatchpairs = set((i,j) for d, i, j, _, _ in pmatches)
        common = matchpairs & pmatchpairs
        score = len(common)/float(len(matchpairs))
        log('Got %d raw matches, %d projected matches, %d common, for score %s' % (len(matchpairs), len(pmatchpairs), len(common), score))
        print >>outf, '%s\t%s\t%s' % (score, im1, im2)

def scoreinvindex(invindex, pvec):
    """Scores a projected vector against an inverted index.
    The invindex can either be the data structure loaded from disk, or a json filename.
    It can also be compressed with gzip.
    Yields sorted list of matches from the index ((i, score, imnum, imfname, invindex))
    """
    from gzip import GzipFile
    from collections import defaultdict
    if isinstance(invindex, basestring):
        f = GzipFile(invindex) if invindex.endswith('.gz') else open(invindex)
        invindex = json.load(f)

    imscores = defaultdict(float)
    for c in pvec:
        for imnum, s in invindex['words'][c]:
            imscores[imnum] += s
    matches = sorted(imscores.items(), key=lambda pair: pair[1], reverse=1)
    for i, (imnum, score) in enumerate(matches):
        yield ((i, score, imnum, invindex['ims'][imnum], invindex))
        #print '%d\t%s\t%s' % (i+1, score, invindex['ims'][imnum])

def makeindex(outfname, pfnames, k, tfidf=1, minscale=0, targetsize=0, minscore=0):
    """Makes an inverted index from the given list of projected individual outputs.
    Parameters:
        outfname: the json filename to write to
        tfidf: if 1, then uses tfidf weighing
        minscale: if > 0, then discards points at a scale smaller than that
        targetsize: if > 0, then sets minscale=max(im.size)/targetsize
        minscore: if the final score is less than this, we don't include it in the output

    """
    import math
    from collections import defaultdict
    ret = dict(k=k, ims=[], words=[{} for i in range(k)])
    if isinstance(pfnames, basestring):
        pfnames = [l.rstrip('\n') for l in open(pfnames)]
    t1 = time.time()
    nimwords = []
    wordcounts = defaultdict(int)
    for i, pfname in enumerate(pfnames):
        if i % 1 == 0:
            log('  On cvec %d, pfname %s, %0.3fs elapsed' % (i+1, pfname, time.time()-t1))
        fvecs = SiftFeat.siftFromFile(pfname, fmt='.projected.gz .projected'.split())
        if targetsize > 0:
            # get the maximum size
            maxsize = 0
            for f in fvecs:
                maxsize = max(maxsize, f.x, f.y)
            minscale = 0 if maxsize < targetsize else maxsize/float(targetsize)
            #print pfname, maxsize, minscale, [(f.data[0], f.scale) for f in fvecs[:5]]
        # get the list of points to use in the index based on minscale
        cvec = [f.data[0] for f in fvecs if f.scale >= minscale]
        log('    Went from %d features down to %d, due to targetsize %s, maxsize %s, and minscale %s' % (len(fvecs), len(cvec), targetsize, maxsize, minscale))
        nimwords.append(len(cvec))
        imnum = len(ret['ims'])
        ret['ims'].append(pfname)
        curwords = set()
        for c in cvec:
            curwords.add(c)
            d = ret['words'][c]
            d.setdefault(imnum, 0.0)
            d[imnum] += 1.0
        # accumulate curcounts into wordcounts
        for word in curwords:
            wordcounts[word] += 1
        #if i > 1: break
    # convert word dicts to lists, also applying tf-idf if wanted
    ret['nims'] = len(ret['ims'])
    log('Accumulating results')
    for i, wd in enumerate(ret['words']):
        lst = sorted(wd.items())
        if tfidf:
            newlst = []
            for imgnum, nid in lst:
                N = ret['nims']
                ni = float(wordcounts[i])
                l = math.log(N/ni)
                nd = float(nimwords[imgnum])
                score = l*nid/nd
                #log('  Got nid %s, nd %s for tf %s, N %s, ni %s, for idf %s, log %s, score %s' % (nid, nd, nid/nd, N, ni, N/ni, l, score))
                newlst.append((imgnum, score))
            lst = newlst
        # filter by minimum score
        oldlen = len(lst)
        lst = [(imgnum, score) for imgnum, score in lst if score > minscore]
        if len(lst) != oldlen:
            log('    For word %d, minscore %s, filtered %d items down to %d' % (i, minscore, oldlen, len(lst)))
        ret['words'][i] = lst
    log('Printing json output to %s' % (outfname))
    json.dump(ret, open(outfname, 'wb'), sort_keys=1, indent=2)

def makeindexmain(outfname, pfnames, k=10000, targetsize=200, minscore=1e-7):
    """Makes inverted sift index."""
    makeindex(outfname, pfnames, k=int(k), tfidf=1, minscale=0, targetsize=int(targetsize), minscore=float(minscore))

def verifyransac(f1, f2, minerror=0.5, maxiters=5000, maxk=10):
    """Runs ransac on the given set of matches.
    Returns dict of best model
    The best model dict has fields:
        inliers: list of inliers, as simple indices
        model: the best HomographyModel
        error: the error (measured as percentage of inliers)
        finalerror: the final error (as reprojection error on only the inliers)
    """
    from ransac import Ransac, HomographyModel
    from utils import getTimeDiffs
    minerror = float(minerror)
    # get matches
    times = [time.time()]
    matches = match(f1, f2, outfname=None, maxratio=None, maxk=10, maxr=0)
    times.append(time.time())
    data = [((f1.x, f1.y), (f2.x, f2.y)) for d, i, j, f1, f2 in matches]
    dataidx = dict((d, i) for i, d in enumerate(data))
    times.append(time.time())
    def callback(niters, elapsed, best, ransac, **kw):
        """Generic callback"""
        perc = len(best['inliers'])/float(len(dataidx))
        perc = min(perc, 0.8)
        ransac.maxiters = min(maxiters, Ransac.estimateNiters(4, perc))
        #print >>sys.stderr, 'On iter %d (%0.3fs elapsed), best model has %d inliers (%0.3f%%), error %0.2f, new maxiters %d' % (niters, elapsed, len(best['inliers']), perc, best['error'], ransac.maxiters)

    r = Ransac(4, HomographyModel, callback=callback, strongthresh=10, weakthresh=30, mininliers=8, minerror=minerror)
    times.append(time.time())
    try:
        ret = r.run(data)
    except KeyboardInterrupt:
        ret = r.best
    times.append(time.time())
    # also compute final error
    if ret['model']:
        ret['finalerror'] = ret['model'].geterror(ret['inliers'])
    times.append(time.time())
    log('  Got final times: %s' % (getTimeDiffs(times)))
    #print 'Error (1-inlier perc): %s, reprojection error: %s, model:\n%s' % (ret['error'], ret['finalerror'], ret['model'])
    return ret

def makescalesmain(targetsize, *fnames):
    """Prints scales for the given images"""
    targetsize = float(targetsize)
    print '#!/bin/bash\n'
    for f in fnames:
        im = Image.open(f)
        m = min(im.size)
        if m < targetsize: continue
        scale = m/targetsize
        print 'qsub -N project -cwd -o ~/outs/ -e ~/errs/ ~/pysub.sh ~/pylib/nksift.py projectindiv ~/projects/photorecall/curvocab.pickle %s %s' % (scale, f.replace('.jpg', '.sift.gz'))
        #print '%s\t%s' % (f, m/targetsize)

def combineprojmain(projnamelst, outfname):
    """Combines multiple projections into a single pickle file"""
    import cPickle as pickle
    from gzip import GzipFile
    fnames = [l.rstrip('\n') for l in open(projnamelst)]
    obj = {}
    for fname in fnames:
        print fname
        els = [l.strip().split() for l in GzipFile(fname)]
        data = [(map(float, el[:4]), map(int, el[4:])) for el in els]
        locs, fvecs = zip(*data)
        if 0:
            feats = SiftFeat.siftFromFile((locs, fvecs))
            obj[fname] = feats
        else:
            obj[fname] = (locs, fvecs)
    if outfname.endswith('.json'):
        json.dump(obj, open(outfname, 'wb'), sort_keys=1, indent=2)
    elif outfname.endswith('.pickle'):
        pickle.dump(obj, open(outfname, 'wb'), -1)



if __name__ == "__main__":
    tasks = 'draw match drawmatches ransac fullsift match1st extract projectindiv projectall makevocab evalvocab aggregate kmcombos makeindex makescales combineproj'.split()
    if len(sys.argv) < 2:
        print 'Usage: python %s <%s> <args> ...' % (sys.argv[0], '|'.join(tasks))
        sys.exit()
    task = sys.argv[1]
    if task not in tasks:
        print 'Invalid task "%s", should be one of: %s' % (task, ','.join(tasks))
        sys.exit()
    func = eval('%smain' % task)
    func(*sys.argv[2:])

