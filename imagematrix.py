"""Reads a list of files on stdin and makes a composite image.
All images on one line are put on one row. All images are resized as specified.

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
from PIL import Image

def getImageMatrix(f, transpose=0, sort=0, xlimit=200, prunelarge=5, delim='\t'):
    """Gets the image matrix from the given file, transposing and sorting by length, if wanted.
    Set sort=1 for sort by longest first, -1 for shortest first, 0 for no sorting."""
    files = [l.strip().split(delim) for l in f]
    sfiles = sorted([(len(f), i) for i, f in enumerate(files)], reverse=1)
    if prunelarge > 0: # prune rows which are way longer than the rest
        while sfiles[0][0] > prunelarge*sfiles[1][0]:
            print 'Popping row %d, with %d images' % (sfiles[0][1], sfiles[0][0])
            del files[sfiles[0][1]] # delete the given element
            del sfiles[0]
    print sfiles

    toadd = [f[xlimit:] for f in files if len(f)>=xlimit]
    files = [f[:xlimit] for f in files if f]
    files.extend(toadd)
    if sort == 1: files.sort(key=len, reverse=1)
    elif sort == -1: files.sort(key=len, reverse=0)
    if transpose:
        numw = max([len(f) for f in files])
        newfiles = [[] for i in range(numw)]
        for flist in files:
            for i, f in enumerate(flist):
                newfiles[i].append(f)
        files = newfiles
    return files

def matrixGrep(matrix, locs, size, uniq=0):
    """Greps the matrix to get filenames at the given locations"""
    def getFname(mx, my):
        """Returns the filename at the given matrix location"""
        try:
            return matrix[my][mx]
        except IndexError: return None
    mlocs = [(x//size[0], y//size[1]) for x, y in locs]
    ret = [getFname(mx, my) for mx, my in mlocs]
    if uniq:
        ret = list(set(ret))
    return ret

def createImageFromImageMatrix(ims, size, replace=None, xpad=0, ypad=0, fill=(0,0,0)):
    """Creates an image from a matrix and returns it"""
    def repfunc(p):
        """Replaces pixel value p with something else"""
        if p == replace[0]: return replace[1]
        return p

    numw = max([len(f) for f in ims])
    numh = len(ims)
    w, h = size
    out = Image.new('RGB', (numw*w + (numw-1)*xpad, numh*h + (numh-1)*ypad), fill)
    x, y = 0, 0
    for row in ims:
        x = 0
        for im in row:
            if im:
                if replace:
                    im.putdata([repfunc(p) for p in im.getdata()])
                out.paste(im, (x, y))
            x += w + xpad
        y += h + ypad
    return out

def createImageFromMatrix(files, size, **kw):
    ims = [[Image.open(f).resize(size, Image.ANTIALIAS) for f in row] for row in files]
    return createImageFromImageMatrix(ims, size, **kw)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Creation Usage: python %s c outfname <width> [<height>=<width>]" % (sys.argv[0])
        print "Grep Usage:     python %s g[u] badlocsfname <width> [<height>=<width>]" % (sys.argv[0])
        sys.exit()

    task = sys.argv[1]
    fname = sys.argv[2]
    w = int(sys.argv[3])
    try:
        h = int(sys.argv[4])
    except IndexError: h = w
    size = w, h

    files = getImageMatrix(sys.stdin, sort=1, transpose=0)
    numw = max([len(f) for f in files])
    numh = len(files)
    print >>sys.stderr, 'Got size %s and nums %s for image size %s' % (size, (numw, numh), (numw*w, numh*h))


    if task == 'c':
        replace = [(255, 0, 0), (255,1,1)]
        out = createImageFromMatrix(files, size, replace=replace)
        out.save(fname)
    elif task.startswith('g'):
        uniq = 1 if 'u' in task else 0
        locs = [map(int, l.strip().split()) for l in open(fname)]
        locs = [l for l in locs if l]
        vals = matrixGrep(files, locs, size, uniq=uniq)
        for v in vals:
            print v
