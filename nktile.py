"""Simple code to tile images.

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

import sys, os, time
from math import sqrt, ceil
from PIL import Image
from itertools import *
from nkutils import *

RESIZE_ALL = (64, 64)

def tile(fnames, resize=RESIZE_ALL, textonly=0, rows=None, cols=None):
    """Tiles images and returns a tiled image"""
    maxsize = [0, 0]
    assert fnames
    todel = set()
    for fname in fnames:
        try:
            im = Image.open(fname)
            maxsize = [max(m, s) for m, s in zip(maxsize, im.size)]
        except Exception: 
            todel.add(fname)
            continue
    fnames = [os.path.realpath(f) for f in fnames if f not in todel] # convert symlinks to real paths
    print >>sys.stderr, "There were %d images (removed %d bad) with maxsize %d x %d" % (len(fnames), len(todel), maxsize[0], maxsize[1])
    # now figure out the right size of the output image
    if not cols and not rows: # if neither dim is given, use the sqrt
        cols = int(sqrt(len(fnames)))
        rows = len(fnames)//cols + (0 if len(fnames)%cols == 0 else 1)
    elif cols and not rows: # only cols is given
        rows = len(fnames)//cols + (0 if len(fnames)%cols == 0 else 1)
    elif not cols and rows: # only rows is given
        cols = len(fnames)//rows + (0 if len(fnames)%rows == 0 else 1)
    else: # both are given
        pass
    if textonly:
        cur = 0
        rows = list(nkgrouper(cols, fnames))
        return rows
    if resize:
        boxsize = resize
    else:
        boxsize = maxsize
    outsize = tuple([s*n for s, n in zip(boxsize, [cols, rows])])
    print >>sys.stderr, "Output will be tiling %d x %d images, with image size %d x %d" % (cols, rows, outsize[0], outsize[1])
    out = Image.new(im.mode, outsize)
    cur = 0
    start = time.time()
    for r in range(rows):
        for c in range(cols):
            print >>sys.stderr, '  At col %d, row %d, cur %d, %0.2f secs elapsed...\r   ' % (c, r, cur, time.time()-start),
            im = Image.open(fnames[cur]).resize(boxsize, Image.ANTIALIAS)
            box = (c*boxsize[0], r*boxsize[1])
            out.paste(im, box)
            cur += 1
            if cur >= len(fnames): break
    print >>sys.stderr
    return out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python %s <outname> <image1> [<image2> ...]" % sys.argv[0]
        sys.exit()
    maxsize = [0, 0]
    outname = sys.argv[1]
    fnames = sys.argv[2:]
    if not fnames: sys.exit()
    outim = tile(fnames)
    outim.save(outname)
