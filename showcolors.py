#!/usr/bin/env python
"""A simple script to show a list of colors.

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

import os, sys, PIL
from PIL import ImageColor, Image

def readcolor(s):
    """Reads a color from the given string or None if it can't parse one.
    Uses ImageColor.getrgb(), and also checks for tab-separated values"""
    try:
        return ImageColor.getrgb(s)
    except ValueError:
        try:
            v = int(s)
            return (v,v,v)
        except ValueError: pass
        try:
            return map(int, s.split())
        except ValueError: pass
    return None

def makecolorim(colors):
    """Makes a color image with the given set of colors"""
    import math
    w = int(math.sqrt(len(colors)))
    h = int(math.ceil(len(colors)/float(w)))
    out = Image.new('RGB', (w,h), (0,0,0))
    pix = out.load()
    i = 0
    try:
        for y in range(h):
            for x in range(w):
                pix[x,y] = tuple(colors[i])
                i += 1
    except IndexError: pass
    return out


if __name__ == '__main__':
    colors = [readcolor(l.strip()) for l in sys.stdin]
    colors = [c for c in colors if c]
    outim = makecolorim(colors)
    try:
        outname = sys.argv[1]
        outim.save(outname)
    except Exception:
        outim.show()

