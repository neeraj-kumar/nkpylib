"""Sed for images. Replaces colors with others, or prints to stdout.

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
from itertools import *

help = '''

Input patterns are groups of () separated by | or , which represents taking the
OR or AND, respectively of the different patterns. Each pattern consists of
valid python expressions using the variables r,g,b to refer to each color
channel, or l to refer to their average, or x,y to refer to the pixel location.

Similarly, output patterns must a set of 3 valid python expressions, each
separated by commas. Each one is eval'ed to get the output r, g, b values. The
same variables as in the input pattern are legal here, and refer to the input
values.'''

def imageiter(im):
    """Returns an generator expression for an image, which returns [(x,y), (im.getpixel((x,y))] pairs for the image"""
    locs = ((x,y) for y in range(im.size[1]) for x in range(im.size[0]))
    return ((loc, im.getpixel(loc)) for loc in locs)

func_template = '''def matchfunc(loc, pix):
    x, y = loc
    r, g, b = pix
    l = (r+g+b)//3
    if %s: return 1
    return 0'''

def parseInputPattern(inpat):
    """Parses an input pattern"""
    s = inpat.strip().lower()
    s = s.replace('|', ' or ')
    s = s.replace(',', ' and ')
    funcs = func_template % s
    #print 'Creating a function:\n%s' % (funcs)
    exec funcs
    return matchfunc

def parseOutputPattern(outpat):
    """Parses an output pattern"""
    r, g, b = outpat.strip().lower().split(',')
    return (r, g, b)

if __name__ == "__main__":
    if len(sys.argv) not in [3, 5]:
        print 'Search only usage:        python %s <inpattern> <infile>' % (sys.argv[0])
        print 'Search and replace usage: python %s <inpattern> <outpattern> <infile> <outfile>' % (sys.argv[0])
        print help
        sys.exit()
    if len(sys.argv) == 3:
        inpat, infname = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 5:
        inpat, outpat = sys.argv[1], sys.argv[2]
        infname, outfname = sys.argv[3], sys.argv[4]
    matchfunc = parseInputPattern(inpat)
    matches = []
    im = Image.open(infname)
    for loc, pix in imageiter(im):
        print >>sys.stderr, '  At row %s with %d matches so far...   \r' % (loc[1], len(matches)),
        sys.stdout.flush()
        if matchfunc(loc, pix):
            matches.append([loc, pix])
    print >>sys.stderr
    if len(sys.argv) == 3: # just print locations
        for loc, pix in matches:
            print loc[0], loc[1]
    elif len(sys.argv) == 5: # write output image
        outfunc = parseOutputPattern(outpat)
        pix = im.load()
        for loc, p in matches:
            pix[loc] = outfunc(loc, p)
        im.save(outfname)
