"""Lots of useful matplotlib utilities, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2010, Neeraj Kumar (neerajkumar.org)
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

from nkpylib.utils import *

import matplotlib
try:
    matplotlib.use('Agg')
except UserWarning: pass

from matplotlib import rc
from pylab import * # type: ignore
from matplotlib.ticker import *
from PIL import Image


def centerel(elsize, contsize):
    """Centers an element of the given size in the container of the given size.
    Returns the coordinates of the top-left corner of the element relative to
    the container."""
    w, h = elsize
    W, H = contsize
    x = (W-w)//2
    y = (H-h)//2
    return (x, y)

def axisfontsize(ax, size):
    """Sets the fontsize for the axis tick labels"""
    for lx in ax.get_xticklabels():
        lx.set_size(size)
    for ly in ax.get_yticklabels():
        ly.set_size(size)

def plotresults(t, rocvals, colors, outname, type, figsize=None, fontfac=1.0):
    """Plots results as an ROC curve."""
    font = {'fontname': 'Arial', 'fontsize': 22*fontfac}
    if not figsize:
        figsize = (10,10)
    figure(3049, figsize=figsize)
    clf()
    xlabel('False Positive Rate', font)
    ylabel('True Positive (Detection) Rate', font)
    font['fontsize'] = 24*fontfac
    ax = gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axisfontsize(ax, 18*fontfac)
    ax.xaxis.set_major_locator(LinearLocator(11))
    ax.yaxis.set_major_locator(LinearLocator(11))
    if fontfac < 1:
        ax.set_position((0.08, 0.12, 0.90, 0.85))
    else:
        ax.set_position((0.08, 0.06, 0.90, 0.92))
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family='sans-serif', size=16*fontfac)
    legend(loc='lower right', prop=fp)

def genNumberedImage(im, rect, num, fsize=10):
    """Generates a new image using the given rect from the image, and puts a number on it"""
    from PIL import ImageFont, ImageDraw
    font = ImageFont.truetype('arial.ttf', fsize)
    ret = im.crop(rect)
    ret.load()
    draw = ImageDraw.Draw(ret)
    s = '%s' % num
    size = draw.textsize(s, font=font)
    pos = centerEl(size, ret.size)
    #log("Got a string of '%s', a size of %s, a pos of %s" % (s, size, pos))
    draw.text(pos, s, fill="blue", font=font)
    del draw
    return ret

def rearrangeim(im, ndivs=4):
    """Rearranges an image to make it more square sized"""
    from PIL import Image
    xincr = im.size[0]//ndivs
    yincr = im.size[1]
    size = (xincr, yincr*ndivs)
    out = Image.new('RGB', size)
    for i in range(ndivs):
        out.paste(im.crop((xincr*i, 0, (xincr*(i+1)-1), yincr-1)), (0, yincr*i))
    print('Converted image of size %s to size %s' % (im.size, out.size))
    return out

