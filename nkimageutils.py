"""Lots of useful image utilities, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2010-2013, Neeraj Kumar (neerajkumar.org)
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

from nkutils import *
from PIL import Image
try:
    import numpy
    NUMPY = 1
except ImportError:
    NUMPY = 0
EPSILON = 0.00000001


# ITERATORS
def imagelociter(im):
    """Returns an image location iterator for the given image.
    Simply a generator giving (x,y) pairs."""
    locs = ((x,y) for y in range(im.size[1]) for x in range(im.size[0]))
    return locs

def imageiter(im, usepix=1):
    """Returns an generator expression for an image, which returns:
    ((x,y), (im.getpixel((x,y))) pairs for the image.  If usepix is true (the
    default), then creates a pix object by calling im.load(), and uses that
    instead of im.getpixel()."""
    locs = list(imagelociter(im))
    if usepix:
        pix = im.load()
        return ((loc, pix[loc]) for loc in locs)
    return ((loc, im.getpixel(loc)) for loc in locs)


# SIMPLE CHECKS
def isimg(fname, nameonly=0):
    """Returns true if the given fname is an image fname.
    If nameonly is 0 (default), then tries opening the file using PIL.
    Otherwise, only checks the filename."""
    if nameonly:
        exts = '.jpg .jpeg .gif .bmp .png .pgm'.split()
        fname = fname.lower()
        for e in exts:
            if fname.endswith(e): return 1
        return 0
    else:
        try:
            im = Image.open(fname)
            return 1
        except IOError: return 0

def tightcrop(im, bg=(0,0,0), dist=0):
    """Makes a tight crop of an image, where background pixels matching bg are removed"""
    pix = im.load()
    valid = []
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            d = lpdist(pix[x,y], bg, 2)
            if d > dist:
                valid.append((x,y))
    xs, ys = zip(*valid)
    limits = (min(xs), min(ys), max(xs), max(ys))
    return im.crop(limits)

def validcolor(c):
    """Takes a color and makes it valid by clamping each value between 0 and 255"""
    try:
        ret = [clamp(int(v+0.5), 0, 255) for v in c]
        return type(c)(ret)
    except TypeError:
        return clamp(int(v+0.5), 0, 255)

def datatoustr(seq, offset=2000):
    """Converts a sequence to a unicode string"""
    return ''.join(unichr(offset+c) for c in seq)

def imgarea(im, zeroel=None):
    """Returns the area of the image, as determined by checking against the 'zeroel'.
    i.e., the number of pixels that are != zeroel.
    If 'zeroel' is not given, it's determined from the image type.
    """
    zeros = {'L':0, '1':0, 'I':0, 'F': 0.0, 'RGB': (0,0,0), 'RGBA': (0,0,0,0)}
    if zeroel is None:
        zeroel = zeros[im.mode]
    nz = sum(1 for p in im.getdata() if p != zeroel)
    return nz

# PATCHES
def getPatchRects(im, size, ss=1):
    """Gets patch rectangles of the given size from the given image, with the given subsampling."""
    w, h = im.size
    ret = []
    for y in range(0, h, ss):
        if y+size[1] >= h: break
        for x in range(0, w, ss):
            if x+size[0] >= w: break
            ret.append([x, y, x+size[0], y+size[1]])
    return ret

def getPatchAsLst(im, rect, type='array'):
    """Returns the given rect from the image as a list.
    For multichannel images, this returns rgbrgbrgb rather than rrrgggbbb.
    The type can be one of 'list', 'tuple', 'array', 'numpy'."""
    from array import array
    d = {'list': list, 'tuple': tuple, 'array': lambda x: array('i', x)}
    try:
        from numpy import array as arr
        d['numpy'] = arr
    except ImportError: pass
    assert type in d
    flatfunc = flatten if len(im.mode) > 1 else lambda i: i
    return d[type](flatfunc(list(im.crop(rect).getdata())))

def colorRects(im, rects, colors, where='center', shape='point', size=1):
    """Colors the given rects with the given colors, and draws on the image.
    Set 'where' to where you want to draw:
        'center': draw at the center of the rect [default]
        'top-left': draw at the top-left of the rect
        'bot-right': draw at the bottom right of the rect
    Set 'shape' and 'size' to what you want to draw:
        'point': a single pixel, size is ignored [default]
        'circle': a circle with radius='size', outline only
        'fillcircle': a circle with radius='size', filled-in
        'rect': a rectangle with width='size'*2, outline only
        'fillrect': a rectangle with width='size'*2, filled-in
    Returns the image (but not that it's modified in-place)
    """
    # generate locations
    if where == 'top-left':
        locs = [(x0,y0) for x0,y0,x1,y1 in rects]
    elif where == 'bot-right':
        locs = [(x1,y1) for x0,y0,x1,y1 in rects]
    elif where == 'center':
        locs = [((x0+x1)//2,(y0+y1)//2) for x0,y0,x1,y1 in rects]
    # draw
    import ImageDraw
    draw = ImageDraw.Draw(im)
    for loc, color in zip(locs, colors):
        if shape == 'point':
            draw.point(loc, fill=color)
        else:
            r = rectat(loc[0], loc[1], size*2, size*2)
            fill = color if shape.startswith('fill') else None
            if 'circ' in shape:
                draw.ellipse(r, fill=fill, outline=color)
            elif 'rect' in shape:
                draw.rectangle(r, fill=fill, outline=color)
            else:
                raise NotImplementedError('Shape must be one of point, circle, fillcircle, rect, fillrect')
    return im


# SIMILARITY
def imsim(im1, im2, thresh=50.0):
    """Returns similarity (between 0.0 and 1.0) of two images.
    Assumes that im1 is roughly a cropped version of im2.
    """
    rescale = lambda v: int(v/thresh)
    data = [rescale(v) for v in im1.getdata()]
    words = [datatoustr(w[10:-10]) for w in grouper(im1.size[0], data)]
    haystack = datatoustr(rescale(v) for v in im2.getdata())
    found = len([1 for w in words if w in haystack])
    #print '%d words, first len %d, and %d in haystack, %d found' % (len(words), len(words[0]), len(haystack), found)
    return found/float(len(words))


# CROPPING AND PADDING
def centerrect(im, aspect=1.0):
    """Returns a rect that is centered at the center of the given image and contains the image, with the given aspect ratio"""
    w, h = im.size
    cx, cy = w/2.0, h/2.0
    if w/float(h) < aspect:
        w = h*aspect
    if w/float(h) > aspect:
        h = w/aspect
    assert w >= im.size[0]
    assert h >= im.size[1]
    r = (int(cx-w/2.0), int(cy-h/2.0), int(cx+w/2.0 + 0.5), int(cy+h/2.0 + 0.5))
    #print 'returning rect %s with cen %s on im size %s' % (r, rectcenter(r), im.size)
    return r

def croprect(im, rect, bg=(0,0,0)):
    """Crops an image with the given rect (x0,y0,x1,y1).
    The rect could extend outside the image, in which case the background is filled in with the given bg"""
    rect = [int(r+0.5) for r in rect]
    w = rect[2]-rect[0]
    h = rect[3]-rect[1]
    out = Image.new('RGB', (w,h), bg)
    offset = [0,0]
    if rect[0] < 0:
        offset[0] = abs(rect[0])
        rect[0] = 0
    if rect[1] < 0:
        offset[1] = abs(rect[1])
        rect[1] = 0
    rect[2] = min(rect[2], im.size[0]-1)
    rect[3] = min(rect[3], im.size[1]-1)
    out.paste(im.crop(tuple(rect)), tuple(offset))
    return out

def padimg(im, padx, pady, fill):
    """Pads the image in each direction with 'fill'.
    padx and pady can each be one of:
        int: # pixels to pad in both directions
        float: percentage of width or height to pad by
        [int or float, int or float]: #/% to pad by in each direction
    """
    def var2vals(var, dim):
        """Converts an input var to a 2-ple, given also the dimensions along this axis"""
        try:
            var = var[:]
        except TypeError:
            var = [var, var]
        ret = []
        for v in var:
            assert v > 0
            if isinstance(v, (int, long)):
                ret.append(v)
            else:
                assert isinstance(v, float)
                ret.append(int(dim*v + 0.5))
        return ret

    xs = var2vals(padx, im.size[0])
    ys = var2vals(pady, im.size[1])
    size = (im.size[0]+sum(xs), im.size[1]+sum(ys))
    ret = Image.new(im.mode, size, fill)
    ret.paste(im, (xs[0], ys[0]))
    return ret


# SAMPLING
def makesampler(im, outvars=('loc', 'val'), filter=Image.BILINEAR):
    """Returns a sampler for the given image using the given filter.
    This function is called with ((x,y)), where x and y can be floats.
    Values outside the image are clamped to the image.
    Values inside the image are interpolated using the given filter.
    You determine the output format with the elements in 'outvars':
        'loc': The interpolated (x,y) as valid ints in the image
        'val': The interpolated output value.
    The default is to return (loc, val). You can instead have it
    return (loc) or (val) if you prefer.

    Options for the interpolation are:
        Image.NEAREST - nearest neighbor sampling
        Image.BILINEAR (default) - bilinear sampling
    """
    assert filter in [Image.BILINEAR, Image.NEAREST]
    pix = im.load()
    def sampler(loc):
        """A sampler created for an image"""
        x, y = loc
        x = clamp(x, 0, im.size[0]-1)
        y = clamp(y, 0, im.size[1]-1)
        outx = int(x+0.5)
        outy = int(y+0.5)
        if filter == Image.NEAREST:
            v = pix[outx, outy]
        elif filter == Image.BILINEAR:
            x0, y0, x1, y1 = int(x), int(y), min(int(x+1), im.size[0]-1), min(int(y+1), im.size[1]-1)
            vy0 = lerp(x, (x0, pix[x0,y0]), (x1, pix[x1,y0]))
            vy1 = lerp(x, (x0, pix[x0,y1]), (x1, pix[x1,y1]))
            v = lerp(y, (y0, vy0), (y1, vy1))
            #m = [['', x0, '', x1], [y0, pix[x0,y0], vy0, pix[x1,y0]], ['', '', v, ''], [y1, pix[x0,y1], vy1, pix[x1,y1]]]
            #print matrixprint(m, sep=' | ')
        #print im.size, loc, x, y, outx, outy, v
        outmap = {'loc': (outx, outy), 'val': v}
        if outvars in outmap:
            return outmap[outvars]
        else:
            return tuple([outmap[v] for v in outvars])

    return sampler

def getImageIndex(im):
    """Returns an index of an image, as a dict.
    This is a mapping from color -> [locations]"""
    ret = {}
    for loc, col in imageiter(im):
        ret.setdefault(col, []).append(loc)
    return ret


# COLOR
def randcolor(mode):
    """Makes a random color for the given mode"""
    import random
    ri = lambda v: random.randint(0, v)
    if mode == 'L':
        return ri(255)
    elif mode == '1':
        return ri(1)
    elif mode == 'I':
        return ri(2**23-1)
    elif mode == 'F':
        return random.random()
    elif mode == 'RGB':
        return (ri(255), ri(255), ri(255))
    elif mode == 'RGBA':
        return (ri(255), ri(255), ri(255), ri(255))
    else:
        assert 1 == 0, 'invalid mode %s' % (mode)

def colormap(v, name='jet'):
    """Takes an input float between 0 and 1 and maps it to an RGB color using the given colormap name.
    Current names are 'jet' or 'hsv' (from matlab),
    or 'gray', which just replicates the color value 3 times."""
    jet = [(0,0,0.5625), (0,0,0.625), (0,0,0.6875), (0,0,0.75), (0,0,0.8125), (0,0,0.875), (0,0,0.9375), (0,0,1), (0,0.0625,1), (0,0.125,1), (0,0.1875,1), (0,0.25,1), (0,0.3125,1), (0,0.375,1), (0,0.4375,1), (0,0.5,1), (0,0.5625,1), (0,0.625,1), (0,0.6875,1), (0,0.75,1), (0,0.8125,1), (0,0.875,1), (0,0.9375,1), (0,1,1), (0.0625,1,0.9375), (0.125,1,0.875), (0.1875,1,0.8125), (0.25,1,0.75), (0.3125,1,0.6875), (0.375,1,0.625), (0.4375,1,0.5625), (0.5,1,0.5), (0.5625,1,0.4375), (0.625,1,0.375), (0.6875,1,0.3125), (0.75,1,0.25), (0.8125,1,0.1875), (0.875,1,0.125), (0.9375,1,0.0625), (1,1,0), (1,0.9375,0), (1,0.875,0), (1,0.8125,0), (1,0.75,0), (1,0.6875,0), (1,0.625,0), (1,0.5625,0), (1,0.5,0), (1,0.4375,0), (1,0.375,0), (1,0.3125,0), (1,0.25,0), (1,0.1875,0), (1,0.125,0), (1,0.0625,0), (1,0,0), (0.9375,0,0), (0.875,0,0), (0.8125,0,0), (0.75,0,0), (0.6875,0,0), (0.625,0,0), (0.5625,0,0), (0.5,0,0)]
    hsv = [(1,0,0), (1,0.09375,0), (1,0.1875,0), (1,0.28125,0), (1,0.375,0), (1,0.46875,0), (1,0.5625,0), (1,0.65625,0), (1,0.75,0), (1,0.84375,0), (1,0.9375,0), (0.96875,1,0), (0.875,1,0), (0.78125,1,0), (0.6875,1,0), (0.59375,1,0), (0.5,1,0), (0.40625,1,0), (0.3125,1,0), (0.21875,1,0), (0.125,1,0), (0.03125,1,0), (0,1,0.0625), (0,1,0.15625), (0,1,0.25), (0,1,0.34375), (0,1,0.4375), (0,1,0.53125), (0,1,0.625), (0,1,0.71875), (0,1,0.8125), (0,1,0.90625), (0,1,1), (0,0.90625,1), (0,0.8125,1), (0,0.71875,1), (0,0.625,1), (0,0.53125,1), (0,0.4375,1), (0,0.34375,1), (0,0.25,1), (0,0.15625,1), (0,0.0625,1), (0.03125,0,1), (0.125,0,1), (0.21875,0,1), (0.3125,0,1), (0.40625,0,1), (0.5,0,1), (0.59375,0,1), (0.6875,0,1), (0.78125,0,1), (0.875,0,1), (0.96875,0,1), (1,0,0.9375), (1,0,0.84375), (1,0,0.75), (1,0,0.65625), (1,0,0.5625), (1,0,0.46875), (1,0,0.375), (1,0,0.28125), (1,0,0.1875), (1,0,0.09375)]
    gray = [(0,0,0), (0.5, 0.5, 0.5), (1,1,1)]
    maps = {'jet': jet, 'hsv': hsv, 'gray':  gray}
    assert name in maps
    assert v >= 0 and v <= 1
    m = maps[name]
    i = (len(m)-1)*v
    from math import ceil, floor
    from_, to = int(floor(i)), int(ceil(i))
    a,b = m[from_], m[to]
    ret = tuple([clamp(int(lerp(i, (from_,col_a), (to, col_b))*255),0,255) for col_a, col_b in zip(a,b)])
    return ret

def indexedcolor(i, num, npersat=15, lightness=60):
    """Returns an rgb color triplet for a given index, with a finite max 'num'.
    Thus if you need 10 colors and want to get color #5, you would call this with (5, 10).
    The colors are "repeatable".
    """
    import math
    from PIL import ImageColor
    nsats = int(math.ceil(num/float(npersat)))
    sat = 100 - int((i//npersat)*(100/nsats))
    l = lightness
    nhues = int(math.ceil(num/float(nsats)))
    hue = (i % nhues) * (360//nhues)
    #print >>sys.stderr, 'For i %d, num %d, got %d sats, %d hues -> %d, %d, %d' % (i, num, nsats, nhues, hue, sat, l)
    return ImageColor.getrgb('hsl(%d,%d%%,%d%%)' % (hue, sat, l))

def rgb2hsv(im):
    """Converts an rgb image to an hsv one"""
    from math import atan2, sqrt, pi
    data = im.getdata()
    fac = 1/255.0
    s3 = sqrt(3.0)
    vfac = 127.5/pi
    outdata = []
    # having a cache helps by a factor of 2-5X
    cache = {}
    for rgb in data:
        if rgb not in cache:
            r, g, b = rgb
            # compute max, min, L of output (range: [0, 1.0])
            M, m = max(rgb), min(rgb) #FIXME this line is ridiculously slow (almost half the total time!)
            #M, m = r, g
            mdiff = (M-m)*fac
            l = (M+m)*0.5*fac
            # compute S of output (range: [0, 255])
            if l == 0 or mdiff < EPSILON:
                s = 0.0
            elif l <= 0.5:
                s = mdiff/(2*l)
            else:
                s = mdiff/(2.0-(2.0*l))
            s = int(s*255.0)
            # compute L of output (range: [0, 255])
            l = int(l*255.0)
            # compute H of output (range: [0, 255])
            # Compute hue from r,g,b, using T. Gever et. al.'s paper
            v = atan2(s3*(g-b), (r-g)+(r+b))
            # go from [-pi, pi] -> [0, 255]
            h = int((v+pi)*vfac)
            cache[rgb] = (h,s,l)
        outdata.append(cache[rgb])
    # create output and set data
    out = Image.new('RGB', im.size)
    out.putdata(outdata)
    return out

def rgb2xyz(im):
    """Converts and rgb image to an xyz one"""
    mat = ( 0.412453, 0.357580, 0.180423, 0,
            0.212671, 0.715160, 0.072169, 0,
            0.019334, 0.119193, 0.950227, 0 )
    return im.convert('RGB', mat)

def fmtcolor(rgb, fmt, percs=0):
    """Formats an rgb color triplet in different formats:
        hex: #rrggbb
        spaces: r g b
        commas: r,g,b
        tabs: r\tg\tb
    If percs is true, then converts to fractions first.
    """
    r, g, b = rgb
    fmts = 'hex spaces commas tabs'.split()
    assert fmt in fmts
    if fmt == 'hex':
        return '#%02x%02x%02x' % (r, g, b)
    # convert to percs if needed
    if percs:
        r /= 255.0
        g /= 255.0
        b /= 255.0
    if fmt in 'spaces commas tabs'.split():
        d = dict(spaces=' ', commas=',', tabs='\t')
        return getListAsStr((r,g,b), sep=d[fmt])


# FILTERING
def numpy_conv2d(mat, kernel):
    """Performs a 2d convolution on the given numpy matrix.
    Output matrix is of same size, with 0s on boundaries"""
    import numpy
    w,h = mat.shape
    kw, kh = kernel.shape
    # temporarily make the output larger than the actual
    out = numpy.zeros((w+kw,h+kh), mat.dtype)
    for y in range(kh):
        for x in range(kw):
            out[x:x+w, y:y+h] += mat*kernel[kw-x-1,kh-y-1]
    out = out[kw//2:kw//2+w, kh//2:kh//2+h]
    out[:kw//2,:] = 0
    out[w-kw//2:,:] = 0
    out[:,:kh//2] = 0
    out[:,h-kh//2:] = 0
    assert out.shape == mat.shape
    return out

def floatFilter(im, size, kernel, scale=None, offset=0):
    """Like applying im.filter(ImageFilter.Kernel(size, kernel, scale, offset)), but for float images"""
    assert im.mode == 'F'
    assert size[0] % 2 == 1
    assert size[1] % 2 == 1
    assert len(kernel) == size[0]*size[1]
    if scale is None:
        try:
            scale = 1.0/sum(kernel)
        except ZeroDivisionError: scale = 1
    ret = Image.new('F', im.size, 0)
    w, h = im.size
    kw, kh = size[0]//2, size[1]//2
    src = im.load()
    dst = ret.load()
    locs = [(x,y) for y in range(-kh, kh+1) for x in range(-kw, kw+1)]
    #print locs, scale, offset
    def val(x,y):
        #orig = x,y
        offs = ((x+lx, y+ly) for lx, ly in locs)
#       vals = (src[x,y] for (x,y) in offs)
#       mult = (v*k for v, k in zip(vals, kernel))
#       ret = sum(mult)
#       print '  Orig: %s, offs: %s, k: %s, vals %s, mult %s, sum %s' % (orig, offs, kernel, vals, mult, ret)
        ret = sum(src[x,y]*k for (x,y), k in zip(offs, kernel))
        return ret

    for y in range(kh, h-kh):
        for x in range(kw, w-kw):
            v = (val(x,y) * scale) + offset
            dst[x,y] = v
    return ret


# EDGES AND BLUR
SOBEL_X = (1, 0, -1, 2, 0, -2, 1, 0, -1)
SOBEL_Y = (1, 2, 1, 0, 0, 0, -1, -2, -1)

SIMPLE_X = (0, 0, 0, 1, 0, -1, 0, 0, 0)
SIMPLE_Y = (0, 1, 0, 0, 0, 0, 0, -1, 0)

def getSobelDerivs(im):
    """Returns the sobel-filtered versions of an image."""
    from PIL import ImageFilter
    assert(im.mode == 'L')
    # get x- and y-filtered edge images
    sx, sy = SOBEL_X, SOBEL_Y
    dx = im.filter(ImageFilter.Kernel((3,3), sx, scale=8.0, offset=127))
    dy = im.filter(ImageFilter.Kernel((3,3), sy, scale=8.0, offset=127))
    #dx.save('dx.png')
    #dy.save('dy.png')
    return dx, dy

def getEdgeMagIm(dx, dy):
    """Returns an edge magnitude L image"""
    assert dx.mode == 'L'
    assert dy.mode == 'L'
    out = Image.new('L', dx.size, 0)
    px = dx.load()
    py = dy.load()
    po = out.load()
    for loc in imagelociter(dx):
        po[loc] = int(math.sqrt((px[loc]**2 + py[loc]**2)/2.0))
    return out

def getFloatDerivs(im, kernels=(SIMPLE_X, SIMPLE_Y), ksizes=((3,3),(3,3)), maxval=255):
    """Returns floating-point x and y derivatives using the given kernels.
    If the input image is not in Float mode, then it is first converted, using the given maxval to normalize."""
    assert len(im.mode) == 1
    if im.mode != 'F':
        im = im.point(lambda p: p/float(maxval), 'F')
        #float2grey(im).save('im.png')
    sx, sy = kernels
    dx = floatFilter(im, ksizes[0], sx)
    dy = floatFilter(im, ksizes[1], sy)
    return dx, dy

def getFloatEdgeMag(dx, dy):
    """Returns a floating-point edge magnitude image using the given derivatives.
    This is just sqrt(dx**2 + dy**2)"""
    from math import sqrt
    import ImageMath
    assert dx.size == dy.size
    assert dx.mode == dy.mode == 'F'
    ret = Image.new('F', dx.size)
    ret.putdata([sqrt(x*x+y*y) for x,y in zip(dx.getdata(), dy.getdata())])
    #ret = ImageMath.eval('(x*x +y*y)**0.5', x=dx, y=dy) #TODO this doesn't work because POW_F is not defined in imaging math!
    return ret

def getFloatEdgeDirs(dx, dy):
    """Returns a floating-point edge directions image using the given derivatives.
    This is just atan(dy / dx)"""
    from math import atan
    assert dx.size == dy.size
    assert dx.mode == dy.mode == 'F'
    ret = Image.new('F', dx.size)
    ret.putdata([atan(y/(x+EPSILON)) for x,y in zip(dx.getdata(), dy.getdata())]) #TODO this doesn't work because atan is not defined in imaging math
    return ret

def getGradientOrientation(im):
    """Returns an image with the gradient orientation of the image (assumed one channel)"""
    from math import atan2, atan, pi
    imx, imy = getSobelDerivs(im)
    # compute the atan at each location, scaling to [0, 255]
    xdat, ydat = imx.getdata(), imy.getdata()
    #print min(xdat), max(xdat)
    #print min(ydat), max(ydat)
    # atan2 returns results from -pi to pi, which is then remapped to between 0-255
    outdat = [remap(atan2(y-127, x-127)) for x, y in izip(xdat, ydat)]
    #print min(outdat), max(outdat)
    # create output
    out = Image.new('L', im.size)
    out.putdata(outdat)
    return out

def getEdgeLaplacianImage(im, outmode='L', imx=None, imy=None):
    """Returns the laplacian of the image: dx^2 + dy^2.
    You can set the outmode to 'L' (default) for displaying the image,
    or 'F' for float (to use it directly).
    You can give the imx and imy if you've already computed them."""
    assert outmode in 'LF'
    im = im.convert('L')
    if not imx or not imy:
        imx, imy = getSobelDerivs(im)
    ret = []
    fac = 1/(127.0*127.0)
    for dx, dy in zip(imx.getdata(), imy.getdata()):
        dx -= 127
        dy -= 127
        out = (dx*dx + dy*dy)*fac # will put it between 0-1
        if outmode == 'L':
            out = clamp(int(out*8*255), 0, 255)
        elif outmode == 'F':
            out = clamp(out, 0.0, 1.0)
        #print dx, dy, out
        ret.append(out)
    outim = Image.new(outmode, im.size)
    outim.putdata(ret)
    return outim

def getNormalizedEdgeDirImages(im, mask, outmode='L'):
    """Returns normalized edge direction images, using laplacian in the given mask to normalize"""
    assert outmode in 'LF'
    assert im.size == mask.size
    # get the sobel images
    im = im.convert('L')
    imx, imy = getSobelDerivs(im)
    # compute the average laplacian within the given mask
    lap = getEdgeLaplacianImage(im, outmode='F', imx=imx, imy=imy)
    vals = [v for v, m in zip(lap.getdata(), mask.getdata()) if m]
    avglap = sum(vals)/float(len(vals))
    # normalize vectors
    fac = 1/(127.0*127.0*avglap)
    def normdata(curim):
        data = [(v-127)*fac for v in curim.getdata()]
        if outmode == 'L':
            data = [clamp(int(v*255), 0, 255) for v in data] #TODO check this factor
        return data

    # create outputs and fill them in
    outx = Image.new(outmode, im.size)
    outy = Image.new(outmode, im.size)
    outx.putdata(normdata(imx))
    outy.putdata(normdata(imy))
    return outx, outy

def getEdgeDiffImage(im1, im2, outmode='L'):
    """Returns an image showing the edge difference between two images.
    We compute dx, dy for each image, then normalize by the sum of gradients
    across the whole image. Finally, we compute distance using L2 between
    the edges.
    You can set the outmode to 'L' (default) for displaying the image, or
    'F' for float (to use it directly)."""
    assert outmode in 'LF'
    dx1, dy1 = getSobelDerivs(im1.convert('L'))
    dx2, dy2 = getSobelDerivs(im2.convert('L'))
    fixeddata = lambda im: [(v-127)/127.0 for v in im.getdata()]
    dx1data, dy1data, dx2data, dy2data = map(fixeddata, [dx1,dy1,dx2,dy2])
    avggrad = lambda dx,dy: sum(x*x+y*y for x,y in zip(dx,dy))/len(dx)
    norm = lambda d, mean: [x/mean for x in d]
    mean1, mean2 = avggrad(dx1data, dy1data), avggrad(dx2data, dy2data)
    dx1data, dy1data, dx2data, dy2data = map(norm, (dx1data, dy1data, dx2data, dy2data), (mean1, mean1, mean2, mean2))
    d1, d2 = zip(dx1data, dy1data), zip(dx2data, dy2data)
    if outmode == 'L':
        outdata = [clamp(int(lpdist(v1,v2)*32), 0, 255) for v1, v2 in zip(d1, d2)]
        outim = Image.new('L', im1.size)
    elif outmode == 'F':
        outdata = [lpdist(v1,v2) for v1, v2 in zip(d1, d2)]
        outim = Image.new('F', im1.size)
    outim.putdata(outdata)
    return outim

def measureblur(im, mask=None):
    """Measures blur of the given image"""
    from PIL import ImageFilter
    from hist import histogram
    im = im.convert('L').filter(ImageFilter.FIND_EDGES).convert('L')
    #im = getEdgeMagIm(*getSobelDerivs(im.convert('L')))
    vals = im.getdata()
    if mask:
        if mask.size != im.size:
            mask = mask.resize(im.size, Image.NEAREST)
        vals = [v for v, m in zip(vals, mask.getdata()) if m]
    h = histogram(vals, binwidth=4.0, normalize=1.0)
    e = entropy(h.values())
    #print h
    #print e
    #im.show()
    #im.save('edge.png')
    return e


# HOG
def getFloatDerivIms(im, outs='dx dy mag dirs binimg magi'.split(), ndirs=8, prenorm=None, kernel=(1.0,0,-1.0), debug=0):
    """Returns various kinds of float derivative images, either as PIL images or numpy 2d-arrays.
    'im' should be either a single channel PIL image or a numpy 2d-array, and the resulting outputs will be of same type.
    'outs' is a list of strings denoting the outputs to return. The options are:
        'dx': The x-derivative (computed using the given 'kernel')
        'dy': The y-derivative (computed using the transpose of the given 'kernel')
        'mag': The magnitude of the derivatives: sqrt(x**2+y**2)
        'dirs': The direction of the derivatives: atan(y/x)
        'binimg': A binned version of 'dirs' with 'ndirs' bins...i.e., each location has an int in range(ndirs)
        'magi': Magnitude images masked by 'binimg'...i.e., one image per 'ndirs', with non-zero values only for that direction

    An optional 'prenorm' function can be given to apply prior to binning, callable as:
        mag, dirs = prenorm(mag, dirs)

    Note that 'mag' and 'dirs' could be either PIL Images or numpy 2d-arrays."""
    from math import pi
    import numpy
    if not outs: return []
    # get the type and bind the castim function
    t1 = time.time()
    t = 'pil' if (type(im) == type(Image.new('1', (1,1)))) else 'numpy'
    castim = float2grey if t == 'pil' else numpyarray2im
    saveim = lambda im, maxval, fname: castim(im, maxval).save(fname.rsplit('.', 1)[0] + '_%s.png' % (t))

    # get the basic derivatives
    if t == 'pil':
        s = (len(kernel), 1)
        dx, dy = getFloatDerivs(im, kernels=(kernel, kernel), ksizes=(s,tuple(reversed(s))))
    elif t == 'numpy':
        kernel = numpy.array([kernel], dtype=numpy.float)
        dx = numpy_conv2d(im, kernel)
        dy = numpy_conv2d(im, kernel.transpose())
    if debug:
        saveim(dx, 0.5, 'dx.png')
        saveim(dy, 0.5, 'dy.png')
    t2 = time.time()

    # now see which other images we need
    if 'mag' in outs or 'magi' in outs:
        # we need the magnitude image
        if t == 'pil':
            mag = getFloatEdgeMag(dx, dy)
        elif t == 'numpy':
            mag = numpy.sqrt((dx**2+dy**2))
        if debug:
            saveim(mag, 1.0, 'mag.png')
        t3 = time.time()

    if 'dirs' in outs or 'binimg' in outs or 'magi' in outs:
        # we need the directions image
        if t == 'pil':
            dirs = getFloatEdgeDirs(dx, dy)
            if debug:
                castim(dirs.point(lambda p:p*(1/pi)+0.5)).save('dirs_%s.png' % (t))
        elif t == 'numpy':
            dirs = numpy.arctan(dy/(dx+EPSILON))
            if debug:
                castim((dirs/numpy.pi)+0.5).save('dirs_%s.png' % (t))
        t4 = time.time()

    if 'binimg' in outs or 'magi' in outs:
        # we need the binimg
        M, m = pi/2, -pi/2
        if t == 'pil':
            binfunc = lambda d: min(int(ndirs*(d-m)/(M-m)), ndirs-1)
            binimg = Image.new('L', mag.size)
            binimg.putdata([binfunc(d) for d in dirs.getdata()])
        elif t == 'numpy':
            binimg = numpy.cast['int'](ndirs*(dirs-m)/(M-m))
        t5 = time.time()

    #if 'dx dy mag dirs binimg magi'
    if 'magi' in outs:
        # we need each of the magi images
        if prenorm: # prenorm if needed
            if debug:
                log("Doing a prenorm...")
            mag, dirs = prenorm(mag, dirs)
        t6 = time.time()

        # now create the magi images
        if t == 'pil':
            zeroim = Image.new('F', mag.size)
            def genmagi(i):
                """Generates the magnitude image for the i'th orientation bin"""
                mask = binimg.point(lambda p: p==i, '1')
                ret = Image.composite(mag, zeroim, mask)
                if debug:
                    saveim(ret, 0.1, 'volmag_%d.png' % (i))
                return ret

        elif t == 'numpy':
            def genmagi(i):
                """Generates the magnitude image for the i'th orientation bin"""
                ret = mag.copy()
                ret[numpy.where(binimg!=i)] = 0 # set other locations to 0
                if debug:
                    saveim(ret, 0.1, 'volmag_%d.png' % (i))
                return ret

        magi = [genmagi(i) for i in range(ndirs)]
        t7 = time.time()

    if debug:
        printTimeDifs([t1, t2, t3, t4, t5, t6, t7])

    # now assemble the outputs
    ret = [locals()[o] for o in outs]
    return ret

def getHOGBlocks(rect, blocksizes, stride=1.0):
    """Returns a set of blocks (rects), with the given (fractional) stride among blocks"""
    x0, y0, x1, y1 = rect[:4]
    ret = []
    for b in blocksizes:
        ss = int(b*stride)
        for y in range(y0, y1+1, ss):
            if y+b-1 > y1: break
            for x in range(x0, x1+1, ss):
                if x+b-1 > x1: break
                ret.append([x, y, x+b-1, y+b-1]) # -1 to make it an inclusive box
    return ret

def hogfeats(im, rects, ndirs=8, blocksizes=[64, 32, 16, 8], prenorm=None, postnorm=None, debug=0):
    """Computes of histogram-of-oriented-gradients features at the given locations.
    The 'im' can be either a single-channel float image or a float numpy array.
    Each el in 'rects' is a 4-tuple of a valid rectangle (inclusive) in the image 'im',
    at which to compute HOG features. 'ndirs' is the number of direction bins.
    'blocksizes' are the blocksizes to use for computation.
    If given, 'prenorm' and 'postnorm' are functions to normalize the magnitude and direction images
    prior to computation (takes mag, dirs as inputs), and after feature computation
    (takes feats, blocks as inputs).
    Returns an array('f') of the feature values for all rects concatenated together."""
    from math import pi
    from array import array
    # compute integral images of the magnitude in each direction
    magi = getFloatDerivIms(im, outs=['magi'], ndirs=ndirs, prenorm=prenorm, debug=debug)[0]
    intmags = [IntegralImage(m) for m in magi]

    # generate block locations and compute features
    blocks = sum([getHOGBlocks(r, blocksizes) for r in rects], [])
    feats = array('f', [m.sum(b)/area(b) for b in blocks for m in intmags])

    # post normalize, if needed
    if postnorm:
        feats = postnorm(feats, blocks)
    return feats

def testFloatEdge(im):
    fidlocs = [(100, 84), (160, 84), (130, 176)]
    rsizes = [(64, 64)]*3
    rects = [rectAt(loc, s) for loc, s in zip(fidlocs, rsizes)]
    t1 = time.time()
    im = im.resize((256,256))
    fim = grey2float(im)
    t2 = time.time()
    nim = im2numpyarray(im)
    t3 = time.time()
    func = lambda im: hogfeats(im, rects, postnorm=lambda f,b: normalize(f, power=1), debug=0)
    feats1 = func(fim)
    t4 = time.time()
    feats2 = func(nim)
    t5 = time.time()
    fdif = sum((f1-f2)**2 for f1, f2 in zip(feats1, feats2))
    log('Feat difference is %s' % (fdif,))
    log('Feats1: %s' % (getListAsStr(feats1[:10], fmt='%0.5f', sep=' ')))
    log('Feats2: %s' % (getListAsStr(feats2[:10], fmt='%0.5f', sep=' ')))
    printTimeDifs([t1,t2,t3,t4,t5])
    def pl(feats, fname):
        from pylab import plot, savefig, xlabel, ylabel, title, legend, hold
        hold(0)
        plot(feats)
        xlabel('Feat #')
        ylabel('Value')
        title('Feature Values')
        savefig(fname)
    pl(feats1, 'testfeats_pil.png')
    pl(feats2, 'testfeats_numpy.png')
    sys.exit()

def computeHog(fname):
    """Computes hogfeatures for the given filename"""
    # open and resize the image, if needed
    im = Image.open(fname).convert('L')
    imsize = (256,256)
    if im.size != imsize:
        im = im.resize(imsize, Image.ANTIALIAS)
    # set parameters
    fidlocs = [(100, 84), (160, 84), (130,176)]
    rsizes = [(64, 64)]*3
    rects = [rectAt(loc, s) for loc, s in zip(fidlocs, rsizes)]
    # compute hogfeats
    feats = hogfeats(im, rects, postnorm=lambda f,b: normalize(f, power=1))
    f = open('curhog', 'a')
    f.write('%s\n' % getListAsStr(feats, sep=' '))
    f.close()
    return feats

def lfwHogTest():
    """A simple test constructing a distance matrix for some images from LFW and using HOG features"""
    '''
    Left corner left eye:   60, 84
    Right corner left eye:  100, 84
    Left corner right eye:  150, 84
    Right corner right eye: 190, 84
    Left corner mouth:      90, 176
    Right corner mouth:     170, 176
    '''
    fidlocs = [(60, 84), (100, 84), (150, 84), (190, 84), (90, 176), (170, 176)]
    fidlocs = [(100, 84), (160, 84), (130,176)]
    rsizes = [(64, 64)]*4 + [(128,64)]*2
    rects = [rectAt(loc, s) for loc, s in zip(fidlocs, rsizes)]
    names = 'John_Allen_Muhammad Britney_Spears Adrien_Brody'.split()
    ims = ['/db/similarity/lfw/%s/%s_%04d_face0.jpg' % (n,n, i) for n in names for i in range(1, 5)]
    t1 = time.time()
    feats = []
    thumbs = []
    imsize = (100,100)
    mask = Image.new('L', (256,256), 0)
    draw = ImageDraw.Draw(mask)

    for r in rects:
        draw.rectangle(r, fill=255)
    mask.save('mask.png')
    for im in ims:
        log('  Doing %s...' % (im))
        im = Image.open(im).convert('L')
        thumbs.append(im.resize(imsize, Image.ANTIALIAS))
        #feats.append(hogfeats(im, rects, postnorm=lambda f,b: normalize(f)))
        feats.append(hogfeats(im, rects))
        #feats.append([i for i, m in zip(im.getdata(), mask.getdata()) if m])
    log('Took %0.2f secs to compute feats for %d ims, with %d rects and %d featlen' % (time.time()-t1, len(ims), len(rects), len(feats[0])))

    feats = [normalize(f, power=1) for f in feats]
    #dmat = [[intersectiondist(x,y) for x in feats] for y in feats]
    dmat = [[lpdist(x,y, 2) for x in feats] for y in feats]
    from pprint import pprint
    print ims
    from pylab import plot, savefig, xlabel, ylabel, title, legend
    print dmat
    if 0:
        for i, im in enumerate(ims):
            plot(range(len(feats[0])), feats[i], label=im.rsplit('/')[-1].replace('_face0.jpg',''))
            xlabel('Feat #')
            ylabel('Value')
            title('Feature Values')
            legend(loc='best')
        savefig('testfeats.png')
    colors = []
    maxval = max(flatten(dmat))
    print maxval
    def colorim(d):
        r = int(255*d/maxval)
        b = 255-r
        if d == 0:
            r = b = 0
        return Image.new('RGB', imsize, (r,0,b))

    mat = [[Image.new('RGB', imsize, (0,0,0))] + thumbs[:]]
    for t1, row in zip(thumbs, dmat):
        cur = [t1]
        for t2, d in zip(thumbs, row):
            cur.append(colorim(d))
        mat.append(cur)
    from imageMatrix import createImageFromImageMatrix
    outim = createImageFromImageMatrix(mat, imsize)
    outim.save('dmatrix.png')
    sys.exit()


# WATERMARKING AND OPACITY
def reduceImageOpacity(im, opacity):
    """Returns an image with reduced opacity.
    The input is not modified, and the output is always of mode 'RGBA'
    Taken from http://code.activestate.com/recipes/362879-watermark-with-pil/
    """
    from PIL import ImageEnhance
    assert opacity >= 0 and opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im

def watermarkImage(im, mark, opacity=1):
    """Adds a watermark to an image with given opacity.
    The watermark should be an RGBA image of the same size as the given image.
    Adapted from http://code.activestate.com/recipes/362879-watermark-with-pil/
    """
    from PIL import Image
    if opacity < 1:
        mark = reduceImageOpacity(mark, opacity)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    # composite the watermark with the image
    return Image.composite(mark, im, mark)

def createTextWatermark(msg, size, loc, fontcolor='white', fontpath='arial.ttf', fontsize=18):
    """Creates a watermark image of the given text.
    Puts it at the given location in an RGBA image of the given size.
    Location should be a 2-tuple denoting the center location of the text."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGBA', size, (0,0,0,0))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(fontpath, fontsize)
    tw, th = draw.textsize(msg, font=font)
    loc = (loc[0] - tw//2, loc[1] - th//2)
    draw.text(loc, msg, font=font, fill=fontcolor)
    return im

def addcaption(im, s, pos='center bottom', font='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf', fontsize=18, color='black', bg='white'):
    """Adds a caption to an image.
    The position is two words, for x, y, with options as follows:
            x       y
        ----------------
          left     top
         center
         right    bottom

    Font can be given as either an ImageFont object or a string.
    If it's a string, then it's interpreted as a path.
    You can also specify a fontsize (ignored if font is an ImageFont).
    You can also specify the color of the text to draw and the bg color.
    """
    from PIL import ImageFont, ImageDraw
    if isinstance(font, basestring):
        font = ImageFont.truetype(font, fontsize)
    size = font.getsize(s)
    if im.size[0] < size[0]:
        # offset image
        imoff = (size[0]-im.size[0])//2
    else:
        imoff = 0
    out = Image.new(im.mode, (max(im.size[0], size[0]), im.size[1]+size[1]), bg)
    draw = ImageDraw.Draw(out)
    x, y = pos.split()
    assert x in 'left center right'.split()
    assert y in 'top bottom'.split()
    if x == 'left':
        x = 0
    elif x == 'center':
        x = out.size[0]//2 - size[0]//2
    elif x == 'right':
        x = out.size[0]-size[0]
    if y == 'top':
        out.paste(im, (imoff, size[1]))
        y = 0
    elif y == 'bottom':
        out.paste(im, (imoff, 0))
        y = im.size[1]
    draw.text((x,y), s, font=font, fill=color)
    return out


# FORMAT CONVERSIONS
def float2grey(im, maxval=1.0):
    """Converts a float image to a grayscale one, normalizing by the given maxval"""
    ret = Image.new('L', im.size)
    if maxval is None:
        maxval = max(im.getdata())
    data = [clamp(v*255/maxval, 0, 255) for v in im.getdata()]
    ret.putdata(data)
    return ret

def grey2float(im, maxval=255):
    """Converts the given greyscale image to a float one"""
    assert im.mode in 'IL'
    return im.point(lambda p: p/float(maxval), 'F')

def im2array(im):
    """Converts an image to an array.
    This is a 2d one for 1-channel images, and a 3-d array for multiple-channel images"""
    if len(im.mode) > 1:
        channels = im.split()
    else:
        channels = [im]
    w, h = im.size
    ret = [list(grouper(w, c.getdata())) for c in channels]
    if len(ret) == 1:
        ret = ret[0]
    return ret

def numpyarray2im(mat, maxval=1.0, minval=0.0):
    """Converts a numpy array to a PIL grayscale image"""
    h, w = mat.shape # the matrix representation is transposed
    im = Image.new('L', (w,h))
    fac = 255.0/(maxval-minval)
    cmat = numpy.clip((mat-minval)*fac, 0, 255)
    im.putdata(map(int, cmat.flatten()))
    return im

def im2numpyarray(im, maxval=255.0, dtype=None):
    """Converts the given image to a numpy array, normalizing by maxval.
    The type is by default numpy.float or it can be user-specified."""
    import numpy
    assert im.mode in 'ILF0'
    if dtype == None:
        dtype = numpy.float
    ret = numpy.asarray(im, dtype=dtype).copy() # copy() because asarray() returns a read-only copy
    ret /= maxval
    return ret

def combineImages(ims):
    """Takes a set of 'L' images and uses them as masks to create a combined 'RGB' image.
    Each image is colored using randcolor().
    If no images are given, then returns None.
    """
    # init and quick checks
    if not ims: return None
    imsize = ims[0].size
    for im in ims:
        assert im.size == imsize
        assert im.mode == 'L'
    # create combined image
    ret = Image.new('RGB', imsize, (0, 0, 0))
    for im in ims:
        ret.paste(randcolor(ret.mode), (0, 0), im)
    return ret


# INTEGRAL IMAGES
class IntegralImage(object):
    """An integral image class.
    Uses PIL's int or float images as underlying representation currently, or numpy arrays."""
    def __init__(self, im, power=1):
        """Creates an integral image of the given image or matrix.
        The type is automatically determined from the input image's type (pil vs numpy, float vs int).
        The input image must be single-channel.
        The power parameter can be changed to 2 to store sum-of-squares, for easy variance comps."""
        if type(im) == type(Image.new('L', (1,1))): # PIL image
            assert len(im.mode) == 1
            w, h = im.size
            mode = 'F' if im.mode == 'F' else 'I'
            self.type = 'image'
            self.intim = Image.new(mode, (w+1,h+1), 0)
            src = im.load()
            self.pix = dst = self.intim.load()
            for y in range(h):
                rowsum = 0
                for x in range(w):
                    rowsum += src[x, y] ** power
                    dst[x+1, y+1] = dst[x+1, y] + rowsum
        else: # must be a numpy array
            import numpy
            w, h = im.shape
            dtype = str(im.dtype)
            assert 'int' in dtype or 'float' in dtype
            newtype = numpy.float if 'float' in dtype else numpy.uint
            self.type = 'numpy'
            self.pix = self.intim = numpy.zeros((w+1, h+1), dtype=newtype)
            whoa = numpy.cumsum(numpy.cumsum(im,0),1)
            self.intim[1:w+1, 1:h+1] = whoa

    def printrange(self, x0,x1,y0,y1):
        """Prints the given range of our internal matrix"""
        for y in range(y0,y1+1):
            for x in range(x0, x1+1):
                print self.pix[x,y],
            print

    def sum(self, rect):
        """Returns the sum of the region with the given coordinates (inclusive)"""
        x0, y0, x1, y1 = rect[:4]
        x1 += 1
        y1 += 1
        if self.type == 'numpy': # flip coords
            x0, y0, x1, y1 = y0, x0, y1, x1
        i = self.pix
        return i[x0,y0] + i[x1,y1] - i[x1,y0] - i[x0,y1]


# I/O
def floatim2color(im, valfunc=lambda f: f, color='jet'):
    """Takes a float image and returns a color image.
    For each pixel, computes a normalized value using the given valfunc.
    It then maps this to a color using the given colormap.
    The color is either a function (float -> (byte, byte, byte)) or a colormap name"""
    assert im.mode == 'F'
    outim = Image.new('RGB', im.size)
    pix = outim.load()
    if isinstance(color, basestring):
        colorfunc = lambda v: colormap(v, name=color)
    else:
        colorfunc = color
    for (x, y), v in imageiter(im):
        v = valfunc(v)
        pix[x, y] = r, g, b = validcolor(colorfunc(v))
    return outim

def writeImageToFile(im, outf, delim=' '):
    """Writes the given image to the open text file and returns the image itself.
    The image must be L or RGB. We write the width, height, then the image data
    in scanline order, all separated by the given delim. No newlines."""
    outf.write('%d%s%d' % (im.size[0], delim, im.size[1]))
    outf.flush()
    if len(im.mode) == 1:
        writefunc = lambda p: outf.write('%s%s' % (delim, p))
    else:
        writefunc = lambda p: outf.write('%s%s' % (delim, getListAsStr(p, sep=delim)))
    for p in im.getdata():
        writefunc(p)
    outf.write('\n')
    outf.flush()
    return im

def readImageFromFile(f, delim=' '):
    """Reads an image from the open text file.
    Reverses writeImageToFile(), by simply reading one line from f
    and splitting it by delim, parsing out the w, h and constructing the image."""
    els = map(int, f.readline().strip().split(delim))
    size = w, h = els[:2]
    els = els[2:]
    if len(els) == w*h:
        im = Image.new('L', size)
    elif len(els) == 3*w*h:
        im = Image.new('RGB', size)
        els = grouper(3, els)
    else:
        raise NotImplementedError('Read width %d and height %d for total of %d els (b/w image) or %d els (rgb image), but instead found %d els' % (w, h, w*h, 3*w*h, len(els)))
    im.putdata(els)
    return im

def matplotlibfig2pil(fig, dpi=100):
    """Converts the given matplotlib figure to a PIL image.
    Note that this returns an RGBA image.
    Taken from http://sebsauvage.net/python/snyppets/#argb_to_rgba
    """
    import matplotlib
    from PIL import Image
    # Ask matplotlib to render the figure to a bitmap using the Agg backend
    fig.set_dpi(dpi)
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    canvas.draw()

    # Get the buffer from the bitmap
    stringImage = canvas.tostring_argb()

    # Convert the buffer from ARGB to RGBA:
    tempBuffer = [None]*len(stringImage) # Create an empty array of the same size as stringImage
    tempBuffer[0::4] = stringImage[1::4]
    tempBuffer[1::4] = stringImage[2::4]
    tempBuffer[2::4] = stringImage[3::4]
    tempBuffer[3::4] = stringImage[0::4]
    stringImage = ''.join(tempBuffer)

    # Convert the RGBA buffer to a PIL Image
    #print dir(canvas.figure.bbox), 'get_bounds' in dir(canvas.figure.bbox)
    try:
        l,b,w,h = canvas.figure.bbox.get_bounds()
    except AttributeError:
        l,b,w,h = canvas.figure.bbox.bounds
    im = Image.fromstring("RGBA", (int(w),int(h)), stringImage)
    return im

def saveTransparentGIF(im, outname):
    """Takes an rgba image and saves a transparent gif.

    Taken from http://sebsauvage.net/python/snyppets/#argb_to_rgba
    """
    im = im.convert('RGB').convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
    # PIL ADAPTIVE palette uses the first color index (0) for the white (RGB=255,255,255),
    # so we use color index 0 as the transparent color.
    im.info["transparency"] = 0
    im.save(outname, transparency=im.info["transparency"])

def makegif(ims, fname):
    """Makes a gif from the given set of images"""
    import gifmaker
    fp = open(fname, 'wb')
    gifmaker.makedelta(fp, ims)
    fp.close()

def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags
    From https://gist.github.com/983821
    """
    from PIL.ExifTags import TAGS, GPSTAGS
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def _convert_to_degrees(value):
    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)
    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)
    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
    lat = None
    lon = None

    def _get_if_exist(data, key):
        if key in data:
            return data[key]
        return None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degrees(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat
            lon = _convert_to_degrees(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon
    return lat, lon

def getgps(im):
    """Returns a GPS (lat, lon) pair from an image or image filename"""
    if isinstance(im, basestring):
        im = Image.open(im)
    exif_data = get_exif_data(im)
    loc = get_lat_lon(exif_data)
    return loc


# TILING
class ImagePair(object):
    """Simple wrapper for making images out of pairs.
    This allows for easy indexing of points across both images.
    In particular, it's optimized for drawing lines across them.
    """
    def __init__(self, im1, im2, orientation='vertical', background=(0,0,0)):
        """Creates the image pair with the given orientation.
        This must be one of 'vertical', 'horizontal'
        You can optionally set:
            background: background color (else black)
        The pair image is 'outim', and it is always of mode RGB
        """
        # set params
        assert orientation in 'vertical horizontal'.split()
        self.ims = [im1, im2]
        self.im1, self.im2 = im1, im2
        self.orientation = orientation
        self.sizes = (w1, h1), (w2, h2) = [im1.size, im2.size]
        self.offsets = [(0,0)]
        if orientation == 'vertical':
            self.offsets.append((0,h1))
            outsize = (max(w1, w2), h1+h2)
        else:
            self.offsets.append((w1, 0))
            outsize = (w1+w2, max(h1, h2))
        # create output image and fill it in
        out = self.outim = Image.new('RGB', outsize, background)
        out.paste(self.im1, self.offsets[0])
        out.paste(self.im2, self.offsets[1])

    def globalcoords(self, imidx, loc):
        """Converts image local coordinates to a global one.
        Returns a tuple (global x, global y)."""
        off = self.offsets[imidx]
        ret = (loc[0]+off[0], loc[1]+off[1])
        return ret

    def localcoords(self, loc):
        """Converts a global location to a local one.
        Negative coordinates are okay; they measure from bottom-right corner
        Returns (imidx, local loc) or None on error."""
        x, y = loc
        w, h = self.outim.size
        if x < 0: x += w
        if y < 0: y += h
        if x < 0 or y < 0 or x >= w or y >= h: return None
        #TODO this is only valid for pairs, not general matrices
        off = self.offsets[1]
        if self.orientation == 'vertical':
            imidx = int(y >= off[1])
        else:
            imidx = int(x >= off[0])
        if imidx == 0:
            return (0, loc)
        else:
            return (1, (loc[0]-off[0], loc[1]-off[1]))

    def drawline(self, locs, **kw):
        """Draws a line between the given locations.
        If the locations are of length 2, then we assume they are global coords.
        If the locations are of length 3, then we assume each is (imidx, local x, local y).
        The kw are passed to PIL's ImageDraw.line() func.
        """
        from PIL import ImageDraw
        def fix(l):
            if len(l) == 2:
                l = l
            elif len(l) == 3:
                l = self.globalcoords(l[0], l[1:3])
            else:
                raise NotImplementedError
            return l

        locs = [fix(l) for l in locs]
        draw = ImageDraw.Draw(self.outim)
        draw.line(locs, **kw)

def tile(fnames, resize=(64,64), textonly=0, rows=None, cols=None):
    """Tiles the given images (by filename) and returns a tiled image"""
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

def ims2matrix(ims, size, replace=None, xpad=0, ypad=0, fill=(0,0,0)):
    """Creates an image from a matrix and returns it.
    Params:
            ims: 2-d array of Images
           size: size of each image
        replace: optional function to replace pixel values
           xpad: pixels between each column
           ypad: pixels between each row
           fill: background color
    Returns the final image.
    """
    def repfunc(p):
        """Replaces pixel value p with something else"""
        if p == replace[0]: return replace[1]
        return p

    numw = max([len(row) for row in ims])
    numh = len(ims)
    w, h = size
    print numw, w, xpad, 'numh', numh, h, ypad, fill
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

def imfnames2matrix(files, size, thumbnail=1, **kw):
    """Simple wrapper for ims2matrix that takes filenames instead of images.
    Each file is opened and resized to the given size.
    If either element of size is a float, then that's the factor applied to the size
    of the first image.
    If thumbnail=0, then the image is simply resized.
    Else (default), the image is thumbnailed (aspect ratio preserved).
    ims2matrix is then called with the given kw args.
    """
    size = list(size[:])
    assert min(size) > 0
    def openfunc(f):
        """Opens the image and resizes/thumbnails, as wished."""
        try:
            im = Image.open(f)
            im.load()
            for i in range(2):
                if isinstance(size[i], float):
                    size[i] = int(im.size[i]*size[i])
            if thumbnail:
                im.thumbnail(size, Image.ANTIALIAS)
            else:
                im = im.resize(size, Image.ANTIALIAS)
        except IOError, e:
            log('Error opening file %s: %s' % (f, e))
            return None
        return im

    ims = [[openfunc(f) for f in row] for row in files]
    ims = [[im for im in row if im] for row in ims]
    return ims2matrix(ims, size, **kw)


# SPECIFIC IMAGE GENERATION
def createRadialMask(imsize):
    """Create a mask image which drops off linearly from the center"""
    center = (imsize[0]/2.0, imsize[1]/2.0)
    mask = Image.new('L', imsize)
    mpix = mask.load()
    maxdist = lpdist((0,0), center, 2)
    for loc in imagelociter(mask):
        dist = lpdist(loc, center, 2)
        val = 255 - int(255*dist/maxdist)
        mpix[loc] = val
    return mask


# PROGRESS IMAGE GENERATOR
class ProgressImager(object):
    """A class to create and update progress images.
    These are images with tiles that are colored to show progress of tasks
    through a set of steps. Although we could create dynamic displays on
    webpages, it's usually much cheaper to just generate it statically and
    display online, especially for large datasets.
    """
    def __init__(self, width=None, height=None, cellsize=(5,5), colors=None, background='transparent'):
        """Creates a new progress imager.
        The generated image has an output size (in cells) specified by 'width'
        or 'height'.  If neither is specified, we make square images. If only
        one is specified, the other is automatically determined. And if both
        are specified, and too small, then they're both scaled proportionally
        to maintain the same aspect ratio.  The 'cellsize' determines how big
        each individual element 'tile' is.  The 'colors' should be a dict
        mapping progress states to colors.  The 'background' should either be a
        color, or the string 'transparent' for a transparent image.

        The way to use this is to manipulate the 'data' instance variable.
        Add or modify elements in it. They should be strings, either mapping to
        a state name in 'colors', or a colorname.
        """
        # initialize parameters
        self.width, self.height = width, height
        self.cellsize = cellsize
        if not colors:
            colors = {}
        self.colors, self.background = colors, background
        # initialize data
        self.data = []
        #TODO make a function get the legend

    def __len__(self):
        """Returns the current number of elements we're tracking"""
        return len(self.data)

    def gen(self):
        """Generates the progress image based on current parameters.
        If the background is 'transparent', then creates an RGBA image.
        Else, creates an RGB image.
        If no data, then returns None."""
        import math
        if not self.data: return None
        # figure out cell dimensions
        n = len(self)
        w, h = 0, 0
        if not self.width and not self.height:
            # neither specified: square image
            w = h = int(math.ceil(math.sqrt(n)))
        if self.width and not self.height:
            # only width specified: figure out height
            w = self.width
            h = int(math.ceil(n/float(w)))
        if self.height and not self.width:
            # only height specified: figure out width
            h = self.height
            w = int(math.ceil(n/float(h)))
        if self.height and self.width:
            # both specified: make sure it's large enough
            w, h = self.width, self.height
            if w*h < n:
                ar = w/float(h)
                w = int(math.ceil(math.sqrt(n*ar)))
                h = int(math.ceil(n/float(w)))
        w, h = int(w), int(h)
        assert w*h >= n
        # create the image
        cw, ch = self.cellsize
        size = (w*cw, h*ch)
        bg = 'white' if self.background == 'transparent' else self.background
        im = Image.new('RGB', size, bg)
        mask = Image.new('L', size, 0)
        #print self.width, self.height, w, h, n, size, im.mode
        # fill out the image
        pix = im.load()
        i = 0
        for y in range(h):
            for x in range(w):
                v = self.data[i]
                c = self.colors.get(v, v)
                box = (x*cw, y*ch, (x+1)*cw, (y+1)*ch)
                try:
                    im.paste(c, box)
                    mask.paste(255, box)
                except Exception:
                    pass
                i += 1
                if i >= n: break
            if i >= n: break
        if self.background == 'transparent':
            im = Image.merge('RGBA', list(im.split())+[mask])
        return im


# SAMPLE MAINS
def printcolors(num, fmt='spaces'):
    """Prints a list of colors"""
    num = int(num)
    for i in range(num):
        rgb = indexedcolor(i, num)
        print fmtcolor(rgb, fmt)
    sys.exit()

def testhsv():
    """Main function to test hsv problems"""
    from PIL import ImageFilter
    fnames = sys.argv[1:]
    fname = fnames[0]
    i = 0
    basemem = procmem()
    print 'Basemem is %s, using image %s of size %s' % (basemem, fname, Image.open(fname).size)
    #while 1:
    for fname in fnames:
        i += 1
        im = Image.open(fname).convert('RGB')#.resize((100,100))
        t1 = time.time()
        #hsv = rgb2hsv(im) # very slow (5s on medium sized image)
        #m = im.filter(ImageFilter.FIND_EDGES).convert('L') # fast (0.22s)
        #o = getGradientOrientation(im.convert('L')) # slow (2.7s)
        t2 = time.time()
        mem = procmem() - basemem
        print 'Finished iter %d, mem: %s, elapsed: %s' % (i, mem/1024.0, t2-t1)
    sys.exit()

def testprogress(args):
    """Tests progress image generation"""
    if len(args) < 4:
        print 'Progress args: <outimg> <num> <w> <h>'
        sys.exit()
    outname = args[0]
    num, w, h = map(int, args[1:4])
    pi = ProgressImager(width=w, height=h, colors=dict(init='red'))
    for i in range(num):
        d = 'init' if i % 2 == 1 else 'blue'
        pi.data.append(d)
    im = pi.gen()
    im.save(outname)


if __name__ == '__main__':
    from pprint import pprint
    TASKS = 'printcolors sample findblur matrix tile gps exif progress'.split(' ')
    if len(sys.argv) < 2:
        print 'Usage: python %s <%s> [<args> ...]' % (sys.argv[0], '|'.join(TASKS))
        sys.exit()
    task = sys.argv[1]
    assert task in TASKS
    if task == 'printcolors':
        printcolors(*sys.argv[2:])
    if task == 'sample':
        if len(sys.argv) < 4:
            print 'Usage: python %s sample <imname> <x1,y1> [<x2,y2> ...]' % (sys.argv[0])
            sys.exit()
        im = Image.open(sys.argv[2])
        sampler = makesampler(im)
        print 'Loaded image %s (mode %s, size %s) and made a sampler' % (sys.argv[2], im.mode, im.size)
        for loc in sys.argv[3:]:
            x, y = map(float, [l.strip() for l in loc.split(',')])
            outloc, v = sampler((x,y))
            print 'For loc %s got outloc %s and v %s' % ((x,y), outloc, v)
    elif task == 'findblur':
        if len(sys.argv) < 4:
            print 'Usage: python %s findblur <mask path or None> <imname or -> [<imname> ...]' % (sys.argv[0])
            sys.exit()
        mask = sys.argv[2]
        try:
            mask = Image.open(mask)
        except Exception:
            mask = None
        fname = sys.argv[3]
        def do(fname):
            print measureblur(Image.open(fname), mask=mask)

        if fname == '-':
            # standard main loop
            stdmainloop(do)
        else:
            # process each one
            for fname in sys.argv[3:]:
                do(fname)
    elif task == 'matrix':
        w = 100
        if len(sys.argv) < 3:
            print 'Usage: python %s matrix <outname> [<width=%d> [<height=width> [<other-args> ...]]] < list of filenames, each row in one line, tab-separated' % (sys.argv[0], w)
            sys.exit()
        files = [l.strip().split('\t') for l in sys.stdin if l]
        outname = sys.argv[2]
        try:
            w = specialize(sys.argv[3])
        except Exception: pass
        try:
            h = specialize(sys.argv[4])
        except Exception:
            h = w
        kw = getKWArgsFromArgs(sys.argv[5:])
        mat = imfnames2matrix(files, (w,h), **kw)
        mat.save(outname)
    elif task == 'tile':
        ar = 1.0
        w = 100
        if len(sys.argv) < 3:
            print 'Usage: python %s tile <outname> [<aspect ratio=%f, or ncols> [<width=%d> [<height=width> [<other-args> ...]]]] < list of filenames, tab- or new-line separated' % (sys.argv[0], ar, w)
            sys.exit()
        files = sum([l.split('\t') for l in sys.stdin if l.strip()], [])
        files = [f.strip() for f in files]
        outname = sys.argv[2]
        try:
            ar = specialize(sys.argv[3])
        except Exception: pass
        try:
            w = specialize(sys.argv[4])
        except Exception: pass
        try:
            h = specialize(sys.argv[5])
        except Exception:
            h = w
        kw = getKWArgsFromArgs(sys.argv[6:])
        log('Got kw args %s' % (kw,))
        num = len(files)
        if isinstance(ar, float):
            # aspect ratio, so figure out ncols and nrows
            ncols = int(math.ceil(math.sqrt(num*ar)))
        else:
            ncols = ar
        files = list(nkgrouper(ncols, files))
        nrows = len(files)
        mat = imfnames2matrix(files, (w,h), **kw)
        mat.save(outname)
    elif task == 'gps':
        for fname in sys.argv[2:]:
            im = Image.open(fname)
            exif_data = get_exif_data(im)
            print get_lat_lon(exif_data)
    elif task == 'exif':
        for fname in sys.argv[2:]:
            im = Image.open(fname)
            exif_data = get_exif_data(im)
            print fname
            pprint(exif_data)
    elif task == 'progress':
        testprogress(sys.argv[2:])

