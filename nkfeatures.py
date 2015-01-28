#!/usr/bin/env python
"""Neeraj's generic feature extractor.
Takes lines of input on stdin and writes a single line to stdout for each one.

Input on stdin is
    <input path>\t<regions>\t<ftypes>\t<fiducials>\t<output parameters>
where:
  - <input path> is a local path (not a url),
  - <regions> is a list of regions to extract features from, separated with ::
  - <ftypes> is a list of feature types to extract from each corresponding region, separated with ::
  - <fiducials> is a list of 'key@value' pairs, joined using ::
      These are used for determining feature locations, which the regions are defined relative to.
  - <output parameters> is an optional list of 'key@value' pairs, joined using ::
      These are used for defining parameters about the output. Currently, we support:
          crop:    'x0,y0,x1,y1' rect from which to extract features from. This is
                   first cut from the image and provides the extents relative to which
                   the feature locations are assumed to be located.
                   [default: no crop]

          width:   the width to scale the cropped input to (aspect ratio NOT preserved)
                   [default: original resolution]

          height:  the height to scale the cropped input to (aspect ratio NOT preserved)
                   [default: original resolution]

          outfmt:  the output format to print on stdout. This is a standard python format string,
                   to which we'll pass a dictionary with the following fields:
                       fvec: space-separated feature vector
                       basename: input file basename
                       path: input file path
                       regions: the passed-in regions string
                       ftypes: the passed-in ftypes string
                       fiducials: the passed-in input parameters string
                       outparams: the passed-in output parameters string
                   [default: '%(fvec)s']

          errfmt:  what to print in case of error, again as a python format string.
                   The fmtdict is like in 'fmt', but without 'fvec', and containing:
                       errortype: a python exception type name
                       errormsg: the error message
                   [default: 'error']

A full input string might look like:
    FIXME

If you want to parallelize this on a single machine, it is best to use
the GNU 'parallel' program: http://www.gnu.org/software/parallel/

This has support for taking lines from stdin and spreading them across
multiple processes, which is what is required here (and can't be done
by a simple pipe to (e.g.) xargs -n1 -P10 -0 bash -c

You want to use a command like this:
    cat featureinputs.txt | parallel --pipe --block 10k -k python nkfeatures.py > features.txt

What's going is the follows:
    --pipe is spreading the stdin onto the processes (splitting by default on newlines),
           but it doesn't know how much to send to each process, and it blocks, so you also need...
    --block 10k i.e., the amount of input data to send to one process (+/- one line)
    -k makes sure the inputs are processed into outputs in the same order



Licensed under the 3-clause BSD License:

Copyright (c) 2012, Neeraj Kumar (neerajkumar.org)
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
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath)
#os.chdir(abspath) #FIXME see if this is really necessary
from itertools import *
from math import pi
from nkpylib.nkutils import *
from nkpylib.nkimageutils import *
from PIL import Image, ImageFilter, ImageDraw, ImageStat
try:
    import numpy
    NUMPY = 1
except ImportError:
    NUMPY = 0

#TODO crop image prior to applying filters/extracting feature values
#TODO add sift features
#TODO add gloh features
#TODO add multi-res histogram features
#TODO add tweaks for current histogram features
#TODO use numpy for hog feats

DEFAULT_HIST_NBINS = 64

REGIONS_FNAME = 'nkregions.json'
DEFAULT_OUTPUT_FMT = '%(fvec)s'
DEFAULT_ERROR_FMT = 'error'

EPSILON = 0.00000001

# SMALL UTILITIES
def area(rect, type='inclusive'):
    """Returns the area of the given box.
    Type is either 'inclusive' (default) or 'exclusive'"""
    x0, y0, x1, y1 = rect[:4]
    if type == 'inclusive':
        x1 += 1
        y1 += 1
    return (y1-y0)*(x1-x0)

def getkvdict(s):
    """Returns a key-value dictionary from the given string"""
    return str2kvdict(s, sep='@', dlm='::')

# EXTRACTION AND NORMALIZATION
def getPixelValues(im, mask, func=lambda p, m: p):
    """Returns the pixel values for the given image and mask, only where mask > 0.
    The function takes the pixel value and mask value, and by default returns just the pixel.
    Returns a list of channels, where each channel contains a set of features"""
    imdata = im.getdata()
    maskdata = mask.getdata()
    ret = [func(p, m) for p, m in izip(imdata, maskdata) if m > 0]
    try:
        return zip(*ret)
    except TypeError:
        return [ret]

def meanNormalize(im, mask):
    """Normalize an image's values within the given mask, equalizing the mean values"""
    # get the values and sort them into mask regions, normalizing each separately
    func = lambda p, m: (p, m)
    vals, masks = getPixelValues(im, mask, func)
    d = lists2dict(masks, vals)
    # get a dictionary of region -> mean
    means = dict([(region, getMean(rvals)) for region, rvals in d.iteritems()])
    # now actually scale the values -- we want them between 0 and roughly 255 to end with
    # if a mean is low but a value is high, we can get very high numbers, so clamp those to 510
    normvals = [clamp(v*(127.5/max(means[m], 1.0)), 0.0, 510) for v, m in izip(vals, masks)]
    return normvals

def energyNormFunc(v, mean, stddev):
    """Energy normalizes the given value.
    For normally distributed data, 95% of it should lie within 2 stddev's of the mean.
    So we want to get 95% of the data between -127 and +127, and then we'll shift it
    up by 127 to get it from 0 to 255"""
    factor = 127.5/max(2*stddev, 0.01)
    ret = (v-mean)*factor + 127.5
    return ret

def energyNormalize(im, mask):
    """Normalize an image's values within the given mask, equalizing the energy (subtract mean and divide by stddev)"""
    # get the values and sort them into mask regions, normalizing each separately
    func = lambda p, m: (p, m)
    vals, masks = getPixelValues(im, mask, func)
    d = lists2dict(masks, vals)
    # get a dictionary of region -> (mean, stddev)
    stats = dict([(region, (getMean(rvals), getStdDev(rvals))) for region, rvals in d.iteritems()])
    func = lambda v, m: energyNormFunc(v, *stats[m])
    normvals = [func(v, m) for v, m in izip(vals, masks)]
    return normvals


# MAIN CLASSES
class FeatureType(object):
    """The base class for all feature types to use from an image"""
    def __init__(self):
        pass

class HistType(FeatureType):
    """Features which are aggregates over some region.
    This includes histograms, etc."""
    def __init__(self, func, binfunc, nbins):
        """Initializes this feature type with a function to get feature values, and the histogram sizes"""
        FeatureType.__init__(self)
        self.func, self.binfunc, self.nbins = func, binfunc, nbins

    def __call__(self, vals):
        """Return a histogram from the set of input values"""
        bins = [int(clamp(self.binfunc(v), 0, self.nbins-1)) for v in vals]
        counts = [0] * self.nbins
        for v in bins:
            counts[v] += 1
        total = float(len(vals))
        # normalize
        ret = [c/total for c in counts]
        return ret

class FeatureSet(object):
    """A set of features to get"""
    @classmethod
    def expandFTypeStr(cls, fstr):
        """Expands a feature string and returns a list of unique feature strings"""
        import re
        ret = []
        for s in fstr.split(';'):
            cur = s.lower().split('.')
            colors = list(cur[0])
            norms = list(cur[1])
            aggrs = [a for a in re.split(r'(\D\d*)', cur[2]) if a]
            #ret.extend(['.'.join(c) for c in xcombine(*cur)])
            ret.extend(['.'.join(c) for c in xcombine(colors, norms, aggrs)])
        return ret

    def __init__(self, typestr):
        """Initializes this feature set using the type string given"""
        from collections import defaultdict
        self.typestr = typestr
        self.ndims = 0
        self.funcs = []
        self.types = self.__class__.expandFTypeStr(typestr)
        #log("For typestr %s, got types of %s" % (typestr, self.types))
        self.funcs = [self.parse(s) for s in self.types]
        self.times = defaultdict(float)
        self.times.update(typestr=typestr, types=self.types)

    def splitStr(self, s):
        """Splits a given string and checks for correctness.
        Also converts a support with an arg into (support char, int param)"""
        colorspaces = ['l', 'rgb', 'hsv', 'xyz', 'mo']
        norms = 'nme' # none, mean, energy
        supports = 'hpnmv' # histogram, pixel with scaling, pixel without scaling, mean, variance
        color, norm, support = s.lower().split('.')
        assert color in ''.join(colorspaces)
        assert norm in norms
        assert support[0] in supports # support can have a modifier after it to specify a param
        if len(support) > 1:
            support = (support[0], int(support[1:]))
        return (color, norm, support)

    def parse(self, s):
        """Parses a string and returns functions [colorspace, normalization, support]"""
        color, norm, support = self.splitStr(s)
        ret = []

        # first handle the color spaces
        def colorfunc(color):
            """Returns a function which returns an image (1 channel, all values from 0-255).
            Uses the cache to store expensive image conversions"""
            def colorfuncret(im, mask, cache, c=color):
                def get(idx, val, cache=cache, im=im, mask=mask):
                    if cache is None: return lazy(val)
                    if idx not in cache:
                        cache[idx] = lazy(val)
                    return cache[idx]
                if c in 'l': # convert to grayscale
                    return get('l', "im.convert('L')")
                elif c in 'rgb': # split into bands and return the appropriate one
                    return get('rgb', "im.split()")['rgb'.index(c)]
                elif c in 'hsv': # convert to hsv and return the appropriate band
                    return get('hsv', "rgb2hsv(im).split()")['hsv'.index(c)]
                elif c in 'xyz': # convert to xyz and return appropriate band
                    return get('xyz', "rgb2xyz(im).split()")['xyz'.index(c)]
                elif c == 'm': # use gradient magnitude
                    return get('m', "im.filter(ImageFilter.FIND_EDGES).convert('L')")
                elif c == 'o': # use gradient orientation
                    return get('o', "getGradientOrientation(im.convert('L'))")
            return colorfuncret

        ret.append(colorfunc(color))

        # now handle the normalization
        def normfunc(norm):
            """Returns a function which returns a sequence (all values from 0 to roughly 255.0)."""
            def normfuncret(im, mask, n=norm):
                if n == 'n': # no normalization
                    return [v for v,m in izip(im.getdata(), mask.getdata()) if m > 0]
                elif n == 'm': # mean normalization
                    return meanNormalize(im, mask)
                elif n == 'e': # energy normalization
                    return energyNormalize(im, mask)
            return normfuncret

        ret.append(normfunc(norm))

        # now handle the aggregation (either pixel, histogram, or statistics)
        def scalefunc(seq):
            mean = getMean(seq)
            stddev = getStdDev(seq)
            seq = [energyNormFunc(x, mean, stddev) for x in seq]
            return [x/255.0 for x in seq]

        identity = lambda x: x
        if support[0] == 'p': # single pixel, with scaling
            if len(support) > 1:
                # subsampling
                f = lambda seq: scalefunc(seq)[::support[1]]
            else:
                f = scalefunc
            self.ndims += 1
        elif support[0] == 'n': # single pixel, with no scaling
            if len(support) > 1:
                # subsampling
                f = lambda seq: identity(seq)[::support[1]]
            else:
                f = scalefunc
            self.ndims += 1
        elif support[0] == 'h': # histogram
            # optional nbins parameter
            nbins = DEFAULT_HIST_NBINS if len(support) == 1 else support[1]
            binsize = (256//nbins) + 1 # to make sure we have no more than nbins
            def binfunc(x, binsize=binsize):
                return x//binsize
            f = HistType(identity, binfunc, nbins)
            self.ndims += nbins
        elif support == 'm': # mean
            f = lambda seq: [getMean(seq)/255.0]
        elif support == 'v': # variance
            def varfunc(seq):
                mean = getMean(seq)
                if mean == 0:
                    var = 0
                else:
                    var = getVariance([s/(mean+EPSILON) for s in seq])
                return [var]

            f = varfunc
        ret.append(f)
        return ret

    def compute(self, im, mask, **kw):
        """Computes features for the given image and mask"""
        from array import array
        ret = array('f')
        t1 = time.time()
        cache = {}
        #TODO deal with outparams, fiducials, fmtdict (all passed in kw)
        for colorfunc, normfunc, aggrfunc in self.funcs:
            c1 = time.time()
            cim = colorfunc(im, mask, cache)
            c2 = time.time()
            #cim.save('color.png')
            vals = normfunc(cim, mask)
            c3 = time.time()
            if NUMPY:
                #vals = numpy.array(vals)
                pass
            #log("Got %s after normfunc, with mean %s, min %s, max %s, stddev %s" % (vals, getMean(vals), min(vals), max(vals), getStdDev(vals)))
            vals = aggrfunc(vals)
            c4 = time.time()
            #log("Got %s after aggrfunc, with mean %s, min %s, max %s, stddev %s" % (vals, getMean(vals), min(vals), max(vals), getStdDev(vals)))
            ret.extend(vals)
            c5 = time.time()
            self.times['colorfunc'] += c2-c1
            self.times['normfunc'] += c3-c2
            self.times['aggrfunc'] += c4-c3
            self.times['Extend ret'] += c5-c4
            #log(len(ret))
        t2 = time.time()
        self.times['Total Fset Compute Time'] += t2-t1
        return ret


class MaskFunc(object):
    """A class to generate masking functions (regions).
    Given a 'mask string', this generates a masking function.
    This function takes in an input image and returns a mask image of the same size.
    The mask image is of type '1' (binary), and is 1 where the mask is ON (valid).

    The typical usage is like this:
        mfunc = MaskFunc('+left eye;+right eye')
        for im in images:
            mask = mfunc.compute(im)

    The mask string is composed of an arbitrary number of elements joined using ';'
    where each element is of the form:
        <bool><shape>
    The <bool> is either '+' or '-' to turn that shape on or off.
    The mask is progressively iterated upon, starting with 0 (completely off),
    with each mask element modifying the mask so far.
    The <shape> is either a primitive shape or a pre-defined region (more common).

    Primitives are defined as:
        <primitive type>:<coords>
    The different primitive types and the coords they take are:
        rect: minx, miny, maxx, maxy
        poly: a series of x,y points
        oval: the bounding rect of the oval (minx, miny, maxx, maxy)
    The coords are always specified in flat comma-separated lists.
    The coordinate system is as follows:
        - the left and right edges of the image are x=-1 and x=1
            - this allows for easy symmetry, since x=0 is the middle
        - the top and bottom edges of the image are y=0 and y=1

    Pre-defined regions are defined in the config file, and are composed
    of an arbitrary number of other regions (which must eventually resolve
    down to primitives). The mask string can contain either:
        <region>
    or:
        <region>*<x-factor>,<y-factor>
    The x- and y-factors are used to scale each primitive in the region by the
    given factor in each direction, relative to the primitive's center. In the
    first form given above, both factors are assumed to be 1.

    The config file is in JSON format, with all the data under the 'regions' field.
    If there is no 'regions' field, then it tries to use the whole json structure.
    Here is an example:
    {"regions": {
        "all": {
            "comps": [{"shape": "rect", "coords": [-1,0,1,1]}],
        },
        "left arm": {
            "align": "front",
            "comps": [{"shape": "rect", "coords": [-0.1,0,-0.5,1]}]
        },
        "right arm": {
            "align": "front",
            "comps": [{"shape": "rect", "coords": [0.5,0,1,1]}]
        },
        "arms": {
            "comps": [{"region": "left arm"}, {"region": "right arm"}]
        }
    }}

    Each region is defined using a name, and contains fields:
        'align': [optional] The name of the alignment this region requires.
        'comps': A list of components this region is made of. Each component
                 is either a primitive or a reference to another region.

    The primitives are defined as:
        'shape': One of the primitive shape types
        'coords': The list of coordinates for this shape

    The references to other regions are defined as:
        'region': The name of the other region to substitute in here.

    Upon parsing, all references will be recursively expanded out to
    primitives.  Make sure there is no infinite loop!

    A single MaskFunc can only have a single 'align' type. This is to keep the
    final interface simple. You can access this using the 'align' parameter.

    """
    SHAPES = 'rect oval poly'.split()

    def __init__(self, maskstr, shapeconfig=REGIONS_FNAME):
        """Creates a new mask function using the given string to initialize it"""
        # read the shape elements
        try:
            shapedict = self.readConfig(shapeconfig)
        except Exception:
            # couldn't read shapeconfig for any reason
            shapedict = {}
        # split our maskstr into elements and simplify them
        els = []
        self.align = None # this will be overwritten later
        for s in maskstr.strip().split(';'):
            els.extend(self.simplify(s.strip(), shapedict))
        # now compute our list of shape funcs
        self.maskstr = maskstr
        self.shapes = els[:]
        self.shapefuncs = [self.getShapeFunc(e, i+1) for i, e in enumerate(els)]
        # initialize our mask cache
        self.masks = {}

    def readConfig(self, fname):
        """Reads the dictionary of shapes.
        Example:
            "right arm": {
                "align": "front",
                "comps": [{"shape": "rect", "coords": [0.5,0,1,1]}]
            },
            "arms": {
                "comps": [{"region": "left arm"}, {"region": "right arm"}]
            }
        """
        # load the json file
        try:
            import simplejson as json
        except ImportError:
            import json
        #print 'Reading shapes file %s' % (fname)
        ret = json.loads(open(fname).read(), strict=False)
        # get the subdict if wanted
        if 'regions' in ret:
            ret = ret['regions']
        # parse out
        for name, r in ret.items():
            # first expand out all references
            hadrefs = 1
            while hadrefs:
                hadrefs = 0
                newc = []
                for c in r['comps']:
                    if 'region' in c:
                        ref = ret[c['region']]
                        newc.extend(ref['comps'])
                        hadrefs = 1
                    else:
                        assert 'shape' in c
                        newc.append(c)
                r['comps'] = newc
        # at this point, all 'comps' should be primitives
        for name, r in ret.items():
            # convert shapes to a list of (shape type, coords)
            shapes = [(c['shape'], c['coords']) for c in r['comps']]
            # the return object for this name is (shapes, align type)
            ret[name] = (shapes, r.setdefault('align', None))
        return ret

    def simplify(self, maskstr, shapes):
        """Simplifies the given mask string if possible"""
        #log("Trying to simplify |%s|, with shapes %s" % (maskstr, shapes))
        assert ';' not in maskstr, 'simplify() operates on single elements only'
        # ignore the boolean flag
        bool = maskstr[0]
        maskstr = maskstr[1:]
        # it's an explicit shape, so just make sure it's a valid shape
        if ':' in maskstr:
            shape, coords = maskstr.split(':', 1)
            assert shape.strip() in self.__class__.SHAPES
            assert len(coords.split(',')) == 4
            return ['%s%s' % (bool, maskstr)]
        # it's a predefined shape
        if '*' in maskstr:
            # it's a predefined shape multiplied by some scale
            type, factors = maskstr.split('*', 1)
            factors = [float(f) for f in factors.split(',', 1)]
        else:
            type = maskstr
            factors = 1.0, 1.0
        realshapes, self.align = shapes[type]
        ret = []
        fx, fy = factors
        for type, coords in realshapes:
            # get the center of this set of coords and translate the shape
            cx, cy = getCenter(coords)
            # multiply the coords out with our given factors
            xs = [clamp(cx + fx*(c-cx), -1.0, 1.0) for i, c in enumerate(coords) if i % 2 == 0]
            ys = [clamp(cy + fy*(c-cy), 0.0, 1.0) for i, c in enumerate(coords) if i % 2 == 1]
            # write out to the output
            ret.append('%s%s:%s' % (bool, type, ','.join(['%s' % f for f in flatten(zip(xs, ys))])))
        return ret

    def getShapeFunc(self, s, fillval):
        """Returns a shape from a given (simplified) string.
        The fillvalue is used to fill in the value for this shape, if it's positive."""
        bool = s[0]
        assert bool in '+-'
        shape, coords = s[1:].split(':', 1)
        coords = [float(c) for c in coords.split(',')]
        assert len(coords) >= 4
        # convert x coordinates to be from origin, instead of center
        for i, c in enumerate(coords):
            if i % 2 != 0: continue
            coords[i] = (coords[i]+1.0)/2.0
        # the function that we'll return
        def ret(mask, c=coords, shape=shape, bool=bool, fillval=fillval):
            """Takes in a mask and "draws" the given shape onto it.
            This can either mask things in or out depending on bool.
            The params are:
                mask: the mask image (binary) to modify
                c: the (origin-based) normalized coordinates of the shape to draw
                shape: the shape to draw (string)
                bool: '+' or '-' on whether to add or subtract an area from the mask
                fillval: the numerical value to fill in for "positive" (visible) areas.
                         You should set this to 1 or 255 depending on mask.mode
            """
            # expand coords out to image size
            w, h = mask.size
            coords = tuple([int(c*d) for c, d in zip(c, cycle([w, h]))])
            # define some convenience dicts
            draw = ImageDraw.Draw(mask)
            funcdict = dict(rect=draw.rectangle, oval=draw.ellipse, poly=draw.polygon)
            booldict = {'+': fillval, '-': 0}
            # draw the shape
            func = funcdict[shape]
            val = booldict[bool]
            func(coords, fill=val, outline=val)
            return mask

        return ret

    @classmethod
    def rectFromMaskIm(cls, im):
        """Returns a rect mask string from a given mask image."""
        locs = [l for l, val in imageiter(im) if val]
        xs, ys = zip(*locs)
        # normalize x/y coords
        w2 = im.size[0]//2
        xs = [(x-w2)/float(w2) for x in xs]
        ys = [y/float(im.size[1]) for y in ys]
        # get the rect
        rect = (min(xs), min(ys), max(xs), max(ys))
        return '+rect:%s,%s,%s,%s' % (rect)

    def compute(self, im, **kw):
        """Returns a mask for the given image.
        For every input image size passed in, we cache the output mask.
        So repeated calls with the same input size will be very fast.
        """
        # cache if needed
        if im.size not in self.masks:
            m = Image.new('1', im.size, 0)
            for func in self.shapefuncs:
                m = func(m)
            self.masks[im.size] = m
        # return from cache
        return self.masks[im.size]


class FeatureComputer(object):
    """A class to compute features from images"""
    def __init__(self, mstr, fstr):
        """Initializes this feature computer"""
        from collections import defaultdict
        from nkalign import Aligner
        # create a feature set and mask function
        self.mstr, self.fstr = mstr, fstr
        self.fsets = [FeatureSet(f) for f in fstr.split('::') if f]
        self.mfuncs = [MaskFunc(m, REGIONS_FNAME) for m in mstr.split('::') if m]
        #print mstr, fstr, self.mfuncs
        # create aligners
        aligntypes = sorted(set(m.align for m in self.mfuncs))
        aligners = [Aligner(name=t) for t in aligntypes]
        self.aligners = dict(zip(aligntypes, aligners))
        # keep track of timings both within ourselves and in our fsets
        self.times = defaultdict(float)
        self.times.update(mstr=mstr, fstr=fstr)
        for i, f in enumerate(self.fsets):
            self.times['fset_%d' % i] = f.times

    def __repr__(self):
        """Returns an "identifier" for this feature computer, which include mstr, fstr"""
        ret = makesafe('%s__%s' % (self.mstr, self.fstr))
        return ret

    def compute(self, fname, **kw):
        """Computes the features for a single file.
        The input fname should already exist and be a legal image.
        The kw are passed to each aligner, each MaskFunc, and each FeatureSet.
        Currently, this returns an array.array('f')
        If there's an error, returns None."""
        from array import array
        results = array('f')
        self.times['Images Attempted'] += 1
        try:
            # open image
            t0 = time.time()
            im = Image.open(fname).convert('RGB')
            t1 = time.time()
            self.times['Image Open'] += t1-t0
            # align image for each aligner
            aligned = dict((t, a.align(im, **kw)[0]) for t, a in self.aligners.items())
            t2 = time.time()
            self.times['Image Align'] += t2-t1
            for mfunc, fset in zip(self.mfuncs, self.fsets):
                im = aligned[mfunc.align]
                c1 = time.time()
                mask = mfunc.compute(im, **kw)
                c2 = time.time()
                #mask.save('mask.png')
                cur = fset.compute(im, mask, **kw)
                c3 = time.time()
                results.extend(cur)
                c4 = time.time()
                self.times['Compute Mask'] += c2-c1
                self.times['Compute Fset'] += c3-c2
                self.times['Concat Feats'] += c4-c3
            t3 = time.time()
            self.times['Compute Feats'] += t3-t2
            self.times['Total Feature Computation'] += t3-t0
        except (IOError, IndexError): return None
        self.times['Images Done'] += 1
        return results


def processImage(path, regions, ftypes, fiducials, outparams, fmtdict):
    """Processes a single image and returns a feature vector, or raises an Exception"""
    fc = FeatureComputer(mstr=regions, fstr=ftypes)
    fvec = fc.compute(path, fiducials=fiducials, outparams=outparams, fmtdict=fmtdict)
    return fvec

def processLine(line):
    """Process a single line of input, returning a single line of output as a string."""
    #<input path>\t<regions>\t<ftypes>\t<input parameters>\t<output parameters>
    from collections import defaultdict
    from nkalign import parseFiducials
    fmtdict = defaultdict(str)
    DEFAULT_OUTPARAMS = defaultdict(str)
    DEFAULT_OUTPARAMS['outfmt'] = DEFAULT_OUTPUT_FMT
    DEFAULT_OUTPARAMS['errfmt'] = DEFAULT_ERROR_FMT

    # parse elements
    els = line.split('\t')
    try:
        # output params
        outparams = dict(**DEFAULT_OUTPARAMS)
        # input
        fmtdict['path'] = path = els.pop(0)
        fmtdict['basename'] = basename = os.path.basename(path)
        #print path, basename, fmtdict
        # regions and ftypes
        fmtdict['regions'] = regions = els.pop(0)
        fmtdict['ftypes'] = ftypes = els.pop(0)
        #print regions, ftypes
        # fiducials
        fmtdict['fiducials'] = fiducials = els.pop(0)
        fiducials = parseFiducials(fiducials)
        #print fiducials
        #print outparams
        if els:
            # output params are optional, so we don't want to raise an exception here
            fmtdict['outparams'] = els.pop(0)
            #print fmtdict['outparams']
            outparams.update(getkvdict(fmtdict['outparams']))
            #print outparams

        # at this stage, we have everything we need
        # first make sure the file exists
        if not os.path.exists(path): raise IOError('Image does not exist')
        # process the image
        fvec = processImage(path, regions, ftypes, fiducials=fiducials, outparams=outparams, fmtdict=fmtdict)
        # return the output
        fmtdict['fvec'] = getListAsStr(fvec, sep=' ')
        outparams['outfmt'] = outparams['outfmt'].replace(r'\t', '\t').replace(r'\n', '\n')
        #print 'Got fmtdict of %s and outparams %s' % (fmtdict, outparams)
        ret = outparams['outfmt'] % (fmtdict)
        return ret
    except Exception, e:
        import traceback
        print >>sys.stderr, 'Exception parsing input line %s:' % (els,)
        traceback.print_exc(file=sys.stderr)

        # add the error values to the fmtdict
        fmtdict['errortype'] = type(e).__name__
        try:
            fmtdict['errormsg'] = e
        except Exception:
            pass
        # generate and return the error string
        outparams['errfmt'] = outparams['errfmt'].replace(r'\t', '\t').replace(r'\n', '\n')
        errstr = outparams['errfmt'] % fmtdict
        return errstr


def mainloop():
    """The real main loop"""
    global REGIONS_FNAME
    if len(sys.argv) < 1:
        print "Usage: python %s [<regions file>=%s]" % (sys.argv[0], REGIONS_FNAME)
        sys.exit()
    try:
        REGIONS_FNAME = sys.argv[1]
    except Exception:
        pass

    def do(line):
        try:
            print processLine(line)
            sys.stdout.flush()
        except IOError:
            pass

    stdmainloop(do)

def dumpmasks():
    """Simple main to dumps masks. Argv should contain, optionally: width, height"""
    w, h = 500, 500
    try:
        w = int(sys.argv[1])
        h = int(sys.argv[2])
    except Exception: pass
    im = Image.new('RGB', (w, h))
    for l in sys.stdin:
        mstr, outpath = l.strip().split('\t')
        mf = MaskFunc(mstr)
        m = mf.compute(im)
        m.save(outpath)
        print 'Saved mask for %s to %s' % (mstr, outpath)


if __name__ == "__main__":
    mainloop()
    #dumpmasks()
