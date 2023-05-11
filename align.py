#!/usr/bin/env python
"""Neeraj's generic image aligner.
Usage: python %s
Some information is written to the saved image, in the UserComment, in our usual format.
This includes:
  - all the parameters passed in via <parameters>,
  - <METHOD>_TIMESTAMP@seconds_since_the_epoch, where <METHOD> is the capitalized method,
  - when <method> is 'oldaffine', 'affine', 'similar', or 'simple', the matrix of the
    transformation, with tag AFFINE.


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

import math, os, sys, time
from nkpylib.utils import *
from nkpylib.exif import getDict, setDict
from PIL import Image, ImageDraw
import numpy as np
assert sys.platform.startswith('linux') # NO WINDOWS OR MAC!
try:
    import simplejson as json
except ImportError:
    import json

# CONFIG PARAMS
ALIGN_CONFIG_FNAME = 'nkaligners.json'
DEFAULT_OUTPUT_FMT = '%(inpath)s\t%(outpath)s'
DEFAULT_ERROR_FMT = 'error'
JPEG_QUALITY = 95
INTERPOLATION = Image.BICUBIC
DEFAULT_DTYPE = np.double


# SMALL UTILITY FUNCTIONS
def md5sum(s):
    """MD5s the given string of data and returns the hexdigest.
    If you want the md5 of a file, call md5sum(open(fname).read())"""
    import hashlib
    return hashlib.md5(s).hexdigest().lower()

def getFidsFromDicts(indict, outdict, outsize=(1,1), dtype=DEFAULT_DTYPE):
    """Returns Nx3 matrices for input and output fiducial points.
    If indict and outdict are dicts, then uses keys that exist in both indict
    and outdict, in sorted order.
    Else, assumes they're sequences of fiducials, and checks that they're same length.
    The outputs are in homogenous coordinates, i.e., with last coordinate = 1.
    The output points are scaled by the width and height in outsize.
    If outsize is None, then outsize is assumed to be (1,1)
    You can optionally specify what datatype to use for the matrices.
    Returns (inpts, outpts), each as a np.array()
    """
    if isinstance(indict, dict) and isinstance(outdict, dict):
        keys = sorted(set(indict) & set(outdict))
    else:
        assert len(indict) == len(outdict)
        keys = range(len(indict))
    n = len(keys)
    inpts = np.ones((n, 3), dtype=dtype)
    outpts = np.ones((n, 3), dtype=dtype)
    for i, k in enumerate(keys):
        inpts[i, :2] = indict[k][:2]
        outpts[i, :2] = outdict[k][:2]
    if outsize is not None:
        outpts[:,0] *= outsize[0]
        outpts[:,1] *= outsize[1]
    return (inpts, outpts)

def transformPts(trans, pts, dtype=DEFAULT_DTYPE):
    """Transforms the given points using the given transform.
    The pts should be a list of pairs or triples (homogenous).
    Returns a numpy Nx2 array of results"""
    p = np.ones((len(pts), 3), dtype=dtype)
    for i, pt in enumerate(pts):
        if len(pt) == 2:
            p[i, :2] = pt[:2]
        elif len(pt) == 3:
            p[i, :3] = pt[:3]
        else:
            raise ValueError('Pt length must be 2 or 3')
    ret = np.dot(p, trans)
    last = ret[:, 2]
    ret /= last.reshape((len(pts), 1))
    return ret[:, :2]


# TRANFORMATIONS
def getSimpleTransform(indict, outdict, outsize=None):
    """Mapping from input points to output points, as only a translation + uniform scaling"""
    #print "Got indict %s, outdict %s, outsize %s" % (indict, outdict, outsize)
    def eyecen(d):
        cens = [d[k] for k in ('LEFT_EYE_OUT', 'LEFT_EYE_IN', 'RIGHT_EYE_IN', 'RIGHT_EYE_OUT')]
        cenx, ceny = zip(*cens)
        cenx, ceny = sum(cenx)/4.0, sum(ceny)/4.0
        w = cens[-1][0] - cens[0][0]
        #print cens, cenx, ceny, w
        return (cenx, ceny), w

    eye_in, w_in = eyecen(indict)
    eye_out, w_out = eyecen(outdict)
    eye_out = [e*o for e, o in zip(eye_out, outsize)]
    #print eye_in, eye_out, w_in, w_out
    if w_in == 0:
        w_in = w_out
    s = w_out/w_in
    t = [(o-i*s) for i, o in zip(eye_in, eye_out)]
    #print s, t
    ret = np.array([[s, 0, 0], [0, s, 0], [t[0], t[1], 1.0]])
    return ret

def getSimilarityTransform(indict, outdict, outsize=None):
    """Returns a similarity transformation to go from input pts to output pts.
    'indict' and 'outdict' should contain identical keys mapping to 2-tuples.
    A similarity transformation matrix has the form
            [k cos  -k sin   dx]          [a  -b   c]
      T  =  [k sin   k cos   dy]  or just [b   a   d]
            [  0       0      1]          [0   0   1].
    Then, for each (xin, yin), (xout, yout) correspondence, we have
        [xin]     [xout]
      T [yin]  =  [yout]
        [ 1 ]     [ 1  ]
    which is equivalent to
                         [a]
      [xin  -yin  1  0]  [b]     [xout]
      [yin   xin  0  1]  [c]  =  [yout].
                         [d]
    We stack up the leftmost and rightmost bits to have 2 * number_of_correspondences rows,
    then find the least squares solution for (a, b, c, d), then build T.  Transformations
    in this code are given as the transpose of this form (so that xt A = xt' rather than A x = x'),
    so we return (T.transpose(), Ax), where the 2nd term is the transformed locations of the inputs."""
    # indict  = {'x1': (1,1), 'x2': (3,1), 'x3': (4,1), 'x4': (6,1), 'x5': (2,5), 'x6': (5,5)}
    # outdict = {'x1': (3,2), 'x2': (5,2), 'x3': (6,2), 'x4': (8,2), 'x5': (4,6), 'x6': (7,6)}
    A = np.zeros((len(outdict)*2, 4), dtype=np.double)
    B = np.zeros((len(outdict)*2, 1), dtype=np.double)
    inpts, outpts = getFidsFromDicts(indict, outdict, outsize)
    for i, (pin, pout) in enumerate(zip(inpts, outpts)):
        A[(2*i),:] = [pin[0], -pin[1], 1, 0]
        B[(2*i), 0] = pout[0]
        A[(2*i+1),:] = [pin[1], pin[0], 0, 1]
        B[(2*i+1), 0] = pout[1]

    # multiply by A transpose on both sides
    At = A.transpose()
    left = np.dot(At, A)
    right = np.dot(At, B)
    # linear least squares solve for the transform, x
    x, resids, rank, s = np.linalg.lstsq(left, right)

    # Transformation matrix is [[a, -b, c], [b, a, d], [0, 0, 1]].
    a = x[0,0]
    b = x[1,0]
    c = x[2,0]
    d = x[3,0]
    T = np.zeros((3, 3), dtype=np.double)
    T[:,:] = [[a, -b, c], [b, a, d], [0, 0, 1]];

    # the other functions expect the transpose of this matrix
    ret = T.transpose()
    Ax = np.dot(inpts, ret)
    return ret, Ax

def getAffineTransform(indict, outdict, outsize=None):
    """Returns a transformation to go from input pts to output pts.
    'indict' and 'outdict' should contain identical keys mapping to 2-tuples.
    Each point is homogenized, then a linear least squares solution to A'Ax=A'B is found,
    where A = inputs, B = outputs, x = affine transformation.
    Returns (x, Ax), where the 2nd term is the transformed locations of the inputs."""
    A, B = getFidsFromDicts(indict, outdict, outsize=outsize)
    # multiply by A transpose on both sides
    At = A.transpose()
    left = np.dot(At, A)
    right = np.dot(At, B)
    # linear least squares solve for the transform, x
    x, resids, rank, s = np.linalg.lstsq(left, right)
    Ax = np.dot(A, x)
    return x, Ax

def getHomography(indict, outdict, outsize=None):
    """Returns a transformation to go from input pts to output pts using a homography.
    'indict' and 'outdict' should contain identical keys mapping to 2-tuples.
    We create A:
        x1 y1 1 0  0  0 -x1*x1' -y1*x1'
        0  0  0 x1 y1 1 -x1*y1' -y1*y1'
        x2 y2 1 0  0  0 -x2*x2' -y2*x2'
        0  0  0 x2 y2 1 -x2*y2' -y2*y2'
        ...
    And b:
        [x1' y1' x2' y2' x3' y3' ...].T
    Then solve for h in Ah = b using linear least squares, where h is:
        [h11 h12 h13 h21 h22 h23 h31 h32].T
    and h33 is 1.
    Returns (h, Ah), where the 2nd term is the transformed locations of the inputs.
    """
    # initialize both matrices
    A = np.zeros((2*len(outdict), 8), dtype=np.double)
    b = np.zeros((2*len(outdict), 1), dtype=np.double)
    inputs, outputs = getFidsFromDicts(indict, outdict, outsize=outsize)
    # copy over data
    for i, ((xi, yi, _), (xo, yo, _)) in enumerate(zip(inputs, outputs)):
        A[2*i,:]    = [xi, yi, 1, 0, 0, 0, -xi*xo, -yi*xo]
        A[2*i+1, :] = [0, 0, 0, xi, yi, 1, -xi*yo, -yi*yo]
        b[2*i]      = xo
        b[2*i+1]    = yo
    #print A, A.shape, b, b.shape, inputs, inputs.shape
    # Linear least squares solve
    h, resids, rank, s = np.linalg.lstsq(A, b)
    h = h.flatten()
    ret = np.ones((3,3), dtype=np.double)
    ret[:, :] = [h[:3], h[3:6], [h[6], h[7], 1.0]]
    ret = ret.transpose()
    # we need transposed version of h throughout
    ah = np.dot(inputs, ret)
    ah /= ah[:, -1:]
    if 0:
        print(h, len(h))
        print('ret\n', ret, ret.shape)
        print('normed ah\n', ah, ah.shape)
        print('outputs\n', outputs)
        print('inputs\n', inputs)
        print('diff %\n', 100.0*(outputs-ah)/outputs)
    return ret, ah

def applyTransform(im, transform, outsize):
    """Apply the given transform (3 x 3 Numpy array) to the given image, cropping the image to the
    given size (a (width, height) 2-tuple)."""
    # compute the inverse transform and transpose it
    inv = np.linalg.inv(transform)
    if 0: # Old code for affine transformations only
        affine = inv.transpose()[:2, :]
        out = im.transform(outsize, Image.AFFINE, affine.flatten(), INTERPOLATION)
    else: # full perspective transformations (note that PERSPECTIVE exists in PIL!)
        homography = inv.transpose().flatten()[:-1]
        out = im.transform(outsize, Image.PERSPECTIVE, homography, INTERPOLATION)
    return out

TRANSFORM_TYPES = dict(simple=getSimpleTransform, similarity=getSimilarityTransform, affine=getAffineTransform, homography=getHomography)


# OUTPUT FUNCTIONS
def drawfids(im, fids, params):
    """Draws fiducials on the output image, depending on 'params'"""
    from PIL import ImageDraw
    # figure out what kind of fids we want to draw
    drawtype = params.get('drawfids', 'none')
    if drawtype == 'none': return im
    assert drawtype in 'circ circle centrect centrectangle rect rectangle'.split()
    # get other params
    fill = params.get('drawfidsfill', 'green')
    line = params.get('drawfidsline', 'green')
    r = int(params.get('drawfidsr', 3))
    # draw the points
    draw = ImageDraw.Draw(im)
    for f in fids:
        x, y = int(round(f[0])), int(round(f[1]))
        if drawtype in 'circ circle'.split():
            draw.ellipse((x-r, y-r, x+r, y+r), outline=line, fill=fill)
        elif drawtype in 'centrect centrectangle'.split():
            draw.rectangle((x-r, y-r, x+r, y+r), outline=line, fill=fill)
        elif drawtype in 'rect rectangle'.split():
            x2, y2 = int(round(f[2])), int(round(f[3]))
            draw.rectangle((x, y, x2, y2), outline=line, fill=fill)
    return im


def printmat(m):
    """Returns a string for a given matrix."""
    s = ''
    for row in m:
        s += '[%s]' % ' '.join('%f' % v for v in row)
    return s

def saveImage(im, path, tags=None):
    """Save the given image to the given path, optionally saving 'tags' in the
    EXIF UserComment in our usual way.
    If tags is given, then the format is JPEG regardless of the filename, since
    only JPEGs have EXIF.  Else, it's determined by the filename.
    """
    if not os.path.basename(path): raise Exception('Invalid save path %s.' % path)
    try:
        os.makedirs(os.path.dirname(path))
    except OSError: pass
    try:
        kw = {}
        if JPEG_QUALITY:
            kw['quality'] = JPEG_QUALITY
        if tags:
            kw['format'] = 'JPEG'
        im.save(path, **kw)
    except IOError as ex:
        raise Exception("Can't save to path %s.  %s" % (path, ex.strerror))
    if tags:
        setDict(path, tags)


# MAIN ALIGNER CLASS
class Aligner(object):
    """A class to align images in different ways.
    Given an 'aligner name', this generates an alignment function.
    This function takes in an input image (and optionally some parameters),
    and returns an aligned image.

    The typical usage is like this:
        aligner = Aligner('left arm')
        for im, inp, outp in zip(images, fiducials, outparams):
            aligned, params = aligner.align(im, fiducials=inp, outparams=outp)

    The aligner names are defined in a config file in JSON format, with all the
    data under the 'aligners' field.  If there is no 'aligners' field, then it
    tries to use the whole json structure. Here is an example:

    {"aligners": {
        "left-eye": {"type": "similarity",
            "fids": {"LEFT_EYE_OUT": [101.84555, 120.42587], "LEFT_EYE_IN": [49,401]},
            "offset": [5, 10]
        },
        "right-eye": {"type": "similarity",
            "fids": {"RIGHT_EYE_OUT": [101.84555, 120.42587], "RIGHT_EYE_IN": [49,401]},
            "offset": ["%(CROP_TOP_LEFT_X)s+5", "%(CROP_TOP_LEFT_Y)s+10"]
        }
        "torso": {"type": "affine",
            "fids": {"LEFT_SHOULDER": [101.84555, 120.42587], "RIGHT_SHOULDER": [49,401], "LEFT_HIP": [59,1], "RIGHT_HIP": [549, 140]}
        }
    }

    Each aligner is defined using a name, and contains fields:
        'fids': A dictionary of fiducial point mappings. It maps fiducial names
                to normalized output [x, y] coordinates, in the range 0-1.
        'offset': [optional] An [x, y] offset to apply to all input fiducials.
                If these are numbers, they are used as-is.
                If they are strings, then they are eval'ed after substitution:
                    val = eval(s % (fidDict))
                where fidDict is the dictionary of fiducial point locations.
                This lets you define things such as:
                    "%(LEFT_EYE_IN)s+5"
        'type': [optional] The underlying alignment type.  One of:
                    None, similarity, quad, affine.
                If not given, it's inferred from the length of 'fids':
                    < 2: None
                    2: similarity
                    4: quad
                    else: affine

    The alignment function optionally needs some input and output parameters.

    The input params are a mapping from fiducial names (corresponding to those
        in the config file) to their locations in the current image. They can
    also include a cropping rect with key 'crop', which is either a string of
    'x0,y0,x1,y1' rect from which to extract features from, or a list of
    numbers directly. This is first cut from the image and provides the extents
    relative to which the feature locations are assumed to be located.

    The output params include:
        width:   the width to scale the cropped input to (aspect ratio NOT preserved)
                 [default: original resolution]
        height:  the height to scale the cropped input to (aspect ratio NOT preserved)
                 [default: original resolution]

    The alignment is done by creating an output image of the right size and
    then mapping the input image into that output based upon the align type:
        None: The image is only cropped (if specified).
        similarity: The fiducials are used to determine a rotation, scaling, and translation.
        quad: The 4 fiducials are mapped as a quadrilateral onto the output.
        affine: The fiducials are used to compute a linear least squares affine alignment.
    """
    def __init__(self, name, alignconfig=ALIGN_CONFIG_FNAME):
        """Creates an aligner of the given name.
        The list of aligner names are defined in the given config file.
        """
        # read the config
        try:
            aligndict = json.load(open(alignconfig), strict=False)
        except Exception:
            aligndict = {}
        # set our align name and params
        if not name:
            name = ''
        d = aligndict.get(name, {})
        self.name = name
        self.outfids = d.get('fids', ())
        self.offsets = d.get('offset', ())
        if 'type' not in d:
            fidlen = len(self.outfids)
            if fidlen < 2:
                d['type'] = ''
            elif fidlen == 2:
                d['type'] = 'similarity'
            elif fidlen == 4:
                d['type'] = 'quad'
            else:
                d['type'] = 'affine'
        # normalize none
        if d['type'].lower() == 'none':
            d['type'] = ''
        self.aligntype = d['type']

    def align(self, im, fiducials=None, outparams=None, **kw):
        """Aligns the given image and returns (aligned image, modified outparams)."""
        from nkpylib.imageutils import croprect
        # normalize inputs
        if not fiducials: fiducials = {}
        if not outparams: outparams = {}

        # Check the md5 checksum, if it exists
        try:
            oldmd5 = outparams['ACTUAL_IMG_MD5'].lower()
            newmd5 = md5sum(open(im.filename).read())
            if oldmd5 != newmd5:
                raise Exception('Input file %s has checksum %s but should be %s' % (im.filename, newmd5, oldmd5))
        except (KeyError, AttributeError):
            pass

        # shift fiducials if needed
        if self.offsets:
            # convert offsets to numbers if they are eval'able format strings
            dx, dy = off = [eval(v % fiducials) if isinstance(v, basestring) else v for v in self.offsets]
            # now apply these offsets
            for name, val in fiducials.items():
                try:
                    x, y = val
                    fiducials[name] = (x+dx, y+dy)
                except TypeError: pass

        # set output size
        outint = lambda varname, defval: int(outparams.get(varname, defval))
        outsize = (outint('width', im.size[0]), outint('height', im.size[1]))
        outparams['width'], outparams['height'] = outsize

        # crop image if wanted
        if 'crop' in fiducials:
            rect = fiducials['crop']
            if isinstance(rect, basestring):
                rect = rect.split(',')
            rect = tuple(map(float, rect))
            im = croprect(im, rect, bg=(0,0,0))

        # do the actual alignment
        try:
            func = TRANSFORM_TYPES[self.aligntype]
            transform, outfids = func(fiducials, self.outfids, outsize)
            out = applyTransform(im, transform, outsize)
            out = drawfids(out, outfids, outparams)
            # add some keys to outparams
            outparams['AFFINE'] = printmat(transform)
            outparams['%s_TIMESTAMP' % self.aligntype.upper()] = str(time.time())
        except KeyError as e:
            #raise e
            # unknown or no transformation -- do no transformation
            # but apply drawfids if necessary
            fids = [f for f in fiducials.values() if type(f) != type(1.0) and type(f) != type(1) and len(f) == 2]
            out = drawfids(im, fids, outparams)

        # resize if our output is not the right size already
        if out.size != outsize:
            out = out.resize(outsize, INTERPOLATION)

        # return image and modified outparams
        return (out, outparams)


def parseFiducials(s):
    """Parses a fiducials dictionary from the given string"""
    from nkpylib.utils import specializeDict, str2kvdict
    # make the type-specialized dict
    fids = specializeDict(str2kvdict(s, sep='@', dlm='::'))
    # convert _x and _y fids to pairs
    names = set(f.rsplit('_', 1)[0] for f in fids if f.lower().endswith('_x') or f.lower().endswith('_y'))
    def popic(name, c):
        """Pops the fids with given name and ending char, ignoring case"""
        try:
            #return fids.pop(name+'_'+c.lower())
            return fids[name+'_'+c.lower()]
        except KeyError:
            #return fids.pop(name+'_'+c.upper())
            return fids[name+'_'+c.upper()]

    for n in names:
        fids[n] = x, y = popic(n, 'x'), popic(n, 'y')
    return fids

def simplemain():
    """A simple main"""
    from nkpylib.utils import specializeDict, str2kvdict
    if len(sys.argv) < 6:
        print('Usage: python %s <aligner name> <input image name> <output image name> <fiducials> <outparams>' % (sys.argv[0]))
        sys.exit()
    name, infname, outfname, fiducials, outparams = sys.argv[1:6]
    a = Aligner(name=name)
    im = Image.open(infname)
    fiducials = parseFiducials(fiducials)
    outparams = str2kvdict(outparams, sep='@', dlm='::')
    print('INFILE:', infname)
    print('FIDUCIALS:', fiducials)
    print('OUTPARAMS:', outparams)
    aligned, params = a.align(im, fiducials=fiducials, outparams=outparams)
    print('PARAMS:', params)
    saveImage(aligned, outfname, params)
    print('OUTFILE:', outfname)
    sys.exit()

def processLine(line):
    """Process a single line of input, returning a single line of output as a string.
    Input on stdin is
        <input path>\t<output fmt>\t<aligner>\t<fiducials>\t<output parameters>
    where:
      - <input path> is a local path of the input image to align (not a url),
      - <output fmt> is a format string which will generate the output path. It's given a dict with:
              dfij:  doifj
              blah: difj
      - <aligner> is the name of the aligner to use,
      - <fiducials> is a list of 'key@value' pairs, joined using ::
          These are used for determining feature locations, which the aligners are defined relative to.
          Any extra fiducials (not needed by the given aligner) are ignored.
          If there is a missing fiducial, an error is returned.
      - <output parameters> is an optional list of 'key@value' pairs, joined using '::'
          These are used for defining parameters about the output. Currently, we support:
                 crop: 'x0,y0,x1,y1' rect from which to extract features from. This is
                       first cut from the image and provides the extents relative to which
                       the feature locations are assumed to be located.
                       [default: no crop]

                width: the width to scale the cropped input to (aspect ratio NOT preserved)
                       [default: original resolution]

               height: the height to scale the cropped input to (aspect ratio NOT preserved)
                       [default: original resolution]

             drawfids: how to draw fiducials on output. options:
                           none: don't draw fiducials [default]
                           circle: draw a circle
                           rectangle: draw a rectangle

         drawfidsline: the color to draw fiducial outlines in, as any valid color string (only if drawfids is on)
                       [default: green]

         drawfidsfill: the color to fill drawn fiducials in, as any valid color string (only if drawfids is on)
                       [default: green]

            drawfidsr: the radius of the circle to draw fiducials in
                       [default: 3]

               outfmt: the output format to print on stdout. This is a standard python format string,
                       to which we'll pass a dictionary with the following fields:
                           basename: input file basename
                           inpath: input file path
                           outpath: output file path
                           outfmt: the passed-in output file format string
                           aligner: the passed-in aligner string
                           fiducials: the passed-in input parameters string
                           outparams: the passed-in output parameters string
                       [default: '%(inpath)s\t%(outpath)s']

               errfmt: what to print in case of error, again as a python format string.
                       The fmtdict is like in 'fmt', and also containing:
                           errortype: a python exception type name
                           errormsg: the error message
                       [default: 'error']

    A full input string might look like:
        FIXME
    """
    #TODO test out various outfmt options
    #TODO how to specify if we want to write EXIF or not?
    from collections import defaultdict
    fmtdict = defaultdict(str)
    DEFAULT_OUTPARAMS = defaultdict(str)
    DEFAULT_OUTPARAMS['outfmt'] = DEFAULT_OUTPUT_FMT
    DEFAULT_OUTPARAMS['errfmt'] = DEFAULT_ERROR_FMT
    DEFAULT_OUTPARAMS['drawfids'] = 'none'
    DEFAULT_OUTPARAMS['drawfidsline'] = 'green'
    DEFAULT_OUTPARAMS['drawfidsfill'] = 'green'
    DEFAULT_OUTPARAMS['drawfidsr'] = 3

    # parse elements
    els = line.split('\t')
    try:
        # input and output
        fmtdict['inpath'] = inpath = els.pop(0)
        fmtdict['basename'] = basename = os.path.basename(inpath)
        fmtdict['outpathfmt'] = outpathfmt = els.pop(0)
        #print path, basename, fmtdict, outfmt
        # aligner
        fmtdict['aligner'] = aligner = els.pop(0)
        #print aligner
        # fiducials
        fmtdict['fiducials'] = fiducials = els.pop(0)
        fiducials = parseFiducials(fiducials)
        # output params
        outparams = dict(**DEFAULT_OUTPARAMS)
        #print outparams
        if els:
            # output params are optional, so we don't want to raise an exception here
            fmtdict['outparams'] = els.pop(0)
            #print fmtdict['outparams']
            outparams.update(str2kvdict(fmtdict['outparams'], sep='@', dlm='::'))
            #print outparams

        # at this stage, we have everything we need
        # first make sure the file exists and open it
        if not os.path.exists(inpath): raise IOError('Image does not exist')
        im = Image.open(inpath)
        # process the image
        a = Aligner(name=aligner)
        aligned, params = a.align(im, fiducials=fiducials, outparams=outparams)
        fmtdict.update(params)
        outparams['outfmt'] = outparams['outfmt'].replace(r'\t', '\t').replace(r'\n', '\n')
        # save the output image
        fmtdict['outpath'] = outpath = outpathfmt % fmtdict
        #print outpathfmt, inpath, basename, fmtdict, outpath
        fmtdict['outpathfmt'] = fmtdict['outpathfmt'].replace(r'\t', '\t').replace(r'\n', '\n')
        saveImage(aligned, outpath, params)
        # generate the output string
        ret = outparams['outfmt'] % (fmtdict)
        return ret
    except Exception as e:
        raise
        # add the error values to the fmtdict
        fmtdict['errortype'] = type(e).__name__
        try:
            fmtdict['errormsg'] = e
        except Exception:
            pass
        # generate and return the error string
        errstr = outparams['errfmt'] % fmtdict
        return errstr

def mainloop():
    """An infinite main loop for running alignment"""
    global ALIGN_CONFIG_FNAME
    if len(sys.argv) < 1:
        print("Usage: python %s [<aligners file>=%s]" % (sys.argv[0], ALIGN_CONFIG_FNAME))
        sys.exit()
    try:
        ALIGN_CONFIG_FNAME = sys.argv[1]
    except IndexError:
        pass
    assert os.path.exists(ALIGN_CONFIG_FNAME), 'The given align config file %s does not exist!' % (ALIGN_CONFIG_FNAME)

    def do(line):
        try:
            print(processLine(line))
            sys.stdout.flush()
        except IOError:
            pass

    stdmainloop(do)


def parseInputs(fname):
    """Parses the input in the given file to get a dict of fiducials."""
    from imageutils import imageiter
    fids = {}
    try:
        # if it's an image, read colored points as fiducials
        im = Image.open(fname)
        assert im.mode == 'RGB'
        # get all colored pixels
        for loc, col in imageiter(im):
            r, g, b = col
            if r==g==b: continue # grayscale points don't count
            if col not in fids:
                fids[col] = []
            fids[col].append(loc)
        # average each color's location to get precise estimate
        for col in fids:
            xs, ys = zip(*fids[col])
            l = float(len(xs))
            fids[col] = (sum(xs)/l, sum(ys)/l)
    except Exception:
        # it's a text file, so get the format
        for i, l in enumerate(open(fname)):
            l = l.rstrip('\n')
            if '\t' in l: # could possibly have string keynames
                els = l.split('\t')
                if len(els) == 2: # only x, y, so use i as the keyname
                    fids[i] = map(float, els)
                elif len(els) > 2: # assume 1st field is keyname and next 2 are x,y
                    fids[els[0]] = map(float, els[1:3])
            else: # must be separated by spaces, so no strings
                fids[i] = map(float, l.split()[:2]) # x,y are first 2 fields
    log('Read %d fids from %s, with first items: %s' % (len(fids), fname, sorted(fids.items())[:3]))
    return fids



TEST_USAGE = '''Usages:
  <%(transtypes)s> <input fids> <output fids> [<transformed fids fname>] => homography
  <%(transtypes)s> <img1 w/colored fids> <img2 w/colored fids> [<transformed img1 fname>] => homography
''' % (dict(transtypes='|'.join(TRANSFORM_TYPES)))

def testhomography(args):
    """Tests out homography estimation"""
    import select
    np.set_printoptions(precision=7, linewidth=150, suppress=1)
    if len(args) < 2:
        print(TEST_USAGE)
        sys.exit()
    # read inputs and parse
    transtype = args.pop(0)
    transfunc = TRANSFORM_TYPES[transtype]
    in1 = args.pop(0)
    in2 = args.pop(0)
    fids1, fids2 = parseInputs(in1), parseInputs(in2)
    h, ah = transfunc(fids1, fids2)
    print(h)
    # any remaining arguments are for outputs
    if args:
        outname = args.pop(0)
        if outname.endswith('.jpg') or outname.endswith('.png'):
            # save output image
            outim = Image.open(in1)
            outim = applyTransform(outim, h, Image.open(in2).size)
            outim.save(outname)
        else:
            # save projected points text file
            pass #TODO
    if select.select([sys.stdin], [], [], 0.0)[0]:
        #TODO also translate from stdin pts to stdout
        # we have stdin data
        pass

def debug():
    a = [[220, 298],
    [427, 313],
    [297, 374],
    [244, 457],
    [379, 469],
    [176, 257],
    [278, 244],
    [191, 282],
    [230, 276],
    [214, 324],
    [256, 315],
    [383, 252],
    [489, 266],
    [392, 318],
    [424, 288],
    [432, 338],
    [474, 307],
    [288, 319],
    [337, 323],
    [277, 354],
    [323, 357],
    [266, 386],
    [347, 396],
    [298, 409],
    [298, 425],
    [298, 443],
    [300, 457]]

    b = [[198.24593, 218.48312],
    [301.75409, 218.48312],
    [250, 288.72064],
    [196.73442, 335.1088],
    [303.26559, 335.1088],
    [152.62563, 206.89514],
    [220.81578, 196.98947],
    [180.65184, 221.26352],
    [196.88483, 213.14243],
    [197.29204, 224.76518],
    [212.424, 220.57774],
    [279.18423, 196.98947],
    [347.37439, 206.89514],
    [287.57602, 220.57774],
    [303.11517, 213.14243],
    [302.70798, 224.76518],
    [319.34818, 221.26352],
    [234.78632, 222.358],
    [265.21368, 222.358],
    [227.47029, 264.40878],
    [272.52972, 264.40878],
    [216.52325, 288.04016],
    [283.47675, 288.04016],
    [250, 329.29788],
    [250, 337.32162],
    [250, 347.60889],
    [250, 361.46271]]

    print(len(a), len(b))

    np.set_printoptions(precision=7, linewidth=150, suppress=1)
    x, ax = getAffineTransform(b, a)
    print(x)
    print(ax)
    out = applyTransform(Image.open('cat.png'), x, (225,250))
    out.save('cat-out.png')

if __name__ == "__main__":
    #simplemain()
    #mainloop()
    #testhomography(sys.argv[1:])
    debug()
