"""A module to get/set our exif information from images.

Currently I use pyexiv2, with a monkey patch to deal with some errors in their code"""

import pyexiv2

COMMENT_TAG = 'Exif.Photo.UserComment'


# fix for older versions of pyexiv2
try:
    pyexiv2.UndefinedToString_ = pyexiv2.UndefinedToString
    def fix(s):
        try:
            return pyexiv2.UndefinedToString_(s)
        except Exception: return s

    pyexiv2.UndefinedToString = fix
except AttributeError: pass

def getAllEXIF(fname, asdict=0):
    """Returns the full exif tags of the given fname.
    If asdict is true, then converts the output into a dict"""
    try:
        im = pyexiv2.Image(fname)
        im.readMetadata()
    except AttributeError:
        # must be newer version of pyexiv2
        im = pyexiv2.ImageMetadata(fname)
        im.read()
    if asdict:
        keys = im.exifKeys() + im.iptcKeys()
        vals = [im[k] for k in keys]
        im = dict(zip(keys, vals))
    return im

def getEXIF(fname, key):
    """Returns the given key of the EXIF tags of the given file"""
    return getAllEXIF(fname)[key]

def setEXIF(fname, key, val):
    """Sets the given EXIF tag for the given filename"""
    try:
        im = pyexiv2.Image(fname)
        im.readMetadata()
        im[key] = val
        im.writeMetadata()
    except AttributeError:
        # must be newer version of pyexiv2
        im = pyexiv2.ImageMetadata(fname)
        im.read()
        im[key] = val
        im.write()

def getDictFromString(s, specialize=1):
    """Returns a tag dict from a string"""
    ret = dict([el.split('@', 1) for el in s.split('|') if el and '@' in el])
    try:
        from utils import specializeDict
        return specializeDict(ret) if specialize else ret
    except ImportError: return ret

def getDict(fname):
    """Returns the dict of values of the UserComment"""
    s = getEXIF(fname, COMMENT_TAG)
    try:
        s = s.value
    except Exception: pass
    return getDictFromString(s)

def setDict(fname, d):
    """Sets the dict of values for the UserComment for the given value"""
    d = dict((k.upper(), v) for k,v in d.iteritems())
    s = '|'.join('%s@%s' % (k, d[k]) for k in sorted(d))
    setEXIF(fname, COMMENT_TAG, s)

def getCropRect(d, prefix=''):
    """Returns the crop rect as (x0, y0, x1, y1) from the given dictionary of tags.
    Assumes tags are named 'CROP_RECT_TOP_LEFT_{X,Y} and CROP_RECT_{WIDTH,HEIGHT}.
    You can give an optional prefix if you want (e.g., ACTUAL_).
    On error, returns None"""
    try:
        x0 = int(d[prefix+'CROP_RECT_TOP_LEFT_X'])
        y0 = int(d[prefix+'CROP_RECT_TOP_LEFT_Y'])
        w = int(d[prefix+'CROP_RECT_WIDTH'])
        h = int(d[prefix+'CROP_RECT_HEIGHT'])
        return (x0, y0, x0+w, y0+h)
    except KeyError:
        return None

def getFiducials(d, offset=None):
    """Returns fiducials from the given dict as a list of (x,y) pairs, or None on error.
    The fiducials are normally translated relative to CROP_RECT_TOP_LEFT_{XY}, but you can
    give an explicit [x,y] offset if you want.
    The order of points returned is:
        left eye out
        left eye in
        right eye in
        right eye out
        mouth left
        mouth right
    """
    KEYS = 'LEFT_EYE_OUT LEFT_EYE_IN RIGHT_EYE_IN RIGHT_EYE_OUT MOUTH_LEFT MOUTH_RIGHT'.split()
    if offset is None:
        offset = [int(d['CROP_RECT_TOP_LEFT_X']), int(d['CROP_RECT_TOP_LEFT_Y'])]
    try:
        ret = [(int(d[k+'_X'])+offset[0], int(d[k+'_Y'])+offset[1]) for k in KEYS]
        return ret
    except KeyError:
        return None


def getQ1(d):
    """Returns the 'q1' quality measure from a dictionary of tags.
    This is defined as the image size in bytes * (# crop pixels) / (# image pixels).
    If any required fields are missing, this will return -1"""
    FIELDS = 'ACTUAL_IMG_SIZE ACTUAL_CROP_RECT_WIDTH ACTUAL_CROP_RECT_HEIGHT ACTUAL_IMG_WIDTH ACTUAL_IMG_HEIGHT'.split()
    cur = [d.get(f, 0) for f in FIELDS]
    cur = [v if v else 0 for v in cur]
    imbytes, cropw, croph, imw, imh = map(float, cur)
    if imw * imh == 0: return -1
    return imbytes * (cropw * croph) / (imw * imh)

def getCamera(fname):
    """Returns the camera type for the given filename"""
    from PIL import Image
    try:
        make = getEXIF(fname, 'Exif.Image.Make')
        model = getEXIF(fname, 'Exif.Image.Model')
        ret = '%s %s' % (make, model)
    except KeyError:
        ret = 'Unknown Camera'
    if 'iphone' in ret.lower():
        im = Image.open(fname)
        res = im.size[0]*im.size[1]
        if res == (1536*2048):
            ret += ' 3GS'
        else:
            # either 2g or 3g #TODO use specific exif tags such as gamma or FlashPixVersion or ColorSpace to figure out
            pass
    return ret

def getTimestamp(fname):
    """Returns a unix-style timestamp of when the photo was taken, or None on error"""
    from time import strptime, mktime, strftime
    LAST_MODIFIED = 'Exif.Image.DateTime'
    CREATED = 'Exif.Photo.DateTimeOriginal'
    try:
        t = getEXIF(fname, CREATED)
        return mktime(t.timetuple())
    except KeyError: 
        try:
            t = getEXIF(fname, LAST_MODIFIED)
            return mktime(t.timetuple())
        except KeyError:
            return None

def getGPS(fname):
    """Returns a tuple of (latitude, longitude) in decimal degrees from exif tags, or None on error"""
    keys = ['Exif.GPSInfo.GPS%s' % (k) for k in 'Latitude Longitude'.split()]
    try:
        e = getAllEXIF(fname)
        # TODO use gpsdeg2dec func in utils
        vals = [e.interpretedExifValue(k) for k in keys]
        vals = [v.replace("'", '').strip().split('deg ') for v in vals]
        vals = [map(float, v) for v in vals]
        vals = [(v[0], int(v[1])/60.0, (v[1]-int(v[1]))/3600.0) for v in vals]
        vals = [sum(v) for v in vals]
        dirs = [e[k+'Ref'] for k in keys]
        vals[0] *= 1 if dirs[0].lower() == 'n' else -1
        vals[1] *= 1 if dirs[1].lower() == 'e' else -1
        return vals
    except KeyError: return None

def prependExifComment(fname, s):
    """Prepends s to the exif UserComment in the given filename.
    Returns the new UserComment
    """
    cur = getEXIF(fname, COMMENT_TAG)
    setEXIF(fname, COMMENT_TAG, s+cur)
    return s+cur

def exifTagsMain():
    """A simple driver to get exif tags from all fnames given in stdin.
    The output is printed to stdout, and is: fname\ttags
    """
    import sys
    for fname in sys.stdin:
        fname = fname.strip()
        try:
            d = getDict(fname)
            s = '|'.join('%s@%s' % (k, d[k]) for k in sorted(d))
            print '%s\t%s' % (fname, s)
            sys.stdout.flush()
        except Exception, e:
            print >>sys.stderr, 'Error with fname %s: %s' % (fname, e)
    sys.exit()

if __name__ == '__main__':
    exifTagsMain()
    import sys, shutil
    from datetime import datetime
    for fname in sys.argv[1:]:
        t = getTimestamp(fname)
        print t, datetime.fromtimestamp(t)
        d = getDict(fname)
        d['aaahello'] = 591
        outfname = 'testexifout.jpg'
        #shutil.copyfile(fname, outfname)
        #setDict(outfname, d)
