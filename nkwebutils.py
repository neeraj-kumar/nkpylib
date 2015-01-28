"""A set of utilities to use for web apps made using web.py.

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

import os, sys, time
import web
try:
    import simplejson as json
except ImportError:
    import json
from nkutils import *

# alias for web's storage
stor = web.storage

# turn off debugging output by default
web.config.debug = False

# create the globals dictionary for use in rendering templates
g = globals()
for func in [zip, sorted, len, range, enumerate, getattr]:
    g[func.__name__] = func
RENDER_GLOBALS = g

def notfound(s='Error: 404 Page Not Found'):
    """Custom 404"""
    return web.notfound(s)

# INITIALIZATION FUNCTIONS
#app = web.application(urls, globals())
#application = app.wsgifunc()

# sets the app's 404 to the notfound function
#app.notfound = notfound

def setupdb(**kw):
    """Sets up the database and adds some default args based on dbn.
    Returns the db"""
    assert 'dbn' in kw
    # setup default options
    dbn = kw['dbn']
    if dbn == 'sqlite':
        defargs = dict(isolation_level=None, timeout=50)
        for k, v in defargs.items():
            if k not in kw:
                kw[k] = v

    db = web.database(**kw)

    # set some other params
    if dbn == 'sqlite':
        #db.query('pragma synchronous=normal') # 'full' is default, but this should be faster...and 'off' is mega fast, but unsafe
        db.query('pragma read_uncommitted=true') # will NOT be ACID on reads, but that's fine...
        #db.query('analyze;') # analyze to optimize query performance

    return db

def initpytz():
    """Initialize pytz if available.
    Right now this is more an example than anything else"""
    import pytz
    dt = datetime.datetime(1980,1,1, tzinfo=pytz.utc)

def applicationWrapperExample():
    """An example of how to wrap an app"""
    _application = app.wsgifunc()
    def application(environ, start_response):
        global APIBASEDIR
        APIBASEDIR = environ.get('APIBASEDIR', '')
        #web.debug('Setting APIBASEDIR to %s, from env %s' % (APIBASEDIR, environ))
        return _application(environ, start_response)


# SMALL UTILITIES
class NKStor(web.Storage):
    """A thin wrapper on web.py's storage class which returns empty strings for non-existent keys"""
    def __getattr__(self, k):
        """Returns empty string if the given one doesn't exist"""
        try:
            return super(NKStor, self).__getattr__(k)
        except AttributeError:
            return ''

    def __getitem__(self, k):
        """Returns empty string if the given one doesn't exist"""
        try:
            return super(NKStor, self).__getitem__(k)
        except KeyError:
            return ''

def mystorify(d):
    """Converts a python dictionary to a web.storage, recursively"""
    if not isinstance(d, dict): return d
    ret = stor(d)
    for k in ret:
        if isinstance(ret[k], dict):
            ret[k] = mystorify(ret[k])
        elif isinstance(ret[k], (list, tuple)):
            ret[k] = [mystorify(el) for el in ret[k]]
    return ret

class WebJSONEncoder(json.JSONEncoder):
    """Custom output for dates, etc."""
    def default(self, obj):
        from datetime import datetime, date
        if isinstance(obj, web.Storage):
            return dict(obj)
        if isinstance(obj, datetime):
            # drop timezone from datetime if it has it
            #FIXME is this what we want?!!
            obj = obj.replace(tzinfo=None)
            return str(obj)
        if isinstance(obj, date):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def jsonp(s, callback=''):
    """Wraps the given string with the callback function name given.
    If callback is not given, then just returns s"""
    if callback:
        s = '%s(%s);' % (callback, s)
    return s

def runinthread(target, args=(), kwargs={}, daemon=1, procs=[]):
    """Runs the target function with the given args and kwargs in a daemon thread"""
    from threading import Thread
    t = Thread(target=target, args=args, kwargs=kwargs)
    t.setDaemon(daemon)
    t.start()
    procs.append(t)

# INPUT/OUTPUT UTILITIES
def html(d, elements, title=None, css=[], js=[], gatag=None, rel='', tmplfname='generic.tmpl', globalvars=None, **kw):
    """Renders the given page with given js, css, etc.
    If gatag is given, then it's used as a google analytics id.
    If rel is given, it's used as the basis for local style sheets and javascripts.
    Otherwise, local scripts are at /static/
    Remote scripts are always fine.
    You can also specify globalvars as needed. zip and enumerate are always included.
    """
    out = stor(css=css, js=js, elements=elements, gatag=gatag, rel=rel)
    import web.template as template
    if not globalvars:
        globalvars = {}
    globalvars.update(dict(zip=zip, enumerate=enumerate))
    render = template.frender(tmplfname, globals=globalvars)
    if not title:
        title = d.title.title()
    out.title = title
    for k, v in kw.iteritems():
        out[k] = v
    return render(out)

def renderDict(d, tmpl):
    """Renders a given dictionary to a string using the given template fname."""
    import web.template as template
    renfunc = template.frender(tmpl, globals=RENDER_GLOBALS)
    s = str(renfunc(d))
    return s

def renderDictToFile(d, tmpl, outname=None, reldir=__file__, **kw):
    """Renders a dictionary using the given template.
    This assumes the tmpl fname ends in .tmpl.
    If no outname is given, then simply replaces .tmpl with .html
    Existing outputs are renamed to '.'+outname temporarily, and then deleted.
    The final output is passed through the html() function, with the given kw.
    Computes the 'rel' parameter using the path given in 'reldir'.
    The reldir defaults to the location of this file.
    Returns the outname.
    """
    s = renderDict(d, tmpl)
    dir, fname = os.path.split(tmpl)
    fname = fname.rsplit('.tmpl')[0]
    if not outname:
        outname = os.path.join(dir, fname + '.html')
    print 'Rendering from %s to %s' % (os.path.join(dir, fname+'.tmpl'), outname)
    # make parent dirs
    try:
        os.makedirs(os.path.dirname(outname))
    except OSError: pass
    # make sure any existing output is first renamed so render doesn't use it
    try:
        os.rename(outname, '.'+outname)
    except OSError: pass
    # compute the relative path to the static dir
    cur = os.path.abspath(os.path.dirname(reldir))
    outdir = os.path.abspath(os.path.dirname(outname))
    prefix = os.path.commonprefix([cur, outdir])
    levels = len(outdir.replace(prefix, '', 1).split('/')) - 1
    if levels == 0:
        rel = '.'
    else:
        rel = '/'.join('..' for i in range(levels))
    #print cur, outdir, prefix, levels, rel
    s = str(html(d, [s], rel=rel, **kw))
    f = open(outname, 'wb')
    f.write(s)
    f.close()
    # remove old output
    try:
        os.remove('.'+outname)
    except OSError: pass
    return outname

def rethandler(data, input, txtfmt=None, htmldat=None, htmlfunc=None, jsoncontent=1, **kw):
    """Returns results using the given dictionary of data.
    The input is used to read the format ('input.fmt') and callback
    for jsonp ('input.callback'). The default format is 'json'.
    If the fmt is text, then txtfmt is either a format string
    or a function that takes the data and returns a string.
    If the format is html, then you have 3 options:
        1. Set txtfmt to None and htmldat to be a pair of (title, renderfunc).
        This is passed to html(), along with **kw.

        2. Set txtfmt to None, htmlfunc to a custom html function, and htmldat
        to be its args. Then htmlfunc(data, *htmldat, **kw) will be called.

        3. Set txtfmt just as you would for 'txt' format and set htmldat to None.
        This is just like formatting text output, but the content-type is set to html

    If jsoncontent is true (the default), then sends a content-type of application/json
    """
    formats = 'json txt html'.split()
    fmt = input.get('fmt', 'json')
    if fmt == 'json':
        if jsoncontent:
            web.header('Content-Type', 'application/json; charset=utf-8')
        else:
            web.header('Content-Type', 'text/plain; charset=utf-8')
        s = json.dumps(data, cls=WebJSONEncoder, indent=2, sort_keys=1)
        s = jsonp(s, input.get('callback', ''))
        #web.debug('About to return json output: %s' % s)
        return s
    elif txtfmt is not None and fmt in 'txt html'.split():
        web.header('Content-Type', 'text/%s; charset=utf-8' % ('plain' if fmt == 'txt' else 'html'))
        if isinstance(txtfmt, basestring):
            return txtfmt % (data)
        else:
            return txtfmt(data)
    if fmt == 'html' and htmldat is not None:
        if htmlfunc:
            return htmlfunc(data, *htmldat, **kw)
        else:
            title, renderfunc = htmldat
            return html(stor(), elements=[renderfunc(data)], title=title, **kw)
    raise web.notfound('Illegal format (%s). Options are: %s' % (fmt, ', '.join(formats)))

def textrethandler(s, fmt='txt'):
    """A convenience function for using a ret handler for simple text or html.
    Set fmt to 'txt' or 'html'."""
    return rethandler({'s': s}, {'fmt': fmt}, txtfmt='%(s)s')

def imrethandlerfname(fname, params={}, cachedir='static/cache/', cachenamefunc=stringize, cacherep='static/', postfunc=None):
    """Returns a local path for the given image, after various manipulations.
    Also checks the params for various image manipulation options:
        'cache': if 0, then no caching is done.
                 if 1, then the image is cached in the given cachedir.
                 Caching is done by replacing cacherep (default='static/') in
                 the fname to the cachedir given.  The default cachedir is
                 'static/cache/'.
                 Then, the other options passed to this function are used to generate a cache fname
                 using the cachenamefunc given (defaulting to stringize).
        'aspect': landscape: rotates the image if needed to make it landscape.
                  portrait: rotates the image if needed to make it portrait.
        'rot': Rotates the image counter-clockwise by this many degrees.
               This must be a multiple of 90.
        'w': Sets the width to this many pixels, maintaining aspect ratio.
             Will not exceed original size.
        'h': Sets the height to this many pixels, maintaining aspect ratio.
             Will not exceed original size.
        'crop': Given as 'x0,y0,x1,y1', this crops the image (at the original size).
                If the crop rectangle given extends outside of the image, those areas
                are filled with black.

    Returns the raw data from the processed image, after setting web.header().
    If there's any problem processing the image, a web.badrequest() is raised.
    """
    from PIL import Image
    import tempfile
    from nkimageutils import croprect
    ext = fname.rsplit('.')[-1].lower()
    caching = 0 if 'cache' in params and int(params['cache']) == 0 else 1
    if not cachenamefunc or not cachedir:
        caching = 0
    # generate the cachename and return the cached image, if it exists
    opts = [(k, params[k]) for k in 'aspect crop rot w h'.split() if k in params]
    if not cachedir.endswith('/'):
        cachedir += '/'
    outputfname = fname.replace(cacherep, cachedir) + '__'+stringize(opts)+'.'+ext
    if caching and os.path.exists(outputfname): return outputfname
    # if not, actually go through and apply all the transformations
    t1 = time.time()
    # first make sure we can open this image and get its size
    try:
        im = Image.open(fname)
        w, h = im.size
    except IOError: raise web.badrequest()
    t2 = time.time()
    # crop the image according to the crop parameters (given as x0,y0,x1,y1)
    if 'crop' in params:
        try:
            rect = map(float, params['crop'].strip().split(','))
            rect = [int(c+0.5) for c in rect]
            assert len(rect) == 4
        except Exception: raise web.badrequest()
        im = croprect(im, rect, bg=(0,0,0))
        w, h = im.size
    # rotate the image (right-angles only)
    if 'rot' in params:
        r = int(params['rot'])
        while r < 0:
            r += 360
        if r % 90 > 0: raise web.badrequest()
        meth = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}[r]
        im = im.transpose(meth)
    # set aspect ratio explicitly to either 'landscape' or 'portrait'
    if 'aspect' in params:
        a = params['aspect'].strip().lower()
        if a == 'landscape':
            if im.size[0] < im.size[1]:
                im = im.transpose(Image.ROTATE_90)
        elif a == 'portrait':
            if im.size[0] > im.size[1]:
                im = im.transpose(Image.ROTATE_90)
    t3 = time.time()
    # resize image down to requested size
    thumbsize = list(im.size)
    if 'w' in params:
        thumbsize[0] = int(params['w'])
    if 'h' in params:
        thumbsize[1] = int(params['h'])
    im.thumbnail(thumbsize, Image.ANTIALIAS)
    t4 = time.time()
    # convert to color if it's a palette-based image
    if im.mode == 'P':
        im = im.convert('RGB')
    if postfunc:
        im = postfunc(im, fname)
    t5 = time.time()
    if caching: # using cache filename
        tempname = outputfname
        try:
            os.makedirs(os.path.dirname(outputfname))
        except OSError: pass
    else: # using temp filename
        f, tempname = tempfile.mkstemp(suffix='.'+ext)
        os.close(f)
    im.save(tempname)
    web.debug('Returning image at %s (cached to %s) with params %s, for final size %s (%0.3f secs to open, %0.3f secs to rotate, %0.3f secs to resize, %0.3f secs to postfunc)' % (fname, tempname, params, im.size, t2-t1, t3-t2, t4-t3, t5-t4))
    return tempname

def imrethandler(fname, params={}, cachedir='static/cache/', cachenamefunc=stringize, cacherep='static/', postfunc=None):
    """Returns the given image, setting the content type appropriately.
    Simply a wrapper on imrethandlerfname().
    Returns the raw data from the processed image, after setting web.header().
    If there's any problem processing the image, a web.badrequest() is raised.
    """
    fname = imrethandlerfname(fname=fname, params=params, cachedir=cachedir, cachenamefunc=cachenamefunc, cacherep=cacherep, postfunc=postfunc)
    ext = fname.rsplit('.')[-1].lower()
    web.header('Content-Type', 'image/%s' % (ext))
    #TODO add etags?
    return open(fname, 'rb').read()


def icongenerator(params={}, cachedir='static/cache/'):
    """Creates an icon with given parameters:
        'cache': if 0, then no caching is done.
                 if 1, then the image is cached in the given cachedir.
                 The options passed to this function are used to generate a cache fname.
        'w': Sets the width to this many pixels.
        'h': Sets the height to this many pixels.
        'fill': Sets the fill color
        'outline': Sets the outline color
        'shape': Sets the shape to draw:
            'rect': rectangle/square
            'oval': oval/circle
            'uptri': triangle pointing up
            'downtri': triangle pointing down
            'lefttri': triangle pointing left
            'righttri': triangle pointing right
            'uppie': pie slice pointing up (tip at center)
            'downpie': pie slice pointing down (tip at center)
            'leftpie': pie slice pointing left (tip at center)
            'rightpie': pie slice pointing right (tip at center)
        'rot': Rotate the figure after generation by given number of degrees (counter-clockwise)

    Returns the raw data from the processed image, after setting web.header().
    If there's any problem processing the image, a web.badrequest() is raised.
    """
    from PIL import Image, ImageDraw
    import tempfile
    web.header('Content-Type', 'image/png')
    caching = 0 if 'cache' in params and int(params['cache']) == 0 else 1
    if not cachedir:
        caching = 0
    # generate the cachename and return the cached image, if it exists
    opts = [(k, params[k]) for k in 'w h fill outline shape rot'.split() if k in params]
    if not cachedir.endswith('/'):
        cachedir += '/'
    outputfname = os.path.join(cachedir, 'icon-'+stringize(opts)+'.png')
    if caching and os.path.exists(outputfname): return open(outputfname, 'rb').read()
    # if not, actually go through and generate the image
    w, h = [int(params.setdefault(f, 32)) for f in 'wh']
    im = Image.new('RGBA', (w,h), (0,0,0,1))
    draw = ImageDraw.Draw(im)
    shape = params.setdefault('shape', 'oval')
    options = {}
    for k in 'fill outline'.split():
        if k in params:
            options[k] = params[k]
    bbox = (0,0,w-1,h-1)
    if shape in 'rect rectangle square'.split():
        draw.rectangle(bbox, **options)
    elif shape in 'oval ellipse circ circle'.split():
        draw.ellipse(bbox, **options)
    elif shape.endswith('pie'):
        dir = shape[:-3]
        start, end = dict(right=(135,225), up=(45,135), down=(225,315), left=(315,45))[dir]
        draw.pieslice(bbox, start, end, **options)
    elif shape.endswith('tri'):
        draw.polygon((w//2,0,w-1,h-1,0,h-1), **options)
        dir = shape[:-3]
        angle = dict(right=Image.ROTATE_270, up=Image.FLIP_LEFT_RIGHT, down=Image.ROTATE_180, left=Image.ROTATE_90)[dir]
        im = im.transpose(angle)
    rot = int(params.get('rot', 0))
    if rot:
        im = im.rotate(rot, Image.BICUBIC, 1)
        im.resize((w,h), Image.ANTIALIAS)
    if caching: # using cache filename
        tempname = outputfname
        try:
            os.makedirs(os.path.dirname(outputfname))
        except OSError: pass
    else: # using temp filename
        f, tempname = tempfile.mkstemp(suffix='.png')
        os.close(f)
    im.save(tempname)
    web.debug('Returning icon of size %s (cached to %s) with params %s' % (im.size, tempname, params))
    return open(tempname, 'rb').read()

def watermarkpostfunc(im, fname, watermark, minsize=(0,0), loc=(-1,-1), opacity=1.0, **kw):
    """A postfunc to use for imrethandler which adds a watermark.
    Options:
        watermark: one of the following:
                       a string - rendered using createTextWatermark() and **kw.
                       an image - must be same size as im - simply composited on.
        minsize: a 2-ple with minimum width and height requirements to create watermark.
        loc: The (x,y) location to put the watermark. If a float, puts it at the given percentage.
             If a positive int, puts it at the given offset to the top-left.
             If a negative int, puts it at the given offset to the bottom-right.
        opacity: determines how opaque to make the watermark (1.0 = fully opaque).
    The output is converted to RGB.
    Use genericBind to bind the 3rd arg onwards.
    """
    from nkimageutils import createTextWatermark, watermarkImage
    # check size
    if im.size[0] < minsize[0] or im.size[1] < minsize[1]: return im

    # figure out loc
    outloc = []
    for cur, lim in zip(loc, im.size):
        # deal with negative values first
        if cur < 0: # relative to bottom-right
            # flip it around
            if isinstance(cur, float):
                cur = 1.0 - cur
            else:
                cur = lim - cur
        # now deal with percentages
        if isinstance(cur, float): # percentage
            cur = int(lim * cur)
        outloc.append(cur)
    loc = tuple(outloc)

    # figure out type of watermark
    if isinstance(watermark, basestring):
        watermark = createTextWatermark(watermark, im.size, loc, **kw)
    im = watermarkImage(im, mark, opacity=opacity).convert('RGB')
    return im

def savefile(input, fname, uploadvar='myfile', urlvar='url'):
    """Saves a file uploaded or from a url to the given fname.
    If uploadvar is given (default 'myfile') and exists, that file is saved as
    an upload. Note that the form must set enctype="multipart/form-data".
    If urlvar is given (default 'url') and exists, that file is downloaded.
    Uses urllib.urlretrieve, so set a custom url opener before-hand if needed.

    Returns the fname the file was saved to.
    If there's an error, raises web.notfound() with message.
    """
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError: pass
    if uploadvar in input and input[uploadvar]:
        #TODO store the original fname somewhere
        try:
            f = open(fname, 'wb')
            f.write(input[uploadvar])
            f.close()
        except IOError, e:
            raise web.notfound('Error getting fileupload from var %s - %s' % (uploadvar, e))
    elif urlvar in input and input[urlvar]:
        try:
            fname, headers = urllib.urlretrieve(input[urlvar], fname)
        except IOError, e:
            raise web.notfound('Error getting url %s - %s' % (input[urlvar], e))
    else:
        raise web.notfound('Error: no url specified in %s and no file uploaded in %s' % (uploadvar, urlvar))
    return fname

def get_content_type(filename, default='application/octet-stream'):
    """Guesses the content type from a filename, or uses the default"""
    import mimetypes
    return mimetypes.guess_type(filename)[0] or default

def encode_multipart_formdata(fields, files):
    """Encodes a multipart form request that contains normal fields as well as files to upload.
    fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return (content_type, body) ready for httplib.HTTP instance
    """
    BOUNDARY = '----------ThIs_Is_tHe_bouNdaRY_$'
    CRLF = '\r\n'
    L = []
    for (key, value) in fields:
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"' % key)
        L.append('')
        L.append(value)
    for (key, filename, value) in files:
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
        L.append('Content-Type: %s' % get_content_type(filename))
        L.append('')
        L.append(value)
        L.append('--' + BOUNDARY + '--')
    L.append('')
    body = CRLF.join(L)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body

def postFilesToURL(url, fields, files):
    """Posts the given fields and files to an http address as multipart/form-data.
    fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return the server's response as an httplib.HTTPResponse object.
    Note that this doesn't catch any exceptions. You should catch them yourself.
    """
    import urlparse, httplib
    # parse url into host and path
    urlparts = urlparse.urlsplit(url)
    host, path = urlparts[1], urlparts[2]
    # encode request
    content_type, body = encode_multipart_formdata(fields, files)
    # make the http request, with the appropriate headers
    h = httplib.HTTPConnection(host)
    h.putrequest('POST', path)
    h.putheader('content-type', content_type)
    h.putheader('content-length', str(len(body)))
    h.endheaders()
    # send the request and return the response.
    h.send(body)
    resp = h.getresponse()
    return resp


# SECURITY/AUTH
def makesalt(saltlen=128):
    """Makes a random salt.
    DEPRECATED: use the bcrypt-based hashpass() instead!
    """
    import random, string
    print >>sys.stderr, 'Warning: makesalt() is deprecated. Use hashpass(), which uses bcrypt, directly'
    rchar = lambda: random.choice(string.letters+string.digits)
    salt = ''.join([rchar() for i in range(saltlen)])
    return salt

def oldhashpass(s, niters=1001):
    """Makes a hashed pass from the given string, stretching by the given number of iterations.
    DEPRECATED: use the bcrypt-based hashpass() instead!"""
    import hashlib
    print >>sys.stderr, 'Warning: oldhashpass() is deprecated. Use hashpass(), which uses bcrypt, directly'
    hash = s
    for i in range(niters):
        hash = hashlib.sha256(hash).hexdigest()
    return hash

def hashpass(s, hashpw=None, workfactor=10):
    """Hashes a given string (usually a password) using bcrypt.
    Call it with just a password to hash it for the first time.  To check a
    user-supplied password for correctness, call it with the user-given
    password and the hashed password from a previous call to hashpass(), and
    check that the output matches the hashed password.

    This is the right way to hash passwords.
    (See http://codahale.com/how-to-safely-store-a-password/ )
    Do NOT use the old makesalt() function...bcrypt generates special salts and
    prepends them to generated hashes, to prevent having to store them
    separately. Thus, you can simply call hashpass(tocheck, hashedpasswd) to
    check a password for validity, without screwing around with concatenating
    salts or storing them separately in databases. This is portable across all
    implementations of bcrypt (tested using python's py-bcrypt and Ruby's
    bcrypt-ruby).  The workfactor determines the exponential cost of hashing a
    password.  In early 2011, a workfactor of 10 (the current default) takes
    about .01 seconds to run, which is a good compromise between speed and
    slowness (for protecting against brute-force attacks).

    The resulting hashed password is 60 characters long, and can include
    letters, digits, and [$./]
    """
    import bcrypt
    if hashpw:
        return bcrypt.hashpw(s, hashpw)
    return bcrypt.hashpw(s, bcrypt.gensalt(workfactor))

class AuthException(Exception): pass
class InvalidUserException(AuthException): pass
class InvalidPasswordException(AuthException): pass

def oldauth(username, passwd, getuserfunc):
    """Checks the given username and password and returns a dict with fields username, userid.
    DEPRECATED: use the bcrypt-based auth() instead!
    Relies on a 'getuserfunc(username)' function which should query the appropriate databases
    and return a dict with {username: username, passwd: hashpass, salt: salt, id: userid} or None on error.
    Note that this can raise exceptions, which are guaranteed to be subclasses of AuthException.
    """
    u = getuserfunc(username)
    if not u:
        raise InvalidUserException
    salt, hash = u['salt'], u['passwd']
    newhash = oldhashpass(passwd+salt)
    if hash != newhash:
        raise InvalidPasswordException
    return dict(username=u['username'], userid=u['id'])

def auth(username, passwd, getuserfunc):
    """Checks the given username and password and returns a dict with fields username, userid.
    Relies on a 'getuserfunc(username)' function which should query the appropriate databases
    and return a dict with {username: username, passwd: hashedpass, id: userid} or None on error.
    Note that this can raise exceptions, which are guaranteed to be subclasses of AuthException.
    """
    u = getuserfunc(username)
    if not u:
        raise InvalidUserException
    if hashpass(passwd, u['passwd']) != u['passwd']:
        raise InvalidPasswordException
    return dict(username=u['username'], userid=u['id'])

def testauth():
    """Tests auth-related functions"""
    # check the hashpass() base function
    p = 'dofij'
    pw = hashpass(p)
    print pw
    x = hashpass(p, pw)
    print x
    assert x == pw, 'The two passes should be identical'
    # check the auth() wrapper
    u = 'user 1'
    p = 'user password'
    hashpw = hashpass(p)
    userfunc = lambda uname: dict(username=uname, passwd=hashpw, id=1)
    x = auth(u, p, userfunc)
    print 'Should be Valid: ', x
    try:
        x = auth(u, 'wrong password', userfunc)
        print 'Should never get here: ', x
    except Exception, e:
        print 'Should get InvalidPasswordException: got %s: %s' % (type(e), e)
    try:
        x = auth(u, 'user password', lambda u: None)
        print 'Should never get here: ', x
    except Exception, e:
        print 'Should get InvalidUserException: got %s: %s' % (type(e), e)

def cleanparameters(parameters, torem='username passwd passwdconf Login Register'.split()):
    """Cleans parameters by removing sensitive information from it"""
    for k in torem:
        if k in parameters:
            del parameters[k]
    return parameters

def simplefilter(lst, start=0, ss=1, num=50, **kw):
    """Given a set of parameters, filters the list."""
    ret = []
    for i, el in enumerate(lst[int(start):]):
        if len(ret) >= int(num): break
        if (i-start) % int(ss) != 0: continue
        ret.append(el)
    return ret

# WEB CLASSES
class robots:
    """A default robots.txt handler that disallows everything"""
    def GET(self):
        return 'User-agent: *\nDisallow: /'


# this is how the main method should look for most webapps:
if __name__ == '__main__':
    app.run()
    #testauth()
