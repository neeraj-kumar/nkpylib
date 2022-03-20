"""Various string utils"""

import codecs
import csv
import fcntl
import hashlib
import json
import locale
import math
import os
import random
import re
import string
import sys
import tempfile
import time

from difflib import SequenceMatcher
from io import StringIO
from pprint import pprint
from random import choice, randint
from shutil import copy2
from subprocess import call, PIPE
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import url2pathname, urlretrieve

#: Mapping from filename extensions to canonical 3-letter form
IMAGE_EXTENSIONS = {
                    'jpg': 'jpg', 'jpeg': 'jpg', 'jp2': 'jpg',
                    'gif': 'gif',
                    'png': 'png',
                    'bmp': 'bmp',
                    'tiff': 'tif', 'tif': 'tif',
                    'ppm': 'ppm',
                    'pgm': 'pgm',
                   }

def getListAsStr(lst, sep=',', fmt='%s'):
    """Returns a list of values as a string using the given separator and formatter"""
    return sep.join([fmt % (x,) for x in lst])

def utf(obj):
    """Converts the given object to a unicode string.
    Handles strings, unicodes, and other things as well."""
    if isinstance(obj, unicode):
        s = obj
    elif isinstance(obj, str):
        s = unicode(obj, 'utf-8')
    else:
        s = unicode(str(obj), 'utf-8')
    return s

def numseqstr(seq, sep=',', fmt='%0.2f'):
    """Takes a number sequence or single number and prints it using the given separator and format."""
    def gets(s):
        """Returns a number formatting string using fmt if a number, else str() version"""
        try: # number
            return fmt % s
        except TypeError: # non-number
            return str(s)
    try:
        # sequence
        seq = [s for s in seq]
        return sep.join(gets(s) for s in seq)
    except TypeError:
        return gets(seq)

def urlquote(url):
    """Quotes a url for submission through forms, etc."""
    if isinstance(url, unicode):
        url = url.encode('utf-8', 'replace')
    if '//' in url:
        prot, rest = url.strip().split('//',1)
        return prot + '//' + quote(rest)
    else:
        return quote(url.strip())

def urlunquote(url):
    """Unquotes a url for submission through forms, etc."""
    if '//' in url:
        prot, rest = url.strip().split('//',1)
        return prot + '//' + unquote(rest)
    else:
        return unquote(url.strip())

def blockindent(s, indent='\t', initial='\t'):
    """Block-indents a string by adding the given indent to the start of every line.
    The default indent is 1 tab.  You can also provide an initial indent to
    apply at the beginning of the string (defaults to 1 tab).
    """
    if initial:
        s = initial + s
    s = s.replace('\n', '\n'+indent)
    return s

def replaceTill(s, anchor, base):
    """Replaces the beginning of this string (until anchor) with base"""
    n = s.index(anchor)
    return base + s[n:]

def convertBasePath(objs, anchor, base):
    """Converts the path of this set of objects so that everything before 'anchor' gets converted to 'base'"""
    if not objs:
        return
    oldpaths = [o.path for o in objs]
    #print "Basename was %s and the first object originally had path %s" % (base, objs[0].path)
    for obj in objs:
        obj.path = replaceTill(obj.path, anchor, base)
    sys.stderr.write("Converted base paths to '%s'\n" % (base))
    return oldpaths

def getKWArgsFromArgs(args=None):
    """Returns a dictionary of keys and values extracted from strings.
    This splits each arg using '=', and then evals the second part.
    WARNING: This is very unsafe, so use at your own-risk!"""
    kw = {}
    if not args:
        return kw
    for a in args:
        k, v = a.split('=', 1)
        try:
            kw[k] = eval(v)
        except (NameError, SyntaxError):
            kw[k] = v
    return kw

def numformat(num, fmt='%d'):
    """Formats a number nicely, with commas.
    You can optionally give a custom format, e.g. for floats"""
    if isinstance(num, basestring):
        if '%d' in num:
            try:
                num = int(num)
            except ValueError:
                return num
        else:
            try:
                num = float(num)
            except ValueError:
                return num
    locale.setlocale(locale.LC_ALL, '')
    return locale.format(fmt, num, grouping=1)

def intOrNone(v, default=0, exctype=Exception):
    """Returns the int value of the given value, or default (which is normally 0) on error.
    Catches exceptions of the given exctype (Exception by default)"""
    try:
        return int(v)
    except exctype:
        return default

def floatOrNone(v, default=0.0, exctype=Exception):
    """Returns the float value of the given value, or default (which is normally 0.0) on error.
    Catches exceptions of the given exctype (Exception by default)"""
    try:
        return float(v)
    except exctype:
        return default

def makesafe(s):
    """Makes the given string "safe" by replacing spaces and lower-casing (less aggressive)"""
    def rep(c):
        if c.isalnum():
            return c.lower()
        else:
            return '_'
    ret = ''.join([rep(c) for c in s])
    return ret

def safestr(s, validchars=string.ascii_letters + string.digits+'-.', rep='_', collapse=1):
    """Makes the given string very safe (super aggressive).
    Limits characters to be in the given set of validchars, and lowercases them.
    Any illegal character is replaced by the given 'rep'.
    If collapse is true, then also goes through and collapses all instances of
    the rep character so there's only 1 at most."""
    # make a dictionary mapping which lowercases
    d = dict([(c,c.lower()) for c in validchars])
    out = []
    if type(s) == str:
        s = unicode(s, 'utf-8')
    for c in s.encode('utf-8', 'replace'):
        # get the appropriate map, or the replacement character
        c = d.get(c, rep)
        # if we want to collapse, don't put copies of the rep character on the output
        if collapse and out and c == rep and out[-1] == rep:
            continue
        out.append(c)
    outs = ''.join(out)
    return outs

def stringize(opts, safefunc=safestr, delim='_'):
    """Stringizes a set of opts by join()-ing safe versions of keys and values in opts.
    opts must be a list of pairs, NOT a dict!
    Both the key and value are run through the given safefunc, defaulting to safestr()."""
    s = delim.join('%s-%s' % (safefunc(k),safefunc(v)) for k, v in opts)
    return s

def randstr(nchars=10):
    """Creates a random string identifier with the given number of chars"""
    chars = string.ascii_letters + string.digits
    ret = ''.join([random.choice(chars) for i in range(nchars)])
    return ret

def host2dirname(url, safefunc=safestr, collapse=1, delport=1):
    """Converts the hostname from a url to a single directory name.
    This gets the hostname from the url, optionally collapses some
    elements from the beginning, and then makes it safe using the given function.
    If delport is true, then also removes port information from the host.

    The collapsing includes:
        - removing www from the beginning
        - keeping the last 2 elements
        - if the 2nd-to-last element is special (.com., .ac., etc.), then keeps last 3 elements.

    This function assumes you're giving it a URL. However, it'll
    check for a // somewhere in the url, and if not found, then assumes it's a host
    """
    h = urlparse(url).netloc if '//' in url else url
    if delport and ':' in h:
        h = h.split(':')[0]
    if collapse:
        # only collapse if it's not a numeric ip
        if [c for c in h if c not in '1234567890.']:
            # first get rid of initial www
            if h.startswith('www.'):
                h = h.replace('www.', '', 1)
            # now figure out how many elements to shorten down to
            minlen = 2
            # these are special cases for other countries, like .com.br, or .ac.uk, etc.
            prefixes = 'com co gov edu org ac net ne go gob unam govt sapo academic'.split()
            if len(h.split('.')) > 2:
                if h.split('.')[-2] in prefixes:
                    minlen = 3
            # chop off all beginning elements, maintaining minlen
            h = '.'.join(h.split('.')[-minlen:])
    h = '.'.join(safefunc(u) for u in h.split('.'))
    return h

def shortenurl(url, maxwidth=50):
    """Shortens the url reasonably"""
    if len(url) <= maxwidth:
        return url
    url = url.replace('http://','')
    if len(url) <= maxwidth:
        return url
    if url.startswith('www'):
        url = url[3:]
    if len(url) <= maxwidth:
        return url
    els = url.split('/')
    els[1] = '...'
    u = '/'.join(els)
    if len(u) <= maxwidth:
        return u
    while len(els) > 3:
        del els[2]
        u = '/'.join(els)
        if len(u) <= maxwidth:
            break
    return u

def strsim(a, b, weights={'difflib':1.0, 'longest_ratio':5.0, 'matching_ratio':6.0, 'exact':5.0}):
    """Returns the similarity between two strings as a float from 0-1 (higher=similar).
    Computes a few different measures and combines them, for optimal matching.
    The methods are linearly weighted using the dictionary 'weights'. It contains:
        'difflib': The ratio given by difflib's SequenceMatcher.ratio()
        'longest_ratio': Ratio of longest match over min(len)
        'matching_ratio': Ratio of sum(len(block)/minlen for block in get_matching_blocks())
        'exact': Scores for exact matches at beginning or end (very high), or in the middle (less high)
    Some default weights are defined for optimal file-renaming performance.
    You can tweak weights as needed.
    """
    minlen = float(min(len(a), len(b)))
    maxlen = float(max(len(a), len(b)))
    sm = SequenceMatcher(None, a, b)
    # compute the ratio given by difflib
    r0 = sm.ratio()
    # compute the ratio of the longest match over the minlen
    lm = sm.find_longest_match(0, len(a), 0, len(b))
    r1 = lm[2] / minlen
    # compute ratio of sum of all matching block lengths over the minlen
    totmatch = sum(m[2] for m in sm.get_matching_blocks())
    r2 = totmatch / minlen
    # add a score if we have an exact substring match
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        r3 = 1.0
    elif a in b or b in a:
        r3 = minlen/maxlen
    else:
        r3 = 0.0
    # add weighted combinations of the different vars
    vars = {'difflib': r0, 'longest_ratio': r1, 'matching_ratio': r2, 'exact': r3}
    total = 0.0
    ret = 0.0
    for k, w in weights.items():
        ret += vars[k] * w
        total += w
    ret /= total
    DEBUG = 0
    if DEBUG:
        print(a, b, '\n\t', r0, r1, r2, r3, ret, file=sys.stderr)
    return ret

def matrixprint(m, width=None, fillchar=' ', sep=' ', rowsep='\n', fmt='%0.2f'):
    """Returns a string representation of a matrix of strings using the given separator.
    Each string is center()-ed using the given width and fillchar.
    If width is None (default), then uses the maximum width of the strings+2"""
    if not width:
        width = max(len(numseqstr(s, fmt=fmt)) for s in flatten(m))
    ret = []
    for row in m:
        # convert Nones to empty strings
        row = [(numseqstr(s, fmt=fmt) if s else '') for s in row]
        row = sep.join(s.center(width, fillchar) for s in row)
        ret.append(row)
    return rowsep.join(ret)

def tempfname(**kw):
    """Creates a temporary filename using mkstemp with nice names.
    You can pass in:
        prefix: [default: int(time.time()*1000)]
        suffix: [default: random 10-char string.ascii_letters]

    Note that this does introduce a race condition, but it's usually okay.
    """
    kw.setdefault('prefix', '%d_' % (int(time.time()*1000)))
    kw.setdefault('suffix', '%s' % (''.join(choice(string.ascii_letters) for i in range(10))))
    tempf, fname = tempfile.mkstemp(**kw)
    os.close(tempf)
    return fname

def cleanimgext(fname):
    """Returns the image filename with the extension fixed.
    If it doesn't seem like a valid image name, then only lowercases"""
    try:
        fname, ext = fname.rsplit('.', 1)
    except ValueError:
        return fname
    ext = ext.lower().split('?')[0].split('&')[0]
    return fname + '.' + IMAGE_EXTENSIONS.get(ext, ext)

def url2fnamefmtdict(url):
    """Converts a url into a dictionary of strings to use for creating a filename.
    This includes:
        scheme, netloc, path, params, query, fragment, username, password, hostname, port - from urlparse
        basefname - the last part of the url, with extension
        basename - the last part of the url, without extension
        ext - the extension
        q-%(query param)s - the value of the given query param
        path%d - the d'th element in the path (stripped and split by '/')
        pathn%d - the -d'th element in the path (stripped and split by '/')
        hel%d - the d'th element in the hostname (split by '.')
        heln%d - the -d'th element in the hostname (split by '.')
        rand%d - random safe string of length d, upto 32
        time - current time, in milliseconds
        md5url - md5 of the url
        md5path - md5 of the path
        md5basefname - md5 of the basefname
    Returns a dict of strings.
    """
    p = urlparse(url)
    ret = dict(url=url)
    for k in 'scheme netloc path params query fragment username password hostname port'.split():
        ret[k] = getattr(p, k)
    ret['basefname'] = os.path.basename(p.path)
    els = ret['basefname'].rsplit('.', 1)
    ret['basename'] = els[0]
    ret['ext'] = els[1] if len(els) > 1 else ''
    pels = p.path.strip('/').split('/')
    for i, pel in enumerate(pels):
        ret['path%d' % i] = ret['pathn%d' % (len(pels)-i)] = pel
    if p.hostname:
        hels = p.hostname.split('.')
        for i, hel in enumerate(hels):
            ret['hel%d' % i] = ret['heln%d' % (len(hels)-i)] = hel
    rs = randstr(nchars=32)
    for i in range(len(rs)):
        ret['rand%d' % (i)] = rs[:i+1]
    ret['time'] = int(time.time()*1000)
    for k in 'url path basefname'.split():
        ret['md5'+k] = hashlib.md5(utf(ret[k]).encode('utf-8')).hexdigest()
    qels = parse_qs(ret['query'], keep_blank_values=1)
    for k, v in qels.items():
        ret['q-'+k] = v
    return ret

def url2fname(url, basedir='', maxlen=250, safefunc=safestr):
    """Returns a SAFE fname to use for downloading the given url. This means:
        Length considerations:
            - Filenames/dirnames are not too long (set by maxlen)
                - Note that names are cut at the end, so you may end up with dupes
                - Also, the final fname len might be up to maxlen + len('.') + len(ext)

        Illegal characters (handled using safefunc, which is safestr by default):
            - No unicode
            - No weird characters like | , @, :, etc.

        Path-specific character handling:
            - No pathname elements start with '.'
            - Filenames end with .ext (lowercase)
            - Image extensions are normalized by type
            - If the fname had a valid extension with the period, we fix it
            - If any path element is empty (including fname without ext), then 'temp_%06d' % (rand) is used.
    """
    fname = url2pathname(os.path.basename(url))
    try:
        fname, ext = fname.rsplit('.', 1)
    except ValueError:
        # see if the fname happens to end with an extension but without the period
        ext = ''
        for test, realext in IMAGE_EXTENSIONS.iteritems():
            if fname.lower().endswith(test):
                fname = fname.rsplit(test, 1)[0]
                ext = realext
                break

    path = os.path.join(basedir.encode('utf-8', 'replace'), fname.encode('utf-8', 'replace'))
    def fix(s):
        """Fixes a given string"""
        #print '  Got input: %s' % (s,),
        s = safefunc(s)[:maxlen]
        while s.startswith('.'):
            s = s[1:]
        if not s:
            s = 'temp_%06d' % (randint(0, 999999))
        #print ' and returning: %s' % (s,)
        return s

    path = u'/'.join(fix(el) for el in path.split('/'))
    if ext:
        path += safefunc(cleanimgext('.'+ext))
    path = re.sub(r'\.+', '.', path)
    return path

def dlFileIfNeeded(f, repfunc=lambda f:replaceTill(f, '/db/', 'http://leaf.cs.columbia.edu')):
    """Downloads a file if needed"""
    if os.path.exists(f):
        return f
    url = repfunc(urllib.quote(f))
    # create the parent directories if needed
    try:
        os.makedirs(os.path.dirname(f))
    except OSError:
        pass
    # download the file
    try:
        fname, headers = urllib.urlretrieve(url, f)
        assert fname == f
        return f
    except IOError as e:
        #print "Error: %s, %s, %s" % (url, f, e)
        return None

def downloadfile(url, outdir='.', outf=sys.stderr, delay=1):
    """Downloads a file from the given url.
    Returns the local file path, or None if doesn't exist/couldn't download.
    This function tries to be smart about things, especially multi-threading issues.
    If the url is a local path, then simply returns that"""
    if url.startswith('http'):
        path = urlparse(url).path[1:] # strip the leading / from the path
        outpath = os.path.join(outdir, path)
        if not os.path.exists(outpath):
            # check for temp file existence, so multiple threads don't all try to download at once
            temp = outpath + '_dl_temp_%d' % (int(time.time())//100) # the temp file is accurate to the 100s of secs
            print('Trying to download path %s to %s via temp name %s' % (url, outpath, os.path.basename(temp)), file=outf)
            while os.path.exists(temp) and not os.path.exists(outpath):
                print('  Detected temp file %s, just sleeping for %s' % (temp, delay), file=outf)
                time.sleep(delay)
                temp = outpath + '_dl_temp_%d' % (int(time.time())//100) # the temp file is accurate to the 100s of secs
            # another check for outpath existence
            if not os.path.exists(outpath):
                # if we're here, then we need to download the file
                try:
                    os.makedirs(os.path.dirname(temp))
                except OSError:
                    pass
                t1 = time.time()
                temp, headers = urlretrieve(url, temp)
                elapsed = time.time()-t1
                # rename it atomically to the right name
                try:
                    os.rename(temp, outpath)
                except Exception:
                    pass
                s = os.stat(outpath).st_size
                print('Downloaded %s to %s in %0.3fs (%s bytes, %0.1f bytes/sec)' % (url, outpath, elapsed, s, s/elapsed), file=outf)
                try:
                    os.remove(outtemp)
                except Exception:
                    pass
    else:
        outpath = url
    # at this point, we've downloaded the file if we needed to
    if os.path.exists(outpath):
        return outpath
    return None

def checkForFiles(fnames, progress=None):
    """Checks to see if the given files exist, otherwise downloads them"""
    ret = []
    for i, f in enumerate(fnames):
        if progress:
            progress('  Downloading %d of %d: %s...' % (i, len(fnames), f))
        f = dlFileIfNeeded(f)
        if f:
            ret.append(f)
    if progress:
        progress("Done downloading files\n")
    return ret

def getArg(seq, index, default=None, func=lambda x: x):
    """Returns func(seq[index]), or 'default' if it's an invalid index"""
    try:
        return func(seq[index])
    except IndexError:
        return default

def cleanDirTree(p, ntimes=-1):
    """Cleans the given directory tree by deleting all directory trees with no files"""
    done = 0
    iters = 0
    while not done:
        ndel = 0
        for root, dirs, files in os.walk(p, topdown=0):
            if not files and not dirs:
                os.rmdir(root)
                ndel +=1
        iters += 1
        if iters == ntimes:
            return # we've done this many repetitions
        if ntimes < 0 and ndel == 0:
            return # we've done all the deletions we can

def specialize(v):
    """Specializes a string value into an int or float or bool"""
    if not isinstance(v, basestring):
        return v
    if v.strip() == 'True':
        return True
    if v.strip() == 'False':
        return False
    try:
        # see it's an int...
        v = int(v)
    except (ValueError,TypeError):
        # maybe it's a float...
        try:
            v = float(v)
        except (ValueError,TypeError):
            pass
    return v

def specializeDict(d):
    """Takes a dictionary and for each value, sees if it can be cast to an int or a float"""
    for k, v in d.iteritems():
        d[k] = specialize(v)
    return d

def readNLines(f):
    """Reads a line which contains the number of future lines to read, followed by that many lines.
    Returns as a list of stripped strings"""
    try:
        n = int(f.readline().strip())
    except ValueError:
        return []
    ret = [f.readline().strip() for i in xrange(n)]
    ret = [l for l in ret if l]
    return ret

def detectdelimiter(fname):
    """Detects the delimiter of the given datafile.
    Returns one of ',', '\t', ' ', or None on error."""
    dlms = ['\t', ',', ' ']
    f = open(fname)
    header = f.readline().strip()
    # keep reading lines until we have a non-header line
    while 1:
        curline = f.readline().strip()
        if not curline.startswith('#'):
            break
        header = curline
    # figure out the format by comparing number of delimiters
    #print 'Got header line %s' % (header)
    for dlm in dlms:
        hnum = len(header.split(dlm))
        cnum = len(curline.split(dlm))
        #print 'For dlm "%s" got hnum %s, cnum %s' % (dlm, hnum, cnum)
        if 1 < cnum <= hnum <= cnum+1: # the +1 because there's an extra delimiter due to the '#'
            return dlm
    return None

def opts2dict(opts):
    """Converts options returned from an OptionParser into a dict"""
    ret = {}
    for k in dir(opts):
        if callable(getattr(opts, k)):
            continue
        if k.startswith('_'):
            continue
        ret[k] = getattr(opts, k)
    return ret

def openVersionedFile(fname, mode='wb'):
    """Opens a file for writing with the given name.
    If the file already exists, it is renamed to 'fname-%Y%m%d-%H%M%S'"""
    if os.path.exists(fname):
        newname = '%s-%s' % (fname, getTimestamp(fmt='%Y%m%d-%H%M%S'))
        os.rename(fname, newname)
    return open(fname, mode)

def saveandrename(fname, func, retfile=1, infork=0, mode='wb'):
    """Opens a new file with a tempfilename, runs the given func, then renames it to the given fname.
    The func is run with the file if retfile=1, else with the tmpfname.
    Creates parent dirs."""
    if infork:
        pid = os.fork()
        if pid != 0:
            return # the parent process just returns out
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    tmpfname = fname+'.tmp-%d' % (int(time.time()*1000))
    if retfile:
        f = open(tmpfname, mode)
        func(f)
        f.close()
    else:
        func(tmpfname)
    os.rename(tmpfname, fname)
    if infork:
        os._exit(0)


def savejson(data, fname, delay=60, lastsave=[0], savebackups=1):
    """Safe save function, which optionally makes backups if elapsed time has exceed delay.
    This is "safe" in that it writes to a temp file, then atomically renames to target fname when done.
    It uses a closure on lastsave[0] to determine when the last time we made a backup was.
    """
    t1 = time.time()
    if time.time()-lastsave[0] > delay:
        try:
            if savebackups:
                copy2(fname, '%s.bak-%d' % (fname, time.time()))
            lastsave[0] = time.time()
        except Exception:
            pass
    tmpname = fname + '.tmp-%d' % (time.time())
    kw = {}
    if json.__name__ == 'ujson':
        kw = dict(encode_html_chars=False, ensure_ascii=False)
    else:
        kw = dict(sort_keys=1, indent=2)
    try:
        os.makedirs(os.path.dirname(tmpname))
    except OSError:
        pass
    json.dump(data, open(tmpname, 'wb'), **kw)
    os.rename(tmpname, fname)

def readLinesOfVals(fname, convfunc=lambda vals, fields: vals, prefunc=lambda l: l, func=lambda d: d, dlm=' ', offset=0, maxlines=-1, onlyheaders=0):
    """Reads data values from the given fname.
    Implementation for readListOfVals and readDictOfVals.

    Input parameters:
           fname - the filename to read data from
         prefunc - if given, then used to filter and remap values (prior to convfunc). i.e.:
                       lines = (prefunc(l) for l in lines if prefunc(l))
                   but note that it doesn't actually call the prefunc twice
        convfunc - function which takes list of values from a line and returns a val (list or dict)
            func - if given, then used to filter and remap values (after convfunc). i.e.:
                        f = func(f)
                        if not f: continue
             dlm - IGNORED. we do automatic delimiter checking now
          offset - the data row offset number
        maxlines - if positive, then only that many datalines are read
     onlyheaders - if true, then we only read the headers and return (fields, dlm)

    Returns (faces, fields), where faces is a list of data items (each corresponding to a row),
    and fields is a list of the data fields.

    The datafile can have any number of comments at the top (lines starting with '#'),
    but none once the data starts. At the top of the file, the last non-data line should have the
    fields in it. The possible delimiters for data (and in the fields row) are '\t', ' ', or ','.
    Note that in the last case, the fields line should not start with #, but in all others, it should.

    This function tries to be efficient about memory by using generators,
    but returns a list in the end, to prevent confusion.
    """
    # figure out the delimiter
    dlm = detectdelimiter(fname)
    assert dlm is not None
    # figure out how many lines of headers there are (the last one is assumed to have the fields)
    skipheaders = -1 # -1 because we expect at least one for the fields
    for i, line in enumerate(open(fname)):
        if not (dlm == ',' and i < 1) and not line.strip().startswith('#'):
            break
        skipheaders += 1
    # now actually read the file
    f = open(fname)
    firstline = f.readline().strip()
    while skipheaders > 0:
        firstline = f.readline().strip()
        skipheaders -= 1
    if ',' not in dlm: # , delimiters means a CSV file, which doesn't use the #
        firstline = firstline.split('#'+dlm, 1)[1]
    fields = firstline.split(dlm)
    if onlyheaders:
        return (fields, dlm)
    # read lines
    lines = (l.strip() for i, l in enumerate(f) if l.strip() and (maxlines < 0 or i < maxlines) and (i >= offset))
    # apply prefunc
    lines = (prefunc(l) for l in lines)
    # filter by prefunc
    lines = (l for l in lines if l)
    # convert to list, specialize
    faces = (convfunc(l.strip().split(dlm), fields) for l in lines)
    # apply function
    faces = (func(f) for f in faces)
    # filter, and start computations
    faces = [f for f in faces if f]
    return (faces, fields)

def readListOfVals(fname, dospecialize=1, **kw):
    """Reads a list of values from the given fname.
    If prefunc is given, then it's used as a prefilter on lines (before specialize()).
    If func is given, then it's used as both a filtering and slicing function:
        f = func(f)
        if not f:
            continue
    This function tries to be efficient about memory by using generators, but
    returns a list in the end, to prevent confusion.
    Returns (faces, fields)"""
    if dospecialize:
        convfunc = lambda vals, fields: map(specialize, vals)
    else:
        convfunc = lambda vals, fields: vals
    return readLinesOfVals(fname, convfunc=convfunc, **kw)

def readDictOfVals(fname, specialize=1, **kw):
    """Reads a dictionary of values from the given fname.
    If prefunc is given, then it's used as a prefilter on lines (before dict()).
    If func is given, then it's used as both a filtering and slicing function:
        f = func(f)
        if not f:
            continue
    This function tries to be efficient about memory by using generators, but
    returns a list in the end, to prevent confusion.
    Returns (faces, fields)"""
    if specialize:
        convfunc = lambda vals,fields: specializeDict(dict(zip(fields, vals)))
    else:
        convfunc = lambda vals,fields: dict(zip(fields,vals))
    return readLinesOfVals(fname, convfunc=convfunc, **kw)

def writeLinesOfVals(linevals, fields, fname, dlm=' ', **kw):
    """Implementation function for writeDictOfVals and writeListOfVals"""
    if fname == '-':
        outf = sys.stdout
    else:
        outf = open(fname, 'w')
    if dlm != ',':
        outf.write('#' + dlm)
    print(dlm.join(fields), file=outf)
    for vals in linevals:
        print(dlm.join(vals), file=outf)

def writeListOfVals(faces, fields, fname, **kw):
    """Prints data in 'faces' (assumed to be lists) using the fields given"""
    linevals = ((str(v) for v in f) for f in faces)
    return writeLinesOfVals(linevals, fields, fname, **kw)

def writeDictOfVals(faces, fields, fname, errfunc=lambda field: 'ERR', **kw):
    """Prints data in 'faces' using the fields given"""
    linevals = ((str(f.get(field, errfunc(field))) for field in fields) for f in faces)
    return writeLinesOfVals(linevals, fields, fname, **kw)

class CSVUnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    Taken from:
    http://stackoverflow.com/questions/15960044/python-write-unicode-to-csv-using-unicodewriter
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def pprintJson(inf, outf=sys.stdout):
    """Pretty-prints a json file"""
    s = inf.read()
    n1 = s.index('{')
    n2 = s.index(';')-1
    f = json.loads(s[n1:n2])
    pprint(f, outf)

def extractToDir(fname, dir):
    """Extracts the given file (zip, tgz, gz, tar.gz, tar) to the given dir.
    Returns the retcode"""
    ends = '.zip .tgz .tar.gz .tar'.split()
    try:
        type = [e for e in ends if fname.lower().endswith(e)][0]
    except IndexError:
        raise TypeError

    try:
        os.makedirs(dir)
    except OSError:
        pass

    if type == '.zip': # unzip
        args = ['unzip', '-qq', '-d', dir, fname]
    elif type in '.tgz .tar.gz .tar'.split(): # tar xf
        args = ['tar', '-C', dir, '-xf', fname]
    ret = call(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    return ret

def filechecksum(f, blocksize=2**20, hashfunc='md5'):
    """Fixed-memory checksum on file, by incrementally updating hash over file.
    The given file can be a string (filename) or a file-like object itself.
    Reads file in chunks of the given size, to handle large files.
    The hashfunc is either given as a string (looked up in hashlib) or should be a
    constructor that initializes with no parameters, and returns an object
    with update() and hexdigest() functions in it.
    """
    h = hashlib.new(hashfunc) if isinstance(hashfunc, basestring) else hashfunc()
    if isinstance(f, basestring):
        f = open(f)
    while 1:
        data = f.read(blocksize)
        if not data:
            break
        h.update(data)
    return h.hexdigest()

class FileLockException(Exception):
    pass

class FileLock(object):
    """A context-manager wrapper on a fcntl.flock()"""
    def __init__(self, f, shared=0):
        """Initializes a lock on the given file or file descriptor or filename.
        If shared is 0 [default], it's an exclusive lock (LOCK_EX).
        Else it's a shared lock (LOCK_SH).
        """
        if isinstance(f, basestring):
            f = open(f)
        self.f = f
        self.shared = shared
        self.locked = 0

    def acquire(self):
        """Acquire this lock, if possible.
        If not, raise a FileLockException."""
        if self.locked:
            return
        try:
            locktype = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
            fcntl.flock(self.f, locktype|fcntl.LOCK_NB)
            self.locked = 1
        except IOError:
            raise FileLockException

    def release(self):
        """Releases our lock"""
        if self.locked:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.locked = 0

    def __enter__(self):
        """Acquire a lock in a 'with' statement"""
        self.acquire()
        return self

    def __exit__(self, type, value, trackback):
        """Release a lock at the end of the 'with' statement"""
        self.release()


