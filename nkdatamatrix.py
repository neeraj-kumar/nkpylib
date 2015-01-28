#!/usr/bin/env python
"""
A way to keep dense matrices of values along with changing data.  In many
situations, we often want to keep a matrix of values around.  For examples,
extracting D features from N images = N x D matrix.  We would like to keep track
of this matrix, along with some metadata, in a way that can be updated in
parallel (at least on one machine), is reasonably fast and compact, but still
dynamic (i.e., we can add or remove rows or columns).

This is a way to do that.

It does this by creating two files, a .json and a .mmap, which contain the
metadata and data, respectively. To deal with the different things we might want
to do, while keeping the program and semantics relatively simple, it operates in
one of two modes:

    1. Update configuration mode. This is used when the structure of the matrix
    changes, i.e., adding or removing rows or cols. When this is happening, both
    files are locked so that no other process can modify or use either.

    2. Update data mode. This is the "normal" mode, i.e., when reading or
    updating the data matrix, but not changing the structure of it. In this
    case, nothing is locked. Note that this means that multiple processes
    reading the data matrix could potentially be overwriting each other. This
    might be what you want, and might not be, but you deal with it.


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

import os, sys, math, time
from array import array
from mmap import mmap, ACCESS_READ, ACCESS_WRITE, ACCESS_COPY
from itertools import *
import fcntl
from nkutils import FileLock, getTimeDiffs, procmem, getmem, nkgrouper, log, getListAsStr
import codecs
try:
    import simplejson as json
except ImportError:
    import json

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

def xcombine(*seqin):
    """Returns a generator which returns combinations of argument sequences
    for example xcombine((1,2),(3,4)) returns a generator; calling the next()
    method on the generator will return [1,3], [1,4], [2,3], [2,4] and
    StopIteration exception.  This will not create the whole list of 
    combinations in memory at once."""
    def rloop(seqin,comb):
        """recursive looping function"""
        if seqin:                  # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]  # add next item to current combination
                # call rloop wth remaining seqs, newcomb
                for item in rloop(seqin[1:],newcomb):   
                    yield item        # seqs and newcomb
        else:                          # processing last sequence
            yield comb                # comb finished, add to list
    return rloop(seqin,[])

def dims(arr):
    """Returns the dimensions of a given (possibly multi-d) array"""
    ret = []
    last = None
    while 1:
        try:
            ret.append(len(arr))
            arr = arr[0]
            if arr == last: # this checks for strings
                ret = ret[:-1]
                break
            last = arr
        except TypeError: break
    return ret

def iterall(arr):
    """Multidimensional-iterator: iterates over single elements in the array"""
    size = dims(arr)
    indices = [xrange(s) for s in size]
    all = xcombine(*indices)
    for lst in all:
        cur = arr
        for i in lst:
            cur = cur[i]
        yield cur
    raise StopIteration

class MmapArray(object):
    """A memory-mapped multi-dimensional array.
    Useful for communication with other processes, where you don't want
    expensive full-memory copies.  Note that python can't really do any useful
    processing on mmap'ed files, so it'll probably cause several small copies if
    you do extensive processing.

    The mmap is indexed using a 'loc', which is a tuple of d-dimensions with
    indices. This is internally mapped to a simple index using the index() func.
    """
    def __init__(self, fname, typecode, size, create=0, init=None, initval=0, access=ACCESS_WRITE, outf=sys.stderr):
        """Initializes this mmap'ed array using the given fname, typecode, and size.
        Parameters:
            typecode: as used in the array.array module.
                size: an array of d-dimensions for a d-dimensional array.
                      Each element is the length of that dimension.
                      If you have the array already, you can call dims(arr).
              create: If true [not the default], creates (overwrites) the mmap
                init: A matrix to init the mmap with.
                      It is an error to specify this and not create.
             initval: A single value to fill the entire matrix with
                      It is an error to specify this and not create.
                      If 'init' is given, this is ignored.
              access: One of ACCESS_READ, ACCESS_WRITE, ACCESS_COPY
                outf: A log file
        """
        from cStringIO import StringIO
        import operator as op
        from math import sqrt
        if not outf:
            outf = StringIO()
        self.fname = fname
        typecode = str(typecode)
        self.typecode = typecode
        self.size = size
        arr = array(typecode)
        self.itemsize = arr.itemsize
        self.nels = reduce(op.mul, size)
        self.rowlen = reduce(op.mul, size[1:]) if len(size) > 1 else 1
        self.memsize = self.nels*self.itemsize
        self.access = access
        if init:
            assert list(dims(init)) == list(size), 'Dims of init were %s, while given size was %s' % (dims(init), size)
            assert create, 'If given an init, create must be 1'
        print >>outf, "For typecode %s, and size %s, got %d itemsize, %d elements total, and %d memsize" % (typecode, size, self.itemsize, self.nels, self.memsize)
        if create:
            # create the file, and init with initvals, unless we have an init
            print >>outf, '  Creating mmap %s' % (fname)
            try:
                os.makedirs(os.path.dirname(fname))
            except OSError:
                pass
            f = open(fname, 'wb')
            # initialize the output in increments of sqrt(nels)
            #initsize = max(size)
            initsize = int(sqrt(self.nels))
            if init:
                # write in groups of the first dimension
                print >>outf, '  Initializing file using given init, in groups of size %d' % (initsize)
                groups = nkgrouper(initsize, iterall(init))
                for g in groups:
                    a = array(typecode, g)
                    a.tofile(f)
            else:
                # create functions for making initialization arrays
                if typecode in 'cu':
                    # strings are already lists
                    if not initval:
                        initval = '\0'
                    makearray = lambda num: array(typecode, initval*num)
                else:
                    makearray = lambda num: array(typecode, [initval]*num)
                print >>outf, '  Initializing file to val %s in groups of size %d...' % (initval, initsize)
                ntodo = self.nels
                a = makearray(initsize)
                while ntodo > initsize:
                    a.tofile(f)
                    ntodo -= initsize
                # finish out the rest of the elements
                if ntodo > 0:
                    a = makearray(ntodo)
                    a.tofile(f)
            f.close()
        # now open the mmap for real (checking to make sure it's the right size)
        f = open(fname, 'rb+')
        f.seek(0, os.SEEK_END)
        # if the file is smaller than the needed size, pad with 'initval'
        diff = self.memsize - f.tell()
        assert diff <= 0
        if diff > 0:
            #print 'diff > 0 %s, %s, %s' % (self.memsize, f.tell(), diff)
            f = open(fname, 'ab+')
            a = array(typecode, [initval]*(diff/self.itemsize))
            a.tofile(f)
            f.flush()
            #diff = self.memsize - f.tell()
            #print self.memsize, f.tell(), diff, initval
            f.close()
            f = open(fname, 'rb+')
            f.seek(0, os.SEEK_END)
        assert f.tell() >= self.memsize, 'Trying to use existing mmap with memsize %s, but file %s is only %s long!' % (self.memsize, fname, f.tell())
        f.seek(0)
        try:
            self.mmap = mmap(f.fileno(), self.memsize, access=access)
        except Exception:
            print f, f.fileno(), self.memsize, access, ACCESS_READ, ACCESS_WRITE, diff
            raise

    def index(self, loc):
        """Returns the index for a given loc. This takes into account the memory size of elements"""
        loc = loc[:]
        #while len(loc) < len(self.size): loc.append(0) # FIXME this seems wasteful
        index = 0
        size = 1
        #TODO all this seems very expensive
        for l, s in reversed(zip(loc, self.size)):
            assert l < s
            index += l*size
            size *= s
        ret = index*self.itemsize
        return ret

    def get(self, loc, nels):
        """Returns the first 'nels' values as a 1-d array, starting from the given loc.
        Wraps around in row-major order, and does not go off the mmap."""
        if nels == 0: return []
        n = self.index(loc)
        self.mmap.seek(n)
        a = array(self.typecode, self.mmap[n:n+(nels*self.itemsize)])
        return a

    def getrow(self, rownum):
        """Returns a full 'row', which is indexed by the first dimension only."""
        #return self.get(self.rowloc(rownum), self.rowlen)
        # we special case this method, as it is called very frequently
        n = rownum * self.rowlen * self.itemsize
        a = array(self.typecode, self.mmap[n:n+(self.itemsize*self.rowlen)])
        return a

    def rowloc(self, rownum):
        """Returns a full location for the start of the given row, as a tuple"""
        loc = (rownum,)+(0,)*self.rowlen
        return loc

    def set(self, loc, arr):
        """Sets values from the given array starting at the given loc.
        If 'arr' is not an array.array, tries to initialize an array using it."""
        if arr is None: return # ignore bad inputs
        try:
            t = arr.typecode
        except AttributeError:
            arr = array(self.typecode, arr)
        assert arr.typecode == self.typecode, "Typecodes don't match! (given: %s, self:%s)" % (arr.typecode, self.typecode)
        n = self.index(loc)
        self.mmap.seek(n)
        self.mmap.write(arr.tostring())

    def __iter__(self):
        """Returns an iterator for all our values"""
        for i in xrange(self.size[0]):
            for v in self.getrow(i):
                yield v

    def flush(self):
        self.mmap.flush()

    def toimage(self, imfunc=None, im=None, valrange=None, rows=None, cache={}):
        """Converts our mmap to an image.
        You can optionally pass in a custom imfunc, which should take in:
            loc: a tuple specifying the current location
            val: the value at the location in the mmap
        The imfunc should return a valid color.

        If not specified, the imfunc maps values using the 'jet' colormap
        (requires nkpylib.nkimageutils). Since this colormap requires inputs
        between 0-1, the actual values are first mapped to this range
        depending on our typecode:
            cB: [0,255]
            b: [-128,127]
            hi: [-32768,32767]
            HI: [0,65535]
            l: [-2**31, 2**31-1]
            L: [0,2**32-1]
            fd: [-10**5, 10**5]
        If you want to use this imfunc, but with a customrange, pass in a pair valrange.
        If you want to use the min and max from the data, set valrange='data'.
        You can optionally pass in a list of rows to render to the output.
        If not given, this is assumed to be the entire DM.
        You can optionally pass in an image to write to.
        If you don't, an 'RGB' image is created with size:
            (len(rows), self.rowlen)
        The imfunc is called for all locations and the result
        is set on the image using im.putdata().
        Returns the image.
        """
        if not imfunc:
            # use a jet colormap imfunc
            from nkpylib.nkimageutils import colormap
            from nkpylib.nkutils import lerp, clamp
            if not valrange:
                # if we're not given a value range, do it based on typecode
                rangedict = dict(c=[0,255], B=[0,255],
                                 b=[-128,127],
                                 h=[-32768,32767], i=[-32768,32767],
                                 H=[0,65535], I=[0,65535],
                                 l=[-2**31, 2**31-1],
                                 L=[0,2**32-1],
                                 f=[-10.0**5, 10.0**5], d=[-10.0**5, 10.0**5])
                valrange = rangedict[self.typecode]
            elif valrange == 'data':
                # do it based on data range
                valrange = min(self), max(self)
            # set up the actual imfunc
            from_ = (valrange[0],0.0)
            to = (valrange[1],1.0)
            def imfunc(loc, val, cache=cache):
                """Image func that lerps val using (from_, to) and returns a 'jet' color.
                Caches outputs for speed."""
                if val not in cache:
                    cache[val] = colormap(clamp(lerp(val, from_, to), 0.0, 1.0), 'jet')
                return cache[val]
        if not rows:
            rows = range(self.size[0])
        # these dims could be useful regardless of if we're creating the image or not
        dims = (len(rows), self.rowlen)
        if not im:
            from PIL import Image
            im = Image.new('RGB', dims)
        ret = []
        # iterate over first coordinate
        for i in rows:
            if i % 100000 == 0:
                #print '  On row %d' % (i)
                pass
            # get all locs for with the first coordinate fixed
            sublocs = [[i]]+[range(s) for s in self.size[1:]]
            locs = xcombine(*sublocs)
            # get all values in this row
            row = map(imfunc, locs, self.getrow(i))
            ret.extend(row)
        # set the data
        im.putdata(ret)
        return im


def testmmaparray():
    a = [1,2,3,2,3,5]
    b = [[5,10,1309],
         [3049,1,4]]
    c = ['cat', 'dog', 'cow', 'pig']
    m = MmapArray('junk', 'i', dims(b), init=b, create=1)
    #m = MmapArray('junk', 'i', [10000,100], create=0)
    print m.get([0,0], 5)
    m.set([1,1], [50, 1000])
    print m.get([0,0], 5)

def relpath(path, base):
    """Returns a path relative to the given base.
    If the given path is absolute, just returns it.
    Else, if base is a dir, then returns base+path.
    If base is a file, then returns os.path.dirname(base)+path.
    """
    if os.path.isabs(path): return path
    if os.path.isdir(base):
        return os.path.join(base, path)
    else:
        return os.path.join(os.path.dirname(base), path)


class DataMatrix(object):
    """A paired MmapArray and a JSON file."""
    def __init__(self, basename, mode='updatedata', errval=-99999, loadfields=1, other={}, outf=None, **initkw):
        """Initializes this datamatrix with the given basename.
        basename+'.json' and basename+'.mmap' will be used.
        The 'mode' must be one of the following:
            'readdata': Open the file only to read data (no changes of any kind)
            'updatedata': Open the file for data update only (no structure changes)
            'updatestruct': Open the file for structure update only (no data changes)
        If creating this for the first time (i.e., no json file yet),
        then make sure you pass in at least the following initializations:
            fields: an array of arrays. Each inside array can be empty, but there should be the right number.
            fieldtypes: an array of strings. The length of this array must match that of fields
            typecode: cannot be updated
            errval: must be valid for the typecode, and cannot be updated
        Optionally:
            fieldfnames: an array of filenames (or None) for each list in fields. This is good
                         if the field list is huge and you don't necessarily want to load it always.
        For 'updatestruct', fields should contain the full data (NO external links), but if fieldfnames
        is set correctly, then it will write out the fields to the external file rather than the json filename.
        For data modes (updatedata, readdata), you can choose whether to load external fields.
        If loadfields is 1 (default), then will load external fields from their files, using fieldfnames.
        Else, stores each external field as simply the length of that field, and fieldlookups for that field will be empty.
        Log messages are sent to the given outf if it's not None.
        """
        from copy import deepcopy
        times = [time.time()]
        self.loadfields = loadfields
        self.basename = basename
        try:
            os.makedirs(os.path.dirname(self.basename))
        except OSError: pass
        self.jsonfname = basename+'.json'
        self.mmapfname = basename+'.mmap'
        self.mode = mode
        assert mode in 'readdata updatedata updatestruct'.split()
        self.access = dict(updatestruct=ACCESS_READ, readdata=ACCESS_READ, updatedata=ACCESS_WRITE)[mode]
        self.outf = outf
        # if either file doesn't exist, we must be in struct mode
        createmmap = 0
        if not os.path.exists(self.jsonfname) or not os.path.exists(self.mmapfname):
            assert mode == 'updatestruct', 'Either json or mmap for %s did not exist, and mode was not "updatestruct"!' % (basename)
            # create files
            open(self.jsonfname, 'a+').close()
            createmmap = 1
        # if the json file is locked, we can't do anything
        self.lock = FileLock(self.jsonfname, shared=(0 if mode == 'updatestruct' else 1))
        with self.lock:
            # (when we die, the lock dies too)
            times.append(time.time())
            # read the json file and init from it
            try:
                self.j = json.load(open(self.jsonfname))
            except Exception:
                self.j = {}
            times.append(time.time())
            if self.j:
                if 'fieldfnames' not in self.j:
                    self.j['fieldfnames'] = [None] * len(self.j['fields'])
                if mode == 'updatestruct' or loadfields:
                    self.loadexternalfields()
                # init ourselves from the json file
                for f in 'fields fieldtypes fieldfnames typecode other errval'.split():
                    if mode == 'updatestruct':
                        self.__dict__[f] = deepcopy(self.j[f])
                    else:
                        self.__dict__[f] = self.j[f]
            times.append(time.time())
            mmapinitval = 0
            if mode == 'updatestruct':
                if not self.j:
                    # init everything
                    self.__dict__.update(initkw)
                    self.errval = errval
                    self.other = other
                else:
                    # update fieldtypes and fields if given
                    if 'fieldtypes' in initkw:
                        self.fieldtypes = initkw['fieldtypes']
                    if 'fields' in initkw:
                        curf, newf = self.fields, initkw['fields']
                        assert len(curf) == len(newf)
                        # add all new fields only
                        for curcol, newcol in zip(curf, newf):
                            existing = set(curcol)
                            for f in newcol:
                                if f not in existing:
                                    curcol.append(f)
                            mmapinitval = self.errval
                for f in self.fields:
                    assert not isinstance(f, (int, long))
            elif mode in 'readdata updatedata'.split():
                # we should have a valid j and ourselves
                assert self.j
                for f in 'fields fieldtypes fieldfnames typecode other errval'.split():
                    assert f in self.j
                    assert f in self.__dict__
            # load mmap
            self.mmap = MmapArray(self.mmapfname, typecode=self.typecode, size=self.fieldlens(), create=createmmap, access=self.access, outf=None, initval=mmapinitval)
            times.append(time.time())
            if 'fieldfnames' not in self.__dict__:
                self.fieldfnames = [None] * len(self.fields)
            # a final sanity check
            self.sanitycheck()
            times.append(time.time())
            #print getTimeDiffs(times)

    def log(self, s):
        """Logs a message to our outf"""
        if self.outf:
            print >>self.outf, s
            self.outf.flush()

    @classmethod
    def readrows(self):
        dm = DataMatrix(dmname, mode='readdata', loadfields=0)
        self.errval = dm.errval
        times.append(time.time())
        # create a lookup of the appropriate type
        self.log('Inputs were: %s' % (inputs[:5],))
        if indexmode == 'datamatrix-fnames':
            inset = set(inset) if inset else set(inputs)
            lookupdict = dm.getlookup(0, keys=inset, keytype='exact')
            lookupfunc = lambda input: lookupdict[unicode(input, 'utf-8', 'ignore')]
        elif indexmode == 'datamatrix-rownums':
            lookupfunc = lambda rownum: int(rownum)
        times.append(time.time())
        #self.log('About to go through inputs, and got times %s' % (getTimeDiffs(times),))
        # Iterate through inputs
        M = dm.mmap
        for input in inputs:
            try:
                rownum = lookupfunc(input)
                fvec = M.getrow(rownum)
                outdict[input].extend(fvec)
            except Exception: continue
        times.append(time.time())

    @classmethod
    def getfields(cls, dmname):
        """Returns (fieldtypes, fields) of the given dm, if valid, else None"""
        try:
            dm = DataMatrix(dmname, 'readdata')
            return (dm.fieldtypes, dm.fields)
        except Exception:
            return None

    def __getitem__(self, key):
        """Returns the item at the given key, based on its length and type:
            int: returns the given row (by index) of the mmap
            str: return given row (by lookup)
            seq and len=1: returns row (by lookup) of the mmap
            seq and len=2: returns the value (by lookup) of the mmap
        """
        try:
            rownum = int(key)
            #print 'For key %s, got rownum %d and returning row' % (key, rownum)
            return self.mmap.getrow(rownum)
        except Exception:
            pass
        # must be a field lookup
        if isinstance(key, basestring):
            key = [key]
        loc = self.fields2loc(key)
        if loc is None:
            raise KeyError('Could not find key %s in our fields' % (key,))
        #print 'For key %s, got loc %s with len %s' % (key, loc, len(key))
        if len(key) == 1:
            return self.mmap.getrow(loc[0])
        else:
            return self.mmap.get(loc)

    def __setitem__(self, key, val):
        """Sets the item at the given key, based on the key's length and type:
            int: sets the given row (by index) of the mmap
            str: sets given row (by lookup)
            seq and len=1: sets row (by lookup) of the mmap
            seq and len=2: sets the value (by lookup) of the mmap
        """
        loc = [0]*len(self.fields)
        try:
            loc[0] = int(key)
        except Exception:
            # must be a field lookup
            if isinstance(key, basestring):
                key = [key]
            loc = self.fields2loc(key)
            while len(loc) < len(self.fields):
                loc.append(0)
        try:
            self.mmap.set(loc, val)
        except Exception, e:
            print 'For key %s, tried setting at loc %s with %d len, but skipped because got %s exception: %s' % (key, loc, len(val), type(e), e)

    def fieldlens(self):
        """Returns lengths of each of our fields"""
        ret = [f if isinstance(f, (int,long)) else len(f) for f in self.fields]
        return ret

    def getlookup(self, fieldtypeOrIndex, keys=None, keytype='exact'):
        """Returns a lookup dict for the given fieldtype, for the given keys.
        If fieldtypeOrIndex is a string, then looks up the index in self.fieldtypes.
        Else, assumes it's a number.
        If that field is stored internally, simply returns our lookup table.
        Else, creates a dict from the external file.
        If keys is given, then limits dict to only 'matching' entries, defined using keytype:
            'exact' (default): each key is a string which has to equal the field
            'prefix': each key is a string prefix to match a line with
            'index': each key is a line number index to include
            'regexp': each key is a regexp to match against
            'func': keys is a function that takes (i, l, done) and returns true if it matches
        The returned dict maps inputs to row numbers.
        """
        import re
        if isinstance(fieldtypeOrIndex, basestring):
            n = self.fieldtypes.index(fieldtypeOrIndex)
        else:
            n = fieldtypeOrIndex
        if self.fieldlookups[n]: return self.fieldlookups[n]
        hadkeys = not (keys is None)
        times = [time.time()]
        if keytype == 'exact':
            keys = set(keys)
            times.append(time.time())
            def matches(i, l):
                ret = l in keys
                if not ret: return ret
                keys.remove(l)
                if not keys:
                    raise StopIteration(1)
                return ret
        elif keytype == 'prefix':
            keys = set(keys)
            def matches(i, l):
                """Returns true if l matches (starts with) any of the keys"""
                for k in keys:
                    if l.startswith(k): return 1
                return 0

        elif keytype == 'index':
            keys = set(keys)
            def matches(i, l):
                """Returns true if i is in keys"""
                ret = i in keys
                if not ret: return ret
                keys.remove(i)
                if not keys:
                    raise StopIteration(1)
                return ret

        elif keytype == 'regexp':
            keys = set(re.compile(k) if isinstance(k, basestring) else k for k in keys)
            def matches(i, l):
                """Returns true if l matches (starts with) any of the keys"""
                for k in keys:
                    if k.match(l): return 1
                return 0
        elif keytype == 'func':
            assert callable(keys)
            matches = keys

        times.append(time.time())
        ret = {}
        try:
            for i, l in enumerate(codecs.open(relpath(self.fieldfnames[n], self.basename), 'rb', 'utf-8')):
                l = l.rstrip('\n')
                if i % 100000 == 0:
                    self.log('  On row %d of %d' % (i, self.fields[n]))
                if hadkeys and not matches(i, l): continue
                ret[l] = i
        except StopIteration, e:
            if e.args and e.args[0]:
                ret[l] = i

        return ret

    def loadexternalfields(self):
        """Loads in external fields into self.j"""
        for i, (fieldfname, lst) in enumerate(zip(self.j['fieldfnames'], self.j['fields'])):
            if not fieldfname: continue
            self.j['fields'][i] = [l.rstrip('\n') for l in codecs.open(relpath(fieldfname, self.basename), 'r', 'utf-8')]

    def sanitycheck(self):
        """Runs a sanity check on ourselves"""
        assert len(self.fields) == len(self.fieldtypes)
        assert len(self.fields) == len(self.fieldfnames)
        for lst in self.fields:
            #FIXME assert len(set(lst)) == len(lst)
            pass
        assert self.typecode in 'cbBuhHiIlLfd'
        assert isinstance(self.other, dict)
        # pre-cache field-lookups
        # Each item in 'fields' is either a list, in which case we make a lookup table
        # Or it's a number, which is the length of that list, externally, and we store None
        dOrNone = lambda lst: dict((name, i) for i, name in enumerate(lst)) if not isinstance(lst, (int, long)) else None
        self.fieldlookups = [dOrNone(lst) for lst in self.fields]

    def fields2loc(self, fieldnames, fields=None, fieldlookups=None):
        """Given an array of field names, returns a loc, or None on error.
        Uses the fields in this class by default.
        You can specify an alternate fields array to check that.
        """
        if not fields:
            fields =  self.fields
            fieldlookups = self.fieldlookups
        try:
            if fieldlookups:
                ret = [lookup[f] for f, lookup in zip(fieldnames, fieldlookups)]
            else:
                ret = [lst.index(f) for f, lst in zip(fieldnames, fields)]
            return ret
        except (ValueError,KeyError):
            return None

    def __iter__(self):
        """Returns tuples of fieldnames iterating through our fields"""
        return xcombine(*self.fields)

    def flush(self):
        """Flushes all changes out to disk"""
        if self.mode == 'updatestruct':
            self.flushstruct()
        elif self.mode == 'updatedata':
            self.mmap.flush()

    def flushstruct(self):
        """Flushes structural changes out to disk.
        This creates a new mmap if structure has changed,
        writes data to it from the old one, and then renames it over.
        It also rewrites the json file, regardless of changes.
        """
        with self.lock:
            self.sanitycheck()
            cur, old = self.__dict__, self.j
            curdims = map(len, cur['fields'])
            # make sure new dims are all positive
            assert min(curdims) > 0
            if old:
                # we can't change sizes or typecodes or errval
                assert len(cur['fields']) == len(old['fields'])
                assert len(cur['fieldtypes']) == len(old['fieldtypes'])
                assert len(cur['fieldfnames']) == len(old['fieldfnames'])
                assert cur['typecode'] == old['typecode']
                assert cur['errval'] == old['errval']
                # if changes to fields, update mmap
                if cur['fields'] != old['fields']:
                    times = [time.time()]
                    oldlookups = [dict((name, i) for i, name in enumerate(lst)) for lst in old['fields']]
                    olddims = map(len, old['fields'])
                    #self.log('dims are %s, %s, fields are %s, %s' % (olddims, curdims, old['fields'], cur['fields']))
                    oldm = MmapArray(self.mmapfname, typecode=old['typecode'], size=olddims, access=ACCESS_READ, outf=None)
                    toup = []
                    times.append(time.time())
                    # special case to deal with only changed rows
                    if cur['fields'][1:] == old['fields'][1:]:
                        self.log( '  Only rows differ -- fast mapping!')
                        oldd = oldlookups[0]
                        oldf = old['fields'][0]
                        guess = 0
                        maxguess = len(oldf)-1
                        for newrow, field in enumerate(cur['fields'][0]):
                            if field == oldf[guess]:
                                # first check our guess, since this is a constant time op
                                oldrow = guess
                            else:
                                # look it up in the dict
                                oldrow = oldd.get(field, None)
                            if oldrow is None: continue # ignore new rows
                            toup.append((oldm.rowloc(oldrow), oldm.rowlen, oldm.rowloc(newrow)))
                            # our next guess is the current row + 1, limited by maxguess
                            guess = min(1+oldrow, maxguess)
                    else:
                        # figure out what oldlocs will map to newlocs
                        self.log('  Need to do full mapping')
                        for fieldnames in self:
                            #TODO perhaps we don't want to iterate on self?
                            oldloc = self.fields2loc(fieldnames, fields=old['fields'], fieldlookups=oldlookups)
                            if not oldloc: continue # ignore new fields
                            newloc = self.fields2loc(fieldnames)
                            toup.append((oldloc, 1, newloc))
                    times.append(time.time())
                    #self.log('toup is %s' % (toup[:100],))
                    # now actually copy over data
                    self.log(' Creating new mmap')
                    tmpfname = '%s_temp-%s' % (self.mmapfname, os.getpid())
                    newm = MmapArray(tmpfname, typecode=cur['typecode'], size=curdims, create=1, initval=cur['errval'], outf=None)
                    times.append(time.time())
                    self.log(' Copying over old data')
                    for oldloc, oldlen, newloc in toup:
                        newm.set(newloc, oldm.get(oldloc, oldlen))
                    times.append(time.time())
                    self.log('  Flushing mmap')
                    newm.flush()
                    times.append(time.time())
                    # close both mmaps and rewrite mmap
                    del newm, oldm
                    self.log('  Renaming %s to %s' % (tmpfname, self.mmapfname))
                    os.rename(tmpfname, self.mmapfname)
                    times.append(time.time())
                    self.log('  Flushstruct (data part) took times: %s' % (getTimeDiffs(times)))
            else:
                # no old, so just create the mmap
                newm = MmapArray(self.mmapfname, typecode=cur['typecode'], size=curdims, create=1, initval=cur['errval'], outf=None)
            # get old struct ready for json writing
            for f in 'fields fieldtypes fieldfnames typecode other errval'.split():
                old[f] = cur[f]
            # if we have any external fields, write them out
            for i, (fieldfname, lst) in enumerate(zip(old['fieldfnames'], old['fields'])):
                if not fieldfname: continue
                fieldfname = relpath(fieldfname, self.basename)
                tmpfname = '%s_temp-%s' % (fieldfname, os.getpid())
                self.log('  Writing %d entries to external fields tempfile %s' % (len(lst), tmpfname))
                f = codecs.open(tmpfname, 'wb', 'utf-8')
                for l in lst:
                    print >>f, l
                    #f.write(l+'\n')
                f.close()
                os.rename(tmpfname, fieldfname)
                old['fields'][i] = len(lst)
            # now write out the json file to a tmpfname and then rename it
            tmpfname = '%s_temp-%s' % (self.jsonfname, os.getpid())
            self.log('Writing new json to tempfile %s' % (tmpfname))
            json.dump(old, open(tmpfname, 'wb'), indent=2, sort_keys=1)
            os.rename(tmpfname, self.jsonfname)
            # finally, reload our own mmap
            self.mmap = MmapArray(self.mmapfname, typecode=self.typecode, size=curdims, create=0, access=self.access, outf=None)

# alias for old code
NKDataMatrix = DataMatrix

def copyDM(curdmname, newdmname):
    """Copies a DataMatrix from curdmname to newdmname"""
    from shutil import copy2
    assert curdmname != newdmname
    for ext in 'json mmap images'.split():
        old = curdmname+'.'+ext
        if not os.path.exists(old): continue
        new = newdmname+'.'+ext
        #print 'Copying %s to %s' % (old, new)
        try:
            os.makedirs(os.path.dirname(new))
        except OSError: pass
        copy2(old, new)

def dm2numpy(dmname, rownames, errval=999999):
    """Creates a numpy matrix from a dm using the given rownames.
    You can optionally pass in the error value to assign.
    """
    import numpy as np
    times = [time.time()]
    typecodemap = dict(b=np.byte, h=np.short, i=np.intc, l=np.int_, B=np.ubyte, H=np.ushort, I=np.uintc, L=np.uint, f=np.single, d=np.float_)
    dm = DataMatrix(dmname, 'readdata', loadfields=0)
    times.append(time.time())
    assert dm.typecode in typecodemap
    ret = np.ones((len(rownames), dm.mmap.rowlen), dtype=typecodemap[dm.typecode])*errval
    times.append(time.time())
    lookup = dm.getlookup(0, keys=rownames, keytype='exact')
    times.append(time.time())
    for i, fname in enumerate(rownames):
        try:
            rownum = lookup[fname]
            ret[i,:] = dm.mmap.getrow(rownum)
            if ret[i,0] == dm.errval:
                ret[i,:] = errval
        except Exception: pass
    return ret


def test():
    if not os.path.exists('testdm.mmap'):
        fields = ['im1 im2 im3'.split(), 'f1 f2 f3'.split()]
        fieldtypes = 'images feats'.split()
        d = DataMatrix('testdm', mode='updatestruct', typecode='f', fields=fields, fieldtypes=fieldtypes)
        d.flush()
    else:
        if 0:
            d = DataMatrix('testdm', mode='updatestruct')
            d.fields[0] = 'im1 im2 im3 im4 im5'.split()
            d.fields[1].append('f5')
            d.flush()
        else:
            d = DataMatrix('testdm', mode='updatedata')
            m = d.mmap
            ndims = len(d.fields[1])
            #m.toimage(valrange=[0,10.0]).save('whoa.png'); sys.exit()
            print d.fields, d.fieldtypes, d.errval, d.typecode, d.other, d.mmap, ndims
            for i, im in enumerate(d.fields[0]):
                v = m.get((i, 0), ndims)
                print i, im, v
                m.set((i,0), [i, i*2, i*3])

def nkfeat2dm(dmname, mstr, fstr, inputiter, incr=1000):
    """Driver for extracting features directly into a data matrix.
    Params:
        dmname: the name of the datamatrix to update
        mstr and fstr: passed to FeatureComputer init
        inputiter: input iterator yielding (fname, fckw) pairs.
    In inputiter, fname is the field name corresponding to an item
    in the first field of the DM. fckw is a dict passed as kwargs to
    FeatureComputer.compute(). If fckw doesn't contain 'fname',
    then 'fname' is used as the fname for the compute().
    The DM is flushed every 'incr' inputs
    """
    from nkpylib.nkfeatures import FeatureComputer
    from pprint import pprint
    import gc
    gc.collect()
    log('Memory at top of nkfeat2dm is %0.2fMB' % (procmem()/1024.0/1024))
    d = DataMatrix(dmname, mode='updatedata')
    gc.collect()
    log('Memory after opening dm is %0.2fMB' % (procmem()/1024.0/1024))
    allmems = dict((f, getmem(getattr(d, f))) for f in dir(d))
    pprint(allmems)
    m = d.mmap
    fnamelookup = dict((fname, i) for i, fname in enumerate(d.fields[0]))
    fc = FeatureComputer(mstr=mstr, fstr=fstr)
    done = 0
    gc.collect()
    log('Memory after creating fc is %0.2fMB' % (procmem()/1024.0/1024))
    del d.j, d.fieldlookups, d.fields
    gc.collect()
    log('Memory after deleting some fields is %0.2fMB' % (procmem()/1024.0/1024))
    allmems = dict((f, getmem(getattr(d, f))) for f in dir(d))
    pprint(allmems)
    for fname, fckw in inputiter:
        try:
            #i = fnames.index(unicode(fname, 'utf-8', 'ignore'))
            i = fnamelookup[unicode(fname, 'utf-8', 'ignore')]
        except (KeyError,UnicodeDecodeError): continue
        loc = m.rowloc(i)
        #log('Memory in main loop with done=%s is %0.2fMB' % (done, procmem()/1024.0/1024))
        fckw.setdefault('fname', fname)
        fvec = fc.compute(**fckw)
        if not fvec:
            # set to error value
            fvec = array(str(d.typecode), [d.errval]*m.rowlen)
            print '  Error on input for row %d (loc %s), so setting to errval %s' % (i, loc, d.errval)
        m.set(loc, fvec)
        done += 1
        if done >= incr:
            m.flush()
            done = 0
        print i, fname, loc, len(fvec)
    m.flush()

def mstrfstrmain(*args):
    """An old driver that reads mstr, fstr, and writes to a DM."""
    incr = 1000
    if len(sys.argv) < 4:
        print 'Usage: python %s <dmname> <mstr> <fstr> [<flush incr>=%s]< inputs' % (sys.argv[0], incr)
        print '  The inputs should be one per line, with filename first.'
        print '  Then, optionally and tab-separated: fiducials, outparams'
        sys.exit()
    dmname, mstr, fstr = sys.argv[2:5]
    try:
        incr = int(sys.argv[5])
    except Exception:
        pass
    def inputiter():
        for l in sys.stdin:
            els = l.rstrip('\n').split('\t')
            fname = els.pop(0)
            kw = {}
            if els:
                kw['fiducials'] = els.pop(0)
            if els:
                kw['outparams'] = els.pop(0)
            yield (fname, kw)

    log('Memory at top of main is %0.2fMB' % (procmem()/1024.0/1024))
    # for 111,195 inputs, memory usage of entire inputiter() is 50MB
    nkfeat2dm(dmname, mstr, fstr, inputiter(), incr=incr)

def create(type, dmname, infname):
    """A driver to create a datamatrix of the given type.
    The type can be one of:
        sift1: Creates 2, 2d DMs with fields "points", {"locations", "descriptors"}, named
               <dmname>-locs and <dmname>-descs. The "points" field has names like 'point-%06d'.
               The input file should be a sift output file

        siftmany: Like sift1, but with first field "imagepoints", with names like '%(imgfname)::point-%06d'
                  The input file should be the output of wc -l.

        gist: Creates a 2d DM with fields "images", "feats" (which are {gist-%04d}).
              The input file should just be image names.
    """
    types = 'sift1 siftmany gist'.split()
    assert type in types
    if type.startswith('sift'):
        ptname = 'points'
        if type == 'siftmany':
            ptname = 'imagepoints'
        fieldtypes = [[ptname, 'locations'], [ptname,'descriptors']]
        if type == 'sift1':
            # the input should be a single sift file
            lines = [l.rstrip().split(' ') for l in codecs.open(infname, 'rb', 'utf-8') if l.rstrip()]
            ptfields = ['point-%06d' % i for i, l in enumerate(lines) if len(l) == 132]
            print 'Read %d lines from %s, down to %d after filtering, last: %s' % (len(lines), infname, len(ptfields), ptfields[-5:])
        elif type == 'siftmany':
            # the input should be the output of 'wc -l <list of descriptor outputs'
            lines = [l.rstrip().split(' ',1) for l in codecs.open(infname, 'rb', 'utf-8') if ' ' in l.rstrip()]
            lines = [(fname, int(num)) for num, fname in lines if fname != 'total']
            total = sum(num for fname, num in lines)
            print 'There are %d descriptors in %d lines total, coming out to %s bytes' % (total, len(lines), total*(128+4*4))
            ptfields = []
            for fname, num in lines:
                ptfields.extend(['%s::point-%06d' % (fname, i) for i in range(num)])
            print 'Read %d lines from %s, %d total, last: %s' % (len(lines), infname, len(ptfields), ptfields[-5:])
        field2 = ['x y scale orientation'.split(), ['sift-%03d' % i for i in range(128)]]
        #sys.exit()
        typecodes = 'fB'
        dmnames = [dmname+'-locs', dmname+'-descs']
        errvals = [-1.0, 255]
        for i in range(2):
            if i == 0: continue
            fields = [ptfields, field2[i]]
            fieldfnames = [os.path.basename(dmname)+'.imgpoints', None]
            dm = DataMatrix(dmnames[i], mode='updatestruct', typecode=typecodes[i], fields=fields, fieldfnames=fieldfnames, fieldtypes=fieldtypes[i], errval=errvals[i]).flush()
    elif type == 'gist':
        fieldtypes = 'images feats'.split()
        fieldfnames = [os.path.basename(dmname)+'.images', None]
        fields = [[l.rstrip() for l in codecs.open(infname, 'rb', 'utf-8')]]
        fields.append(['gist-%04d' % i for i in range(960)])
        print 'Creating gist DM at %s with field lens %s' % (dmname, map(len, fields))
        dm = DataMatrix(dmname, mode='updatestruct', typecode='f', fields=fields, fieldfnames=fieldfnames, fieldtypes=fieldtypes, errval=-1.0).flush()

def readsiftdescriptors(fname, sortbyloc=1):
    """Reads sift descriptors from the given fname and returns (locs, descs).
    locs is a list with [x, y, scale, orientation] entries.
    descs is a corresponding list with length-128 feature vectors.
    If sortbyloc is true (default), then sorts so that descrips are in scanline-order.
    """
    locs, descs = [], []
    lines = [l.rstrip().split() for l in open(fname) if l.rstrip()]
    lines = [(map(float, l[:4]), map(int, l[4:])) for l in lines]
    if lines:
        if sortbyloc:
            # sort by (y, x, scale, rotation)
            lines.sort(key=lambda p: (p[0][1], p[0][0], p[0][2], p[0][3]))
        locs, descs = zip(*lines)
    return locs, descs

def oldupdatesift(dmprefix, *args):
    """Updates sift descriptors in the given DM (by prefix).
    The args should be pairs of (imname, descfname).
    The imname indexes into the DM (using the 1st field).
    The descfname is the filename where descriptors are read from.
    """
    times = [time.time()]
    print 'Reading dm %s' % (dmprefix+'-locs')
    dmlocs = DataMatrix(dmprefix+'-locs', mode='updatedata', loadfields=0)
    mlocs = dmlocs.mmap
    times.append(time.time())
    dmdescs = DataMatrix(dmprefix+'-descs', mode='updatedata', loadfields=0)
    mdescs = dmdescs.mmap
    times.append(time.time())
    print 'Times to read DMs from prefix %s: %s' % (dmprefix, getTimeDiffs(times))
    inputs = list(nkgrouper(2, args))
    # precache rowlookups
    imnames = set(zip(*inputs)[0])
    imnames = set('|'.join(imnames))
    loclookup = dmlocs.getlookup(0, keys=imnames, keytype='regexp')
    desclookup = dmdescs.getlookup(0, keys=imnames, keytype='regexp')
    times.append(time.time())
    print 'Times to precache lookups: %s' % (getTimeDiffs(times))
    t1 = time.time()
    for iter, (imname, descfname) in enumerate(inputs):
        print 'On input %d of %d, with im %s, descfname %s' % (iter+1, len(inputs), imname, descfname)
        pname = lambda num: imname+'::point-%06d' % (num)
        lookup = lambda num: pname(num) in loclookup
        if not lookup(0):
            print '  This image not found in DM, so skipping'
            continue
        locs, descs = readsiftdescriptors(descfname)
        print '  Got %d locs, %d descs: %s' % (len(locs), len(descs), locs[:2])
        if not lookup(len(locs)-1):
            print '  This descfile has %d points, but the DM has less, so skipping all' % (len(locs))
            continue
        if lookup(len(locs)):
            print '  This descfile has %d points, but the DM has more, so skipping all' % (len(locs))
            continue
        # at this point, we're ready to write
        locrownum = loclookup[pname(0)]
        descrownum = desclookup[pname(0)]
        print '  Updating starting from rownums %d, %d' % (locrownum, descrownum)
        for i, (loc, desc) in enumerate(zip(locs, descs)):
            #print '    ', i, loc, desc[:8], mlocs.rowloc(locrownum+i), mdescs.rowloc(descrownum+i)
            mlocs.set(mlocs.rowloc(locrownum+i), loc)
            mdescs.set(mdescs.rowloc(descrownum+i), desc)
        times.append(time.time())
        sys.stdout.flush()
    print 'Finished updating %d images in %0.3fs, flushing now...' % (iter, time.time()-t1)
    dmlocs.flush()
    times.append(time.time())
    dmdescs.flush()
    times.append(time.time())
    print 'Final times: %s' % (getTimeDiffs(times))

def getsiftrowspecs(ptsfname, descriplookup):
    """Reads the given list of points and descriptor lookup and outputs rowspecs.
    The ptsfname is the external fields file of a SIFT DM.
    descriplookup is a file containing imname -> descripfname lookups.
    This prints to stdout a list of corresponding rowspecs and descripfnames.
    This is what updatesift() takes in as *args."""
    ret = {}
    order = []
    for i, l in enumerate(open(ptsfname)):
        if i % 1000000 == 0:
            log(' On line %d of %s: %s' % (i, ptsfname, l.rstrip()))
        im, ptnum = l.rstrip('\n').split('::point-')
        ptnum = int(ptnum)
        if im not in ret:
            ret[im] = [i, i]
            order.append(im)
        ret[im][1] = i+1
    lines = (l.rstrip().split() for l in open(descriplookup))
    im2desc = dict((im, d) for im, d in lines if im in ret)
    for im in order:
        try:
            print '%d-%d\t%s' % (ret[im][0], ret[im][1], im2desc[im])
        except KeyError:
            log('Image %s did not exist in %s' % (im, descriplookup))

def updatesift(dmprefix, *args):
    """Updates sift descriptors in the given DM (by prefix).
    The args should be pairs of (rowspec, descfname).
    The rowspec is something 59-3913, i.e., inputs to range(a,b).
    The descfname is the filename where descriptors are read from.
    """
    times = [time.time()]
    print 'Reading dm %s' % (dmprefix+'-locs')
    dmlocs = DataMatrix(dmprefix+'-locs', mode='updatedata', loadfields=0)
    dmdescs = DataMatrix(dmprefix+'-descs', mode='updatedata', loadfields=0)
    mlocs, mdescs = dmlocs.mmap, dmdescs.mmap
    inputs = list(nkgrouper(2, args))
    times.append(time.time())
    print 'Times to read DMs from prefix %s, and %d inputs: %s' % (dmprefix, len(inputs), getTimeDiffs(times))
    t1 = time.time()
    for iter, (rowspec, descfname) in enumerate(inputs):
        print 'On input %d of %d, with rowspec %s, descfname %s' % (iter+1, len(inputs), rowspec, descfname)
        locs, descs = readsiftdescriptors(descfname)
        rownums = range(*map(int, rowspec.split('-')))
        print '  Got %d locs, %d descs, and %d rownums: %s' % (len(locs), len(descs), len(rownums), locs[:2])
        if len(rownums) != len(locs) != len(descs):
            print '  Mismatched number of rows/locs/descs, so skipping'
            continue
        # at this point, we're ready to write
        for rownum, loc, desc in zip(rownums, locs, descs):
            #print '    ', i, loc, desc[:8], mlocs.rowloc(locrownum+i), mdescs.rowloc(descrownum+i)
            mlocs.set(mlocs.rowloc(rownum), loc)
            mdescs.set(mdescs.rowloc(rownum), desc)
        times.append(time.time())
        sys.stdout.flush()
    print 'Finished updating %d images in %0.3fs, flushing now...' % (iter, time.time()-t1)
    dmlocs.flush()
    times.append(time.time())
    dmdescs.flush()
    times.append(time.time())
    print 'Final times: %s' % (getTimeDiffs(times))

def checksift(dmprefix, num, descriplookup):
    """Checks the given DM for sift completion/correctness.
    num determines how many randomly sampled descriptors to use.
    descriplookup is a file which contains imname -> descripfname lookups
    The imname indexes into the DM (using the 1st field).
    The descfname is the filename where descriptors are read from.
    """
    import random
    times = [time.time()]
    print 'Reading dm %s' % (dmprefix+'-locs')
    dmlocs = DataMatrix(dmprefix+'-locs', mode='readdata', loadfields=0)
    mlocs = dmlocs.mmap
    times.append(time.time())
    dmdescs = DataMatrix(dmprefix+'-descs', mode='readdata', loadfields=0)
    mdescs = dmdescs.mmap
    times.append(time.time())
    print 'Times to read DMs from prefix %s: %s' % (dmprefix, getTimeDiffs(times))
    # precache rowlookups
    num = int(num)
    todo = set(random.sample(xrange(dmlocs.fields[0]), num))
    #todo = range(num) + [80000]
    maxrow = max(todo)
    print '%d Todo, with max %s' % (len(todo), maxrow)
    loclookup = dmlocs.getlookup(0, keys=todo, keytype='index')
    desclookup = dmdescs.getlookup(0, keys=todo, keytype='index')
    times.append(time.time())
    print 'Times to precache %d row lookups: %s' % (len(loclookup), getTimeDiffs(times))
    # now precache descrip lookups
    imnames = set(im.split('::')[0] for im in desclookup)
    lines = (l.rstrip().split() for l in open(descriplookup))
    im2desc = dict((im, d) for im, d in lines if im in imnames)
    times.append(time.time())
    print 'Times to precache %d descrip lookups for %d imnames: %s' % (len(im2desc), len(imnames), getTimeDiffs(times))
    t1 = time.time()
    alldescs = {}
    aresimilar = lambda a,b: sum(abs(x-y) for x, y in zip(a, b)) < 0.001
    done = 1
    for pt in sorted(loclookup):
        if done % 1000 == 0:
            log('%d done' % (done,))
        im, ptnum = pt.split('::')
        ptnum = int(ptnum.split('-')[1])
        descfname = im2desc[im]
        #print pt, im, descfname, ptnum, loclookup[pt], desclookup[pt]
        if descfname not in alldescs:
            alldescs[descfname] = readsiftdescriptors(descfname)
        locs, descs = alldescs[descfname]
        loc = locs[ptnum]
        desc = descs[ptnum]
        sloc = mlocs.getrow(loclookup[pt])
        sdesc = mdescs.getrow(desclookup[pt])
        if not aresimilar(sloc, loc) or not aresimilar(sdesc, desc):
            print '%s had a problem:\n\t%s\n\t%s\n\t%s\n\t%s' % (pt, sloc, loc, sdesc, desc)
        done += 1

def updategist(dmname, infname, loadfields=0, incr=5000):
    """Updates gist descriptors in the given DM.
    The input file should contain rownumber and then the descriptor
    """
    loadfields = int(loadfields)
    times = [time.time()]
    dm = DataMatrix(dmname, mode='updatedata', loadfields=loadfields)
    m = dm.mmap
    t1 = time.time()
    times.append(t1)
    print 'Loaded DM %s with loadfields=%d in %0.3fs and starting to read outputs from %s' % (dmname, loadfields, t1-times[0], infname)
    ndone = 0
    for iter, l in enumerate(open(infname)):
        try:
            first, fvec = l.rstrip().split('\t', 1)
            fvec = map(float, fvec.split())
            assert len(fvec) == m.size[1]
            rownum = dm.fieldlookups[0][first] if loadfields else int(first)
        except (KeyError, ValueError, AssertionError):
            print '  On line %d, could not read rownum or fvec: %s' % (iter, l.rstrip())
            continue
        if iter % 1 == 0:
            print '  On input %d, with rownum %s' % (iter+1, rownum)
        m.set(m.rowloc(rownum), fvec)
        ndone += 1
        if ndone >= incr:
            log('Flushing DM')
            dm.flush()
            ndone = 0
        sys.stdout.flush()
    print 'Finished updating %d images in %0.3fs, flushing now...' % (iter, time.time()-t1)
    dm.flush()
    times.append(time.time())
    print 'Final times: %s' % (getTimeDiffs(times))

def fnames2rownums(fnamelst):
    """Reads fnamelst and makes a mapping of fname -> rownum.
    Then for each line in the input, replaces the first field with the row num.
    Prints out the converted file to stdout"""
    d = dict((f.rstrip(), i) for i, f in enumerate(open(fnamelst)))
    log('Read %d entries from %s' % (len(d), fnamelst))
    for i, l in enumerate(sys.stdin):
        if i  % 100000:
            log('On input row %d'  %(i))
        fname, rest = l.rstrip('\n').split('\t',1)
        if fname not in d:
            log('Error on line %d: %s not in dict' % (i, fname))
            continue
        print '%s\t%s' % (d[fname], rest)


def squaredims(dims):
    """Takes a pair of dimensions and normalizes them so that they're almost square."""
    import math
    w, h = dims
    total = w*h
    outw = int(math.sqrt(total))
    outw -= outw % min(w,h)
    outh = total//outw
    if total % outw:
        outh += 1
    return (outw, outh)

def siftimages(dmprefix, outfmt, nrows=250000):
    """Creates images representing a sift DM.
    """
    from PIL import Image
    dm = DataMatrix(dmprefix+'-descs', mode='readdata', loadfields=0)
    dims = [len(f) if isinstance(f, list) else f for f in dm.fields]
    rowsets = list(nkgrouper(nrows, xrange(dims[0])))
    print len(rowsets)
    cache = {}
    for i, rows in enumerate(rowsets):
        w, h = squaredims((len(rows),dims[1]))
        im = Image.new('RGB', (w, h))
        fname = outfmt % i
        print 'On image %d of %d, with dims %d x %d: %s' % (i+w, len(rowsets), w, h, fname)
        im = dm.mmap.toimage(im=im, rows=rows, cache=cache, valrange=[0, 128])
        im.save(fname)

def externalize(dmname, field, externalfname):
    """Externalizes a given field from the given datamatrix"""
    dm = DataMatrix(dmname, 'updatestruct')
    i = dm.fieldtypes.index(field)
    dm.fieldfnames[i] = externalfname
    dm.flush()

def testnn(dmname, datarowsfname):
    """Tests NN speeds.
    May 30, on arnold (1 core), 128-byte sift descriptors
    n data | n test | time
    ------------------------
     50000 |  1000  |  8.5s
    100000 |  1000  | 17  s
    100000 |   500  |  9.8s
    200000 |   500  | 19.8s
    300000 |   500  | 30.8s
    300000 |   100  | 12.6s
    Or in summary: linear in datasize, sublinear in testsize
    """
    from nkutils import bulkNNl2
    times = [time.time()]
    datarows = [k.rstrip() for k in open(datarowsfname) if k.rstrip()]
    keys = [k.rstrip() for k in sys.stdin if k.rstrip()]
    dm = DataMatrix(dmname, 'readdata', loadfields=0)
    times.append(time.time())
    print 'Times: %s' % (getTimeDiffs(times))
    datamat = dm2numpy(dmname, datarows, errval=dm.errval)
    times.append(time.time())
    print 'Times: %s' % (getTimeDiffs(times))
    print 'Datamat:',datamat, datamat.shape, datamat.dtype
    testmat = dm2numpy(dmname, keys, errval=dm.errval)
    times.append(time.time())
    print 'Times: %s' % (getTimeDiffs(times))
    print 'Testmat:',testmat, testmat.shape, testmat.dtype
    dists = bulkNNl2(testmat, datamat)
    times.append(time.time())
    print 'Times: %s' % (getTimeDiffs(times))
    print dists, dists.shape, dists.dtype

def dm2redis(dmname, rediscfgfname):
    """Puts a datamatrix into redis.
    Reads fnames from stdin."""
    from nkpylib.nkredisutils import redis
    name = os.path.basename(dmname)
    db = redis.Redis(**json.load(open(rediscfgfname)))
    db.sadd('ftypes', name)
    print 'Opened redis db connection using params in %s and added ftype %s' % (rediscfgfname, name)
    # read fnames and map them to ids
    incr = 100000
    fnames = [l.rstrip('\n') for l in sys.stdin if l.rstrip('\n')]
    print 'Read %d fnames: %s' % (len(fnames), fnames[:5])
    groups = list(nkgrouper(incr, fnames))
    idmap = {}
    for i, g in enumerate(groups):
        print '  On group %d of %d' % (i+1, len(groups))
        ids = db.hmget('fnameids', g)
        toset = {}
        for id, fname in zip(ids, g):
            if not id:
                idmap[fname] = toset[fname] = int(db.incr('id_counter'))
            else:
                idmap[fname] = int(id)
        print '    Setting %d ids, %d total in idmap' % (len(toset), len(idmap))
        if toset:
            db.hmset('fnameids', toset)
    print 'Set and got ids for %d fnames' % (len(idmap))
    # load DM and go through it
    dm = DataMatrix(dmname, 'readdata', loadfields=1)
    print 'Loaded dm %s' % (dmname)
    p = db.pipeline()
    for i, fname in enumerate(dm.fields[0]):
        if fname not in idmap: continue
        key = 'f:%s:%s' % (name, idmap[fname])
        fvec = dm.mmap.getrow(i)
        p.set(key, fvec.tostring())
        if i % 1000 == 0:
            print '  Executing pipeline for i=%d' % (i)
            p.execute()
    p.execute()
    print 'Finished updating redis'

def view(dmname, filt=None):
    """Views a datamatrix"""
    dm = DataMatrix(dmname, 'readdata', loaddata=0)
    m = dm.mmap
    for i, fname in enumerate(dm.fields[0]):
        if filt and filt not in fname: continue
        r = m.getrow(i)
        print u'%s\t%s' % (fname, getListAsStr(r, '\t'))


if __name__ == '__main__':
    import inspect
    callspec = lambda f: inspect.getsource(f).split('\n')[0].strip().replace('def ','').rstrip(':')
    drivers = [create, getsiftrowspecs, updatesift, checksift, updategist, fnames2rownums, siftimages, externalize, mstrfstrmain, testnn, test, dm2redis, view]
    dnames = [f.__name__ for f in drivers]
    if len(sys.argv) < 2 or sys.argv[1] not in dnames:
        print 'Usages:\n%s' % ('\n'.join('\t'+callspec(f) for f in drivers))
        sys.exit()
    func = eval(sys.argv[1])
    func(*sys.argv[2:])
    #DataMatrix('testdm', mode='readdata').mmap.toimage(valrange='data').save('testdm.png'); sys.exit()
