#!/usr/bin/env python
"""A simple but parallelized nearest-neighbors program.
Originally written by Neeraj Kumar <me@neerajkumar.org>.

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

import os, sys, random, math, time
from nkpylib.nkutils import log, getListAsStr, simplenn
from optparse import OptionParser
import numpy

VERSION = '0.1'

FMTS = 'foo bar baz blah'.split()
DEFAULT_DATAFMT = 'blah'
DEFAULT_INPUTFMT = 'blah'
DEFAULT_OUTPUTFMT = 'blah'

METRICS = 'l1 l2 chisq int bhatt'.split()
DEFAULT_METRIC = 'l2'

SEARCH_TYPES = 'k radius radius-k sorted unsorted'.split()

def readData(f=sys.stdin, dtype=float):
    """Reads the data matrix from the given file.
    The first line should contain the # rows (height) and # cols (width) of the matrix"""
    nrows, ncols = map(int, f.readline().strip().split())
    data = []
    for i in range(nrows):
        row = numpy.array(map(dtype, f.readline().strip().split()))
        assert len(row) == ncols
        data.append(row)
    data = numpy.vstack(data)
    return data

def readfmt(s, fmt=DEFAULT_INPUTFMT):
    """Reads a given string into an array of floats using the given format"""
    ret = map(float, s.strip().split())
    return ret

def writefmt(fvec, fmt=DEFAULT_OUTPUTFMT):
    """Writes a given array to a string using the given format"""
    return getListAsStr(fvec, sep=' ')

def readdata(fname, fmt=DEFAULT_DATAFMT):
    """Reads the data from the given filename with the given format"""
    data = [readfmt(row, fmt=fmt) for row in open(fname)]
    return data

def computedists(fvecs, datafname, minparallel=15, blocksizefac=3, **kw):
    """Computes distances, in a parallel way if possible.
    Yields distance vectors one at a time, corresponding to inputs
    """
    from subprocess import Popen, PIPE
    from Queue import Queue
    # create the process
    mainargs = ['python', os.path.abspath(__file__), datafname]
    try:
        # if too few items, just do them serially
        if len(fvecs) <= minparallel: raise OSError
        # first try opening it using parallel
        # we want the write size to be roughly blocksizefac times the length of the fvec
        fveclen = len(writefmt(fvecs[0]))
        blocksize = str(blocksizefac * fveclen)
        args = ['parallel', '--pipe', '-k', '--block', blocksize] + mainargs
        #print 'Running parallel with: "%s"' % (' '.join(args))
        p = Popen(args, stdin=PIPE, stdout=PIPE)
    except OSError:
        # otherwise just run the process directly
        p = Popen(mainargs, stdin=PIPE, stdout=PIPE)

    dists = Queue()
    def readouts(dists=dists):
        for i, fv in enumerate(fvecs):
            d = readfmt(p.stdout.readline())
            dists.put(d)

    from nkpylib.nkthreadutils import spawnWorkers
    dlt = spawnWorkers(1, readouts, interval=0)

    # feed it data
    for i, fvec in enumerate(fvecs):
        print >>p.stdin, writefmt(fvec)
        p.stdin.flush()

    # close the input stream on the process, since we're done with it
    p.stdin.close()

    # yield results
    for i, fv in enumerate(fvecs):
        yield dists.get()


DESCRIPTION = """This program allows you to compute nearest neighbors (NN) in a parallel and flexible way on a single machine (for now). Since this is a common operation in lots of applications, this is written in a very generic way, at a fairly low-level, so you will most likely need to write your own wrapper on top.

The following data formats are supported for the data file, the inputs, and the outputs: (a) blah: doij, (b) foo: doifj, (c) bar: idfojio
"""


def main():
    """Main method"""
    # setup command parser
    usage = 'Usage: python %s [opts] <data filename>' % (sys.argv[0])
    parser = OptionParser(usage=usage, version=VERSION, description=DESCRIPTION)
    parser.add_option('-m', '--metric', dest='metric', choices=METRICS, default=DEFAULT_METRIC, help='the distance metric to use [default %s]' % (DEFAULT_METRIC))
    parser.add_option('-d', '--datafmt', dest='datafmt', choices=FMTS, default=DEFAULT_DATAFMT, help='the format of the data file [default %s]' % (DEFAULT_DATAFMT))
    parser.add_option('-i', '--inputfmt', dest='inputfmt', choices=FMTS, default=DEFAULT_INPUTFMT, help='the format of the input data from stdin [default %s]' % (DEFAULT_INPUTFMT))
    parser.add_option('-o', '--outputfmt', dest='outputfmt', choices=FMTS, default=DEFAULT_OUTPUTFMT, help='the format of the output to stdout [default %s]' % (DEFAULT_OUTPUTFMT))
    opts, args = parser.parse_args()
    if len(args) < 1:
        parser.print_help()
        parser.error('Need to specify the data filename.')
    #log('%s, %s' % (opts, args))
    datafname = args[0]
    data = readdata(datafname, opts.datafmt)
    ndata, ndims = len(data), len(data[0])
    #print data
    i = 0
    for line in sys.stdin:
        fvec = readfmt(line, fmt=opts.inputfmt)
        assert len(fvec) == ndims
        dists = simplenn(data, fvec, metric=opts.metric, normalize=None)
        print writefmt(dists, fmt=opts.outputfmt)
        try:
            sys.stdout.flush()
        except IOError: break


if __name__ == '__main__':
    main()
