#!/usr/bin/env python
"""Converts a tab-separated input into a format suitable for cdb.
CDB is Daniel J Bernstein's "constant data base":
    http://cr.yp.to/cdb.html
In practice, we typically use the 'mcdb' implementation:
    https://github.com/gstrauss/mcdb
because the original only supports upto 4GB files, and this one has no limits.

This version also uses a mmap()-based implementation, which has benefits if
running multiple processes for searching an index.

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
from optparse import OptionParser

def fmtcdb(key, val):
    """Formats a key and value into cdb format."""
    key = str(key)
    val = str(val)
    s = '+%d,%d:%s->%s' % (len(key), len(val), key, val)
    return s

if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <input file or - for stdin> <output file or - for stdout>')
    parser.add_option('-c', '--collapse', dest='collapse', default='', help='If specified, collapses duplicate keys using the provided separator. Else, does not collapse multiple keys')
    parser.add_option('-g', '--grouped', dest='grouped', action='store_true', default=0, help='If specified, collapse assumes all keys are grouped together, to prevent having to store a giant dict. Default false.')
    parser.add_option('-2', '--twopass', dest='twopass', action='store_true', default=0, help='If specified, runs a two-pass algorithm to save memory. Default false.')
    opts, args = parser.parse_args()
    if len(args) < 2:
        parser.print_help()
        sys.exit()
    infile = sys.stdin if args[0] == '-' else open(args[0])
    if args[1] == '-':
        outfile = sys.stdout
    else:
        try:
            os.makedirs(os.path.dirname(args[1]))
        except OSError:
            pass
        outfile = open(args[1], 'wb')
    dlens = {}
    if opts.twopass:
        # first pass to compute value lengths
        assert infile is not sys.stdin
        assert opts.grouped
        for i, line in enumerate(infile):
            if i % 1000 == 0:
                print >>sys.stderr, '[Pass 1] On line %d of input...       \r' % (i+1),
            k, v = line.rstrip('\n').split('\t', 1)
            if k not in dlens:
                dlens[k] = len(v)
            else:
                dlens[k] += len(v) + len(opts.collapse)
        infile.seek(0)
        print >>sys.stderr, 'In pass 1, read %d lines and %d keys with total dlen %d' % (i+1, len(dlens), sum(dlens.itervalues()))

    keys = {}
    last = None
    def printkeys():
        """print out keys->values if we've collapsed"""
        for k, vlist in keys.iteritems():
            v = opts.collapse.join(vlist)
            s = fmtcdb(k, v)
            print >>outfile, s

    for i, line in enumerate(infile):
        if i % 10000 == 0:
            print >>sys.stderr, 'On line %d of input...       \r' % (i+1),
            outfile.flush()
        k, v = line.rstrip('\n').split('\t', 1)
        #print k, v
        # if we're not collapsing, we just stream through the file
        if not opts.collapse:
            s = fmtcdb(k, v)
            print >>outfile, s
            continue
        # otherwise, we have to accumulate values
        if k == last:
            # existing key
            if opts.twopass:
                outfile.write('%s%s' % (opts.collapse, v))
        else:
            # new key
            if opts.twopass:
                # print end of last line
                if last is not None:
                    outfile.write('\n')
                # print beginning of this line
                outfile.write('+%d,%d:%s->%s' % (len(k), dlens[k], k, v))
            else:
                # flush out the previous entry, as it's done
                printkeys()
                try:
                    del keys[last]
                except KeyError: pass
        if not opts.twopass:
            keys.setdefault(k, []).append(str(v))
        last = k
    print >>sys.stderr, 'Finished reading input with %d lines.     ' % (i+1)
    if opts.twopass:
        # we need to end the last line
        outfile.write('\n')
    printkeys()
    print >>outfile # final newline is needed for valid cdb input
