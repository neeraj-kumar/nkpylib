#!/usr/bin/env python
"""A script for downloading lots of urls fast, written by Neeraj Kumar.
This is just a wrapper around nkthreadutils.dlmany()

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
from nkpylib.nkthreadutils import dlmany


#def dlmany(urls, fnames, nprocs=10, callback=None, validfunc=os.path.exists, checkexists=0, timeout=None):
if __name__ == '__main__':
    nthreads = 15
    if len(sys.argv) < 2:
        print 'Usage: python %s <output dir> [<nthreads>=%d] < <list of urls>' % (sys.argv[0], nthreads)
        sys.exit()
    outdir = sys.argv[1]
    try:
        nthreads = int(sys.argv[2])
    except Exception: pass
    urls = []
    fnames = []
    for l in sys.stdin:
        url = l.rstrip('\n')
        if '\t' in url:
            url, path = url.split('\t')
            path = os.path.join(outdir, path) #TODO here!
            print url, path
            urls.append(url)
            fnames.append(path)

    print 'Getting ready to download %d urls to outdir %s using %d threads' % (len(urls), outdir, nthreads)
    #sys.exit()

    def callback(i, u, f):
        print '  Downloaded %s %s %s' % (i, u, f)
        sys.stdout.flush()

    t1 = time.time()
    dlmany(urls, fnames, nprocs=nthreads, callback=callback, checkexists=1)
    print time.time() - t1

