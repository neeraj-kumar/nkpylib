#!/usr/bin/env python
"""Mechanical Turk-related utilities, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2013, Neeraj Kumar (neerajkumar.org)
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
from pprint import pprint, pformat

OUTHTML_FRAME = '''<html>
    <head><title>Rendered mturk template</title></head>
    <body>
        %s
    </body>
</html>'''

def readcsv(fname):
    """Reads the CSV file given and returns a list of dicts"""
    import csv
    reader = csv.DictReader(open(fname))
    ret = [row for row in reader]
    return ret

def renderhtml(tmpl, data, rowspec=None):
    """Renders html from the given template and data (list of dicts).
    The rowspec should be an expression involving i and r, which are
    the row index, and a random float, resp. This will be eval'ed and
    only if true will the row be output.
    An empty or None rowspec outputs all rows.
    """
    from random import random
    import re
    # convert template to a python-style template
    var = re.compile(r'\${(.*?)}')
    matches = var.split(tmpl)
    s = ''
    for i, m in enumerate(matches):
        if i%2 == 0:
            s += m
        else:
            s += '%%(%s)s' % (m)
    # go through data
    rows = []
    for i, row in enumerate(data):
        r = random()
        if rowspec and not eval(rowspec, locals()): continue
        rows.append(s % row)
    out = OUTHTML_FRAME % ('\n'.join(rows))
    return out

def demultiplex(row, nperhit):
    """Demultiplexes this dict and returns a list of dicts."""
    import re
    end = re.compile(r'_\d+$')
    # de-multiplex data
    ret = []
    for i in range(nperhit):
        # copy all data
        d = dict(**row)
        for k, v in sorted(d.items()):
            # find input and output fields and delete them initially
            if not k.startswith('Input.') and not k.startswith('Answer.'): continue
            del d[k]
            # rename to simplified keys
            k = k.replace('Input.','').replace('Answer.','')
            if end.search(k):
                # if it's the current one, we want to add it back in
                if k.endswith('_%d' % i):
                    k = k.rsplit('_', 1)[0]
                else: continue # remove multiplexed keys
            # add field back in
            d[k] = v
        ret.append(d)
    return ret

def renderout(tmplfname, data, groupby, nperhit):
    """Renders mturk output"""
    import web, web.template
    from nkutils import partitionByFunc
    from nkwebutils import NKStor, mystorify
    # de-multiplex and simplify data
    data = sum([demultiplex(row, nperhit) for row in data], [])
    # group by common key
    grouped, _ = partitionByFunc(data, lambda d: d[groupby])
    results = []
    Cls = NKStor
    # build up list of results
    for gkey, g in sorted(grouped.items()):
        # build up a list of common keys for this result group
        r = Cls(g[0])
        for el in g:
            for k, v in r.items():
                if el[k] != v:
                    del r[k]
        # now create each individual sub-output
        r['outputs'] = [Cls(el) for el in g]
        results.append(r)
    #pprint(results)
    # render results
    renfunc = web.template.frender(tmplfname)
    s = renfunc(results)
    return s



if __name__ == '__main__':
    from pprint import pprint
    TASKS = 'renderhit renderout'.split(' ')
    if len(sys.argv) < 2:
        print 'Usage: python %s <%s> [<args> ...]' % (sys.argv[0], '|'.join(TASKS))
        sys.exit()
    task = sys.argv[1]
    assert task in TASKS
    if task == 'renderhit':
        if len(sys.argv) < 4:
            print 'Usage: python %s renderhit <template> <data csv> [<rowspec>]' % (sys.argv[0])
            print "  rowspec is an expression involving 'i' (index) and/or 'r' (random float) which is eval'ed"
            sys.exit()
        tmpl = open(sys.argv[2]).read()
        data = readcsv(sys.argv[3])
        try:
            rowspec = sys.argv[4]
        except Exception:
            rowspec = None
        html = renderhtml(tmpl, data, rowspec)
        print html
    elif task == 'renderout':
        if len(sys.argv) < 5:
            print 'Usage: python %s renderout <template> <data csv> <groupby> <nperhit>' % (sys.argv[0])
            sys.exit()
        tmplfname = sys.argv[2]
        data = readcsv(sys.argv[3])
        groupby = sys.argv[4]
        nperhit = int(sys.argv[5])
        html = renderout(tmplfname, data, groupby, nperhit)
        print html


