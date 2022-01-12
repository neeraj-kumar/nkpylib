"""Parses links from html files, and other graph-related tooling for website downloads

Licensed under the 3-clause BSD License:

Copyright (c) 2022, Neeraj Kumar (http://neerajkumar.org)
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

import sys

from os.path import join, exists, isfile
from urllib.parse import urljoin
from typing import Set, Optional

from pyquery import PyQuery as pq

def find_page(loc: str) -> Optional[str]:
    """Finds the page path at the given `loc`"""
    to_check = [loc]
    if loc.endswith('/'):
        to_check.append(loc+'index.html')
    else:
        to_check.append(loc+'/index.html')
    to_check.append(loc.replace('/?', '/index.html?'))
    for loc in to_check:
        #print(f'  Checking {loc}')
        if exists(loc) and isfile(loc):
            return loc
    return None

def parse_page(loc: str) -> Set[str]:
    """Parses html page at given `loc`"""
    while '//' in loc:
        loc = loc.replace('//', '/')
    path = find_page(loc)
    #print(f'For loc {loc} got path {path}')
    if not path:
        return set()
    d = pq(filename=path)
    def joinlink(next):
        if next.startswith('/'):
            base = loc.split('/')[0]
            return base+next
        else:
            return next
    links = {joinlink(link.get('href')) for link in d('a') if link.get('href')}
    return links

def get_all_edges(base: str):
    """Gets all edges starting at given `base`"""
    done = set()
    todo = [base]
    ret = []
    for loc in todo:
        if loc in done:
            continue
        #if len(done) > 5: break
        print(f'Processing page {loc}, w/{len(ret)} in ret, {len(done)} done, {len(todo)} todo')
        for link in parse_page(loc):
            if 'http' in link or link.startswith('#'):
                continue
            ret.append((loc, link))
            todo.append(link)
        done.add(loc)
    return ret

if __name__ == '__main__':
    dir, out_path = sys.argv[1:3]
    edges = get_all_edges(dir)
    with open(out_path, 'w') as f:
        for src, dst in edges:
            f.write(f'{src}\t{dst}\n')
