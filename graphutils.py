#!/usr/bin/env python
"""Utilities for dealing with graphs in python, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2014, Neeraj Kumar (neerajkumar.org)
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

import os, sys

import numpy as np

def floydwarshall(g, v):
    """An implementation of the Floyd-Warshall algorithm for finding all-pairs shortest paths.
    The input graph g should be something indexable by (i,j) to get distance between nodes i and j.
    This should be pre-filled with the costs between known nodes, 0 for the diagonal, and infinity elsewhere.
    v is the number of vertices.
    Modifies the given g directly and returns it.
    """
    for k in xrange(v):
        for i in xrange(v):
            for j in xrange(v):
                g[i,j] = min(g[i,j], g[i,k] + g[k,j])
    return g

def bipartite_matching(a, b, score_func, symmetric=False):
    """Does bipartite matching between lists `a` and `b` using `score_func`.

    This computes scores between all elements in a and b using `score_func(x, y)`, and then goes
    through the score matrix in descending order of score, yielding `(score, a_i, b_j)` tuples.
    These are filtered such that no duplicate i or j are returned.

    If `symmetric` is True, then assumes `a` and `b` refer to the same set of objects, and order
    doesn't matter.
    """
    # first compute scores
    na, nb = len(a), len(b)
    scores = np.zeros((na, nb))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            scores[i,j] = score_func(x, y)
    inds = np.unravel_index(np.argsort(scores, axis=None), scores.shape)
    inds = np.array([x[::-1] for x in inds])
    i_left = set(range(na))
    j_left = set(range(nb))
    if symmetric:
        assert na == nb
        j_left = i_left
    for i, j in inds.T:
        if symmetric and i == j:
            continue
        if i not in i_left:
            continue
        if j not in j_left:
            continue
        yield (scores[i, j], a[i], b[j])
        i_left.remove(i)
        j_left.remove(j)
