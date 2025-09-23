#!/usr/bin/env python
"""Utilities for dealing with graphs in python, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2014-, Neeraj Kumar (neerajkumar.org)
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

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from typing import Callable, Iterator, Any, Optional, TypeAlias, TypeVar, Hashable
from queue import Empty, Queue

import numpy as np

# some helpful type aliases
GraphT: TypeAlias = dict[tuple[Any, Any], float]

# Type variables for bipartite matching functions
A = TypeVar('A')
B = TypeVar('B')
ScoreFunc: TypeAlias = Callable[[A, B], float]

def floydwarshall(g: GraphT, v: int) -> GraphT:
    """An implementation of the Floyd-Warshall algorithm for finding all-pairs shortest paths. The
    input graph g should be something indexable by (i,j) to get distance between nodes i and j. This
    should be pre-filled with the costs between known nodes, 0 for the diagonal, and infinity
    elsewhere. v is the number of vertices.

    Modifies the given g directly and returns it.
    """
    for k in range(v):
        for i in range(v):
            for j in range(v):
                g[i, j] = min(g[i, j], g[i, k] + g[k, j])
    return g


def dijkstras_shortest_path(graph: GraphT, src: Any) -> tuple[dict[Any, Any], dict[Any, float]]:
    """An implementation of Dijkstra's shortest path algorithm.

    This starts at `src` and either computes all shortest paths from there.

    The `graph` should be a dict-like object with distances from a to b as `graph[a,b]`

    Returns `(previous_nodes, shortest_path)` where the former is a dict from node to previous node,
    and the latter is a dict from node to shortest path length from src
    """
    # initialization
    todo = set()
    neighbors_by_node = defaultdict(set)
    for a, b in graph.keys():
        neighbors_by_node[a].add(b)
        neighbors_by_node[b].add(a)
        todo.add(a)
        todo.add(b)
    shortest_path: dict[Any, float] = {}
    for node in todo:
        shortest_path[node] = sys.maxsize
    assert src in todo
    shortest_path[src] = 0
    previous_nodes = {}
    while todo:
        # pick node with min distance so far
        cur = None
        for node in todo:
            if cur is None:
                cur = node
            elif shortest_path[node] < shortest_path[cur]:
                cur = node
        # get neighbors from cur and evaluate distances
        for neighbor in neighbors_by_node[cur]:
            dist = shortest_path[cur] + graph[cur, neighbor]
            if dist < shortest_path[neighbor]:
                shortest_path[neighbor] = dist
                previous_nodes[neighbor] = cur
        todo.remove(cur)
    return previous_nodes, shortest_path


def bipartite_matching(a: list[A], b: list[B], score_func: ScoreFunc[A, B], symmetric: bool = False, threshold: float = 0.0) -> Iterator[tuple[float, A, B]]:
    """Does bipartite matching between lists `a` and `b` using `score_func`.

    This computes scores between all elements in a and b using `score_func(x, y)`, and then goes
    through the score matrix in descending order of score, yielding `(score, a_i, b_j)` tuples.
    These are filtered such that no duplicate i or j are returned. We stop if the `score <
    threshold`.

    If `symmetric` is True, then assumes `a` and `b` refer to the same set of objects, and order
    doesn't matter.
    """
    # first compute scores
    na, nb = len(a), len(b)
    scores = np.zeros((na, nb))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            scores[i, j] = score_func(x, y)
    inds: Any = np.unravel_index(np.argsort(scores, axis=None), scores.shape)
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
        if scores[i, j] < threshold:
            break
        yield (scores[i, j], a[i], b[j])
        i_left.remove(i)
        j_left.remove(j)


def bipartite_matching_wrapper(a: list[A], b: list[B], score_func: ScoreFunc[A, B], symmetric: bool = False, threshold: float = 0.0) -> tuple[list[tuple[float, A, B]], set[A], set[B]]:
    """A wrapper to `bipartite_matching()` that returns `(matches, unmatched_in_a, unmatched_in_b)`

    The list of `matches` contains tuples of `(score, a_element, b_element)`. The two unmatched
    lists are elements from each of the respective input lists.
    """
    found_a, found_b = set(), set()
    matches = []
    for score, i, j in bipartite_matching(a, b, score_func, symmetric=symmetric, threshold=threshold):
        matches.append((score, i, j))
        found_a.add(i)
        found_b.add(j)
    unmatched_in_a = set(a) - found_a
    unmatched_in_b = set(b) - found_b
    return matches, unmatched_in_a, unmatched_in_b


class PathNotFoundError(Exception):
    pass


NodeT = TypeVar('NodeT', bound='Hashable')
class Router:
    """A routing class to find paths matching certain criteria in graphs"""

    def __init__(self, nn_func: Callable[[Any], Iterator[tuple[NodeT, float]]],
                 nn_pred: Optional[Callable[[NodeT, float], bool]] = None):
        """Initializes the router with a function to get neighbors.
        `nn_func(node)` should take a node and return an iterator of (neighbor, distance) tuples.
        `nn_pred(node, distance) -> valid` is an optional function to filter neighbors based on distance.
        """
        self._nn_func = nn_func
        self._nn_pred = nn_pred

    def find_path(self, a: NodeT, b: NodeT) -> tuple[list[NodeT], float]:
        """Finds a path from `a` to `b` matching our criteria.

        Returns `(path, cost)`, where the path is a list of node ids, and cost is total cost.

        If no path is found, then raises `PathNotFoundError`.
        """
        logging.info(f"Trying to find a path from {a} to {b}")
        q: Queue[Any] = Queue()
        dists = {}
        done = set()

        def add(el):
            """util function to add `el` and expand its neighbors, returning if we found `b`"""
            done.add(el)
            for key, dist in self._nn_func(el):
                if self._nn_pred and not self._nn_pred(key, dist):
                    continue
                dists[el, key] = dists[key, el] = dist
                if key == b:
                    return True
                if key not in done:
                    q.put(key)
            return False

        # add the initial neighbors from a
        found = add(a)
        i = 0
        while not found:
            try:
                found = add(q.get(block=False))
                logging.debug(
                    f"On iter {i}, {len(done)} done, {len(dists)} dists, {q.qsize()} in q, found {found}"
                )
            except Empty:
                break
            i += 1
        # at this point, we either found a path, or are otherwise done
        if not found:
            raise PathNotFoundError()
        # get the actual path and cost
        prev, costs = dijkstras_shortest_path(dists, a)
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        path = path[::-1]
        return path, costs[b]

def load_graphml_nodes(graph_path: str):
    """Loads only nodes from a graphml file.

    Example format:

    ?xml version='1.0' encoding='utf-8'?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"><key id="d3" for="edge" attr.name="weight" attr.type="double"/>
    <key id="d2" for="edge" attr.name="type" attr.type="string"/>
    <key id="d1" for="node" attr.name="name" attr.type="string"/>
    <key id="d0" for="node" attr.name="type" attr.type="string"/>
    <graph edgedefault="undirected"><node id="tt0104674">
      <data key="d0">movie</data>
      <data key="d1">Lakota Moon</data>
    </node>
    <node id="nm0310390">
      <data key="d0">person</data>
      <data key="d1">David Gautreaux</data>
    </node>
    <node id="tt0121194">
      <data key="d0">movie</data>
      <data key="d1">Decaf</data>
    </node>
    """
    import networkx as nx
    from lxml import etree
    from tqdm import tqdm
    with open(graph_path, 'rb') as f:
        tree = etree.parse(f)
    G = nx.Graph()
    root = tree.getroot()
    for node in tqdm(root.findall('node')):
        key = node.attrib['id']
        G.add_node(key)
        for data in node.findall('data'):
            key = data.attrib['key']
            value = data.text
            if key == 'd0':
                G.nodes[key]['type'] = value
            elif key == 'd1':
                G.nodes[key]['name'] = value
    return G
