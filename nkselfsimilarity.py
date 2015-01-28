"""Some code to compute self similarity between patches in an image.

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

import Image, sys, os, math, pickle, itertools

DEFAULT_SIZE = (5, 5)
NUM_BINS = 2

# caches the method using a closure
def cachemethod(fn):
    """ Returns a function that caches calls to the function passed in.
    It's important to only cache functional type methods, that is methods that always return the same value given the same parameters """
    cache = {}
    def newfn(*args):
        try: 
            return cache[args]
        except KeyError:
            val = cache[args] = fn(*args)
            return val
    return newfn

def grouper(n, iterable, padvalue=None):
    """Taken from Python's itertools recipes.
    >>> list(grouper(3, 'abcdefg', 'x'))
    [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'x', 'x')]"""
    from itertools import izip, chain, repeat
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee, izip
    a, b = tee(iterable)
    try:
        b.next()
    except StopIteration:
        pass
    return izip(a, b)

def ssd(a, b):
    """Returns the sum-of-squared-differences between the two patches"""
    ret = 0
    for i, j in zip(a, b):
        ret += (i-j)**2
    return math.sqrt(ret)

def getStartingLocation(im, size):
    """Returns a starting template of the given size from the image"""
    return tuple([i//2 for i in im.size])

def getLocations(im, size):
    """Generator to return locations of patches of a given size"""
    for y in range(size[1], im.size[1]-size[1]):
        for x in range(size[0], im.size[0]-size[0]):
            yield (x, y)

@cachemethod
def getPatch(im, loc, size):
    """Returns the patch at the given location and size, as a list"""
    ret = []
    pix = im.load()
    for y in range(loc[1]-size[1], loc[1]+size[1]+1):
        for x in range(loc[0]-size[0], loc[0]+size[0]+1):
            ret.append(pix[x, y])
    return tuple(ret)

class BTreeNode(object):
    """One node of a b-tree"""
    def __init__(self, template, parent=None, breaks=[]):
        """Creates a new node with the given breakpoints"""
        self.template = template
        self.children = []
        self.breaks = breaks
        self.parent = parent
        self.data = []

    def getParent(self): return self._parent
    def setParent(self, parent):
        self._parent = parent
        if self._parent:
            self._parent.children.append(self)
    parent = property(getParent, setParent)

    def getName(self):
        """Returns a suitable name for this node"""
        return "node_%d_%d" % (self.template[0], self.template[1])

    def getGraphString(self, root=1):
        """Returns a string which can be fed to graphviz to display this graph"""
        ret = []
        if root:
            ret.append('digraph {')
        # the information for this node
        s = '%s [label = "<f0> %s' % (self.getName(), self.template)
        if self.children:
            for i, b in enumerate(self.breaks):
                s += '| <f%d> %0.1f' % (i+1, b)
        else:
            for i, d in enumerate(self.data):
                s += '| <f%d> %s' % (i+1, d)
        s += '" shape = "record"];'
        ret.append(s)
        for i, c in enumerate(self.children):
            ret.append('%s:f%d -> %s:f0' % (self.getName(), i, c.getName()))
            ret.append(c.getGraphString(0))
        if root:
            ret.append("}")
        return '\n'.join(ret)

def equalBin(dists, num=NUM_BINS):
    """Returns breakpoints between the given distances"""
    if len(dists) < num: return None
    # divide up the list into 'num' different segments
    incr = int(math.ceil(len(dists)/float(num)))
    groups = list(grouper(incr, dists))
    # go through each pair and find the breakpoint between them
    breaks = []
    for a, b in pairwise(groups):
        breaks.append((a[-1][0]+b[0][0])/2.0)
    #print "Equal bin called with %d dists and num=%d returning with %d groups and %d breaks" % (len(dists), num, len(groups), len(breaks))
    assert len(groups) <= num
    assert len(breaks) <= num-1
    groups[-1] = tuple([i for i in groups[-1] if i])
    return (breaks, groups)
    
class SelfSimilarity(object):
    """Simple class to encapsulate finding self-similar portions"""
    def __init__(self, im, locs, startloc, size=DEFAULT_SIZE, getDist=ssd, getBreaks=equalBin):
        """Basic init"""
        from math import floor, ceil, log
        self.im = im
        self.locs = locs
        self.size = size
        self.lookups = {}
        self.getDist = ssd
        self.getBreaks = getBreaks
        for l in locs:
            self.lookups[l] = []
        # do the first level of the tree
        print "Getting root node"
        self.rootnode, groups = self.doOneLevel(startloc, locs, None)
        # now recursively do the rest
        numLevels = int(floor(log(len(locs))/log(NUM_BINS))) - 2
        self.nodes = [self.rootnode]
        print "Building tree"
        self.go(self.rootnode, groups)
        print "Done"
        for k in self.lookups.keys()[:10]:
            print k, self.lookups[k]
        print len(self.nodes)
        s = self.rootnode.getGraphString()
        f = open('graph', 'w')
        f.write(s)
        f.close()
        self.outputImage()

    def go(self, curnode, groups):
        """Builds the subtree rooted at the given node"""
        lens = [len(g) for g in groups]
        if min(lens) < NUM_BINS:
            # we've reached a leaf node, so set the data
            for g in groups:
                curnode.data.extend(g)
            print "  Returning because we only got min %d items in a group" % (min(lens))
            return None
        #print "Recursing on a node with %d groups" % (len(groups))
        for g in groups:
            # select a patch from this group to match against
            node, newgroups = self.doOneLevel(g[len(g)//2], g, curnode)
            self.nodes.append(node)
            # now recurse
            self.go(node, newgroups)

    def doOneLevel(self, patchloc, locs, parent):
        """Does one level of the segmentation"""
        # get distances to all other patches
        patch = getPatch(self.im, patchloc, self.size)
        print "Doing a level with %d locs" % (len(locs))
        dists = []
        for i, l in enumerate(locs):
            d = self.getDist(patch, getPatch(im, l, self.size))
            dists.append((d, l))
        dists.sort()
        # separate these out into groups and find the breakpoints between them
        breaks, groups = self.getBreaks(dists)
        #print "Breaks: ", breaks
        # add the bin numbers for each group to the lookup table
        for i, g in enumerate(groups):
            temp = []
            #print "Group %d: %s" % (i, g[-4:])
            for d, l in g:
                self.lookups[l].append(i)
                temp.append(l)
            # also remove the distances from the groups, since we don't care about them anymore
            groups[i] = temp
        btn = BTreeNode(patchloc, parent, breaks)
        return btn, groups

    def outputImage(self):
        """Writes an image to disk, marking the leaf nodes of this tree in one color"""
        from random import randint
        im = Image.new('RGB', self.im.size, (0, 0, 0))
        pix = im.load()
        for node in self.nodes:
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            if not node.data: continue
            for x, y in node.data:
                pix[x, y] = (r, g, b)
        im.save('similar.png')

def selfSimilarity(im, size=DEFAULT_SIZE, getDist=ssd, getBreaks=equalBin):
    """Returns an image and a datastructure that represents self-similarity within the image"""
    outim, outds = None, None
    # choose a starting location
    l0 = getStartingLocation(im, size)
    locs = tuple(getLocations(im, size))
    ss = SelfSimilarity(im, locs, l0, size)
    #print breaks
    return (outim, outds)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python %s <input image> [<output image name> [<output datastruct name>]]" % (sys.argv[0])
        sys.exit()
    
    infname = sys.argv[1]
    try:
        outimfname = sys.argv[2]
    except IndexError:
        outimfname = infname.rsplit('.', 1)[0] + '_out.png'
    try:
        outdsfname = sys.argv[3]
    except IndexError:
        outdsfname = infname.rsplit('.', 1)[0] + '_out.txt'
    
    im = Image.open(infname).convert('L')
    assert(im.size[0] > 0 and im.size[1] > 0)
    outim, outds = selfSimilarity(im)
    #outim.save(outimfname)
    #pickle.dump(outds, outdsfname)
