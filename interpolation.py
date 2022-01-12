"""Various interpolation utilities.

Written by Neeraj Kumar
"""

import os, sys, time
from nkpylib.utils import lpdist, lerp

class Interpolation(object):
    """Abstract base class"""
    def __init__(self, indexes, pts, distfunc=lpdist):
        """Creates an interpolation class using the given set of pts, which are saved.
        The pts are indexed by the given indexes.
        A distance function used to get distances between pts can be given.
        Default distance function is lpdist (which is L2 by default)"""
        assert len(indexes) == len(pts)
        self.indexes = indexes
        self.pts = pts
        self.idims = len(indexes) if indexes else 0
        self.pdims = len(pts[0]) if pts else 0
        self.distfunc = distfunc

    def copypt(self, pt):
        """Copies the given point to a new point.
        This handles different data types"""
        if isinstance(pt, list):
            return pt[:]
        if isinstance(pt, tuple):
            return tuple(pt[:])
        try:
            import numpy
            if isinstance(pt, numpy.ndarray):
                return pt.copy()
        except ImportError: pass
        raise NotImplementedError('No copy handler for point type %s' % (type(pt)))

    def interpolate(self, qi):
        """Interpolates using a single query index."""
        raise NotImplementedError

    def curvature(self, qi, idist):
        """Returns the curvature of the manifold at the given query index,
        as computed within the given index distance. The returned curvature
        should be non-negative, with larger values = more curved."""
        raise NotImplementedError

    def distance(self, qpt):
        """Returns the distance of a given query pt (NOT index) to our "manifold"."""
        raise NotImplementedError

    def project(self, qpt):
        """Projects a query point onto our "manifold" to find its equivalent index and pt.
        Returns (projected index, projected point, projection distance)"""
        raise NotImplementedError

class NNInterpolation(Interpolation):
    """Simple Nearest Neigbhors interpolation.
    Keeps points as points and simply finds the nearest neighbor
    for interpolations and distance calculations."""
    def interpolate(self, qi):
        """Returns the point with closest index value"""
        best = min((abs(qi-i), pt) for i, pt in zip(self.indexes, self.pts))[1]
        return best

    def distance(self, qpt):
        """Returns the distance of a given query pt (NOT index) to the closest point"""
        d = min(self.distfunc(qpt, pt) for pt in self.pts)
        return d

    def project(self, qpt, k=1):
        """Takes the average of the k-nearest neighbors and their indices.
        Returns (avg(indexes(close_k)), avg(pts(close_k)), avg(dist(pts(close_k))))"""
        close = sorted([(self.distfunc(qpt, pt), i) for i, pt in enumerate(self.pts)])[:k]
        avgi = sum(self.indexes[i] for d, i in close)/len(close)
        avgpt = sum(self.pts[i] for d, i in close)/len(close)
        avgd = sum(d for d, i in close)/len(close)
        return (avgi, avgpt, avgd)

class PiecewiseLinearInterpolation(Interpolation):
    """Piecewise linear interpolation.
    Forms linear segments between points by increasing index for interpolation."""
    def __init__(self, indexes, pts, *args, **kw):
        """Saves indexes, points, etc. but reordered so that they can be lerped"""
        indexes, pts = zip(*sorted(zip(indexes, pts)))
        Interpolation.__init__(self, indexes, pts, *args, **kw)

    def _nearestindexlocs(self, qi):
        """Finds the nearest index locations to the given query index.
        Returns a pair (i1, i2) where indexes[i1] <= qi <= indexes[i2].
        If qi is outside the full range, i1 == i2."""
        from bisect import bisect
        idxs = self.indexes
        # return the first pt if our query index is at or before the start
        if qi <= idxs[0]: return (0, 0)
        # return the last pt if our query index is at or past the end
        if qi >= idxs[-1]: return (len(idxs)-1, len(idxs)-1)
        # find the right spot
        I = bisect(idxs, qi)
        i = I-1
        return (i, I)

    def interpolate(self, qi):
        """Finds the linear segment where this query index falls and interpolates values"""
        idxs = self.indexes
        i, I = self._nearestindexlocs(qi)
        if i == I: return self.pts[i] # one of the endpoints
        # index
        x, X = idxs[i], idxs[I]
        # lerp each dim of the output
        ret = self.copypt(self.pts[0])
        for dim, (y, Y) in enumerate(zip(self.pts[i], self.pts[I])):
            ret[dim] = lerp(qi, (x,y), (X,Y))
        return ret

    def _linesegproj(self, qpt, p1, p2):
        """Projects the query point onto the line segment spanned by p1 and p2.
        Returns (projected pt, percentage along line p1 -> p2 (which are at 0 and 1), min dist)."""
        from nkpylib.utils import getTriAltitude
        # get the projection
        proj = getTriAltitude(qpt, p1, p2)
        d = -1
        for i in range(len(p1)):
            if p1[i] == p2[i]: continue # find a dimension with some change
            loc = lerp(proj[i], (p1[i], 0.0), (p2[i], 1.0))
            if loc < 0: # off the segment, closer to p1
                d = self.distfunc(qpt, p1)
                break
            if loc > 1.0: # off the segment, closer to p2
                d = self.distfunc(qpt, p2)
                break
            # valid projection, so d = its length
            d = self.distfunc(qpt, proj)
            break
        # at this point, if we didn't find any non-zero dimensions,
        # then we can pick either endpoint as our target. We set location to be None
        if d < 0:
            d = self.distfunc(qpt, p1)
            loc = None
        return (proj, loc, d)

    def distance(self, qpt):
        """Returns the distance from the query point to the closest line segment"""
        # find minimum over all segments
        d = min(self._linesegproj(qpt, p1, p2)[-1] for p1, p2 in zip(self.pts, self.pts[1:]))
        return d

    def curvature(self, qi, idist):
        """Returns the curvature of the manifold at the given query index within the given index dist.
        This currently finds projections at qi-idist and qi+idist and computes their angle with qi.
        Note that if idist is less than the nearest stored indexes, you get 0.
        You can set idist to a negative integer if you want to look that many indexes away.
        """
        from math import acos
        qp = self.interpolate(qi)
        if idist > 0:
            qp1 = self.interpolate(qi-idist)
            qp2 = self.interpolate(qi+idist)
        else:
            # find nearest index pts and increment them
            i, I = self._nearestindexlocs(qi)
            i = max(0, i+idist) # remember that idist is negative
            I = min(len(self.pts)-1, I-idist)
            qp1, qp2 = self.pts[i], self.pts[I]
        # use dot product: a . b = |a| |b| cos(theta)
        # approximate curvature by: cos(theta) + 1
        v1 = [a-q for a, q in zip(qp1, qp)]
        v2 = [a-q for a, q in zip(qp2, qp)]
        dot = sum(a*b for a, b in zip(v1, v2))
        if dot == 0:
            ret = 5
        else:
            ret = (dot/(lpdist(qp, qp1) * lpdist(qp, qp2))) + 1
        assert ret >= 0
        return ret

    def project(self, qpt):
        """Projects a query point onto the closest line segment to find its equivalent index and pt.
        Returns (projected index, projected point, projection distance)"""
        mind = self.distfunc(qpt, self.pts[0])
        ret = (self.indexes[0], self.pts[0], mind)
        for p1, p2, i1, i2 in zip(self.pts, self.pts[1:], self.indexes, self.indexes[1:]):
            proj, loc, d = self._linesegproj(qpt, p1, p2)
            if d > mind: continue
            # possible candidate for best
            mind = d
            # check loc to see whether the projected point is one of the endpoints
            if loc < 0:
                # the first point
                ret = (i1, p1, d)
            elif loc > 1:
                # the 2nd point
                ret = (i2, p2, d)
            else:
                # somewhere in between
                ret = (lerp(loc, (0.0, i1), (1.0, i2)), proj, d)
        return ret
