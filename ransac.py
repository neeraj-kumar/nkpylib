# -*- coding: utf-8 -*-
"""Ransac utilities"""

import os, sys, time
import numpy as np

class Model(object):
    """A baseclass for a model"""
    def __init__(self, hypothesis, **kw):
        """Initializes this model with the given hypothesis"""
        pass

    def getinliers(self, pts, thresh):
        """Computes the inliers from the given pts, assuming the given threshold.
        Should return (inliers, error)."""
        raise NotImplementedError


class HomographyModel(Model):
    """A homography-based model"""
    from align import getHomography, transformPts # type: ignore
    def __init__(self, hyp, **kw):
        """Takes a hypothesis, which should have at least 4 pair.
        Each pair should be (x1, y1), (x2, y2).
        """
        assert len(hyp) >= 4
        self.hyp = hyp[:]
        # get the input and output locations
        inputs, outputs = zip(*hyp)
        self.trans, proj = getHomography(inputs, outputs)
        #print 'From %d hyp pairs, got trans\n%s' % (len(hyp), self.trans)

    def __repr__(self):
        """Returns a string representation of ourself"""
        return str(self.trans)

    def _projectAndGetDists(self, pts):
        """Projects points and returns distances"""
        inputs, outputs = zip(*pts)
        proj = transformPts(self.trans, inputs)
        outputs = np.array(outputs)
        dists = np.sqrt(np.sum((outputs-proj)**2, 1))
        return dists

    def getinliers(self, pts, thresh):
        """Returns the list of inliers within the given thresh.
        The error is computed as simply 1 - ninliers/npts.
        If you need to get an actual error value on a set of inliers,
        use geterror().
        """
        dists = self._projectAndGetDists(pts)
        inliers = [pt for pt, d in zip(pts, dists) if d < thresh]
        error = 1.0 - len(inliers)/float(len(pts))
        return inliers, error

    def geterror(self, inliers):
        """Returns the error for the given set of inputs (all assumed to be inliers).
        This is average reprojection error in L2, calculated as:
            sum(l2dist(input.project(), output))/len(inliers)
        """
        dists = self._projectAndGetDists(inliers)
        error = sum(dists)/float(len(inliers))
        return error


class Ransac(object):
    """Basic RANSAC"""
    def __init__(self, hypfunc, modelfunc, callback=None, **kw):
        """Initializes this ransac.
        Parameters:
            hypfunc - A function that takes the full 'data', and some kw, and
                      returns a list of potential inliers to build a model with.
                      You can also optionally pass in an integer.
                      In this case, it randomly picks that many items from data and returns them.
            modelfunc - takes a hypothesis list and returns a model.
                        This can be just the name of a Model subclass.
            callback - Called at the end of each loop iteration, where you can change params. Args:
                         niters - number of iterations done
                         elapsed - secs since we started ransac
                         best - dict with 'model', 'error', 'inliers'
                         checked - did we verify this round or not
                         ransac - this ransac instance
        Iteration parameters (instance vars):
            strongthresh [20]: the threshold to accept a strong inlier (for initial model)
            weakthresh [5]: the threshold to accept a weak inlier (for final count model)
            mininliers [1]: at least this many inliers must be found to even verify a model
            maxiters [1000]: maximum number of iterations to run. You can use estimateNiters() to set this.
            minerror [0.0]: if the error drops below this, we return
        """
        import random
        if isinstance(hypfunc, int):
            self.minpts = hypfunc
            hypfunc = lambda data, **kw: random.sample(data, self.minpts)
        self.hypfunc = hypfunc
        self.modelfunc = modelfunc
        self.callback = callback
        # parameters
        self.strongthresh = kw.get('strongthresh', 20)
        self.weakthresh = kw.get('weakthresh', 5)
        self.mininliers = kw.get('mininliers', 1)
        self.maxiters = kw.get('maxiters', 1000)
        self.minerror = kw.get('minerror', 0.0)
        # output
        self.best = dict(model=None, inliers=[], error=1e99, finalerror=1e99)

    def run(self, data, **kw):
        """Runs ransac with the given data.
        Returns a dict with keys 'model', 'inliers', 'error'.
        Sets any instance variables given in kw.
        """
        import time
        niters = 0
        self.__dict__.update(kw)
        best = self.best
        start = time.time()
        while niters < self.maxiters:
            # fit the model
            hyp = self.hypfunc(data, niters=niters)
            model = self.modelfunc(hyp)
            # get set of inliers
            inliers, error = model.getinliers(data, self.strongthresh)
            # if we have enough inliers, check for completion
            checked = 0
            if len(inliers) > self.mininliers:
                # fit model with all inliers and update if better than best
                checked = 1
                model = self.modelfunc(inliers)
                inliers, error = model.getinliers(data, self.weakthresh)
                if error < best['error']:
                    best.update(model=model, inliers=inliers, error=error)
            niters += 1
            if self.callback:
                self.callback(niters, elapsed=time.time()-start, best=best, checked=checked, ransac=self)
            if best['error'] <= self.minerror: break
        return best

    @classmethod
    def estimateNiters(cls, ptsPerHyp, inlierPerc, confidence=0.95, stddevs=2):
        """Estimates the maximum number of iterations needed to run.
        Parameters:
            ptsPerHyp - the number of points picked in each hypothesis (minimum number to fit a model)
            inlierPerc - the estimated fraction of inliers
            confidence - the desired confidence (0-1)
            stddevs - number of standard deviations to add (for robustness)
        """
        from math import log, sqrt
        #TODO set inlierperc = best # inliers found so far/num elements
        #TODO then p(no outliers) = 1.0 - w**ptsPerHyp
        inlierPerc = min(0.9, inlierPerc)
        wn = inlierPerc**ptsPerHyp # w^n
        try:
            n = int(log(1-confidence)/log(1-wn) + 0.5)
        except ZeroDivisionError:
            n = 1e99 # overflow!
        if stddevs > 0:
            n += stddevs*int(sqrt(1-wn)/(wn+1e-5) + 0.5)
        return n


def testRansacNiters():
    """Tests generation of niters for Ransac.
    We use data from Ondrej Chum's PhD thesis:
        http://cmp.felk.cvut.cz/~chum/Teze/Chum-PhD.pdf
    Here is what it should be:
                                        perc
           =========================================================================
        N  | 15%          20%            30%        40%         50%         70%
        ============================================================================
        2  | 132          73             32         17          10          4
        4  | 5916         1871           368        116         46          11
        7  | 1.75e6      2.34e5          1.37e4     1827        382         35
        8  | 1.17e7      1.17e6          4.57e4     4570        765         50
        12 | 2.31e10     7.31e8          5.64e6     1.79e5      1.23e4      215
        18 | 2.08e15     1.14e13         7.73e9     4.36e7      7.85e5      1838
        30 |    ∞            ∞           1.35e16    2.60e12     3.22e9      1.33e5
        40 |    ∞            ∞           ∞          2.70e16     3.29e12     4.71e6

    """
    r = Ransac(1, 1)
    for n in [2,4,7,8,12,18,30,40]:
        vals = []
        for perc in [.15, .2, .3, .4, .5, .7]:
            vals.append(r.estimateNiters(n, perc, stddevs=0))
        print('N %s: %s' % (n, vals))

if __name__ == "__main__":
    testRansacNiters()
