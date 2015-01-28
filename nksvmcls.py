#!/usr/bin/env python
"""SVM Classification Script, written by Neeraj Kumar.

Licensed under the 3-clause BSD License:

Copyright (c) 2012, Neeraj Kumar (neerajkumar.org)
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

from nktrainutils import *
from optparse import OptionParser

VERSION = '1.0'
DEFAULT_OUTFMT = '%(cls)s'
DEFAULT_DELAY = 1.0
DEFAULT_ERRVAL = -99999
DEFAULT_CACHE_DIR = 'cache/'

USAGE = '''

Classifies data using a trained SVM model.
The data is read from stdin, and can be given as raw feature vectors,
or indices to lookup into a set of feature dictionaries.
Classification outputs are printed to stdout, in the same order as inputs.
Processing can optionally be batched for efficiency.
'''

class SVMClassifier(FeatureParser):
    """A class encapsulating the functionality of SVM classification.
    This is here for convenience in using this module from other
    python scripts, although the module is most optimized for
    running on the command line.
    """
    def __init__(self, model, featfiles=None, indexmode='', columns='', urlbase='', cachedir=DEFAULT_CACHE_DIR, errval=DEFAULT_ERRVAL, prescaled=0):
        """Initialize this with the given model, and feature dictionary filenames.
        The model can either be a filename or an actual model."""
        FeatureParser.__init__(self, featfiles=featfiles, indexmode=indexmode, columns=columns, urlbase=urlbase, cachedir=cachedir)
        if isinstance(model, basestring):
            try:
                model = SGDSVM.load(model) #TODO see how to deal with non-SGD classes
                self.SVMCls = SGDSVM()
            except:
                model = readSVMModelAndParams(model)
                self.SVMCls = LibSVM()
        self.model = model
        self.errval = errval
        self.prescaled = prescaled

    def _process(self, rows, callback=None):
        """Processes a batch of rows and returns a list of outputs.
        Note that you should use classify() for compatibility.
        This also calls the callback as necessary.
        The rows given to this function should NOT be from a generator."""
        #TODO check for ndims
        fvecs = self.getfeatures(rows)
        results = [self.errval]*len(rows)
        # partition fvecs into valid and invalid
        partfvecs, indices = partitionByFunc(fvecs, lambda f: 0 if f is None else 1)
        # run classification
        #rawresults = bulkclassifyscores(self.model, partfvecs[1], applyscales=not self.prescaled)
        #print str(partfvecs[1])[:200]
        #print self.prescaled
        rawresults = self.SVMCls.classify(self.model, partfvecs[1], applyscales=not self.prescaled)
        for i, score in enumerate(rawresults):
            results[indices[(1, i)]] = score
        if callback:
            for row, output in zip(rows, results):
                callback(row, output)
        return results

    def classify(self, inputs, delay=DEFAULT_DELAY, callback=None):
        """Classifies the given inputs and returns a list of classification outputs.
        Note that this is a generator function (uses 'yield').
        This tries to batch things by waiting for 'delay' seconds
        to accumulate inputs before bulk-classifying all of them.
        If you set this to 0, it will call classify separately for each item.
        If you set this to < 0, it will call classify on all inputs together.
        The inputs can be a generator if you wish.
        You can optionally pass in a callback function that is called
        with (input, classification output).
        """
        if delay < 0: # batch all
            inputs = list(inputs)
            results = self._process(inputs, callback=callback)
            for c in results:
                yield c
        elif delay == 0: # one-by-one
            for row in inputs:
                result = self._process([row], callback=callback)[0]
                yield result
        else: # batched by time
            inputs = iter(inputs)
            while 1:
                #FIXME this is broken when reading inputs from stdin that doesn't close
                #print >>sys.stderr, 'at top of loop'
                last = time.time()
                rows = []
                try:
                    # build up the list of rows
                    for row in inputs:
                        rows.append(row)
                        if time.time() - last >= delay: break
                    else:
                        if not rows:
                            raise StopIteration
                    # we've hit our delay, so process now
                    if rows:
                        results = self._process(rows, callback=callback)
                        for c in results:
                            yield c
                except StopIteration:
                    break

def readinputs():
    """Reads inputs and returns them as an iterator"""
    while 1:
        try:
            l = sys.stdin.readline().rstrip('\n')
            if not l: break
            yield l
        except IOError:
            break


def main():
    """Main method"""
    # setup command parser
    usage = 'Usage: python %s [opts] <svm model>' % (sys.argv[0]) + USAGE
    parser = OptionParser(usage=usage, version=VERSION)
    parser.add_option('-f', '--featfiles', dest='featfiles', action='append', help='feature dictionary filenames, or assume input contains raw feature vectors if not given')
    parser.add_option('-i', '--indexmode', dest='indexmode', default='', help='how to interpret input indices [default: '']')
    parser.add_option('-C', '--columns', dest='columns', help='0-indexed columns to extract from input (inclusive ranges and/or comma-separated)')
    parser.add_option('-u', '--urlbase', dest='urlbase', default='http://s3.amazonaws.com/neerajkumar-pub/projects/clothing/', help='base url from where to download feature dictionaries')
    parser.add_option('-c', '--cachedir', dest='cachedir', default=DEFAULT_CACHE_DIR, help='cachedir to store downloaded feature dictionaries to [default: %s]' % (DEFAULT_CACHE_DIR))
    parser.add_option('-o', '--outfmt', dest='outfmt', default=DEFAULT_OUTFMT, help='format string defining what to print (python style), with vars "cls" (classification output) and "fname" (input filename) [default: %s]' % (DEFAULT_OUTFMT))
    parser.add_option('-d', '--delay', dest='delay', type='float', default=DEFAULT_DELAY, help='delay, in secs, between classification runs. If 0, process things one-by-one. If < 0, batch all items. [default: %f]' % DEFAULT_DELAY)
    parser.add_option('-e', '--errval', dest='errval', type='float', default=DEFAULT_ERRVAL, help='error value (float) if problem computing features on some input. [default: %f]' % DEFAULT_ERRVAL)
    parser.add_option('-p', '--prescaled', dest='prescaled', action='store_true', default=0, help='features are prescaled for classifier [default: apply scaling]')
    # parse arguments
    opts, args = parser.parse_args()
    try:
        c = SVMClassifier(args[0], opts.featfiles, indexmode=opts.indexmode, columns=opts.columns, cachedir=opts.cachedir, urlbase=opts.urlbase, errval=opts.errval, prescaled=opts.prescaled)
    except Exception:
        parser.error('First argument must be a valid SVM trained model!')

    # set up the printing
    opts.outfmt = opts.outfmt.replace(r'\t', '\t').replace(r'\n', '\n')
    def callback(input, output):
        """Callback which prints results, formatted as the user requested"""
        fmtdict = dict(cls=output, fname=input)
        s = opts.outfmt % fmtdict
        print s
        sys.stdout.flush()

    # set up the classification and start it
    inputs = readinputs()
    out = c.classify(inputs, delay=opts.delay, callback=callback)
    for o in out:
        pass


if __name__ == '__main__':
    main()
