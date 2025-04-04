"""Verification using attributes.

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

import os, sys, time
from utils import *
from webutils import stor

# GLOBALS
CACHEDIR = 'cache/'
MAX_ROWS = 10000000
FACESERVICE = 'http://aphex.cs.columbia.edu:45455'

MODELS = {'lfw_v1.1': 'LFW Attribute Verification v1.1'}
DEFAULT_MODEL = 'lfw_v1.1'
DATASETS = {'lfw': 'LFW', 'pubfig': 'PubFig'}

# UTILS
@memoize
def getmodelfields(model):
    """For a given model, returns a list of fields needed."""
    if model not in MODELS:
        raise web.notfound('No model %s. Choices are: %s' % (model, ', '.join(MODELS)))
    if model == 'lfw_v1.1':
        mapping = dict(l.strip().split('\t', 1) for l in open('attrnames.txt'))
        fields = [mapping[l.strip()] for l in open('fields_lfw_v1.1.txt')]
    return fields

@memoize
def getmodel(model):
    ret = stor(fields=getmodelfields(model))
    ret.update(id=model, name=MODELS[model])
    return ret

def createAttrDicts():
    """Creates the attribute dictionaries"""
    ret = {}
    # lfw v1.1
    ret['lfw_v1.1'] = d = {}
    fields = getmodelfields('lfw_v1.1')
    for l in open('attrnames.txt'):
        num, name = l.strip().split('\t', 1)
        if name not in fields: continue
        d[num] = d[int(num)] = d[name] = name
    return ret

ATTRDICTS = createAttrDicts()

# MAIN WEB FUNCTIONALITY
def myrethandler(data, input, **kw):
    return rethandler(data, input, jsoncontent=1, **kw)

@memoize
def getdataset(id):
    """Returns a dataset dictionary for the one with the given id"""
    try:
        name = DATASETS[id]
    except KeyError:
        raise web.notfound('No dataset with id %s. Choices are: %s' % (id, ', '.join(DATASETS)))
    d = stor(name=name, id=id, imgs=[], names=[], groups={}, prefix='')
    # read relevant data from disk
    fname = 'dataset_%s.txt' % (id)
    if id == 'pubfig':
        d.prefix = 'http://faceserv.cs.columbia.edu/private/localpubfig/aligned/'
        def func(f):
            """Takes a dictionary from a line and extracts the info we need from it"""
            if len(d.imgs) > MAX_ROWS: return None
            p = f['person'].replace(' ', '_')
            name = '%s_%04d' % (p, int(f['imagenum']))
            url = '%s/%s.jpg' % (p, name)
            d.groups.setdefault(f['person'], []).append(len(d.names))
            d.imgs.append(url)
            d.names.append(name)
            return None
    elif id == 'lfw':
        d.prefix = 'http://faceserv.cs.columbia.edu/db/newsimilarity/lfw/'
        d.prefix = 'http://leaf.cs.columbia.edu/db/similarity_cropped/lfw/'
        def func(f):
            """Takes a dictionary from a line and extracts the info we need from it"""
            if len(d.imgs) > MAX_ROWS: return None
            url = f['fname'].split('/', 4)[-1]
            person, name = url.split('/', 1)
            name = name.rsplit('_', 1)[0]
            d.groups.setdefault(person, []).append(len(d.names))
            d.imgs.append(url)
            d.names.append(name)
            return None

    # make the actual read call here, ignoring its output (since we've already extracted all info)
    readDictOfVals(fname, specialize=0, func=func)
    return d


#TODO this function is copied from facesearch/www/simtrain.py...we should link to it if possible?
def getSimFeaturesFromFvals(fvals1, fvals2, meths):
    """Returns similarity features computed from the features for two objects.
    'meths' are one or more of 'absdiff', 'diffsq', 'prod', 'avg', 'concat' with a weight val, as a 2-ple.
    If 'weighted' is > 0, then weights differences using a gaussian of the given variance
    """
    from array import array
    allmeths = {'absdiff': lambda a,b: abs(a-b), 'diffsq': lambda a,b: (a-b)**2, 'prod': lambda a,b: a*b, 'avg': lambda a,b: (a+b)/2.0, 'concat': None}
    ret = array('d')
    for meth, weighted in meths:
        assert meth in allmeths
        variance = weighted
        try:
            if meth == 'concat':
                ret.extend(fvals1+fvals2) # concat
            else:
                func = allmeths[meth]
                if weighted > 0:
                    func = lambda a,b: allmeths[meth](a, b) * gaussian((a+b)/2.0, 0.0, weighted)
                ret.extend([func(f1,f2) for f1, f2 in zip(fvals1, fvals2)])
        except KeyError: pass
    return ret

#TODO this function is copied from facesearch/www/trainutils.py...we should link to it if possible?
def applyScales(fvals, scales):
    """Applies the given normalization scales to the set of feature values and returns a normalized set."""
    from array import array
    if not scales: return fvals
    assert len(fvals) == len(scales), 'Scales had length %d but fvals had length %d' % (len(scales), len(fvals))
    fvals = array('f', [(f-m)/(2*(s+0.000001)) for f, (m,s) in zip(fvals, scales)])
    return fvals

#TODO this function is copied from facesearch/www/trainutils.py...we should link to it if possible?
def bulkclassify(model, seqOfFvals):
    """Classify multiple items in bulk. Returns (label, value) pairs for each input."""
    try:
        seqOfFvals = [applyScales(fvals, model.scales) for fvals in seqOfFvals]
    except AttributeError: pass
    return model.predict_many(seqOfFvals)

@memoize
def readmodel(model):
    """Reads the model and parameters for the given model name.
    Returns (model, simmeths)"""
    if model not in MODELS:
        raise web.notfound('No model %s. Choices are: %s' % (model, ', '.join(MODELS)))
    modelfname = model+'.model'
    from svm import svm_model
    t1 = time.time()
    model = svm_model(modelfname)
    f = open(modelfname.replace('.model', '.params'))
    model.scales = eval(f.readline().strip())
    simmeths = eval(f.readline().strip())
    f.close()
    log('Loaded verification model for %s from %s with %d dims and simmeths %s in %0.3f secs' % (model, modelfname, len(model.scales), simmeths, time.time()-t1))
    return (model, simmeths)

@memoize
def getfdict(dataset, name):
    """Returns the feature vector (as a dict) for the given name in the given dataset"""
    import simplejson as json
    from urllib import urlopen
    from pprint import pprint
    if dataset not in DATASETS:
        raise web.notfound('No dataset with dataset %s. Choices are: %s' % (dataset, ', '.join(DATASETS)))
    fname = 'dataset_%s.txt' % (dataset)
    if dataset == 'pubfig':
        def func(f):
            """Takes a dictionary from a line and extracts the info we need from it"""
            p = f['person'].replace(' ', '_')
            n = '%s_%04d' % (p, int(f['imagenum']))
            if n != name: return None
            return specializeDict(f)
    elif dataset == 'lfw':
        def func(f):
            """Takes a dictionary from a line and extracts the info we need from it"""
            url = f['fname'].split('/', 4)[-1]
            person, n = url.split('/', 1)
            n = n.rsplit('_', 1)[0]
            if n != name: return None
            return specializeDict(f)

    # make the actual read call here, ignoring its output (since we've already extracted all info)
    fvec, junk = readDictOfVals(fname, specialize=0, func=func)
    if len(fvec) != 1:
        raise web.notfound('Invalid name %s for dataset %s' % (name, dataset))
    fvec = fvec[0]
    return fvec

def getfvec(fdict, model):
    """Takes a feature dict and converts it to a feature vector using the given model"""
    from array import array
    attrs = ATTRDICTS[model]
    vals = dict((attrs[k], v) for k, v in fdict.items() if k in attrs)
    ret = array('d', [vals[f] for f in getmodelfields(model)])
    return ret

@memoize
def verifypair(a, b, model, debug=0):
    """Verifies a pair using the given model.
    Each item of the pair is given as (dataset, name)."""
    def log(s):
        if debug:
            print >>sys.stderr, s
    ret = stor(fields=getmodelfields(model), modelid=model, modelname=MODELS[model])
    # read feature dicts
    fd1 = getfdict(a[0], a[1])
    log('fdict1  : %s' % (fd1))
    fd2 = getfdict(b[0], b[1])
    log('fdict2  : %s' % (fd2))
    # convert to feature vectors
    f1 = getfvec(fd1, model)
    log('fvec1   : %s' % (f1))
    ret.update(dataset1=a[0], name1=a[1], feats1=list(f1))
    f2 = getfvec(fd2, model)
    log('fvec2   : %s' % (f1))
    ret.update(dataset2=b[0], name2=b[1], feats2=list(f2))
    # read verification classifier
    svm_model, simmeths = readmodel(model)
    # get similarity fvec
    fvec = getSimFeaturesFromFvals(f1, f2, simmeths)
    log('simfvec : %s' % (fvec))
    # compute results
    label, score = bulkclassify(svm_model, [fvec])[0]
    ret.score = score * label
    log('score   : %s' % (ret.score,))
    # set same or diff and correct or not
    ret.same = ret.correct = 0
    if not ret.dataset1.startswith('job') and not ret.dataset2.startswith('job'):
        name = lambda n: n.lower().rsplit('_', 1)[0]
        same = (name(ret.name1) == name(ret.name2))
        ret.same = 1 if same else -1
        ret.correct = 1 if ret.same*ret.score > 0 else -1
    else:
        ret.same = ret.correct = None
    log('For %s and %s using model %s, got score %s, same %s, correct %s' % (a, b, model, ret.score, ret.same, ret.correct))
    return ret

def verifyall(fvec, dataset, svm_model, simmeths):
    """Verifies a fvec against an entire dataset using the given model.
    Results are a sorted list of (score, fname from the dataset)."""
    fnames, datafvecs = zip(*dataset.items())
    # get similarity fvecs
    simfvecs = [getSimFeaturesFromFvals(fvec, dfv, simmeths) for dfv in datafvecs]
    # compute results
    results = [(label*score, fname) for (label, score), fname in zip(bulkclassify(svm_model, simfvecs), fnames)]
    results.sort(reverse=1)
    return results

def readdataset(datasetfname):
    """Reads the dataset from the given fname.
    It must have a single non-attr field, called 'fname'
    Returns a dict of {fname: fvec}
    The fvecs are in order of fields from dataset.
    """
    faces, fields = readDictOfVals(datasetfname, specialize=1)
    fields.remove('fname')
    #log('Got %d fields: %s' % (len(fields), fields))
    #log('Got %d faces: %s' % (len(faces), faces[0]))
    log('Got %d faces and %d fields' % (len(faces), len(fields)))
    d2fvec = lambda d: [d[f] for f in fields]
    dataset = dict((f['fname'], d2fvec(f)) for f in faces)
    return dataset


def mainloop(dataset, svm_model, simmeths, inputs=sys.stdin):
    """Main loop which reads inputs from given stream or list of strings and yields results one-by-one"""
    #log('Read model %s with %d scales, %s' % (modelname, len(svm_model.scales), simmeths))
    for l in inputs:
        l = l.rstrip()
        # get format
        if '\t' in l:
            els = l.split('\t')
        elif ' ' in l:
            els = l.split(' ')
        else:
            els = [l]
        # figure out the format
        fname = ''
        if len(els) == 1: # must be a valid fname from the dataset
            fname = els[0]
            fvec = dataset[fname]
        elif len(els) == len(fields): # only fvec
            fvec = map(float, els)
        elif len(els) == len(fields)+1: # fname + fvec
            fname = els[0]
            fvec = map(float, els[1:])
        else:
            print 'Error'
            continue
        #print fvec
        results = verifyall(fvec, dataset, svm_model, simmeths)
        yield results



if __name__ == '__main__':
    modelname = DEFAULT_MODEL
    if len(sys.argv) < 2:
        print 'Usage: python %s <datasetfname> <model=[%s]> < inputs' % (sys.argv[0], model)
        sys.exit()
    datasetfname = sys.argv[1]
    try:
        modelname = sys.argv[2]
        assert modelname in MODELS
    except Exception: pass
    dataset = readdataset(datasetfname)
    # read model
    svm_model, simmeths = readmodel(modelname)
    for results in mainloop(dataset, svm_model, simmeths):
        s = '\t'.join('%s:%s' % (d, fname) for d, fname in results)
        print s
        sys.stdout.flush()
