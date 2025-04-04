#!/usr/bin/env python
"""This module allows for easy use of PCA
"""
import os, sys, time
from PIL import Image
from numpy import *

# SIMPLE MATRIX MANIPULATION
def clamp(x):
    return min(max(x, 0), 255)

def submean(data):
    """Subtracts the mean from the data.
    Samples should be in cols, and dimensions as rows"""
    means = mean(data, 1).reshape(data.shape[0],1)
    data = data - means
    return data, means

def getCertainRows(data, toget):
    """Returns a matrix with and without certain rows in it."""
    yes = data[toget, :]
    opposite = sorted(list(set(range(len(data)))-set(toget)))
    no = data[opposite, :]
    return yes, no

def normcols(m):
    """Normalizes each column to have unit length"""
    s = sum(m**2, 0)
    s[abs(s)<0.00001] = 1.0
    m /= s
    return m

def sortvals(values, vecs, ndims=-1):
    """Sorts the given set of values and vectors (in columns) by decreasing value.
    Keeps only the given number of dimensions (or all if ndims < 0)"""
    if ndims <= 0:
        ndims = len(values)
    perm = argsort(-values)  # sort in descending order
    perm = perm[:ndims]
    values = values[perm]
    vecs = vecs[:, perm]
    return (values, vecs)

def mat2file(mat, fname):
    f = open(fname, 'wb')
    for row in mat:
        print >>f, ' '.join('%f' % x for x in row)
    f.close()

# CORE PCA FUNCTIONS
def sortedeigs(m, dtype=float64):
    """Returns sorted eigenvalues and -vectors"""
    log('    Starting eigendecomposition on matrix of shape %s now' % (m.shape,))
    t1 = time.time()
    values, vecs = linalg.eigh(m)
    log('    Finished eigendecomposition in %0.3f secs' % (time.time()-t1))
    if vecs.dtype != dtype:
        vecs = vecs.astype(dtype)
    vecs = normcols(vecs)
    return sortvals(values, vecs)

def sparseeigs(m, ndims):
    """Does a sparse eigenvalue problem, with the given number of dimensions"""
    import cvxopt
    import cvxopt.lapack as lp

    log('    Starting sparse eigendecomposition on matrix of shape %s now using %d dims' % (m.shape, ndims))
    t1 = time.time()
    N = len(m)
    if ndims < 0 or ndims > N:
        ndims = N
    m = cvxopt.base.matrix(m)
    evals = cvxopt.base.matrix([0.0]*N)
    evecs = cvxopt.base.matrix(0.0, (N, ndims))
    ret = lp.syevx(m, evals, jobz='V', range='I', il=N-ndims+1, iu=N, Z=evecs)
    log('    Finished sparse eigendecomposition in %0.3f secs' % (time.time()-t1))
    evals = array(evals)[:ndims, 0]
    evecs = array(evecs)
    return sortvals(evals, evecs)

def prunedims(values, vecs, keep=-1):
    """Prunes by user-defined number of dimensions. Also normalizes values (pre-cutoff).
    Assumes values and vecs are sorted in descending order already.

    'keep' determines how many dimensions to keep:
        <= 0: all dimensions (default)
        0.0 < keep <= 1.0: given percentage of total variance (must be a float)
        >= 1: given number of dimensions (must be an int)
    
    In all cases, at most len(values) will be kept.
    """
    t1 = time.time()
    tokeep = len(values)
    values /= sum(values)
    #log(values[:100])
    if keep >= 1 and int(keep) == keep:
        tokeep = min(tokeep, keep)
    elif keep > 0 and keep <= 1.0 and int(keep) != keep:
        normvals = cumsum(values)/sum(values)
        i = 1
        while normvals[i-1] < keep:
            i += 1
        tokeep = min(i, tokeep)
    assert tokeep > 0
    values = values[:tokeep]
    vecs = vecs[:, :tokeep]
    t2 = time.time()
    log('   For keep %s, got tokeep %s and pruned in %0.3f secs' % (keep, tokeep, t2-t1))
    return (values, vecs)

def pca(data, submeans=0, keep=-1, flip=1, sparse=0):
    """Auto-selecting PCA, with data in columns.
    The 'data' matrix should be ndims X npts.
    If npts > ndims, then does a PCA directly.
    If ndims > npts, then does PCA on transpose, and does appropriate normalization.
    Returns (eigvals, eigvecs)...or something equivalent.
    You can apply dot(vecs.T, data) to transform data, and dot(evecs, t) to transform back.

    'keep' determines how many dimensions to keep:
        <= 0: all dimensions (default)
        0.0 < keep <= 1.0: given percentage of total variance (must be a float)
        >= 1: given number of dimensions (must be an int)

    In all cases, at most min(data.shape) will be kept.

    If you definitely don't want to flip data, then set flip=0.
    """
    t1 = time.time()
    if submeans:
        data, means = submean(data)
    t2 = time.time()
    log('    Done subtracting means in %0.3f secs...' % (t2-t1))
    ndims, npts = data.shape
    if npts >= ndims or not flip:
        if sparse:
            assert type(keep) == type(123)
            values, vecs = sparseeigs(cov(data), keep)
            return prunedims(values, vecs, keep)
        else:
            values, vecs = sortedeigs(cov(data))
            return prunedims(values, vecs, keep)
    else:
        #TODO this path is broken right now
        assert 1 == 0, 'This path is broken!'
        d2 = data.transpose()
        t3 = time.time()
        log('    Computed transpose in %0.3f secs' % (t3-t2))
        c = cov(d2) # 3.35 secs
        t4 = time.time()
        log('    Computed covariance in %0.3f secs' % (t4-t3))
        values, vecs = linalg.eigh(c) # 0.34 secs
        t5 = time.time()
        log('    Computed eigendecomposition in %0.3f secs' % (t5-t4))
        del c
        ndims = len(values)

        #log(values)
        #log(vecs)
        for i in range(ndims):
            if abs(values[i]) > 0.00001:
                values[i] = sqrt(1/values[i]/(max(data.shape))) # needed for normalization...why? # fast
        values, vecs = sortvals(values, vecs)
        # TODO prune dimensions here (before matrix-mult)?
        v2 = dot(data, vecs) * values # 2.53
        t6 = time.time()
        v2 = normcols(v2)
        t7 = time.time()
        if 0:
            log(values)
            log('Should be I: %s' % (dot(v2.T, v2),)) # identity...good
            t = dot(v2.T, data)
            log('T: ', t)
            log('%s' % (data.shape, d2.shape, values.shape, vecs.shape, v2.shape,))
            # (5788, 500) (500, 5788) (500,) (500, 500) (500, 500) (5788, 500) (5788, 500)
            log(dot(v2, t))
            log(data) # this should equal the previous line
        log('    Times were %s' % ([t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6],))
        sys.stdout.flush()
        return prunedims(values, v2, keep)

def ipca_unthreaded(means, datacallback, keep=-1, dtype=float64, init=None, sonly=0, sparse=0):
    """Iterative PCA, which builds up the scatter matrix iteratively.
    This iterates over datacallback repeatedly to get chunks of data.
    Each chunk C should be of size ndims X chunk_size (where chunk_size 
    can vary between chunks). A scatter matrix is computed by first 
    subtracting means, and then taking C * C.T. This is added to a running
    total scatter matrix S. Finally, when the datacallback is exhausted
    (e.g., end of generator or list), then PCA is run on S, and eigvalues are sorted.
    
    Returns (eigvals, eigvecs).
    
    'keep' determines how many dimensions to keep:
        <= 0: all dimensions (default)
        0.0 < keep <= 1.0: given percentage of total variance (must be a float)
        >= 1: given number of dimensions (must be an int)
    
    In all cases, at most min(len(means)) will be kept.

    You can also pass in an S matrix as initialization (otherwise zeros() is used).
    If sonly is true, then only returns the complete S matrix.
    """
    t1 = time.time()
    ndims = max(means.shape)
    means = means.reshape((ndims,1))
    if init:
        assert init.shape[0] == ndims == init.shape[1], 'Given init must have shape (%d, %d) but had shape %s' % (ndims, ndims, s.shape)
        s = init
    else:
        s = zeros((ndims, ndims), dtype=dtype)
    t2 = time.time()
    log('    Done initializing S matrix of shape %s in %0.3f secs...' % (s.shape, t2-t1))
    tot = 0
    num = 0
    for i, d in enumerate(datacallback):
        d -= means
        num += d.shape[1]
        t3 = time.time()
        s += dot(d, d.T)
        tot += time.time()-t3
        log('      On iteration %d of datacallback and got %d total elements, added in %0.3f secs' % (i+1, num, tot))
    if sonly: return s
    if sparse:
        assert type(keep) == type(123)
        values, vecs = sparseeigs(s, keep)
    else:
        values, vecs = sortedeigs(s, dtype=dtype)
    return prunedims(values, vecs, keep)

def ipca_threaded(means, datacallback, keep=-1, dtype=float64, init=None, sonly=0, sparse=0):
    """Iterative PCA, which builds up the scatter matrix iteratively.
    This is a threaded version, which is faster if the datacallback takes some time...
    This iterates over datacallback repeatedly to get chunks of data.
    Each chunk C should be of size ndims X chunk_size (where chunk_size 
    can vary between chunks). A scatter matrix is computed by first 
    subtracting means, and then taking C * C.T. This is added to a running
    total scatter matrix S. Finally, when the datacallback is exhausted
    (e.g., end of generator or list), then PCA is run on S, and eigvalues are sorted.
    
    Returns (eigvals, eigvecs).
    
    'keep' determines how many dimensions to keep:
        <= 0: all dimensions (default)
        0.0 < keep <= 1.0: given percentage of total variance (must be a float)
        >= 1: given number of dimensions (must be an int)
    
    In all cases, at most min(len(means)) will be kept.

    You can also pass in an S matrix as initialization (otherwise zeros() is used).
    If sonly is true, then only returns the complete S matrix.
    """
    t1 = time.time()
    ndims = max(means.shape)
    means = means.reshape((ndims,1))
    if init:
        assert init.shape[0] == ndims == init.shape[1], 'Given init must have shape (%d, %d) but had shape %s' % (ndims, ndims, s.shape)
        s = init
    else:
        s = zeros((ndims, ndims), dtype=dtype)
    t2 = time.time()
    log('    Done initializing S matrix of shape %s in %0.3f secs...' % (s.shape, t2-t1))
    tot = 0
    num = 0
    from Queue import Queue
    qsize=5
    q = Queue(qsize)
    from threadutils import spawnWorkers
    def loadq(q, callback):
        for d in callback:
            d -= means
            q.put(d)
        q.put(None)

    datathread = spawnWorkers(1, loadq, args=(q, datacallback))[0]
    i = 0
    while 1:
        d = q.get()
        i += 1
        if d is None: break
        num += d.shape[1]
        t3 = time.time()
        s += dot(d, d.T)
        tot += time.time()-t3
        #log('      On iteration %d of datacallback and got %d total elements, added in %0.3f secs' % (i, num, tot))
    if sonly: return s
    if sparse:
        assert type(keep) == type(123)
        values, vecs = sparseeigs(s, keep)
    else:
        values, vecs = sortedeigs(s, dtype=dtype)
    return prunedims(values, vecs, keep)

def ipca(*args, **kw):
    if kw.get('threaded', 0):
        try:
            del kw['threaded']
        except ValueError: pass
        return ipca_threaded(*args, **kw)
    return ipca_unthreaded(*args, **kw)

def pcatransform(data, evecs):
    """Transforms data into PCA coordinates using the given evecs."""
    ret = dot(evecs.T, data)
    return ret

def pcainvtransform(trans, evecs):
    """Transforms data back from PCA coordinates to original ones."""
    ret = dot(evecs, trans)
    return ret


# OUTPUT FUNCTIONS
def evec2im(arr, mask):
    """Makes a grayscale eigenimage from the given principal component vector.
    The vector must have exactly as many elements as the number of '1' pixels in the mask,
    which should be of type '1'"""
    im = Image.new('L', mask.size, 0)
    mdat = tuple(mask.getdata())
    dat = list(im.getdata())
    cur = 0
    m, M = min(arr), max(arr)
    fac = 255.0/(M-m)
    arr = [int((a-m)*fac) for a in arr]
    for i in xrange(len(dat)):
        if not mdat[i]: continue
        dat[i] = arr[cur]
        cur += 1
    assert cur == len(arr)
    im.putdata(dat)
    return im

def distances2im(m, func=lambda a: clamp(int(log(a+1)*128/log(2))), fac=25):
    """Returns a distance matrix as an image.
    The given function is used to map matrix values to numbers from 0-255.
    These numbers are then converted to a spectrum from red to blue.
    The fac is the factor by which to multiply each pixel in the distance matrix."""
    n = len(m)
    im = Image.new('RGB', (n, n))
    all = []
    for row in m:
        all.extend(row)
    all = [func(a) for a in all]
    def getc(x):
        if x < 128: return (256-x, 0, 0)
        if x > 128: return (0, 0, x)
        if x == 128: return (0, 0, 0)
    colors = [getc(x) for x in all]
    im.putdata(colors)
    nsize = (n*fac, n*fac)
    return im.resize(nsize)


# TEST FUNCTION
def checkpca():
    """Tests pca, and shows how to use all the different functions"""
    set_printoptions(precision=7, linewidth=150, suppress=1)
    M = array([[1,2,3,4,5,6], [2.2,4,6,8,10,12], [3,6,9,12,15,18]]).transpose()
    #M = array([[1,2,3,4,5,6], [2.2,4,6,8,10,12], [3,9,18,24,30,36]]).transpose()

    m, means = submean(M)
    log('M is %s, %s, mean is %s, %s and m is %s, %s' % (M.shape, M, means.shape, means, m.shape, m))
    evals, evecs = pca(m, keep=2, flip=0)
    log('Got evals %s, %s, evecs %s, %s' % (evals.shape, evals, evecs.shape, evecs))
    t = pcatransform(m, evecs)
    log('Got Transformed matrix %s, %s' % (t.shape, t))
    newm = pcainvtransform(t, evecs)
    log(newm)
    final = newm + means
    log('After adding back mean, got %s' % (final,))
    res = final-M
    resp = res/M
    log('Residuals are %s and Percentages are %s' % (res, resp))

def checkipca(ndims=100, npts=1000):
    """Checks ipca"""
    set_printoptions(precision=7, linewidth=150, suppress=1)
    # generate gaussian data with 0 mean and random variances
    vars = 5*abs(random.rand(ndims))
    #log('Variances: ', vars)
    m = random.normal(zeros((npts, ndims)), scale=vars).T
    #log(m)
    #log('Means: ', mean(m,1))
    log('Shape: ', m.shape)
    if 0:
        # first do it the old-fashioned way
        t1 = time.time()
        evals, evecs = pca(m, flip=0)
        t2 = time.time()
        log('Dense  Eigs in %0.3f secs, evals: %s' % (t2-t1, evals[:5]))
        log(evecs)

        # now using cvxopt
        t3 = time.time()
        evals, evecs = pca(m, flip=0, sparse=1, keep=10)
        t4 = time.time()
        log('Sparse Eigs in %0.3f secs (%0.3fX faster), evals: %s' % (t4-t3, ((t2-t1)/(t4-t3)), evals[:5]))
        log(evecs)

    # now do it the ipca way
    def gen(x, ss=100, lag=0.04):
        from_ = 0
        for i in range(ss):
            to = from_ + npts//ss
            time.sleep(lag)
            yield x[:,from_:to]
            from_ = to
    
    t5 = time.time()
    evals, evecs = ipca(zeros(ndims), gen(m))
    t6 = time.time()
    log('New eigs in %0.3f secs, evals: %s' % (t6-t5, evals[:5]))
    log(evecs)

    t7 = time.time()
    evals, evecs = ipca(zeros(ndims), gen(m), sparse=1, keep=10)
    t8 = time.time()
    log('New eigs in %0.3f secs, evals: %s' % (t8-t7, evals[:5]))
    log(evecs)

    t9 = time.time()
    evals, evecs = ipca_threaded(zeros(ndims), gen(m), sparse=1, keep=10)
    t10 = time.time()
    log('New eigs in %0.3f secs, evals: %s' % (t10-t9, evals[:5]))
    log(evecs)

def checkprune():
    """Checks the prune function to make sure it's right"""
    vals = [0.08234926,0.05706,0.02484902,0.021391,0.02049739,0.01856545,0.01694183,0.01646052,0.01470734,0.01352113,0.01330168,0.0127809,0.01255655,0.01089848,0.01018878,0.00985537,0.00914789,0.00885832,0.00842793,0.00825101,0.00790971,0.00740127,0.00725265,0.00715757,0.00708191,0.00676072,0.00643255,0.00617226,0.00613682,0.00599727,0.00583192,0.00579706,0.0055662,0.00547047,0.00529036,0.00520864,0.00502711,0.0048735,0.00482254,0.00454305,0.00448793,0.00441308,0.00425876,0.00424894,0.00418519,0.00407819,0.00393067,0.00386274,0.00374639,0.00370774,0.00363166,0.00359066,0.00351069,0.00347037,0.00336404,0.00331573,0.00330471,0.00327859,0.00322214,0.00317321,0.0031603,0.00310918,0.00306605,0.00301577,0.00293041,0.0028991,0.0028835,0.00279775,0.00275721,0.00272268,0.0026815,0.00261214,0.00257703,0.00255544,0.00252881,0.00246848,0.00244149,0.00241617,0.0023928,0.00234767,0.00233682,0.00232134,0.00227421,0.00222894,0.00220016,0.00217564,0.00216032,0.00208491,0.00208388,0.00205186,0.00202657,0.0020237,0.00200573,0.0019495,0.00193219,0.00189462,0.00187787,0.0018654,0.00182752,0.00181351]
    vecs = zeros((10,len(vals)))
    prunedims(vals, vecs, keep=0.95)

if __name__ == '__main__':
    #checkprune()
    checkipca(1000,10000)
    #checkpca()
