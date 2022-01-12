"""A layered graph, backed by redis.

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
import redis
from nkpylib.redisutils import *
import nkpylib.utils as utils
import numpy as np
from line_profiler import LineProfiler
profile = LineProfiler()

GENERIC_NODE_KEY_FMT = 'nodes:%(layername)s:%(id)s'

def spark(vals, f=sys.stdout):
    """Custom spark that wraps to 201 and scales by 1000"""
    return utils.spark(vals, wrap=201, scale=1000.0, f=f)

class RedisLayeredGraph(object):
    """A layered graph backed by redis.
    Connections should only be within the same level or adjoining level.
    However, you can have multiple "layers" at the same "level".
    """
    def __init__(self, db=None, reset=0, symmetric=1, prefix='', callbacks=[], **kw):
        """Initializes ourself.
        You can either pass in a redis connection in 'db', or give kw params.
        They should include 'host', 'port', 'db', and 'password'.
        Optionally, they can include 'charset' and 'socket_timeout'
        If you set reset=1, then it will reset the database.
        You can also add a prefix, which will be used whenever we need to save files.
        """
        self.db = db if db else redis.Redis(**kw)
        self.symmetric = symmetric
        if prefix:
            self.prefix = prefix
        else:
            # try to read it from the database
            self.prefix = self.db.get('prefix')
        self.edgematrices = {}
        self.layerkeys = {}
        self.dirty = {}
        self.callbacks = callbacks[:]
        if reset:
            print 'RESETTING DATABASE!!! Press control-c to cancel, within 5 seconds!'
            time.sleep(5)
            self.db.flushdb()
        self.db.set('prefix', self.prefix)

    @memoize
    def layerkey(self, layername):
        """Returns the key for a given layer name"""
        return 'layers:%s' % (layername)

    def getLayer(self, layername):
        """Returns a dict of the given layer"""
        ret = self.db.hgetall(self.layerkey(layername))
        ret = specializeDict(ret)
        return ret

    def layers(self):
        """Returns a list of (layername, level) pairs, in sorted order"""
        layers = [(name, int(level)) for name, level in self.db.hgetall('layers').items()]
        return sorted(layers, key=lambda pair: pair[1])

    @memoize
    def nodekeyfmt(self, layername):
        """Returns the node key format for the given layer"""
        return self.db.hget(self.layerkey(layername), 'nodekeyfmt')

    def nodekey(self, nodeprimary, layername):
        """Returns the node key given the primary id for the node and the layer.
        If this does not exist, then returns None"""
        id = self.db.hget('layers:%s:nodes' % (layername), nodeprimary)
        if not id: return None
        return self.nodekeyfmt(layername) % dict(layername=layername, id=id)

    def nodekeyFromID(self, nodeid, layername):
        """Returns the node key given the node id and the layer"""
        return self.nodekeyfmt(layername) % dict(layername=layername, id=nodeid)

    @memoize
    def splitNodekey(self, nodekey):
        """Splits a node key into (layername, node id)"""
        _, layername, nodeid = nodekey.split(':')
        return (layername, nodeid)

    def nodes(self, layername):
        """Returns a list of nodekeys for a given layer"""
        ids = sorted(map(int, self.db.hvals('layers:%s:nodes' % (layername))))
        ret = [self.nodekeyFromID(id, layername) for id in ids]
        return ret

    def addcallback(self, callback, predfunc=None):
        """Adds the given node callback and optionally a predicate function.
        The callback takes (nodekey, nodedict, layer, action), where:
            nodekey - The key to the node
            nodedict - The dict of elements in the node
            layer - The layer the node is in
            action - The action that was performed on this node. One of:
                create - Created for the first time
                update - Update a node that already existed
                init - Called on first init of the current process
        The predicate takes the same args and if False, the callback is not called.
        With no predicate, the callback is always executed.
        """
        self.callbacks.append((callback, predfunc))

    def runcallbacks(self, nodekey, nodedict, layer, action):
        """Runs all applicable callbacks for the given args.
        This does the appropriate checking with the predicate functions.
        """
        for callback, predfunc in self.callbacks:
            if not predfunc(nodekey, nodedict, layer, action): continue
            callback(nodekey, nodedict, layer, action)

    def addLayer(self, name, level, nodekeyfmt=GENERIC_NODE_KEY_FMT):
        """Creates a new layer with given name and level.
        Optionally specify a fmt that generates nodekeys given a dict containing (id, layername).
        Returns the layer name.
        """
        # check if we already added this level
        oldl = self.db.hget('layers', name)
        if oldl is not None:
            # we did, so check the level
            if int(oldl) != level:
                raise ValueError('Layer "%s" already exists at level %s, not %d!' % (name, oldl, level))
            # level matches, so just return the name of the layer
            #print 'Layer "%s" already existed' % (name)
            return name
        key = self.layerkey(name)
        p = self.db.pipeline()
        l = dict(name=name, level=level, key=key, nodekeyfmt=nodekeyfmt)
        p.hmset(key, l)
        p.hset('layers', name, level)
        p.execute()
        #print 'Added layer "%s" at level %s: %s' % (name, level, l)
        return name

    def addNode(self, layername, primary=None, checkexisting=1, pipeline=None, added=None, addindicator=None, **nodekw):
        """Adds a node to the given layer.
        You can optionally pass in a "primary key", which can be used
        to lookup the node id in O(1) time. Else, this becomes the node id.

        If checkexisting is true (default), then it first checks
        to see if a node with that primary exists. If so, it sets the given
        nodekw on that existing node, and doesn't create a new id, etc.

        If added is given, then checkexisting also tries to lookup the
        'primary' value in the given dict, which should return a key.
        If the key was not found, then the primary->key mapping is added to it.

        The actual data to add should be given as key-value pairs in nodekw.
        Note that the values should be serializable to fit in a redis hash.
        Also, the data is ALWAYS set, even if 'checkexisting' was true.

        If pipeline is given, then all db modifications are done within the
        given pipeline.  Else, creates a new pipeline for the duration of this
        function, and then executes it at the end.

        If addindicator is true, then it's assumed to be an array with at least 1 element.
        This element is set to 1 if we actually added the node, else 0.

        Returns the node keyname."""
        # see if this key exists
        key = None
        if checkexisting and primary:
            try:
                primary = str(primary)
            except Exception:
                pass
            if isinstance(added, dict) and primary in added:
                # first check the given 'added' dict, if it exists
                key = added[primary]
            else:
                # now check the database
                key = self.nodekey(primary, layername=layername)
                if not key or not self.db.exists(key):
                    #log('  Key %s did not exist!' % (key,))
                    key = None
        p = pipeline if pipeline else self.db.pipeline()
        if addindicator:
            addindicator[0] = 0
        # if we don't have a key yet, generate one
        action = 'update'
        if not key:
            action = 'create'
            lkey = self.layerkey(layername)
            keyfmt = self.nodekeyfmt(layername)
            id = self.db.incr('counters:layernodes:%s' % (layername))
            key = keyfmt % dict(id=id, layername=layername)
            # some bookkeeping
            p.hincrby(lkey, 'nnodes', 1)
            if not primary:
                primary = id
            try:
                primary = str(primary)
            except Exception: pass
            #log('Actually adding node %s with primary %s to layer %s' % (key, primary, layername))
            p.hset('layers:%(layername)s:nodes' % (locals()), primary, id)
            # add to the 'added' cache of added keys
            if isinstance(added, dict):
                added[primary] = key
            # mark layerkeys dirty for this layer
            self.dirty['layerkeys-%s' % (layername)] = 1
            # mark all edges connected to this layer as dirty
            for l1, _ in self.layers():
                for l2, _ in self.layers():
                    self.dirty['edgematrix-%s-%s' % (l1, l2)] = 1
            # set the indicator
            if addindicator:
                addindicator[0] = 1
            # also delete all flows and edgematrices
            self.deletePickles()
        # set the main kw for this node
        p.hmset(key, nodekw)
        if not pipeline:
            p.execute()
            # run callbacks
            self.runcallbacks(key, nodekw, layername, action)
        return key

    def addScores(self, scorename, layername, hashkey=None, **kw):
        """Adds a zset for the given layername.
        If you give a hashkey, then it initializes this using the given key
        extracted from all nodes. Otherwise, it uses the kwargs to initialize it.
        """
        if hashkey:
            nodes = self.nodes(layername)
            vals = pipefunc(self.db, nodes, 'hget', hashkey)
            for n, v in zip(nodes, vals):
                nodeid = n.rsplit(':', 1)[-1]
                kw[nodeid] = v
        if not kw: return
        key = 'scores:%s:%s' % (layername, scorename)
        p = self.db.pipeline()
        p.zadd(key, **kw)
        p.sadd('layers:%s:scores' % (layername), key)
        p.execute()

    def addEdges(self, edges, pipeline=None):
        """Adds edges, each of which is (nodekey1, nodekey2, weight).
        This does various normalization and then adds relevant entries.
        Returns a list of (edgekey, edgefield) entries.
        Note that if self.symmetric is true, then adds the symmetric entries as well,
        but will still return only as many return pairs as inputs.
        Also, we will filter out any edges that connect a node to itself.
        """
        p = pipeline if pipeline else self.db.pipeline()
        ret = []
        for nk1, nk2, w in edges:
            if not nk1 or not nk2: continue
            (l1, l2), (n1, n2) = zip(*[self.splitNodekey(nk) for nk in [nk1, nk2]])
            if nk1 == nk2: continue
            ekey1, efield1 = ('%s:edges:%s' % (nk1, l2), str(n2))
            ret.append((ekey1, efield1))
            p.zadd(ekey1, **{efield1:w})
            p.hincrby('layeredges:%s' % (l1), l2, 1)
            # mark this edgematrix as dirty
            self.dirty['edgematrix-%s-%s' % (l1, l2)] = 1
            if self.symmetric:
                ekey2, efield2 = ('%s:edges:%s' % (nk2, l1), str(n1))
                p.zadd(ekey2, **{efield2:w})
                p.hincrby('layeredges:%s' % (l2), l1, 1)
                # mark this edgematrix as dirty
                self.dirty['edgematrix-%s-%s' % (l2, l1)] = 1
            #print '  ', nk1, nk2, n1, n2, l1, l2, ekey1, efield1, ekey2, efield2, w
            self.deletePickles()
        if not pipeline:
            p.execute()
        return ret

    def deletePickles(self):
        """Deletes all our pickles"""
        from shutil import rmtree
        rmtree(os.path.join(self.prefix, 'edgematrices'), ignore_errors=1)
        rmtree(os.path.join(self.prefix, 'flows'), ignore_errors=1)


    def deleteEdges(self, layer1, layer2, dosym=1):
        """Deletes edges from layer1 to layer2.
        If self.symmetric and dosym, then also deletes edges the other way."""
        p = self.db.pipeline()
        l1keys = self.nodes(layer1)
        for k in l1keys:
            p.delete('%s:edges:%s' % (k, layer2))
        self.dirty['edgematrix-%s-%s' % (layer1, layer2)] = 1
        p.hdel('layeredges:%s' % (layer1), layer2)
        p.execute()
        if self.symmetric and dosym:
            self.deleteEdges(layer2, layer1, dosym=0) # so that we don't keep iterating forever

    def getEdges(self, nodekeys, valid=None, sort=1):
        """Gets edges from the given nodekeys, optionally limited to some layers.
        Returns a dict mapping layer names to lists of results.
        Each result list has the same length as 'nodekeys', and consists of
        edge lists, which are (nodeid, weight) pairs.

        If 'valid' is a string, then only returns edges that connect to that layer.
        If 'valid' is a list, then only returns edges that connect to one of those layers.
        if 'sort' is true (default), then each list is sorted by highest weight first.

        All input nodekeys must be in the same layer."""
        if not nodekeys: return []
        # basic init and quick checks
        layername, _ = self.splitNodekey(nodekeys[0])
        elayers = self.db.hkeys('layeredges:%s' % (layername))
        if valid:
            if isinstance(valid, basestring): # single valid layer
                elayers = [l for l in elayers if l==valid]
            else: # list of layers
                elayers = [l for l in elayers if l in valid]
        if not elayers: return {}
        ret = {}
        for l in elayers:
            edges = pipefunc(self.db, ['%s:edges:%s' % (nk, l) for nk in nodekeys], 'zrevrangebyscore', withscores=1, min=0.00001, max='inf')
            assert len(edges) == len(nodekeys)
            ret[l] = [[(int(k), float(v)) for k, v in e] for e in edges]
            if sort:
                for lst in ret[l]:
                    lst.sort(key=lambda pair: pair[1], reverse=1)
        return ret

    def summedEdges(self, keys, dstlayer):
        """Returns a summed list of edges out from the given key inputs.
        Essentially one stage of a flow computation, but done without matrices.
        The inputs are either a list of keys (assumed weight=1), a list of (key,score) pairs,
        or a dict of key->weights. The edges to the given `dstlayer` are retrieved,
        summed, and then multiplied by these scores.
        The output is a dict of key->scores.
        """
        from collections import defaultdict
        if not keys: return {}
        if isinstance(keys, dict): # already a dict
            inputs = keys
        else:
            if isinstance(keys[0], basestring): # only keys
                inputs = dict.fromkeys(keys, 1.0)
            else: # (key,score) pairs
                inputs = dict(keys)
        ikeys = sorted(inputs)
        edges = self.getEdges(ikeys, dstlayer, sort=0)[dstlayer]
        ret = defaultdict(float)
        for ikey, curedges in zip(ikeys, edges):
            inscore = inputs[ikey]
            #print '  %s : score %f, %d edges' % (ikey, inscore, len(curedges))
            for nodeid, s in curedges:
                ret[self.nodekeyFromID(nodeid, dstlayer)] += inscore * s
        return dict(ret)

    def recursiveFlow(self, keys, layers):
        """Repeated calls to summedEdges() with initial keys, going through many `layers`.
        The outputs of one call are then fed to the next.
        Returns a dict of key->scores at the last layer."""
        if not keys: return {}
        ret = keys
        for dstlayer in layers:
            ret = self.summedEdges(ret, dstlayer)
        return ret


    def updateLayerKeys(self, layername):
        """Updates the cached layer keys for the given layer"""
        l = layername
        dkey = 'layerkeys-%s' % (l)
        if l not in self.layerkeys:
            self.dirty[dkey] = 1
        if self.dirty.get(dkey, 0):
            #nnodes = self.db.hlen('layers:%s:nodes' % (l)) + 1
            try:
                nnodes = max(map(int, self.db.hvals('layers:%s:nodes' % (l)))) + 1
            except Exception:
                nnodes = 0
            self.layerkeys[l] = [self.nodekeyFromID(id, l) for id in range(nnodes)]
            #log('  Precached %d layerkeys for layer %s' % (len(self.layerkeys[l]), l))
            if dkey in self.dirty:
                del self.dirty[dkey]

    #@timed
    def getEdgeMatrix(self, srclayer, dstlayer, srckeys=None, dstkeys=None, usesparse=1):
        """Computes a matrix of weights that transforms from srclayer to dstlayer.
        i.e., you have a vector V_s of values from srclayer, and this function
        returns M_ds. Then you can do V_d = np.dot(M_ds, V_s).
        Returns (M_ds, list of srclayer keys, list of dstlayer keys).

        You can optionally pass in lists of srckeys and dstkeys.
        If so, then only fills in values that exist in these lists.

        If src and dst layers are the same, then initializes the matrix with identity.
        Otherwise, initializes the matrix with 0s.

        If usesparse is true (default), then uses sparse matrices. Notes:
            - we initialize data using lil_matrix, because it's fastest to modify
            - we convert to csr_matrix at the end, because that's fastest to multiply
        """
        import scipy as sp
        import scipy.sparse as sparse
        from scipy.sparse import lil_matrix as sparsemat
        times = [time.time()]
        # init keys and matrix
        if not srckeys:
            srckeys = self.nodes(srclayer)
        if not dstkeys:
            dstkeys = self.nodes(dstlayer)
        dstrows = dict((int(self.splitNodekey(dk)[1]), i) for i, dk in enumerate(dstkeys))
        times.append(time.time())
        ns, nd = len(srckeys), len(dstkeys)
        assert ns > 0 and nd > 0
        if srclayer == dstlayer:
            if usesparse:
                M = sparsemat((nd,nd))
                M.setdiag(np.ones(nd))
            else:
                M = np.eye(nd)
        else:
            if usesparse:
                M = sparsemat((nd,ns))
            else:
                M = np.zeros((nd, ns))
        times.append(time.time())
        # fill in the matrix, only if we have something to fill
        if self.db.hexists('layeredges:%s' % (srclayer), dstlayer):
            edges = self.getEdges(srckeys, valid=dstlayer, sort=0)[dstlayer]
            for col, row in enumerate(edges):
                for nodeid, w in row:
                    if nodeid not in dstrows: continue
                    row = dstrows[nodeid]
                    M[row, col] = w
        times.append(time.time())
        nz = len(M.nonzero()[0])
        nels = M.shape[0]*M.shape[1]
        if nz == 0:
            M = None
        else:
            if ns == nd:
                # check if it's identity
                if usesparse:
                    eye = sparsemat((nd,nd))
                    eye.setdiag(np.ones(nd))
                else:
                    eye = np.eye(nd)
                eye -= M
                iseye = (len(eye.nonzero()[0]) == 0)
                if iseye:
                    M = None
            else:
                iseye = 0
            log('  Matrix from %s (%d) to %s (%d) had %d/%d nonzeros (%0.5f%%) (iseye=%s)' % (srclayer, len(srckeys), dstlayer, len(dstkeys), nz, nels, nz*100.0/float(nels), iseye))
        log('  Matrix took: %s' % (getTimeDiffs(times)))
        if sparse.issparse(M):
            M = M.tocsr()
        return (M, srckeys, dstkeys)

    def cachedEdgeMatrix(self, l1, l2):
        """Updates the cached edge matrix between the given layers (if needed).
        Assumes associated layerkeys are already up-to-date.
        Returns the matrix."""
        import cPickle as pickle
        #FIXME if things are symmetric, then only compute one half of the symmetries, and generate the others on-the-fly
        dkey = 'edgematrix-%s-%s' % (l1, l2)
        picklename = os.path.join(self.prefix, 'edgematrices', dkey+'.pickle')
        try:
            os.makedirs(os.path.dirname(picklename))
        except OSError:
            pass
        if (l1, l2) not in self.edgematrices:
            self.dirty[dkey] = 1
        if self.dirty.get(dkey, 0): #FIXME the pickles are always assumed to be up-to-date right now!
            try:
                M = pickle.load(open(picklename))
                #log('  Loaded %s of size %s' % (picklename, M.shape if M is not None else 0))
            except Exception, e:
                M, _, _ = self.getEdgeMatrix(l1, l2, self.layerkeys[l1], self.layerkeys[l2])
                try:
                    os.makedirs(os.path.dirname(picklename))
                except Exception: pass
                #pickle.dump(M, open(picklename, 'wb'), -1)
                #log('  Due to exception "%s", saved matrix of shape %s, with pickle size %d to "%s"' % (e, M.shape if M is not None else 0, os.stat(picklename).st_size, picklename))
                self.edgematrices[(l1, l2)] = M #FIXME experiment to not use all this memory
                #log('  Precached edgematrix %s x %s from layer %s to %s' % (M.shape[0], M.shape[1], l1, l2))
                if dkey in self.dirty:
                    del self.dirty[dkey]
        else:
            M = self.edgematrices[(l1, l2)]
        if 0 and l1 != l2 and M is not None: #debugging
            log('Got M of type %s' % (type(M),))
            import array
            scores = array.array('f', [])
            import numpy.random as random
            for i in range(M.shape[1]):
                scores.append(random.random())
            log('Got %d scores: %s' % (len(scores), scores[:5]))
            try:
                t1 = time.time()
                v = M.dot(scores)
                t2 = time.time()
            except Exception:
                log('in exc')
                t1 = time.time()
                v = np.dot(M, scores)
                t2 = time.time()

            M = M.todense()
            t3 = time.time()
            v1 = np.dot(M, scores)
            t4 = time.time()
            log('Got %d in v, in %0.4f secs, compared to %0.4fs for dense: %s, %s, %s' % (len(v), t2-t1, t4-t3, v[:5], v1[:5], v==v1))
            sys.exit()
        return M

    def nodeweights(self, layername, lkeys=None):
        """Returns the nodeweights for the given layer.
        If lkeys is given, then the weights are returned in that order.
        Otherwise, returns weights for all nodes in this layer, as returned by nodes()"""
        if not lkeys:
            lkeys = self.nodes(layername)
        weights = np.ones(len(lkeys))
        key = 'layers:%s:weights' % (layername)
        if not self.db.exists(key): return weights
        ids = [self.splitNodekey(k)[1] for k in lkeys]
        for i, w in enumerate(self.db.hmget(key, ids)):
            if w is None: continue
            weights[i] = float(w)
        #log('For layer %s, got %s' % (layername, zip(lkeys, ids, weights)))
        #log('For layer %s, with %d lkeys, got %d weights: %s' % (layername, len(lkeys), len(weights), weights))
        return weights

    def updateCache(self):
        """Updates our cache"""
        # update our list of layerkeys as needed
        for l, _ in self.layers():
            self.updateLayerKeys(l)

    def createFlow(self, *args, **kw):
        """Creates a flow object.
        If args and/or kw are given, then calls flow.add() with those params.
        Note that for now, we precache layerkeys and edge matrices here."""
        #log('In create flow, got dirty: %s' % (self.dirty,))
        self.updateCache()
        f = RGLFlow(self)
        if args or kw:
            f.add(*args, **kw)
        return f

    def updateIfDirty(self, dkey, func, *args, **kw):
        """Runs the given func if the dirty bit is set for the given key"""
        if dkey in self.dirty:
            func(*args, **kw)
            del self.dirty[dkey]
        else:
            log('Got precached dkey %s' % (dkey))

class RGLFlow(object):
    """A flow object for a given RedisLayeredGraph (RGL)"""
    def __init__(self, g, id=None, tempflow=0, debugfmt='str'):
        """Initialize this flow object from the given graph.
        If an id is given, then tries to load the values from disk.
        If tempflow is true (default false), then save() and load() become no-ops.
        The debugfmt determines how to print out ourself:
            'spark': using spark lines
            'str': using get()
        """
        self.g = g
        self.db = g.db
        self.scores = {}
        self.tempflow = tempflow
        self.debugfmt = debugfmt
        if id:
            # load from disk
            self.id = id
            try:
                self.load()
            except Exception:
                # could not load, so just reset
                self.reset()
        else:
            # create new id and reset
            self.newid()
            self.reset()

    def reset(self, *layernames):
        """Resets the score arrays.
        Optionally, you can give a list of layernames to reset.
        Otherwise, it resets all layers."""
        # update the cached list of layerkeys as needed
        for lname, level in self.g.layers():
            if layernames and lname not in layernames: continue
            nnodes = len(self.g.layerkeys[lname])
            a = self.scores[lname] = np.zeros(nnodes)
            #print 'Reset scores for layer %s (level %d) with %d nodes' % (lname, level, len(a))
        self.save()

    def binop(self, other, binfunc):
        """Base function for binary operators.
        Does all the necessary checks, and then calls the binfunc(v1, v2) to get the output.
        'other' can be either another flow, or a scalar."""
        ret = RGLFlow(g=self.g, debugfmt=self.debugfmt, tempflow=1)
        assert sorted(self.scores) == sorted(ret.scores)
        if isinstance(other, RGLFlow):
            # combine two flows
            assert self.g == other.g
            assert sorted(self.scores) == sorted(other.scores)
            for layer in self.scores:
                s1, s2, out = self.scores[layer], other.scores[layer], ret.scores[layer]
                assert len(s1) == len(s2) == len(out)
                ret.scores[layer] = binfunc(s1, s2)
        elif isinstance(other, (float, long, int)):
            # apply the given scalar to this flow
            for layer in self.scores:
                s, out = self.scores[layer], ret.scores[layer]
                assert len(s) == len(out)
                ret.scores[layer] = binfunc(s, other)
        else:
            raise NotImplementedError('cannot handle type %s for RGLFlow.binop()' % (type(other)))
        return ret


    def __add__(self, other):
        """Adds 'other' rgl flow to this one and returns new RGLFlow"""
        return self.binop(other, binfunc=lambda v1, v2: v1+v2)

    def __radd__(self, other):
        """Addition with flipped order"""
        return self.__add__(other)

    def __iadd__(self, other):
        """Runs the normal __add__, and then resets our variables"""
        temp = self+other
        self.scores = temp.scores
        self.save()
        return self

    def __sub__(self, other):
        """Subtracts 'other' flow from this one and returns the result.
        Note that values are clamped to remain positive."""
        def binfunc(v1, v2):
            c = v1-v2
            c[c < 0] = 0.0
            return c

        return self.binop(other, binfunc=binfunc)

    def __mul__(self, other):
        """Multiplies two flows, or this flow and a scalar"""
        return self.binop(other, binfunc=lambda v1, v2: v1*v2)

    def __rmul__(self, other):
        """Multiplication with flipped order"""
        return self.__mul__(other)

    def __eq__(self, other):
        """Returns true if our layers are the same and the values are (almost) the same."""
        if sorted(self.scores) != sorted(other.scores): return False
        for l in self.scores:
            s1 = self.scores[l]
            s2 = other.scores[l]
            if not np.allclose(s1, s2): return False
        return True

    def newid(self):
        """Changes our id"""
        import uuid
        self.id = uuid.uuid1().hex

    def save(self):
        """Saves ourself to disk"""
        import cPickle as pickle
        #from scipy.sparse import lil_matrix as sparsemat
        from scipy.sparse import csr_matrix as sparsemat
        #M = M.tocsr()
        if self.tempflow: return
        fname = os.path.join(self.g.prefix, 'flows', self.id+'.pickle')
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError: pass
        if 0:
            todump = {}
            for k in self.scores:
                todump[k] = sparsemat(self.scores[k])
        else:
            todump = self.scores
        pickle.dump(todump, open(fname, 'wb'), -1)
        #log('Saving flow with id %s' % (self.id))

    def load(self):
        """Loads ourself from disk. Our id must be set"""
        import cPickle as pickle
        fname = os.path.join(self.g.prefix, 'flows', self.id+'.pickle')
        self.scores = pickle.load(open(fname))
        # check if the lengths of these scores match RLG's list of layerkeys
        for l in self.g.layerkeys:
            alen, blen = len(self.g.layerkeys[l]), len(self.scores.get(l,[]))
            if alen != blen:
                log('### checking l "%s": exists %s, lens %d vs %d' % (l, l in self.scores, alen, blen))
        for l in self.scores:
            if l not in self.g.layerkeys:
                log('### layer "%s" from flow %s does not exist in layerkeys' % (l, self.id))
        pass#TODO HERE
        #data = pickle.load(open(fname))
        #self.scores = dict((k, v.todense()) for k, v in data.iteritems())
        #log('Loading flow with id %s' % (self.id))

    def __str__(self):
        """Returns our values as a string"""
        from StringIO import StringIO
        s = StringIO()
        print >>s, 'Flow %s, tempflow %s, debugfmt %s' % (self.id, self.tempflow, self.debugfmt)
        for lname, level in self.g.layers():
            sc = self.get(lname)
            if self.debugfmt == 'spark' or sc:
                print >>s, '%s (%d nodes, %d nz):' % (lname, len(self.scores[lname]), len(sc))
                s.flush()
            if self.debugfmt == 'spark':
                spark(self.scores[lname], f=s)
            elif self.debugfmt == 'str':
                if sc:
                    print >>s, sc
        return s.getvalue()

    def incr(self, op='add', **scores):
        """Increments scores, given as a mapping of nodekey=increment.
        Increment type depends on 'op':
            'add': add to existing [default]
            'mul': multiply to existing
        """
        for nk, incr in scores.iteritems():
            lname, nodeid = self.g.splitNodekey(nk)
            a = self.scores[lname]
            if int(nodeid) >= len(a): continue
            if op == 'add':
                try:
                    a[int(nodeid)] += incr
                except Exception:
                    log('Got nk %s, incr %s, lname %s, nodeid %s, a len %d, lkeys %d: %s' % (nk, incr, lname, nodeid, len(a), len(self.g.layerkeys[lname]), self.g.layerkeys[lname][-5:]))
                    raise
            elif op == 'mul':
                a[int(nodeid)] *= incr
            else:
                raise NotImplementedError()

    @timed
    #@profile
    def add(self, dir, destlevel, layerfuncs=None, debug=0, **flows):
        """Adds the given flows.
        The scores are calculated going in the given dir (+1 or -1).
        Computation proceeds until the given destination level.
        Each flow should be given as nodekey=score.
        You can optionally pass in a dict of layerfuncs.
        These are applied at the given layer as:
            self.scores[layer] = layerfuncs[layer](self.scores[layer])
        Returns self.
        """
        import numpy as np
        import operator as op
        #if not flows: return
        mylog = lambda s: log(s, funcindent=-1)
        if debug > 0:
            mylog('Adding %d flows in dir %d to destination level %d' % (len(flows), dir, destlevel))
        # basic init
        g = self.g
        db = self.db
        layers = dict((layer, g.getLayer(layer)) for layer, level in g.layers())
        layerOrder = sorted(layers, key=lambda l: layers[l]['level'], reverse=(dir<0))
        if debug > 0:
            mylog('Layers: %s' % (layers,))
            mylog('Layer order: %s' % (layerOrder,))
        # add all todo flows
        if flows:
            self.incr(**flows)
        # start accumulating flows
        for l in layerOrder:
            curlevel = layers[l]['level']
            if dir > 0 and curlevel > destlevel: break
            if dir < 0 and curlevel < destlevel: break
            if debug > 0:
                mylog('Adding flows to layer %s (level %d)' % (l, curlevel))
            lkeys = g.layerkeys[l] # these are guaranteed to line up with our scores array.
            # quick check for non-zero elements
            nz = len(self.scores[l].nonzero()[0])
            if nz == 0: continue
            if 0: #FIXME temporarily disabled
                # get the self-edge matrix
                if debug > 1:
                    mylog('  Running local flows')
                #M, lkeys2, _ = g.getEdgeMatrix(l, l, lkeys, lkeys)
                M = g.cachedEdgeMatrix(l, l)
                #assert lkeys == _ == lkeys2
                #print M, M.shape, len(lkeys), lkeys[:5]
                #FIXME see if we run into any issues due to M being None for identity
                if M is not None:
                    # multiply scores by this matrix
                    try:
                        v = M.dot(self.scores[l])
                    except Exception, e:
                        v = np.dot(M, self.scores[l])
                        mylog('  *****  Hit exception %s: %s vs %s' % (e, M.shape, len(self.scores[l])))
                        mylog('%d Layerkeys: %s, %s' % (len(g.layerkeys[l]), g.layerkeys[l][:3], g.layerkeys[l][-3:]))
                        sys.exit()
                    #assert len(v) == len(lkeys) == M.shape[0] == M.shape[1] == len(self.scores[l])
                    assert len(v) == len(self.scores[l])
                    self.scores[l] = v
                #print len(v), v, v.max(), v.argmax(), v.sum()
                # at this point, multiply by our weights
                self.scores[l] *= g.nodeweights(l, lkeys)
            # now apply the layerfunc, if we have one
            if layerfuncs and l in layerfuncs and layerfuncs[l]:
                self.scores[l] = layerfuncs[l](self.scores[l])
            # another quick check for nonzeros
            nz = len(self.scores[l].nonzero()[0])
            if nz == 0: continue
            # now run flows from this layer to all others in dir
            if debug > 1:
                mylog('  Running neighboring flows')
            for l2 in db.hkeys('layeredges:%s' % (l)):
                if l2 == l: continue
                l2level = layers[l2]['level']
                if dir > 0 and (l2level > destlevel or l2level < curlevel): continue
                if dir < 0 and (l2level < destlevel or l2level > curlevel): continue
                l2keys = g.layerkeys[l2]
                if debug > 2:
                    mylog('    Neighboring flow from %s (%d) to %s (%d), dir %s, destlevel %s' % (l, curlevel, l2, l2level, dir, destlevel))
                # get the edge matrix
                #M, _, _ = g.getEdgeMatrix(l, l2, lkeys, l2keys)
                M = g.cachedEdgeMatrix(l, l2) #TODO most time spent here
                if M is not None:
                    #print M, M.shape, len(l2keys), l2keys[:5]
                    # multiply scores by this matrix to get dst scores
                    try:
                        v = M.dot(self.scores[l])
                    except Exception, e:
                        v = np.dot(M, self.scores[l])
                        log('  **** In exception: %s' % (e,))
                        raise
                    assert len(v) == len(self.scores[l2])
                    # add these scores to existing scores at that level
                    self.scores[l2] += v
                # at this point, multiply by the weights in the 2nd layer
                #self.scores[l2] *= g.nodeweights(l2, l2keys) #FIXME I think this will cause a double-weighting
        self.save()
        return self

    def get(self, layername, thresh=0.0, withscores=1, tokeys=0):
        """Returns (nodeid, score) pairs from the given layer, where score > thresh.
        Results are sorted from high score to low.
        If withscores is true (default) the returns scores as well.
        If tokeys is true, then maps ids to keys.
        """
        a = self.scores[layername]
        #ret = [(i, score) for i, score in enumerate(a) if score > thresh]
        #ret.sort(key=lambda pair: pair[1], reverse=1)
        # only process rest if we have any values above the threshold
        if not np.any(a > thresh): return []
        inds = np.argsort(a)[::-1]
        scores = a[inds]
        scores = scores[scores > thresh]
        if tokeys:
            inds = [self.g.nodekeyFromID(id, layername) for id in inds]
        if withscores:
            ret = zip(inds, scores)
        else:
            ret = inds[:len(scores)]
        return ret

    def outliers(self, layers=None):
        """Returns the "outliers" amongst the given layers (or all if None)."""
        if not layers:
            layers = self.scores.keys()
        ret = []
        for l in layers:
            scores = [(self.g.nodekeyFromID(id, l), score) for id, score in self.get(l)]
            if not scores: continue
            nk, s = scores[0]
            oscore = 1.0 if len(scores) == 1 else 1.0-(scores[1][1]/s)
            if oscore == 0: continue
            #print l, scores, nk, s, oscore
            ret.append((nk, oscore, s))
        ret.sort(key=lambda r: (r[1], r[2]), reverse=1)
        return ret

    @classmethod
    def combine(cls, tocomb, op='add', **kw):
        """Combines a list of (factor, flow) flows into one:
            ret = op(factor*flow for factor, flow in tocomb)
        Where 'op' is one of:
            'add': adds flows
            'mul': multiplies flows
        If you given any other keywords, they are used in the initialization.
        Returns None on error.
        """
        if not tocomb: return None
        first = tocomb[0][1]
        defkw = dict(tempflow=1, debugfmt=first.debugfmt)
        defkw.update(kw)
        f = RGLFlow(first.g, **defkw)
        for layer, a in f.scores.iteritems():
            if op == 'mul': # re-initialize the layer to be ones if we're multiplying
                a += 1.0
            for fac, flow in tocomb:
                if op == 'add':
                    a += fac*flow.scores[layer]
                elif op == 'mul':
                    a *= fac*flow.scores[layer]
            # clamp back to positive
            a[a < 0] = 0.0
        f.save()
        return f
