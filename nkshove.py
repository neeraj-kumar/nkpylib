"""A replacement for the PYPI shove module, which unfortunately seems not to be thread safe.

Licensed under the 3-clause BSD License:

Copyright (c) 2011, Neeraj Kumar (neerajkumar.org)
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

class SQLiteStore(object):
    """Sqlite-backing for Shove"""
    def __init__(self, uri):
        """Creates a sqlite store"""
        import sqlite3
        self.uri = uri
        self.c = sqlite3.connect(uri).cursor()
        self.c.execute('create table if not exists store ("key" varchar(1024) not null, value blob not null, primary key ("key"))')

    def execute(self, q, arg=None, **kw):
        """Executes a query and returns a list, retrying on error upto retries times.
        Finally raises an operational error if not successful"""
        import sqlite3
        for i in range(kw.get('retries', 10)):
            try:
                if arg:
                    ret = list(self.c.execute(q, arg))
                else:
                    ret = list(self.c.execute(q))
                return ret
            except sqlite3.OperationalError: pass
        raise sqlite3.OperationalError('Could not execute query %s' % (q,))

    def selectone(self, q, *args, **kw):
        """Executes a query and returns the first element of the first row"""
        ret = self.execute(q, *args, **kw)
        return ret[0][0]

    def commit(self, retries=10):
        """Commits, with given number of retries"""
        import sqlite3
        for i in range(retries):
            try:
                self.c.connection.commit()
            except sqlite3.OperationalError: pass

    def __len__(self):
        ret = self.selectone('select count(*) from store')
        return ret

    def __getitem__(self, k):
        import cPickle as pickle
        try:
            ret = str(self.selectone('select value from store where key=? limit 1', (k,)))
            #print 'got ret %s for k %s' % (len(ret), k)
            return pickle.loads(ret)
        except IndexError: raise KeyError('%s not found in sqlite' % (k,))

    def __setitem__(self, k, v):
        import cPickle as pickle
        import sqlite3
        v = sqlite3.Binary(pickle.dumps(v, protocol=1))
        self.execute('insert or replace into store (key, value) values (?, ?)', (k, v))
        self.commit()

    def __delitem__(self, k):
        try:
            ret = self.selectone('select key from store where key=?', (k,))
        except IndexError: raise KeyError('%s not found in sqlite' % (k,))
        self.execute('delete from store where key=?', (k,))
        self.commit()
    
    def __contains__(self, k):
        try:
            ret = self.selectone('select key from store where key=?', (k,))
            return true
        except IndexError: return false
    
    def clear(self):
        self.execute('delete from store')
        self.commit()

    def get(self, k, default=None):
        try:
            ret = self[k]
        except KeyError: return default

    def setdefault(self, k, default=None):
        try:
            ret = self[k]
        except KeyError: 
            ret = self[k] = default
        return ret

    def __iter__(self):
        keys = self.execute('select key from store')
        for k in keys:
            yield k

    def iterkeys(self):
        return iter(self)

    def keys(self):
        return list(self.iterkeys())

    def itervalues(self):
        for k in self:
            yield self[k]

    def values(self):
        return list(self.itervalues())
        
    def iteritems(self):
        for k in self:
            yield (k, self[k])

    def items(self):
        return list(self.iteritems())

    def pop(self, k, default='no one will ever pick this'):
        try:
            v = self[k]
            del self[k]
            return v
        except KeyError:
            if default == 'no one will ever pick this':
                raise
            return default

    def popitem(self):
        for k, v in self.iteritems():
            del self[k]
            return v
        raise KeyError

    def update(self, other, **kw):
        for k, v in other.iteritems():
            self[k] = v
        if kw:
            for k, v in kw.iteritems():
                self[k] = v


class Shove(object):
    """A storage-backed mapping object"""
    def __init__(self, storeuri='sqlite://:memory:', cacheuri=None):
        """Initializes this shove with the given storage uri and cache uri.
        Cache is ignored right now"""
        import sqlite3
        self.storeuri, self.cacheuri = storeuri, cacheuri
        if cacheuri: raise NotImplementedError('MyShove does not support cache uris right now')
        assert storeuri.startswith('sqlite://')
        if storeuri.startswith('sqlite://'):
            uri = storeuri.replace('sqlite://', '')
            if uri.startswith('/'):
                uri = uri[1:]
            # try a few times to connect
            self.store = {}
            for i in range(10):
                try:
                    self.store = SQLiteStore(uri)
                    break
                except sqlite3.OperationalError: pass
        #print 'store is %s' % (self.store,)

    def __getattr__(self, k):
        if k in 'storeuri cacheuri store'.split():
            return self.__dict__[k]
        return getattr(self.store, k)

    def __setattr__(self, k, v):
        if k in 'storeuri cacheuri store'.split():
            self.__dict__[k] = v
            return
        setattr(self.store, k, v)

    def __getitem__(self, k):
        return self.store[k]

    def __setitem__(self, k, v):
        self.store[k] = v

