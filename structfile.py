"""A way to read and write structs to a binary file, with fast access

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

import os, sys, time
import struct

class StructFile(object):
    """A file which contains structs"""
    def __init__(self, structfmt, fname):
        """Initializes a structfile using the given structfmt and fname.
        The file is opened in the given mode ('rb' as default)."""
        self.struct = struct.Struct(structfmt)
        self.size = self.struct.size
        self.fname = fname
        if not os.path.exists(fname):
            open(fname, 'wb').close()
        self.readptr = open(fname, 'rb')
        try:
            self.writeptr = open(fname, 'r+b')
        except IOError:
            self.writeptr = None

    def __len__(self):
        """Returns the number of structs in this file"""
        f = self.readptr
        f.seek(0, os.SEEK_END)
        n = f.tell()
        return n/self.size

    def __iter__(self):
        """Iterates over structs in this file, from the beginning"""
        f = open(self.fname, 'rb')
        while 1:
            try:
                yield self.struct.unpack(f.read(self.size))
            except EOFError:
                break

    def __getitem__(self, i):
        """Returns the i'th struct.
        Negative indices work as well.
        Raised IndexError on invalid index.
        """
        l = len(self)
        if i < 0:
            i += l
        if i >= l: raise IndexError
        f = self.readptr
        f.seek(self.size*i)
        return self.struct.unpack(f.read(self.size))

    def __setitem__(self, i, val):
        """Sets the i'th struct. The file must already have this many structs.
        Negative indices work as well.
        Raised IndexError on invalid index.
        Raises IOError if the file doesn't have write permissions.
        """
        l = len(self)
        if i < 0:
            i += l
        if i >= l: raise IndexError
        f = self.writeptr
        if not f: raise IOError
        f.seek(self.size*i)
        f.write(self.struct.pack(*val))

    def flush(self):
        """Flushes the file if any changes have been made.
        Raises IOError if the file doesn't have write permissions.
        """
        if not self.writeptr: raise IOError
        self.writeptr.flush()

    def append(self, val):
        """Adds the given value to the end of the file.
        Raises IOError if the file doesn't have write permissions.
        """
        f = self.writeptr
        if not f: raise IOError
        f.seek(0, os.SEEK_END)
        f.write(self.struct.pack(*val))
