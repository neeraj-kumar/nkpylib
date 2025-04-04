#!/usr/bin/env python
"""A simple script to run a process and restart it if it dies for any reason.

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
from subprocess import Popen, PIPE

class ProcMon(object):
    def __init__(self, args):
        """Creates a procmon with the given arguments.
        Does not run the program. Call mainloop() for that"""
        self.args = args
        self.proc = None
        self.nruns = 0

    def restartProc(self):
        """Restarts the program"""
        if self.proc: # try killing it explicitly
            try:
                os.system('kill -9 %d' % (self.proc.pid))
            except Exception: pass
        del self.proc
        try:
            self.proc = Popen(self.args)
        except OSError:
            print 'Invalid program: %s' % (' '.join(self.args))
            sys.exit()
        self.nruns += 1

    def mainloop(self):
        """Runs the process repeatedly"""
        while 1:
            self.restartProc()
            print 'polling now'
            self.proc.communicate()
            print 'done polling, returncode is %s' % (self.proc.returncode)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: %s <program> [<arg 1> ...]' % (sys.argv[0])
        sys.exit()
    args = sys.argv[1:]
    pm = ProcMon(args)
    pm.mainloop()
