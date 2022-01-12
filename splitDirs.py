import os, sys
from dircache import listdir
from subprocess import *
from itertools import *

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

MAX_FILES_PER_CMD = 200

def splitIntoDirs(dirfmt, num, files):
    """Splits the given set of files into directories.
    dirfmt defines the naming scheme of the directories (it's used as dirfmt % i), and num
    is the max number per directory"""
    cur = 0
    total = len(files)
    tomove = {}
    # first figure out how many files to move into each directory
    while total > 0:
        # get the number of files in this directory
        curdir = dirfmt % cur
        try:
            curnum = len(listdir(curdir))
        except OSError:
            # if it didn't exist, create it
            call(['mkdir', '-p', curdir])
            curnum = 0
        dif = min(num-curnum, total)
        if dif > 0:
            tomove[curdir] = dif
            total -= dif
        cur += 1
    # now actually do the moves
    i = 0
    total = len(files)
    #print tomove
    for d in sorted(tomove.keys()):
        curfiles = files[i:i+tomove[d]]
        set = ' '.join(curfiles)
        #print "Set was %s" % set
        groups = grouper(MAX_FILES_PER_CMD, curfiles)
        for g in groups:
            set = ' '.join([f for f in g if f])
            cmd = 'mv %s %s/' % (set, d)
            os.system(cmd)
        i += tomove[d]
        print "Moved %d files out of %d.\r" % (i, total) ,


if __name__ == "__main__":
    from fnmatch import filter
    if len(sys.argv) < 4:
        print "Usage: python %s <dirname fmt> <num per dir> <filename glob>" % sys.argv[0]
        sys.exit()
    files = filter(listdir('.'), sys.argv[3])
    splitIntoDirs(sys.argv[1], int(sys.argv[2]), files)
