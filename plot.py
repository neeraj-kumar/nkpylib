import pylab
import sys, os
from math import *
import matplotlib

COLORS = 'rbgcmyk'
MARKERS = 'oxvs+D1'
LINES = ['-', '--', '-.', ':']

"""Args: (defaults at end of each line)
         bg: background color - 'white'
    figsize: figure size - (10,10)
   plotstrs: array of plot strings - ['%s-o' % c for c in COLORS][:ndims]
    aligned: aligned data or not- 1
     update: whether to update on the initial draw - 0
  legendloc: location of legend - 'best'
     labels: array of plot labels - ['Col %d' % i for i in range(1, len(cols)+1)]
       func: plotting function - 'plot'
       name: name of the graph - None
      title: title of graph - 'Plot of %d-D Data' % (ndims)
          x: x-axis label - 'x'
          y: y-axis label - 'y'
    outfile: name of output file - None
"""
def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"""
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

def barGraph(data, **kw):
    """Draws a bar graph for the given data"""
    from pylab import bar
    kw.setdefault('barw', 0.5)
    kw.setdefault('log', 0)
    kw.setdefault('color', 'blue')
    xs = [i+1 for i, row in enumerate(data)]
    names, ys = zip(*data)
    names = [n.replace('_', '\n') for n in names]
    #print 'got xs %s, ys %s, names %s' % (xs, ys, names)
    bar(xs, ys, width=kw['barw'], color=kw['color'], align='center', log=kw['log'])
    ax = pylab.gca()
    def f(x, pos=None):
        n = int(x) - 1
        if n+1 != x: return ''
        if n < 0: return ''
        try:
            return names[n]
        except IndexError: return ''

    ax.xaxis.set_major_formatter(pylab.FuncFormatter(f))
    ax.xaxis.set_major_locator(pylab.MultipleLocator(1))
    ax.set_xlim(0.5, len(names)+0.5)
    for l in ax.get_xticklabels():
        pylab.setp(l, rotation=90)
    start = 0.08, 0.18
    pos = (start[0], start[1], 0.99-start[0], 0.95-start[1])
    ax.set_position(pos)

def plotfunc(data, kw={}, **morekw):
    """Plots the given data, using the given plotstrs"""
    if data is None or len(data) == 0: return
    kw.update(morekw)
    bgcolor = kw.get('bg', 'white')
    fontcolor = 'white' if bgcolor in ('transparent', 'black') else 'black'
    frameon = 0 if bgcolor == 'transparent' else 1
    fig = pylab.figure(30241, figsize=kw.get('figsize', (10, 10)), frameon=frameon)
    if 'update' not in kw:
        pylab.clf()
    try:
        ndims = len(data[0])
    except (TypeError, KeyError):
        ndims = 1
    if isinstance(data, dict):
        # do a bar graph instead
        keys = map(str, sorted(data))
        vals = [data[k] for k in sorted(data)]
        barGraph(zip(keys, vals), **kw)
    elif isinstance(data[0][0], str):
        # do a bar graph instead
        barGraph(data, **kw)
    else:
        plotstrs = kw.get('plotstrs', ['%s-o' % c for c in COLORS][:ndims])
        func = eval('pylab.%s' % kw.get('func', 'plot'))
        aligned = kw.get('aligned', 1)
        if 'update' not in kw:
            pylab.clf()
        if ndims == 1: # if it's 1-d, assume we just want to plot vs indices
            data = [d[0] for d in data]
            func(range(len(data)), data, plotstrs[0])
        elif ndims == 2 and aligned: # if it's 2-d, assume we're given x and y in each row
            xs, ys = zip(*data)
            func(xs, ys, plotstrs[0])
        else: # otherwise assume it's multiple sets of 2d data
            pylab.hold(1)
            if aligned:
                # the data is aligned, with first col=xs, and rest are individual ys
                xs = [d[0] for d in data]
                labels = kw.get('labels', ['A. Col %d' % i for i in range(1, ndims)])
                if type(labels) == type('abcd'): labels = [labels]
                for i in range(1, ndims):
                    ys = [d[i] for d in data]
                    func(xs, ys, plotstrs[(i-1)%len(plotstrs)], label=labels[(i-1)%len(labels)])
            else:
                # unaligned, data is list of list of pairs
                cols = data
                labels = kw.get('labels', ['Col %d' % i for i in range(1, len(cols)+1)])
                if type(labels) == type('abcd'): labels = [labels]
                for i, col in enumerate(cols):
                    xs, ys = zip(*col)
                    func(xs, ys, plotstrs[i%len(plotstrs)], label=labels[i%len(labels)])
            leg = pylab.legend(loc=kw.get('legendloc', 'best'))
            if bgcolor == 'transparent':
                leg.get_frame().set_facecolor('None')
            for t in leg.get_texts():
                t.set_color(fontcolor)
    for name, func in zip('xmin xmax ymin ymax'.split(), [pylab.xlim, pylab.xlim, pylab.ylim, pylab.ylim]):
        if name in kw:
            cur = {name: kw[name]}
            func(**cur)
    pylab.title(kw.get('title', 'Plot of %d-D Data' % (ndims)), color=fontcolor)
    pylab.xlabel(kw.get('x', 'x'), color=fontcolor)
    pylab.ylabel(kw.get('y', 'y'), color=fontcolor)
    pylab.xticks(color=fontcolor)
    pylab.yticks(color=fontcolor)
    ax = pylab.gca()
    if 0:
        ax.get_frame().set_edgecolor(fontcolor)
    if bgcolor == 'transparent':
        ax.axesPatch.set_facecolor('None')
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color(fontcolor)
    return pylab.gcf()

def getKWArgsFromArgs(args):
    """Returns a dictionary of values extracted from strings"""
    kw = {}
    for a in args:
        k, v = a.split('=', 1)
        try:
            kw[k] = eval(v)
        except (NameError, SyntaxError): kw[k] = v
    return kw

def specialize(v):
    """Takes a value and sees if it can be cast to an int or a float"""
    try:
        # see if it's an int...
        v = int(v)
    except ValueError:
        # maybe it's a float...
        try:
            v = float(v)
        except ValueError: pass
    return v

def readlines(f):
    data = [map(specialize, l.strip().split()) for l in f if not l.strip().startswith('#')]
    return data

if __name__ == "__main__":
    data = readlines(sys.stdin)
    kw = getKWArgsFromArgs(sys.argv[1:])
    p = plotfunc(data, kw)
    outfile = kw.get('outfile', None)
    if outfile:
        pylab.savefig(outfile)
    else:
        pylab.show()
