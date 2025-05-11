"""Geometry-related utilities."""

from PIL import Image, ImageDraw, ImageChops

from nkpylib.utils import lerp, timed, uniqueize, lpdist
from nkpylib.imageutils import combineImages, createRadialMask # type: ignore

## GEOMETRY UTILS
# All triangle functions take (x,y) pairs as inputs for points
def getDistance(pt1, pt2):
    """Returns euclidean distance between two points"""
    return lpdist(pt1, pt2, 2)

def ptLineDist(pt, line):
    """Returns distance between `pt` ``(x,y)`` to `line` ``((x0,y0), (x1,y1))``, and the closest point on the line.

    Adapted from http://paulbourke.net/geometry/pointlineplane/

    Example::
        >>> ptLineDist((0.5, 1.0), [(0,0), (1, 0)])
        (1.0, (0.5, 0.0))
        >>> ptLineDist((0.0, 0.0), [(0,0), (1, 0)])
        (0.0, (0.0, 0.0))
        >>> ptLineDist((1.0, 0.0), [(0,0), (1, 1)])
        (0.70710678118654757, (0.5, 0.5))
        >>> ptLineDist((-5, 0.0), [(0,0), (1, 0)])
        (5.0, (0.0, 0.0))
    """
    x, y = pt
    (x0, y0), (x1, y1) = line
    dx, dy = x1-x0, y1-y0
    t = ((x-x0)*dx + (y-y0)*dy)/(dx**2 + dy**2)
    t = clamp(t, 0.0, 1.0)
    intersection = intx, inty = (x0+t*dx, y0+t*dy)
    d = getDistance(pt, intersection)
    return (d, intersection)

def distAlong(d, pt1, pt2):
    """Returns the coordinate going distance `d` from `pt1` to `pt2`.
    Works for any dimensionalities.
    """
    dist = getDistance(pt1, pt2)
    ret = [(d/dist * (pt2[dim]-pt1[dim])) + pt1[dim] for dim in range(len(pt1))]
    return ret

def expandBox(box, facs):
    """Expands a `box` about its center by the factors ``(x-factor, y-factor)``.
    The box is given as ``(x0, y0, x1, y1)``"""
    w, h = box[2]-box[0], box[3]-box[1]
    cen = cx, cy = (box[2]+box[0])/2.0, (box[1]+box[3])/2.0
    nw2 = w*facs[0]/2.0
    nh2 = h*facs[1]/2.0
    box = [cx-nw2, cy-nh2, cx+nw2, cy+nh2]
    return box

def rectarea(r, incborder=1):
    """Returns the area of the given ``(x0, y0, x1, y1)`` rect.
    If `incborder` is true (default) then includes that in calc. Otherwise doesn't.
    If either width or height is not positive, returns 0."""
    w = r[2]-r[0] + incborder
    h = r[3]-r[1] + incborder
    if w <= 0 or h <= 0: return 0
    return w * h

def rectcenter(rect, cast=float):
    """Returns the center ``[x,y]`` of the given `rect`.
    Applies the given `cast` function to each coordinate."""
    return [cast((rect[0]+rect[2]-1)/2.0), cast((rect[1]+rect[3]-1)/2.0)]

def rectintersection(r1, r2):
    """Returns the rect corresponding to the intersection between two rects.
    Returns `None` if non-overlapping.
    """
    if r1[0] > r2[2] or r1[2] < r2[0] or r1[1] > r2[3] or r1[3] < r2[1]: return None
    ret = [max(r1[0], r2[0]), max(r1[1], r2[1]), min(r1[2], r2[2]), min(r1[3], r2[3])]
    return ret

def rectoverlap(r1, r2, meth='min'):
    """Returns how much the two rects overlap, using different criteria:

        - 'min': ``intersection/min(a1, a2)``
        - 'max': ``intersection/max(a1, a2)``
    """
    a1 = rectarea(r1)
    a2 = rectarea(r2)
    i = rectintersection(r1, r2)
    if not i: return 0
    ai = float(rectarea(i))
    if meth == 'min':
        return ai/min(a1, a2)
    if meth == 'max':
        return ai/max(a1, a2)

def rectAt(cen, size):
    """Returns a rectangle of the given `size` centered at the given location.
    The coordinates are inclusive of borders."""
    x, y = cen[:2]
    w, h = size[:2]
    return [x-w//2, y-h//2, x-w//2+w-1, y-h//2+h-1]

def trilengths(pt1, pt2, pt3):
    """Returns the lengths of the sides opposite each corner"""
    d1 = getDistance(pt2, pt3)
    d2 = getDistance(pt1, pt3)
    d3 = getDistance(pt1, pt2)
    ret = [d1, d2, d3]
    return ret

def triarea(pt1, pt2, pt3):
    """Returns the area of the triangle.
    Uses `Heron's formula <http://en.wikipedia.org/wiki/Heron%27s_formula>`_
    """
    a, b, c = trilengths(pt1, pt2, pt3)
    s = (a+b+c)/2.0
    return math.sqrt(s*(s-a)*(s-b)*(s-c))

def getTriAngles(pt1, pt2, pt3):
    """Returns the angles (in rads) of each corner"""
    lens = l1, l2, l3 = trilengths(pt1, pt2, pt3)
    a1 = acos((l2**2 + l3**2 - l1**2)/(2 * l2 * l3))
    a2 = acos((l1**2 + l3**2 - l2**2)/(2 * l1 * l3))
    a3 = acos((l1**2 + l2**2 - l3**2)/(2 * l1 * l2))
    angles = [a1, a2, a3]
    return angles

def trialtitude(pt1, pt2, pt3):
    """Returns the coordinates of the other end of the altitude starting at `p1`."""
    lens = l1, l2, l3 = trilengths(pt1, pt2, pt3)
    angles = a1, a2, a3 = getTriAngles(pt1, pt2, pt3)
    dfrom2 = cos(a2)*l3
    return distAlong(dfrom2, pt2, pt3)

def haversinedist(loc1, loc2):
    """Returns the haversine great circle distance (in meters) between two locations.
    The input locations must be given as ``(lat, long)`` pairs (decimal values).

    See http://en.wikipedia.org/wiki/Haversine_formula
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    R = 6378100.0 # mean radius of earth, in meters
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    sdlat2 = sin(dlat/2)
    sdlon2 = sin(dlon/2)
    a = sdlat2*sdlat2 + cos(radians(lat1))*cos(radians(lat2))*sdlon2*sdlon2
    d = R * 2 * atan2(sqrt(a), sqrt(1-a))
    return d

def polyarea(poly):
    """Returns the signed area of the given polygon.
    The polygon is given as a list of ``(x, y)`` pairs.
    Counter-clockwise polys have positive area, and vice-versa.
    """
    area = 0.0
    p = poly[:]
    # close the polygon
    if p[0] != p[-1]:
        p.append(p[0])
    for (x1, y1), (x2, y2) in zip(p, p[1:]):
        area += x1*y2 - y1*x2
    area /= 2.0
    return area

def pointInPolygon(pt, poly, bbox=None):
    """Returns `True` if the point is inside the polygon.
    If `bbox` is passed in (as ``(x0,y0,x1,y1)``), that's used for a quick check first.
    Main code adapted from http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
    """
    x, y = pt
    if bbox:
        x0, y0, x1, y1 = bbox
        if not (x0 <= x <= x1) or not (y0 <= y <= y1): return 0
    c = 0
    i = 0
    nvert = len(poly)
    j = nvert-1
    while i < nvert:
        if (((poly[i][1]>y) != (poly[j][1]>y)) and (x < (poly[j][0]-poly[i][0]) * (y-poly[i][1]) / (poly[j][1]-poly[i][1]) + poly[i][0])):
            c = not c
        j = i
        i += 1
    return c

def pointPolygonDist(pt, poly, bbox=None):
    """Returns the distance from a given point to a polygon, and the closest point.
    If the point is inside the polygon, returns a distance of 0.0, and the point itself.
    The point should be ``(x,y)``, and the poly should be a series of ``(x,y)`` pairs.
    You can optionally pass-in a bounding box ``[x0,y0,x1,y1]`` to run a quick check first.
    (If you don't, it's computed and checked.)

    Returns ``(distance, (x,y))`` of the closest point on the polygon (if outside), else `pt` itself.
    If the polygon is degenerate, then returns ``(0.0, pt)``

    .. note::
        This is not the most efficient function (linear in number of edges of the `poly`).
    """
    if not bbox:
        xs, ys = zip(*poly)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    x, y = pt
    inside = pointInPolygon(pt, poly, bbox=bbox)
    if inside: return (0.0, pt)
    # else, it's outside, so compute distance
    lines = zip(poly, poly[1:]+[poly[0]])
    lines = [(p1, p2) for p1, p2 in lines if p1 != p2]
    dists = [ptLineDist(pt, l) for l in lines]
    if not dists: return (0.0, pt)
    return min(dists)

def distInMeters(dist):
    """Converts distances to a numeric distance in meters.
    If the input is a string, then it can have the following suffixes:
        - 'm': meters
        - 'meter': meters
        - 'meters': meters
        - 'metre': meters
        - 'metres': meters
        - 'km': kilometers
        - 'kilometer': kilometers
        - 'kilometers': kilometers
        - 'kilometre': kilometers
        - 'kilometres': kilometers
        - 'mi': miles
        - 'mile': miles
        - 'miles': miles
        - 'ft': feet
        - 'feet': feet
        - 'foot': feet

    Assumes the string is in the form of a number, optional spaces (of any sort), then the suffix.
    Else, assumes it's numeric and returns it as is.
    """
    if not isinstance(dist, basestring): return dist
    # else, it's a string, so map it
    mPerMile = 1609.34
    mPerFoot = 0.3048
    UNITS = dict(m=1.0, meter=1.0, meters=1.0, metre=1.0, metres=1.0,
        km=1000.0, kilometer=1000.0, kilometers=1000.0, kilometre=1000.0, kilometres=1000.0,
        mi=mPerMile, mile=mPerMile, miles=mPerMile,
        ft=mPerFoot, feet=mPerFoot, foot=mPerFoot,
    )
    # has units, so parse
    match = re.match(r'([-+]?\d*\.\d+|\d+)\s*([a-zA-Z]*)', dist.lower().strip())
    val, unit = match.group(1, 2)
    val = float(val)*UNITS[unit]
    return val

def boxAroundGPS(loc, dist):
    """Returns a bounding box around the given GPS location, within the given distance.
    The location is ``(latitude, longitude)`` and the distance is either a
    single value, or a pair of values ``(lat_dist, lon_dist)``.
    These can be floats (i.e., degrees), or strings, which are assumed to be
    degrees if there is no suffix, or mapped to meters using
    :func:`distInMeters()` if there is a suffix.

    .. note::
        If you give no units, then the returned bbox will be symmetrical in
        degrees around the center, but this is NOT symmetrical in terms of
        distance, since longitudinal distance varies with latitude.

    In contrast, giving units should give symmetric (in terms of distance) bounds.

    For reference:
        - 1 degree latitude = 111.319 km = 69.170 miles.
        - 1 degree longitude = 69.170 miles * cos(`lat`)

    Returns ``[lat0, lon0, lat1, lon1]``
    """
    assert len(loc) == 2
    try:
        xdist, ydist = dist
    except (ValueError, TypeError):
        xdist = ydist = dist
    ret = []
    mPerDeg = 111318.845 # meters per degree
    for i, (cen, d) in enumerate(zip(loc, [xdist, ydist])):
        try:
            d = float(d)
            # no units -- is degrees
            # easy to calculate ret
            ret.extend([cen-d, cen+d])
        except ValueError:
            # has units, so parse
            val = distInMeters(d)/mPerDeg
            #print 'd %s: Val %s, unit %s' % (d.lower().strip(), val, unit)
            if i == 0:
                # latitude just needs equal increments
                ret.extend([cen-val, cen+val])
            else:
                # longitude needs special computation
                minlat, maxlat = ret # get min and max latitudes
                minlon = val/math.cos(math.radians(minlat))
                maxlon = val/math.cos(math.radians(maxlat))
                #print minlat, maxlat, minlon, maxlon
                ret.extend([cen-minlon, cen+maxlon])
    # permute into right order
    ret = [ret[0], ret[2], ret[1], ret[3]]
    return ret

def getBoxProjection(loc, dist, imsize):
    """Creates a box around the given location and projects points to it.
    The loc is (latitude, longitude).
    The dist is a string that is interpretable by boxAroundGPS().
    The imsize is the size of the images created.

    Returns (project, polyim), which are both functions:
        project(loc): takes a (lat, lon) pair and returns an image (x,y) pair.
        polyim(coords): takes a project()-ed set of coordinates and returns a
                        1-channel image with the polygon drawn in it.
    """
    lat, lon = loc
    box = boxAroundGPS(loc, dist)
    w, h = imsize
    lon2x = lambda lon: int(lerp(lon, (box[1], 0), (box[3], w)))
    lat2y = lambda lat: int(lerp(lat, (box[0], 0), (box[2], h)))
    project = lambda loc: (lon2x(loc[1]), lat2y(loc[0]))
    def polyim(coords):
        """Returns a single channel image for this polygon (already projected)"""
        im = Image.new('L', (w, h), 0)
        if coords:
            draw = ImageDraw.Draw(im)
            draw.polygon(coords, outline=255, fill=255)
        return im

    return (project, polyim)

def createNearMask(imsize):
    """Cached and memoized "near" mask generation.
    This is simply a wrapper on createRadialMask().
    Note that we invert the mask, so that later on we can simply paste(),
    rather than have to composite() with a black image.
    """
    fname = 'mask-%d-%d.png' % (imsize[0], imsize[1])
    try:
        return Image.open(fname)
    except Exception:
        mask = createRadialMask(imsize)
        mask = ImageChops.invert(mask)
        mask.save(fname)
    return mask

def projectAndGetExtrema(p, project, polyim, fname=None, mask=None):
    """Takes a polygon and projects it and gets extrema.
    Uses project() to project the coordinates,
    polyim() to get the polygon image.
    If mask is given, then composites the image with the mask.
    If fname is given, then saves the (possibly composited) image to that name.
    Finally, computes the extrema.
    Returns (max value, polygon image, projected coordinates).
    """
    coords = map(project, p)
    pim = polyim(coords)
    if mask:
        pim.paste(0, (0,0), mask)
    if fname:
        pass #pim.save(fname) #FIXME this takes too long...
    m, M = pim.getextrema()
    return (M, pim, coords)

def locateGPS(loc, objs, imsize=(1000,1000), indist='50 meters', neardist='1 km', imdir=None):
    """Figures out what objects this location is "in" and "near".
    'loc' is a (latitude, longitude) pair.
    'objs' is a list of (objkey, polygon) tuples.
    For both "in" and "near", projects a box around the given location to an image.
    This image has size 'imsize'. Also projects all given object polygons to this image.

    For "in", checks for any objects that intersect a box within distance
    "indist" from the given location.

    For "near", computes distance from loc to any objects within 'neardist'
    (that were not 'in').

    Returns (objsin, objsnear), where each is a sorted list of (objkey, score) pairs.
    For "in", the score is 1.0. [Should it be (area of intersection)/(area of obj)?]
    The objects are sorted from least area to greatest area.
    For "near", the score is minimum distance between location and obj
    boundaries as a fraction of 'indist', squared to get a faster fall-off.

    If imdir is given, then saves debugging images within that directory.
    """
    #TODO check if done?
    #log('Trying to locate %s with %d objs, imsize %s, dists %s, %s, imdir %s: %s' % (loc, len(objs), imsize, indist, neardist, imdir, objs[:2]))
    # init
    # create imdir if needed
    if imdir:
        try:
            os.makedirs(imdir)
        except OSError:
            pass
    # setup projection for "in" and run on all objects
    project, polyim = getBoxProjection(loc, indist, imsize)
    objsin = []
    for objkey, p in objs:
        fname = os.path.join(imdir, 'in-%s.png' % (objkey.rsplit(':', 1)[-1])) if imdir else ''
        M, pim, coords = projectAndGetExtrema(p, project, polyim, fname=fname)
        if M == 0: continue # ignore things that don't match at all
        objsin.append([objkey, abs(polyarea(coords)), pim])
    # sort "in" objects by area
    objsin.sort(key=lambda o: o[1])
    if imdir:
        comb = combineImages([o[2] for o in objsin])
        if comb:
            comb.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(imdir, 'in-poly.png'))
    # remap to get scores instead of areas and pims
    objsin = [(o[0], 1.0) for o in objsin]
    log('    Got %d objects "in": %s' % (len(objsin), objsin[:5]))
    # now do "near"
    project, polyim = getBoxProjection(loc, neardist, imsize)
    mask = createNearMask(imsize)
    doneobjs = set([o for o, s in objsin])
    objsnear = []
    for objkey, p in objs:
        if objkey in doneobjs: continue # skip objects we're in
        fname = os.path.join(imdir, 'near-%s.png' % (objkey.rsplit(':', 1)[-1])) if imdir else ''
        M, pim, coords = projectAndGetExtrema(p, project, polyim, fname=fname, mask=mask)
        if M == 0: continue # ignore things that weren't close enough
        objsnear.append([objkey, M/255.0, pim])
    # sort "near" objects by closevalue
    objsnear.sort(key=lambda o: o[1], reverse=1)
    if imdir:
        comb = combineImages([o[2] for o in objsnear])
        if comb:
            comb.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(imdir, 'near-poly.png'))
    # remap to get final scores
    objsnear = [(o[0], o[1]*o[1]) for o in objsnear] # we square the score to get a steeper falloff
    log('    Got %d objects "near": %s' % (len(objsnear), objsnear[:5]))
    return objsin, objsnear


