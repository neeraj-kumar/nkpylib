"""Simple script ot allow uploading of images to a site.

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
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(abspath)
os.chdir(abspath)
import web

# DEFINE URL MAPPING
urls = (
        r'/?', 'index',
        r'/upload/?', 'upload',
)

# CONSTANTS
IMAGEDIR = 'static/uploadimages'
IMAGE_EXTS = 'png jpg'.split()

# CREATE APP
app = web.application(urls, globals())
application = app.wsgifunc()


# PROCESSING FUNCTIONS
def process(im, fname, params=None):
    """Processes the given image in some way, and returns rendered HTML.
    'im' is the uploaded Image.
    'fname' is the path to the image.
    'params' is an optional dict with other parameters.
    To refer to the uploaded image in the HTML, use '/'+fname
    """
    from PIL import Image, ImageFilter
    import subprocess
    # cleanup params into a printable string
    if not params:
        params = {}
    p = '<br>'.join(['<b>%s:</b> %s' % (k, v) for k, v in params.items()])
    # do some processing with the image in python itself:
    edgeim = im.convert('L').filter(ImageFilter.FIND_EDGES)
    edgefname = fname.replace('.jpg', '.png')
    edgeim.save(edgefname)
    # do some processing using an external program:
    extoutfname = edgefname.replace('.png', '-external.png')
    try:
        subprocess.call(['convert', '-median', '3', edgefname, extoutfname])
    except Exception:
        # couldn't run the external program
        web.debug('Error with external program')
    s = '''
    <html><body>
        <h2>We got the following uploaded image:</h2>
        <img src="/%s" /><br>
        <h2>Running edge detection in Python:</h2>
        <img src="/%s" /><br>
        <h2>Running median filtering on the edges using an external program:</h2>
        <img src="/%s" /><br>
        <h2>The other parameters in the form were:</h2>
            %s
    </body></html>''' % (fname, edgefname, extoutfname, p)
    return s


# WEB CLASSES
class index(object):
    """The index page"""
    def GET(self):
        """Simple page with an upload form"""
        # Note that you can add various hidden fields, and set them using javascript on the client.
        # their values will be read upon submission, and you use them for processing.
        # For getting GPS location, see http://www.w3schools.com/html5/html5_geolocation.asp
        s = '''\
<html><body>
    <h1>Simple Image Upload Test</h1>
    <form action="http://arnold.cs.washington.edu:37123/upload/" method="POST" enctype="multipart/form-data">
        <input type="file" id="fileinput" name="myfile" accept="image/*"> </input>
        <input type="submit" value="submit" />
        <input type="hidden" id="latitudeval" name="latitude" value="" />
        <input type="hidden" id="longitudeval" name="longitude" value="" />
        <input type="hidden" id="altitudeval" name="altitude" value="" />
    </form>
    <div id="output"></div>
</body></html>
'''
        web.header('Content-Type', 'text/html; charset=utf-8')
        return s

class upload(object):
    """The upload page"""
    def POST(self):
        """Called when the user uploads a file using POST"""
        from PIL import Image
        # get the data from the form
        x = web.input(myfile=web.Storage(file=''))
        # check the input file to make sure it's legal
        if 'myfile' not in x or not x.myfile.file: raise web.badrequest()
        # create the filename (and containing dir) based on the current time
        fname = os.path.join(IMAGEDIR, '%d.jpg' % (time.time()*1000))
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass
        # save the image
        fout = open(fname, 'w')
        fout.write(x.myfile.file.read())
        fout.close()
        # remove the image from the uploaded params
        del x['myfile']
        try:
            # if it's a valid image, process it
            im = Image.open(fname)
            web.debug('Saved valid image to %s, about to call process() on it' % (fname))
            results = process(im, fname, dict(x))
            return results
        except IOError:
            # if not, delete the image and return bad request
            try:
                os.remove(fname)
            except IOError:
                pass
            raise web.badrequest()


if __name__ == '__main__':
    # Main driver
    # If a command line argument is specified, it is assumed to be the port.
    # The default port is 8080
    app.run()
