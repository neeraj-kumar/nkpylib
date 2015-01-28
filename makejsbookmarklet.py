#!/usr/bin/env python
"""Simple script to make a bookmarklet from a set of js and/or css files passed in on stdin.

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
import json

BASE_TEMPLATE = '''
/* asynchronously loads a javascript script */
/* adapted from http://stackoverflow.com/questions/7943004/how-to-load-javascript-dynamically */
function asyncloadjs(url, handler){
    var s = document.createElement('script');
    s.setAttribute("type","text/javascript");
    s.setAttribute("src", url);
    s.onreadystatechange = function () {
        if (this.readyState == 'complete' && handler) handler();
    };
    s.onload = handler;
    if (typeof s!="undefined"){
        document.getElementsByTagName("head")[0].appendChild(s);
    }
}

/* Loads several javascript files in order, finally calling the finalhandler */
function loadmultjs(urls, finalhandler){
    if (urls.length > 0){
        asyncloadjs(urls[0], function(){
            loadmultjs(urls.slice(1), finalhandler);
        });
    } else {
        if (finalhandler) finalhandler();
    }
}

/* simple function to load javascript */
/* from http://ntt.cc/2008/02/10/4-ways-to-dynamically-load-external-javascriptwith-source.html */
function simpleloadjs(url){
    document.write('<script src="'+url+'"><\/script>');
}

/* dynamically load css, once jquery has been loaded */
/* from http://stackoverflow.com/questions/2685614/load-external-css-file-like-scripts-in-jquery-which-is-compatible-in-ie-also */
function loadcss(url){
    $("<link/>", {rel: "stylesheet", type: "text/css", href: url}).appendTo("head");
}


/* load all the javascript files, and then load the css files */
var toloadjs = %(js)s;

loadmultjs(toloadjs, function(){
    var toloadcss = %(css)s;
    for (var i = 0; i < toloadcss.length; i++){
        loadcss(toloadcss[i]);
    }
});
console.log(toloadjs, 'hello!');
'''

BOOKMARKLET_TEMPLATE = '''
javascript:(function(){%s})();
'''

DEFAULT = dict(jquery=['//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js'],
             jqueryui=['//ajax.googleapis.com/ajax/libs/jqueryui/1.10.3/jquery-ui.min.js',
                       '//ajax.googleapis.com/ajax/libs/jqueryui/1.8/themes/smoothness/jquery-ui.css'],
                json2=['//raw.github.com/douglascrockford/JSON-js/master/json2.js'],
                 none=[])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <%s> [...]' % (sys.argv[0], '|'.join(DEFAULT))
        sys.exit()
    lines = []
    for a in sys.argv[1:]:
        if a in DEFAULT:
            lines.extend(DEFAULT[a])
    lines.extend([l.strip() for l in sys.stdin if l.strip()])
    js, css = [], []
    for url in lines:
        if url.endswith('.js'):
            js.append(url)
        elif url.endswith('.css'):
            css.append(url)
    print js
    print css
    code = BASE_TEMPLATE % (dict(js=json.dumps(js), css=json.dumps(css)))
    bk = BOOKMARKLET_TEMPLATE % code.replace('\n','');
    print bk
