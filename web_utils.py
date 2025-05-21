"""Various web-related utilities.

Note that I'm starting over in 2024, as the old webutils.py (no underscore) is mostly working with
web.py, not tornado.
"""

#TODO generic LLM searcher does multiple searches, and runs them (in parallel if possible)
#TODO   see if it makes sense to have generic search results combiner

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import json
import logging
import os
import re
import tempfile
import time
import traceback

from abc import ABC, abstractmethod
from collections import Counter
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from os.path import dirname, splitext
from typing import Any, Optional, Callable
from urllib.parse import urlparse

import requests

from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, StaticFileHandler

from nkpylib.constants import USER_AGENT
from nkpylib.utils import specialize
from nkpylib.ml.client import call_llm
from nkpylib.ml.llm_utils import load_llm_json
from nkpylib.thread_utils import sync_or_async, run_async

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LOG_FILE = 'search-logs.jsonl'

DEFAULT_LLM_MODEL = 'turbo'

REQUEST_TIMES: dict[str, float] = {}

async def make_request_async(url: str,
                             method='get',
                             min_delay=1,
                             request_times=REQUEST_TIMES,
                             headers=None,
                             files=None,
                             **kwargs) -> requests.Response:
    """Makes an (async) request to the given `url` with `method` with the given kwargs.

    Note the `kwargs` are passed directly to the requests.request() function, and are NOT the same
    as JSON post data or query params. In general, you usually want one of the following:
    - `params`={dict} for query params (in a get request)
    - `json`={dict} for a JSON post request
    - `data`={dict or list of tuples} for a form post request

    We maintain a mapping from host to last request time in request_times (stored by default as a
    global in this module, but you can pass your own), and we wait `min_delay` seconds before
    contacting the same host again.
    """
    host = urlparse(url).hostname
    elapsed = time.time() - request_times.get(host, 0)
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    request_times[host] = time.time()
    _headers = {'User-Agent': USER_AGENT}
    if headers is not None:
        _headers.update(headers)
    resp = await asyncio.to_thread(requests.request, method, url, headers=_headers, files=files, **kwargs)
    return resp

def make_request(url: str,
                 method='get',
                 min_delay=1,
                 request_times=REQUEST_TIMES,
                 headers=None,
                 files=None,
                 **kwargs) -> requests.Response:
    """Makes a (synchronous) request to the given `url` with `method` with the given kwargs.

    Note the `kwargs` are passed directly to the requests.request() function, and are NOT the same
    as JSON post data or query params. In general, you usually want one of the following:
    - `params`={dict} for query params (in a get request)
    - `json`={dict} for a JSON post request
    - `data`={dict or list of tuples} for a form post request

    We maintain a mapping from host to last request time in request_times (stored by default as a
    global in this module, but you can pass your own), and we wait `min_delay` seconds before
    contacting the same host again.
    """
    #FIXME figure out if we can just use the async version here, robustly
    host = urlparse(url).hostname
    elapsed = time.time() - request_times.get(host, 0)
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    request_times[host] = time.time()
    _headers = {'User-Agent': USER_AGENT}
    if headers is not None:
        _headers.update(headers)
    resp = requests.request(method, url, headers=_headers, files=files, **kwargs)
    return resp


def resolve_url(url: str, method='head', **kwargs) -> str:
    """Follows the url through all redirects and returns the ultimate url"""
    r = make_request(url, method, **kwargs)
    r.raise_for_status()
    return r.url

@asynccontextmanager
async def dl_temp_file(url_or_path):
    """Makes the given `url_or_path` available as a temporary file in a context manager.

    Use as:
        with dl_temp_file(url_or_path) as path:
            # do something with path

    This will download the file at `url_or_path` to a temporary file (with proper extension), and
    yield the path to that file. The file will be deleted when the context manager exits.

    The url can be a data url, normal url, or a local file (in which case, this just yields the
    path to that file).
    """
    if isinstance(url_or_path, bytes): # bytes -> actual file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(url_or_path)
            try:
                yield temp_path
            finally:
                os.remove(temp_path)
    else: # string -> path or url
        if url_or_path.startswith('http') or url_or_path.startswith('data:'):
            if url_or_path.startswith('data:'):
                ext = url_or_path.split(';')[0].split('/')[-1]
            else:
                ext = url_or_path.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as f:
                temp_path = f.name
            try:
                resp = await make_request_async(url_or_path, method='get', stream=True)
                resp.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                yield temp_path
            finally:
                os.remove(temp_path)
        else: # local file
            yield url_or_path

class BaseSearcher(ABC):
    """Searcher base class.

    This is a generic implementation of a custom search engine that can be plugged into a webapp.
    It assumes that the searching is done on the server (as opposed to in the frontend), and that
    there are different implementations (which inherit from this).

    You can initialize this in your search handler (or in the app, if you want to keep it around),
    and then call the `search()` method with a query string and the tornado request object to get
    the results. The tornado request object is stored as `self.req`

    By default, `search()` does the following:
    - parses the query using `parse()`
    - calls `_search()` with the query, the parser results, and number of results ('n')
    - log the query and results to a log file
    - creates a list of messages (arbitrary JSON objects) that are sent back to the client. These
      are helpful for showing what kinds of operations were run on the backend, how the query was
      parsed, etc.
    - creates a dict of timings
    - return the results

    The default parser simply strips the query.

    In practice, you might want to create your base searcher that inherits from this to do some more
    logging/processing, and then have different subclasses that implement the actual search/parse
    logic.
    """
    def __init__(self,
                 log_file_path: str=DEFAULT_SEARCH_LOG_FILE,
                 **kw: Any) -> None:
        """Initializes a searcher with given `log_file_path` and any other `kw` args.

        If a `log_file_path` is given, it will be a .jsonl file (one json object per line) that
        can be written to using `self.log()`. If not, no logging is done.

        The `kw` args are stored as instance variables on the searcher.
        """
        self.logf = open(log_file_path, 'a') if log_file_path else None
        self.req: Optional[RequestHandler] = None
        for k, v in kw.items():
            setattr(self, k, v)

    def log(self, **kw: Any) -> None:
        """Logs the given `kw` dict to the log file, adding some metadata."""
        if self.logf is None:
            return
        if self.req is not None:
            kw['ip'] = self.req.request.remote_ip
        kw['ts'] = time.time()
        kw['uname'] = os.uname()
        kw['class'] = self.__class__.__name__
        kw['caller'] = inspect.stack()[1].function
        self.logf.write(json.dumps(kw) + '\n')
        self.logf.flush()

    def add_msg(self, msg: str) -> None:
        """Adds a message to the current request object."""
        if self.req is not None:
            self.req.msgs.append(msg) # type: ignore
            # also print it to the log
            logger.info(f'Req msg: {msg}')

    def search(self, q: str, req: RequestHandler, **kw: Any) -> dict[str, object]:
        """Search wrapper which takes the query and request and returns results.

        This calls _search() for the underlying implementation.

        This method does the following:
        - store the request object in `self.req`
        - create an empty list in the req called `msgs`
          - you can add to this using `self.add_msg()`
        - parse the `q` string using `self.parse()`
        - call `self._search()` with the parsed query and any other `kw` args
        - logs the query and results, and some metadata.
        - collect some basic timings
        - returns a dict with q, parsed, ret, msgs, and times.

        A common pattern in your subclass is to call this method using super().search() and then
        do other post-processing. You can also update the `times` and `msgs` fields in the returned
        dict if you want to add more info.

        Note that there is no expectation of what the results should look like, other than being a
        dict, so you can return whatever you want. It will be available to the client as `ret` in the
        returned response.
        """
        t0 = time.time()
        self.req = req
        self.req.msgs = [] # type: ignore
        parsed = self.parse(q)
        t1 = time.time()
        ret = self._search(q=q, parsed=parsed, **kw)
        t2 = time.time()
        times = dict(parse=t1-t0, search=t2-t1, total=t2-t0)
        self.log(q=q,
                 parsed=parsed,
                 ret=ret,
                 times=times,
                 **kw)
        return dict(q=q,
                    parsed=parsed,
                    ret=ret,
                    times=times,
                    msgs=self.req.msgs) # type: ignore

    def parse(self, q: str) -> object:
        """Parses a search query.

        By default this returns q.strip(), but you can override this for more complex parsing.
        """
        return q.strip()

    @abstractmethod
    def _search(self, q: str, parsed: object, **kw: Any) -> dict[str, object]:
        """Takes a search query `q` and the `parsed` version, and returns a dict of results.

        You can also pass in any other `kw` args.
        """
        pass


class LLMSearcher(BaseSearcher):
    """A searcher that uses an LLM to parse natural language search queries.

    This takes in a "full" searcher which is assumed to be a fairly structured searcher, and helps
    with translating natural language queries into structured queries for the full searcher. If the
    parse fails, it then falls back to a "simple" searcher instead, which is assumed to accept any
    kind of query.
    """
    def __init__(self,
                 full_searcher: BaseSearcher,
                 simple_searcher: BaseSearcher,
                 prompt_fmt: str,
                 log_file_path: str=DEFAULT_SEARCH_LOG_FILE,
                 model_name: str=DEFAULT_LLM_MODEL,
                 history_length_s: float=900,
                 n_queries: int=5,
                 **kw: Any):
        """Inits this with the underlying searchers to use, the prompt_fmt, and some other init.

        The `prompt_fmt` should include {}-style format strings for various args that will be used
        to generate the actual prompt sent to the LLM. The args passed in will include at least:
        - `q`: the query string
        - `human_ts`: the current ts in 'YYYY-MM-DD HH:MM:SS' format
        - `n_queries`: the maximum number of queries to generate

        Uses the `model_name` for the LLM model to use, and the `log_file_path` for logging searches.

        If `history_length_s` is non-negative, it will load past searches from the log
        file `log_file_path` that happened within the last `history_length_s` seconds (0 for
        infinite) into `self.history` (filtering to those that include 'q' and 'ts' keys). This can
        be useful for giving the LLM some context for the current search.
        """
        super().__init__(log_file_path=log_file_path, **kw)
        self.full_searcher = full_searcher
        self.simple_searcher = simple_searcher
        self.prompt_fmt = prompt_fmt
        self.model_name = model_name
        self.n_queries = n_queries
        self.pool = ThreadPoolExecutor(max_workers=self.n_queries)
        # load past searches if requested
        self.history = []
        if log_file_path and history_length_s >= 0:
            try:
                with open(log_file_path) as f:
                    lines = f.readlines()
                    searches = [json.loads(line) for line in lines]
                    # filter to searches that happened within the last `history_length_s` seconds
                    now = time.time()
                    in_time = lambda s: (now - s['ts'] < history_length_s) if history_length_s > 0 else True
                    self.history = [s['q'] for s in searches if 'q' in s and 'ts' in s and in_time(s)]
                    logger.info(f'Loaded {len(self.history)} search history items from {log_file_path}')
            except Exception as e:
                logger.error(f'Could not load search history: {e}')

    def search(self, q: str, req: RequestHandler, **kw: Any) -> dict[str, object]:
        """Small wrapper to set `req` on our underlying searchers."""
        self.full_searcher.req = req
        self.simple_searcher.req = req
        return super().search(q=q, req=req, **kw)

    def gen_prompt_kw(self, q: str) -> dict[str, object]:
        """Generates the prompt kw for the LLM.

        This is called during `parse()` and should return the prompt string to send to the LLM.
        (Note that we have can't send arbitrary kw here, so you should do your work elsewhere.)

        You can subclass this, but make sure to call this method via super().gen_prompt_kw() to
        get some standard kw.
        """
        return dict(
            q=q,
            history='\n'.join(self.history),
            human_ts=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            n_queries=self.n_queries,
        )

    def parse(self, q: str) -> object:
        """Parses the `q` by calling the llm.

        This first generates the prompt kw using `gen_prompt_kw()`, and then formats our
        `prompt_fmt` with them to get the actual prompt. It then calls the LLM with that prompt to
        generate the structured search query.

        It then tries to parse the structured search query using the `full_searcher`, and if that
        fails, falls back to the `simple_searcher`.
        """
        assert self.req is not None
        kwargs = self.gen_prompt_kw(q)
        prompt = self.prompt_fmt.format(**kwargs)
        logger.debug(f'From q {q} got kwargs {kwargs}, and prompt: {prompt}')
        resp = call_llm.single(prompt, model=self.model_name).strip()
        lst = load_llm_json(resp)
        logger.info(f'LLM response: {lst}')
        self.add_msg(f'Translated "{q}" to "{lst}" using {self.model_name}')
        self.log(llm_translations={q: lst})
        try:
            parsed = [self.full_searcher.parse(structured_q) for structured_q in lst]
            self.req.searcher_used = self.full_searcher # type: ignore
        except Exception as e:
            self.add_msg(f'Could not run full search, switching to simple search: {e}')
            traceback.print_exc()
            parsed = [self.simple_searcher.parse(q)]
            self.req.searcher_used = self.simple_searcher # type: ignore
        return parsed

    @abstractmethod
    def _combine_results(self, results: list[dict[str, object]], **kw) -> dict[str, object]:
        """Takes a list of results and combines them into a single results dict.

        Implement in your subclass.
        """
        ...

    def _search(self, q: str, parsed: object, **kw: Any) -> dict[str, object]:
        """Runs the search using whichever searcher was used in `parse()`.

        Note that because we're generating multiple queries, we run many searches in parallel using
        our thread pool.
        """
        assert self.req is not None
        func = self.req.searcher_used._search
        #return func(q=q, parsed=parsed, **kw) # type: ignore
        futures = [self.pool.submit(func, q=q, parsed=p, **kw) for p in parsed]
        # wait for all the futures to finish
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f'*** Error running search: {e}')
                traceback.print_exc()
                self.add_msg(f'*** Error running search: {e}')
        # combine the results
        combined = self._combine_results(results)
        return combined


def run_search(q: str,
               searchers: list[tuple[str, Callable[[], BaseSearcher]]],
               **kw: Any) -> dict[str, object]:
    """Generic search handler that can flexibly switch between different searchers.

    This takes the `q` to run and then checks for certain prefixes to determine which searcher to
    use. This mapping is specified in the `searchers` list, which is a list of tuples where the
    first element is the prefix and the second element is a callable which evaluates to a
    searcher instance.

    It then calls the `search()` method on the chosen searcher with the `q` (with prefix removed)
    and any other `kw` args.
    """
    for prefix, make_searcher in searchers:
        if q.startswith(prefix):
            searcher = make_searcher()
            q = q[len(prefix):]
            break
    return searcher.search(q=q, **kw)

class BaseHandler(RequestHandler):
    """A base tornado handler that sets up some common functionality.
    - CORS headers
    """
    def set_default_headers(self):
        """allow CORS"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

INDEX_HTML_FMT = '''<!doctype html>
<html>
  <head>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/react-router-dom@4/umd/react-router-dom.min.js"></script>
    <script src="https://unpkg.com/babel-standalone/babel.js"></script>
    <script src="https://unpkg.com/prop-types/prop-types.min.js"></script>
    <script src="https://unpkg.com/localforage/dist/localforage.js"></script>
    <script src="https://unpkg.com/immer@9/dist/immer.umd.development.js"></script>

    <script src="https://unpkg.com/ag-grid-community@28/dist/ag-grid-community.js"></script>
    <script src="https://unpkg.com/ag-grid-react@28/bundles/ag-grid-react.min.js"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js" integrity="sha512-ZwR1/gSZM3ai6vCdI+LVF1zSq/5HznD3ZSTk7kajkaj4D292NLuduDCO1c/NT8Id+jE58KYLKT7hXnbtryGmMg==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="robots" content="noindex">
    <link rel="stylesheet" href="{static_path}/{css_filename}">
  </head>
<body>
  <div id="main" />
</body>
<script type="text/babel">{nk_utils_js}</script>
<script src="{static_path}/{js_filename}" type="text/babel"></script>
</html>'''

def default_index(static_path='/static',
                  js_filename='app.jsx',
                  css_filename='app.css',
                  ) -> str:
    """Returns default HTML index page that loads react etc from CDNs.

    This also inlines the nk js utils code.

    Your jsx and css code should be at /static/app.{jsx,css} by default, but you can change all of
    these by passing in the `static_path`, `js_filename`, and `css_filename` args.
    """
    # load 'nk-utils.js' from the same directory as this file
    with open(f'{dirname(__file__)}/nk-utils.js') as f:
        nk_utils_js = f.read()
    ret = INDEX_HTML_FMT.format(static_path=static_path,
                               js_filename=js_filename,
                               css_filename=css_filename,
                               nk_utils_js=nk_utils_js)
    return ret

def setup_and_run_server(parser: Optional[argparse.ArgumentParser]=None,
                         make_app: Callable[[], Application]=lambda: Application(),
                         default_port: int=8000,
                         post_parse_fn: Callable[[dict],None]|None=None) -> None:
    """Creates a web server and runs it.

    We create an `Application` instance using the `make_app` callable (by default just
    `Application()`), and then parse the command line arguments using the `parser` (we create a
    standard parser if not given). We then start the server on the port specified in the arguments
    (or `default_port` if not specified, which is added to the arg parser).

    You can also set a `post_parse_fn` which is called with the `args` object after arg parsing.

    We also setup logging.
    """
    # include line number
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d: %(message)s')
    if parser is None:
        parser = argparse.ArgumentParser(description='Web server')
    parser.add_argument('-p', '--port', type=int, default=default_port, help='Port to listen on')
    args = parser.parse_args()
    if post_parse_fn:
        post_parse_fn(vars(args))
    logger.info(f'Starting server on port {args.port}')
    app = make_app()
    app.listen(args.port)
    IOLoop.current().start()

def simple_react_tornado_server(jsx_path: str,
                                port: int,
                                more_handlers: list|None=None,
                                parser: argparse.ArgumentParser|None=None,
                                post_parse_fn: Callable[[dict],None]|None=None,
                                data_dir: str|None='.',
                                **kw):
    """Call this to start a tornado server to serve a single page react app from.

    Starts the server on `port` and serves the index page at / , which is a basic html page which
    loads react and other common libs from unpkg. The tornado server sets the static path to be the
    parent dir of `jsx_path`, and the index page will then load the jsx file from there.

    It also sets the `data_dir` (default: cur dir) as the static path for /data/ requests, so you
    can load up data files relative to this dir. If you set `data_dir` to None, it won't add a /data/
    handler.

    You can also pass in additional handlers to add to the tornado server using the `more_handlers`
    arg, which should be a list of tuples where the 1st element is the path spec (i.e., with
    regexps), 2nd element is the handler class, and 3rd (optional) one is params. This is useful for
    having an API.

    Any **kw you pass in are stored as instance variables on the application class. Your handlers
    can access these via `self.application.<varname>`.


    You probably want to call this function something like this from your main() function:

        simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/state_logger.jsx', port=11555)

    And then run it from a directory containing the data file care about.
    """
    parent, basename = os.path.split(jsx_path)
    logger.debug(f'Setting parent to {parent} and base to {basename}')
    class DefaultIndexHandler(RequestHandler):
        def get(self):
            self.write(default_index(js_filename=basename))

    class DefaultApplication(Application):
        def __init__(self):
            handlers = [
                (r"/", DefaultIndexHandler),
            ]
            if data_dir is not None:
                handlers.append((r"/data/(.*)", StaticFileHandler, {'path': data_dir}))
            if more_handlers:
                handlers.extend(more_handlers)
            settings = {
                "debug": True,
                "static_path": parent,
                "compress_response": True,
            }
            for k, v in kw.items():
                setattr(self, k, v)
            Application.__init__(self, handlers, **settings)

    setup_and_run_server(parser=parser,
                         make_app=DefaultApplication,
                         default_port=port,
                         post_parse_fn=post_parse_fn)

def bundle_js(js_path: str, css_path: str|None=None) -> str:
    """Bundles standard html index given js & css files into a single html file.

    This inlines the given js inside the html. Note that the output still contains CDN references to
    react, etc.

    If `css_path` is none, we look for it at the same location as the `js_path` just with the
    extension changed to css.

    This is useful for creating a standalone html file that can be opened in a browser without
    needing to run a server.
    """
    if css_path is None:
        css_path = splitext(js_path)[0] + '.css'
    with open(js_path) as f:
        js = f.read()
    try:
        with open(css_path) as f:
            css = f.read()
    except Exception:
        css = ''
    ret = default_index(static_path='XXX', js_filename='XXX', css_filename='XXX')
    # replace the stylesheet line with our css code
    ret = re.sub(r'<link rel="stylesheet" href=".*?">', f'<style>{css}</style>', ret)
    # replace the js line with our js code
    ret = re.sub(r'<script src=".*?" type="text/babel"></script>', f'<script type="text/babel">{js}</script>', ret)
    # convert all instances of text/babel with text/javascript
    #ret = re.sub(r'text/babel', 'text/javascript', ret)
    return ret


def clean_html_for_llm(s: str, max_length=200000, **kw) -> str:
    """Cleans up html for LLM input, primarily for length.

    - `max_length`: truncate the string to this length

    Pass in some `kw` else we do all of them. These are:
    - `image_data=True`: replace b64 image data with a placeholder
    - `quoted_image_data=True`: replace quoted b64 image data with a placeholder
    - `remove_svg=True`: remove all SVG elements
    - `remove_links=True`: remove all <link> elements (not actual <a> links)
    """
    orig_len = len(s)
    default_kw = dict(image_data=True,
                      quoted_image_data=True,
                      remove_svg=True,
                      remove_links=True,
                      remove_style=True,
                      remove_script=True,
                      )
    if not kw:
        kw = default_kw
    img_exts = ['png', 'jpg', 'jpeg', 'gif']
    img_data_prefix = 'data:image/(' + '|'.join(img_exts) + ')'
    def remove_tag(tag, s):
        # note that tags can be nested
        return re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', s, flags=re.DOTALL)

    for key, value in kw.items():
        if key == 'image_data' and value:
            data_re = re.compile(img_data_prefix + r';base64,[^"]+')
            s = data_re.sub('data:image/png;base64,PLACEHOLDER', s)
        if key == 'quoted_image_data' and value:
            quoted_data_re = re.compile(r'data:image/png%3bbase64%2c[^"]+')
            s = quoted_data_re.sub('data:image/png%3bbase64%2cPLACEHOLDER', s)
        if key == 'remove_links' and value:
            s = re.sub(r'<link [^>]*>', '', s)
        if key == 'remove_svg' and value:
            s = remove_tag('svg', s)
        if key == 'remove_style' and value:
            s = remove_tag('style', s)
        if key == 'remove_script' and value:
            s = remove_tag('script', s)

    msg = f'Got input string type {type(s)}, len {orig_len} -> {len(s)}'
    s = s[:max_length]
    msg += f' -> {len(s)}, {s[-100:]}'
    logger.info(msg)
    return s

def simplify_html(html: str, idx: int|None=None) -> str:
    """Simplifies htmls by checking against text content.

    For each node, if the node has no text content, we remove it.
    Or, if it has the exact same text content as one of its children, we remove it.
    """
    soup = BeautifulSoup(html, "html.parser")
    orig_clean = soup.prettify()
    counts: Counter[str] = Counter()
    for node in soup.find_all():
        counts['total'] += 1
        if isinstance(node, Comment):
            #print(f"  Killing comment node with {len(node.text)} bytes: {node.text[:100]}...")
            counts['comment'] += 1
            counts['killed'] += 1
            node.extract()
#       elif node.name in ['img']:
#           counts['img'] += 1
#           counts['alive'] += 1
        elif node.name in ['style', 'script', 'footer']:
            #print(f"  Killing {node.name} node with {len(node.text)} bytes: {node.text[:100]}...")
            counts[node.name] += 1
            counts['killed'] += 1
            node.extract()
        elif not node.text.strip():
            #print(f'    Killing {node.name} node with no text content: {str(node)[:100]}...')
            node.extract()
            counts['empty'] += 1
            counts['killed'] += 1
        # check for any nodes with a style attribute that contains 'display: none' or 'visibility: hidden'
        elif node.has_attr('style') and ('display: none' in node['style'] or 'visibility: hidden' in node['style']):
            #print(f'    Killing {node.name} node with hidden style: {str(node)[:100]}...')
            node.extract()
            counts['hidden'] += 1
            counts['killed'] += 1
        else:
            counts['alive'] += 1
            #counts[f'type-{node.name}'] += 1

    def simplify_node(node):
        """Recursive function to check if a node can be simplified.

        If this node has a single child with text content identical to this node itself, then we can
        remove this node and replace it with the child.
        """
        if not hasattr(node, 'children'):
            return
        if not node.children:
            return
        children = list(node.children)
        if len(children) == 1:
            child = children[0]
            if child.text == node.text:
                #print(f"  Simplifying node {node.name} with {len(node.text)} bytes: {node.text[:100]}...")
                node.replace_with(child)
                counts['simplified'] += 1
        for child in children:
            simplify_node(child)

    first = str(soup)
    C = lambda s: f'{len(s)}b/{count_tokens(s)}t ({100.0*count_tokens(s) / len(s):.0f}%)'
    simplify_node(soup)
    ret = str(soup)
    logger.info(f"Simplified html from {C(html)} -> {C(first)} -> {C(ret)}, {pformat(counts)}")
    # write versions to disk if we were given an idx
    if idx is not None:
        with open(f'streeteasy_raw_{idx}.html', 'w') as f:
            f.write(orig_clean)
        with open(f'streeteasy_clean_{idx}.html', 'w') as f:
            f.write(soup.prettify())
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple react web server (default port 13000)')
    parser.add_argument('jsx_path', help='Path to jsx file to load')
    parser.add_argument('-d', '--data_dir', default='.', help='Path to data dir [.]')
    parser.add_argument('-b', '--bundle_path', help='Bundle the html jsx file to this filename (no serving in that case)')
    args = parser.parse_args()
    if args.bundle_path:
        bundled = bundle_js(args.jsx_path)
        try:
            os.makedirs(dirname(args.bundle_path), exist_ok=True)
        except Exception:
            pass
        with open(args.bundle_path, 'w') as f:
            f.write(bundled)
    else:
        simple_react_tornado_server(jsx_path=args.jsx_path,
                                    port=13000,
                                    parser=parser,
                                    data_dir=args.data_dir)
