"""A resumable, multi-threaded download manager.

Licensed under the 3-clause BSD License:

Copyright (c) 2023-, Neeraj Kumar (neerajkumar.org)
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
import json
import logging
import os
import threading
import time

from enum import Enum
from hashlib import md5
from os.path import join, dirname, basename

import requests

from pony.orm import *

logger = logging.getLogger(__name__)

HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0'}

class Status(Enum):
    READY = 'ready'
    DOWNLOADING = 'downloading'
    PAUSED = 'paused'
    COMPLETED = 'completed'
    ERROR = 'error'

# define the File entity, which is the only table in the database
db = Database()

class File(db.Entity):
    id = PrimaryKey(int, auto=True)
    status = Required(str, default=Status.READY.value)
    # urls must be unique
    url = Required(str, unique=True)
    filename = Required(str)
    # default empty dir
    dir = Required(str)
    size = Required(int, default=0, size=64)
    cur_size = Required(int, default=0, size=64)
    # default added ts to now
    added_ts = Required(float, default=lambda: time.time())
    update_ts = Required(float, default=0)
    # optional md5 hash of the file
    md5 = Optional(str, nullable=True)
    # optional kwargs for requests
    kwargs = Optional(Json)


class Downloader:
    def __init__(self, db_path, dir='', n_threads=1, callback=None, stale_timeout=60 * 60):
        self.n_threads = n_threads
        self.db_path = db_path
        self.db = db
        # bind and generate tables if they don't already exist
        self.db.bind(provider='sqlite', filename=db_path, create_db=True)
        self.db.generate_mapping(create_tables=True)
        # other params
        self.callback = callback
        self.dir = dir
        # if there are any stale files, mark them as ready
        with db_session:
            for f in File.select(lambda f: f.status == Status.DOWNLOADING.value and time.time() - f.update_ts > stale_timeout):
                f.status = Status.READY.value
        # spawn download threads as daemons, pausing after each one
        self.threads = []
        for i in range(n_threads):
            t = threading.Thread(target=self.main_loop, daemon=True)
            self.threads.append(t)
            t.start()
            time.sleep(1)

    def add_file(self, url, filename, dir=None, size=0, status=Status.READY.value, md5=None, **kwargs):
        """Adds a file to this downloader database"""
        if dir is None:
            dir = self.dir
        # try adding the file, but ignore if it already exists
        try:
            with db_session:
                File(url=url, filename=filename, dir=dir, size=size, status=status, md5=md5, kwargs=kwargs)
        except Exception as e:
            logger.info(f'Error adding file {filename}: {e}')
            pass

    @db_session()
    def _download_file(self, url):
        """Downloads a file, updating its status as needed"""
        # get the file
        file = File.get(url=url)
        # bail if the status is not right
        if file.status != Status.DOWNLOADING.value:
            return
        logger.info(f'Downloading file {file.filename} from {file.url} to {file.dir}')
        path = join(file.dir, file.filename)
        kwargs = dict(file.kwargs)
        if 'auth' in kwargs:
            kwargs['auth'] = tuple(kwargs['auth'])
        # make parent dirs
        try:
            os.makedirs(dirname(path), exist_ok=True)
        except Exception as e:
            pass
        # make sure we have a file size
        if not file.size:
            # fetch size by doing a HEAD request
            logger.info(f'Fetching size of {file.filename}')
            r = requests.head(file.url, headers=HEADERS, allow_redirects=True, **kwargs)
            file.size = int(r.headers.get('Content-Length', 0))
            logger.info(f'Got size of {file.size} for {file.filename}')
            commit()
        # check if the file already exists and is the right size
        if os.path.exists(path):
            cursize = os.path.getsize(path)
            if file.size == cursize:
                logger.info(f'File {file.filename} already exists, skipping')
                file.status = Status.COMPLETED.value
                file.cur_size = file.size
                return
            # check if we have a partial file
            file.cur_size = os.path.getsize(path)
            commit()
        # download the file, starting from the current size
        logger.info(f'Downloading {file.filename} to {path}, starting at {file.cur_size}')
        headers = dict(Range=f'bytes={file.cur_size}-', **HEADERS)
        try:
            r = requests.get(file.url, headers=headers, stream=True, allow_redirects=True, **kwargs)
            # write the file
            with open(path, 'ab') as f:
                for chunk in r.iter_content(chunk_size=1024000):
                    f.write(chunk)
                    file.cur_size += len(chunk)
                    file.update_ts = time.time()
                    commit()
            # check if the file is complete
            if file.cur_size == file.size:
                logger.info(f'Completed download of {file.filename}')
                file.status = Status.COMPLETED.value
                file.update_ts = time.time()
                file.md5 = md5(open(path, 'rb').read()).hexdigest()
                commit()
            else:
                raise Exception(f'File {file.filename} is incomplete')
        except Exception as e:
            logger.info(f'Error downloading {file.filename}: {e}')
            file.status = f'{Status.ERROR.value}: {e}'
            file.update_ts = time.time()
        try:
            # call callback with url, path
            if self.callback is not None:
                self.callback(file.url, path)
        except Exception as e:
            logger.info(f'Error calling callback for {file.filename}: {e}')

    def main_loop(self):
        """The main downloader loop.

        This is run in a separate thread for each downloader thread.
        Note that this is a daemon thread, so it will exit when the main thread exits.
        """
        while True:
            # get the next file to download
            with db_session:
                file = File.select(lambda f: f.status == Status.READY.value).order_by(File.added_ts).first()
                if file is None:
                    # if there are no files to download, sleep for a bit
                    time.sleep(1)
                    continue
                logger.info(f'Found file {file.filename} to download')
                file.status = Status.DOWNLOADING.value
                file.update_ts = time.time()
            # download the file
            self._download_file(file.url)
            # pause for a bit
            time.sleep(1)

    def __del__(self):
        """If this download is deleted, pause all downloads and mark as ready"""
        with db_session:
            for f in File.select(lambda f: f.status == Status.DOWNLOADING.value):
                f.status = Status.READY.value
