"""A state-and-event logger to allow replaying logs to reconstruct state at different points.

"""

import json
import logging
import os
import re
import shutil
import sys
import time

from argparse import ArgumentParser
from collections import defaultdict
from os.path import join, dirname
from pprint import pprint
from queue import Queue, Empty
from threading import Thread

import termcolor
import tornado.web

from tornado.web import RequestHandler

from nkpylib.stringutils import parse_num_spec
from nkpylib.web_utils import simple_react_tornado_server

logger = logging.getLogger(__name__)

#TODO partial history matching
#TODO for now, we just write to a single logfile, but in the future we should rotate

class StateLogger:
    """A class to manage state and event update logging.

    The idea is to be able to recreate the state of the system at any point in time by replaying
    these logs. This is useful for debugging, and for training models that predict the next state.
    """
    def __init__(self, name_field='_name', log_dir='logs', recency=3600):
        """Initializes the state logger.

        Logs are stored inside `log_dir`. We use `recency` to determine how far back in history to
        copy objects from (for deduplication).
        """
        self.name_field = name_field
        self.log_dir = log_dir
        log_path = join(log_dir, 'log.jsonl')
        try:
            os.makedirs(dirname(log_path), exist_ok=True)
        except Exception as e:
            pass
        self.recency = recency
        self.history_by_signature = defaultdict(dict)
        self.ignore_keys = ['_ts', '_human_ts']
        # parse the log file to get the history
        ts_regexp = re.compile(r'"_ts": ([0-9.]+)')
        try:
            with open(log_path) as f:
                for line in f:
                    # filter by ts
                    match = ts_regexp.search(line)
                    if not match or float(match.group(1)) < time.time() - recency:
                        continue
                    obj = json.loads(line)
                    sig = self.signature(obj)
                    if sig:
                        self.history_by_signature[sig][obj['_ts']] = obj
        except Exception as e:
            pass
        self.logf = open(log_path, 'a')
        # create a background thread to write to the log file (daemon)
        self.queue = Queue()
        self.thread = Thread(target=self.write_to_log, daemon=True)
        self.thread.start()

    def __del__(self):
        self.queue.put((None, None))
        self.thread.join()
        self.logf.close()

    def write_to_log(self):
        """Writes to the log file. Start this in a background thread."""
        while True:
            try:
                ts, obj = self.queue.get(timeout=0.2)
                if ts is None:
                    break
                self._log(ts, obj)
            except Empty:
                pass

    def signature(self, obj):
        """Computes the signature of an object"""
        if self.name_field not in obj:
            return None
        keys = [k for k in obj if not k in self.ignore_keys]
        return (obj[self.name_field], tuple(sorted(keys)))

    def jsonable(self, obj):
        """Returns a jsonable version of the object (doesn't modify original object)"""
        ret = dict()
        for key, value in obj.items():
            try:
                json.dumps(value)
                ret[key] = value
            except:
                pass
        return ret

    def _log(self, ts, obj):
        """Actual logging implementation"""
        human_ts = time.ctime(ts)
        obj = self.jsonable(obj)
        sig = self.signature(obj)
        if sig:
            to_del = []
            for old_ts, prev in self.history_by_signature[sig].items():
                if old_ts < time.time() - self.recency:
                    to_del.append(old_ts)
                else:
                    if self.equals(prev, obj):
                        obj = {'_copy_of':old_ts, self.name_field:obj[self.name_field]}
                        break
            # delete expired entries
            for old_ts in to_del:
                del self.history_by_signature[sig][old_ts]
        # only add items to history that are not copies
        if '_copy_of' not in obj:
            self.history_by_signature[sig][ts] = obj
        # now add _ts and _human_ts fields to object
        obj.update(_ts=ts, _human_ts=human_ts)
        # write to log file
        self.logf.write(json.dumps(obj) + '\n')
        self.logf.flush()

    def log(self, obj):
        """Logs the given object"""
        self.queue.put((time.time(), obj))

    def equals(self, prev, obj):
        """Compares two log objects for equality"""
        keys1 = set(prev.keys())
        keys1 -= set(self.ignore_keys)
        keys2 = set(obj.keys())
        keys2 -= set(self.ignore_keys)
        if keys1 != keys2:
            return False
        for key in keys1:
            if prev[key] != obj[key]:
                return False
        return True

    def log_external_file(self, rel_path, name, **kw):
        """Returns a path to a file that will be logged externally.

        The `rel_path` should be the suffix (including dirs, potentially).
        We'll create parent dirs and internally create a log entry with `name` (required) and
        `**kw`.
        """
        path = join(self.log_dir, rel_path)
        try:
            os.makedirs(dirname(path), exist_ok=True)
        except Exception as e:
            pass
        self.log({self.name_field:name, 'path':path, **kw})
        return path


class StateLogReader:
    """A class to read state logs and step through them"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []
        self.idx = 0
        self.idx_by_ts = {}
        self.load()

    def load(self):
        """Loads the log file"""
        with open(self.log_path) as f:
            for line in f:
                obj = json.loads(line)
                # do some filtering
                # no event name
                if '_name' not in obj:
                    continue
                name = obj['_name'].lower()
                # log handler writing is not interesting
                if name == 'write-loghandler':
                    continue
                # main handler with / uri is not interesting
                if name == 'init-mainhandler' and obj.get('uri') == '/':
                    continue
                # skip audio events for now (both client and server)
                if 'audio' in name or 'voice' in name:
                    continue
                self.idx_by_ts[obj['_ts']] = len(self.history)
                self.history.append(obj)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        ret = dict(**self.history[idx])
        if '_copy_of' in ret:
            # this is a copy, so get the original
            old_ts = ret['_copy_of']
            idx = self.idx_by_ts.get(old_ts, -1)
            if idx >= 0:
                prev = self.history[idx]
                # update the ret with the fields from this
                ret.update({k:v for k, v in prev.items() if k not in ret})
        return ret

    def __iter__(self):
        return self

    def cli_print_obj(self, obj, idx):
        """Prints an object for the CLI.

        This first prints the idx in blue, human ts in green, then the object in json format
        (without ts or human ts).
        """
        print(termcolor.colored(f"Index: {idx}", 'blue'), termcolor.colored(obj['_human_ts'], 'green'))
        obj = {k:v for k, v in obj.items() if k not in ['_ts', '_human_ts']}
        pprint(obj, width=shutil.get_terminal_size().columns)

    def cli(self):
        """Main interactive CLI to step through the log"""
        print(f"Loaded {len(self)} log entries")
        while True:
            if self.idx < 0:
                self.idx = 0
            if self.idx >= len(self):
                self.idx = len(self) - 1
            obj = self[self.idx]
            self.cli_print_obj(obj, idx=self.idx)
            print("Commands: [n]ext, [p]rev, [j]ump(idx), [q]uit")
            cmd = input("Command: ")
            cmd, rest = cmd[0], cmd[1:]
            if cmd == 'n':
                self.idx += 1
            elif cmd == 'p':
                self.idx -= 1
            elif cmd == 'j':
                self.idx = int(rest)
                if self.idx < 0:
                    self.idx += len(self)
            elif cmd == 'q':
                break
            else:
                print(f"Unknown command: {cmd}")


def cli_main():
    """CLI state log reader"""
    parser = ArgumentParser(description="Step through a state log file")
    parser.add_argument('log_path', help="The path to the log file")
    args = parser.parse_args()
    reader = StateLogReader(args.log_path)
    reader.cli()

class BaseHandler(RequestHandler):
    @property
    def reader(self):
        """Returns the state log reader"""
        return self.application.more_kw['reader']

class GetHandler(BaseHandler):
    def get(self, indices):
        nums = sorted(parse_num_spec(indices))
        items = {i: self.reader[i] for i in nums if i < len(self.reader)}
        msg = f'Our state log reader has {len(self.reader)} entries, reading {nums}'
        self.write(dict(msg=msg, items=items, length=len(self.reader)))

def web_main():
    # load the data file from first arg
    parser = ArgumentParser(description="Web state log reader")
    parser.add_argument('data_path', help="The path to the data file")
    kw = {}
    def post_parse_fn(args):
        reader = StateLogReader(args.data_path)
        kw['reader'] = reader

    simple_react_tornado_server(jsx_path=f'{dirname(__file__)}/state_logger.jsx',
                                port=11555,
                                more_handlers=[(r'/get/(.*)', GetHandler)],
                                parser=parser,
                                post_parse_fn=post_parse_fn,
                                more_kw=kw)


if __name__ == '__main__':
    #cli_main()
    web_main()
