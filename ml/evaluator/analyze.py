"""A script to help analyze results"""

from __future__ import annotations

import json
import logging
import os

from argparse import ArgumentParser
from glob import glob
from os.path import dirname, abspath, join

from objexplore import explore

from nkpylib.ml.feature_set import JsonLmdb

logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/'

def get_latest_result_path(dir=RESULTS_DIR) -> str:
    """Get the path to the latest results JsonLMDB database."""
    result_files = sorted(glob(join(dir, 'results_*.lmdb')))
    if not result_files:
        raise FileNotFoundError('No results files found in results directory')
    return result_files[-1]

def explore_results(db: JsonLmdb, **kw):
    """Uses objexplore to explore the results database."""
    from rich.tree import Tree
    from rich import print
    # get all results
    results = {k: db[k] for k in db.keys() if k.startswith('key:')}
    # filter down to 5 results
    results = {k: v for i, (k, v) in enumerate(results.items()) if i < 5}
    logger.info(f'Exploring {len(results)} results')
    tree = Tree(results)
    #explore(list(results.items())[:5], **kw)
    print(tree)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    funcs = {f.__name__: f for f in [explore_results]}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('func', type=str, choices=funcs, help=f'Function to run [choices: {", ".join(funcs)}]')
    parser.add_argument('--result_path', type=str, help='Path to the results JsonLMDB database [default latest]')
    args = parser.parse_args()
    if not args.result_path:
        args.result_path = get_latest_result_path()
    db = JsonLmdb.open(args.result_path)
    kw = vars(args)
    kw.pop('result_path')
    func = kw.pop('func')
    funcs[func](db=db, **kw)
