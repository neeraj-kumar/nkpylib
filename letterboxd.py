"""Interface to letterboxd (movie tracking site).

Currently we only deal with downloaded archives from the site.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil

from argparse import ArgumentParser
from collections import Counter
from csv import DictReader
from os.path import dirname, join, exists
from typing import Any, Iterator

from tqdm import tqdm

from movies.searcher import search_movies # type: ignore

logger = logging.getLogger(__name__)

Entry = dict[str, Any]

# json serializer that deals with dates
def json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def json_dumps(obj, **kw):
    return json.dumps(obj, default=json_serial, **kw)

class LetterboxdArchive:
    """Interface to a letterboxd archive."""
    def __init__(self, path: str):
        self.path = path
        self.diary: dict[tuple[str, str], dict[str, Any]] = {}
        self.diary_by_imdb_id: dict[str, list[dict[str, Any]]] = {}
        self.read_data()
        logger.debug(f'Read {len(self.diary)} diary entries: {json_dumps(sorted(self.diary.items())[:5], indent=2)}')

    def iter_csv(self, rel_path: str) -> Iterator[dict[str, Any]]:
        """Iterates through the CSV file at `rel_path`.

        Also tries the "orphaned" version of the file.
        """
        try:
            with open(join(self.path, rel_path)) as f:
                yield from DictReader(f)
        except FileNotFoundError:
            pass
        try:
            with open(join(self.path, 'orphaned', rel_path)) as f:
                yield from DictReader(f)
        except FileNotFoundError:
            pass

    def read_data(self):
        """Reads various kinds of data from the archive"""
        # parse date in YYYY-MM-DD format
        date_parser = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date()
        casts = {
            'Year': int,
            'Rating': float,
            'Rewatch': lambda s: s.lower() == 'yes',
            'Tags': lambda s: s.split(',') if s else [],
            'Date': date_parser,
            'Watched Date': date_parser,
        }
        def cast_row(row):
            for k, cast in casts.items():
                try:
                    row[k] = cast(row[k])
                except ValueError:
                    pass
            return row

        # read diary
        for row in self.iter_csv("diary.csv"):
            row = cast_row(row)
            key = (row['Date'], row['Name'])
            self.diary[key] = row
        # read reviews and add them in
        for row in self.iter_csv("reviews.csv"):
            row = cast_row(row)
            key = (row['Date'], row['Name'])
            if key in self.diary:
                self.diary[key]['Review'] = row['Review']
            else:
                self.diary[key] = row
        # read/load imdb ids
        self.add_imdb_ids()

    def add_imdb_ids(self):
        """Adds imdb ids to the diary entries"""
        # read existing imdb ids
        imdb_ids_path = join(dirname(self.path), 'imdb_ids.json')
        try:
            with open(imdb_ids_path) as f:
                imdb_ids = json.load(f)
                imdb_map = {row['letterboxd_uri']: row['imdb_id'] for row in imdb_ids if row.get('imdb_id') and row.get('letterboxd_uri')}
        except FileNotFoundError:
            imdb_ids = []
            imdb_map = {}
        for entry in self.diary.values():
            entry['imdb_id'] = imdb_map.get(entry['Letterboxd URI'])
            self.diary_by_imdb_id.setdefault(entry['imdb_id'], []).append(entry)
        # search for missing imdb ids
        queries = []
        to_search = []
        for entry in self.diary.values():
            if entry.get('imdb_id') or not entry.get('Letterboxd URI'):
                continue
            queries.append(dict(title=entry['Name'], titles=[entry['Name']], year=entry['Year']))
            to_search.append(entry)
        logger.info(f'Read {len(imdb_ids)} imdb ids, searching for {len(queries)} missing ones')
        if not queries:
            return
        #queries = queries[:15]
        try:
            ret = search_movies(queries, max_workers=1)
            # add new imdb ids to the diary, and to our list which we will save
            for entry, matches in tqdm(zip(to_search, ret)):
                if not matches:
                    continue
                logger.debug(f'For entry {entry}: {matches}')
                entry['imdb_id'] = matches[0].id
                self.diary_by_imdb_id.setdefault(entry['imdb_id'], []).append(entry)
                imdb_ids.append(dict(title=entry['Name'], year=entry['Year'], letterboxd_uri=entry['Letterboxd URI'], imdb_id=entry['imdb_id']))
        except Exception as e:
            logger.error(f'Error searching for movies: {e}')
            return
        # save the new list of imdb ids to a temp file then rename
        with open(imdb_ids_path+'.tmp', 'w') as f:
            json.dump(imdb_ids, f, indent=2)
        shutil.move(imdb_ids_path+'.tmp', imdb_ids_path)

    def __iter__(self) -> Iterator[Entry]:
        """Iterates through all our diary entries (in chronological order)."""
        for key in sorted(self.diary.keys()):
            yield self.diary[key]

    def __len__(self) -> int:
        """The number of watches (not necessarily unique movies)"""
        return len(self.diary)

    def __getitem__(self, key: str) -> list[Entry]:
        """Get diary entries (a list) by imdb id"""
        imdb_id_pattern = re.compile(r"tt\d+")
        matches = imdb_id_pattern.search(key)
        if not matches:
            raise KeyError(f'No imdb id in {key}')
        imdb_id = matches.group(0)
        return self.diary_by_imdb_id[imdb_id]

    def get(self, key: str, default=None) -> list[Entry]:
        """Get diary entries (a list) by imdb id"""
        try:
            return self[key]
        except KeyError:
            return default

def test_main(archive: LetterboxdArchive, **kw) -> None:
    """Simple driver to list movies in this archive"""
    # read our mementodb and get list of movies that we've seen
    from nkpylib.memento import MovieDB
    mdb = MovieDB()
    i = 0
    for m in (mdb):
        f = m['fields']
        if 'imdb link' not in f or f['status']:
            continue
        matches = archive.get(f['imdb link'])
        if not matches:
            continue
        #print(f'got movie: {m} vs {matches}')
        print(f"Seen movie {f['title']} ({f.get('recommended by')})")
        i += 1

def stats_main(archive: LetterboxdArchive, **kw) -> None:
    """Simple driver to show my stats from this year about this archive.

    This includes:
    - number of movies watched
    - number of new movies/re-watches
    - number of movies per rating, subdivided by new vs re-watch
    """
    year = datetime.date.today().year
    entries = [e for e in archive if e['Date'].year == year]
    counts = Counter()
    top_new = []
    for e in entries:
        #print(e)
        counts['total'] += 1
        stars = str(e.get('Rating', 0))
        counts[f'stars={stars}'] += 1
        is_new = e.get('Year', 0) >= year - 1
        if is_new:
            counts[f'New movie'] += 1
            counts[f'New movie stars={stars}'] += 1
        else:
            counts[f'Old movie'] += 1
            counts[f'Old movie stars={stars}'] += 1
        if e.get('Rewatch'):
            counts['rewatch'] += 1
            counts[f'rewatch={stars}'] += 1
        else:
            counts['new'] += 1
            counts[f'new={stars}'] += 1
            if stars in ('4.5', '5.0'):
                top_new.append(e)

    print(f'Stats for year {year}:')
    for key, count in counts.items():
        if '=' not in key:
            print(f'  {key}: {count}')
    for key, count in sorted(counts.items()):
        if '=' in key:
            print(f'  {key}: {count}')
    print('Top new 4.5+ star movies:')
    for e in top_new:
        print(f"  {e['Name']} ({e.get('Year', '')}): {e.get('Rating', '')}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    funcs = {fn.__name__: fn for fn in [test_main, stats_main]}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("func", choices=funcs.keys(), help="Function to run")
    parser.add_argument("path", nargs='?', default='/home/neeraj/dp/projects/movies/letterboxd/latest', help="Path to the letterboxd archive")
    args = parser.parse_args()
    archive = LetterboxdArchive(args.path)
    print(f'Read letterboxd archive with {len(archive)} entries')
    kw = vars(args)
    func = funcs[kw.pop('func')]
    kw.pop('path')
    func(archive, **kw)
