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
from csv import DictReader
from os.path import dirname, join, exists
from typing import Any, Iterator

from movies.imdb import search_movies

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
        self.diary = {}
        self.diary_by_imdb_id = {}
        self.read_data()
        print(f'Read {len(self.diary)} diary entries: {json_dumps(sorted(self.diary.items())[:5], indent=2)}')

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
        print(f'Read {len(imdb_ids)} imdb ids, searching for {len(queries)} missing ones')
        if not queries:
            return
        #queries = queries[:15]
        ret = search_movies(queries, n_results=1)
        # add new imdb ids to the diary, and to our list which we will save
        for entry, matches in zip(to_search, ret):
            if not matches:
                continue
            entry['imdb_id'] = matches[0][0]
            self.diary_by_imdb_id.setdefault(entry['imdb_id'], []).append(entry)
            imdb_ids.append(dict(title=entry['Name'], year=entry['Year'], letterboxd_uri=entry['Letterboxd URI'], imdb_id=entry['imdb_id']))
        # save the new list of imdb ids to a temp file then rename
        with open(imdb_ids_path+'.tmp', 'w') as f:
            json.dump(imdb_ids, f, indent=2)
        shutil.move(imdb_ids_path+'.tmp', imdb_ids_path)

    def __iter__(self) -> Iterator[Entry]:
        """Iterates through all our diary entries."""
        return iter(self.diary.values())

    def __len__(self) -> int:
        """The number of watches (not necessarily unique movies)"""
        return len(self.diary)

    def __getitem__(self, key: str) -> list[Entry]:
        """Get diary entries (a list) by imdb id"""
        imdb_id_pattern = re.compile(r"tt\d+")
        imdb_id = imdb_id_pattern.search(key).group(0)
        return self.diary_by_imdb_id[imdb_id]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the letterboxd archive")
    args = parser.parse_args()
    archive = LetterboxdArchive(args.path)
    print(f'Read archive with {len(archive)} entries')
    for i, m in enumerate(archive):
        print(f'movie {i}: {m}')
        if i > 4:
            break
    print(f'For interstellar, got following diary entries: {archive["tt0816692"]}')
