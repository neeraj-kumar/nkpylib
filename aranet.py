"""Utilities to interface with Aranet4 data dumps.


The raw csv files look like this:
Time(dd/mm/yyyy),Carbon dioxide(ppm),Temperature(°F),Relative humidity(%),Atmospheric pressure(mmHg)
05/10/2023 13:38:35,"721","77.4","61","766"
05/10/2023 13:39:35,"704","77.4","60","765"
05/10/2023 13:40:35,"681","77.5","60","766"

We parse each row into a `Reading` dataclass, which contains the timestamp (as epoch seconds) and the other readings.
"""

from __future__ import annotations

import bisect

from argparse import ArgumentParser
from csv import DictReader
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytz

DATA_DIR = Path('/home/neeraj/dp/Aranet4')
DEFAULT_TZ = pytz.timezone('America/New_York')

def parse_ts(ts: str, tz=DEFAULT_TZ) -> int:
    """Parse a timestamp string in the format 'dd/mm/yyyy HH:MM:SS' into epoch seconds.

    Since there is no timezone information in the raw data, we assume the timestamps are in the
    default timezone. For efficiency, we first parse as naive datetime, get UTC timestamp, then
    apply timezone offset at the end. During DST transitions, we assume the first occurrence."""
    naive_dt = datetime.strptime(ts, '%d/%m/%Y %H:%M:%S')
    # Get UTC timestamp without timezone info
    ts_utc = int(naive_dt.timestamp())
    # Add timezone offset, using is_dst=True for ambiguous times
    offset = tz.utcoffset(naive_dt, is_dst=True).total_seconds()  # type: ignore
    return int(ts_utc + offset)

@dataclass
class Reading:
    ts: int # epoch seconds
    co2: int|None # ppm
    temp: float|None # °F
    humidity: float|None # %
    pressure: float|None # mmHg

    @classmethod
    def from_dict(cls, data: dict) -> Reading|None:
        """Create a Reading instance from a dictionary expected from the raw file.

        The ts must be valid, else we return `None` from this method.
        Any other fields are allowed to be empty strings, in which case those values in the
        `Reading` will be set to None, but we will still return a valid Reading object.
        """
        ts = parse_ts(data.get('Time(dd/mm/yyyy)'))
        if ts is None:
            print(f'Invalid timestamp in row {data}, skipping')
            return None
        # mapping from CSV field names to Reading attributes and cast types
        field_map = {
            'Carbon dioxide(ppm)': ('co2', int),
            'Temperature(°F)': ('temp', float),
            'Relative humidity(%)': ('humidity', float),
            'Atmospheric pressure(mmHg)': ('pressure', float),
        }
        # Create a dictionary with the parsed values
        parsed_data = {'ts': ts}
        for field, (attr, cast_type) in field_map.items():
            value = data.get(field)
            if value == '':
                parsed_data[attr] = None
            else:
                try:
                    parsed_data[attr] = cast_type(value)
                except ValueError as e:
                    print(f'Error parsing field {field} with value {value} in row {data}: {e}, skipping')
                    return None
        return cls(**parsed_data)

    def __repr__(self):
        return (f'Reading<ts={self.ts}, co2={self.co2}, temp={self.temp}, '
                f'humidity={self.humidity}, pressure={self.pressure}>')

def read_dump(file_path: Path, existing: list[Reading], ts_to_idx: dict[int, int]|None=None) -> tuple[list[Reading], dict[int, int]]:
    """Read a single CSV dump file and extends its `Readings` to the existing list of `Readings`.

    We overwrite any existing readings with the same timestamp, keeping only the latest one.
    Uses a timestamp->index mapping for O(1) lookups of existing readings.

    Args:
        file_path: Path to the CSV file to read
        existing: List of existing readings
        ts_to_idx: Optional mapping of timestamp->index in existing list. Will be created if None.

    Returns:
        Tuple of (updated readings list, updated timestamp->index mapping)
    """
    if ts_to_idx is None:
        ts_to_idx = {r.ts: i for i, r in enumerate(existing)}

    new_readings = []  # Buffer for readings that need to be appended
    with file_path.open('r') as f:
        reader = DictReader(f)
        for row in reader:
            reading = Reading.from_dict(row)
            if not reading:
                continue
            if reading.ts in ts_to_idx:
                # Update existing reading
                existing[ts_to_idx[reading.ts]] = reading
            else:
                # Add to buffer of new readings
                new_readings.append(reading)

    if new_readings:
        # Add all new readings at once and update mapping
        start_idx = len(existing)
        existing.extend(new_readings)
        for i, reading in enumerate(new_readings, start=start_idx):
            ts_to_idx[reading.ts] = i
        # Sort once at the end
        existing.sort(key=lambda r: r.ts)
        # Rebuild mapping after sort
        ts_to_idx = {r.ts: i for i, r in enumerate(existing)}

    return existing, ts_to_idx

def read_all_dumps(data_dir: Path, path_filter: Callable = lambda p: True):
    """Read all CSV dump files under the specified directory and return their contents.

    Note that the data dir has folders per year, and those have the csv files. There's also overlap
    in the data, so we want to always use the latest one to overwrite the existing, but going in
    sorted order by dir and filename.

    You can optionally pass a `path_filter` function to filter out files you don't want to read.
    It is given the full path of the file, and should return `True` if the file should be read,
    """
    readings = []
    ts_to_idx = {}
    for year_dir in sorted(data_dir.iterdir()):
        print('Processing year directory:', year_dir)
        if not year_dir.is_dir():
            continue
        for file_path in sorted(year_dir.glob('*.csv')):
            if not file_path.is_file():
                continue
            if not path_filter(file_path):
                print(f'Skipping file {file_path} due to filter')
                continue
            print(f'Reading file {file_path} ({len(readings)} existing readings)')
            readings, ts_to_idx = read_dump(file_path, readings, ts_to_idx)
    print(f'Reading {len(readings)} readings from {data_dir}, first {readings[0]}, last {readings[-1]}')
    return readings


if __name__ == '__main__':
    parser = ArgumentParser(description='Aranet4 data dump utilities')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Directory containing Aranet4 data dumps')
    parser.add_argument('-f', '--filter', type=str, help='Optional substring filter for file paths to read', default='/2024/')
    args = parser.parse_args()
    print(args.filter)
    readings = read_all_dumps(args.data_dir, lambda p: args.filter in str(p) if args.filter else True)
