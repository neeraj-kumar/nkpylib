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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytz

DATA_DIR = Path('/home/neeraj/dp/Aranet4')
DEFAULT_TZ = pytz.timezone('America/New_York')

def parse_ts(ts: str, tz=DEFAULT_TZ) -> int:
    """Parse a timestamp string in the format 'dd/mm/yyyy HH:MM:SS' into epoch seconds.

    Since there is no timezone information in the raw data, we assume the timestamps are in the default timezone."""
    dt = datetime.strptime(ts, '%d/%m/%Y %H:%M:%S')
    dt = tz.localize(dt)  # Localize to the default timezone
    return int(dt.timestamp())

@dataclass
class Reading:
    ts: int # epoch seconds
    co2: int # ppm
    temp: float # °F
    humidity: float # %
    pressure: float # mmHg

    @classmethod
    def from_dict(cls, data: dict) -> Reading:
        """Create a Reading instance from a dictionary expected from the raw file."""
        try:
            return cls(
                ts=parse_ts(data['Time(dd/mm/yyyy)']),
                co2=int(data['Carbon dioxide(ppm)']),
                temp=float(data['Temperature(°F)']),
                humidity=float(data['Relative humidity(%)']),
                pressure=float(data['Atmospheric pressure(mmHg)'])
            )
        except Exception as e:
            print(f'Error parsing row {data}: {e}')
            raise

def read_dump(file_path: Path, existing: list[Reading]) -> list[Reading]:
    """Read a single CSV dump file and return its contents as a list of dicts.

    We overwrite any existing readings with the same timestamp, keeping only the latest one.
    We also sort all readings into ascending order by timestamp.
    """
    with file_path.open('r') as f:
        reader = DictReader(f)
        new_readings = []
        for row in reader:
            reading = Reading.from_dict(row)
            # Find where this timestamp would go in existing readings
            idx = bisect.bisect_left(existing, reading.ts, key=lambda r: r.ts)
            # Check if we found an exact match
            if idx < len(existing) and existing[idx].ts == reading.ts:
                # Replace the existing reading
                existing[idx] = reading
            else:
                new_readings.append(reading)
        # Combine new readings with existing ones and sort by timestamp
        combined_readings = sorted(existing + new_readings, key=lambda r: r.ts)
    return combined_readings

def read_all_dumps(data_dir: Path):
    """Read all CSV dump files under the specified directory and return their contents.

    Note that the data dir has folders per year, and those have the csv files. There's also overlap
    in the data, so we want to always use the latest one to overwrite the existing, but going in
    sorted order by dir and filename.
    """
    readings = []
    for year_dir in sorted(data_dir.iterdir()):
        print('Processing year directory:', year_dir)
        if not year_dir.is_dir():
            continue
        for file_path in sorted(year_dir.glob('*.csv')):
            if not file_path.is_file():
                continue
            print(f'Reading file {file_path}')
            readings.extend(read_dump(file_path, readings))
    print(f'Reading {len(readings)} readings from {data_dir}, first {readings[0]}, last {readings[-1]}')
    return readings


if __name__ == '__main__':
    parser = ArgumentParser(description='Aranet4 data dump utilities')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Directory containing Aranet4 data dumps')
    args = parser.parse_args()
    readings = read_all_dumps(args.data_dir)
