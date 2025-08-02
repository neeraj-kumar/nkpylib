"""Utilities to interface with Aranet4 data dumps.


The raw csv files look like this:
Time(dd/mm/yyyy),Carbon dioxide(ppm),Temperature(°F),Relative humidity(%),Atmospheric pressure(mmHg)
05/10/2023 13:38:35,"721","77.4","61","766"
05/10/2023 13:39:35,"704","77.4","60","765"
05/10/2023 13:40:35,"681","77.5","60","766"

Or the first field name might be in the format:
Time(DD/MM/YYYY H:mm:ss)

We parse each row into a `Reading` dataclass, which contains the timestamp (as epoch seconds) and the other readings.
"""

from __future__ import annotations

import json
import logging

from argparse import ArgumentParser
from csv import DictReader
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from nkpylib.stringutils import GeneralJSONEncoder

logger = logging.getLogger(__name__)

DATA_DIR = Path('/home/neeraj/dp/Aranet4')
DEFAULT_TZ = pytz.timezone('America/New_York')
# Pre-calculate DST and non-DST offsets for our timezone
DST_OFFSET = -4 * 3600  # EDT is UTC-4
STD_OFFSET = -5 * 3600  # EST is UTC-5

def get_offset(ts: int, transition_cache: dict[int, tuple[int, int]] = {}) -> int:
    """Get timezone offset for timestamp, caching DST transitions by year.

    Args:
        ts: Timestamp in epoch seconds
        transition_cache: Optional cache of DST transitions by year

    Returns:
        Offset in seconds from UTC (e.g. -14400 for EDT, -18000 for EST)
    """
    year = datetime.fromtimestamp(ts, pytz.UTC).year
    if year not in transition_cache:
        logger.debug(f'Calculating DST transitions for year {year}')
        # Find transitions by checking March and November
        # Use naive datetimes for checking offsets
        spring = datetime(year, 3, 1)
        summer = datetime(year, 7, 1)
        # Check each day until we find the transitions
        for dt in (spring + timedelta(days=i) for i in range(45)):
            if DEFAULT_TZ.utcoffset(dt) != DEFAULT_TZ.utcoffset(spring):
                spring_forward = int(DEFAULT_TZ.localize(dt).timestamp())
                break
        for dt in (summer + timedelta(days=i) for i in range(180)):
            if DEFAULT_TZ.utcoffset(dt) != DEFAULT_TZ.utcoffset(summer):
                fall_back = int(DEFAULT_TZ.localize(dt).timestamp())
                break
        transition_cache[year] = (spring_forward, fall_back)

    spring, fall = transition_cache[year]
    return DST_OFFSET if spring <= ts < fall else STD_OFFSET

def fast_parse_ts(ts: str) -> int:
    """Fast parse of timestamp string in format 'dd/mm/yyyy HH:MM:SS' into UTC seconds.

    Much faster than strptime by doing direct integer math instead of datetime objects.
    Handles leap years correctly."""
    # Split into date and time
    date_str, time_str = ts.split(' ')
    # Split date and time parts
    day, month, year = map(int, date_str.split('/'))
    hour, minute, second = map(int, time_str.split(':'))
    # Create timestamp using direct integer math
    # This avoids datetime object creation until the end
    days = (year - 1970) * 365 + day - 1  # rough days since epoch
    # Add leap years
    leap_years = (year - 1968) // 4 - (year - 1900) // 100 + (year - 1600) // 400
    days += leap_years
    # Add days for months
    days_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    days += days_in_month[month - 1]
    # Convert everything to seconds
    ts_utc = days * 86400 + hour * 3600 + minute * 60 + second
    # Add timezone offset based on whether DST was in effect
    return ts_utc + get_offset(ts_utc)

def parse_ts(ts: str, fmt: str='%D/%m/%Y %H:%M:%S') -> int:
    """Parse a timestamp string in the format 'dd/mm/yyyy HH:MM:SS' into epoch seconds.

    Since there is no timezone information in the raw data, we determine the correct
    offset based on whether DST was in effect at that time."""
    if fmt == '%D/%m/%Y %H:%M:%S':
        # Fast path for our specific format
        return fast_parse_ts(ts)
    else:
        # Use strptime for other formats
        dt = datetime.strptime(ts, fmt)
        # Convert to UTC seconds, adjusting for DST
        ts_utc = int(dt.timestamp())
        return ts_utc + get_offset(ts_utc)


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
        try:
            time_str = data.get('Time(dd/mm/yyyy)') or data.get('Time(DD/MM/YYYY H:mm:ss)')
            ts = parse_ts(time_str) if time_str else None
        except Exception as e:
            logger.debug(f'Error parsing timestamp in row {data}: {e}, skipping')
            return None
        if ts is None:
            logger.debug(f'Invalid timestamp in row {data}, skipping')
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
                    logger.debug(f'Error parsing field {field} with value {value} in row {data}: {e}, skipping')
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
        logger.debug('Processing year directory:', year_dir)
        if not year_dir.is_dir():
            continue
        for file_path in sorted(year_dir.glob('*.csv')):
            if not file_path.is_file():
                continue
            if not path_filter(file_path):
                #logger.debug(f'Skipping file {file_path} due to filter')
                continue
            logger.debug(f'Reading file {file_path} ({len(readings)} existing readings)')
            readings, ts_to_idx = read_dump(file_path, readings, ts_to_idx)
    logger.debug(f'Reading {len(readings)} readings from {data_dir}, first {readings[0]}, last {readings[-1]}')
    return readings


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description='Aranet4 data dump utilities')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Directory containing Aranet4 data dumps')
    parser.add_argument('-f', '--filter', type=str, help='Optional substring filter for file paths to read', default='')
    parser.add_argument('output_path', type=Path, nargs='?', default=None,
                        help='Optional output path to save the readings as JSON')
    args = parser.parse_args()
    print(args.filter)
    readings = read_all_dumps(args.data_dir, lambda p: args.filter in str(p) if args.filter else True)
    print(f'Loaded {len(readings)} readings from {args.data_dir}, first: {readings[0]}, last: {readings[-1]}')
    if args.output_path:
        with args.output_path.open('w') as f:
            json.dump(readings, f, cls=GeneralJSONEncoder, indent=2)
        print(f'Saved {len(readings)} readings to {args.output_path}')
