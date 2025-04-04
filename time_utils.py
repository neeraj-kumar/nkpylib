"""Time-related utilities and constants"""

import datetime
import time

DAY_SECS = 24*3600
YEAR_SECS = 365*DAY_SECS

def parse_ts(ts: float|int|str) -> float:
    """Parses the timestamp to a float"""
    if isinstance(ts, str):
        # strip Z
        if ts.endswith('Z'):
            ts = ts[:-1]
        # remove fractional part for adding later
        tz: int|str
        frac: float|str
        if '.' in ts:
            ts, frac = ts.split('.', 1)
            frac = float(f'0.{frac}')
        else:
            frac = 0.0
        # figure out how to deal with timezones
        if '+' in ts:
            ts, tz = ts.split('+', 1)
            tz = int(tz)
        ret = datetime.datetime.fromisoformat(ts).timestamp()
        return ret + frac
    return ts

