"""Utils for working with google calendar (.ics) files"""

from __future__ import annotations

import logging

from argparse import ArgumentParser
from pathlib import Path

from ics import Calendar

from nkpylib.google.constants import BACKUPS_DIR

logger = logging.getLogger(__name__)

CALENDARS_PATH = Path(BACKUPS_DIR) / Path('Takeout/Calendar/')
MAIN_CALENDARS = ['NKAK Health', 'N+A', 'Neeraj_s Fun Calendar']

"""
Takeout/Calendar/NK Food.ics
Takeout/Calendar/Meetings_Work.ics
Takeout/Calendar/NKAK Health.ics
Takeout/Calendar/N+A.ics
Takeout/Calendar/NK Reminders.ics
Takeout/Calendar/Neeraj_s Fun Calendar.ics
"""

def read_calendar(path: str) -> Calendar:
    """Reads a calendar file and returns a Calendar object."""
    with open(path, 'r') as file:
        calendar_data = file.read()
    cal = Calendar(calendar_data)
    # remove any cancelled events
    cal.events = [event for event in cal.events if not event.status or event.status != 'CANCELLED']
    logger.debug(f'Read {len(cal.events)} events from {path}')
    if not cal.events:
        raise ValueError(f'No events found in the calendar at {path}!')
    # print earliest and latest event
    earliest_event = min(cal.events, key=lambda e: e.begin)
    latest_event = max(cal.events, key=lambda e: e.end)
    logger.debug(f'Earliest event: {earliest_event} at {earliest_event.begin}')
    logger.debug(f'Latest event: {latest_event} at {latest_event.end}')
    return cal

def read_all_calendars(names: list[str]=MAIN_CALENDARS, dir: Path = CALENDARS_PATH) -> list[Calendar]:
    ret = {name: read_calendar(dir / f'{name}.ics') for name in names}
    return ret

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description='Read a calendar file and print its events.')
    parser.add_argument('--path', type=str, help='Path to the calendar file (.ics)')
    args = parser.parse_args()
    if args.path:
        calendar = read_calendar(args.path)
    else:
        ret = read_all_calendars()
