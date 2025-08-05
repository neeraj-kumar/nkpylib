"""Access to Google Timeline, the history of your location.

This actually is not from the takeout, but rather exported from the phone directly:
On your phone, go to Settings -> Location -> Location Services -> Timeline -> Export Timeline. This
is not identical to takeout, but this export also contains a json file with your entire location
history, which can be visualized using online tools like Dawarich (not my website, use at own risk).

https://dawarich.app/

The timeline file is a JSON file with the following structure:
- "semanticSegments": [
    {
      "startTime": "2014-07-15T18:00:00.000-04:00",
      "endTime": "2014-07-15T20:00:00.000-04:00",
      "timelinePath": [
        {
          "point": "47.619773°, -122.3212269°",
          "time": "2014-07-15T19:34:00.000-04:00"
        },
        {
          "point": "47.6140505°, -122.3219479°",
          "time": "2014-07-15T19:47:00.000-04:00"
        },
        {
          "point": "47.6175282°, -122.3210819°",
          "time": "2014-07-15T19:56:00.000-04:00"
        }
      ]
    },
    ...
    {
      "startTime": "2014-07-15T19:34:20.000-04:00",
      "endTime": "2014-07-15T20:02:22.000-04:00",
      "startTimeTimezoneUtcOffsetMinutes": -420,
      "endTimeTimezoneUtcOffsetMinutes": -420,
      "activity": {
        "start": {
          "latLng": "47.6197896°, -122.3213708°"
        },
        "end": {
          "latLng": "47.6225353°, -122.321511°"
        },
        "distanceMeters": 0.0,
        "topCandidate": {
          "type": "UNKNOWN_ACTIVITY_TYPE",
          "probability": 0.0
        }
      }
    },
    ...
    {
      "startTime": "2025-07-29T15:29:31.000-04:00",
      "endTime": "2025-07-29T15:38:52.000-04:00",
      "startTimeTimezoneUtcOffsetMinutes": -240,
      "endTimeTimezoneUtcOffsetMinutes": -240,
      "activity": {
        "start": {
          "latLng": "40.773598°, -73.9833402°"
        },
        "end": {
          "latLng": "40.7640552°, -73.9774017°"
        },
        "distanceMeters": 1465.828369140625,
        "probability": 0.9936237931251526,
        "topCandidate": {
          "type": "IN_BUS",
          "probability": 0.31177178025245667
        }
      }
    },
    {
      "startTime": "2025-07-29T15:38:52.000-04:00",
      "endTime": "2025-07-29T18:47:15.000-04:00",
      "startTimeTimezoneUtcOffsetMinutes": -240,
      "endTimeTimezoneUtcOffsetMinutes": -240,
      "visit": {
        "hierarchyLevel": 0,
        "probability": 0.7579012513160706,
        "topCandidate": {
          "placeId": "ChIJzyuMivlYwokRBC6U4D4a0Xc",
          "semanticType": "HOME",
          "probability": 0.4542723596096039,
          "placeLocation": {
            "latLng": "40.7627908°, -73.9771957°"
          }
        }
      }
    },
    ...
  ]
- "rawSignals": [
    {
      "position": {
        "LatLng": "40.7633763°, -73.9767325°",
        "accuracyMeters": 100,
        "altitudeMeters": 4.900000095367432,
        "source": "WIFI",
        "timestamp": "2025-06-29T23:40:16.000-04:00",
        "speedMetersPerSecond": 0.0
      }
    },
    {
      "position": {
        "LatLng": "40.7633763°, -73.9767325°",
        "accuracyMeters": 100,
        "altitudeMeters": 4.900000095367432,
        "source": "WIFI",
        "timestamp": "2025-06-29T23:50:16.000-04:00",
        "speedMetersPerSecond": 0.0
      }
    },
    ...
    {
      "wifiScan": {
        "deliveryTime": "2025-07-29T23:19:40.000-04:00",
        "devicesRecords": [
          {
            "mac": 84301499987454,
            "rawRssi": -36
          },
          {
            "mac": 51316151154174,
            "rawRssi": -37
          },
          {
            "mac": 194522237825917,
            "rawRssi": -50
          },
          ...
          {
            "mac": 40247473037776,
            "rawRssi": -91
          }
        ]
      }
    },
    {
      "activityRecord": {
        "probableActivities": [
          {
            "type": "STILL",
            "confidence": 0.8199999928474426
          },
          {
            "type": "IN_VEHICLE",
            "confidence": 0.05999999865889549
          },
          {
            "type": "IN_RAIL_VEHICLE",
            "confidence": 0.05999999865889549
          },
          {
            "type": "ON_FOOT",
            "confidence": 0.05000000074505806
          },
          {
            "type": "WALKING",
            "confidence": 0.05000000074505806
          },
          {
            "type": "IN_ROAD_VEHICLE",
            "confidence": 0.05000000074505806
          },
          {
            "type": "ON_BICYCLE",
            "confidence": 0.009999999776482582
          },
          {
            "type": "UNKNOWN",
            "confidence": 0.0
          }
        ],
        "timestamp": "2025-07-29T23:23:56.000-04:00"
      }
    },
    {
      "position": {
        "LatLng": "40.763385°, -73.9767747°",
        "accuracyMeters": 100,
        "altitudeMeters": 4.900000095367432,
        "source": "WIFI",
        "timestamp": "2025-07-29T23:37:14.000-04:00",
        "speedMetersPerSecond": 0.0
      }
    }
  ]
- "userLocationProfile": {
    "frequentPlaces": [
      {
        "placeId": "ChIJAAAAAAAAAAARBC6U4D4a0Xc",
        "placeLocation": "40.7627908°, -73.9771957°",
        "label": "HOME"
      },
      {
        "placeId": "ChIJAAAAAAAAAAAReZxCM_fHDGM",
        "placeLocation": "40.6971521°, -73.9768077°",
        "label": "WORK"
      },
      {
        "placeId": "ChIJAAAAAAAAAAAR2Xw3BLuSO2c",
        "placeLocation": "40.701378°, -73.986603°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAARefq6VNIOrlY",
        "placeLocation": "40.715143°, -73.9911467°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAARg1iQGDTSgyw",
        "placeLocation": "40.7637614°, -73.9742923°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAARwnpMeNyWLFs",
        "placeLocation": "40.7284097°, -74.00433°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAARTS3NUw8-59o",
        "placeLocation": "40.7312056°, -74.0016722°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAAR9VDpbdSqQZQ",
        "placeLocation": "40.698879°, -73.979346°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAARsywIOc28k3Q",
        "placeLocation": "40.732338°, -74.000496°"
      },
      {
        "placeId": "ChIJAAAAAAAAAAAROyEKYwbPEwo",
        "placeLocation": "40.7647437°, -73.9789166°"
      }
    ],
    "persona": {
      "travelModeAffinities": [
        {
          "mode": "WALKING",
          "affinity": 0.7917981147766113
        },
        {
          "mode": "IN_SUBWAY",
          "affinity": 0.27760252356529236
        },
        {
          "mode": "IN_PASSENGER_VEHICLE",
          "affinity": 0.10094637423753738
        },
        {
          "mode": "IN_BUS",
          "affinity": 0.04416403919458389
        },
        {
          "mode": "IN_TRAIN",
          "affinity": 0.012618296779692173
        },
        {
          "mode": "FLYING",
          "affinity": 0.0063091483898460865
        }
      ]
    }
  }
}
"""

from __future__ import annotations

import bisect
import json
import logging

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar, Protocol

import pytz

from nkpylib.geo import haversine_dist
from nkpylib.google.constants import BACKUPS_DIR

logger = logging.getLogger(__name__)

GPS = tuple[float, float]  # Latitude, Longitude

class ActivityType(Enum):
    UNKNOWN_ACTIVITY_TYPE = "UNKNOWN_ACTIVITY_TYPE"
    CYCLING = "CYCLING"
    FLYING = "FLYING"
    IN_BUS = "IN_BUS"
    IN_FERRY = "IN_FERRY"
    IN_PASSENGER_VEHICLE = "IN_PASSENGER_VEHICLE"
    IN_SUBWAY = "IN_SUBWAY"
    IN_TRAIN = "IN_TRAIN"
    IN_TRAM = "IN_TRAM"
    MOTORCYCLING = "MOTORCYCLING"
    WALKING = "WALKING"
    # the following are from raw signals, not sure why they are different
    ON_FOOT = "ON_FOOT"
    IN_VEHICLE = "IN_VEHICLE"
    IN_RAIL_VEHICLE = "IN_RAIL_VEHICLE"
    IN_ROAD_VEHICLE = "IN_ROAD_VEHICLE"
    ON_BICYCLE = "ON_BICYCLE"
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    STILL = "STILL"
    TILTING = "TILTING"
    EXITING_VEHICLE = "EXITING_VEHICLE"

class PlaceType(Enum):
    UNKNOWN = "UNKNOWN"
    HOME = "HOME"
    INFERRED_HOME = "INFERRED_HOME"
    WORK = "WORK"
    INFERRED_WORK = "INFERRED_WORK"
    SEARCHED_ADDRESS = "SEARCHED_ADDRESS"
    ALIASED_LOCATION = "ALIASED_LOCATION"

@dataclass
class Estimate:
    prob: float

@dataclass
class TimePoint:
    t0: float

@dataclass
class SemanticSegment(TimePoint):
    t1: float

@dataclass
class TimelinePath(SemanticSegment):
    points: list[tuple[GPS, float]]  # List of tuples (GPS, timestamp in seconds since epoch)

@dataclass
class Activity(SemanticSegment, Estimate):
    start: GPS
    end: GPS
    distance: float
    type: ActivityType

@dataclass
class Visit(SemanticSegment, Estimate):
    level: int
    place_id: str
    type: PlaceType
    loc: GPS
    place_prob: float

@dataclass
class TimelineMemory(SemanticSegment):
    place_id: str
    distance: float

class SourceType(Enum):
    WIFI = "WIFI"
    GPS = "GPS"
    UNKNOWN = "UNKNOWN"
    CELL = "CELL"

@dataclass
class RawSignal(TimePoint):
    pass

@dataclass
class RawPosition(RawSignal):
    source: SourceType
    loc: GPS
    accuracy: float
    altitude: float
    speed: float

@dataclass
class WifiScan(RawSignal):
    devices: list[tuple[int, int]] # List of tuples (MAC address, RSSI)

@dataclass
class ActivityRecord(RawSignal):
    activities: list[tuple[ActivityType, float]]  # List of tuples (ActivityType, probability)

class HasTimestamp(Protocol):
    """Protocol for objects with a single timestamp."""
    t0: float

class HasTimeRange(HasTimestamp, Protocol):
    """Protocol for objects with a time range (start and end timestamps)."""
    t1: float

TimeT = TypeVar("TimeT", bound=HasTimestamp, covariant=True)


class TimeSortedLst(Generic[TimeT]):
    """A class to keep a list of items sorted by their timestamp(s).

    Works with both single-timestamp objects (t0) and time-range objects (t0, t1).
    Items are primarily sorted by t0, and secondarily by t1 if it exists.
    """
    def __init__(self, items: list[TimeT]):
        self.items = sorted(items, key=self._get_sort_key)

    def _get_sort_key(self, x: TimeT) -> tuple[float, float]:
        if hasattr(x, 't1'):
            return (x.t0, x.t1)  # type: ignore
        return (x.t0, x.t0)

    def __repr__(self):
        return f"TSList<{len(self.items)} items>"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TimeT:
        return self.items[idx]

    def find_at_time(self, ts: float|str) -> list[TimeT]:
        """Find items that contain or are closest to the given timestamp.

        Args:
            ts: Either seconds since epoch, or a timestamp string parseable by ts_to_seconds()

        Returns:
            List of matching items, sorted by proximity to timestamp.
            For time ranges (t0,t1), returns exact matches that contain the timestamp.
            For single timestamps (t0), returns the closest items.
        """
        if isinstance(ts, str):
            ts = ts_to_seconds(ts)

        key = (ts, ts)  # Create search key in same format as _get_sort_key
        idx = bisect.bisect_left(self.items, key, key=self._get_sort_key)

        # Check for exact matches first
        matches = []
        if idx < len(self.items):
            item = self.items[idx]
            if hasattr(item, 't1'):  # TimeRange
                if item.t0 <= ts <= item.t1:  # type: ignore
                    return [item]
            elif item.t0 == ts:  # Single timestamp exact match
                return [item]

        # No exact match, return closest items
        closest = []
        if idx > 0:
            closest.append(self.items[idx - 1])
        if idx < len(self.items):
            closest.append(self.items[idx])

        return sorted(closest, key=lambda x: abs(x.t0 - ts))

@dataclass
class Timeline:
    semantic: TimeSortedLst[SemanticSegment]
    raw: TimeSortedLst[RawSignal]

def ts_to_seconds(ts: str) -> float:
    """Convert a timestamp string (with tz) to seconds since epoch.
    
    The input timestamp should include timezone info. If it ends with 'Z',
    it's interpreted as UTC."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.astimezone(pytz.utc)  # Convert to UTC
        return dt.timestamp()  # Return seconds since epoch
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp '{ts}': {e}")

def parse_gps(gps_str: str) -> GPS:
    """Parse a GPS string in the format 'lat,lon' into a tuple of floats."""
    # remove degree symbols and whitespace
    gps_str = gps_str.replace('°', '').replace(' ', '')
    lat, lon = map(float, gps_str.split(','))
    return (lat, lon)

# Parsers for different types of semantic segments
SEMANTIC_SEGMENT_PARSERS = {
    'timelinePath': {
        'class': TimelinePath,
        'fields': {
            'points': lambda obj: [(parse_gps(p['point']), ts_to_seconds(p['time']))
                                 for p in obj['timelinePath']]
        }
    },
    'activity': {
        'class': Activity,
        'fields': {
            'start': lambda a: parse_gps(a['activity']['start']['latLng']),
            'end': lambda a: parse_gps(a['activity']['end']['latLng']),
            'distance': lambda a: a['activity'].get('distanceMeters', 0.0),
            'type': lambda a: ActivityType(a['activity']['topCandidate']['type']),
            'prob': lambda a: a['activity']['topCandidate'].get('probability', 0.0)
        }
    },
    'visit': {
        'class': Visit,
        'fields': {
            'level': lambda a: a['visit']['hierarchyLevel'],
            'place_id': lambda a: a['visit']['topCandidate']['placeId'],
            'type': lambda a: PlaceType(a['visit']['topCandidate']['semanticType']),
            'prob': lambda a: a['visit'].get('probability', 0.0),
            'loc': lambda a: parse_gps(a['visit']['topCandidate']['placeLocation']['latLng']),
            'place_prob': lambda a: a['visit']['topCandidate'].get('probability', 0.0)
        }
    },
    'timelineMemory': {
        'class': TimelineMemory,
        'fields': {
            'place_id': lambda a: a['timelineMemory']['trip']['destinations'][0]['identifier']['placeId'],
            'distance': lambda a: a['timelineMemory']['trip'].get('distanceFromOriginKms', 0.0) * 1000.0
        }
    }
}

# Parsers for different types of raw signals
RAW_SIGNAL_PARSERS = {
    'position': {
        'class': RawPosition,
        'time_field': lambda a: ts_to_seconds(a['position']['timestamp']),
        'fields': {
            'source': lambda a: SourceType(a['position']['source']),
            'loc': lambda a: parse_gps(a['position']['LatLng']),
            'accuracy': lambda a: a['position'].get('accuracyMeters', 0.0),
            'altitude': lambda a: a['position'].get('altitudeMeters', 0.0),
            'speed': lambda a: a['position'].get('speedMetersPerSecond', 0.0)
        }
    },
    'wifiScan': {
        'class': WifiScan,
        'time_field': lambda a: ts_to_seconds(a['wifiScan']['deliveryTime']),
        'fields': {
            'devices': lambda a: [(d['mac'], d['rawRssi']) for d in a['wifiScan']['devicesRecords']]
        }
    },
    'activityRecord': {
        'class': ActivityRecord,
        'time_field': lambda a: ts_to_seconds(a['activityRecord']['timestamp']),
        'fields': {
            'activities': lambda a: [(ActivityType(act['type']), act['confidence'])
                                   for act in a['activityRecord']['probableActivities']]
        }
    }
}

def parse_segment(obj: dict, parsers: dict, is_semantic: bool=True) -> TimePoint:
    """Parse a segment using the appropriate parser from the given parsers dict."""
    for key, parser in parsers.items():
        if key in obj:
            fields = {
                name: fn(obj) for name, fn in parser['fields'].items()
            }
            if is_semantic:
                return parser['class'](
                    t0=ts_to_seconds(obj['startTime']),
                    t1=ts_to_seconds(obj['endTime']),
                    **fields
                )
            else:
                return parser['class'](
                    t0=parser['time_field'](obj),
                    **fields
                )
    raise NotImplementedError(f"Unknown segment type: {obj}")

def read_timeline(path: str) -> Timeline:
    """Reads the timeline json from `path` and converts to Timeline object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse semantic segments
    counts: Counter = Counter()
    semantic_segments = []
    for obj in data['semanticSegments']:
        segment = parse_segment(obj, SEMANTIC_SEGMENT_PARSERS, is_semantic=True)
        semantic_segments.append(segment)
        counts[next(k for k in SEMANTIC_SEGMENT_PARSERS if k in obj)] += 1
    logger.debug(f'Got {len(semantic_segments)} semantic segments: {json.dumps(dict(counts), indent=2, sort_keys=True)}')

    # Parse raw signals
    counts.clear()
    raw_signals = []
    for obj in data['rawSignals']:
        signal = parse_segment(obj, RAW_SIGNAL_PARSERS, is_semantic=False)
        raw_signals.append(signal)
        counts[next(k for k in RAW_SIGNAL_PARSERS if k in obj)] += 1
    logger.debug(f'Got {len(raw_signals)} raw signals: {json.dumps(dict(counts), indent=2, sort_keys=True)}')

    return Timeline(
        semantic=TimeSortedLst(semantic_segments),
        raw=TimeSortedLst(raw_signals),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description="Read Google Timeline JSON file.")
    parser.add_argument("path", type=str, help="Path to the timeline JSON file.")
    args = parser.parse_args()
    timeline = read_timeline(args.path)
    print(timeline)
    for ts in [
        '2025-07-29T23:23:56.000-04:00',
        '2025-07-28T14:17:14.000-04:00',
    ]:
        print(f"\nFinding semantic at {ts}:")
        ts = ts_to_seconds(ts)
        found = timeline.semantic.find_at_time(ts)
        for item in found:
            print(f"  - {item} (t0: {item.t0}, t1: {getattr(item, 't1', 'N/A')}, inside? {item.t0 <= ts <= getattr(item, 't1', item.t0)})")
        print("Finding raw at {ts}:")
        found = timeline.raw.find_at_time(ts)
        for item in found:
            print(f"  - {item} (t0: {item.t0} vs {ts} = {item.t0 - ts})")
