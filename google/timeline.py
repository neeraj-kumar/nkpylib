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

import json

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import pytz

from nkpylib.google.constants import BACKUPS_DIR

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
class Place(Estimate):
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

TimeT = TypeVar("TimeT", bound=TimePoint)

class TimeSortedLst(Generic[TimeT]):
    """A class to keep a list of items sorted by their timestamp."""
    def __init__(self, items: list[TimeT]):
        self.items = sorted(items, key=lambda x: x.t0) #FIXME what about t1?

    def __repr__(self):
        return f"TSList<{len(self.items)} items>"

@dataclass
class Timeline:
    semantic: TimeSortedLst[SemanticSegment]
    raw: TimeSortedLst[RawSignal]

def ts_to_seconds(ts: str) -> float:
    """Convert a timestamp string (with tz) to seconds since epoch."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    dt = dt.astimezone(pytz.utc)  # Convert to UTC
    return dt.timestamp()  # Return seconds since epoch

def parse_gps(gps_str: str) -> GPS:
    """Parse a GPS string in the format 'lat,lon' into a tuple of floats."""
    # remove degree symbols and whitespace
    gps_str = gps_str.replace('°', '').replace(' ', '')
    lat, lon = map(float, gps_str.split(','))
    return (lat, lon)

def read_timeline(path: str) -> Timeline:
    """Reads the timeline json from `path` and converts to Timeline object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(data.keys())
    semantic_segments: list[SemanticSegment] = []
    counts: Counter = Counter()
    for obj in data['semanticSegments']:
        start_ts = ts_to_seconds(obj['startTime'])
        end_ts = ts_to_seconds(obj['endTime'])
        if 'timelinePath' in obj:
            points = [(parse_gps(p['point']), ts_to_seconds(p['time'])) for p in obj['timelinePath']]
            semantic_segments.append(TimelinePath(t0=start_ts, t1=end_ts, points=points))
            counts['timelinePath'] += 1
        elif 'activity' in obj:
            a = obj['activity']
            semantic_segments.append(Activity(
                t0=start_ts,
                t1=end_ts,
                start=parse_gps(a['start']['latLng']),
                end=parse_gps(a['end']['latLng']),
                distance=a.get('distanceMeters', 0.0),
                type=ActivityType(a['topCandidate']['type']),
                prob=a['topCandidate'].get('probability', 0.0),
            ))
            counts['activity'] += 1
            #counts[f'activity-{a["topCandidate"]["type"]}'] += 1
        elif 'visit' in obj:
            visit = obj['visit']
            semantic_segments.append(Visit(
                t0=start_ts,
                t1=end_ts,
                level=visit['hierarchyLevel'],
                place_id=visit['topCandidate']['placeId'],
                type=PlaceType(visit['topCandidate']['semanticType']),
                prob=visit.get('probability', 0.0),
                loc=parse_gps(visit['topCandidate']['placeLocation']['latLng']),
                place_prob=visit['topCandidate'].get('probability', 0.0),
            ))
            counts['visit'] += 1
            #counts[f'place-{visit["topCandidate"]["semanticType"]}'] += 1
        elif 'timelineMemory' in obj: # very few of these, so probably don't matter
            mem = obj['timelineMemory']['trip']
            dst = mem['destinations'][0]['identifier']
            #print(json.dumps(obj, indent=2))
            semantic_segments.append(TimelineMemory(
                t0=start_ts,
                t1=end_ts,
                place_id=dst['placeId'],
                distance=mem.get('distanceFromOriginKms', 0.0)*1000.0,
            ))
            counts['timelineMemory'] += 1
        else:
            raise NotImplementedError(f"Unknown semantic segment type: {obj}")
    print(f'Got {len(semantic_segments)} semantic segments: {json.dumps(dict(counts), indent=2, sort_keys=True)}')
    counts.clear()
    raw_signals: list[RawSignal] = []
    for obj in data['rawSignals']:
        if 'position' in obj:
            pos = obj['position']
            raw_signals.append(RawPosition(
                t0=ts_to_seconds(pos['timestamp']),
                source=SourceType(pos['source']),
                loc=parse_gps(pos['LatLng']),
                accuracy=pos.get('accuracyMeters', 0.0),
                altitude=pos.get('altitudeMeters', 0.0),
                speed=pos.get('speedMetersPerSecond', 0.0),
            ))
            counts['position'] += 1
        elif 'wifiScan' in obj:
            scan = obj['wifiScan']
            devices = [(d['mac'], d['rawRssi']) for d in scan['devicesRecords']]
            raw_signals.append(WifiScan(
                t0=ts_to_seconds(scan['deliveryTime']),
                devices=devices,
            ))
            counts['wifiScan'] += 1
        elif 'activityRecord' in obj:
            act_rec = obj['activityRecord']
            activities = [(ActivityType(a['type']), a['confidence']) for a in act_rec['probableActivities']]
            raw_signals.append(ActivityRecord(
                t0=ts_to_seconds(act_rec['timestamp']),
                activities=activities,
            ))
            counts['activityRecord'] += 1
        else:
            raise NotImplementedError(f"Unknown raw signal type: {obj}")
    print(f'Got {len(raw_signals)} raw signals: {json.dumps(dict(counts), indent=2, sort_keys=True)}')
    return Timeline(
        semantic=TimeSortedLst(semantic_segments),
        raw=TimeSortedLst(raw_signals),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Read Google Timeline JSON file.")
    parser.add_argument("path", type=str, help="Path to the timeline JSON file.")
    args = parser.parse_args()
    timeline = read_timeline(args.path)
    print(timeline)
