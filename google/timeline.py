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

from dataclasses import dataclass
from enum import Enum
from typing import Any


GPS = tuple[float, float]  # Latitude, Longitude

@dataclass
class Estimate:
    prob: float

@dataclass
class Place(Estimate):
    prob: float

@dataclass
class TimePoint:
    ts0: float

@dataclass
class SemanticSegment(TimePoint):
    ts1: float

@dataclass
class Path(SemanticSegment):
    points: list[tuple[GPS, float]]  # List of tuples (GPS, timestamp in seconds since epoch)

class ActivityType(Enum):
    UNKNOWN = "UNKNOWN"
    STILL = "STILL"
    IN_VEHICLE = "IN_VEHICLE"
    IN_RAIL_VEHICLE = "IN_RAIL_VEHICLE"
    ON_FOOT = "ON_FOOT"
    WALKING = "WALKING"
    IN_ROAD_VEHICLE = "IN_ROAD_VEHICLE"
    ON_BICYCLE = "ON_BICYCLE"
    IN_BUS = "IN_BUS"
    IN_SUBWAY = "IN_SUBWAY"
    IN_TRAIN = "IN_TRAIN"
    FLYING = "FLYING"

class PlaceType(Enum):
    UNKNOWN = "UNKNOWN"
    HOME = "HOME"
    WORK = "WORK"
    SHOPPING = "SHOPPING"
    RESTAURANT = "RESTAURANT"
    PARK = "PARK"
    OTHER = "OTHER"

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

class SourceType(Enum):
    WIFI = "WIFI"
    CELLULAR = "CELLULAR"
    GPS = "GPS"
    UNKNOWN = "UNKNOWN"

@dataclass
class RawPosition(TimePoint):
    source: SourceType
    loc: GPS
    accuracy: float
    altitude: float
    speed: float

@dataclass
class WifiScan(TimePoint):
    devices: list[tuple[int, int]] # List of tuples (MAC address, RSSI)

@dataclass
class ActivityRecord(TimePoint):
    activities: list[tuple[ActivityType, float]  # List of tuples (ActivityType, probability)
