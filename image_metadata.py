"""Code to get image metadata.

Currently this processes exif from JPEGs, and various gif metadata.
"""

from typing import Any, Optional, Sequence

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from PIL.GifImagePlugin import GifImageFile

def convert_to_degrees(value: Sequence[float]) -> float:
    """Helper function to convert the GPS coordinates stored in the EXIF to degrees"""
    degrees = value[0]
    minutes = value[1] / 60.0
    seconds = value[2] / 3600.0
    return degrees + minutes + seconds

def get_gps_location(exif_data) -> tuple[Optional[float], Optional[float]]:
    """Returns the latitude and longitude, if available, from the provided exif_data.

    Returns a tuple of `(latitude, longitude)`, or `(None, None)` if not found.
    """
    lat = None
    lon = None
    gps_info = exif_data.get("GPSInfo")
    if gps_info:
        gps_latitude = gps_info.get("GPSLatitude")
        gps_latitude_ref = gps_info.get("GPSLatitudeRef")
        gps_longitude = gps_info.get("GPSLongitude")
        gps_longitude_ref = gps_info.get("GPSLongitudeRef")
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = convert_to_degrees(gps_latitude)
            if gps_latitude_ref.lower() != "n":
                lat = -lat
            lon = convert_to_degrees(gps_longitude)
            if gps_longitude_ref.lower() != "e":
                lon = -lon
    return lat, lon

def get_exif_data(image) -> dict[str, Any]:
    """Returns a dictionary from the exif data of a PIL Image.

    Also converts the GPS Tags
    """
    exif_data = {}
    try:
        info = image._getexif()
        if not info:
            raise ValueError("No EXIF data found")
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[sub_decoded] = value[gps_tag]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
        # now add the gps info
    except Exception as e:
        pass
    return exif_data

def get_gif_metadata(image):
    ret = {}
    if isinstance(image, GifImageFile):
        metadata = image.info
        for key, value in metadata.items():
            ret[key] = value
    return ret

def get_image_metadata(image) -> dict[str, Any]:
    """Extracts image metadata from the given PIL `image`.

    This stores the following top-level keys:
    - raw_exif: raw exif data (with gps expansion)
    - raw_gif_info: raw gif metadata
    - latitude, longitude: if found, else None for each
    - width, height, resolution: size information
    - format, mode: output from PIL
    - is_animated, has_transparency_data, n_frames: output from PIL
    - length: in seconds
    - camera: combination of "Make" and "Model" fields in exif, or empty string
    """
    exif = get_exif_data(image)
    gif = get_gif_metadata(image)
    ret = dict(
        raw_exif=exif,
        raw_gif_info=gif,
    )
    for field in 'format mode width height is_animated has_transparency_data n_frames'.split():
        ret[field] = getattr(image, field, False)
    ret['resolution'] = ret['width'] * ret['height']
    if ret['format'] == 'MPO': # some bug with photos from my phone
        ret['n_frames'] = 1
    ret['n_frames'] = ret['n_frames'] or 1
    if ret['n_frames'] == 1:
        ret['is_animated'] = False
    if ret['is_animated'] and gif:
        ret['length'] = (gif.get('duration', 1) or 1) * 0.001 * ret['n_frames']
    else:
        ret['length'] = 0.0
    camera_vals = [exif.get('Make', ''), exif.get('Model', '')]
    ret['camera'] = ' '.join(x for x in camera_vals if x)
    ret['latitude'], ret['longitude'] = get_gps_location(ret['raw_exif'])
    return ret
