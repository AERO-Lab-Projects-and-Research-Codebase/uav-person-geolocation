"""Read the camera pose from a JPEG: full Skydio XMP, or partial DJI/EXIF data."""
import math
import re
from pathlib import Path

from PIL import Image

from metadata_geometry import read_capture

FIELDS = ("lat", "lon", "altitude_hae", "heading_deg", "pitch_deg", "roll_deg", "hfov_deg", "height_above_ground_m")
FULL_FRAME_DIAGONAL_MM = 43.267


def _xmp_number(xmp, name):
    match = re.search(rf'{re.escape(name)}\s*=\s*"([-+0-9.eE]+)"', xmp)
    return float(match.group(1)) if match else None


def _exif_latlon(exif):
    gps = exif.get_ifd(0x8825)
    try:
        values = []
        for dms, ref in ((gps[2], gps[1]), (gps[4], gps[3])):
            degrees = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
            values.append(-degrees if ref in ("S", "W") else degrees)
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None
    return (None, None) if values == [0.0, 0.0] else tuple(values)  # DJI writes 0,0 without a fix


def read_pose(path):
    """Return {source, fields, warnings, capture}; capture is the Skydio record or None."""
    path = Path(path)
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
    match = re.search(rb"<x:xmpmeta\b.*?</x:xmpmeta>", path.read_bytes(), re.S)
    xmp = match.group().decode("utf-8", "replace") if match else ""
    warnings = []
    if exif.get(274, 1) != 1:
        warnings.append("EXIF orientation is not 1; pixel coordinates follow the stored, unrotated image.")
    if "CameraOrientationNED" in xmp:
        capture = read_capture(path)
        if capture["geometry_status"] == "provisional":
            t, ground = capture["telemetry"], capture["assumed_launch_ground_hae_m"]
            fields = {"lat": t["latitude"], "lon": t["longitude"], "altitude_hae": t["altitude_hae"],
                      "heading_deg": t["sensor_azimuth"], "pitch_deg": t["sensor_elevation"],
                      "roll_deg": t["sensor_roll"], "hfov_deg": t["sensor_hfov"],
                      "height_above_ground_m": None if ground is None else t["altitude_hae"] - ground}
            warnings.append("Skydio metadata read; ground height assumes the launch plane (AbsoluteAltitude minus RelativeAltitude).")
            return {"source": "skydio", "fields": fields, "warnings": warnings, "capture": capture}
        warnings += [f"Skydio metadata unusable: {e}" for e in capture["geometry_errors"]]
    lat, lon = _exif_latlon(exif)
    f35 = exif.get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
    hfov = None
    if f35:
        # The 35 mm equivalent is quoted for the 4:3 sensor; a wider frame (16:9) is a crop of it with the same width.
        # ponytail: a native 3:2 sensor is also treated as 4:3 (about 2 deg too narrow); read the sensor size if one appears.
        aspect = min(width / height, 4 / 3)
        sensor_width = FULL_FRAME_DIAGONAL_MM * aspect / math.hypot(aspect, 1)
        hfov = math.degrees(2 * math.atan(sensor_width / 2 / float(f35)))
    dji = "drone-dji:" in xmp
    number = lambda name: _xmp_number(xmp, "drone-dji:" + name)
    gimbal_yaw, flight_yaw = number("GimbalYawDegree"), number("FlightYawDegree")
    heading = gimbal_yaw
    # Mini-series gimbals have no yaw axis and write GimbalYawDegree relative to the body (~0) instead of
    # absolute (~FlightYawDegree); when it sits nearer 0 than the flight yaw, add the two.
    if gimbal_yaw is not None and flight_yaw is not None and abs(gimbal_yaw) < abs((gimbal_yaw - flight_yaw + 180) % 360 - 180):
        heading = (flight_yaw + gimbal_yaw + 180) % 360 - 180
    fields = {"lat": lat, "lon": lon, "altitude_hae": number("AbsoluteAltitude"),
              "heading_deg": heading, "pitch_deg": number("GimbalPitchDegree"),
              "roll_deg": number("GimbalRollDegree"), "hfov_deg": hfov,
              "height_above_ground_m": number("RelativeAltitude")}
    if dji:
        warnings.append("DJI altitude is barometric, not ellipsoid height; HFOV comes from the 35 mm equivalent focal length; heading is the gimbal yaw (plus the flight yaw when the gimbal reports it body-relative).")
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        warnings.append("Not found in metadata, enter by hand: " + ", ".join(missing))
    return {"source": "dji" if dji else "exif", "fields": fields, "warnings": warnings, "capture": None}
