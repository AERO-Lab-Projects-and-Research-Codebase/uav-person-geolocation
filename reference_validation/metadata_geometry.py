"""Offline JPEG adapter. Estimates are provisional, never accuracy validation.

Source: https://support.skydio.com/hc/en-us/articles/20866347470491
Skydio documents pixel intrinsics, WGS84 ellipsoid altitude, FLU/NED frames,
and radial k1/k2/k3 distortion (no tangential/denominator terms). The stored
quaternion is interpreted as camera FLU -> NED and checked against its Euler
counterpart on every capture. The optical RDF ray is [right, down, forward].
Neither this consistency check nor factory calibration proves field accuracy.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skydio live data"))
from human_geolocation import fuse_estimates  # noqa: E402  the live module owns the estimator rules

SOURCE_URL = "https://support.skydio.com/hc/en-us/articles/20866347470491-Skydio-X10-camera-and-metadata-overview"
BASE_LIMITATIONS = [
    "Provisional coordinates: capture pose, intrinsics and geolocation accuracy have no surveyed validation.",
    "Monocular depth scale and axial-depth convention are unvalidated for these visible and thermal stills.",
    "The ground cross-check assumes a flat plane at AbsoluteAltitude minus RelativeAltitude; relative altitude is not terrain AGL.",
    "The depth point samples the torso; the ground point uses bounding-box bottom centre, which may not be the feet.",
    "GPS and camera pose are embedded in the photo; sensor timing and absolute alignment are not independently verified.",
]


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def quaternion_matrix(values):
    """Normalized Hamilton xyzw quaternion; reject absent/degenerate input."""
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > .01:
        raise ValueError("invalid_camera_quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_rdf_to_ned(roll, pitch, yaw):
    """Skydio Euler angles in degrees, with pitch interpreted as elevation."""
    roll, el, az = map(math.radians, (roll, pitch, yaw))
    right = np.array([-math.sin(az), math.cos(az), 0])
    down = np.array([math.sin(el)*math.cos(az), math.sin(el)*math.sin(az), math.cos(el)])
    forward = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), -math.sin(el)])
    return np.column_stack((math.cos(roll)*right + math.sin(roll)*down,
                            -math.sin(roll)*right + math.cos(roll)*down, forward))


def _offset_gps(lat, lon, hae, north, east):
    a, e2 = 6378137., 6.6943799901413165e-3
    phi = math.radians(lat)
    d = 1-e2*math.sin(phi)**2
    meridian, prime = a*(1-e2)/d**1.5, a/math.sqrt(d)
    return (lat+math.degrees(north/(meridian+hae)),
            (lon+math.degrees(east/((prime+hae)*math.cos(phi)))+180)%360-180)


def read_capture(path):
    """Read an original JPEG without altering pixels, metadata or live code."""
    path = Path(path)
    with Image.open(path) as im:
        width, height = im.size
        orientation = int(im.getexif().get(274, 1))
        zoom = float(im.getexif().get_ifd(0x8769).get(0xA404) or 1)  # EXIF DigitalZoomRatio
    match = re.search(rb"<x:xmpmeta\b.*?</x:xmpmeta>", path.read_bytes(), re.S)
    if not match:
        raise ValueError("missing_embedded_xmp")
    root = ET.fromstring(match.group())
    description = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    if description is None:
        raise ValueError("missing_xmp_description")
    def parse(node):
        return ({child.tag.split("}")[-1]: parse(child) for child in node}
                if len(node) else (node.text or "").strip())
    raw = parse(description)
    def vector(tag, keys):
        resource = raw.get(tag, {})
        return [_number(resource.get(key)) for key in keys] if isinstance(resource, dict) else [None]*len(keys)
    fx, fy = vector("CalibratedFocalLength", ["X", "Y"])
    cx, cy = vector("CalibratedOpticalCenter", ["X", "Y"])
    if zoom > 1 and None not in (fx, fy, cx, cy):
        # The calibration describes the full sensor; a digital zoom crops the central 1/zoom about the image centre and
        # resamples it to full size, scaling the focal lengths and moving the optical centre toward the middle.
        fx, fy = fx * zoom, fy * zoom
        cx, cy = (cx - width / 2) * zoom + width / 2, (cy - height / 2) * zoom + height / 2
    roll, pitch, yaw = vector("CameraOrientationNED", ["Roll", "Pitch", "Yaw"])
    lat, lon, altitude, relative = [_number(raw.get(k)) for k in
                                  ("Latitude", "Longitude", "AbsoluteAltitude", "RelativeAltitude")]
    errors, limitations = [], list(BASE_LIMITATIONS)
    if orientation != 1:
        errors.append("unsupported_exif_orientation")
    if None in (fx, fy, cx, cy) or not (fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height):
        errors.append("invalid_intrinsics")
    if None in (lat, lon, altitude) or not (-89.9 < lat < 89.9 and -180 <= lon <= 180 and -12000 <= altitude <= 100000) or (lat == 0 and lon == 0):
        errors.append("invalid_real_gps_origin")
    if None in (roll, pitch, yaw) or not -90 <= pitch <= 90:
        errors.append("invalid_camera_euler")
    distortion = [_number(v) for v in str(raw.get("DewarpData", "")).split(",")]
    if len(distortion) != 3 or any(v is None for v in distortion):
        errors.append("invalid_radial_distortion")
    if raw.get("PhotoDewarpResult") != "PHOTO_DEWARP_NONE":
        errors.append("unsupported_dewarped_image")
    rotation, rotation_error = None, None
    try:
        flu = quaternion_matrix(vector("CameraOrientationQuatNED", ["X", "Y", "Z", "W"]))
        rotation = flu[:, [1, 2, 0]] * [-1, -1, 1]
        if None not in (roll, pitch, yaw):
            rotation_error = float(np.max(np.abs(rotation-euler_rdf_to_ned(roll, pitch, yaw))))
            if rotation_error > 1e-4:
                errors.append("camera_quaternion_euler_disagreement")
    except (TypeError, ValueError):
        errors.append("invalid_camera_quaternion")
    # GPS tags describe the aircraft; use the same-frame camera-vehicle offset.
    camera, vehicle = vector("CameraPositionNED", ["X", "Y", "Z"]), vector("VehiclePositionNED", ["X", "Y", "Z"])
    offset = None
    if None not in camera + vehicle and not any(e == "invalid_real_gps_origin" for e in errors):
        offset = np.asarray(camera)-np.asarray(vehicle)
        if np.linalg.norm(offset) > 2:
            errors.append("implausible_camera_vehicle_offset")
        else:
            lat, lon = _offset_gps(lat, lon, altitude, offset[0], offset[1])
    else:
        limitations.append("Camera-to-aircraft position offset unavailable; the aircraft GPS is used as camera origin.")
    camera_altitude = altitude - float(offset[2]) if altitude is not None and offset is not None else altitude
    ground = altitude-relative if altitude is not None and relative is not None and relative > 0 else None
    telemetry = {"latitude": lat, "longitude": lon, "altitude_hae": camera_altitude,
                 "sensor_azimuth": yaw, "sensor_elevation": pitch, "sensor_roll": roll,
                 "sensor_hfov": math.degrees(2*math.atan(width/(2*fx))) if fx and fx > 0 else None,
                 "sensor_vfov": math.degrees(2*math.atan(height/(2*fy))) if fy and fy > 0 else None,
                 "uid": raw.get("VehicleId")}
    return {"path": str(path.resolve()), "width": width, "height": height,
            "exif_orientation": orientation, "digital_zoom": zoom, "camera_source": raw.get("CameraSource"),
            "photo_kind": raw.get("PhotoKind"), "capture_time": raw.get("CreateDate"),
            "capture_utime": raw.get("CaptureUtime"), "request_nonce": raw.get("RequestNonce"),
            "intrinsics": dict(fx=fx, fy=fy, cx=cx, cy=cy), "distortion_k1_k2_k3": distortion,
            "camera_rdf_to_ned": rotation.tolist() if rotation is not None else None,
            "quaternion_euler_max_matrix_error": rotation_error,
            "camera_vehicle_offset_ned_m": offset.tolist() if offset is not None else None,
            "telemetry": telemetry, "assumed_launch_ground_hae_m": ground,
            "geometry_status": "unavailable" if errors else "provisional",
            "geometry_errors": errors, "limitations": limitations,
            "metadata_source_url": SOURCE_URL, "xmp": raw}


def calibrated_ray(pixel, capture, frame_shape):
    """Inverse radial model after undoing cv2 resize's pixel-centre transform."""
    h, w = frame_shape[:2]
    if h <= 1 or w <= 1:
        raise ValueError("invalid_frame_shape")
    u, v = pixel
    if _number(u) is None or _number(v) is None:
        raise ValueError("invalid_sample_pixel")
    u = (u+.5)*capture["width"]/w-.5
    v = (v+.5)*capture["height"]/h-.5
    intr = capture["intrinsics"]
    xd, yd = (u-intr["cx"])/intr["fx"], (v-intr["cy"])/intr["fy"]
    rd = math.hypot(xd, yd)
    k1, k2, k3 = capture["distortion_k1_k2_k3"]
    r = rd
    for _ in range(30):
        r2 = r*r
        f = r*(1+k1*r2+k2*r2*r2+k3*r2*r2*r2)-rd
        derivative = 1+3*k1*r2+5*k2*r2*r2+7*k3*r2*r2*r2
        if derivative <= 0 or not math.isfinite(derivative):
            raise ValueError("radial_inverse_not_monotonic")
        step = f/derivative
        r -= step
        if r < 0 or not math.isfinite(r):
            raise ValueError("radial_inverse_failed")
        if abs(step) < 1e-12:
            break
    residual = abs(r*(1+k1*r*r+k2*r**4+k3*r**6)-rd)
    if residual > 1e-8:
        raise ValueError("radial_inverse_did_not_converge")
    factor = r/rd if rd else 1.
    return np.array([xd*factor, yd*factor, 1.])


def _clear_position(item, reason):
    item.update(status="unavailable", reason=reason, origin_mode="unavailable")
    for key in ("latitude", "longitude", "altitude_hae", "north_m", "east_m", "down_m", "offsets_m"):
        item[key] = None


def _position(item, ned, capture):
    north, east, down = map(float, ned)
    t = capture["telemetry"]
    lat, lon = _offset_gps(t["latitude"], t["longitude"], t["altitude_hae"], north, east)
    item.update(status="estimated", reason=None, geometry_status="provisional",
                latitude=lat, longitude=lon, altitude_hae=t["altitude_hae"]-down,
                north_m=north, east_m=east, down_m=down,
                offsets_m={"north": north, "east": east, "down": down},
                origin_mode="embedded_photo_gps_with_camera_offset", uid=t["uid"])


GROUND_SCALE_MIN_DEPRESSION_DEG = 15.0   # rays this steep meet the plane reliably; shallower ones are where it drifts


def ground_scale(depth_map, capture, frame_shape, boxes=(), samples=4000):
    """Factor that moves an axial depth map's ground onto the assumed ground plane, or None.

    A monocular depth model gets the shape of a scene but can misjudge its overall scale. Each sampled pixel
    whose ray points at least ``GROUND_SCALE_MIN_DEPRESSION_DEG`` below the horizon, outside the person boxes,
    implies a camera height (axial depth times the ray's downward component); the factor is the plane's height
    over the median of those. None when the plane is unknown or too little steep ground is in view.
    """
    plane = capture.get("assumed_launch_ground_hae_m")
    if plane is None or capture["geometry_status"] == "unavailable":
        return None
    height = capture["telemetry"]["altitude_hae"]-plane
    depth_map = np.asarray(depth_map, dtype=float)
    h, w = frame_shape[:2]
    if height <= 0 or depth_map.shape != (h, w):
        return None
    rotation = np.asarray(capture["camera_rdf_to_ned"])
    min_down = math.sin(math.radians(GROUND_SCALE_MIN_DEPRESSION_DEG))
    step = max(1, int(math.sqrt(h*w/samples)))
    implied = []
    for v in range(step//2, h, step):
        for u in range(step//2, w, step):
            depth = depth_map[v, u]
            if not math.isfinite(depth) or depth <= 0 or any(x1 <= u <= x2 and y1 <= v <= y2 for x1, y1, x2, y2 in boxes):
                continue
            try:
                ray = rotation@calibrated_ray((u, v), capture, frame_shape)
            except ValueError:
                continue
            if ray[2]/np.linalg.norm(ray) >= min_down:
                implied.append(depth*ray[2])
    return height/float(np.median(implied)) if len(implied) >= 100 else None


def apply_capture_geometry(result, capture, frame_shape, *, max_depth_m=100., depth_map=None):
    """Use existing torso depth samples; do not rerun or change either model.

    Limits match the live defaults (100 m monocular / 1 km ground ray). No
    surveyed ground is supplied: launch-plane intersections remain conditional.
    With ``depth_map`` (the axial depth map the torso samples came from) each
    person also gets ``ground_scaled``: the depth-ray position with the depth
    multiplied by ``ground_scale``, stored on the result, and ``scaled_fusion``:
    the fusion rule fed with that scaled ray in place of the raw one.
    """
    result = copy.deepcopy(result)
    max_depth_m = _number(max_depth_m)
    if max_depth_m is None or not 0 < max_depth_m <= 1000:
        raise ValueError("invalid_depth_limit")
    result["geometry_status"] = capture["geometry_status"]
    result["ground_hae_m"] = capture["assumed_launch_ground_hae_m"]
    result["ground_source"] = "absolute_minus_relative_altitude_assumed_launch_plane"
    result["warnings"] = list(capture["limitations"])
    result["telemetry_uid"] = capture["telemetry"].get("uid")
    result["telemetry_age_s"] = None
    scale = None
    if depth_map is not None:
        scale = ground_scale(depth_map, capture, frame_shape, [p["bbox"] for p in result.get("people", []) if p.get("bbox")])
        result["ground_scale"] = scale
    for person in result.get("people", []):
        _clear_position(person, "capture_geometry_unavailable")
        person["assumptions"] = list(capture["limitations"])
        person["estimate_disagreement_m"] = None
        ground = {"ground_hae_m": result["ground_hae_m"], "ground_range_m": None,
                  "height_above_ground_m": None, "depression_deg": None,
                  "assumptions": list(capture["limitations"])}
        _clear_position(ground, "capture_geometry_unavailable")
        person["ground_plane"] = ground
        if depth_map is not None:
            person["ground_scaled"] = {"slant_range_m": None}
            _clear_position(person["ground_scaled"], "capture_geometry_unavailable")
        if capture["geometry_status"] == "unavailable":
            continue
        rotation = np.asarray(capture["camera_rdf_to_ned"])
        vector = None
        try:
            depth = _number(person.get("depth_m"))
            if depth is None or not 0 < depth <= max_depth_m:
                raise ValueError("missing_or_invalid_torso_depth")
            if person.get("sample_pixel") is None:
                raise ValueError("missing_depth_sample_pixel")
            ray = calibrated_ray(person["sample_pixel"], capture, frame_shape)
            mode = person.get("depth_mode", "axial")
            if mode not in ("axial", "range"):
                raise ValueError("unsupported_depth_mode")
            vector = ray*depth if mode == "axial" else ray/np.linalg.norm(ray)*depth
            person["slant_range_m"] = float(np.linalg.norm(vector))
            if person["slant_range_m"] > max_depth_m:
                raise ValueError("slant_range_exceeds_limit")
            _position(person, rotation@vector, capture)
        except (ValueError, TypeError) as exc:
            _clear_position(person, str(exc))
        if depth_map is not None:
            scaled = person["ground_scaled"]
            if person["status"] != "estimated":
                _clear_position(scaled, person["reason"])
            elif scale is None:
                _clear_position(scaled, "ground_scale_unavailable")
            elif float(np.linalg.norm(vector))*scale > max_depth_m:
                _clear_position(scaled, "slant_range_exceeds_limit")
            else:
                scaled["slant_range_m"] = float(np.linalg.norm(vector))*scale
                _position(scaled, rotation@(vector*scale), capture)
        try:
            plane = ground["ground_hae_m"]
            if plane is None:
                raise ValueError("ground_height_unknown")
            height = capture["telemetry"]["altitude_hae"]-plane
            ground["height_above_ground_m"] = height
            if height <= 0:
                raise ValueError("camera_not_above_assumed_plane")
            x1, y1, x2, y2 = person["bbox"]
            h, w = frame_shape[:2]
            x1, x2, y2 = max(0, x1), min(w, x2), min(h, y2)
            pixel = [(x1+x2)/2-.5, y2-.5]
            ground["sample_pixel"] = pixel
            direction = rotation@calibrated_ray(pixel, capture, frame_shape)
            direction /= np.linalg.norm(direction)
            ground["depression_deg"] = math.degrees(math.asin(float(np.clip(direction[2], -1, 1))))
            if direction[2] <= 1e-6:
                raise ValueError("ray_does_not_reach_ground")
            distance = height/direction[2]
            ground["ground_range_m"] = float(distance)
            if distance > 1000:
                raise ValueError("ground_range_exceeds_limit")
            _position(ground, direction*distance, capture)
        except (ValueError, TypeError) as exc:
            _clear_position(ground, str(exc))
        if person["status"] == ground["status"] == "estimated":
            person["estimate_disagreement_m"] = math.hypot(person["north_m"]-ground["north_m"], person["east_m"]-ground["east_m"])
    for person in result.get("people", []):
        person["fusion"] = fuse_estimates(person, person["ground_plane"])
        if depth_map is not None:
            person["scaled_fusion"] = fuse_estimates(person["ground_scaled"], person["ground_plane"])
    return result
