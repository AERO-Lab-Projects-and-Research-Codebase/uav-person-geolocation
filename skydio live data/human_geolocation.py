"""Estimate the position of a visible human from a Skydio frame plus CoT pose.

Two independent estimators share the same camera model, attitude rotation and
WGS84 conversion, so that their disagreement isolates the depth source:

``estimate_person_position`` uses metric monocular depth. This is an estimate,
not a GPS measurement of the person. Depth must be registered to the complete
input frame (possibly at lower resolution), in metres after ``depth_scale``.
The default depth convention is axial camera Z; select ``range`` for a model
whose values are Euclidean camera-to-surface distances. Verify that convention
and metric scale against stationary, measured targets before using the output.
The estimate refers to the sampled torso patch, not the person's feet.

``estimate_ground_plane_position`` intersects the pixel ray through the bottom
centre of the box with a horizontal plane at a known ground ellipsoid height.
It uses no depth network, so it is unaffected by depth-model scale or zoom-
induced depth error; it depends on the aircraft HAE, the ground height, the
CoT attitude and the CoT FOV, and it assumes the feet pixel lies on that plane.

Camera coordinates are right/down/forward. CoT azimuth is assumed clockwise
from true north, elevation positive up, and roll positive clockwise looking
along the optical axis. CoT attitude and FOV must describe the streamed camera.
WGS84 positions use a first-order local tangent conversion, limited to 1 km.
"""

import math
from collections.abc import Mapping

import numpy as np


_SHARED_ASSUMPTIONS = [
    "sensor_fov_and_attitude_describe_streamed_camera",
    "azimuth_true_north_clockwise_elevation_positive_up",
    "roll_positive_clockwise_looking_along_optical_axis",
    "wgs84_first_order_local_tangent_conversion_max_1km",
]


def _finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _frame_and_box(bbox, frame_shape):
    """Return ``(h, w, x1, y1, x2, y2)`` clamped to the frame, or a failure reason."""
    try:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        box = [_finite(value) for value in bbox]
    except (TypeError, ValueError, IndexError, OverflowError):
        return None, "invalid_frame_shape_or_bbox"
    if h <= 1 or w <= 1 or len(box) != 4 or any(value is None for value in box):
        return None, "invalid_frame_shape_or_bbox"
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, "empty_or_too_small_bbox"
    return (h, w, x1, y1, x2, y2), None


def _pixel_ray(u, v, w, h, telemetry):
    """Unit-Z camera ray ``[x, y, 1]`` for pixel-index coordinates, or a reason."""
    hfov = _finite(telemetry.get("sensor_hfov"))
    vfov = _finite(telemetry.get("sensor_vfov"))
    if hfov is None or vfov is None or not (0 < hfov < 179 and 0 < vfov < 179):
        return None, "missing_or_invalid_sensor_fov"
    fx = w / (2 * math.tan(math.radians(hfov) / 2))
    fy = h / (2 * math.tan(math.radians(vfov) / 2))
    return np.array([(u - (w - 1) / 2) / fx, (v - (h - 1) / 2) / fy, 1.0]), None


def _gps_origin(telemetry):
    """Return ``(lat, lon, origin_mode, note)`` or ``(None, None, None, reason)``."""
    lat, lon = _finite(telemetry.get("latitude")), _finite(telemetry.get("longitude"))
    if lat is None or lon is None:
        return None, None, None, "missing_or_nonfinite_gps_origin"
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None, None, None, "gps_origin_out_of_bounds"
    if abs(lat) >= 89.9:
        return None, None, None, "origin_too_close_to_pole_for_local_tangent_conversion"
    mode = "assumed_zero_origin" if lat == 0 and lon == 0 else "telemetry_origin"
    note = "zero_gps_used_as_0_0_origin_at_user_request_not_actual_location" if mode == "assumed_zero_origin" else None
    return lat, lon, mode, note


def _attitude(telemetry):
    """Return ``(az, el, roll, note)`` in radians, or ``(None, None, None, reason)``."""
    az = _finite(telemetry.get("sensor_azimuth"))
    el = _finite(telemetry.get("sensor_elevation"))
    if az is None or el is None or not -90 <= el <= 90:
        return None, None, None, "missing_or_invalid_sensor_orientation"
    raw_roll = telemetry.get("sensor_roll")
    roll = _finite(raw_roll)
    note = None
    if raw_roll is None:
        roll = 0.0
        note = "missing_sensor_roll_assumed_zero"
    elif roll is None:
        return None, None, None, "nonfinite_sensor_roll"
    az, el, roll = map(math.radians, (az % 360, el, roll % 360))
    return az, el, roll, note


def _camera_to_ned(vector, az, el, roll):
    """Rotate a right/down/forward camera vector into north/east/down."""
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    down = np.array([math.sin(el) * math.cos(az), math.sin(el) * math.sin(az), math.cos(el)])
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
    right_rolled = math.cos(roll) * right + math.sin(roll) * down
    down_rolled = -math.sin(roll) * right + math.cos(roll) * down
    return vector[0] * right_rolled + vector[1] * down_rolled + vector[2] * forward


def _origin_hae(telemetry, assumptions):
    """Aircraft HAE used for curvature; out-of-bounds or missing values become None."""
    hae = _finite(telemetry.get("altitude_hae"))
    if hae is not None and not -12000 <= hae <= 100000:
        hae = None
        assumptions.append("out_of_bounds_origin_altitude_ignored")
    if hae is None:
        assumptions.append("origin_altitude_unknown_surface_ellipsoid_used_for_horizontal_conversion")
    return hae


def _ned_to_wgs84(lat, lon, hae, north, east):
    """First-order local tangent offset on the WGS84 ellipsoid. Returns (lat, lon) or None."""
    # WGS84 meridional and prime-vertical radii. HAE adjusts curvature only
    # when present; unknown HAE does not suppress an otherwise valid lat/lon.
    a, e2 = 6378137.0, 6.6943799901413165e-3
    lat_r = math.radians(lat)
    denom = 1 - e2 * math.sin(lat_r) ** 2
    meridional = a * (1 - e2) / denom ** 1.5
    prime_vertical = a / math.sqrt(denom)
    height = hae if hae is not None else 0.0
    target_lat = lat + math.degrees(north / (meridional + height))
    target_lon = (lon + math.degrees(east / ((prime_vertical + height) * math.cos(lat_r))) + 180) % 360 - 180
    if not -90 <= target_lat <= 90:
        return None
    return target_lat, target_lon


def estimate_person_position(
    bbox, depth_map, telemetry, frame_shape, *, depth_mode="axial",
    depth_scale=1.0, max_depth_m=100.0,
):
    """Return a JSON-safe estimate for one ``(x1,y1,x2,y2)`` person box.

    ``status`` is ``estimated`` or ``unavailable``; ``reason`` explains failures.
    Explicit latitude=longitude=0 is honored as an assumed origin. Absent,
    null, nonfinite, and out-of-range origins do not become zero. A missing
    altitude permits horizontal coordinates but leaves target HAE unknown.
    Missing roll alone is allowed, with an explicit zero-roll assumption.
    Depth sampling is independent of telemetry: a missing GPS/attitude/FOV
    leaves position unavailable while retaining the valid depth and sample ROI.
    """
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    assumptions = [
        "monocular_metric_depth_is_an_estimate_not_measured_person_range",
        "depth_registered_to_full_video_frame",
        *_SHARED_ASSUMPTIONS,
        "position_refers_to_sampled_torso_not_feet",
    ]
    result = {
        "status": "unavailable", "reason": None,
        "latitude": None, "longitude": None, "altitude_hae": None,
        "north_m": None, "east_m": None, "down_m": None,
        "offsets_m": None, "depth_m": None, "slant_range_m": None,
        "sample_pixel": None, "sample_roi": None, "depth_roi": None,
        "valid_depth_fraction": None, "origin_mode": "unavailable",
        "uid": str(telemetry["uid"]) if telemetry.get("uid") is not None else None,
        "depth_mode": str(depth_mode), "assumptions": assumptions,
    }

    def fail(reason):
        result["reason"] = reason
        return result

    if depth_mode not in ("axial", "range"):
        return fail("unsupported_depth_mode")
    scale, limit = _finite(depth_scale), _finite(max_depth_m)
    if scale is None or scale <= 0 or limit is None or not 0 < limit <= 1000:
        return fail("invalid_depth_scale_or_limit")
    assumptions.append("depth_values_are_" + ("axial_camera_z" if depth_mode == "axial" else "euclidean_ray_range"))

    geometry, reason = _frame_and_box(bbox, frame_shape)
    if reason:
        return fail(reason)
    h, w, x1, y1, x2, y2 = geometry
    try:
        depth = np.asarray(depth_map, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        return fail("invalid_depth_map")
    if depth.ndim != 2 or min(depth.shape) < 1:
        return fail("invalid_depth_map")
    dh, dw = depth.shape

    # Use the central upper-middle body rather than edges, feet or background.
    torso = (x1 + .35 * (x2 - x1), y1 + .25 * (y2 - y1),
             x1 + .65 * (x2 - x1), y1 + .60 * (y2 - y1))
    dx1 = min(dw - 1, max(0, int(math.floor(torso[0] * dw / w))))
    dy1 = min(dh - 1, max(0, int(math.floor(torso[1] * dh / h))))
    dx2 = min(dw, max(dx1 + 1, int(math.ceil(torso[2] * dw / w))))
    dy2 = min(dh, max(dy1 + 1, int(math.ceil(torso[3] * dh / h))))
    with np.errstate(over="ignore", invalid="ignore"):
        patch = depth[dy1:dy2, dx1:dx2] * scale
    valid = np.isfinite(patch) & (patch > 0) & (patch <= limit)
    fraction = float(np.count_nonzero(valid) / patch.size)
    result["valid_depth_fraction"] = fraction
    result["depth_roi"] = [dx1, dy1, dx2, dy2]
    # Express the actual rounded sample region in video pixel coordinates.
    result["sample_roi"] = [dx1 * w / dw, dy1 * h / dh, dx2 * w / dw, dy2 * h / dh]
    u, v = (dx1 + dx2) * w / (2 * dw) - .5, (dy1 + dy2) * h / (2 * dh) - .5
    result["sample_pixel"] = [u, v]
    if fraction < .5:
        return fail("insufficient_valid_torso_depth")
    distance = float(np.median(patch[valid]))
    result["depth_m"] = distance

    ray, reason = _pixel_ray(u, v, w, h, telemetry)
    if reason:
        return fail(reason)
    camera = ray * distance if depth_mode == "axial" else ray / np.linalg.norm(ray) * distance
    slant = float(np.linalg.norm(camera))
    result["slant_range_m"] = slant
    if slant > min(1000.0, limit):
        return fail("slant_range_exceeds_limit")

    lat, lon, origin_mode, note = _gps_origin(telemetry)
    if lat is None:
        return fail(note)
    result["origin_mode"] = origin_mode
    if note:
        assumptions.append(note)

    az, el, roll, note = _attitude(telemetry)
    if az is None:
        return fail(note)
    if note:
        assumptions.append(note)
    north, east, down_offset = map(float, _camera_to_ned(camera, az, el, roll))

    hae = _origin_hae(telemetry, assumptions)
    projected = _ned_to_wgs84(lat, lon, hae, north, east)
    if projected is None:
        return fail("projected_latitude_out_of_bounds")
    result.update(
        status="estimated", reason=None, latitude=projected[0], longitude=projected[1],
        altitude_hae=hae - down_offset if hae is not None else None,
        north_m=north, east_m=east, down_m=down_offset,
        offsets_m={"north": north, "east": east, "down": down_offset},
    )
    return result


def estimate_ground_plane_position(bbox, telemetry, frame_shape, *, ground_hae, max_range_m=1000.0):
    """Intersect the box's bottom-centre pixel ray with a flat plane at ``ground_hae``.

    Requires aircraft HAE above the plane and a ray pointing below the horizon.
    ``ground_range_m`` is the camera-to-ground-point distance along the ray and
    ``depression_deg`` the ray's angle below horizontal; shallow angles amplify
    attitude and altitude error. Returns the same failure reasons as the depth
    estimator for GPS, FOV and attitude problems. Never uses depth or zoom scale.
    """
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    assumptions = [
        "ray_intersects_flat_horizontal_plane_at_ground_hae",
        "position_refers_to_box_bottom_centre_assumed_on_ground",
        "aircraft_hae_and_ground_hae_share_the_same_datum",
        *_SHARED_ASSUMPTIONS,
    ]
    result = {
        "status": "unavailable", "reason": None,
        "latitude": None, "longitude": None, "altitude_hae": None,
        "north_m": None, "east_m": None, "down_m": None, "offsets_m": None,
        "ground_range_m": None, "height_above_ground_m": None,
        "depression_deg": None, "ground_hae_m": _finite(ground_hae),
        "pitch_sigma_deg": _finite(telemetry.get("sensor_pitch_sigma_deg")),  # pose quality, for the fusion gate
        "sample_pixel": None, "origin_mode": "unavailable",
        "uid": str(telemetry["uid"]) if telemetry.get("uid") is not None else None,
        "assumptions": assumptions,
    }

    def fail(reason):
        result["reason"] = reason
        return result

    limit = _finite(max_range_m)
    if limit is None or not 0 < limit <= 1000:
        return fail("invalid_range_limit")
    geometry, reason = _frame_and_box(bbox, frame_shape)
    if reason:
        return fail(reason)
    h, w, x1, y1, x2, y2 = geometry
    u, v = (x1 + x2) / 2 - .5, y2 - .5
    result["sample_pixel"] = [u, v]
    ground = result["ground_hae_m"]
    if ground is None:
        return fail("ground_height_unknown")

    ray, reason = _pixel_ray(u, v, w, h, telemetry)
    if reason:
        return fail(reason)
    lat, lon, origin_mode, note = _gps_origin(telemetry)
    if lat is None:
        return fail(note)
    result["origin_mode"] = origin_mode
    if note:
        assumptions.append(note)
    hae = _finite(telemetry.get("altitude_hae"))
    if hae is None or not -12000 <= hae <= 100000:
        return fail("missing_aircraft_altitude")
    az, el, roll, note = _attitude(telemetry)
    if az is None:
        return fail(note)
    if note:
        assumptions.append(note)

    height = hae - ground
    result["height_above_ground_m"] = height
    if height <= 0:
        return fail("aircraft_not_above_ground_plane")
    direction = _camera_to_ned(ray, az, el, roll)
    direction = direction / np.linalg.norm(direction)
    result["depression_deg"] = float(math.degrees(math.asin(max(-1.0, min(1.0, direction[2])))))
    if direction[2] <= 1e-6:
        return fail("ray_does_not_reach_ground")
    scale = height / direction[2]
    result["ground_range_m"] = float(scale)
    if scale > limit:
        return fail("ground_range_exceeds_limit")
    north, east, down_offset = map(float, direction * scale)

    projected = _ned_to_wgs84(lat, lon, hae, north, east)
    if projected is None:
        return fail("projected_latitude_out_of_bounds")
    result.update(
        status="estimated", reason=None, latitude=projected[0], longitude=projected[1],
        altitude_hae=ground, north_m=north, east_m=east, down_m=down_offset,
        offsets_m={"north": north, "east": east, "down": down_offset},
    )
    return result


def horizontal_disagreement_m(first, second):
    """Horizontal metres between two estimates' NED offsets, or None if either is missing."""
    values = [_finite(item.get(key)) if isinstance(item, Mapping) else None
              for item in (first, second) for key in ("north_m", "east_m")]
    if any(value is None for value in values):
        return None
    return float(math.hypot(values[0] - values[2], values[1] - values[3]))


# ponytail: these five constants were chosen on the 149-detection Skydio set (reference good to ~20 m) and the
# 12-detection flight 4 (good to 1-2 m); re-tune them once surveyed references exist.
FUSION_MIN_DEPRESSION_DEG = 10.0       # averaging floor when the pose's pitch uncertainty is unknown (= 2.9 deg sigma)
FUSION_GROSS_DEPRESSION_DEG = 5.0      # plane trusted this close to the horizon when the depth ray is off by > 2x
FUSION_MODERATE_DEPRESSION_DEG = 15.0  # plane needs this much when the two disagree but not grossly
FUSION_AGREE_TOL = 0.3                 # relative range gap under which the two estimates are averaged
FUSION_GROSS_RATIO = 2.0               # depth/plane range ratio beyond which the depth scale is taken as wrong


def _horizontal_range(item, range_key):
    north, east = _finite(item.get("north_m")), _finite(item.get("east_m"))
    return math.hypot(north, east) if north is not None and east is not None else _finite(item.get(range_key))


def fuse_estimates(depth, ground):
    """Combine the depth-ray and ground-plane estimates by how steep the plane ray is and how far they agree.

    A depth model's error is a scale bias on every ray; the plane's is small on steep rays and unbounded near the
    horizon. So: when the two ranges agree within ``FUSION_AGREE_TOL`` and the plane is steady enough, their mean
    (independent errors partly cancel). The plane is steady when its predicted relative range error, the pose's
    ``pitch_sigma_deg`` over sin and cos of the depression, is within the same tolerance; a pose with no stated
    uncertainty falls back to the ``FUSION_MIN_DEPRESSION_DEG`` floor. When they disagree by more than ``FUSION_GROSS_RATIO`` the depth
    scale is the suspect and the plane is taken down to ``FUSION_GROSS_DEPRESSION_DEG``; when they disagree
    moderately the plane needs ``FUSION_MODERATE_DEPRESSION_DEG``; otherwise the depth ray. ``source`` names the
    choice (``ground_plane``, ``depth_ray`` or ``mean``) and ``disagreement_m`` is carried for callers.
    """
    depth = depth if isinstance(depth, Mapping) else {}
    ground = ground if isinstance(ground, Mapping) else {}
    depression = _finite(ground.get("depression_deg"))
    depth_ok, plane_ok = depth.get("status") == "estimated", ground.get("status") == "estimated" and depression is not None
    ray_m, plane_m = _horizontal_range(depth, "slant_range_m"), _horizontal_range(ground, "ground_range_m")
    source = None
    if depth_ok and plane_ok and ray_m and plane_m:
        gap, ratio = abs(ray_m - plane_m) / max(ray_m, plane_m), ray_m / plane_m
        sigma = _finite(ground.get("pitch_sigma_deg"))
        if sigma is not None and 0 < depression < 90:  # range = h cot(d), so dR/R = d_pitch / (sin d cos d)
            steady = math.radians(sigma) / (math.sin(math.radians(depression)) * math.cos(math.radians(depression))) <= FUSION_AGREE_TOL
        else:
            steady = depression >= FUSION_MIN_DEPRESSION_DEG
        if gap <= FUSION_AGREE_TOL:
            source = "mean" if steady else "depth_ray"
        else:
            needed = FUSION_GROSS_DEPRESSION_DEG if (ratio < 1 / FUSION_GROSS_RATIO or ratio > FUSION_GROSS_RATIO) else FUSION_MODERATE_DEPRESSION_DEG
            source = "ground_plane" if depression >= needed else "depth_ray"
    elif plane_ok and depression >= FUSION_MODERATE_DEPRESSION_DEG:  # nothing cross-checks a lone plane
        source = "ground_plane"
    elif depth_ok:
        source = "depth_ray"
    keys = ("latitude", "longitude", "altitude_hae", "north_m", "east_m", "down_m")
    result = {"status": "unavailable", "reason": None, "source": source, "depression_deg": depression,
              "disagreement_m": horizontal_disagreement_m(depth, ground), **{key: None for key in keys}, "range_m": None}
    if source is None:
        ground_reason = ground.get("reason") or ("ray_below_min_depression" if ground.get("status") == "estimated" else None)
        result["reason"] = f"ground_plane:{ground_reason};depth_ray:{depth.get('reason')}"
        return result
    if source == "mean":
        pairs = [(depth.get(key), ground.get(key)) for key in keys] + [(ray_m, plane_m)]
        values = [(a + b) / 2 if _finite(a) is not None and _finite(b) is not None else (a if _finite(a) is not None else b) for a, b in pairs]
        result.update(status="estimated", **dict(zip(keys + ("range_m",), values)))
        return result
    chosen, range_key = (ground, "ground_range_m") if source == "ground_plane" else (depth, "slant_range_m")
    result.update(status="estimated", range_m=chosen.get(range_key), **{key: chosen.get(key) for key in keys})
    return result
