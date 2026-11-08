"""Pose -> the capture record used by metadata_geometry, and person placement on top of it."""
import math

from metadata_geometry import apply_capture_geometry, euler_rdf_to_ned
from person_geolocator import ROOT
from terrain import TerrainModel, apply_terrain_geometry

REQUIRED = ("lat", "lon", "heading_deg", "pitch_deg", "hfov_deg")
TERRAIN_PATH = ROOT / "reference_validation" / "terrain" / "site_dem.npz"
_terrain = []


def build_capture(pose, width, height):
    """Pinhole capture (no distortion) from the pose form; unavailable if a required field is bad."""
    errors = [f"missing_{key}" for key in REQUIRED if pose.get(key) is None]
    if not errors:
        if not 0 < pose["hfov_deg"] < 179:
            errors.append("invalid_hfov_deg")
        if not -90 <= pose["pitch_deg"] <= 90:
            errors.append("invalid_pitch_deg")
        if not (-89.9 < pose["lat"] < 89.9 and -180 <= pose["lon"] <= 180):
            errors.append("invalid_lat_lon")
    altitude, above = pose.get("altitude_hae"), pose.get("height_above_ground_m")
    fx = None if errors else width / (2 * math.tan(math.radians(pose["hfov_deg"]) / 2))
    return {
        "width": width, "height": height,
        "intrinsics": {"fx": fx, "fy": fx, "cx": (width - 1) / 2, "cy": (height - 1) / 2},
        "distortion_k1_k2_k3": [0., 0., 0.],
        "geometry_status": "unavailable" if errors else "provisional", "geometry_errors": errors,
        "limitations": ["Pose from metadata or manual entry; no surveyed validation.",
                        "No lens distortion correction for this camera."],
        "telemetry": {"latitude": pose.get("lat"), "longitude": pose.get("lon"),
                      "altitude_hae": 0.0 if altitude is None else altitude, "uid": None},
        "assumed_launch_ground_hae_m": None if altitude is None or above is None else altitude - above,
        "camera_rdf_to_ned": None if errors else euler_rdf_to_ned(
            pose.get("roll_deg") or 0.0, pose["pitch_deg"], pose["heading_deg"]).tolist(),
    }


def terrain_for(capture):
    """The site DEM if the camera lies inside it, else None."""
    t = capture["telemetry"]
    if capture["geometry_status"] != "provisional" or not TERRAIN_PATH.is_file():
        return None
    if not _terrain:
        _terrain.append(TerrainModel.load(TERRAIN_PATH))
    return _terrain[0] if _terrain[0].height_hae(t["latitude"], t["longitude"]) is not None else None


def _method(item, range_key):
    return {"status": item["status"], "reason": item.get("reason"), "lat": item.get("latitude"),
            "lon": item.get("longitude"), "range_m": item.get(range_key)}


def place_people(people, capture, shape, max_depth_m=100., depth_map=None):
    """Depth-ray, ground-plane and fusion positions; with the depth map, also the ground-scaled depth ray."""
    result = apply_capture_geometry({"people": [dict(p, depth_mode="axial") for p in people]},
                                    capture, shape, max_depth_m=max_depth_m, depth_map=depth_map)
    terrain = terrain_for(capture)
    terrain_items = apply_terrain_geometry(result, capture, shape, terrain) if terrain else None
    placed = []
    for i, person in enumerate(result["people"]):
        methods = {"depth_ray": _method(person, "slant_range_m"),
                   "ground_plane": _method(person["ground_plane"], "ground_range_m"),
                   "fusion": dict(_method(person["fusion"], "range_m"), source=person["fusion"]["source"])}
        if terrain_items:
            methods["terrain"] = _method(terrain_items[i], "ground_range_m")
        if depth_map is not None:
            methods["ground_scaled"] = _method(person["ground_scaled"], "slant_range_m")
        placed.append({**people[i], "methods": methods, "disagreement_m": person.get("estimate_disagreement_m")})
    return placed
