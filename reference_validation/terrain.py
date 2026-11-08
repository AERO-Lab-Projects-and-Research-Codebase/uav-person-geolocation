"""Bare-earth terrain on a WGS84 lat/lon grid, and box-bottom ray intersection.

Grid heights are NAVD88 orthometric (USGS 3DEP). geoid_height_m (GEOID18 N)
converts them to ellipsoid heights, HAE = NAVD88 + N, so they compare with
Skydio altitudes. Earth curvature is ignored (under 0.1 m within 1 km).
This sits beside the flat launch-plane estimate; it never replaces it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from metadata_geometry import _clear_position, _offset_gps, _position, calibrated_ray

NODATA_BELOW_M = -1000.


class TerrainModel:
    """Row 0 is the northern edge; bounds are outer pixel edges in degrees."""

    def __init__(self, elevation, west, south, east, north, geoid_height_m, metadata=None):
        grid = np.asarray(elevation, dtype=float)
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise ValueError("invalid_terrain_grid")
        if not (west < east and south < north):
            raise ValueError("invalid_terrain_bounds")
        if not math.isfinite(geoid_height_m):
            raise ValueError("invalid_geoid_height")
        self.elevation = np.where(np.isfinite(grid) & (grid > NODATA_BELOW_M), grid, np.nan)
        self.west, self.south, self.east, self.north = map(float, (west, south, east, north))
        self.geoid_height_m = float(geoid_height_m)
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            west, south, east, north = map(float, data["bounds_wsen"])
            return cls(data["elevation_navd88_m"], west, south, east, north,
                       float(data["geoid_height_m"]), json.loads(str(data["metadata_json"])))

    def save(self, path):
        np.savez_compressed(Path(path), elevation_navd88_m=self.elevation.astype(np.float32),
                            bounds_wsen=np.array([self.west, self.south, self.east, self.north]),
                            geoid_height_m=np.array(self.geoid_height_m),
                            metadata_json=np.array(json.dumps(self.metadata)))

    def height_hae(self, lat, lon):
        """Bilinear ellipsoid height at a point, or None outside the grid / on no data."""
        rows, cols = self.elevation.shape
        col = (lon-self.west)/(self.east-self.west)*cols-.5
        row = (self.north-lat)/(self.north-self.south)*rows-.5
        if not (0 <= col <= cols-1 and 0 <= row <= rows-1):
            return None
        c0, r0 = min(int(col), cols-2), min(int(row), rows-2)
        fc, fr = col-c0, row-r0
        g = self.elevation[r0:r0+2, c0:c0+2]
        value = (g[0, 0]*(1-fc)+g[0, 1]*fc)*(1-fr)+(g[1, 0]*(1-fc)+g[1, 1]*fc)*fr
        return float(value)+self.geoid_height_m if math.isfinite(value) else None


GRAZING_DEG = 5.
TERRAIN_LIMITATIONS = [
    "Terrain-map ground uses USGS 3DEP bare-earth heights converted with a single GEOID18 value; drone GPS altitude error shifts every hit.",
    "Shallow (grazing) terrain hits magnify small camera-angle errors into large range errors.",
]


def intersect_ray(terrain, lat, lon, hae, direction_ned, *, max_range_m=1000., step_m=.5, vertical_offset_m=0.):
    """Slant distance to the first terrain crossing along a NED ray from (lat, lon, hae)."""
    d = np.asarray(direction_ned, dtype=float)
    d = d/np.linalg.norm(d)

    def gap(s):
        """Ray height above the (offset) terrain at distance s, or None off the grid."""
        north, east, down = d*s
        ground = terrain.height_hae(*_offset_gps(lat, lon, hae, north, east))
        return None if ground is None else hae-down-(ground+vertical_offset_m)

    previous = gap(0.)
    if previous is None:
        raise ValueError("camera_outside_terrain_model")
    if previous <= 0:
        raise ValueError("camera_below_terrain")
    start = 0.
    while start < max_range_m:
        end = min(start+step_m, max_range_m)
        current = gap(end)
        if current is None:
            raise ValueError("ray_left_terrain_model")
        if current <= 0:
            for _ in range(40):
                middle = (start+end)/2
                value = gap(middle)
                if value is None:
                    raise ValueError("ray_left_terrain_model")
                start, end = (middle, end) if value > 0 else (start, middle)
            return end
        start = end
    raise ValueError("ground_range_exceeds_limit")


def incidence_deg(terrain, lat, lon, hae, point_ned, direction_ned, delta_m=1.):
    """Angle between the ray and the local terrain surface (90 = head-on, 0 = grazing)."""
    north, east = float(point_ned[0]), float(point_ned[1])
    heights = [terrain.height_hae(*_offset_gps(lat, lon, hae, north+dn, east+de))
               for dn, de in ((delta_m, 0), (-delta_m, 0), (0, delta_m), (0, -delta_m))]
    if None in heights:
        return None
    normal = np.array([-(heights[0]-heights[1])/(2*delta_m), -(heights[2]-heights[3])/(2*delta_m), 1.])
    d = np.asarray(direction_ned, dtype=float)
    ray = np.array([d[0], d[1], -d[2]])/np.linalg.norm(d)  # north, east, up
    return math.degrees(math.asin(min(1., abs(float(ray@normal))/np.linalg.norm(normal))))


def apply_terrain_geometry(result, capture, frame_shape, terrain, *, max_range_m=1000., vertical_offset_m=0.):
    """Terrain-map ground estimate per person; the flat-plane result is left untouched."""
    estimates = []
    for person in result.get("people", []):
        item = {"index": person.get("index"), "bbox": person.get("bbox"), "bbox_original": person.get("bbox_original"),
                "sample_pixel": None, "ground_hae_m": None, "ground_range_m": None, "height_above_ground_m": None,
                "depression_deg": None, "incidence_deg": None, "grazing": None,
                "vertical_offset_m": vertical_offset_m,
                "assumptions": list(capture.get("limitations", []))+TERRAIN_LIMITATIONS}
        _clear_position(item, "capture_geometry_unavailable")
        estimates.append(item)
        if capture.get("geometry_status") == "unavailable":
            continue
        try:
            if vertical_offset_m is None:
                raise ValueError("vertical_offset_unavailable")
            # Same box-bottom pixel as the flat plane in metadata_geometry.apply_capture_geometry.
            x1, y1, x2, y2 = person["bbox"]
            h, w = frame_shape[:2]
            x1, x2, y2 = max(0, x1), min(w, x2), min(h, y2)
            item["sample_pixel"] = pixel = [(x1+x2)/2-.5, y2-.5]
            direction = np.asarray(capture["camera_rdf_to_ned"])@calibrated_ray(pixel, capture, frame_shape)
            direction /= np.linalg.norm(direction)
            item["depression_deg"] = math.degrees(math.asin(float(np.clip(direction[2], -1, 1))))
            t = capture["telemetry"]
            distance = intersect_ray(terrain, t["latitude"], t["longitude"], t["altitude_hae"], direction,
                                     max_range_m=max_range_m, vertical_offset_m=vertical_offset_m)
            point = direction*distance
            angle = incidence_deg(terrain, t["latitude"], t["longitude"], t["altitude_hae"], point, direction)
            item.update(ground_range_m=float(distance), ground_hae_m=float(t["altitude_hae"]-point[2]),
                        height_above_ground_m=float(point[2]), incidence_deg=angle,
                        grazing=None if angle is None else angle < GRAZING_DEG)
            _position(item, point, capture)
        except (ValueError, TypeError, KeyError) as exc:
            _clear_position(item, str(exc))
    return estimates
