"""Terrain grid and ray-terrain intersection on synthetic, analytically solvable terrain."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from metadata_geometry import _offset_gps, apply_capture_geometry, euler_rdf_to_ned
from terrain import TerrainModel, apply_terrain_geometry

LAT0, LON0 = 34., -84.


def synthetic_terrain(height_fn, half_m=600., step_m=2., geoid_height_m=0.):
    """Grid centred on (LAT0, LON0); height_fn(north_m, east_m) -> NAVD88 metres.

    Pixel centres sit exactly on the sampled metre offsets, so bilinear
    sampling reproduces any planar height_fn exactly.
    """
    count = int(round(2*half_m/step_m))+1
    top, right = _offset_gps(LAT0, LON0, 0., half_m, half_m)
    bottom, left = _offset_gps(LAT0, LON0, 0., -half_m, -half_m)
    dlat, dlon = (top-bottom)/(count-1), (right-left)/(count-1)
    rows = [[height_fn(half_m-i*step_m, -half_m+j*step_m) for j in range(count)] for i in range(count)]
    return TerrainModel(np.array(rows), left-dlon/2, bottom-dlat/2, right+dlon/2, top+dlat/2, geoid_height_m)


class TerrainModelTests(unittest.TestCase):
    def test_bilinear_sampling_of_a_plane_is_exact(self):
        terrain = synthetic_terrain(lambda n, e: 300.+.25*n-.1*e)
        lat, lon = _offset_gps(LAT0, LON0, 0., 37.3, -12.9)
        self.assertAlmostEqual(terrain.height_hae(lat, lon), 300.+.25*37.3+.1*12.9, places=2)

    def test_geoid_height_converts_navd88_to_hae(self):
        terrain = synthetic_terrain(lambda n, e: 370., geoid_height_m=-30.)
        self.assertAlmostEqual(terrain.height_hae(LAT0, LON0), 340., places=6)

    def test_outside_and_no_data_return_none(self):
        terrain = synthetic_terrain(lambda n, e: -9999. if n > 100 else 340.)
        self.assertIsNone(terrain.height_hae(*_offset_gps(LAT0, LON0, 0., 700., 0.)))
        self.assertIsNone(terrain.height_hae(*_offset_gps(LAT0, LON0, 0., 300., 0.)))
        self.assertAlmostEqual(terrain.height_hae(LAT0, LON0), 340.)

    def test_save_load_round_trip(self):
        terrain = synthetic_terrain(lambda n, e: 300.+n*.01, half_m=50., geoid_height_m=-30.4)
        terrain.metadata = {"source": "test"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"dem.npz"
            terrain.save(path)
            loaded = TerrainModel.load(path)
        self.assertEqual(loaded.metadata, {"source": "test"})
        self.assertAlmostEqual(loaded.geoid_height_m, -30.4)
        self.assertAlmostEqual(loaded.height_hae(LAT0, LON0), terrain.height_hae(LAT0, LON0), places=3)

    def test_invalid_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_terrain_bounds"):
            TerrainModel(np.zeros((3, 3)), 1., 0., 0., 1., 0.)
        with self.assertRaisesRegex(ValueError, "invalid_terrain_grid"):
            TerrainModel(np.zeros(3), 0., 0., 1., 1., 0.)


def capture(pitch=-45., yaw=0.):
    """Camera 10 m above a 340 m HAE launch plane, looking along yaw (0 = north); centre pixel is the box bottom."""
    return {"width": 640, "height": 512,
            "intrinsics": {"fx": 500., "fy": 500., "cx": 311.5, "cy": 255.5},
            "distortion_k1_k2_k3": [0., 0., 0.],
            "geometry_status": "provisional", "limitations": ["test_assumptions"],
            "telemetry": {"latitude": LAT0, "longitude": LON0, "altitude_hae": 350., "uid": "test"},
            "assumed_launch_ground_hae_m": 340.,
            "camera_rdf_to_ned": euler_rdf_to_ned(0, pitch, yaw).tolist()}


def detections():
    return {"people": [{"index": 1, "bbox": [280, 200, 344, 256], "bbox_original": [1120, 800, 1376, 1024],
                        "depth_m": 20., "depth_mode": "axial", "sample_pixel": [311.5, 255.5]}]}


SHAPE = (512, 640)


class TerrainIntersectionTests(unittest.TestCase):
    def terrain_person(self, height_fn, pitch=-45., yaw=0., **kwargs):
        terrain = synthetic_terrain(height_fn, geoid_height_m=kwargs.pop("geoid_height_m", 0.))
        return apply_terrain_geometry(detections(), capture(pitch, yaw), SHAPE, terrain, **kwargs)[0]

    def test_flat_terrain_matches_existing_flat_plane(self):
        flat = apply_capture_geometry(detections(), capture(), SHAPE)["people"][0]["ground_plane"]
        item = self.terrain_person(lambda n, e: 340.)
        self.assertEqual(item["status"], "estimated")
        self.assertAlmostEqual(item["north_m"], flat["north_m"], places=2)
        self.assertAlmostEqual(item["east_m"], flat["east_m"], places=2)
        self.assertAlmostEqual(item["north_m"], 10., places=2)
        self.assertAlmostEqual(item["ground_hae_m"], 340., places=2)
        self.assertAlmostEqual(item["incidence_deg"], 45., places=1)
        self.assertFalse(item["grazing"])
        self.assertEqual((item["index"], item["bbox"]), (1, [280, 200, 344, 256]))

    def test_rising_terrain_shortens_the_hit(self):
        # Ray: height 350 - n; terrain: 340 + 0.5 n  ->  n = 20/3.
        item = self.terrain_person(lambda n, e: 340.+.5*n)
        self.assertAlmostEqual(item["north_m"], 20/3, places=2)
        self.assertAlmostEqual(item["ground_hae_m"], 340.+10/3, places=2)

    def test_level_ray_hits_rising_hill_where_flat_plane_fails(self):
        flat = apply_capture_geometry(detections(), capture(0.), SHAPE)["people"][0]["ground_plane"]
        self.assertEqual(flat["reason"], "ray_does_not_reach_ground")
        item = self.terrain_person(lambda n, e: 340.+.5*n, pitch=0.)
        self.assertEqual(item["status"], "estimated")
        self.assertAlmostEqual(item["north_m"], 20., places=2)
        self.assertAlmostEqual(item["incidence_deg"], math.degrees(math.atan(.5)), places=1)

    def test_looking_east_uses_the_east_slope_for_the_incidence_normal(self):
        # Level ray 10 m up, heading east, meets terrain 340 + 0.5 e at e = 20; the north slope is zero.
        item = self.terrain_person(lambda n, e: 340.+.5*e, pitch=0., yaw=90.)
        self.assertEqual(item["status"], "estimated")
        self.assertAlmostEqual(item["east_m"], 20., places=2)
        self.assertAlmostEqual(item["north_m"], 0., places=2)
        self.assertAlmostEqual(item["incidence_deg"], math.degrees(math.atan(.5)), places=1)

    def test_shallow_hit_is_flagged_grazing(self):
        item = self.terrain_person(lambda n, e: 340.+.05*n, pitch=0.)
        self.assertAlmostEqual(item["north_m"], 200., places=1)
        self.assertTrue(item["grazing"])

    def test_geoid_and_vertical_offset_shift_the_surface(self):
        self.assertAlmostEqual(self.terrain_person(lambda n, e: 370., geoid_height_m=-30.)["north_m"], 10., places=2)
        self.assertAlmostEqual(self.terrain_person(lambda n, e: 340., vertical_offset_m=5.)["north_m"], 5., places=2)

    def test_failures_have_reasons(self):
        self.assertEqual(self.terrain_person(lambda n, e: 360.)["reason"], "camera_below_terrain")
        self.assertEqual(self.terrain_person(lambda n, e: 340., pitch=0.)["reason"], "ray_left_terrain_model")
        self.assertEqual(self.terrain_person(lambda n, e: 340., vertical_offset_m=None)["reason"],
                         "vertical_offset_unavailable")
        self.assertEqual(self.terrain_person(lambda n, e: 340.+.5*n, pitch=0., max_range_m=10.)["reason"],
                         "ground_range_exceeds_limit")
        bad = capture()
        bad["geometry_status"] = "unavailable"
        item = apply_terrain_geometry(detections(), bad, SHAPE, synthetic_terrain(lambda n, e: 340.))[0]
        self.assertEqual(item["reason"], "capture_geometry_unavailable")
        self.assertIsNone(item["latitude"])

    def test_input_result_is_not_modified(self):
        source = detections()
        before = copy.deepcopy(source)
        apply_terrain_geometry(source, capture(), SHAPE, synthetic_terrain(lambda n, e: 340.))
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
