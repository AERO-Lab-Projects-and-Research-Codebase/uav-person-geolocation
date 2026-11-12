"""Analytical geometry, metadata rejection and real-capture consistency checks."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from metadata_geometry import (apply_capture_geometry, calibrated_ray,
                               euler_rdf_to_ned, ground_scale, read_capture)


def fixture():
    return {"width": 640, "height": 512,
            "intrinsics": {"fx": 500., "fy": 500., "cx": 311.5, "cy": 255.5},
            "distortion_k1_k2_k3": [0., 0., 0.],
            "geometry_status": "provisional", "limitations": ["test_assumptions"],
            "telemetry": {"latitude": 34., "longitude": -84., "altitude_hae": 350., "uid": "test"},
            "assumed_launch_ground_hae_m": 340.,
            "camera_rdf_to_ned": euler_rdf_to_ned(0, -45, 0).tolist()}


def result_fixture():
    return {"people": [{"depth_m": 20., "depth_mode": "axial", "sample_pixel": [311.5, 255.5],
                        "bbox": [280, 200, 344, 256], "latitude": 0., "longitude": 0.}]}


def flat_ground_depth(capture, shape, factor=1.):
    """Axial depth of flat ground 10 m below the fixture camera, times ``factor``; 0 where the ray misses the ground."""
    rotation = np.asarray(capture["camera_rdf_to_ned"])
    depth = np.zeros(shape)
    for v in range(shape[0]):
        for u in range(shape[1]):
            down = (rotation@calibrated_ray((u, v), capture, shape))[2]
            depth[v, u] = factor*10./down if down > 0 else 0.
    return depth


class GroundScaleTests(unittest.TestCase):
    def test_scale_restores_a_depth_map_that_reads_short(self):
        capture, shape = fixture(), (64, 80)
        self.assertAlmostEqual(ground_scale(flat_ground_depth(capture, shape, .6), capture, shape), 1/.6, places=6)
        self.assertAlmostEqual(ground_scale(flat_ground_depth(capture, shape), capture, shape), 1., places=6)

    def test_no_scale_without_a_plane_or_steep_ground(self):
        capture, shape = fixture(), (64, 80)
        depth = flat_ground_depth(capture, shape)
        self.assertIsNone(ground_scale(depth, dict(capture, assumed_launch_ground_hae_m=None), shape))
        raised = dict(capture, camera_rdf_to_ned=euler_rdf_to_ned(0, 30, 0).tolist())  # pitched 30 deg up:
        self.assertIsNone(ground_scale(np.full(shape, 30.), raised, shape))            # no ray is 15 deg down
        self.assertIsNone(ground_scale(depth, capture, (32, 40)))                     # map and frame sizes differ

    def test_people_are_not_ground(self):
        capture, shape = fixture(), (64, 80)
        depth = flat_ground_depth(capture, shape)
        depth[:, :60] = 1.                                                             # a near object over most of the frame
        self.assertGreater(ground_scale(depth, capture, shape), 2.)                    # the object is taken for ground
        self.assertAlmostEqual(ground_scale(depth, capture, shape, boxes=[(0, 0, 59, 63)]), 1., places=6)

    def test_scaled_estimate_moves_the_person_along_the_same_ray(self):
        capture, shape = fixture(), (512, 640)
        small = flat_ground_depth(capture, (64, 80), .5)
        depth = np.kron(small, np.ones((8, 8)))                                        # same map at the frame size
        placed = apply_capture_geometry(result_fixture(), capture, shape, depth_map=depth)
        person = placed["people"][0]
        self.assertAlmostEqual(placed["ground_scale"], 2., delta=.05)
        self.assertEqual(person["ground_scaled"]["status"], "estimated")
        self.assertAlmostEqual(person["ground_scaled"]["north_m"], person["north_m"]*placed["ground_scale"], places=6)
        self.assertAlmostEqual(person["ground_scaled"]["slant_range_m"], person["slant_range_m"]*placed["ground_scale"], places=6)
        self.assertNotIn("ground_scaled", apply_capture_geometry(result_fixture(), capture, shape)["people"][0])

    def test_scaled_fusion_feeds_the_fusion_rule_with_the_scaled_ray(self):
        capture, shape = fixture(), (512, 640)
        depth = np.kron(flat_ground_depth(capture, (64, 80), .5), np.ones((8, 8)))    # the model reads half the true depth
        result = result_fixture()
        result["people"][0]["depth_m"] = 8.                                            # 5.7 m out raw, 11.3 m scaled; plane 10 m
        person = apply_capture_geometry(result, capture, shape, depth_map=depth)["people"][0]
        scaled, ground, fused = person["ground_scaled"], person["ground_plane"], person["scaled_fusion"]
        self.assertEqual(person["fusion"]["source"], "ground_plane")                   # raw ray too short to average
        self.assertEqual((fused["status"], fused["source"]), ("estimated", "mean"))    # scaled, the two agree and average
        self.assertAlmostEqual(fused["north_m"], (scaled["north_m"]+ground["north_m"])/2, places=9)
        self.assertAlmostEqual(fused["latitude"], (scaled["latitude"]+ground["latitude"])/2, places=12)
        self.assertNotIn("scaled_fusion", apply_capture_geometry(result_fixture(), capture, shape)["people"][0])

    def test_scaled_estimate_respects_the_depth_limit(self):
        capture, shape = fixture(), (512, 640)
        depth = np.kron(flat_ground_depth(capture, (64, 80), .1), np.ones((8, 8)))    # scale 10 puts 20 m at 200 m
        person = apply_capture_geometry(result_fixture(), capture, shape, depth_map=depth)["people"][0]
        self.assertEqual(person["status"], "estimated")
        self.assertEqual(person["ground_scaled"]["reason"], "slant_range_exceeds_limit")


class GeometryTests(unittest.TestCase):
    def test_cardinal_camera_basis_and_roll(self):
        r = euler_rdf_to_ned(0, 0, 0)
        np.testing.assert_allclose(r@[0, 0, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(r@[1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(r@[0, 1, 0], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(euler_rdf_to_ned(0, -90, 90)@[0, 0, 1], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(euler_rdf_to_ned(90, 0, 0)@[1, 0, 0], [0, 0, 1], atol=1e-12)

    def test_resize_preserves_off_centre_principal_point(self):
        capture = fixture()
        original = [481.2, 120.7]
        resized = [(original[0]+.5)*.25-.5, (original[1]+.5)*.5-.5]
        np.testing.assert_allclose(calibrated_ray(original, capture, (512, 640)),
                                   calibrated_ray(resized, capture, (256, 160)), atol=1e-12)
        np.testing.assert_allclose(calibrated_ray([311.5, 255.5], capture, (512, 640)), [0, 0, 1])

    def test_radial_inverse_recovers_known_undistorted_ray(self):
        capture = fixture()
        capture["distortion_k1_k2_k3"] = [.318731, -2.647725, 5.515893]
        x, y = .32, -.24
        r2 = x*x+y*y
        factor = 1+.318731*r2-2.647725*r2*r2+5.515893*r2**3
        pixel = [311.5+500*x*factor, 255.5+500*y*factor]
        np.testing.assert_allclose(calibrated_ray(pixel, capture, (512, 640)), [x, y, 1], atol=1e-10)

    def test_depth_and_plane_have_distinct_geometries(self):
        source = result_fixture()
        result = apply_capture_geometry(source, fixture(), (512, 640))
        person = result["people"][0]
        self.assertEqual(person["status"], "estimated")
        self.assertAlmostEqual(person["north_m"], 20/math.sqrt(2))
        self.assertAlmostEqual(person["east_m"], 0.)
        self.assertAlmostEqual(person["altitude_hae"], 350-20/math.sqrt(2))
        ground = person["ground_plane"]
        self.assertAlmostEqual(ground["north_m"], 10.)
        self.assertAlmostEqual(ground["altitude_hae"], 340.)
        self.assertGreater(person["latitude"], 34.)
        self.assertAlmostEqual(person["estimate_disagreement_m"], 20/math.sqrt(2)-10)
        self.assertEqual(source["people"][0]["latitude"], 0.)

    def test_range_depth_is_not_axial_depth(self):
        capture = fixture()
        source = result_fixture()
        source["people"][0].update(sample_pixel=[561.5, 255.5], depth_mode="range")
        person = apply_capture_geometry(source, capture, (512, 640))["people"][0]
        self.assertAlmostEqual(person["slant_range_m"], 20.)
        self.assertAlmostEqual(np.linalg.norm(list(person["offsets_m"].values())), 20.)

    def test_unknown_ground_and_horizon_are_rejected_independently(self):
        capture = fixture()
        capture["assumed_launch_ground_hae_m"] = None
        person = apply_capture_geometry(result_fixture(), capture, (512, 640))["people"][0]
        self.assertEqual(person["status"], "estimated")
        self.assertEqual(person["ground_plane"]["reason"], "ground_height_unknown")
        capture = fixture()
        capture["camera_rdf_to_ned"] = euler_rdf_to_ned(0, 0, 0).tolist()
        ground = apply_capture_geometry(result_fixture(), capture, (512, 640))["people"][0]["ground_plane"]
        self.assertEqual(ground["reason"], "ray_does_not_reach_ground")

    def test_unavailable_capture_clears_stale_coordinates(self):
        capture = fixture()
        capture["geometry_status"] = "unavailable"
        person = apply_capture_geometry(result_fixture(), capture, (512, 640))["people"][0]
        self.assertIsNone(person["latitude"])
        self.assertEqual(person["depth_m"], 20.)
        self.assertEqual(person["ground_plane"]["status"], "unavailable")

    def test_missing_metadata_does_not_invent_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"no_metadata.jpg"
            Image.new("RGB", (20, 20)).save(path)
            with self.assertRaisesRegex(ValueError, "missing_embedded_xmp"):
                read_capture(path)

    def test_configured_limit_and_nonfinite_pixel_rejected(self):
        person = apply_capture_geometry(result_fixture(), fixture(), (512, 640), max_depth_m=10)["people"][0]
        self.assertEqual(person["status"], "unavailable")
        self.assertEqual(person["ground_plane"]["status"], "estimated")
        with self.assertRaisesRegex(ValueError, "invalid_sample_pixel"):
            calibrated_ray([float("nan"), 2], fixture(), (512, 640))

    def test_incomplete_xmp_rejected_without_zero_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"incomplete.jpg"
            Image.new("RGB", (20, 20)).save(path)
            with path.open("ab") as out:
                out.write(b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:RDF><rdf:Description><foo>bar</foo></rdf:Description></rdf:RDF></x:xmpmeta>')
            capture = read_capture(path)
            self.assertEqual(capture["geometry_status"], "unavailable")
            self.assertIsNone(capture["telemetry"]["latitude"])
            self.assertIn("invalid_real_gps_origin", capture["geometry_errors"])

    def test_all_reference_captures_have_consistent_embedded_geometry(self):
        paths = list((Path(__file__).resolve().parents[1]/"Geolocation Reference Data").glob("*.JPG"))
        if not paths:
            self.skipTest("local reference data absent")
        self.assertEqual(len(paths), 120)
        for path in paths:
            capture = read_capture(path)
            with self.subTest(file=path.name):
                self.assertEqual(capture["geometry_status"], "provisional", capture["geometry_errors"])
                self.assertLess(capture["quaternion_euler_max_matrix_error"], 2e-6)
                self.assertNotEqual(capture["telemetry"]["latitude"], 0)
                self.assertGreater(capture["assumed_launch_ground_hae_m"], 300)
                self.assertEqual(capture["exif_orientation"], 1)


if __name__ == "__main__":
    unittest.main()
