"""Offline analytic checks; no models, dashboard, network or hardware needed."""
import json
import math
import unittest

import numpy as np

from human_geolocation import estimate_ground_plane_position, estimate_person_position, fuse_estimates


class HumanGeolocationTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = dict(latitude=0, longitude=0, altitude_hae=20,
                              sensor_hfov=90, sensor_vfov=90,
                              sensor_azimuth=0, sensor_elevation=0, sensor_roll=0,
                              uid="camera")
        self.depth = np.full((100, 100), 10.0)
        # Torso ROI y=40..60 puts its sample on the camera principal point.
        self.box = [40, 180 / 7, 60, 580 / 7]

    def estimate(self, **kwargs):
        return estimate_person_position(self.box, self.depth, self.telemetry, (100, 100, 3), **kwargs)

    def test_cardinal_directions(self):
        for az, north, east in ((0, 10, 0), (90, 0, 10), (180, -10, 0), (270, 0, -10)):
            with self.subTest(az=az):
                self.telemetry["sensor_azimuth"] = az
                result = self.estimate()
                self.assertEqual(result["status"], "estimated")
                self.assertAlmostEqual(result["north_m"], north, places=8)
                self.assertAlmostEqual(result["east_m"], east, places=8)
                self.assertAlmostEqual(result["down_m"], 0, places=8)

    def test_zero_origin_is_explicit_and_wgs84_radius(self):
        result = self.estimate()
        self.assertEqual(result["origin_mode"], "assumed_zero_origin")
        self.assertAlmostEqual(result["latitude"], math.degrees(10 / (6335439.32729282 + 20)), places=10)
        self.assertEqual(result["uid"], "camera")

    def test_nonzero_origin_and_dateline_wrap(self):
        self.telemetry.update(latitude=42, longitude=179.99999, sensor_azimuth=90)
        result = self.estimate()
        self.assertEqual(result["origin_mode"], "telemetry_origin")
        self.assertLess(result["longitude"], -179.99)

    def test_absent_and_null_gps_are_not_zero(self):
        for invalid in (None, float("nan"), float("inf"), "", False):
            with self.subTest(invalid=invalid):
                self.telemetry["latitude"] = invalid
                result = self.estimate()
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["latitude"])
                json.dumps(result, allow_nan=False)
        del self.telemetry["latitude"]
        self.assertEqual(self.estimate()["reason"], "missing_or_nonfinite_gps_origin")

    def test_invalid_gps_and_pole(self):
        self.telemetry["longitude"] = 181
        self.assertEqual(self.estimate()["reason"], "gps_origin_out_of_bounds")
        self.telemetry.update(longitude=0, latitude=90)
        self.assertIn("pole", self.estimate()["reason"])

    def test_right_pixel_moves_east_and_range_normalizes(self):
        self.box[0], self.box[2] = 80, 100
        axial, rayrange = self.estimate(), self.estimate(depth_mode="range")
        self.assertAlmostEqual(axial["north_m"], 10)
        self.assertAlmostEqual(axial["east_m"], 8)
        self.assertAlmostEqual(axial["slant_range_m"], math.sqrt(164))
        self.assertAlmostEqual(rayrange["slant_range_m"], 10)
        self.assertAlmostEqual(rayrange["east_m"] / rayrange["north_m"], .8)

    def test_elevation_positive_up(self):
        self.telemetry["sensor_elevation"] = 30
        result = self.estimate()
        self.assertAlmostEqual(result["north_m"], math.sqrt(75))
        self.assertAlmostEqual(result["down_m"], -5)
        self.assertAlmostEqual(result["altitude_hae"], 25)

    def test_downward_camera_uses_depth_not_hae_as_floor(self):
        self.telemetry["sensor_elevation"] = -90
        result = self.estimate()
        self.assertAlmostEqual(result["down_m"], 10)
        self.assertAlmostEqual(result["altitude_hae"], 10)
        self.assertAlmostEqual(result["north_m"], 0)

    def test_clockwise_roll_moves_right_pixel_down(self):
        self.box[0], self.box[2] = 80, 100
        self.telemetry["sensor_roll"] = 90
        result = self.estimate()
        self.assertAlmostEqual(result["east_m"], 0)
        self.assertAlmostEqual(result["down_m"], 8)

    def test_missing_attitude_and_invalid_fov(self):
        for key, value in (("sensor_azimuth", None), ("sensor_elevation", 91),
                           ("sensor_hfov", 180), ("sensor_vfov", 0),
                           ("sensor_roll", float("inf"))):
            with self.subTest(key=key):
                original = self.telemetry[key]
                self.telemetry[key] = value
                self.assertEqual(self.estimate()["status"], "unavailable")
                self.telemetry[key] = original

    def test_missing_roll_is_labelled_and_missing_altitude_is_unknown(self):
        del self.telemetry["sensor_roll"]
        del self.telemetry["altitude_hae"]
        result = self.estimate()
        self.assertEqual(result["status"], "estimated")
        self.assertIsNone(result["altitude_hae"])
        self.assertIn("missing_sensor_roll_assumed_zero", result["assumptions"])

    def test_depth_preserved_when_gps_or_attitude_missing(self):
        for key in ("latitude", "longitude", "sensor_azimuth", "sensor_elevation"):
            with self.subTest(key=key):
                original = self.telemetry.pop(key)
                result = self.estimate()
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["latitude"])
                self.assertEqual(result["depth_m"], 10)
                self.assertAlmostEqual(result["slant_range_m"], 10)
                self.assertIsNotNone(result["sample_pixel"])
                self.assertIsNotNone(result["sample_roi"])
                self.telemetry[key] = original

    def test_depth_preserved_when_fov_or_all_telemetry_missing(self):
        del self.telemetry["sensor_hfov"]
        result = self.estimate()
        self.assertEqual(result["reason"], "missing_or_invalid_sensor_fov")
        self.assertEqual(result["depth_m"], 10)
        self.assertIsNone(result["slant_range_m"])
        self.telemetry = {}
        result = self.estimate()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["depth_m"], 10)
        self.assertIsNotNone(result["sample_roi"])

    def test_depth_median_rejects_outlier_and_sensor_range_unused(self):
        self.depth[45, 49] = 99
        self.telemetry["sensor_range"] = 100
        self.assertEqual(self.estimate()["depth_m"], 10)

    def test_invalid_depth_and_bad_map_json_safe(self):
        for invalid in (0, -1, np.nan, np.inf, 101):
            with self.subTest(invalid=invalid):
                self.depth[:] = invalid
                result = self.estimate()
                self.assertEqual(result["reason"], "insufficient_valid_torso_depth")
                json.dumps(result, allow_nan=False)
        self.depth = None
        self.assertEqual(self.estimate()["reason"], "invalid_depth_map")

    def test_slant_range_bound_at_fov_edge(self):
        self.box[0], self.box[2] = 80, 100
        self.depth[:] = 90
        self.assertEqual(self.estimate()["reason"], "slant_range_exceeds_limit")

    def test_depth_scale_and_bad_config(self):
        self.assertAlmostEqual(self.estimate(depth_scale=.1)["north_m"], 1)
        for options in ({"depth_mode": "unknown"}, {"depth_scale": -1}, {"max_depth_m": 1001}):
            self.assertEqual(self.estimate(**options)["status"], "unavailable")

    def test_lower_resolution_depth_uses_actual_patch_center(self):
        self.depth = np.full((20, 20), 10.0)
        result = self.estimate()
        x1, y1, x2, y2 = result["sample_roi"]
        self.assertEqual(result["sample_pixel"], [(x1 + x2) / 2 - .5, (y1 + y2) / 2 - .5])
        self.assertAlmostEqual(result["north_m"], 10)

    def test_invalid_box(self):
        self.box = [110, 10, 120, 90]
        self.assertEqual(self.estimate()["reason"], "empty_or_too_small_bbox")
        self.box = [0, 1, np.nan, 99]
        self.assertEqual(self.estimate()["reason"], "invalid_frame_shape_or_bbox")


class GroundPlaneTests(unittest.TestCase):
    """Ray/ground-plane intersection from pose only; no depth model involved."""

    def setUp(self):
        self.telemetry = dict(latitude=0, longitude=0, altitude_hae=30,
                              sensor_hfov=90, sensor_vfov=90,
                              sensor_azimuth=0, sensor_elevation=-45, sensor_roll=0,
                              uid="camera")
        # Feet pixel (box bottom centre) lands on the 100x100 principal point.
        self.box = [40, 10, 60, 50]
        self.ground = 20.0

    def estimate(self, **kwargs):
        kwargs.setdefault("ground_hae", self.ground)
        return estimate_ground_plane_position(self.box, self.telemetry, (100, 100, 3), **kwargs)

    def test_centre_ray_at_45_degrees(self):
        result = self.estimate()
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["north_m"], 10)
        self.assertAlmostEqual(result["east_m"], 0)
        self.assertAlmostEqual(result["down_m"], 10)
        self.assertAlmostEqual(result["ground_range_m"], math.sqrt(200))
        self.assertAlmostEqual(result["height_above_ground_m"], 10)
        self.assertAlmostEqual(result["depression_deg"], 45)
        self.assertEqual(result["altitude_hae"], 20)
        self.assertEqual(result["ground_hae_m"], 20)
        self.assertEqual(result["sample_pixel"], [49.5, 49.5])
        self.assertIn("position_refers_to_box_bottom_centre_assumed_on_ground", result["assumptions"])
        json.dumps(result, allow_nan=False)

    def test_nadir_and_azimuth(self):
        self.telemetry["sensor_elevation"] = -90
        result = self.estimate()
        self.assertAlmostEqual(result["north_m"], 0)
        self.assertAlmostEqual(result["east_m"], 0)
        self.assertAlmostEqual(result["ground_range_m"], 10)
        self.telemetry.update(sensor_elevation=-45, sensor_azimuth=90)
        result = self.estimate()
        self.assertAlmostEqual(result["east_m"], 10)
        self.assertAlmostEqual(result["north_m"], 0)

    def test_centre_pixel_ignores_zoom_but_fov_scales_offcentre_pixel(self):
        wide = self.estimate()
        self.telemetry.update(sensor_hfov=40, sensor_vfov=40)
        self.assertAlmostEqual(self.estimate()["north_m"], wide["north_m"])
        # Nadir, feet 25 px right of centre: 90 deg FOV gives fx=50 -> 5 m east;
        # tan(hfov/2)=0.5 gives fx=100 -> 2.5 m east.
        self.telemetry["sensor_elevation"] = -90
        self.box[0], self.box[2] = 65, 85
        self.telemetry.update(sensor_hfov=90, sensor_vfov=90)
        self.assertAlmostEqual(self.estimate()["east_m"], 5)
        self.telemetry.update(sensor_hfov=2 * math.degrees(math.atan(.5)), sensor_vfov=90)
        self.assertAlmostEqual(self.estimate()["east_m"], 2.5)

    def test_rays_at_or_above_horizon_never_reach_ground(self):
        for elevation in (0, 10):
            with self.subTest(elevation=elevation):
                self.telemetry["sensor_elevation"] = elevation
                result = self.estimate()
                self.assertEqual(result["reason"], "ray_does_not_reach_ground")
                self.assertIsNone(result["latitude"])

    def test_ground_height_and_aircraft_altitude_requirements(self):
        self.assertEqual(self.estimate(ground_hae=None)["reason"], "ground_height_unknown")
        self.assertEqual(self.estimate(ground_hae=float("nan"))["reason"], "ground_height_unknown")
        self.assertEqual(self.estimate(ground_hae=30)["reason"], "aircraft_not_above_ground_plane")
        self.assertEqual(self.estimate(ground_hae=35)["reason"], "aircraft_not_above_ground_plane")
        del self.telemetry["altitude_hae"]
        self.assertEqual(self.estimate()["reason"], "missing_aircraft_altitude")

    def test_grazing_ray_exceeds_range_limit(self):
        self.telemetry["sensor_elevation"] = -1
        result = self.estimate(max_range_m=100)
        self.assertEqual(result["reason"], "ground_range_exceeds_limit")
        self.assertGreater(result["ground_range_m"], 500)

    def test_shares_gps_fov_and_attitude_failures(self):
        for key, reason in (("latitude", "missing_or_nonfinite_gps_origin"),
                            ("sensor_hfov", "missing_or_invalid_sensor_fov"),
                            ("sensor_elevation", "missing_or_invalid_sensor_orientation")):
            with self.subTest(key=key):
                original = self.telemetry.pop(key)
                result = self.estimate()
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["reason"], reason)
                self.telemetry[key] = original
        self.box = [110, 10, 120, 90]
        self.assertEqual(self.estimate()["reason"], "empty_or_too_small_bbox")

    def test_wgs84_offsets_match_depth_estimator(self):
        self.telemetry.update(latitude=42, longitude=-84)
        result = self.estimate()
        self.assertEqual(result["origin_mode"], "telemetry_origin")
        self.assertAlmostEqual(result["latitude"], 42 + math.degrees(10 / (6364033.4 + 20)), places=6)
        self.assertAlmostEqual(result["longitude"], -84, places=9)


class FuseEstimatesTests(unittest.TestCase):
    DEPTH = {"status": "estimated", "latitude": 1.0, "longitude": 2.0, "altitude_hae": 3.0,
             "north_m": 10.0, "east_m": 0.0, "down_m": 5.0, "slant_range_m": 11.2}
    # plane 16 m against the ray's 10 m: a moderate disagreement (ratio 0.62, gap 0.38)
    GROUND = {"status": "estimated", "latitude": 1.1, "longitude": 2.1, "altitude_hae": 0.0,
              "north_m": 16.0, "east_m": 0.0, "down_m": 6.0, "ground_range_m": 17.1, "depression_deg": 30.0}

    def test_steep_ray_takes_ground_plane_and_carries_disagreement(self):
        fused = fuse_estimates(self.DEPTH, self.GROUND)
        self.assertEqual((fused["status"], fused["source"], fused["range_m"], fused["latitude"]), ("estimated", "ground_plane", 17.1, 1.1))
        self.assertEqual(fused["disagreement_m"], 6.0)

    def test_moderate_disagreement_needs_15_degrees(self):
        self.assertEqual(fuse_estimates(self.DEPTH, {**self.GROUND, "depression_deg": 12.0})["source"], "depth_ray")

    def test_gross_disagreement_trusts_plane_down_to_5_degrees(self):
        gross = {**self.GROUND, "north_m": 30.0, "ground_range_m": 30.6}  # ray/plane 0.33
        self.assertEqual(fuse_estimates(self.DEPTH, {**gross, "depression_deg": 7.0})["source"], "ground_plane")
        self.assertEqual(fuse_estimates(self.DEPTH, {**gross, "depression_deg": 4.0})["source"], "depth_ray")

    def test_agreeing_estimates_are_averaged(self):
        close = {**self.GROUND, "north_m": 12.0, "latitude": 1.2, "ground_range_m": 13.4}  # gap 0.17
        fused = fuse_estimates(self.DEPTH, close)
        self.assertEqual((fused["source"], fused["north_m"], fused["latitude"], fused["range_m"]), ("mean", 11.0, 1.1, 11.0))
        self.assertEqual(fuse_estimates(self.DEPTH, {**close, "depression_deg": 8.0})["source"], "depth_ray")

    def test_stated_pitch_uncertainty_replaces_the_averaging_floor(self):
        close = {**self.GROUND, "north_m": 12.0, "latitude": 1.2, "ground_range_m": 13.4, "depression_deg": 2.0}
        # 0.5 deg over sin(2) cos(2) is a 25% range error, inside the 30% tolerance; 1 deg is 50% and is not
        self.assertEqual(fuse_estimates(self.DEPTH, {**close, "pitch_sigma_deg": 0.5})["source"], "mean")
        self.assertEqual(fuse_estimates(self.DEPTH, {**close, "pitch_sigma_deg": 1.0})["source"], "depth_ray")
        self.assertEqual(fuse_estimates(self.DEPTH, {**close, "pitch_sigma_deg": 0.5, "depression_deg": 0.0})["source"], "depth_ray")

    def test_shallow_ray_takes_depth_ray(self):
        fused = fuse_estimates(self.DEPTH, {**self.GROUND, "depression_deg": 4.0})
        self.assertEqual((fused["source"], fused["range_m"], fused["latitude"]), ("depth_ray", 11.2, 1.0))

    def test_unavailable_ground_plane_takes_depth_ray(self):
        fused = fuse_estimates(self.DEPTH, {"status": "unavailable", "reason": "ray_does_not_reach_ground"})
        self.assertEqual((fused["source"], fused["disagreement_m"]), ("depth_ray", None))

    def test_neither_estimate_reports_both_reasons(self):
        fused = fuse_estimates({"status": "unavailable", "reason": "depth_missing"}, {**self.GROUND, "depression_deg": 4.0})
        self.assertEqual(fused["status"], "unavailable")
        self.assertEqual(fused["reason"], "ground_plane:ray_below_min_depression;depth_ray:depth_missing")
        self.assertIsNone(fused["latitude"])



if __name__ == "__main__":
    unittest.main()
