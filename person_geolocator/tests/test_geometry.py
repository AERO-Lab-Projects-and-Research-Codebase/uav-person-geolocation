import unittest

from person_geolocator.geometry import build_capture, place_people

POSE = {"lat": 34.0, "lon": -84.0, "altitude_hae": 100.0, "heading_deg": 0.0, "pitch_deg": 0.0,
        "roll_deg": 0.0, "hfov_deg": 90.0, "height_above_ground_m": 1.5}
PERSON = {"index": 1, "bbox": [450, 150, 550, 300], "depth_m": 20.0, "sample_pixel": [499.5, 249.5]}
METRES_PER_DEG_LAT = 110_900.0  # approx at 34 deg


class GeometryTests(unittest.TestCase):
    def test_level_camera_places_person_north(self):
        capture = build_capture(POSE, 1000, 500)
        self.assertEqual(capture["geometry_status"], "provisional")
        person = place_people([PERSON], capture, (500, 1000))[0]
        ray = person["methods"]["depth_ray"]
        self.assertEqual(ray["status"], "estimated")
        self.assertAlmostEqual((ray["lat"] - 34.0) * METRES_PER_DEG_LAT, 20.0, delta=0.4)
        self.assertAlmostEqual(ray["lon"], -84.0, places=5)
        ground = person["methods"]["ground_plane"]  # box bottom 50 px below centre, fx 500 -> 15 m
        self.assertEqual(ground["status"], "estimated")
        self.assertAlmostEqual((ground["lat"] - 34.0) * METRES_PER_DEG_LAT, 15.0, delta=0.4)
        self.assertNotIn("terrain", person["methods"])
        fusion = person["methods"]["fusion"]  # that box-bottom ray is only 5.7 deg down: too shallow for the plane
        self.assertEqual((fusion["status"], fusion["source"]), ("estimated", "depth_ray"))
        self.assertEqual((fusion["lat"], fusion["range_m"]), (ray["lat"], ray["range_m"]))

    def test_steep_ray_fuses_to_ground_plane(self):
        person = place_people([PERSON], build_capture({**POSE, "pitch_deg": -45.0}, 1000, 500), (500, 1000))[0]
        fusion, ground = person["methods"]["fusion"], person["methods"]["ground_plane"]
        self.assertEqual((fusion["status"], fusion["source"]), ("estimated", "ground_plane"))
        self.assertEqual((fusion["lat"], fusion["range_m"]), (ground["lat"], ground["range_m"]))

    def test_nothing_to_fuse_gives_reason(self):
        capture = build_capture({**POSE, "height_above_ground_m": None}, 1000, 500)
        fusion = place_people([{**PERSON, "depth_m": None}], capture, (500, 1000))[0]["methods"]["fusion"]
        self.assertEqual(fusion["status"], "unavailable")
        self.assertEqual(fusion["reason"], "ground_plane:ground_height_unknown;depth_ray:missing_or_invalid_torso_depth")

    def test_missing_pose_makes_positions_unavailable(self):
        capture = build_capture({**POSE, "lat": None}, 1000, 500)
        self.assertEqual(capture["geometry_status"], "unavailable")
        self.assertIn("missing_lat", capture["geometry_errors"])
        person = place_people([PERSON], capture, (500, 1000))[0]
        self.assertEqual(person["methods"]["depth_ray"]["status"], "unavailable")
        self.assertIsNone(person["methods"]["depth_ray"]["lat"])

    def test_missing_depth_keeps_ground_plane(self):
        person = place_people([{**PERSON, "depth_m": None}], build_capture(POSE, 1000, 500), (500, 1000))[0]
        self.assertEqual(person["methods"]["depth_ray"]["status"], "unavailable")
        self.assertEqual(person["methods"]["ground_plane"]["status"], "estimated")

    def test_no_height_means_no_ground_plane(self):
        capture = build_capture({**POSE, "height_above_ground_m": None}, 1000, 500)
        person = place_people([PERSON], capture, (500, 1000))[0]
        self.assertEqual(person["methods"]["ground_plane"]["status"], "unavailable")
        self.assertEqual(person["methods"]["depth_ray"]["status"], "estimated")


if __name__ == "__main__":
    unittest.main()
