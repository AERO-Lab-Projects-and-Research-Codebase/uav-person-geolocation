import unittest

from person_geolocator import ROOT
from person_geolocator.metadata import FIELDS, read_pose

DJI = ROOT / "geolocation_experiment_image" / "Images" / "Human_15m.JPG"
SKYDIO = ROOT / "Geolocation Reference Data" / "S1014277.JPG"


class MetadataTests(unittest.TestCase):
    @unittest.skipUnless(DJI.exists(), "lab photograph not distributed")
    def test_dji_partial_pose(self):
        pose = read_pose(DJI)
        self.assertEqual(pose["source"], "dji")
        self.assertEqual(set(pose["fields"]), set(FIELDS))
        f = pose["fields"]
        self.assertIsNone(f["lat"])  # DJI wrote 0,0 indoors
        self.assertIsNone(f["lon"])
        self.assertAlmostEqual(f["altitude_hae"], 287.98)
        self.assertAlmostEqual(f["heading_deg"], -57.9)  # FlightYaw -58.9 + body-relative gimbal yaw 1.0
        self.assertAlmostEqual(f["pitch_deg"], 2.5)
        self.assertAlmostEqual(f["height_above_ground_m"], 0.7)
        self.assertAlmostEqual(f["hfov_deg"], 71.6, delta=0.1)  # 4000x2250 crop of the 4:3 sensor, 24 mm equivalent
        self.assertTrue(any("lat" in w for w in pose["warnings"]))
        self.assertIsNone(pose["capture"])

    @unittest.skipUnless(SKYDIO.exists(), "local reference data absent")
    def test_skydio_full_pose(self):
        pose = read_pose(SKYDIO)
        self.assertEqual(pose["source"], "skydio")
        f = pose["fields"]
        self.assertAlmostEqual(f["lat"], 34.3874, places=3)
        self.assertAlmostEqual(f["hfov_deg"], 79.09, places=1)
        self.assertAlmostEqual(f["height_above_ground_m"], 9.34, places=1)
        self.assertIsNotNone(pose["capture"])


if __name__ == "__main__":
    unittest.main()
