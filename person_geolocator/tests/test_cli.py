import contextlib
import io
import unittest

from person_geolocator import ROOT
from person_geolocator.cli import main

DJI = str(ROOT / "geolocation_experiment_image" / "Images" / "Human_15m.JPG")


@unittest.skipUnless((ROOT / "geolocation_experiment_image" / "Images" / "Human_15m.JPG").exists(), "lab photograph not distributed")
class CliTests(unittest.TestCase):
    def test_overrides_merge_over_metadata(self):
        seen = {}

        def fake(path, model, pose):
            seen.update(model=model, pose=pose)
            return {"people": [], "geometry_status": "provisional", "geometry_errors": [], "annotated_jpeg": b"x", "seconds": 0.0}

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([DJI, "--model", "depth_pro", "--lat", "34.5", "--lon", "-84.5", "--heading", "90"], locate_fn=fake)
        self.assertEqual(code, 0)
        self.assertEqual(seen["model"], "depth_pro")
        self.assertEqual(seen["pose"]["lat"], 34.5)
        self.assertEqual(seen["pose"]["heading_deg"], 90.0)          # override
        self.assertAlmostEqual(seen["pose"]["pitch_deg"], 2.5)        # from metadata
        self.assertIn("0 person", out.getvalue())


if __name__ == "__main__":
    unittest.main()
