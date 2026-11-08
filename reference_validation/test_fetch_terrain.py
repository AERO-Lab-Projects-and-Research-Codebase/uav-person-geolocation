"""Pure helpers of the terrain download; no network."""
import json
import math
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import fetch_terrain
from fetch_terrain import camera_points, export_params, geoid_params, grid_size, site_bounds, main


class FetchHelperTests(unittest.TestCase):
    def test_bounds_add_margin_in_metres(self):
        west, south, east, north = site_bounds([(34.387, -84.455), (34.388, -84.454)], 600.)
        self.assertAlmostEqual((north-34.388)*111_320., 600., delta=1.)
        self.assertAlmostEqual((-84.454-east)*-111_320.*math.cos(math.radians(34.3875)), 600., delta=1.)
        self.assertLess(west, -84.455)
        self.assertLess(south, 34.387)

    def test_grid_size_is_about_one_metre_and_capped(self):
        width, height = grid_size(site_bounds([(34.387, -84.455)], 600.), 1.)
        self.assertAlmostEqual(width, 1200, delta=2)
        self.assertAlmostEqual(height, 1200, delta=2)
        with self.assertRaisesRegex(ValueError, "terrain_grid_too_large"):
            grid_size(site_bounds([(34.387, -84.455)], 5000.), 1.)

    def test_request_parameters(self):
        params = export_params((-84.46, 34.38, -84.45, 34.39), (900, 1100))
        self.assertEqual((params["bboxSR"], params["imageSR"]), (4326, 4326))
        self.assertEqual(params["size"], "900,1100")
        self.assertEqual((params["format"], params["pixelType"], params["f"]), ("tiff", "F32", "json"))
        self.assertEqual(geoid_params(34.387, -84.455)["model"], 14)

    def test_camera_points_read_saved_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/"results").mkdir()
            (run/"manifest.json").write_text(json.dumps({"records": [{"result_json": "results/a.json"},
                                                                    {"result_json": "results/b.json"}]}))
            (run/"results"/"a.json").write_text(json.dumps({"capture_metadata": {"telemetry": {"latitude": 34.1, "longitude": -84.2}}}))
            (run/"results"/"b.json").write_text(json.dumps({"capture_metadata": {"telemetry": {"latitude": None, "longitude": None}}}))
            self.assertEqual(camera_points(run), [(34.1, -84.2)])


class OverwriteGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/"site_dem.npz"
        self.output.write_bytes(b"existing terrain")

        def no_network(*args, **kwargs):
            raise AssertionError("network must not be touched")

        for target, replacement in (("_get", no_network),
                                    ("camera_points", lambda run: [(34.387, -84.455)])):
            patcher = mock.patch.object(fetch_terrain, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_existing_output_is_refused_without_force_and_before_any_request(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--output", str(self.output)])
        self.assertIn("--force", str(caught.exception))
        self.assertEqual(self.output.read_bytes(), b"existing terrain")

    def test_print_only_still_works_when_the_output_exists(self):
        main(["--output", str(self.output), "--print-only"])
        self.assertEqual(self.output.read_bytes(), b"existing terrain")

    def test_force_reaches_the_download_step(self):
        with mock.patch.object(fetch_terrain, "download", side_effect=RuntimeError("download reached")):
            with self.assertRaisesRegex(RuntimeError, "download reached"):
                main(["--output", str(self.output), "--force"])


if __name__ == "__main__":
    unittest.main()
