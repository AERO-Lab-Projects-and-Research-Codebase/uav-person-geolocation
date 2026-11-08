"""Terrain run reads a saved detection run and writes only to its own folder."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from run_terrain_ground import main, run
from test_terrain import capture, synthetic_terrain


class TerrainRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.detections, self.output, self.dem = root/"photos", root/"terrain_photos", root/"dem.npz"
        (self.detections/"results").mkdir(parents=True)
        saved = {"run_signature": "det", "capture_metadata": capture(),
                 "source": {"relative_path": "photo.JPG", "sha256": "abc", "processed_shape": [512, 640, 3]},
                 "people": [{"index": 1, "bbox": [280, 200, 344, 256], "bbox_original": [1120, 800, 1376, 1024]}]}
        (self.detections/"results"/"photo_1.json").write_text(json.dumps(saved))
        (self.detections/"manifest.json").write_text(json.dumps(
            {"status": "complete", "run_signature": "det",
             "records": [{"source": "photo.JPG", "result_json": "results/photo_1.json"}]}))
        synthetic_terrain(lambda n, e: 343.).save(self.dem)  # 3 m above the launch plane

    def digest(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.detections.rglob("*") if p.is_file()}

    def test_absolute_mode_writes_separate_signed_results(self):
        before = self.digest()
        manifest = run(self.detections, self.dem, self.output)
        self.assertEqual(self.digest(), before)
        self.assertEqual((manifest["status"], manifest["detector_run_signature"]), ("complete", "det"))
        self.assertAlmostEqual(manifest["datum_check"]["median_m"], -3., places=2)
        saved = json.loads((self.output/"results"/"photo_1.json").read_text())
        self.assertEqual(saved["run_signature"], manifest["run_signature"])
        self.assertEqual(saved["source"]["sha256"], "abc")
        person = saved["people"][0]
        self.assertEqual(person["status"], "estimated")
        self.assertAlmostEqual(person["north_m"], 7., places=2)  # camera 7 m above terrain at 45 deg
        with (self.output/"terrain_ground.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual((rows[0]["source"], rows[0]["status"]), ("photo.JPG", "estimated"))

    def test_nadir_anchor_moves_terrain_to_launch_plane(self):
        run(self.detections, self.dem, self.output, vertical_mode="nadir_anchor")
        person = json.loads((self.output/"results"/"photo_1.json").read_text())["people"][0]
        self.assertAlmostEqual(person["north_m"], 10., places=2)
        self.assertAlmostEqual(person["vertical_offset_m"], -3., places=2)

    def test_incomplete_detection_run_rejected(self):
        manifest = json.loads((self.detections/"manifest.json").read_text())
        manifest["status"] = "running"
        (self.detections/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not complete"):
            run(self.detections, self.dem, self.output)

    def test_output_may_not_overwrite_or_nest_with_the_detection_run(self):
        before = self.digest()
        for output in (self.detections, self.detections/"results", self.detections/"new_terrain",
                       self.detections/".."/"photos", self.detections.parent):
            with self.subTest(output=str(output)), self.assertRaisesRegex(ValueError, "separate from the detection run"):
                run(self.detections, self.dem, output)
        self.assertEqual(self.digest(), before)

    def test_existing_non_terrain_output_is_refused(self):
        self.output.mkdir()
        (self.output/"manifest.json").write_text(json.dumps({"status": "complete", "run_signature": "x"}))
        with self.assertRaisesRegex(ValueError, "not a terrain"):
            run(self.detections, self.dem, self.output)
        self.assertEqual(json.loads((self.output/"manifest.json").read_text()), {"status": "complete", "run_signature": "x"})

    def test_rerun_into_existing_terrain_output_still_works(self):
        first = run(self.detections, self.dem, self.output)
        second = run(self.detections, self.dem, self.output)
        self.assertEqual((second["status"], second["run_signature"]), ("complete", first["run_signature"]))

    def test_main_rejects_unsafe_output_names(self):
        def must_not_run(*args, **kwargs):
            raise AssertionError("run() must not be reached for an unsafe --output")

        for name in (str(self.output), "../terrain_x", "sub/terrain_x", "photos"):
            with self.subTest(name=name), mock.patch("run_terrain_ground.run", must_not_run),                     self.assertRaises(SystemExit):
                main(["--run", "photos", "--terrain", str(self.dem), "--output", name])


if __name__ == "__main__":
    unittest.main()
