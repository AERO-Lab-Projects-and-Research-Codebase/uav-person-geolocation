"""Check camera/depth alignment and absolute projection, without model weights."""
import copy
import csv
import json
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_depth_comparison import (scaled_intrinsics, prepare_depth_input, restore_depth_map,
                                  merge_manifests, export_report)
from run_reference_batch import flatten_result
from depth_backends.lingbot_depth import LingBotUniDepthBackend
from metadata_geometry import calibrated_ray, apply_capture_geometry


def capture():
    return {"width": 100, "height": 80,
            "intrinsics": {"fx": 80., "fy": 85., "cx": 49.5, "cy": 39.5},
            "distortion_k1_k2_k3": [.1, .01, .001],
            "geometry_status": "provisional", "limitations": [],
            "camera_rdf_to_ned": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "assumed_launch_ground_hae_m": 0.,
            "telemetry": {"latitude": 30., "longitude": -97., "altitude_hae": 50., "uid": "test"}}


class DepthComparisonTests(unittest.TestCase):
    def test_resize_intrinsics_preserve_pixel_centres(self):
        k = scaled_intrinsics(capture(), (40, 50, 3))
        np.testing.assert_allclose(k, [[40, 0, 24.5], [0, 42.5, 19.5], [0, 0, 1]])

    def test_rectified_pixel_map_matches_independent_calibrated_ray(self):
        c = capture()
        _, k, reverse, _ = prepare_depth_input(np.zeros((40, 50, 3), np.uint8), c)
        x, y = 38, 10
        ray = calibrated_ray((x, y), c, (40, 50, 3))
        expected = k @ ray
        np.testing.assert_allclose([reverse[0][y, x], reverse[1][y, x]], expected[:2], atol=2e-5)

    def test_depth_restore_retains_metric_units_and_invalid_pixels(self):
        c = capture()
        _, _, reverse, _ = prepare_depth_input(np.zeros((40, 50, 3), np.uint8), c)
        restored = restore_depth_map(np.full((40, 50), 42., dtype=np.float32), reverse, (40, 50, 3))
        self.assertAlmostEqual(float(restored[20, 25]), 42., places=5)
        self.assertTrue(np.isfinite(restored).all())
        self.assertTrue(np.all((restored == 0) | np.isclose(restored, 42)))
        bad = np.full((40, 50), np.nan, dtype=np.float32)
        self.assertEqual(float(restore_depth_map(bad, reverse, (40, 50, 3)).sum()), 0.)
        with self.assertRaises(ValueError):
            restore_depth_map(np.ones((20, 25)), reverse, (40, 50, 3))

    def test_axial_depth_drives_calibrated_person_gps(self):
        c = capture()
        c["distortion_k1_k2_k3"] = [0., 0., 0.]
        record = {"people": [{"depth_m": 20., "depth_mode": "axial", "sample_pixel": [49.5, 39.5],
                              "bbox": [40., 20., 60., 60.]}]}
        person = apply_capture_geometry(record, c, (80, 100, 3))["people"][0]
        self.assertEqual(person["status"], "estimated")
        self.assertAlmostEqual(person["north_m"], 20.)
        self.assertAlmostEqual(person["east_m"], 0.)
        self.assertAlmostEqual(person["slant_range_m"], 20.)
        self.assertGreater(person["latitude"], 30.)
        self.assertAlmostEqual(person["longitude"], -97.)


class FusionRowTests(unittest.TestCase):
    def test_report_emits_a_gated_fusion_row_per_person_from_saved_estimates(self):
        c = capture()
        c["distortion_k1_k2_k3"] = [0., 0., 0.]
        # Camera level, 50 m up: box bottom at row 60 is 13 deg down (plane), at row 45 only 3 deg (depth ray).
        record = apply_capture_geometry({"people": [
            {"index": 1, "confidence": .9, "depth_m": 20., "depth_mode": "axial", "sample_pixel": [49.5, 39.5], "bbox": [40., 20., 60., 60.]},
            {"index": 2, "confidence": .8, "depth_m": 20., "depth_mode": "axial", "sample_pixel": [49.5, 39.5], "bbox": [40., 20., 60., 45.]}]},
            c, (80, 100, 3))
        record.update(capture_metadata={"camera_source": "MAIN_VISIBLE", "photo_kind": "NORMAL"},
                      source={"relative_path": "a.JPG", "sha256": "x", "processed_shape": [80, 100, 3], "original_shape": [80, 100, 3]})
        saved = copy.deepcopy(record)
        for person in saved["people"]:
            del person["fusion"]  # saved runs predate the gate
        rows = list(flatten_result(saved, ""))
        self.assertEqual([r["fusion_source"] for r in rows], ["ground_plane", "depth_ray"])
        self.assertEqual((rows[0]["fusion_latitude"], rows[0]["fusion_range_m"]), (rows[0]["ground_latitude"], rows[0]["ground_range_m"]))
        self.assertEqual((rows[1]["fusion_latitude"], rows[1]["fusion_range_m"]), (rows[1]["latitude"], rows[1]["slant_range_m"]))
        with tempfile.TemporaryDirectory() as folder:
            export_report(Path(folder), {}, {"a.JPG": record})
            with (Path(folder) / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
                fusion = [r for r in csv.DictReader(handle) if r["model"] == "Fusion (YOLO26-depth + ground plane)"]
        self.assertEqual([(r["person_index"], r["estimate_point"], r["status"]) for r in fusion],
                         [("1", "bbox_bottom_centre", "estimated"), ("2", "torso", "estimated")])
        self.assertEqual(fusion[0]["latitude"], str(rows[0]["ground_latitude"]))


class LingBotSeedTests(unittest.TestCase):
    class Seed:
        def __init__(self):
            self.calls, self.loaded = [], False

        def load(self):
            self.loaded = True

        def predict(self, frame, intrinsics=None):
            self.calls.append((frame.shape, intrinsics))
            return {"depth": np.full(frame.shape[:2], 12., np.float32), "metadata": {"backend": "unidepth_v2", "runtime_ms": 5.}}

    class Refiner:
        def __init__(self):
            self.calls, self.loaded = [], False

        def load(self):
            self.loaded = True

        def predict(self, frame, seed, intrinsics=None):
            self.calls.append((seed.copy(), intrinsics))
            return {"depth": seed * 1.5, "metadata": {"backend": "lingbot_depth", "runtime_ms": 7.,
                                                      "independent_depth_estimate": True}}

    def test_unidepth_depth_seeds_lingbot_with_same_image_and_intrinsics(self):
        seed, refiner = self.Seed(), self.Refiner()
        backend = LingBotUniDepthBackend(seed_backend=seed, refiner=refiner).load()
        self.assertTrue(seed.loaded and refiner.loaded)
        k = np.eye(3, dtype=np.float32)
        result = backend.predict(np.zeros((6, 8, 3), np.uint8), intrinsics=k)
        np.testing.assert_allclose(refiner.calls[0][0], 12.)
        self.assertIs(refiner.calls[0][1], k)
        np.testing.assert_allclose(result["depth"], 18.)
        meta = result["metadata"]
        self.assertEqual(meta["backend"], "lingbot_depth")
        self.assertFalse(meta["independent_depth_estimate"])
        self.assertEqual(meta["seed"]["backend"], "unidepth_v2")
        self.assertEqual(meta["runtime_ms"], 12.)


class ManifestMergeTests(unittest.TestCase):
    def test_single_model_run_keeps_other_completed_models_for_the_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            baseline = {"a.JPG": {}}
            detection = {"run_signature": "detector"}
            def save(name, **extra):
                (output / name).mkdir()
                manifest = {"status": "complete", "selection": ["a.JPG"], "detector_run_signature": "detector", **extra}
                (output / name / "manifest.json").write_text(json.dumps(manifest))
                return manifest
            save("unidepth_v2")
            save("depth_pro", detector_run_signature="other")
            fresh = {"status": "complete", "marker": "fresh"}
            merged = merge_manifests(output, baseline, detection, {"lingbot_depth": fresh})
            self.assertEqual(list(merged), ["unidepth_v2", "lingbot_depth"])
            self.assertIs(merged["lingbot_depth"], fresh)


if __name__ == "__main__":
    unittest.main()
