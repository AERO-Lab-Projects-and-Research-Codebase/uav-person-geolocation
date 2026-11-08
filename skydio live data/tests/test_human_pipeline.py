"""Worker behavior tests with local fake models: no network, GPU, or vehicle."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_pipeline import HumanPipeline


def wait_until(predicate, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Worker condition did not arrive before timeout")


def frame_snapshot(frame_id=1):
    return {"frame": np.zeros((100, 160, 3), dtype=np.uint8),
            "frame_id": frame_id, "captured_at": time.time()}


def telemetry(received_at):
    return {"uid": "drone-test", "_received_at": received_at,
            "latitude": 0.0, "longitude": 0.0, "altitude_hae": 17.2,
            "sensor_hfov": 78.2, "sensor_vfov": 62.7,
            "sensor_azimuth": 90, "sensor_elevation": 0, "sensor_roll": 0}


class FakeDetector:
    def __init__(self, entered=None, release=None):
        self.calls = []
        self.entered, self.release = entered, release

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(2)
        # Include one non-person to check independent class filtering.
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=np.array([[55, 10, 105, 95], [0, 0, 30, 30]], dtype=float),
            conf=np.array([0.9, 0.99]), cls=np.array([0, 1]))) ]


class FakeDepth:
    def __init__(self, bad=False):
        self.calls = []
        self.bad = bad

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.bad:
            return [SimpleNamespace(depth=None)]
        return [SimpleNamespace(depth=SimpleNamespace(
            data=np.full(kwargs["source"].shape[:2], 5.0, dtype=np.float32)))]


class HumanPipelineTests(unittest.TestCase):
    def make_pipeline(self, provider, telem=None, detector=None, depth=None, **config):
        config = {"log_dir": None, "max_fps": 100, **config}
        instance = HumanPipeline(provider, telem or (lambda stamp: telemetry(stamp)), config)
        fake_detector = detector or FakeDetector()
        fake_depth = depth or FakeDepth()

        def load():
            instance._device = "cpu"
            instance._detector = fake_detector
            instance._depth_model = fake_depth
            instance._calibration = {"status": "pretrained_metric_scale_camera_unvalidated"}

        instance._load_models = load
        self.addCleanup(instance.stop)
        return instance, fake_detector, fake_depth

    def test_zero_origin_success_person_filter_and_once_per_frame_log(self):
        snapshot = frame_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            pipeline, detector, depth = self.make_pipeline(lambda: snapshot, log_dir=directory)
            pipeline.start()
            wait_until(lambda: pipeline.get_result()["status"] == "running")
            time.sleep(0.08)
            result = pipeline.get_result()
            self.assertEqual(len(result["people"]), 1)
            person = result["people"][0]
            self.assertEqual(person["origin_mode"], "assumed_zero_origin")
            self.assertEqual(person["status"], "estimated")
            self.assertGreater(person["longitude"], 0)
            self.assertEqual(person["index"], 1)
            self.assertNotIn("track_id", person)
            self.assertEqual(len(detector.calls), 1)
            self.assertEqual(len(depth.calls), 1)
            self.assertEqual(detector.calls[0]["classes"], [0])
            self.assertEqual(detector.calls[0]["quantize"], 32)
            self.assertIsNotNone(pipeline.get_frame())
            pipeline.stop()
            log = list(Path(directory).glob("human_estimates_*.jsonl"))
            self.assertEqual(len(log), 1)
            lines = log[0].read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["frame_id"], 1)

    def test_worker_never_blocks_snapshot_read_and_drops_intermediate_frames(self):
        snapshot = [frame_snapshot(1)]
        entered, release = threading.Event(), threading.Event()
        detector = FakeDetector(entered, release)
        pipeline, _, _ = self.make_pipeline(lambda: snapshot[0], detector=detector)
        self.addCleanup(release.set)
        before = time.monotonic()
        pipeline.start()
        self.assertLess(time.monotonic() - before, 0.5)
        self.assertTrue(entered.wait(2))
        before = time.monotonic()
        self.assertIsNone(pipeline.get_frame())
        pipeline.get_result()
        self.assertLess(time.monotonic() - before, 0.1)
        snapshot[0] = frame_snapshot(2)
        snapshot[0] = frame_snapshot(3)
        release.set()
        wait_until(lambda: pipeline.get_result()["frame_id"] == 3)
        self.assertEqual(len(detector.calls), 2)

    def test_telemetry_is_snapshotted_before_inference(self):
        snapshot = frame_snapshot()
        live_telem = telemetry(snapshot["captured_at"])
        entered, release = threading.Event(), threading.Event()
        pipeline, _, _ = self.make_pipeline(lambda: snapshot, lambda stamp: live_telem,
                                             detector=FakeDetector(entered, release))
        self.addCleanup(release.set)
        pipeline.start()
        self.assertTrue(entered.wait(2))
        live_telem["uid"] = "changed-after-capture"
        live_telem["longitude"] = 120
        release.set()
        wait_until(lambda: pipeline.get_result()["status"] == "running")
        result = pipeline.get_result()
        self.assertEqual(result["telemetry_uid"], "drone-test")
        self.assertEqual(result["people"][0]["origin_mode"], "assumed_zero_origin")

    def test_stale_or_future_telemetry_retains_detection_withholds_position(self):
        for time_offset in (-10, 10):
            with self.subTest(time_offset=time_offset):
                snapshot = frame_snapshot()
                pipeline, _, _ = self.make_pipeline(
                    lambda: snapshot, lambda stamp: telemetry(stamp + time_offset))
                pipeline.start()
                wait_until(lambda: pipeline.get_result()["status"] == "running")
                result = pipeline.get_result()
                self.assertEqual(len(result["people"]), 1)
                self.assertIsNone(result["telemetry_uid"])
                self.assertIsNone(result["people"][0]["latitude"])
                self.assertTrue(any("Telemetry" in note for note in result["warnings"]))
                pipeline.stop()

    def test_stale_result_not_overlaid_on_newer_video(self):
        snapshot = frame_snapshot()
        pipeline, _, _ = self.make_pipeline(lambda: snapshot, frame_max_age_s=0.3)
        pipeline.start()
        wait_until(lambda: pipeline.get_result()["status"] == "running")
        self.assertIsNotNone(pipeline.get_frame())
        time.sleep(0.35)
        result = pipeline.get_result()
        self.assertEqual(result["status"], "stale")
        self.assertTrue(result["stale"])
        self.assertIsNone(pipeline.get_frame())

    def test_bad_depth_model_is_visible_and_video_thread_survives(self):
        snapshot = frame_snapshot()
        pipeline, _, _ = self.make_pipeline(lambda: snapshot, depth=FakeDepth(bad=True))
        pipeline.start()
        wait_until(lambda: pipeline.get_result()["status"] == "error")
        self.assertIn("Results.depth", pipeline.get_result()["error"])
        self.assertIsNone(pipeline.get_frame())

    def test_missing_model_has_actionable_error_without_download(self):
        pipeline = HumanPipeline(lambda: None, lambda stamp: {},
                                 {"detector_weights": "does-not-exist-for-test.pt", "log_dir": None})
        self.addCleanup(pipeline.stop)
        pipeline.start()
        wait_until(lambda: pipeline.get_result()["status"] == "error")
        self.assertIn("Missing local detector_weights", pipeline.get_result()["error"])

    def test_getters_return_copies_and_strict_json(self):
        snapshot = frame_snapshot()
        pipeline, _, _ = self.make_pipeline(lambda: snapshot)
        pipeline.start()
        wait_until(lambda: pipeline.get_result()["status"] == "running")
        result = pipeline.get_result()
        result["people"].clear()
        self.assertEqual(len(pipeline.get_result()["people"]), 1)
        image = pipeline.get_frame()
        image[:] = 123
        self.assertFalse(np.all(pipeline.get_frame() == 123))
        with pipeline._lock:
            pipeline._result["inference_ms"] = float("nan")
        self.assertIsNone(json.loads(json.dumps(pipeline.get_result(), allow_nan=False))["inference_ms"])

    def test_checkpoint_buffers_identify_baked_metric_scale(self):
        head = SimpleNamespace(cal_a=np.array([1.]), cal_b=np.array([-0.3167]))
        model = SimpleNamespace(ckpt={}, model=SimpleNamespace(model=[head]))
        result = HumanPipeline._read_calibration(model)
        self.assertEqual(result["status"], "pretrained_metric_scale_camera_unvalidated")

    def test_ground_plane_estimate_logged_alongside_depth(self):
        snapshot = frame_snapshot()

        def pitched(stamp):
            return {**telemetry(stamp), "altitude_hae": 27.2, "sensor_elevation": -45}

        with tempfile.TemporaryDirectory() as directory:
            pipeline, _, _ = self.make_pipeline(lambda: snapshot, pitched, log_dir=directory,
                                                ground_hae_m=17.2)
            pipeline.start()
            wait_until(lambda: pipeline.get_result()["status"] == "running")
            result = pipeline.get_result()
            self.assertEqual(result["ground_source"], "configured")
            self.assertEqual(result["ground_hae_m"], 17.2)
            person = result["people"][0]
            self.assertEqual(person["status"], "estimated")
            ground = person["ground_plane"]
            self.assertEqual(ground["status"], "estimated")
            self.assertAlmostEqual(ground["height_above_ground_m"], 10)
            self.assertGreater(ground["ground_range_m"], 10)
            self.assertIsNotNone(ground["latitude"])
            self.assertIsInstance(person["estimate_disagreement_m"], float)
            self.assertGreaterEqual(person["estimate_disagreement_m"], 0)
            pipeline.stop()
            line = json.loads(list(Path(directory).glob("*.jsonl"))[0].read_text().splitlines()[0])
            self.assertEqual(line["people"][0]["ground_plane"]["status"], "estimated")
            self.assertIn("estimate_disagreement_m", line["people"][0])

    def test_launch_hae_fallback_is_labelled_and_level_camera_has_no_ground_fix(self):
        snapshot = frame_snapshot()
        fixed = lambda stamp: {**telemetry(stamp), "latitude": 33.93, "longitude": -84.52}
        pipeline, _, _ = self.make_pipeline(lambda: snapshot, fixed)
        pipeline.start()
        wait_until(lambda: pipeline.get_result()["status"] == "running")
        result = pipeline.get_result()
        self.assertEqual(result["ground_source"], "launch_hae_assumed_ground")
        self.assertEqual(result["ground_hae_m"], 17.2)
        self.assertTrue(any("GROUND_HAE_M" in note for note in result["warnings"]))
        ground = result["people"][0]["ground_plane"]
        self.assertEqual(ground["status"], "unavailable")
        self.assertEqual(ground["reason"], "aircraft_not_above_ground_plane")
        self.assertIsNone(result["people"][0]["estimate_disagreement_m"])
        # Depth estimate is untouched by the ground-plane addition.
        self.assertEqual(result["people"][0]["status"], "estimated")

    def test_ground_plane_without_telemetry_or_gps_fix_is_unavailable_not_an_error(self):
        # The 0,0 no-fix placeholder must never become launch ground height.
        for provider in (lambda stamp: {}, lambda stamp: telemetry(stamp)):
            snapshot = frame_snapshot()
            pipeline, _, _ = self.make_pipeline(lambda: snapshot, provider)
            pipeline.start()
            wait_until(lambda: pipeline.get_result()["status"] == "running")
            result = pipeline.get_result()
            self.assertEqual(result["ground_source"], "unavailable")
            self.assertIsNone(result["ground_hae_m"])
            self.assertEqual(result["people"][0]["ground_plane"]["status"], "unavailable")
            self.assertEqual(result["people"][0]["ground_plane"]["reason"], "ground_height_unknown")
            self.assertTrue(any("GROUND_HAE_M" in note for note in result["warnings"]))
            pipeline.stop()

    def test_invalid_ground_hae_config_rejected(self):
        with self.assertRaises(ValueError):
            HumanPipeline(lambda: None, lambda stamp: {}, {"ground_hae_m": "high", "log_dir": None})

    def test_stopping_during_load_does_not_publish_after_stop(self):
        entered, release = threading.Event(), threading.Event()
        pipeline, _, _ = self.make_pipeline(lambda: frame_snapshot())
        original_load = pipeline._load_models

        def slow_load():
            entered.set()
            release.wait(3)
            original_load()

        pipeline._load_models = slow_load
        self.addCleanup(release.set)
        pipeline.start()
        self.assertTrue(entered.wait(2))
        # Release after stop's bounded join so the late loader path is exercised.
        pipeline.stop()
        release.set()
        pipeline._thread.join(1)
        self.assertEqual(pipeline.get_result()["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
