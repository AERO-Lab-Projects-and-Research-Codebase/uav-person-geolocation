"""Offline checks for person-depth sampling and frame/result association.

No camera, pretrained model, checkpoint, or CUDA allocation is used.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).with_name("run_webcam_depth.py")
SPEC = importlib.util.spec_from_file_location("webcam_runner_under_test", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TensorLike:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def detections(coords=(), scores=(), classes=()):
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxy=TensorLike(np.asarray(coords).reshape(-1, 4)),
        conf=TensorLike(scores), cls=TensorLike(classes)))


class TorsoSamplingTests(unittest.TestCase):
    def test_background_and_feet_are_excluded_from_person_depth(self):
        depth = np.full((100, 100), 50., dtype=np.float32)
        depth[25:60, 35:65] = 2.5
        sample = runner.sample_torso(depth, [0, 0, 100, 100])
        self.assertEqual(sample["depth_m"], 2.5)
        self.assertEqual(sample["valid_fraction"], 1.)

    def test_nonfinite_and_nonpositive_pixels_are_never_meters(self):
        depth = np.zeros((20, 20), dtype=np.float32)
        depth[5:12, 7:13] = np.nan
        depth[5, 7:13] = [np.inf, -1., 0., 0., 0., 3.]
        result = runner.sample_torso(depth, [0, 0, 20, 20])
        self.assertIsNone(result["depth_m"])
        self.assertLess(result["valid_fraction"], .5)

    def test_exactly_half_valid_pixels_returns_their_median(self):
        depth = np.zeros((20, 20), dtype=np.float32)
        depth[5:12, 7:10] = 3.
        result = runner.sample_torso(depth, [0, 0, 20, 20])
        self.assertEqual(result["valid_fraction"], .5)
        self.assertEqual(result["depth_m"], 3.)

    def test_invalid_or_offscreen_box_has_no_depth(self):
        depth = np.ones((20, 20), dtype=np.float32)
        for box in ([0, 0, 0, 20], [20, 0, 0, 20], [0, 0, np.nan, 20],
                    [100, 100, 120, 120], [-120, -120, -100, -100]):
            with self.subTest(box=box):
                self.assertIsNone(runner.sample_torso(depth, box)["depth_m"])

    def test_clipping_does_not_wrap_negative_indices_into_background(self):
        depth = np.full((20, 20), 99., dtype=np.float32)
        depth[:10, :10] = 2.
        self.assertEqual(runner.sample_torso(depth, [-5, -5, 10, 10])["depth_m"], 2.)

    def test_multi_channel_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HxW"):
            runner.sample_torso(np.ones((20, 20, 3)), [0, 0, 20, 20])


class IntrinsicsTests(unittest.TestCase):
    def write(self, folder, **changes):
        data = {"width": 60, "height": 40,
                "K": [[50., 0., 30.], [0., 50., 20.], [0., 0., 1.]]}
        data.update(changes)
        path = Path(folder) / "intrinsics.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_unspecified_camera_does_not_invent_calibration(self):
        self.assertIsNone(runner.pixel_intrinsics(None, (40, 60)))

    def test_exact_resolution_preserves_camera_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            matrix = runner.pixel_intrinsics(self.write(folder), (40, 60))
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_array_equal(matrix, [[50., 0., 30.], [0., 50., 20.], [0., 0., 1.]])

    def test_wrong_processed_resolution_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "actual processed frame"):
                runner.pixel_intrinsics(self.write(folder), (60, 40))

    def test_invalid_camera_matrix_is_rejected(self):
        matrices = [None, [[1., 0.], [0., 1.]],
                    [[0., 0., 30.], [0., 50., 20.], [0., 0., 1.]],
                    [[50., 0., 30.], [0., 50., 20.], [0., 1., 1.]],
                    [[50., 0., float("nan")], [0., 50., 20.], [0., 0., 1.]]]
        with tempfile.TemporaryDirectory() as folder:
            for matrix in matrices:
                with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                    runner.pixel_intrinsics(self.write(folder, K=matrix), (40, 60))


class PipelineTests(unittest.TestCase):
    def engine(self, found):
        engine = runner.DepthEngine.__new__(runner.DepthEngine)
        engine.args = SimpleNamespace(imgsz=640, depth_imgsz=640, confidence=.4,
                                      intrinsics=None, yolo_weights=Path("fine_depth.pt"),
                                      yolo_original_weights=Path("original_depth.pt"))
        engine.device = "cpu"
        engine.detector, engine.yolo, engine.yolo_original = object(), object(), object()
        engine.calibration = {"cal_a": 1., "cal_b": 3.833984375}
        engine.original_calibration = {"cal_a": 2., "cal_b": .5}
        engine.calls = []
        engine.clear_cuda = lambda: None

        def yolo_predict(model, frame, **options):
            name = ("detector" if model is engine.detector else
                    "yolo_original" if model is engine.yolo_original else "yolo")
            engine.calls.append((name, frame))
            if model is engine.detector:
                return found
            value = 2. if model is engine.yolo_original else 3.
            return SimpleNamespace(depth=SimpleNamespace(data=TensorLike(np.full(frame.shape[:2], value))))

        def uni_predict(frame, intrinsics=None):
            engine.calls.append(("unidepth", frame))
            return {"depth": np.full(frame.shape[:2], 4., dtype=np.float32),
                    "metadata": {"predicted_intrinsics": [[50., 0., 30.], [0., 50., 20.], [0., 0., 1.]]}}

        def ling_predict(frame, depth, intrinsics=None):
            engine.calls.append(("lingbot", frame))
            np.testing.assert_array_equal(depth, np.full(frame.shape[:2], 4.))
            np.testing.assert_array_equal(intrinsics, [[50., 0., 30.], [0., 50., 20.], [0., 0., 1.]])
            return {"depth": np.full(frame.shape[:2], 5., dtype=np.float32), "metadata": {}}

        engine.yolo_predict = yolo_predict
        engine.unidepth = SimpleNamespace(model=SimpleNamespace(to=lambda device: None), predict=uni_predict)
        engine.lingbot = SimpleNamespace(predict=ling_predict)
        return engine

    def test_four_models_measure_the_same_captured_frame_and_people(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        engine = self.engine(detections([[30, 0, 60, 40], [0, 0, 30, 40]], [.8, .9], [0, 0]))
        record = engine.process(frame, 23, 100.)
        self.assertEqual(record["frame_id"], 23)
        self.assertEqual(record["captured_at"], 100.)
        self.assertEqual(record["lingbot_input"], "unidepth_v2_predicted_depth_same_frame")
        self.assertEqual([name for name, _ in engine.calls],
                         ["detector", "yolo_original", "yolo", "unidepth", "lingbot"])
        self.assertTrue(all(value is frame for _, value in engine.calls))
        self.assertEqual([person["bbox"][0] for person in record["people"]], [0., 30.])
        for person in record["people"]:
            self.assertEqual([person["estimates"][name]["depth_m"]
                              for name in ("yolo_original", "yolo", "unidepth", "lingbot")],
                             [2., 3., 4., 5.])
            self.assertEqual(len({tuple(estimate["roi_xyxy"]) for estimate in person["estimates"].values()}), 1)
        self.assertEqual(record["metadata"]["yolo_original"]["weights"], "original_depth.pt")
        self.assertEqual(record["metadata"]["yolo"]["weights"], "fine_depth.pt")
        self.assertEqual(record["metadata"]["yolo_original"]["saved_calibration"],
                         engine.original_calibration)
        self.assertEqual(record["metadata"]["yolo"]["saved_calibration"], engine.calibration)
        self.assertFalse(record["metadata"]["yolo_original"]["extra_scale_fitted"])
        self.assertFalse(record["metadata"]["yolo"]["extra_scale_fitted"])
        self.assertIn("yolo_original", record["timings_ms"])
        self.assertIn("yolo", record["timings_ms"])

    def test_missing_original_depth_fails_without_substituting_finetuned_result(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        engine = self.engine(detections([[0, 0, 60, 40]], [.8], [0]))
        predict = engine.yolo_predict

        def missing_original(model, captured_frame, **options):
            if model is engine.yolo_original:
                return SimpleNamespace(depth=None)
            return predict(model, captured_frame, **options)

        engine.yolo_predict = missing_original
        with self.assertRaisesRegex(ValueError, "YOLO original checkpoint returned no Results.depth"):
            engine.process(frame, 23, 100.)
        self.assertEqual([name for name, _ in engine.calls], ["detector"])

    def test_original_depth_map_alignment_failure_stops_comparison(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        engine = self.engine(detections([[0, 0, 60, 40]], [.8], [0]))
        predict = engine.yolo_predict

        def misaligned_original(model, captured_frame, **options):
            if model is engine.yolo_original:
                return SimpleNamespace(depth=SimpleNamespace(data=TensorLike(np.ones((20, 30)))))
            return predict(model, captured_frame, **options)

        engine.yolo_predict = misaligned_original
        with self.assertRaisesRegex(ValueError, "yolo_original depth shape .* does not match RGB"):
            engine.process(frame, 23, 100.)
        self.assertEqual([name for name, _ in engine.calls], ["detector"])

    def test_empty_frame_produces_fresh_empty_result_without_depth_inference(self):
        engine = self.engine(detections())
        record = engine.process(np.zeros((40, 60, 3), dtype=np.uint8), 24, 101.)
        self.assertEqual(record["frame_id"], 24)
        self.assertEqual(record["people"], [])
        self.assertEqual([name for name, _ in engine.calls], ["detector"])

    def test_invalid_lowconfidence_and_nonperson_boxes_do_not_trigger_depth(self):
        found = detections([[0, 0, 30, 40], [0, 0, 30, 40], [0, 0, np.nan, 40], [20, 0, 10, 40]],
                           [.3, .9, .9, .9], [0, 1, 0, 0])
        record = self.engine(found).process(np.zeros((40, 60, 3), dtype=np.uint8), 25, 102.)
        self.assertEqual(record["people"], [])

    def test_depth_map_alignment_failure_is_not_silently_resized(self):
        with self.assertRaisesRegex(ValueError, "does not match RGB"):
            runner.DepthEngine.sample([], "unidepth", np.ones((20, 30)), (40, 60))


class RenderingAndLogTests(unittest.TestCase):
    def test_historical_analysis_uses_its_image_and_keeps_live_preview_separate(self):
        live = np.full((480, 640, 3), 80, dtype=np.uint8)
        measured = np.full((480, 640, 3), 160, dtype=np.uint8)
        record = {"captured_at": 10., "people": []}
        original = measured.copy()
        result = runner.render(live, (measured, record), "Running", now=20.)
        np.testing.assert_array_equal(result[35:515, :640], live)
        np.testing.assert_array_equal(result[35:515, 640:], measured)
        np.testing.assert_array_equal(measured, original)

    def test_empty_results_remain_in_jsonl_without_fake_csv_person(self):
        args = runner.parser().parse_args([])
        engine = PipelineTests().engine(detections([[0, 0, 60, 40]], [.8], [0]))
        full = engine.process(np.zeros((40, 60, 3), dtype=np.uint8), 1, 10.)
        empty = copy.deepcopy(full)
        empty.update(frame_id=2, people=[])
        with tempfile.TemporaryDirectory() as folder:
            log = runner.ResultLog(Path(folder) / "run", args)
            try:
                with patch("builtins.print"):
                    log.write(full)
                    log.write(empty)
            finally:
                log.close()
            records = [json.loads(line) for line in (log.path / "estimates.jsonl").read_text().splitlines()]
            with (log.path / "people.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([item["frame_id"] for item in records], [1, 2])
            self.assertEqual(records[1]["people"], [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["yolo_original_m"], "2.0")
            self.assertEqual(rows[0]["yolo_m"], "3.0")
            self.assertEqual(rows[0]["lingbot_refined_unidepth_m"], "5.0")


class ArgumentValidationTests(unittest.TestCase):
    def test_unreasonable_numeric_options_fail_before_model_loading(self):
        cases = [("width", 31), ("height", 0), ("imgsz", 0), ("depth_imgsz", 0),
                 ("confidence", 0), ("confidence", 1.1), ("confidence", float("nan")),
                 ("interval", -1.), ("interval", float("inf")), ("torch_threads", 0),
                 ("timeout", -1.), ("timeout", float("nan")), ("max_results", 0)]
        for name, value in cases:
            args = runner.parser().parse_args([])
            setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                runner.validate(args)


if __name__ == "__main__":
    unittest.main()
