"""Offline dashboard contracts; never start RTSP, CoT sockets, or a model."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as dashboard


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.receiver = dashboard.CoTReceiver(aircraft_uid="")

    def test_aircraft_selected_without_operator_or_target_merge(self):
        self.receiver.record_telemetry({"uid": "air", "event_type": "a-f-A-M-H-Q",
                                       "latitude": 1, "longitude": 2, "sensor_hfov": 78}, 100)
        self.receiver.record_telemetry({"uid": "operator", "event_type": "a-f-G-U-C",
                                       "latitude": 50, "sensor_range": 999}, 101)
        self.receiver.record_telemetry({"uid": "target", "event_type": "b-m-p-s-p-i",
                                       "latitude": 80}, 102)
        selected = self.receiver.get_geolocation_telemetry(102)
        self.assertEqual(selected["uid"], "air")
        self.assertEqual(selected["latitude"], 1)
        self.assertNotIn("sensor_range", selected)
        # Even same-UID partial messages do not inherit stale pose fields.
        self.receiver.record_telemetry({"uid": "air", "event_type": "a-f-A-M-H-Q", "heading": 42}, 102.5)
        self.assertNotIn("latitude", self.receiver.get_geolocation_telemetry(103))
        self.assertNotIn("sensor_range", self.receiver.get_telemetry())

    def test_receive_time_selection_excludes_future_and_stale(self):
        for when, lat in [(100, 1), (102, 2), (105, 3)]:
            self.receiver.record_telemetry({"uid": "air", "event_type": "a-f-A", "latitude": lat}, when)
        self.assertEqual(self.receiver.get_geolocation_telemetry(101)["latitude"], 1)
        self.assertEqual(self.receiver.get_geolocation_telemetry(104)["latitude"], 2)
        self.assertEqual(self.receiver.get_geolocation_telemetry(108)["latitude"], 3)
        self.assertEqual(self.receiver.get_geolocation_telemetry(108.01), {})
        self.assertEqual(self.receiver.get_geolocation_telemetry(99), {})
        self.assertEqual(self.receiver.get_geolocation_telemetry(None), {})

    def test_explicit_uid_overrides_type_but_never_other_uid(self):
        receiver = dashboard.CoTReceiver(aircraft_uid="verified-aircraft")
        receiver.record_telemetry({"uid": "verified-aircraft", "latitude": 1}, 100)
        receiver.record_telemetry({"uid": "other", "event_type": "a-f-A", "latitude": 9}, 101)
        self.assertEqual(receiver.get_geolocation_telemetry(102)["latitude"], 1)

    def test_namespace_and_nonfinite_xml(self):
        parsed = dashboard.parse_cot_xml('''<cot:event xmlns:cot="urn:test" uid="air" type="a-f-A">
          <cot:point lat="0" lon="0" hae="nan" ce="inf"/>
          <cot:detail><cot:sensor fov="78" vfov="60" elevation="-20"/></cot:detail>
        </cot:event>''')
        self.assertEqual(parsed["uid"], "air")
        self.assertEqual(parsed["latitude"], 0)
        self.assertIsNone(parsed["altitude_hae"])
        self.assertIsNone(parsed["ce"])
        self.assertEqual(parsed["sensor_hfov"], 78)

    def test_stable_csv_columns_and_escaping(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = dashboard.CSVLogger(directory)
            logger.log({"uid": "air", "latitude": 1, "callsign": "air, one"})
            logger.log({"longitude": 2, "uid": "operator"})
            logger.close()
            with open(logger.path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["callsign"], "air, one")
            self.assertEqual(rows[1]["uid"], "operator")
            self.assertEqual(rows[1]["longitude"], "2")
            self.assertEqual(rows[1]["latitude"], "")


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.patches = patch.multiple(dashboard, video_capture=None, cot_receiver=None,
                                     csv_logger=None, human_pipeline=None, human_start_error=None)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.client = dashboard.app.test_client()

    def test_import_never_starts_resources(self):
        spec = importlib.util.spec_from_file_location("offline_dashboard_import", ROOT / "main.py")
        module = importlib.util.module_from_spec(spec)
        with patch("socket.socket") as socket_mock, patch("threading.Thread.start") as start_mock:
            spec.loader.exec_module(module)
        socket_mock.assert_not_called()
        start_mock.assert_not_called()
        self.assertIsNone(module.human_pipeline)

    def test_disabled_and_error_states_preserve_raw_route(self):
        with patch.object(dashboard, "HUMAN_ANALYSIS_ENABLED", False):
            result = self.client.get("/api/people").get_json()
            self.assertEqual(result["status"], "disabled")
            self.assertTrue(result["raw_fallback"])
        dashboard.human_pipeline = Mock()
        dashboard.human_pipeline.get_result.side_effect = RuntimeError("model unavailable")
        result = self.client.get("/api/people").get_json()
        self.assertEqual(result["error"], "model unavailable")
        self.assertTrue(result["raw_fallback"])
        response = self.client.get("/video_feed", buffered=False)
        self.assertIn(b"Content-Type: image/jpeg", next(response.response))
        response.close()

    def test_inference_frame_is_separate_from_raw_and_falls_back(self):
        raw = np.zeros((24, 32, 3), dtype=np.uint8)
        annotated = np.full((24, 32, 3), 255, dtype=np.uint8)
        dashboard.video_capture = Mock()
        dashboard.video_capture.get_frame.return_value = raw
        dashboard.human_pipeline = Mock()
        dashboard.human_pipeline.get_frame.return_value = annotated
        def mean_frame(human):
            generator = dashboard.generate_mjpeg(human=human)
            packet = next(generator)
            generator.close()
            jpeg = packet.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
            return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).mean()
        self.assertGreater(mean_frame(True), 250)
        self.assertLess(mean_frame(False), 5)
        dashboard.human_pipeline.get_frame.return_value = None
        self.assertLess(mean_frame(True), 5)
        dashboard.human_pipeline.get_frame.side_effect = RuntimeError("worker failed")
        self.assertLess(mean_frame(True), 5)

    def test_frame_snapshot_is_copied_and_keeps_decode_identity(self):
        capture = dashboard.VideoCapture("offline://never-opened")
        capture._frame = np.ones((2, 3, 3), dtype=np.uint8)
        capture._frame_count = 7
        capture._captured_at = 100
        snapshot = capture.get_frame_snapshot()
        snapshot["frame"][:] = 0
        self.assertEqual(capture.get_frame().sum(), 18)
        self.assertEqual(snapshot["frame_id"], 7)
        self.assertEqual(snapshot["captured_at"], 100)

    def test_ui_has_safe_text_and_estimate_labels(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("/human_video_feed", html)
        self.assertIn("Show raw video", html)
        self.assertIn("ASSUMED ZERO ORIGIN", html)
        self.assertIn("not a calibrated camera measurement", html)
        self.assertIn("Ground-plane", html)
        self.assertNotIn(".innerHTML", html)

    def test_ground_hae_env_is_optional_float(self):
        import os
        with patch.dict("os.environ", {"GROUND_HAE_M": ""}):
            self.assertIsNone(dashboard.human_config()["ground_hae_m"])
        with patch.dict("os.environ", {"GROUND_HAE_M": "276.3"}):
            self.assertEqual(dashboard.human_config()["ground_hae_m"], 276.3)
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("GROUND_HAE_M", None)
            self.assertIsNone(dashboard.human_config()["ground_hae_m"])


if __name__ == "__main__":
    unittest.main()
