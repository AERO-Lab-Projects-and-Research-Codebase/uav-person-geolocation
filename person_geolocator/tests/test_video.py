import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from person_geolocator.video import load_log, main, pick_recording, pose_at, recordings

HEADER = ["CUSTOM.dateTime", "OSD.latitude", "OSD.longitude", "OSD.height", "OSD.altitude",
          "GIMBAL.pitch", "GIMBAL.roll", "OSD.yaw", "CAMERA.isVideo", "CAMERA.isPhoto"]


def write_log(path):
    """10 Hz log, 0-20 s: no GPS fix for the first 0.5 s, recording 2-6 s and 10-13 s, photos at 4 s and 15 s,
    yaw crossing 180."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerow(["1970-01-01T00:00:00Z", 34.0, -84.0, 0, 0, 0, 0, 0, "false", "false"])  # dropped: no clock yet
        for i in range(201):
            s = i / 10
            fix = s >= 0.5
            w.writerow([f"2026-10-27T15:00:{int(s):02d}.{round(s * 1000) % 1000:03d}Z",
                        34.0 + 1e-5 * s if fix else 0, -84.0 if fix else 0, 30.0, 330.0,
                        -45.0, 0.0, (170 + s) if s < 10 else (s - 190),
                        "true" if 2 <= s <= 6 or 10 <= s <= 13 else "false",
                        "true" if i in (40, 150) else "false"])


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        write_log(self.dir / "log.csv")
        self.log = load_log(self.dir / "log.csv")

    def test_recordings_and_pick(self):
        spans = recordings(self.log)
        t0 = self.log["t"][0]
        self.assertEqual([(round(a - t0, 1), round(b - t0, 1)) for a, b in spans], [(2.0, 6.0), (10.0, 13.0)])
        self.assertEqual(pick_recording(spans, 3.1), spans[1])
        self.assertEqual(pick_recording(spans, 3.1, index=0), spans[0])

    def test_pose_interpolates_and_wraps_yaw(self):
        t0 = self.log["t"][0]
        pose = pose_at(self.log, t0 + 9.95, 70.0)
        self.assertAlmostEqual(pose["heading_deg"], 179.95, places=3)
        pose = pose_at(self.log, t0 + 10.05, 70.0)       # wraps to -180 side, not a 360 sweep
        self.assertAlmostEqual(pose["heading_deg"], -179.95, places=3)
        self.assertAlmostEqual(pose["lat"], 34.0 + 1e-5 * 10.05, places=9)
        self.assertEqual((pose["pitch_deg"], pose["height_above_ground_m"], pose["hfov_deg"]), (-45.0, 30.0, 70.0))
        self.assertAlmostEqual(pose_at(self.log, t0, 70.0)["lat"], 34.0 + 5e-6)  # pre-fix rows skipped

    def test_main_feeds_each_sampled_frame_its_log_pose(self):
        video = self.dir / "clip.mp4"
        out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
        for _ in range(30):                               # 3 s at 10 fps -> the 10-13 s recording
            out.write(np.zeros((36, 64, 3), np.uint8))
        out.release()
        seen = []

        def fake(path, model, pose):
            seen.append((Path(path).name, pose))
            est = {"status": "estimated", "lat": pose["lat"], "lon": pose["lon"]}
            return {"people": [{"index": 1, "confidence": 0.9, "depth_m": 12.0,
                                "methods": {"depth_ray": est, "ground_plane": {"status": "unavailable"}}}],
                    "annotated_jpeg": b"jpg"}

        with contextlib.redirect_stdout(io.StringIO()):
            code = main([str(video), str(self.dir / "log.csv"), "--every", "1", "--out", str(self.dir / "run")],
                        locate_fn=fake)
        self.assertEqual(code, 0)
        self.assertEqual([name for name, _ in seen], ["frame_000000.jpg", "frame_000010.jpg", "frame_000020.jpg"])
        self.assertAlmostEqual(seen[1][1]["lat"], 34.0 + 1e-5 * 11, places=9)
        with open(self.dir / "run" / "people.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]["ground_plane_lat"], "")
        self.assertTrue((self.dir / "run" / "frames" / "frame_000020_annotated.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
