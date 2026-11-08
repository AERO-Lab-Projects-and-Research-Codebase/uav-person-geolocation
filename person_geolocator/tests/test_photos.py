import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from person_geolocator.photos import main, match_times, photo_time
from person_geolocator.tests.test_video import write_log
from person_geolocator.video import load_log

T0 = 1793113200.0  # 2026-10-27T15:00:00Z, the first row of write_log


def write_photo(path, stamp=None, subsec=None):
    exif = Image.Exif()
    if stamp:
        exif.get_ifd(0x8769)[0x9003] = stamp
    if subsec:
        exif.get_ifd(0x8769)[0x9291] = subsec
    Image.new("RGB", (64, 36)).save(path, exif=exif)


class PhotoTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        write_log(self.dir / "log.csv")

    def test_photo_time_reads_local_exif_in_the_given_zone(self):
        write_photo(self.dir / "a.jpg", "2026:10:27 11:00:04", "25")
        self.assertAlmostEqual(photo_time(self.dir / "a.jpg", utc_offset=-4), T0 + 4.25)
        write_photo(self.dir / "b.jpg")
        self.assertIsNone(photo_time(self.dir / "b.jpg", utc_offset=0))

    def test_log_rows_start_at_t0(self):
        self.assertEqual(load_log(self.dir / "log.csv")["t"][0], T0)

    def test_match_times(self):
        shutters = [T0 + 4, T0 + 15]
        self.assertEqual(match_times([T0 + 4.9, T0 + 8], shutters, 2.0), [(T0 + 4, "shutter"), (T0 + 8, "exif")])
        self.assertEqual(match_times([None, T0 + 15], shutters, 2.0), [(T0 + 4, "log order"), (T0 + 15, "log order")])
        with self.assertRaises(ValueError):
            match_times([None], shutters, 2.0)

    def test_main_takes_pose_from_log_and_skips_photos_outside_it(self):
        write_photo(self.dir / "DJI_0001.JPG", "2026:10:27 15:00:05")   # snaps to the 4 s shutter
        write_photo(self.dir / "DJI_0002.JPG", "2026:10:27 15:00:09")   # no shutter near: EXIF time
        write_photo(self.dir / "DJI_0003.JPG", "2026:10:27 16:00:00")   # after the log ends
        seen = []

        def fake(path, model, pose):
            seen.append((Path(path).name, pose))
            est = {"status": "estimated", "lat": pose["lat"], "lon": pose["lon"]}
            return {"people": [{"index": 1, "confidence": 0.9, "depth_m": 12.0, "methods": {"depth_ray": est}}],
                    "annotated_jpeg": b"jpg"}

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main([str(self.dir), "--log", str(self.dir / "log.csv"), "--utc-offset", "0", "--hfov", "70", "--snap", "2",
                  "--out", str(self.dir / "run")], locate_fn=fake)
        self.assertEqual([name for name, _ in seen], ["DJI_0001.JPG", "DJI_0002.JPG"])
        self.assertAlmostEqual(seen[0][1]["lat"], 34.0 + 1e-5 * 4, places=9)
        self.assertAlmostEqual(seen[1][1]["lat"], 34.0 + 1e-5 * 9, places=9)
        self.assertEqual(seen[0][1]["hfov_deg"], 70.0)
        self.assertIn("DJI_0003.JPG: taken outside the log", out.getvalue())
        with open(self.dir / "run" / "people.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["photo"], r["time_source"]) for r in rows], [("DJI_0001.JPG", "shutter"), ("DJI_0002.JPG", "exif")])
        self.assertTrue((self.dir / "run" / "DJI_0001_annotated.jpg").is_file())
        saved = json.loads((self.dir / "run" / "DJI_0002.json").read_text())
        self.assertEqual((saved["photo"], saved["time_source"]), ("DJI_0002.JPG", "exif"))
        self.assertAlmostEqual(saved["pose"]["lat"], 34.0 + 1e-5 * 9, places=9)


if __name__ == "__main__":
    unittest.main()
