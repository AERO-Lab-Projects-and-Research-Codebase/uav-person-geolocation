"""Offline checks for split tar reads, paired resizing, depth units and leakage."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dataset as preparation


def image_and_depth(depth):
    image = Image.new("RGB", (depth.shape[1], depth.shape[0]), (100, 120, 140))
    rgb = io.BytesIO()
    image.save(rgb, format="JPEG")
    numeric = io.BytesIO()
    np.save(numeric, depth, allow_pickle=False)
    return rgb.getvalue(), numeric.getvalue()


class PreparationChecks(unittest.TestCase):
    def test_joined_reads_across_part_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a", Path(directory) / "b"]
            paths[0].write_bytes(b"01234")
            paths[1].write_bytes(b"56789")
            with preparation.JoinedParts(paths) as joined:
                joined.seek(3)
                self.assertEqual(joined.read(5), b"34567")
                joined.seek(-3, 2)
                self.assertEqual(joined.read(), b"789")

    def test_metric_units_invalid_mask_and_cap_after_quantization(self):
        depth = np.full((32, 64, 1), 35.555, dtype=np.float32)
        depth[0, :5, 0] = [np.nan, np.inf, -1, 500, 499.999]
        image_bytes, depth_bytes = image_and_depth(depth)
        image, encoded, details = preparation.encode_pair(image_bytes, depth_bytes, 64, 500)
        self.assertEqual(image.size, (64, 32))
        self.assertEqual(encoded.dtype, np.uint16)
        self.assertTrue((encoded[0, :5] == 0).all())
        self.assertAlmostEqual(encoded[1, 1] / 100, 35.56)
        self.assertEqual(details["range_excluded_source_pixels"], 1)

    def test_location_suffixes_never_split(self):
        splits = preparation.choose_splits(["park1", "park5", "campus", "factory", "farm"], 2026, .15, .15)
        self.assertEqual(splits["park1"], splits["park5"])
        self.assertEqual(set(splits.values()), {"train", "val", "test"})

    def test_single_pixel_source_size_rounding_is_explicit(self):
        rgb, _ = image_and_depth(np.full((32, 64, 1), 10, dtype=np.float32))
        numeric = io.BytesIO()
        np.save(numeric, np.full((33, 65, 1), 10, dtype=np.float32))
        image, encoded, details = preparation.encode_pair(rgb, numeric.getvalue(), 64, 500)
        self.assertTrue(details["one_pixel_alignment_adjusted"])
        self.assertEqual(details["source_depth_shape"], [33, 65])
        self.assertEqual(encoded.shape, (32, 64))
        np.testing.assert_array_equal(encoded, np.full((32, 64), 1000))
        numeric = io.BytesIO()
        np.save(numeric, np.full((35, 65), 10, dtype=np.float32))
        with self.assertRaises(ValueError):
            preparation.encode_pair(rgb, numeric.getvalue(), 64, 500)

    def test_weighted_holdouts_keep_dominant_location_in_train(self):
        counts = {"longhua": 7500, "park1": 700, "park5": 800, "campus": 1000,
                  "factory": 1000, "farm": 500, "airport": 500}
        split = preparation.choose_splits(list(counts), 2026, .15, .15, counts)
        self.assertEqual(split["longhua"], "train")
        self.assertEqual(split["park1"], split["park5"])
        for holdout in ("val", "test"):
            fraction = sum(counts[scene] for scene in counts if split[scene] == holdout) / sum(counts.values())
            self.assertLess(abs(fraction - .15), .05)
        self.assertEqual(split, preparation.choose_splits(list(counts), 2026, .15, .15, counts))

    def test_unsafe_paths_and_links_rejected(self):
        for name in ("../outside", "/absolute", "C:/outside", "safe/../../outside"):
            with self.assertRaises(ValueError):
                preparation.safe_member(tarfile.TarInfo(name))
        link = tarfile.TarInfo("safe/link")
        link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            preparation.safe_member(link)

    def test_complete_pipeline_with_split_archive_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloads = root / "downloads"
            downloads.mkdir()
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as tar:
                for scene in ("park1", "park5", "farm", "factory", "campus"):
                    image, depth = image_and_depth(np.full((64, 96, 1), 35.55, dtype=np.float32))
                    for name, value in (("image.jpg", image), ("depth.npy", depth), ("meta.json", b'{"intrinsics": []}')):
                        member = tarfile.TarInfo(f"TrainingData_Final_MoGe-All/{scene}/000001/{name}")
                        member.size = len(value)
                        tar.addfile(member, io.BytesIO(value))
            blob = payload.getvalue()
            boundary = len(blob) // 2 + 137
            ledger = {}
            for name, value in zip(preparation.PART_NAMES, [blob[:boundary], blob[boundary:]]):
                path = downloads / name
                path.write_bytes(value)
                ledger[name] = {"size": len(value), "mtime_ns": path.stat().st_mtime_ns,
                                "sha256": hashlib.sha256(value).hexdigest()}
            (downloads / "verified_files.json").write_text(json.dumps(ledger))
            with (downloads / "final_merged.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Scene_Name", "Renamed_Image", "Split"])
                writer.writerows((scene, "000001.jpg", "train") for scene in ("park1", "park5", "farm", "factory", "campus"))
            old_root = preparation.ROOT
            preparation.ROOT = root
            try:
                args = argparse.Namespace(max_side=64, max_depth=500., seed=2026,
                    val_fraction=.15, test_fraction=.15, remove_archives=False)
                preparation.prepare(args)
                preparation.prepare(args)  # Interrupted/resumed jobs keep exactly the same pairs.
            finally:
                preparation.ROOT = old_root
            dataset = root / "dataset"
            status = json.loads((dataset / "preparation_status.json").read_text())
            self.assertTrue(status["complete"])
            self.assertEqual(status["total"], 5)
            self.assertEqual(sum(status["counts"].values()), 5)
            self.assertNotIn(str(root), (dataset / "data.yaml").read_text())
            for png in (dataset / "depth").rglob("*.png"):
                with Image.open(png) as image:
                    numeric = np.asarray(image)
                self.assertEqual(numeric.shape, (43, 64))
                self.assertTrue((numeric == 3555).all())


if __name__ == "__main__":
    unittest.main()
