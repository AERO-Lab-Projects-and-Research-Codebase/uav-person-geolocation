import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftdata import GREY_TO_M, PairDataset, mixed_batches, pinhole_normalized, skyscenes_depth_cm, unit_rays  # noqa: E402


class SkyScenesDecodingTest(unittest.TestCase):
    def test_grey_levels_decode_to_carla_logarithmic_depth(self):
        self.assertAlmostEqual(GREY_TO_M[255], 1000.0)
        self.assertAlmostEqual(GREY_TO_M[67], 14.9, delta=0.1)      # a camera 15 m up looking straight down
        cm = skyscenes_depth_cm(np.array([[0, 1, 67, 255]], np.uint8))
        self.assertEqual(cm.tolist(), [[0, 0, round(GREY_TO_M[67] * 100), 0]])   # too near, too near, ground, sky

    def test_depth_beyond_the_cap_has_no_label(self):
        self.assertEqual(skyscenes_depth_cm(np.array([[250]], np.uint8)).item(), 0)   # about 894 m

    def test_110_degree_camera(self):
        k = pinhole_normalized(110.0, 2160, 1440)
        self.assertAlmostEqual(k[0][0] * 2160, 756.2, delta=0.1)
        self.assertAlmostEqual(k[1][1] * 1440, 756.2, delta=0.1)   # square pixels


class PairDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for kind in ("images", "depth", "metadata"):
            (root / kind / "train").mkdir(parents=True)
        Image.fromarray(np.full((60, 120, 3), 128, np.uint8)).save(root / "images/train/a.jpg")
        depth = np.zeros((60, 120), np.uint16)
        depth[:, 60:] = 2500                                       # right half is 25 m, left half unlabelled
        Image.fromarray(depth).save(root / "depth/train/a.png")
        (root / "metadata/train/a.json").write_text(json.dumps(
            {"original_metadata": {"intrinsics": [[0.5, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]}}), encoding="utf-8")
        (root / "train.txt").write_text("./images/train/a.jpg\n", encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_item_is_resized_with_depth_in_metres_and_scaled_intrinsics(self):
        item = PairDataset(self.root, "train", shape=(28, 42), train=False)[0]
        self.assertEqual(tuple(item["image"].shape), (3, 28, 42))
        self.assertEqual(tuple(item["depth"].shape), (1, 28, 42))
        self.assertEqual(sorted(set(item["depth"].unique().tolist())), [0.0, 25.0])   # nearest resize: no blended depths
        # a 90 x 60 centre crop of the 120 x 60 image, resized to 42 x 28: fx 60 px -> 28, cx 60 - 15 = 45 px -> 21
        self.assertTrue(torch.allclose(item["params"], torch.tensor([28.0, 28.0, 21.0, 14.0])))

    def test_central_ray_points_along_the_camera_axis(self):
        rays = unit_rays(torch.tensor([[28.0, 28.0, 21.0, 14.0]]), 28, 42)
        self.assertEqual(tuple(rays.shape), (1, 3, 28, 42))
        self.assertTrue(torch.allclose(rays.norm(dim=1), torch.ones(1, 28, 42)))
        self.assertGreater(rays[0, 2, 14, 21].item(), 0.999)

    def test_mixed_batches_take_from_every_set(self):
        make = lambda: torch.utils.data.DataLoader(PairDataset(self.root, "train", shape=(28, 42)), batch_size=1)  # noqa: E731
        batch = next(mixed_batches([make(), make()]))
        self.assertEqual(batch["image"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
