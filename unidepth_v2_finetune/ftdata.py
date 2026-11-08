"""Shared pieces of the UniDepth V2 fine-tune: SkyScenes depth decoding and the image/depth pair dataset.

Both training sets are stored in the layout the AerialMetric preparation already uses:
    images/<split>/<name>.jpg, depth/<split>/<name>.png (uint16 centimetres, 0 = no label),
    metadata/<split>/<name>.json (intrinsics divided by image width and height), <split>.txt
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEPTH_SCALE = 100       # depth PNGs are centimetres
MAX_DEPTH_M = 500.0     # the AerialMetric set is capped here, so SkyScenes is too
SKYSCENES_FOV_DEG = 110.0   # horizontal, from the SkyScenes paper's appendix; images are 2160 x 1440
# CARLA LogarithmicDepth: grey = 255 * (1 + ln(depth / 1000 m) / 5.70378), clamped to 0..255
GREY_TO_M = np.exp((np.arange(256) / 255 - 1) * 5.70378) * 1000
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def skyscenes_depth_cm(grey):
    """uint16 centimetres from a SkyScenes grey depth image; 0 marks a pixel with no usable label."""
    metres = GREY_TO_M[grey]
    # grey 0-1 is everything nearer than 3.4 m (clamped) and 255 is sky; neither is a depth
    valid = (grey > 1) & (grey < 255) & (metres < MAX_DEPTH_M)
    return np.where(valid, np.rint(metres * DEPTH_SCALE), 0).astype(np.uint16)


def pinhole_normalized(fov_deg, width, height):
    """Pinhole intrinsics with row 0 divided by the width and row 1 by the height, as AerialMetric stores them."""
    f = width / (2 * math.tan(math.radians(fov_deg) / 2))
    return [[f / width, 0.0, 0.5], [0.0, f / height, 0.5], [0.0, 0.0, 1.0]]


def unit_rays(params, height, width):
    """Unit viewing rays (B, 3, H, W) for pinhole params (B, 4) = fx, fy, cx, cy; pixel centres at +0.5 as UniDepth uses."""
    v, u = torch.meshgrid(torch.arange(height, device=params.device) + 0.5,
                          torch.arange(width, device=params.device) + 0.5, indexing="ij")
    fx, fy, cx, cy = (params[:, i, None, None] for i in range(4))
    rays = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like((u - cx) / fx)], dim=1)
    return rays / rays.norm(dim=1, keepdim=True)


class PairDataset(torch.utils.data.Dataset):
    """Normalised image, metric depth (m, 0 = no label) and pixel intrinsics, cropped and resized to one shape."""

    def __init__(self, root, split, shape=(504, 756), train=True, limit=None):
        self.root, self.shape, self.train = Path(root), tuple(shape), train
        with open(self.root / f"{split}.txt", encoding="utf-8") as fh:
            self.items = [line.strip() for line in fh if line.strip()]
        if limit:
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        rel = Path(self.items[index])                       # ./images/<split>/<name>.jpg
        split, name = rel.parent.name, rel.stem
        image = np.asarray(Image.open(self.root / rel).convert("RGB"))
        depth = np.asarray(Image.open(self.root / "depth" / split / f"{name}.png"))
        with open(self.root / "metadata" / split / f"{name}.json", encoding="utf-8") as fh:
            k = json.load(fh)["original_metadata"]["intrinsics"]
        h, w = image.shape[:2]
        fx, fy, cx, cy = k[0][0] * w, k[1][1] * h, k[0][2] * w, k[1][2] * h
        # crop to the training aspect ratio (random place when training, centre otherwise), then resize
        out_h, out_w = self.shape
        crop_w, crop_h = min(w, round(h * out_w / out_h)), min(h, round(w * out_h / out_w))
        left = random.randint(0, w - crop_w) if self.train else (w - crop_w) // 2
        top = random.randint(0, h - crop_h) if self.train else (h - crop_h) // 2
        image = Image.fromarray(image[top:top + crop_h, left:left + crop_w]).resize((out_w, out_h), Image.BILINEAR)
        rows = (np.arange(out_h) + 0.5) * crop_h / out_h
        cols = (np.arange(out_w) + 0.5) * crop_w / out_w
        depth = depth[top:top + crop_h, left:left + crop_w][rows.astype(int)][:, cols.astype(int)]   # nearest: never blend depths
        fx, cx = fx * out_w / crop_w, (cx - left) * out_w / crop_w
        fy, cy = fy * out_h / crop_h, (cy - top) * out_h / crop_h
        image = np.asarray(image, np.float32) / 255
        depth = depth.astype(np.float32) / DEPTH_SCALE
        if self.train and random.random() < 0.5:
            image, depth, cx = image[:, ::-1], depth[:, ::-1], out_w - cx
        image = (image - MEAN) / STD
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "depth": torch.from_numpy(np.ascontiguousarray(depth[None])),
                "params": torch.tensor([fx, fy, cx, cy], dtype=torch.float32)}


def mixed_batches(loaders):
    """Endless batches that take one batch from every loader and join them, so each set has the same weight."""
    iterators = [iter(loader) for loader in loaders]
    while True:
        parts = []
        for i, loader in enumerate(loaders):
            try:
                parts.append(next(iterators[i]))
            except StopIteration:
                iterators[i] = iter(loader)
                parts.append(next(iterators[i]))
        yield {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
