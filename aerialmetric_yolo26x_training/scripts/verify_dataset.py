"""Verify every prepared image/depth pair, split separation and portable manifests."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from train_depth import BUNDLE_ROOT, read_data_yaml, split_image_paths, depth_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BUNDLE_ROOT / "dataset" / "data.yaml")
    parser.add_argument("--sample-only", action="store_true", help="inspect 30 evenly spaced pairs per split")
    args = parser.parse_args()
    data = read_data_yaml(args.data.resolve())
    dataset = Path(data["path"])
    status = json.loads((dataset / "preparation_status.json").read_text())
    if not status.get("complete"):
        raise ValueError("Dataset preparation is not complete")
    scene_splits = json.loads((dataset / "scene_splits.json").read_text())
    groups = {}
    for scene, split in scene_splits["scene_to_split"].items():
        group = scene_splits["scene_to_location_group"][scene]
        if group in groups and groups[group] != split:
            raise ValueError(f"Location group {group} leaks across splits")
        groups[group] = split
    image_sets = {}
    summary = {}
    with (dataset / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        pair_rows = list(csv.DictReader(stream))
    pair_scenes = {row["image"]: row["scene"] for row in pair_rows}
    if len(pair_scenes) != len(pair_rows):
        raise ValueError("Duplicate image entry in pairs.csv")
    for split in ("train", "val", "test"):
        paths = split_image_paths(data, split)
        image_sets[split] = set(paths)
        if len(paths) != status["counts"][split]:
            raise ValueError(f"Manifest count mismatch in {split}")
        selected = paths
        if args.sample_only and len(paths) > 30:
            selected = [paths[index] for index in np.linspace(0, len(paths)-1, 30).astype(int)]
        valid_pixels = 0
        maximum = 0.0
        for index, image_path in enumerate(selected):
            scene = pair_scenes[image_path.relative_to(dataset).as_posix()]
            if scene_splits["scene_to_split"].get(scene) != split:
                raise ValueError(f"Scene assigned to wrong split: {image_path}")
            with Image.open(image_path) as source:
                source.load()
                shape = (source.height, source.width)
            paired = depth_path(image_path)
            with Image.open(paired) as source:
                raw = np.asarray(source)
            if raw.ndim != 2 or raw.shape != shape or raw.dtype != np.uint16:
                raise ValueError(f"Expected aligned uint16 depth PNG: {paired}, {raw.shape}, {raw.dtype}")
            depth = raw.astype(np.float32) / data["depth_scale"]
            valid = depth[depth > 0.001]
            if valid.size < 10 or valid.max() >= data["max_depth"]:
                raise ValueError(f"Invalid depth range or insufficient valid pixels: {paired}")
            maximum = max(maximum, float(valid.max()))
            valid_pixels += int(valid.size)
            if index % 500 == 0:
                print(f"Verified {split} {index + 1}/{len(selected)}", flush=True)
        summary[split] = {"pairs": len(paths), "inspected": len(selected), "max_m": maximum,
                          "valid_pixels_inspected": valid_pixels}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if image_sets[a] & image_sets[b]:
            raise ValueError(f"Image path leakage between {a}/{b}")
    report = {"passed": True, "mode": "sample" if args.sample_only else "all_pairs", "splits": summary,
              "depth_scale": data["depth_scale"], "max_depth": data["max_depth"]}
    report_name = "verification_sample.json" if args.sample_only else "verification.json"
    (dataset / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
