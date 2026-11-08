"""Converts the downloaded SkyScenes archives into the layout of the prepared AerialMetric set.

Reads downloads/skyscenes/{Images,Depth,metaData}/<setting>/<town>.tar.gz and writes dataset/skyscenes with
images resized to at most 768 px (JPEG 95), depth as uint16 centimetre PNG decoded from CARLA's logarithmic grey
image, and one metadata JSON per frame holding the 110 degree pinhole intrinsics. One town is held out for
validation, so no validation scene is seen in training.

Usage (from unidepth_v2_finetune):  python scripts/prepare_skyscenes.py [--val-town Town10HD] [--limit N]
"""
import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ftdata import SKYSCENES_FOV_DEG, pinhole_normalized, skyscenes_depth_cm  # noqa: E402

SRC, OUT, MAX_SIDE = HERE / "downloads" / "skyscenes", HERE / "dataset" / "skyscenes", 768
PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10])


def members(tar):
    """(frame id, bytes) of every file in a tar; some SkyScenes archives hold a second tar named like a picture or a JSON."""
    for m in tar:
        if not m.isfile() or Path(m.name).name.startswith("._"):   # "._" files are macOS resource forks, not data
            continue
        data = tar.extractfile(m).read()
        if data[:1] == b"{" or data[:8] == PNG or data[:2] == bytes([255, 216]):   # JSON, PNG or JPEG, judged by content
            frame = re.search(r"\d+", Path(m.name).name)
            if frame:
                yield frame.group(), data
        else:
            with tarfile.open(fileobj=io.BytesIO(data)) as nested:
                yield from members(nested)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-town", default="Town10HD")
    ap.add_argument("--limit", type=int, help="frames per archive, for a quick trial")
    args = ap.parse_args()
    lists = {"train": [], "val": []}
    for archive in sorted((SRC / "Images").glob("*/*.tar.gz")):
        setting, town = archive.parent.name, archive.name.split(".")[0]
        split = "val" if town == args.val_town else "train"
        with tarfile.open(SRC / "Depth" / setting / archive.name) as tar:
            depth = {frame: data for frame, data in members(tar)}
        with tarfile.open(SRC / "metaData" / setting / archive.name) as tar:
            meta = {frame: json.loads(data) for frame, data in members(tar)}
        for kind in ("images", "depth", "metadata"):
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
        done, seen = 0, set()
        with tarfile.open(archive) as tar:
            for frame, data in members(tar):
                if args.limit and done >= args.limit:
                    break
                if frame in seen or frame not in depth or frame not in meta:   # some archives repeat their frames in a nested tar
                    continue
                seen.add(frame)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                grey = np.asarray(Image.open(io.BytesIO(depth[frame])))
                grey = grey if grey.ndim == 2 else grey[..., 0]
                assert grey.shape == image.size[::-1], (archive, frame, grey.shape, image.size)
                scale = MAX_SIDE / max(image.size)
                size = (round(image.width * scale), round(image.height * scale))
                name = f"{setting}_{town}_{frame}"
                image.resize(size, Image.LANCZOS).save(OUT / "images" / split / f"{name}.jpg", quality=95)
                # nearest keeps each label a real depth; the grey image is resized before decoding
                small = np.asarray(Image.fromarray(grey).resize(size, Image.NEAREST))
                Image.fromarray(skyscenes_depth_cm(small)).save(OUT / "depth" / split / f"{name}.png")
                m = meta[frame]
                record = {"width": size[0], "height": size[1], "scene": town, "setting": setting, "frame": frame,
                          "actual_height": m.get("actual_height"), "actual_pitch": m.get("actual_pitch"),
                          "source": "SkyScenes ClearNoon (hoffman-lab/SkyScenes, MIT)",
                          "original_metadata": {"intrinsics": pinhole_normalized(SKYSCENES_FOV_DEG, image.width, image.height)}}
                (OUT / "metadata" / split / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
                lists[split].append(f"./images/{split}/{name}.jpg")
                done += 1
        print(f"{setting} {town}: {done} frames -> {split}", flush=True)
    for split, items in lists.items():
        (OUT / f"{split}.txt").write_text("\n".join(sorted(items)) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in lists.items()})


if __name__ == "__main__":
    main()
