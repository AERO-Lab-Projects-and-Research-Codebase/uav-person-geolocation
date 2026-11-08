"""Convert the complete dense AerialMetric tar into a portable YOLO depth dataset.

The split tar is read virtually, without concatenating or extracting another 56 GB.
Original meter depths become uint16 centimeter PNGs; zeros retain unknown pixels.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tarfile
import time

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
PART_NAMES = ["TrainingData_Final_MoGe-All.tar.part-aa", "TrainingData_Final_MoGe-All.tar.part-ab"]
REVISION = "8ad32c3b83f48c89fc8eb11eb5cc7f9c024ca788"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class JoinedParts:
    """Seekable virtual concatenation, including reads crossing part boundaries."""
    def __init__(self, paths: list[Path]):
        self.files = [p.open("rb") for p in paths]
        self.starts = [0]
        for p in paths:
            self.starts.append(self.starts[-1] + p.stat().st_size)
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.starts[-1] + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.starts[-1] - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size:
            index = bisect.bisect_right(self.starts, self.position) - 1
            take = min(size, self.starts[index + 1] - self.position)
            self.files[index].seek(self.position - self.starts[index])
            block = self.files[index].read(take)
            if len(block) != take:
                raise OSError("Truncated archive part")
            chunks.append(block)
            self.position += take
            size -= take
        return b"".join(chunks)

    def close(self):
        for stream in self.files:
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in name:
        raise ValueError(f"Unsafe archive path: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"Unsupported archive link/device: {member.name}")
    if member.size > 256 * 1024**2:
        raise ValueError(f"Unexpectedly large single member: {member.name}")
    return path


def scene_group(scene: str) -> str:
    # Conservatively keep numbered portions of a named location together.
    return re.sub(r"[ _-]*\d+$", "", scene.casefold()) or scene.casefold()


def choose_splits(scenes: list[str], seed: int, val_fraction: float, test_fraction: float,
                  scene_counts: dict[str, int] | None = None) -> dict[str, str]:
    groups = sorted({scene_group(scene) for scene in scenes})
    if len(groups) < 3:
        raise ValueError("At least 3 location groups are needed for train/val/test; supply more data.")
    if scene_counts is not None:
        weights = {group: sum(scene_counts[scene] for scene in scenes if scene_group(scene) == group)
                   for group in groups}
        largest = max(groups, key=lambda group: (weights[group], group))
        eligible = [group for group in groups if group != largest]
        total = sum(weights.values())
        randomizer = random.Random(seed)
        best = None

        def pick(order, target, reserved_count):
            chosen, count = [], 0
            for group in order:
                if len(chosen) >= len(order) - reserved_count:
                    break
                if abs(count + weights[group] - target) < abs(count - target):
                    chosen.append(group)
                    count += weights[group]
            if not chosen:
                chosen = [min(order, key=lambda group: abs(weights[group] - target))]
                count = weights[chosen[0]]
            return chosen, count

        for _ in range(600):
            order = eligible.copy()
            randomizer.shuffle(order)
            val, val_count = pick(order, total * val_fraction, 1)
            remaining = [group for group in order if group not in val]
            test, test_count = pick(remaining, total * test_fraction, 0)
            score = abs(val_count / total - val_fraction) + abs(test_count / total - test_fraction)
            candidate = (score, tuple(sorted(val)), tuple(sorted(test)))
            if best is None or candidate < best:
                best = candidate
        _, val, test = best
        return {scene: "val" if scene_group(scene) in val else "test" if scene_group(scene) in test else "train"
                for scene in scenes}
    random.Random(seed).shuffle(groups)
    val_count = max(1, round(len(groups) * val_fraction))
    test_count = max(1, round(len(groups) * test_fraction))
    if val_count + test_count >= len(groups):
        raise ValueError("Holdout fractions leave no training scenes")
    group_split = {group: "val" if i < val_count else "test" if i < val_count + test_count else "train"
                   for i, group in enumerate(groups)}
    return {scene: group_split[scene_group(scene)] for scene in scenes}


def encode_pair(image_bytes: bytes, depth_bytes: bytes, max_side: int, max_depth: float):
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
    depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.dtype.kind != "f":
        raise ValueError(f"Expected floating-point HxW depth, received {depth.shape} {depth.dtype}")
    width, height = image.size
    source_depth_shape = list(depth.shape)
    alignment_adjusted = False
    if depth.shape != (height, width):
        if abs(depth.shape[0] - height) <= 1 and abs(depth.shape[1] - width) <= 1:
            # A few upstream L1 pairs round image/depth dimensions differently.
            # Their pairing and normalized FOV are retained; resample the entire
            # depth field by nearest neighbor, recording the one-pixel adjustment.
            depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(
                (width, height), Image.Resampling.NEAREST))
            alignment_adjusted = True
        else:
            raise ValueError(f"Misaligned RGB/depth: RGB {(height, width)}, depth {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0.001) & (depth < max_depth)
    range_excluded = int(np.count_nonzero(np.isfinite(depth) & (depth >= max_depth)))
    numeric = np.where(valid, depth, 0).astype(np.float32)
    factor = min(1.0, max_side / max(width, height))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
        numeric = np.asarray(Image.fromarray(numeric).resize(size, Image.Resampling.NEAREST))
    encoded = np.rint(numeric * 100).astype(np.uint16)
    encoded[encoded.astype(np.float32) / 100 >= max_depth] = 0
    if np.count_nonzero(encoded) < 10:
        raise ValueError("Fewer than 10 valid depth pixels after resizing/range masking")
    return image, encoded, {"source_width": width, "source_height": height,
        "source_depth_shape": source_depth_shape, "one_pixel_alignment_adjusted": alignment_adjusted,
        "width": size[0], "height": size[1], "valid_pixels": int(np.count_nonzero(encoded)),
        "range_excluded_source_pixels": range_excluded,
        "valid_fraction": float(np.count_nonzero(encoded) / encoded.size)}


def index_archive(archive: tarfile.TarFile):
    records = {}
    for member in archive:
        path = safe_member(member)
        if not member.isfile():
            continue
        if len(path.parts) != 4 or path.parts[0] != "TrainingData_Final_MoGe-All":
            continue
        if path.name not in {"image.jpg", "image.png", "depth.npy", "meta.json"}:
            continue
        scene, frame = path.parts[1:3]
        if not re.fullmatch(r"[A-Za-z0-9_. -]+", scene) or not re.fullmatch(r"[A-Za-z0-9_. -]+", frame):
            raise ValueError(f"Unsafe scene/frame identifier: {path}")
        record = records.setdefault((scene, frame), {})
        field = "image" if path.name.startswith("image.") else path.name
        if field in record:
            raise ValueError(f"Duplicate member: {path}")
        record[field] = member
    pairs = {key: value for key, value in records.items() if "image" in value and "depth.npy" in value}
    if len(pairs) != len(records):
        raise ValueError(f"Found {len(records) - len(pairs)} incomplete image/depth pairs")
    if not pairs:
        raise ValueError("No dense image/depth pairs; check the selected archive parts")
    return pairs


def save_encoded_pair(image_bytes: bytes, depth_bytes: bytes, metadata_bytes: bytes | None,
                      max_side: int, max_depth: float, scene: str, frame: str, split: str,
                      upstream_csv_split: str, target_image: Path, target_depth: Path,
                      target_meta: Path) -> dict:
    """Encode/write one independent pair; this worker never accesses the tar reader."""
    image, depth, details = encode_pair(image_bytes, depth_bytes, max_side, max_depth)
    meta = json.loads(metadata_bytes) if metadata_bytes is not None else {}
    details.update({"scene": scene, "frame": frame, "split": split,
        "upstream_csv_split": upstream_csv_split,
        "original_metadata": meta, "depth_convention": "source metric camera depth; no ray/ground-range conversion"})
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_depth.parent.mkdir(parents=True, exist_ok=True)
    target_image_tmp = target_image.with_suffix(".jpg.tmp")
    target_depth_tmp = target_depth.with_suffix(".png.tmp")
    image.save(target_image_tmp, format="JPEG", quality=95, subsampling=0)
    Image.fromarray(depth).save(target_depth_tmp, format="PNG", compress_level=3)
    target_image_tmp.replace(target_image)
    target_depth_tmp.replace(target_depth)
    # The receipt is written last, retaining the existing interruption/resume protocol.
    atomic_json(target_meta, details)
    return details


def read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    """Read archive bytes on the calling main thread, before submitting work."""
    with archive.extractfile(member) as stream:
        return stream.read()


def official_splits(metadata: Path) -> dict[tuple[str, str], str]:
    if not metadata.exists():
        raise FileNotFoundError("Run download_dataset.py first to fetch final_merged.csv")
    result = {}
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["Scene_Name"], Path(row["Renamed_Image"]).stem)
            result[key] = row.get("Split", "").strip().casefold()
    return result


def prepare(args) -> None:
    downloads = ROOT / "downloads"
    parts = [downloads / name for name in PART_NAMES]
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(f"Missing {part}; run scripts/download_dataset.py")
    verified_path = downloads / "verified_files.json"
    if not verified_path.exists():
        raise ValueError("Missing checksum-verification manifest; run download_dataset.py --verify-only")
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    for part in parts:
        record = verified.get(part.name, {})
        if record.get("size") != part.stat().st_size or record.get("mtime_ns") != part.stat().st_mtime_ns:
            raise ValueError(f"Archive changed since verification: {part.name}; reverify")
    dataset = ROOT / "dataset"
    settings = {"revision": REVISION, "max_side": args.max_side, "max_depth": args.max_depth,
                "seed": args.seed, "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
                "depth_scale": 100, "split_policy": "weighted_location_groups_largest_group_kept_in_train",
                "dataset_scope": "AerialMetric Oblique complete dense training subset"}
    state_path = dataset / "preparation_settings.json"
    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing dataset uses different settings; prepare into a fresh bundle to avoid mixed data")
    atomic_json(state_path, settings)
    source_splits = official_splits(downloads / "final_merged.csv")
    with JoinedParts(parts) as joined, tarfile.open(fileobj=joined, mode="r:") as archive:
        print("Indexing dense tar headers without extracting the archive...", flush=True)
        pairs = index_archive(archive)
        scenes = sorted({scene for scene, _ in pairs})
        scene_counts = dict(Counter(scene for scene, _ in pairs))
        splits = choose_splits(scenes, args.seed, args.val_fraction, args.test_fraction, scene_counts)
        for (scene, frame) in pairs:
            recorded_split = source_splits.get((scene, frame))
            # The pinned CSV has 196 L1 entries marked NAN. The upstream dense
            # archive explicitly lives in Train-Split; retain those entries while
            # recording the missing CSV assignment. Explicit holdouts are refused.
            if recorded_split not in {"train", "nan", ""}:
                raise ValueError(f"{scene}/{frame}: explicit/missing CSV Split={recorded_split!r}; refusing split leakage")
        atomic_json(dataset / "scene_splits.json", {"scene_to_split": splits,
            "scene_to_location_group": {scene: scene_group(scene) for scene in scenes},
            "source_scene_pair_counts": scene_counts,
            "note": "Count-weighted location holdouts target 15% val/test; largest group reserved for train"})
        print(f"Indexed {len(pairs):,} pairs across {len(scenes)} scenes. Converting...", flush=True)
        rows = []
        counts = {"train": 0, "val": 0, "test": 0}
        workers = getattr(args, "workers", 4)
        if workers < 1:
            raise ValueError("workers must be a positive integer")
        pending = deque()

        def consume_oldest() -> None:
            # Ordered consumption keeps rows, counters and status on the main thread.
            index, scene, frame, split, target_image, target_depth, job = pending.popleft()
            details = job.result() if isinstance(job, Future) else job
            counts[split] += 1
            rows.append({"scene": scene, "frame": frame, "split": split,
                "image": target_image.relative_to(dataset).as_posix(),
                "depth": target_depth.relative_to(dataset).as_posix(),
                "upstream_csv_split": source_splits[(scene, frame)],
                "valid_fraction": details["valid_fraction"],
                "range_excluded_source_pixels": details["range_excluded_source_pixels"]})
            if index % 100 == 0 or index + 1 == len(pairs):
                atomic_json(dataset / "preparation_status.json", {"complete": False, "converted": index + 1,
                    "total": len(pairs), "counts": counts, "updated_unix": time.time()})
                print(f"Prepared {index + 1:,}/{len(pairs):,} pairs; {counts}", flush=True)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="depth-pair") as executor:
            for index, ((scene, frame), members) in enumerate(sorted(pairs.items())):
                if shutil.disk_usage(ROOT).free < 4 * 1024**3:
                    raise OSError("Less than 4 GiB free; conversion paused. Resume after freeing space.")
                split = splits[scene]
                # Ultralytics sorts paths, so hash-first names interleave scenes even
                # after that sort. This also balances its first-200-frame calibration.
                prefix = hashlib.sha256(f"{args.seed}/{scene}/{frame}".encode()).hexdigest()[:16]
                stem = f"{prefix}_{scene}_{frame}"
                target_image = dataset / "images" / split / f"{stem}.jpg"
                target_depth = dataset / "depth" / split / f"{stem}.png"
                target_meta = dataset / "metadata" / split / f"{stem}.json"
                if target_image.exists() and target_depth.exists() and target_meta.exists():
                    job = json.loads(target_meta.read_text(encoding="utf-8"))
                else:
                    # JoinedParts/tarfile are seekable but shared: read all bytes here,
                    # then let workers operate only on their independent byte buffers.
                    image_bytes = read_member_bytes(archive, members["image"])
                    depth_bytes = read_member_bytes(archive, members["depth.npy"])
                    metadata_bytes = read_member_bytes(archive, members["meta.json"]) if "meta.json" in members else None
                    job = executor.submit(save_encoded_pair, image_bytes, depth_bytes, metadata_bytes,
                        args.max_side, args.max_depth, scene, frame, split, source_splits[(scene, frame)],
                        target_image, target_depth, target_meta)
                pending.append((index, scene, frame, split, target_image, target_depth, job))
                # Bound memory and outstanding writes: eight jobs with the default four workers.
                if len(pending) >= workers * 2:
                    consume_oldest()
            while pending:
                consume_oldest()
        # Stable hash-first paths preserve interleaving when the loader sorts.
        random.Random(args.seed).shuffle(rows)
        for split in counts:
            (dataset / f"{split}.txt").write_text("".join("./" + row["image"] + "\n" for row in rows
                                                         if row["split"] == split), encoding="utf-8")
        with (dataset / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        data_yaml = {"path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt",
                     "nc": 1, "names": {0: "depth"}, "channels": 3,
                     "depth_scale": 100, "max_depth": args.max_depth}
        (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        atomic_json(dataset / "preparation_status.json", {"complete": True, "converted": len(rows),
            "total": len(rows), "counts": counts, "scene_count": len(scenes), "settings": settings,
            "csv_split_counts": dict(Counter(source_splits[key] for key in pairs)),
            "achieved_split_fractions": {split: count / len(rows) for split, count in counts.items()},
            "note": "Internal location holdouts from official Train-Split; not the official benchmark test set"})
    if args.remove_archives:
        for part in parts:
            resolved = part.resolve()
            if resolved.parent != downloads.resolve() or resolved.name not in PART_NAMES:
                raise ValueError("Refusing archive cleanup outside this bundle downloads directory")
            resolved.unlink()
        print("Removed verified dense tar parts after completed conversion; dataset remains portable.")
    print("Dataset prepared. Run scripts/verify_dataset.py before copying or training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-depth", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4, help="Parallel pair encoders; main thread alone reads the tar (default: 4)")
    parser.add_argument("--remove-archives", action="store_true", help="Remove only verified tar parts after full successful preparation")
    args = parser.parse_args()
    if not 0.01 < args.max_depth <= 655.35 or args.max_side < 32:
        parser.error("max-depth must be <=655.35 m and max-side >=32")
    if not 0 < args.val_fraction < 1 or not 0 < args.test_fraction < 1:
        parser.error("Holdout fractions must be between 0 and 1")
    if args.workers < 1:
        parser.error("workers must be a positive integer")
    prepare(args)


if __name__ == "__main__":
    main()
