"""Run the existing RGB person/depth models on saved reference photographs.

This offline entry point never starts Flask, RTSP, CoT, or aircraft controls.
Human identities and ground-truth coordinates are not inputs to inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "skydio live data"
sys.path.insert(0, str(LIVE_DIR))

import cv2
import torch

from human_geolocation import fuse_estimates
from human_pipeline import HumanPipeline, _json_safe
from metadata_geometry import apply_capture_geometry, read_capture


SCHEMA_VERSION = 1
CSV_FIELDS = [
    "source", "source_sha256", "photo_kind", "camera_source", "capture_time",
    "request_nonce", "person_index", "identity", "confidence", "bbox_x1",
    "bbox_y1", "bbox_x2", "bbox_y2", "processed_width", "processed_height",
    "original_width", "original_height", "depth_m", "depth_mode", "status",
    "reason", "latitude", "longitude", "north_m", "east_m", "slant_range_m",
    "ground_status", "ground_reason", "ground_latitude", "ground_longitude",
    "ground_north_m", "ground_east_m", "ground_range_m", "ground_source",
    "fusion_status", "fusion_reason", "fusion_source", "fusion_latitude", "fusion_longitude",
    "fusion_north_m", "fusion_east_m", "fusion_range_m",
    "estimate_disagreement_m", "geometry_status", "result_json",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Publish a complete strict-JSON record atomically."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers/antivirus can briefly hold a destination without delete
    # sharing. Preserve the atomic publication and retry the transient lock.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def resize_frame(frame, max_side):
    height, width = frame.shape[:2]
    if max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    return cv2.resize(frame, (max(2, round(width * scale)), max(2, round(height * scale))),
                      interpolation=cv2.INTER_AREA)


def result_key(relative_path):
    path = Path(relative_path)
    suffix = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{suffix}"


def flatten_result(record, result_path):
    source = record["source"]
    capture = record["capture_metadata"]
    for person in record["people"]:
        ground = person.get("ground_plane") or {}
        # Runs saved before the gate existed carry no fusion; it is a pure function of the saved estimates.
        fusion = person.get("fusion") or fuse_estimates(person, ground)
        bbox = person["bbox"]
        yield {
            "source": source["relative_path"], "source_sha256": source["sha256"],
            "photo_kind": capture.get("photo_kind"), "camera_source": capture.get("camera_source"),
            "capture_time": capture.get("capture_time"), "request_nonce": capture.get("request_nonce"),
            "person_index": person["index"], "identity": "", "confidence": person["confidence"],
            **dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), bbox)),
            "processed_width": source["processed_shape"][1], "processed_height": source["processed_shape"][0],
            "original_width": source["original_shape"][1], "original_height": source["original_shape"][0],
            **{key: person.get(key) for key in (
                "depth_m", "depth_mode", "status", "reason", "latitude", "longitude",
                "north_m", "east_m", "slant_range_m", "estimate_disagreement_m")},
            **{"ground_" + key: ground.get(key) for key in (
                "status", "reason", "latitude", "longitude", "north_m", "east_m")},
            "ground_range_m": ground.get("ground_range_m"),
            "ground_source": record.get("ground_source"),
            **{"fusion_" + key: fusion.get(key) for key in (
                "status", "reason", "source", "latitude", "longitude", "north_m", "east_m", "range_m")},
            "geometry_status": capture.get("geometry_status"), "result_json": result_path,
        }


def write_detections_csv(output):
    """Re-flatten a finished run's saved result JSONs into detections.csv (no inference)."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    with (output / "detections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entry in manifest["records"]:
            record = json.loads((output / entry["result_json"]).read_text(encoding="utf-8"))
            writer.writerows(flatten_result(record, entry["result_json"]))


def summarize(records, failures, expected):
    groups = defaultdict(lambda: Counter(images=0, detected_people=0, images_with_people=0,
                                         depth_positions=0, ground_positions=0))
    reasons = Counter()
    for record in records:
        capture = record["capture_metadata"]
        kind = f"{capture.get('camera_source', 'unknown')} / {capture.get('photo_kind', 'unknown')}"
        group = groups[kind]
        people = record["people"]
        group.update(images=1, detected_people=len(people), images_with_people=int(bool(people)))
        for person in people:
            group["depth_positions"] += int(person.get("status") == "estimated")
            ground = person.get("ground_plane") or {}
            group["ground_positions"] += int(ground.get("status") == "estimated")
            if person.get("reason"):
                reasons["depth:" + person["reason"]] += 1
            if ground.get("reason"):
                reasons["ground:" + ground["reason"]] += 1
    return {
        "expected_images": expected, "successful_images": len(records), "failed_images": len(failures),
        "detected_people_total": sum(len(record["people"]) for record in records),
        "counts_by_photo_kind": {key: dict(value) for key, value in sorted(groups.items())},
        "unavailable_geometry_reasons": dict(reasons),
        "meaning": "Counts are per-photo detections, not unique people, recall, or validated GPS accuracy.",
    }


def annotate(frame, record):
    annotated = HumanPipeline._annotate(frame, record)
    h, w = frame.shape[:2]
    label = (f"OFFLINE | {record['source']['relative_path']} | "
             f"{record['capture_metadata'].get('photo_kind', 'unknown')} | "
             "identities unassigned; geometry provisional")
    cv2.rectangle(annotated, (0, max(0, h - 28)), (w, h), (15, 15, 15), -1)
    cv2.putText(annotated, label, (8, max(12, h - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (30, 215, 255), 1, cv2.LINE_AA)
    return annotated


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Geolocation Reference Data")
    parser.add_argument("--output", type=Path, default=ROOT / "reference_validation" / "outputs" / "baseline")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index")
    parser.add_argument("--max-side", type=int, default=2048, help="Downscale only; preserve image aspect ratio")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--detector-no-upscale", action="store_true",
                        help="Cap the detector size at each photo's long side; keeps 640x512 thermal frames native")
    parser.add_argument("--depth-imgsz", type=int, default=768)
    parser.add_argument("--confidence", type=float, default=.4)
    parser.add_argument("--depth-mode", choices=("axial", "range"), default="axial")
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--torch-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Optional first N files for an explicitly labelled subset run")
    parser.add_argument("--files", nargs="+", help="Optional exact relative filenames for an explicitly labelled subset")
    parser.add_argument("--resume", action="store_true", help="Reuse successful records only when hashes/configuration match")
    parser.add_argument("--csv-only", action="store_true",
                        help="Rewrite detections.csv in --output from its saved result JSONs, without inference")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.max_side < 64 or args.detector_imgsz < 32 or args.depth_imgsz < 32:
        parser.error("Image sizes are too small")
    if args.torch_threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--torch-threads and --limit must be positive")
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        parser.error("--input must be an existing directory")
    all_files = sorted((p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")),
                       key=lambda path: path.relative_to(input_dir).as_posix())
    files = all_files
    if args.files:
        requested = set(args.files)
        files = [path for path in files if path.relative_to(input_dir).as_posix() in requested]
        missing = requested - {path.relative_to(input_dir).as_posix() for path in files}
        if missing:
            parser.error(f"Requested files not found: {sorted(missing)}")
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        parser.error("No JPEG photographs selected")
    output = args.output.resolve()
    if args.csv_only:
        write_detections_csv(output)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    (output / "annotated").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error("Output already has a manifest; choose a new directory or pass --resume")

    torch.set_num_threads(args.torch_threads)
    config = {
        "device": args.device, "log_dir": None, "detector_imgsz": args.detector_imgsz,
        "detector_no_upscale": args.detector_no_upscale,
        "depth_imgsz": args.depth_imgsz, "confidence": args.confidence,
        "depth_mode": args.depth_mode, "max_depth_m": args.max_depth_m,
    }
    worker = HumanPipeline(lambda: None, lambda _: {}, config)
    model_hashes = {key: {"path": str(Path(worker.config[key]).resolve()),
                          "sha256": sha256_file(worker.config[key])}
                    for key in ("detector_weights", "depth_weights")}
    code_paths = [Path(__file__), Path(__file__).with_name("metadata_geometry.py"),
                  LIVE_DIR / "human_pipeline.py", LIVE_DIR / "human_geolocation.py"]
    code_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths}
    runtime = {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": version("ultralytics"), "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }
    selection = [path.relative_to(input_dir).as_posix() for path in files]
    signature_payload = {
        "schema_version": SCHEMA_VERSION, "input": str(input_dir), "selection": selection,
        "config": worker.config, "max_side": args.max_side, "model_hashes": model_hashes,
        "code_hashes": code_hashes, "runtime": runtime,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != signature:
            parser.error("Resume signature differs (selected files, code, models, settings, or runtime); use a new output directory")
    manifest = {
        **signature_payload, "run_signature": signature, "status": "running", "started_at": utc_now(),
        "updated_at": utc_now(), "output": str(output), "dataset_jpeg_count": len(all_files),
        "subset_run": len(files) != len(all_files), "validation_scope": "offline_saved_photographs",
        "ground_truth_used_for_fitting": False, "identity_matching": "unassigned",
        "limitations": [
            "The existing person detector and depth checkpoint were pretrained on RGB; thermal performance is unvalidated.",
            "Monocular depth retains pretrained metric scale; no calibration is fitted to these reference coordinates.",
            "Metadata-derived geolocation is provisional and depends on camera pose, intrinsics and ground assumptions.",
            "No automatic ground-truth matching is performed; detection indices restart in every image.",
            "Images are downscaled before the existing full-frame detector/depth path; no tile or crop refinement is used.",
            "A complete batch means every selected JPEG was processed, not that geolocation accuracy was established.",
        ],
        "records": [], "failures": [],
    }
    write_json(manifest_path, manifest)
    records, failures = [], []
    started = time.perf_counter()
    try:
        print(json.dumps({"event": "loading_models", "selected_images": len(files), "runtime": runtime}), flush=True)
        worker._load_models()
        manifest.update(effective_device=worker._device, depth_calibration=worker._calibration)
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(input_dir).as_posix()
            key = result_key(relative)
            result_path = output / "results" / f"{key}.json"
            annotated_path = output / "annotated" / f"{key}.jpg"
            try:
                source_hash = sha256_file(path)
                reused = False
                if args.resume and result_path.exists() and annotated_path.exists():
                    candidate = json.loads(result_path.read_text(encoding="utf-8"))
                    if candidate.get("run_signature") == signature and candidate.get("source", {}).get("sha256") == source_hash:
                        record = candidate
                        reused = True
                if not reused:
                    capture = read_capture(path)
                    original = cv2.imread(str(path))
                    if original is None:
                        raise ValueError("OpenCV could not decode this JPEG")
                    original_shape = list(original.shape)
                    frame = resize_frame(original, args.max_side)
                    del original
                    # Only the adapter interprets capture pose/intrinsics. Empty
                    # telemetry ensures the live code cannot invent a ground reference.
                    worker._launch_hae = None
                    record, _ = worker._process_frame(frame, relative, time.time(), {}, None, None)
                    record = apply_capture_geometry(record, capture, frame.shape,
                                                    max_depth_m=args.max_depth_m)
                    record.update(
                        schema_version=SCHEMA_VERSION, run_signature=signature,
                        validation_mode="offline_reference_image_provisional_metadata_geometry",
                        capture_metadata=capture,
                        source={"relative_path": relative, "absolute_path": str(path), "sha256": source_hash,
                                "size_bytes": path.stat().st_size, "original_shape": original_shape,
                                "processed_shape": list(frame.shape),
                                "resize_scale_x": frame.shape[1] / original_shape[1],
                                "resize_scale_y": frame.shape[0] / original_shape[0]},
                        model_config=worker.config, identity_matching="unassigned",
                    )
                    record["captured_at"] = None
                    record["frame_id"] = relative
                    record["alignment"] = "Pose/intrinsics embedded in this photograph; no live CoT timestamp pairing"
                    for person in record["people"]:
                        person["identity"] = None
                        person["bbox_original"] = [
                            coordinate * (original_shape[1] / frame.shape[1] if axis % 2 == 0 else original_shape[0] / frame.shape[0])
                            for axis, coordinate in enumerate(person["bbox"])
                        ]
                    write_json(result_path, record)
                    if not cv2.imwrite(str(annotated_path), annotate(frame, record), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError("OpenCV could not write annotated image")
                records.append(record)
                entry = {"source": relative, "sha256": source_hash,
                         "photo_kind": record["capture_metadata"].get("photo_kind"),
                         "detections": len(record["people"]), "reused": reused,
                         "result_json": result_path.relative_to(output).as_posix(),
                         "annotated_image": annotated_path.relative_to(output).as_posix()}
                manifest["records"].append(entry)
                print(json.dumps({"event": "processed", "index": index, "total": len(files), **entry}), flush=True)
            except Exception as exc:
                failure = {"source": relative, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                failures.append(failure)
                print(json.dumps({"event": "failed", "index": index, "total": len(files), **failure}), flush=True)
            manifest.update(updated_at=utc_now(), failures=failures, summary=summarize(records, failures, len(files)),
                            elapsed_s=time.perf_counter() - started)
            write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        manifest["fatal_traceback"] = traceback.format_exc()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            print(f"Batch interrupted: {exc}", file=sys.stderr, flush=True)
        else:
            print(manifest["fatal_traceback"], file=sys.stderr, flush=True)
    finally:
        with (output / "detections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record, entry in zip(records, manifest["records"]):
                writer.writerows(flatten_result(record, entry["result_json"]))
        complete = len(records) == len(files) and not failures and not manifest.get("fatal_error")
        manifest.update(status="complete" if complete else "partial", finished_at=utc_now(), updated_at=utc_now(),
                        failures=failures, summary=summarize(records, failures, len(files)),
                        elapsed_s=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        write_json(output / "summary.json", {key: manifest[key] for key in (
            "status", "subset_run", "run_signature", "dataset_jpeg_count", "summary", "limitations", "elapsed_s")})
        print(json.dumps({"event": "finished", "status": manifest["status"], "summary": manifest["summary"],
                          "manifest": str(manifest_path)}), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
