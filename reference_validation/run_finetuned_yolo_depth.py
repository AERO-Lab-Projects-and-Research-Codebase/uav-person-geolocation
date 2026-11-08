"""Run the AerialMetric fine-tuned YOLO26x-depth checkpoint on saved photographs.

Reuses the boxes of a completed detection run (same source hashes, same processed
dimensions), so the review page can show the fine-tuned depth next to the other
models for every detection. Depth is predicted on the processed frame without
extra lens rectification, as in the live pipeline. GPS is provisional.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import run_depth_comparison as rdc
from human_pipeline import HumanPipeline
from metadata_geometry import read_capture
from run_reference_batch import resize_frame, result_key, sha256_file, summarize, utc_now, write_json

ROOT = rdc.ROOT
NAME = "finetuned"
LABEL = "AerialMetric fine-tuned YOLO26x-depth"
SCHEMA_VERSION = 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=ROOT / "Geolocation Reference Data")
    result.add_argument("--detection-run", type=Path, default=ROOT / "reference_validation/outputs/photos")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--weights", type=Path, default=ROOT / "aerialmetric_yolo26x_training/weights/best.pt")
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--device", default="0")
    result.add_argument("--max-side", type=int, default=2048)
    result.add_argument("--max-depth-m", type=float, default=300.)
    return result


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    input_dir, output, detection_run = args.input.resolve(), args.output.resolve(), args.detection_run.resolve()
    if (output / "manifest.json").exists():
        ap.error(f"Output already has a manifest: {output}")
    files = sorted((p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")),
                   key=lambda p: p.relative_to(input_dir).as_posix())
    baseline, detection_manifest = rdc.load_baseline(input_dir, detection_run, files, args.max_side)
    folder = output / NAME
    for child in ("results", "annotated", "depth", "depth_preview"):
        (folder / child).mkdir(parents=True, exist_ok=True)

    weights_sha = sha256_file(args.weights)
    quantize = 32
    signature = f"{weights_sha}:{detection_manifest['run_signature']}:{args.imgsz}:{quantize}:{args.max_depth_m}"
    model = YOLO(str(args.weights), task="depth")
    calibration = {**HumanPipeline._read_calibration(model), "weights": str(args.weights.resolve()),
                   "sha256": weights_sha}
    rdc.LABELS.setdefault(NAME, LABEL)
    model_config = {"depth_weights": str(args.weights.resolve()), "depth_imgsz": args.imgsz,
                    "quantize": quantize, "device": args.device, "depth_mode": "axial",
                    "depth_scale": 1.0, "max_depth_m": args.max_depth_m}
    model_manifest = {"status": "running", "label": LABEL, "weights": str(args.weights.resolve()),
                      "sha256": weights_sha, "records": [], "failures": [], "calibration": calibration,
                      "run_signature": signature}
    run = {"schema_version": SCHEMA_VERSION, "status": "running", "started_at": utc_now(),
           "input": str(input_dir), "detection_run": str(detection_run), "output": str(output),
           "selection": list(baseline), "selection_policy": "all_photos_in_detection_run",
           "selected_images": len(baseline),
           "selected_detections": sum(len(r["people"]) for r in baseline.values()),
           "detector_run_signature": detection_manifest["run_signature"],
           "input_hashes": {k: v["source"]["sha256"] for k, v in baseline.items()},
           "max_side": args.max_side, "imgsz": args.imgsz, "device": args.device, "quantize": quantize,
           "max_depth_m": args.max_depth_m, "depth_mode": "axial", "ground_truth_used_for_fitting": False,
           "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
           "models": {NAME: model_manifest}, "failures": []}
    write_json(output / "manifest.json", run)

    records, started = [], time.perf_counter()
    for index, path in enumerate(files, 1):
        relative = path.relative_to(input_dir).as_posix()
        key = result_key(relative)
        try:
            capture = read_capture(path)
            original = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if original is None:
                raise ValueError("JPEG decoding failed")
            frame = resize_frame(original, args.max_side)
            del original
            if list(frame.shape) != baseline[relative]["source"]["processed_shape"]:
                raise ValueError("Current processed image differs from detector image shape")
            begin = time.perf_counter()
            result = model.predict(source=frame, imgsz=args.imgsz, device=args.device, quantize=quantize,
                                   verbose=False, save=False)[0]
            runtime_ms = (time.perf_counter() - begin) * 1000
            if getattr(result, "depth", None) is None:
                raise ValueError("Checkpoint returned no Results.depth")
            depth = result.depth.data.detach().cpu().numpy().squeeze().astype(np.float32)
            if depth.shape != frame.shape[:2]:
                raise ValueError(f"Depth map shape {depth.shape} differs from image {frame.shape[:2]}")
            prediction = {"depth": depth, "metadata": {
                "runtime_ms": runtime_ms, "predictor_speed_ms": result.speed, "depth_units": "meters",
                "calibration_already_applied": True}}
            record = rdc.build_record(frame, prediction, capture, baseline[relative], NAME, signature,
                                      args.max_depth_m,
                                      {"method": "original_processed_pixels_no_extra_rectification"})
            valid = np.isfinite(depth) & (depth > 0)
            record.update(depth_calibration=calibration, model_config=model_config,
                          depth_map=f"depth/{key}.npz",
                          depth_map_statistics={"shape": list(depth.shape), "units": "meters", "uncapped": True,
                                                "finite_positive_fraction": float(valid.mean()),
                                                "above_person_cap_fraction": float((depth > args.max_depth_m).mean())})
            write_json(folder / "results" / f"{key}.json", record)
            np.savez_compressed(folder / "depth" / f"{key}.npz", depth=depth)
            if not cv2.imwrite(str(folder / "annotated" / f"{key}.jpg"), rdc.annotate(frame, record, LABEL)):
                raise OSError("Annotated image write failed")
            if not cv2.imwrite(str(folder / "depth_preview" / f"{key}.jpg"), rdc.depth_preview(depth)):
                raise OSError("Depth preview write failed")
            records.append(record)
            model_manifest["records"].append({
                "source": relative, "source_sha256": record["source"]["sha256"],
                "detections": len(record["people"]), "result_json": f"results/{key}.json",
                "annotated_image": f"annotated/{key}.jpg", "depth_map": f"depth/{key}.npz",
                "depth_preview": f"depth_preview/{key}.jpg"})
            print(json.dumps({"event": "processed", "index": index, "total": len(files), "source": relative,
                              "detections": len(record["people"]),
                              "positions": sum(p["status"] == "estimated" for p in record["people"])}), flush=True)
        except Exception as exc:
            failure = {"source": relative, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            model_manifest["failures"].append(failure)
            print(json.dumps({"event": "failed", **failure}), flush=True)
    done = not model_manifest["failures"] and len(records) == len(files)
    model_manifest.update(status="complete" if done else "partial",
                          summary=summarize(records, model_manifest["failures"], len(files)))
    write_json(folder / "manifest.json", model_manifest)
    run.update(status="complete" if done else "partial", failures=model_manifest["failures"],
               finished_at=utc_now(), elapsed_s=time.perf_counter() - started,
               comparison_rows=sum(len(r["people"]) for r in records))
    write_json(output / "manifest.json", run)
    print(json.dumps({"event": "finished", "status": run["status"], "output": str(output)}), flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
