"""Run image-only metric depth backends on saved Skydio photographs.

Reuse source-verified person detections for an identical-box comparison.
Each model runs sequentially on the GPU. GPS uses embedded capture geometry;
estimated torso positions are provisional, not surveyed accuracy results.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skydio live data"))
from human_geolocation import estimate_person_position
from human_pipeline import HumanPipeline
from metadata_geometry import read_capture, apply_capture_geometry
from run_reference_batch import (CSV_FIELDS, flatten_result, result_key,
                                 sha256_file, summarize, utc_now, write_json, resize_frame)

BACKENDS = ("unidepth_v2", "depth_pro", "lingbot_depth")
DEFAULT_BACKENDS = ("unidepth_v2", "depth_pro")
LABELS = {"unidepth_v2": "UniDepth V2", "depth_pro": "Depth Pro",
          "lingbot_depth": "LingBot refined UniDepth"}
SCHEMA_VERSION = 1


def scaled_intrinsics(capture, shape):
    """Scale pixel K using the same pixel-centre convention as cv2.resize."""
    intr = capture["intrinsics"]
    values = np.array([intr.get(k) for k in ("fx", "fy", "cx", "cy")], dtype=float)
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("invalid_intrinsics")
    h, w = shape[:2]
    sx, sy = w / capture["width"], h / capture["height"]
    fx, fy, cx, cy = values
    return np.array([[fx * sx, 0, (cx + .5) * sx - .5],
                     [0, fy * sy, (cy + .5) * sy - .5], [0, 0, 1]], dtype=np.float32)


def prepare_depth_input(frame, capture):
    """Rectify radial lens distortion for pinhole models, preserving pixel K.

    The return map takes raw image pixels to rectified pixels. Remapping model
    predictions through it restores alignment with the original detection boxes.
    """
    matrix = scaled_intrinsics(capture, frame.shape)
    k1, k2, k3 = capture["distortion_k1_k2_k3"]
    distortion = np.array([k1, k2, 0, 0, k3], dtype=np.float64)
    if not np.isfinite(distortion).all():
        raise ValueError("invalid_radial_distortion")
    h, w = frame.shape[:2]
    metadata = {"method": "radial_rectification_same_K_then_depth_remap_to_original_pixels",
                "processed_intrinsics": matrix.tolist(),
                "opencv_distortion_k1_k2_p1_p2_k3": distortion.tolist(),
                "principal_point_resize": "(c+0.5)*scale-0.5",
                "source": "embedded_photo_calibration"}
    if not np.any(distortion):
        return frame, matrix, None, metadata
    map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix, (w, h), cv2.CV_32FC1)
    rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    input_valid = cv2.remap(np.ones((h, w), dtype=np.float32), map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
    y, x = np.indices((h, w), dtype=np.float32)
    points = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    # More iterations than OpenCV's default improve inversion of thermal lenses.
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10)
    if hasattr(cv2, "undistortPointsIter"):
        rectified_pixels = cv2.undistortPointsIter(points, matrix, distortion, None, matrix, criteria)
    else:  # OpenCV 5 merged the iterative API into undistortPoints.
        rectified_pixels = cv2.undistortPoints(points, matrix, distortion, R=None, P=matrix, criteria=criteria)
    rectified_pixels = rectified_pixels.reshape(h, w, 2)
    reverse = (np.ascontiguousarray(rectified_pixels[..., 0]),
               np.ascontiguousarray(rectified_pixels[..., 1]), input_valid)
    metadata["rectified_valid_fraction"] = float(input_valid.mean())
    return rectified, matrix, reverse, metadata


def restore_depth_map(depth, reverse, shape):
    array = np.asarray(depth, dtype=np.float32)
    if array.shape != tuple(shape[:2]):
        raise ValueError(f"Depth shape {array.shape} differs from image {shape[:2]}")
    valid = np.isfinite(array) & (array > 0)
    array = np.where(valid, array, 0).astype(np.float32)
    if reverse is None:
        return array
    map_x, map_y, input_valid = reverse
    valid = valid & input_valid
    array = np.where(valid, array, 0).astype(np.float32)
    restored = cv2.remap(array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    restored_valid = cv2.remap(valid.astype(np.float32), map_x, map_y,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
    return np.where(restored_valid, restored, 0).astype(np.float32)


def load_baseline(input_dir, detection_run, files, max_side):
    """Require a complete detector run and unchanged sources; zero boxes is valid."""
    manifest = json.loads((detection_run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Detection run must be complete")
    # The detection run may have been made from a copy of the photos at another path
    # (for example before the project moved drives). Identity is decided by the
    # per-file SHA-256 check below, not by where the photos live.
    if manifest.get("max_side") != max_side:
        raise ValueError("--max-side must match the detection run to preserve identical boxes")
    entries = {entry["source"]: entry for entry in manifest["records"]}
    result = {}
    for path in files:
        relative = path.relative_to(input_dir).as_posix()
        entry = entries.get(relative)
        if entry is None:
            raise ValueError(f"No detector record for {relative}")
        source_hash = sha256_file(path)
        if source_hash != entry["sha256"]:
            raise ValueError(f"Source changed since detection: {relative}")
        saved = json.loads((detection_run / entry["result_json"]).read_text(encoding="utf-8"))
        if saved["source"]["sha256"] != source_hash or saved.get("run_signature") != manifest["run_signature"]:
            raise ValueError(f"Detector record provenance differs for {relative}")
        result[relative] = saved
    return result, manifest


def create_backend(name, args):
    if name == "unidepth_v2":
        from depth_backends.unidepth_v2 import UniDepthV2Backend
        return UniDepthV2Backend(device=args.device, model_id=str(args.unidepth_model.resolve()),
                                 resolution_level=args.unidepth_resolution)
    if name == "lingbot_depth":
        from depth_backends.lingbot_depth import LingBotUniDepthBackend
        return LingBotUniDepthBackend(device=args.device, unidepth_model_id=args.unidepth_model.resolve(),
                                      unidepth_resolution=args.unidepth_resolution,
                                      lingbot_model_id=args.lingbot_checkpoint.resolve(),
                                      lingbot_resolution=args.lingbot_resolution)
    from depth_backends.depth_pro import DepthProBackend
    return DepthProBackend(device=args.device, checkpoint_path=args.depthpro_checkpoint,
                           precision=args.depthpro_precision)


def depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    lower, upper = np.percentile(depth[valid], (2, 98)) if valid.any() else (0, 1)
    if upper <= lower:
        upper = lower + 1
    normalized = np.clip((depth - lower) / (upper - lower), 0, 1)
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    cv2.putText(image, f"Axial depth: {lower:.1f}-{upper:.1f} m (2-98 percentiles)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def annotate(frame, record, label):
    image = HumanPipeline._annotate(frame, record)
    for person in record["people"]:
        depth = person.get("depth_m")
        y = min(frame.shape[0] - 35, max(40, round(person["bbox"][3]) + 17))
        x = max(0, min(frame.shape[1] - 280, round(person["bbox"][0])))
        text = f"{label}: {depth:.2f} m Z" if depth is not None else f"{label}: depth unavailable"
        cv2.rectangle(image, (x, y - 13), (min(frame.shape[1], x + 290), y + 4), (15, 15, 15), -1)
        cv2.putText(image, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 27), (20, 20, 20), -1)
    cv2.putText(image, f"{label} | estimated torso depth/GPS | geometry provisional", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def build_record(frame, depth_output, capture, baseline, name, signature, max_depth_m, input_metadata):
    people = []
    for box in baseline["people"]:
        # This function computes the established median torso ROI and preserves
        # the sample even with empty live telemetry. GPS is then calculated below.
        person = estimate_person_position(box["bbox"], depth_output["depth"], {}, frame.shape,
                                          depth_mode="axial", max_depth_m=max_depth_m)
        person.update(index=box["index"], bbox=box["bbox"], confidence=box["confidence"],
                      identity=None, bbox_original=box.get("bbox_original"))
        confidence = depth_output.get("confidence")
        roi = person.get("depth_roi")
        if confidence is not None and roi:
            x1, y1, x2, y2 = roi
            values = confidence[y1:y2, x1:x2]
            local_depth = depth_output["depth"][y1:y2, x1:x2]
            values = values[np.isfinite(values) & np.isfinite(local_depth) &
                            (local_depth > 0) & (local_depth <= max_depth_m)]
            person["depth_confidence"] = float(np.median(values)) if values.size else None
        people.append(person)
    record = {"schema_version": SCHEMA_VERSION, "backend": name, "depth_model": LABELS[name],
              "run_signature": signature, "frame_id": baseline["frame_id"],
              "captured_at": None, "processed_at": utc_now(), "people": people,
              "source": baseline["source"], "capture_metadata": capture,
              "depth_metadata": {**depth_output["metadata"], "input_geometry": input_metadata},
              "detection_provenance": {"run_signature": baseline["run_signature"],
                                       "model": baseline["model_config"]["detector_weights"],
                                       "boxes": "reused_same_hash_and_same_processed_dimensions"},
              "validation_mode": "offline_metric_depth_provisional_metadata_geometry",
              "identity_matching": "unassigned", "alignment": baseline["alignment"],
              "depth_calibration": {"status": "pretrained_metric_scale_camera_unvalidated"},
              "inference_ms": depth_output["metadata"].get("runtime_ms")}
    return apply_capture_geometry(record, capture, frame.shape, max_depth_m=max_depth_m)


def load_complete_manifests(output, baseline, detection_manifest, names):
    """Saved, complete model runs for exactly these photos and this detector run."""
    result = {}
    for name in names:
        path = output / name / "manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("status") == "complete" and manifest.get("selection") == list(baseline)
                and manifest.get("detector_run_signature") == detection_manifest["run_signature"]):
            result[name] = manifest
    return result


def merge_manifests(output, baseline, detection_manifest, manifests):
    """Report every completed model, so a single-model run never drops the others."""
    saved = load_complete_manifests(output, baseline, detection_manifest,
                                    [name for name in BACKENDS if name not in manifests])
    return {name: manifests.get(name) or saved[name] for name in BACKENDS if name in manifests or name in saved}


def export_csv(path, records, name):
    fields = ["backend", "model", "depth_confidence", *CSV_FIELDS]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            relative = f"results/{result_key(record['source']['relative_path'])}.json"
            for person, row in zip(record["people"], flatten_result(record, relative)):
                writer.writerow({"backend": name, "model": LABELS[name],
                                 "depth_confidence": person.get("depth_confidence"), **row})


def export_report(output, manifests, baseline):
    """Build an offline comparison CSV and an interactive per-photo review."""
    fields = ["source", "photo_kind", "camera_source", "person_index", "model", "depth_m",
              "slant_range_m", "latitude", "longitude", "status", "reason", "depth_confidence",
              "north_m", "east_m", "ground_latitude", "ground_longitude", "result_json",
              "estimate_point"]
    rows, images = [], {}

    def fusion_row(row, label):
        # Gated pick between this model's depth ray and the shared plane; estimate_point names the pick.
        return {"source": row["source"], "photo_kind": row["photo_kind"], "camera_source": row["camera_source"],
                "person_index": row["person_index"], "model": f"Fusion ({label} + ground plane)",
                "estimate_point": "bbox_bottom_centre" if row["fusion_source"] == "ground_plane" else "torso",
                "depth_m": None, "slant_range_m": row["fusion_range_m"],
                "latitude": row["fusion_latitude"], "longitude": row["fusion_longitude"],
                "status": row["fusion_status"], "reason": row["fusion_reason"], "depth_confidence": None,
                "north_m": row["fusion_north_m"], "east_m": row["fusion_east_m"],
                "ground_latitude": row["ground_latitude"], "ground_longitude": row["ground_longitude"],
                "result_json": row["result_json"]}

    for relative, record in baseline.items():
        images[relative] = {"source": relative, "camera": record["capture_metadata"]["camera_source"],
                            "kind": record["capture_metadata"]["photo_kind"],
                            "detection_count": len(record["people"]), "models": {}}
        for person, row in zip(record["people"], flatten_result(record, "")):
            rows.append({key: row.get(key) for key in fields} |
                        {"model": "YOLO26-depth (saved baseline)", "estimate_point": "torso"})
            rows.append(fusion_row(row, "YOLO26-depth"))
            ground = person.get("ground_plane") or {}
            # This independent geometric estimate is shared by every model.
            # ground_range_m is the camera-to-plane slant range, not axial Z.
            rows.append({"source": relative, "photo_kind": images[relative]["kind"],
                         "camera_source": images[relative]["camera"], "person_index": person["index"],
                         "model": "Ground plane (box bottom)", "estimate_point": "bbox_bottom_centre",
                         "depth_m": None, "slant_range_m": ground.get("ground_range_m"),
                         "latitude": ground.get("latitude"), "longitude": ground.get("longitude"),
                         "status": ground.get("status", "unavailable"),
                         "reason": ground.get("reason") or ("ground_estimate_missing" if not ground else None),
                         "north_m": ground.get("north_m"), "east_m": ground.get("east_m"),
                         "ground_latitude": ground.get("latitude"),
                         "ground_longitude": ground.get("longitude"), "result_json": ""})
    for name, manifest in manifests.items():
        for entry in manifest["records"]:
            record = json.loads((output / name / entry["result_json"]).read_text(encoding="utf-8"))
            relative = entry["source"]
            images[relative]["models"][name] = {"annotated": f"{name}/{entry['annotated_image']}",
                                               "depth": f"{name}/{entry['depth_preview']}",
                                               "label": LABELS[name]}
            for person, row in zip(record["people"], flatten_result(record, f"{name}/{entry['result_json']}")):
                rows.append({key: row.get(key) for key in fields} |
                            {"model": LABELS[name], "depth_confidence": person.get("depth_confidence"),
                             "estimate_point": "torso"})
                rows.append(fusion_row(row, LABELS[name]))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    visible_images = [image for image in images.values() if image["detection_count"] == 4]
    visible_sources = {image["source"] for image in visible_images}
    overview = {"models": {name: {"status": m["status"], "summary": m.get("summary")}
                           for name, m in manifests.items()},
                "input_images": len(baseline), "detections_per_model": sum(len(r["people"]) for r in baseline.values()),
                "ground_plane": {"total": sum(r["model"] == "Ground plane (box bottom)" for r in rows),
                                 "estimated": sum(r["model"] == "Ground plane (box bottom)" and r["status"] == "estimated" for r in rows),
                                 "unavailable": sum(r["model"] == "Ground plane (box bottom)" and r["status"] == "unavailable" for r in rows)},
                "meaning": "Pretrained torso depth/GPS and provisional box-bottom ground-plane estimates. No measured accuracy ranking.",
                "viewer_filter": {"person_detections": 4, "matching_images": len(visible_images)},
                "report": "review_depth_comparison.html", "csv": "comparison.csv"}
    write_json(output / "summary.json", overview)
    data = json.dumps({"images": visible_images,
                       "rows": [row for row in rows if row["source"] in visible_sources]},
                      allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Skydio depth comparison</title>
<style>body{font:15px system-ui;margin:24px;color:#182333;background:#f5f7fa}h1{font-size:24px}
select{padding:8px;min-width:320px}.views{display:flex;gap:18px;flex-wrap:wrap}.card{flex:1;min-width:300px;background:white;padding:12px;border-radius:8px}
img{width:100%;max-height:460px;object-fit:contain;background:#121212}table{border-collapse:collapse;width:100%;background:white;margin-top:20px}
td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}tr.ground{background:#edf6f0}a{color:#165b9c}.note{max-width:1050px;line-height:1.6}</style>
<h1>Person depth and GPS comparison</h1><p class="note">UniDepth V2, Depth Pro and LingBot refined UniDepth (a refinement of the UniDepth V2 depth map, not an independent measurement) use the same saved YOLO person boxes, calibrated camera parameters and embedded photo pose.
Model depth means camera-axis Z at the torso; model GPS is an estimated torso position. Ground plane uses the box-bottom centre ray intersected with an assumed flat launch plane (absolute minus relative altitude).
Its Slant value is camera-to-plane range; its Depth Z is left blank because it uses a different sample point. The box bottom may not be the feet. Thermal model accuracy and metadata geometry remain unvalidated.
Fusion rows take the ground plane when the box-bottom ray is at least 10° below the horizon and that model's depth ray otherwise.
The dataset includes visible, infrared normal and infrared RJPEG images. This viewer shows only photos with exactly four saved person detections. Person indices are local to each image.</p>
<p><a href="comparison.csv">Download all estimates (CSV)</a> · <a href="summary.json">Run summary</a></p>
<p id="filter-note"></p><label>Photograph: <select id="photo"></select></label><span id="kind"></span><div id="views" class="views"></div><table><thead><tr>
<th>Person</th><th>Estimator</th><th>Depth Z (m)</th><th>Slant (m)</th><th>Latitude</th><th>Longitude</th><th>Status</th></tr></thead><tbody id="people"></tbody></table>
<script>const data=DATA;const photo=document.querySelector('#photo');
const fmt=(v,n)=>v==null?'—':Number(v).toFixed(n);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const order={'YOLO26-depth (saved baseline)':0,'UniDepth V2':1,'LingBot refined UniDepth':2,'Depth Pro':3,'Ground plane (box bottom)':4};
data.images.forEach((im,i)=>{const o=document.createElement('option');o.value=i;o.textContent=im.source;photo.append(o)});
photo.disabled=!data.images.length;document.querySelector('#filter-note').textContent=data.images.length ? `Showing ${data.images.length} photos with exactly 4 person detections.` : 'No photos have exactly 4 person detections.';
function show(){const im=data.images[photo.value];if(!im){document.querySelector('#kind').textContent='';document.querySelector('#views').innerHTML='';document.querySelector('#people').innerHTML='';return}document.querySelector('#kind').textContent='  '+im.camera+' / '+im.kind;
document.querySelector('#views').innerHTML=Object.values(im.models).map(m=>`<div class="card"><h2>${esc(m.label)}</h2><img src="${esc(m.annotated)}" alt="Annotated person estimates"><details><summary>Full-frame depth map</summary><img src="${esc(m.depth)}" alt="Depth visualization"></details></div>`).join('');
document.querySelector('#people').innerHTML=data.rows.filter(r=>r.source===im.source).sort((a,b)=>a.person_index-b.person_index || (order[a.model]??99)-(order[b.model]??99)).map(r=>`<tr class="${r.estimate_point==='bbox_bottom_centre'?'ground':''}"><td>${r.person_index}</td><td>${esc(r.model)}</td><td>${fmt(r.depth_m,2)}</td><td>${fmt(r.slant_range_m,2)}</td><td>${fmt(r.latitude,7)}</td><td>${fmt(r.longitude,7)}</td><td>${esc(r.status)} ${esc(r.reason)}</td></tr>`).join('')}
photo.onchange=show;const first=data.images.findIndex(im=>data.rows.some(r=>r.source===im.source));photo.value=first<0?0:first;show();</script></html>""".replace("DATA", data)
    (output / "review_depth_comparison.html").write_text(page, encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=ROOT / "Geolocation Reference Data")
    result.add_argument("--detection-run", type=Path, default=ROOT / "reference_validation/outputs/photos")
    result.add_argument("--output", type=Path, default=ROOT / "reference_validation/outputs/unidepth_depthpro_latest")
    result.add_argument("--models", nargs="+", choices=BACKENDS, default=list(DEFAULT_BACKENDS),
                        help="lingbot_depth refines UniDepth V2 and is opt-in; completed runs in --output are always reported")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--max-side", type=int, default=2048)
    result.add_argument("--max-depth-m", type=float, default=300., help="Explicit validity limit for axial depth AND slant range")
    result.add_argument("--unidepth-model", type=Path, default=ROOT / "models/unidepth_v2")
    result.add_argument("--unidepth-resolution", type=int, default=5)
    result.add_argument("--depthpro-checkpoint", type=Path, default=ROOT / "models/depth_pro.pt")
    result.add_argument("--depthpro-precision", choices=("float16", "float32"), default="float16")
    result.add_argument("--lingbot-checkpoint", type=Path, default=ROOT / "models/lingbot_depth/model.pt")
    result.add_argument("--lingbot-resolution", type=int, default=0)
    result.add_argument("--limit", type=int)
    result.add_argument("--files", nargs="+")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--report-only", action="store_true",
                        help="Rebuild the comparison CSV/HTML from completed saved results, without model inference")
    return result


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.max_side < 64 or not 0 < args.max_depth_m <= 1000 or (args.limit is not None and args.limit < 1):
        ap.error("Invalid image size, depth limit, or subset limit")
    if len(set(args.models)) != len(args.models):
        ap.error("Duplicate model selection")
    input_dir, output = args.input.resolve(), args.output.resolve()
    all_files = sorted((p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")),
                       key=lambda p: p.relative_to(input_dir).as_posix())
    files = all_files
    if args.files:
        requested = set(args.files)
        files = [p for p in files if p.relative_to(input_dir).as_posix() in requested]
        if len(files) != len(requested):
            ap.error("One or more selected files are missing")
    if args.limit:
        files = files[:args.limit]
    if not files:
        ap.error("No JPEG photographs selected")
    baseline, detection_manifest = load_baseline(input_dir, args.detection_run.resolve(), files, args.max_side)
    if args.report_only:
        manifests = {}
        for name in args.models:
            manifest = json.loads((output / name / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or manifest.get("selection") != list(baseline):
                ap.error("Report-only requires completed model results matching the selected photos")
            if manifest.get("detector_run_signature") != detection_manifest["run_signature"]:
                ap.error("Saved model results use a different detector run")
            records = []
            for entry in manifest["records"]:
                saved = json.loads((output / name / entry["result_json"]).read_text(encoding="utf-8"))
                if (saved.get("run_signature") != manifest["run_signature"] or
                        saved["source"]["sha256"] != baseline[entry["source"]]["source"]["sha256"]):
                    ap.error("Saved model result provenance differs")
                records.append(saved)
            export_csv(output / name / "detections.csv", records, name)
            manifests[name] = manifest
        export_report(output, merge_manifests(output, baseline, detection_manifest, manifests), baseline)
        print(json.dumps({"event": "report_regenerated", "output": str(output)}), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    packages = {}
    for package in ("torch", "torchvision", "numpy", "timm", "huggingface-hub", "unidepth", "depth_pro", "opencv-python"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "packages": packages, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    code_paths = [Path(__file__), Path(__file__).with_name("metadata_geometry.py"),
                  ROOT / "skydio live data/human_geolocation.py",
                  Path(__file__).parent / "depth_backends/unidepth_v2.py",
                  Path(__file__).parent / "depth_backends/depth_pro.py"]
    if "lingbot_depth" in args.models:
        code_paths += [Path(__file__).parent / "depth_backends/lingbot_depth.py",
                       ROOT / "webcam_depth/lingbot_backend.py"]
    code_hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in code_paths}
    official_artifacts = json.loads((ROOT / "models/depth_model_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {str(Path(e["path"]).resolve()): e["sha256"] for a in official_artifacts for e in a["files"]}
    lingbot_manifest = ROOT / "models/lingbot_depth/manifest.json"
    if lingbot_manifest.is_file():
        pinned = json.loads(lingbot_manifest.read_text(encoding="utf-8"))
        expected_hashes[str(Path(pinned["path"]).resolve())] = pinned["sha256"]
    selected_paths = {"unidepth_v2": [args.unidepth_model.resolve() / "config.json",
                                     args.unidepth_model.resolve() / "model.safetensors"],
                      "depth_pro": [args.depthpro_checkpoint.resolve()],
                      "lingbot_depth": [args.lingbot_checkpoint.resolve(),
                                        args.unidepth_model.resolve() / "config.json",
                                        args.unidepth_model.resolve() / "model.safetensors"]}
    artifacts, official_sources = {}, {}
    for name in args.models:
        artifacts[name] = []
        for path in selected_paths[name]:
            actual_hash = sha256_file(path)
            expected_hash = expected_hashes.get(str(path.resolve()))
            if expected_hash is not None and expected_hash != actual_hash:
                raise ValueError("Model artifact changed since download manifest")
            artifacts[name].append({"path": str(path), "bytes": path.stat().st_size,
                                    "sha256": actual_hash, "official_download_match": expected_hash == actual_hash})
        def source_record(folder):
            source_dir = ROOT / "external" / folder
            revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
            return {"path": str(source_dir), "revision": revision,
                    "python_hashes": {p.relative_to(source_dir).as_posix(): sha256_file(p)
                                      for p in sorted(source_dir.rglob("*.py"))}}
        source_folder = {"unidepth_v2": "UniDepth", "depth_pro": "DepthPro", "lingbot_depth": "LingBotDepth"}[name]
        official_sources[name] = source_record(source_folder)
        if name == "lingbot_depth":
            official_sources[name]["seed_source"] = source_record("UniDepth")
    manifests, complete = {}, True
    for name in args.models:
        folder = output / name
        for child in ("results", "annotated", "depth", "depth_preview"):
            (folder / child).mkdir(parents=True, exist_ok=True)
        signature_payload = {"schema_version": SCHEMA_VERSION, "backend": name,
            "input": str(input_dir), "selection": list(baseline),
            "input_hashes": {k: v["source"]["sha256"] for k, v in baseline.items()},
            "detector_record_hashes": {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                                       for k, v in baseline.items()},
            "detector_run_signature": detection_manifest["run_signature"],
            "max_side": args.max_side, "max_depth_m": args.max_depth_m,
            "unidepth_resolution": args.unidepth_resolution, "depthpro_precision": args.depthpro_precision,
            "device": args.device, "model_artifacts": artifacts[name], "official_source": official_sources[name],
            "code_hashes": code_hashes, "runtime": runtime}
        if name == "lingbot_depth":
            signature_payload["lingbot_resolution"] = args.lingbot_resolution
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not args.resume:
                ap.error(f"Output exists: {folder}; use new output or --resume")
            if old["run_signature"] != signature:
                ap.error("Resume signature differs; choose a new output directory")
        manifest = {**signature_payload, "run_signature": signature, "status": "running", "started_at": utc_now(),
                    "detection_run_input": detection_manifest["input"],
                    "selected_images": len(files), "dataset_images": len(all_files),
                    "subset_run": len(files) != len(all_files), "records": [], "failures": [],
                    "ground_truth_used_for_fitting": False,
                    "limitations": ["RGB pretrained depth on aerial thermal is unvalidated.",
                        "GPS is a model-estimated torso point, using provisional embedded capture geometry.",
                        "Same source-verified detector boxes, including zero-detection images, are reused for every model.",
                        "No depth scale fitted to reference coordinates; no accuracy ranking is inferred.",
                        "Depth validity limit is explicitly recorded; values beyond it yield unavailable GPS."]}
        write_json(manifest_path, manifest)
        records, failures, backend = [], [], None
        started = time.perf_counter()
        try:
            print(json.dumps({"event": "loading_model", "backend": name, "images": len(files)}), flush=True)
            backend = create_backend(name, args).load()
            for index, path in enumerate(files, 1):
                relative = path.relative_to(input_dir).as_posix()
                key = result_key(relative)
                destination = folder / "results" / f"{key}.json"
                annotated_path = folder / "annotated" / f"{key}.jpg"
                depth_path = folder / "depth" / f"{key}.npz"
                preview_path = folder / "depth_preview" / f"{key}.jpg"
                try:
                    reused = False
                    if args.resume and all(p.exists() for p in (destination, annotated_path, depth_path, preview_path)):
                        candidate = json.loads(destination.read_text(encoding="utf-8"))
                        if candidate.get("run_signature") == signature and candidate["source"]["sha256"] == baseline[relative]["source"]["sha256"]:
                            record, reused = candidate, True
                    if not reused:
                        capture = read_capture(path)
                        original = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        if original is None:
                            raise ValueError("JPEG decoding failed")
                        frame = resize_frame(original, args.max_side)
                        del original
                        if list(frame.shape) != baseline[relative]["source"]["processed_shape"]:
                            raise ValueError("Current processed image differs from detector image shape")
                        model_frame, matrix, reverse, preprocessing = prepare_depth_input(frame, capture)
                        prediction = backend.predict(model_frame, intrinsics=matrix)
                        prediction["depth"] = restore_depth_map(prediction["depth"], reverse, frame.shape)
                        if "confidence" in prediction:
                            prediction["confidence"] = restore_depth_map(prediction["confidence"], reverse, frame.shape)
                        record = build_record(frame, prediction, capture, baseline[relative], name, signature,
                                              args.max_depth_m, preprocessing)
                        record["depth_map"] = depth_path.relative_to(folder).as_posix()
                        write_json(destination, record)
                        arrays = {k: prediction[k] for k in ("depth", "confidence") if k in prediction}
                        np.savez_compressed(depth_path, **arrays)
                        if not cv2.imwrite(str(annotated_path), annotate(frame, record, LABELS[name])):
                            raise OSError("Annotated image write failed")
                        if not cv2.imwrite(str(preview_path), depth_preview(prediction["depth"])):
                            raise OSError("Depth preview write failed")
                    records.append(record)
                    entry = {"source": relative, "result_json": destination.relative_to(folder).as_posix(),
                             "annotated_image": annotated_path.relative_to(folder).as_posix(),
                             "depth_map": depth_path.relative_to(folder).as_posix(),
                             "depth_preview": preview_path.relative_to(folder).as_posix(),
                             "detections": len(record["people"]), "reused": reused}
                    manifest["records"].append(entry)
                    print(json.dumps({"event": "processed", "backend": name, "index": index,
                        "total": len(files), "source": relative, "detections": len(record["people"]),
                        "positions": sum(p["status"] == "estimated" for p in record["people"]),
                        "inference_ms": record.get("inference_ms"), "reused": reused}), flush=True)
                except Exception as exc:
                    failure = {"source": relative, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    failures.append(failure)
                    print(json.dumps({"event": "failed", "backend": name, **failure}), flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                manifest.update(updated_at=utc_now(), elapsed_s=time.perf_counter()-started,
                                failures=failures, summary=summarize(records, failures, len(files)))
                write_json(manifest_path, manifest)
        except BaseException as exc:
            manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
            manifest["fatal_traceback"] = traceback.format_exc()
            print(json.dumps({"event": "fatal", "backend": name, "error": manifest["fatal_error"],
                              "traceback": manifest["fatal_traceback"]}), flush=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            export_csv(folder / "detections.csv", records, name)
            done = len(records) == len(files) and not failures and not manifest.get("fatal_error")
            complete = complete and done
            manifest.update(status="complete" if done else "partial", finished_at=utc_now(),
                            elapsed_s=time.perf_counter()-started, summary=summarize(records, failures, len(files)))
            write_json(manifest_path, manifest)
            manifests[name] = manifest
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({"event": "model_finished", "backend": name, "status": manifest["status"],
                              "summary": manifest["summary"]}), flush=True)
    export_report(output, merge_manifests(output, baseline, detection_manifest, manifests), baseline)
    print(json.dumps({"event": "finished", "complete": complete, "output": str(output)}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
