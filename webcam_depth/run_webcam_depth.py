"""Webcam person depth: original/fine-tuned YOLO26, UniDepth V2, LingBot.

All models sample the same torso pixels in the same captured frame. LingBot
receives UniDepth's predicted depth, not a sensor measurement. No GPS, aircraft
controls, live dashboard imports, or calibration fitting are involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
MODEL_LABELS = {
    "yolo_original": "YOLO original",
    "yolo": "YOLO fine-tuned",
    "unidepth": "UniDepth V2",
    "lingbot": "LingBot refined UniDepth",
}
MODEL_COLUMNS = (("yolo_original", 130), ("yolo", 395), ("unidepth", 650), ("lingbot", 905))
COLORS = [(60, 220, 255), (80, 255, 90), (255, 180, 80), (255, 100, 220)]


def sample_torso(depth, bbox):
    """Median axial depth from central torso; invalid pixels never become 0 m."""
    if depth.ndim != 2:
        raise ValueError("Depth must be an HxW map")
    height, width = depth.shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return {"depth_m": None, "valid_fraction": 0.0, "roi_xyxy": None}
    left = max(0, min(width, math.floor(x1 + .35 * (x2 - x1))))
    right = max(0, min(width, math.ceil(x1 + .65 * (x2 - x1))))
    top = max(0, min(height, math.floor(y1 + .25 * (y2 - y1))))
    bottom = max(0, min(height, math.ceil(y1 + .60 * (y2 - y1))))
    patch = depth[top:bottom, left:right]
    valid = np.isfinite(patch) & (patch > 0)
    fraction = float(valid.mean()) if patch.size else 0.0
    value = float(np.median(patch[valid])) if fraction >= .5 else None
    return {"depth_m": value, "valid_fraction": fraction,
            "roi_xyxy": [left, top, right, bottom]}


def pixel_intrinsics(path, shape):
    """Optional pixel K for the exact processed resolution; no guessed FOV."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if [data.get("height"), data.get("width")] != list(shape):
        raise ValueError("Intrinsics width/height must match the actual processed frame")
    matrix = np.asarray(data.get("K"), dtype=np.float32)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError("Intrinsics JSON needs a finite 3x3 pixel K with positive fx/fy")
    return matrix


class DepthEngine:
    """One serial inference worker; hold only the active large model on CUDA."""
    def __init__(self, args):
        import torch
        from ultralytics import YOLO
        from reference_validation.depth_backends.unidepth_v2 import UniDepthV2Backend
        from lingbot_backend import LingBotDepthBackend
        self.args = args
        self.torch = torch
        torch.set_num_threads(args.torch_threads)
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if self.device.isdigit():
            self.device = "cuda:" + self.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu or the supplied launcher")
        self.detector = YOLO(str(args.detector_weights), task="detect")
        self.yolo_original = YOLO(str(args.yolo_original_weights), task="depth")
        self.yolo = YOLO(str(args.yolo_weights), task="depth")
        if (self.detector.task != "detect" or self.yolo.task != "depth"
                or self.yolo_original.task != "depth"):
            raise ValueError("Need separate person-detection and depth checkpoints")
        if self.detector.names[0].lower() != "person":
            raise ValueError("Detector class 0 must be person")
        self.unidepth = UniDepthV2Backend(
            device="cpu", model_id=str(args.unidepth_weights),
            source_dir=ROOT / "external" / "UniDepth",
            resolution_level=args.resolution_level, local_files_only=True,
        ).load()
        self.lingbot = LingBotDepthBackend(
            device=self.device, model_id=str(args.lingbot_weights),
            source_dir=ROOT / "external" / "LingBotDepth", offload=True,
        ).load()
        self.calibration = self.read_calibration(self.yolo)
        self.original_calibration = self.read_calibration(self.yolo_original)

    @staticmethod
    def read_calibration(model):
        calibration = {}
        head = model.model.model[-1]
        for name in ("cal_a", "cal_b"):
            value = getattr(head, name, None)
            calibration[name] = None if value is None else float(value.detach().cpu().item())
        return calibration

    def clear_cuda(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.empty_cache()

    def yolo_predict(self, model, frame, **kwargs):
        try:
            return model.predict(source=frame, device=self.device, quantize=32,
                                 verbose=False, save=False, **kwargs)[0]
        finally:
            # Wrapper .to also drops the cached predictor/backend GPU reference.
            model.to("cpu")
            self.clear_cuda()

    def process(self, frame, frame_id, captured_at):
        start = time.perf_counter()
        timings = {}
        detection = self.yolo_predict(self.detector, frame, imgsz=self.args.imgsz,
                                     classes=[0], conf=self.args.confidence)
        timings["detector"] = (time.perf_counter() - start) * 1000
        boxes = detection.boxes
        people = []
        if boxes is not None:
            coords = boxes.xyxy.detach().float().cpu().numpy()
            scores = boxes.conf.detach().float().cpu().numpy()
            classes = boxes.cls.detach().float().cpu().numpy()
            detections = [(box, score) for box, score, cls in zip(coords, scores, classes)
                          if cls == 0 and np.isfinite(box).all() and np.isfinite(score)
                          and score >= self.args.confidence and box[2] > box[0] and box[3] > box[1]]
            # Per-frame indices, sorted left-to-right. They are not tracked IDs.
            detections.sort(key=lambda pair: float(pair[0][0]))
            people = [{"index": i + 1, "bbox": box.tolist(), "confidence": float(score),
                       "estimates": {}} for i, (box, score) in enumerate(detections)]
        record = {"frame_id": frame_id, "captured_at": captured_at,
                  "depth_units": "meters", "depth_convention": "axial_z",
                  "lingbot_input": "unidepth_v2_predicted_depth_same_frame",
                  "device": self.device, "shape_hw": list(frame.shape[:2]),
                  "people": people, "timings_ms": timings, "metadata": {}}
        if people:
            intrinsics = pixel_intrinsics(self.args.intrinsics, frame.shape[:2])
            for name, model, weights, calibration in (
                ("yolo_original", self.yolo_original, self.args.yolo_original_weights, self.original_calibration),
                ("yolo", self.yolo, self.args.yolo_weights, self.calibration),
            ):
                stage = time.perf_counter()
                result = self.yolo_predict(model, frame, imgsz=self.args.depth_imgsz)
                if getattr(result, "depth", None) is None:
                    raise ValueError(f"{MODEL_LABELS[name]} checkpoint returned no Results.depth")
                depth = result.depth.data.detach().float().cpu().numpy().squeeze()
                self.sample(people, name, depth, frame.shape[:2])
                timings[name] = (time.perf_counter() - stage) * 1000
                record["metadata"][name] = {"weights": str(weights),
                    "saved_calibration": calibration, "extra_scale_fitted": False}
                del result, depth
            stage = time.perf_counter()
            try:
                self.unidepth.device = self.device
                self.unidepth.model.to(self.device)
                uni = self.unidepth.predict(frame, intrinsics=intrinsics)
            finally:
                self.unidepth.model.to("cpu")
                self.clear_cuda()
            self.sample(people, "unidepth", uni["depth"], frame.shape[:2])
            timings["unidepth"] = (time.perf_counter() - stage) * 1000
            record["metadata"]["unidepth"] = uni["metadata"]
            if intrinsics is None:
                predicted = uni["metadata"].get("predicted_intrinsics")
                if predicted is None:
                    raise ValueError("UniDepth did not supply estimated intrinsics for LingBot")
                intrinsics = np.asarray(predicted, dtype=np.float32)
            stage = time.perf_counter()
            ling = self.lingbot.predict(frame, uni["depth"], intrinsics=intrinsics)
            self.sample(people, "lingbot", ling["depth"], frame.shape[:2])
            timings["lingbot"] = (time.perf_counter() - stage) * 1000
            record["metadata"]["lingbot"] = ling["metadata"]
        record["processing_ms"] = (time.perf_counter() - start) * 1000
        record["processed_at"] = time.time()
        return record

    @staticmethod
    def sample(people, name, depth, shape):
        if depth.shape != shape:
            raise ValueError(f"{name} depth shape {depth.shape} does not match RGB {shape}")
        for person in people:
            person["estimates"][name] = sample_torso(depth, person["bbox"])


class LatestCapture:
    """Drain a camera on its own thread, keeping one copied latest frame."""
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="webcam-capture")

    def run(self):
        frame_id = 0
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "Camera stopped delivering frames"
                return
            frame_id += 1
            with self.lock:
                self.latest = (frame.copy(), frame_id, time.time())

    def snapshot(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.cap.release()


class InferenceWorker:
    def __init__(self, args, capture):
        self.args, self.capture = args, capture
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.status = "Loading four depth models..."
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="person-depth")

    def run(self):
        try:
            engine = DepthEngine(self.args)
            self.status = "Waiting for camera..."
            last_id = -1
            while not self.stop_event.is_set():
                snapshot = self.capture.snapshot()
                if snapshot is None or snapshot[1] == last_id:
                    self.stop_event.wait(.02)
                    continue
                frame, frame_id, captured_at = snapshot
                last_id = frame_id
                self.status = "Analyzing captured frame..."
                record = engine.process(frame, frame_id, captured_at)
                if not self.stop_event.is_set():
                    with self.lock:
                        self.latest = (frame, record)
                self.status = "Running"
                self.stop_event.wait(self.args.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


def fit_panel(frame, width=640, height=480):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    sized = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                               max(1, round(frame.shape[0] * scale))))
    top, left = (height - sized.shape[0]) // 2, (width - sized.shape[1]) // 2
    panel[top:top + sized.shape[0], left:left + sized.shape[1]] = sized
    return panel


def render(live, measured, status="", now=None):
    now = time.time() if now is None else now
    annotation = live.copy() if measured is None else measured[0].copy()
    record = None if measured is None else measured[1]
    people = [] if record is None else record["people"]
    for i, person in enumerate(people):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = [round(v) for v in person["bbox"]]
        cv2.rectangle(annotation, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotation, f"Person {person['index']}", (max(0, x1), max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
        roi = person["estimates"].get("yolo", {}).get("roi_xyxy")
        if roi is not None:
            cv2.rectangle(annotation, tuple(roi[:2]), tuple(roi[2:]), color, 1)
    rows = max(1, len(people))
    canvas = np.full((580 + rows * 35, 1280, 3), 20, dtype=np.uint8)
    canvas[35:515, :640] = fit_panel(live)
    canvas[35:515, 640:] = fit_panel(annotation)

    def label(text, x, y, color=(230, 230, 230), size=.55):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    label("LIVE webcam", 12, 25)
    age = 0 if record is None else max(0, now - record["captured_at"])
    label(f"ANALYZED frame | age {age:.1f}s | {status}", 652, 25,
          (80, 150, 255) if age > 3 else (230, 230, 230))
    label("Torso axial depth (m) | estimates | Q / Esc: quit", 12, 540)
    label("Person", 12, 570)
    for name, x in MODEL_COLUMNS:
        label(MODEL_LABELS[name], x, 570)
    if not people:
        label("No person detected" if record else "Waiting for first result...", 12, 603)
    for i, person in enumerate(people):
        y = 603 + i * 35
        color = COLORS[i % len(COLORS)]
        label(str(person["index"]), 12, y, color)
        for name, x in MODEL_COLUMNS:
            value = person["estimates"].get(name, {}).get("depth_m")
            label("unavailable" if value is None else f"{value:.2f} m", x, y, color)
    return canvas


class ResultLog:
    def __init__(self, path, args):
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.jsonl = (path / "estimates.jsonl").open("w", encoding="utf-8")
        self.csv_file = (path / "people.csv").open("w", newline="", encoding="utf-8")
        fields = ["frame_id", "captured_at", "person_index", "confidence", "yolo_original_m", "yolo_m",
                  "unidepth_m", "lingbot_refined_unidepth_m", "processing_ms"]
        self.csv = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.csv.writeheader()
        manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "depth_convention": "axial_z_meters", "lingbot_seed": "UniDepth prediction",
                    "person_indices": "per-frame, left-to-right; not persistent identities"}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def write(self, record):
        self.jsonl.write(json.dumps(record, allow_nan=False) + "\n")
        self.jsonl.flush()
        for person in record["people"]:
            estimates = person["estimates"]
            self.csv.writerow({"frame_id": record["frame_id"], "captured_at": record["captured_at"],
                "person_index": person["index"], "confidence": person["confidence"],
                "yolo_original_m": estimates["yolo_original"]["depth_m"],
                "yolo_m": estimates["yolo"]["depth_m"], "unidepth_m": estimates["unidepth"]["depth_m"],
                "lingbot_refined_unidepth_m": estimates["lingbot"]["depth_m"],
                "processing_ms": record["processing_ms"]})
        self.csv_file.flush()
        values = [f"person {p['index']}: " + ", ".join(
            f"{MODEL_LABELS[key]}={value['depth_m']:.2f}m" if value['depth_m'] is not None
            else f"{MODEL_LABELS[key]}=unavailable" for key, value in p["estimates"].items())
            for p in record["people"]]
        print(f"Frame {record['frame_id']} ({record['processing_ms'] / 1000:.2f}s): "
              + ("; ".join(values) or "no person detected"), flush=True)

    def close(self):
        self.jsonl.close()
        self.csv_file.close()


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--camera", type=int, default=0)
    command.add_argument("--image", type=Path, help="Process one saved image instead of opening a webcam")
    command.add_argument("--width", type=int, default=640)
    command.add_argument("--height", type=int, default=480)
    command.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or GPU index")
    command.add_argument("--confidence", type=float, default=.4)
    command.add_argument("--imgsz", type=int, default=640)
    command.add_argument("--depth-imgsz", type=int, default=640)
    command.add_argument("--resolution-level", type=int, default=2, choices=range(10), help="UniDepth resolution 0..9")
    command.add_argument("--interval", type=float, default=.1, help="Worker pause after a completed estimate")
    command.add_argument("--torch-threads", type=int, default=8)
    command.add_argument("--intrinsics", type=Path, help="JSON {width,height,K}; exact processed pixel intrinsics")
    command.add_argument("--detector-weights", type=Path, default=ROOT / "skydio live data" / "yolo26x.pt")
    command.add_argument("--yolo-weights", type=Path, default=ROOT / "aerialmetric_yolo26x_training" / "weights" / "best.pt")
    command.add_argument("--yolo-original-weights", type=Path, default=ROOT / "skydio live data" / "yolo26x-depth.pt")
    command.add_argument("--unidepth-weights", type=Path, default=ROOT / "models" / "unidepth_v2")
    command.add_argument("--lingbot-weights", type=Path, default=ROOT / "models" / "lingbot_depth" / "model.pt")
    command.add_argument("--output", type=Path, help="New output directory; default webcam_depth/outputs/timestamp")
    command.add_argument("--headless", action="store_true")
    command.add_argument("--max-results", type=int, help="Exit after N completed estimates")
    command.add_argument("--timeout", type=float, default=0, help="Optional maximum run seconds, including loading")
    command.add_argument("--check", action="store_true", help="Check local files/imports and GPU without opening camera")
    return command


def validate(args):
    if args.width < 32 or args.height < 32 or args.imgsz < 32 or args.depth_imgsz < 32:
        raise ValueError("Frame and inference sizes must be >=32")
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        raise ValueError("Confidence must be within (0,1]")
    if not math.isfinite(args.interval) or args.interval < 0 or args.torch_threads < 1:
        raise ValueError("Interval must be finite/nonnegative; torch-threads must be positive")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        raise ValueError("Timeout must be finite/nonnegative")
    if args.max_results is not None and args.max_results < 1:
        raise ValueError("max-results must be positive")
    for field in ("detector_weights", "yolo_original_weights", "yolo_weights", "lingbot_weights"):
        if not getattr(args, field).is_file():
            raise FileNotFoundError(f"Missing {field}: {getattr(args, field)}. Run webcam_depth/setup_webcam_depth.ps1")
    for file in ("config.json", "model.safetensors"):
        if not (args.unidepth_weights / file).is_file():
            raise FileNotFoundError(f"Missing UniDepth file: {args.unidepth_weights / file}")


def open_camera(args):
    apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for api in apis:
        cap = cv2.VideoCapture(args.camera, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open webcam {args.camera}; close other camera apps or choose --camera 1")


def main(argv=None):
    args = parser().parse_args(argv)
    log = capture = worker = None
    window = "Person depth: original YOLO / fine-tuned YOLO / UniDepth / LingBot"
    try:
        validate(args)
        if args.check:
            import torch
            sys.path.insert(0, str(ROOT / "external" / "UniDepth"))
            from unidepth.models import UniDepthV2
            import os
            os.environ["XFORMERS_DISABLED"] = "1"
            sys.path.insert(0, str(ROOT / "external" / "LingBotDepth"))
            from mdm.model.v2 import MDMModel
            print(f"Local files/imports OK | Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
            return 0
        output = args.output or ROOT / "webcam_depth" / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log = ResultLog(output, args)
        print(f"Logs: {output}\nLingBot refines UniDepth's predicted depth. All values are torso axial-depth estimates in meters.", flush=True)
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f"Cannot read image: {args.image}")
            if frame.shape[1] > args.width or frame.shape[0] > args.height:
                scale = min(args.width / frame.shape[1], args.height / frame.shape[0])
                frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            record = DepthEngine(args).process(frame, 1, time.time())
            log.write(record)
            canvas = render(frame, (frame, record), "Complete")
            cv2.imwrite(str(output / "comparison.jpg"), canvas)
            if not args.headless:
                cv2.imshow(window, canvas)
                cv2.waitKey(0)
            return 0
        capture = LatestCapture(open_camera(args))
        capture.thread.start()
        worker = InferenceWorker(args, capture)
        worker.thread.start()
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        started = time.monotonic()
        last_id, count = -1, 0
        while True:
            if worker.error:
                raise RuntimeError(worker.error)
            if capture.error:
                raise RuntimeError(capture.error)
            measured = worker.snapshot()
            if measured is not None and measured[1]["frame_id"] != last_id:
                log.write(measured[1])
                last_id = measured[1]["frame_id"]
                count += 1
                if args.max_results is not None and count >= args.max_results:
                    break
            if args.timeout and time.monotonic() - started >= args.timeout:
                raise TimeoutError(f"Run timeout reached after {count} completed estimates")
            live = capture.snapshot()
            if not args.headless and live is not None:
                cv2.imshow(window, render(live[0], measured, worker.status))
                if cv2.waitKey(20) & 0xff in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(.02)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if worker is not None:
            worker.stop()
        if capture is not None:
            capture.stop()
        if log is not None:
            log.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
