"""Latest-frame human detection, YOLO26x depth and a ground-plane cross-check.

RTSP capture and Flask call only the nonblocking snapshot methods. One worker
loads the local models and serially infers on a copied frame. CoT is selected
before inference by local receive time; that is not sensor-clock synchronization.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from human_geolocation import (estimate_ground_plane_position, estimate_person_position, fuse_estimates,
                               horizontal_disagreement_m)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "detector_weights": str(BASE_DIR / "yolo26x.pt"),
    "depth_weights": str(BASE_DIR / "yolo26x-depth.pt"),
    "device": "auto",
    "detector_imgsz": 640,
    # True caps the detector input at the frame's own long side (stride 32), so a
    # large detector_imgsz helps big photos without upscaling small thermal frames.
    "detector_no_upscale": False,
    "depth_imgsz": 768,
    "confidence": 0.4,
    "max_fps": 3.0,
    "depth_mode": "axial",
    "depth_scale": 1.0,
    "max_depth_m": 100.0,
    "frame_max_age_s": 3.0,
    "telemetry_max_age_s": 3.0,
    # Surveyed ground ellipsoid height (m) for the ray/ground-plane estimate.
    # None falls back to the first aircraft HAE seen with a real GPS fix (not the
    # 0,0 no-fix placeholder), labelled as launch-point ground.
    "ground_hae_m": None,
    "log_dir": str(BASE_DIR / "telemetry_logs"),
}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _json_safe(value):
    """Strict JSON only: bad numeric input becomes null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


class HumanPipeline:
    """Bounded single-worker inference, with copied thread-safe result snapshots.

    ``frame_provider()`` returns frame/frame_id/captured_at or None.
    ``telemetry_provider(captured_at)`` returns one coherent CoT snapshot or {}.
    ``stop()`` signals cancellation and waits at most two seconds; an in-progress
    model call finishes on its daemon thread without publishing after shutdown.
    """

    def __init__(self, frame_provider, telemetry_provider, config=None):
        self.config = dict(DEFAULT_CONFIG)
        supplied = dict(config or {})
        for alias, canonical in (("max_depth", "max_depth_m"),
                                 ("frame_max_age", "frame_max_age_s"),
                                 ("telemetry_max_age", "telemetry_max_age_s")):
            if alias in supplied and canonical not in supplied:
                supplied[canonical] = supplied.pop(alias)
        self.config.update(supplied)
        for key in ("max_fps", "max_depth_m", "frame_max_age_s",
                    "telemetry_max_age_s", "depth_scale"):
            number = _finite(self.config[key])
            if number is None or number <= 0:
                raise ValueError(f"{key} must be finite and positive")
            self.config[key] = number
        confidence = _finite(self.config["confidence"])
        if confidence is None or not 0 < confidence <= 1:
            raise ValueError("confidence must be within (0, 1]")
        self.config["confidence"] = confidence
        if self.config["ground_hae_m"] is not None:
            ground = _finite(self.config["ground_hae_m"])
            if ground is None or not -12000 <= ground <= 100000:
                raise ValueError("ground_hae_m must be a finite ellipsoid height in metres, or None")
            self.config["ground_hae_m"] = ground
        self._launch_hae = None
        self.frame_provider = frame_provider
        self.telemetry_provider = telemetry_provider
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._detector = None
        self._depth_model = None
        self._device = "pending"
        self._half = False
        self._calibration = {"status": "not_loaded"}
        self._log_path = None
        self._result = self._empty_result("stopped")

    def _empty_result(self, status, error=None):
        return {
            "status": status, "error": error, "frame_id": None,
            "captured_at": None, "processed_at": None, "inference_ms": None,
            "fps": 0.0, "people": [], "telemetry_uid": None,
            "telemetry_age_s": None, "device": self._device,
            "depth_model": Path(self.config["depth_weights"]).name,
            "detector_model": Path(self.config["detector_weights"]).name,
            "depth_calibration": self._calibration,
            "ground_hae_m": self._ground_reference()[0],
            "ground_source": self._ground_reference()[1],
            "warnings": [], "result_age_s": None, "stale": False,
            "alignment": "local capture and CoT receive times; not sensor synchronized",
        }

    @staticmethod
    def _has_gps_fix(telemetry):
        """True for finite lat/lon/HAE that is not the 0,0 no-fix placeholder."""
        lat, lon = _finite(telemetry.get("latitude")), _finite(telemetry.get("longitude"))
        hae = _finite(telemetry.get("altitude_hae"))
        return lat is not None and lon is not None and hae is not None and (lat, lon) != (0.0, 0.0)

    def _ground_reference(self):
        """Ground HAE for the plane intersection and where it came from."""
        if self.config["ground_hae_m"] is not None:
            return self.config["ground_hae_m"], "configured"
        if self._launch_hae is not None:
            return self._launch_hae, "launch_hae_assumed_ground"
        return None, "unavailable"

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._launch_hae = None
        with self._lock:
            self._frame = None
            self._result = self._empty_result("loading")
        self._thread = threading.Thread(target=self._run,
                                        name="human-depth-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._frame = None
            self._result = self._empty_result("stopped")

    def get_result(self):
        with self._lock:
            result = copy.deepcopy(self._result)
        captured_at = result.get("captured_at")
        if captured_at is not None:
            result["result_age_s"] = max(0.0, time.time() - captured_at)
            result["stale"] = result["result_age_s"] > self.config["frame_max_age_s"]
            if result["stale"] and result["status"] == "running":
                result["status"] = "stale"
                result["warnings"].append("Inference frame is stale; estimates are historical.")
        return _json_safe(result)

    def get_frame(self):
        with self._lock:
            captured_at = self._result.get("captured_at")
            if self._frame is None or captured_at is None:
                return None
            if time.time() - captured_at > self.config["frame_max_age_s"]:
                return None
            return self._frame.copy()

    def _set_status(self, status, error=None):
        with self._lock:
            self._frame = None
            self._result = self._empty_result(status, error)

    def _load_models(self):
        # Refuse missing paths before YOLO can initiate an automatic download.
        for key in ("detector_weights", "depth_weights"):
            if not Path(self.config[key]).is_file():
                raise FileNotFoundError(f"Missing local {key}: {self.config[key]}")
        import torch
        from ultralytics import YOLO

        requested = str(self.config["device"])
        self._device = ("0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        self._half = self._device.lower() not in ("cpu", "mps")
        self._detector = YOLO(str(self.config["detector_weights"]), task="detect")
        self._depth_model = YOLO(str(self.config["depth_weights"]), task="depth")
        if self._detector.task != "detect" or self._depth_model.task != "depth":
            raise ValueError("Need separate YOLO26x detect and YOLO26x-depth checkpoints")
        names = self._detector.names
        if str(names.get(0, "") if isinstance(names, dict) else names[0]).lower() != "person":
            raise ValueError("Detector class 0 must be person")
        self._calibration = self._read_calibration(self._depth_model)

    @staticmethod
    def _read_calibration(model):
        checkpoint = getattr(model, "ckpt", None) or {}
        provenance = checkpoint.get("depth_calibration")
        head = model.model.model[-1]
        cal_a = _finite(_numpy(head.cal_a).item()) if hasattr(head, "cal_a") else None
        cal_b = _finite(_numpy(head.cal_b).item()) if hasattr(head, "cal_b") else None
        present = provenance is not None or (cal_a is not None and cal_b is not None and
                                            (cal_a != 1.0 or cal_b != 0.0))
        return _json_safe({
            "status": "pretrained_metric_scale_camera_unvalidated" if present else "calibration_unverified",
            "cal_a": cal_a, "cal_b": cal_b,
            "checkpoint_provenance": provenance,
        })

    def _select_telemetry(self, captured_at):
        snapshot = copy.deepcopy(self.telemetry_provider(captured_at) or {})
        received_at = _finite(snapshot.get("_received_at"))
        age = captured_at - received_at if received_at is not None else None
        if received_at is None:
            return {}, None, "No timestamped telemetry matched this frame."
        if age < -0.05:
            return {}, age, "Telemetry arrived after this frame; position withheld."
        if age > self.config["telemetry_max_age_s"]:
            return {}, age, "Telemetry is too old for this frame; position withheld."
        return snapshot, max(age, 0.0), None

    def _run(self):
        try:
            self._load_models()
            if self.config["log_dir"]:
                log_dir = Path(self.config["log_dir"])
                log_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                self._log_path = log_dir / f"human_estimates_{stamp}.jsonl"
        except Exception as exc:
            if not self._stop.is_set():
                self._set_status("error", f"Model initialization failed: {type(exc).__name__}: {exc}")
            return
        if self._stop.is_set():
            return
        self._set_status("waiting_for_frame")
        last_frame_id = None
        previous_completed = None
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                snapshot = self.frame_provider()
                if not snapshot or snapshot.get("frame") is None:
                    self._stop.wait(0.05)
                    continue
                frame_id = snapshot.get("frame_id")
                if frame_id is None or frame_id == last_frame_id:
                    self._stop.wait(0.02)
                    continue
                captured_at = _finite(snapshot.get("captured_at"))
                if captured_at is None or abs(time.time() - captured_at) > self.config["frame_max_age_s"]:
                    last_frame_id = frame_id
                    self._set_status("waiting_for_fresh_frame")
                    self._stop.wait(0.05)
                    continue
                frame = np.asarray(snapshot["frame"]).copy()
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                    raise ValueError("frame_provider must return a nonempty HxWx3 BGR frame")
                # Copy telemetry now, not when slow GPU/CPU inference eventually completes.
                telemetry, telemetry_age, telemetry_warning = self._select_telemetry(captured_at)
                last_frame_id = frame_id
                result, annotated = self._process_frame(frame, frame_id, captured_at,
                                                        telemetry, telemetry_age, telemetry_warning)
                completed = time.perf_counter()
                interval = completed - previous_completed if previous_completed is not None else 0.0
                result["fps"] = 1.0 / interval if interval > 0 else 0.0
                previous_completed = completed
                if self._stop.is_set():
                    break
                self._write_log(result)
                with self._lock:
                    self._result = _json_safe(result)
                    self._frame = annotated
            except Exception as exc:
                if not self._stop.is_set():
                    self._set_status("error", f"Inference failed: {type(exc).__name__}: {exc}")
                self._stop.wait(1.0)
            self._stop.wait(max(0.0, 1.0 / self.config["max_fps"] - (time.perf_counter() - started)))

    def _process_frame(self, frame, frame_id, captured_at, telemetry, telemetry_age, telemetry_warning):
        begin = time.perf_counter()
        options = {"device": self._device, "quantize": 16 if self._half else 32,
                   "verbose": False, "save": False}
        detector_imgsz = self.config["detector_imgsz"]
        if self.config.get("detector_no_upscale"):
            detector_imgsz = min(detector_imgsz, -(-max(frame.shape[:2]) // 32) * 32)
        detections = self._detector.predict(source=frame, imgsz=detector_imgsz,
                                            classes=[0], conf=self.config["confidence"], **options)[0]
        depth_result = self._depth_model.predict(source=frame, imgsz=self.config["depth_imgsz"], **options)[0]
        if getattr(depth_result, "depth", None) is None:
            raise ValueError("YOLO26x-depth returned no Results.depth; verify checkpoint and Ultralytics version")
        depth_map = _numpy(depth_result.depth.data).squeeze()
        if depth_map.ndim != 2 or depth_map.shape != frame.shape[:2]:
            raise ValueError(f"Depth map shape {depth_map.shape} does not match video {frame.shape[:2]}")
        if self._launch_hae is None and self._has_gps_fix(telemetry):
            self._launch_hae = _finite(telemetry.get("altitude_hae"))
        ground_hae, ground_source = self._ground_reference()
        result = self._empty_result("running")
        result.update(frame_id=frame_id, captured_at=captured_at,
                      telemetry_uid=telemetry.get("uid"), telemetry_age_s=telemetry_age,
                      ground_hae_m=ground_hae, ground_source=ground_source)
        result["warnings"].append("Camera intrinsics, depth and camera attitude accuracy are unvalidated.")
        if ground_source == "launch_hae_assumed_ground":
            result["warnings"].append("Ground-plane estimate treats the first aircraft HAE with a GPS fix as "
                                      "ground level; set GROUND_HAE_M to the surveyed ground height.")
        elif ground_source == "unavailable":
            result["warnings"].append("Ground-plane estimate unavailable: no ground height (set GROUND_HAE_M).")
        if self._calibration.get("status") == "calibration_unverified":
            result["warnings"].append("Depth checkpoint metric calibration could not be verified.")
        if telemetry_warning:
            result["warnings"].append(telemetry_warning)
        boxes = getattr(detections, "boxes", None)
        if boxes is not None:
            xyxy, confidence, classes = _numpy(boxes.xyxy), _numpy(boxes.conf), _numpy(boxes.cls)
            for bbox, score, cls in zip(xyxy, confidence, classes):
                if not np.isfinite(bbox).all() or _finite(score) is None or _finite(cls) != 0:
                    continue
                if float(score) < self.config["confidence"]:
                    continue
                person = estimate_person_position(
                    bbox, depth_map, telemetry, frame.shape,
                    depth_mode=self.config["depth_mode"], depth_scale=self.config["depth_scale"],
                    max_depth_m=self.config["max_depth_m"],
                )
                ground = estimate_ground_plane_position(bbox, telemetry, frame.shape, ground_hae=ground_hae)
                person.update(index=len(result["people"]) + 1,
                              bbox=[float(v) for v in bbox], confidence=float(score),
                              ground_plane=ground, fusion=fuse_estimates(person, ground),
                              estimate_disagreement_m=horizontal_disagreement_m(person, ground))
                result["people"].append(person)
        result["inference_ms"] = (time.perf_counter() - begin) * 1000
        result["processed_at"] = time.time()
        annotated = self._annotate(frame, result)
        return result, annotated

    def _write_log(self, result):
        if self._log_path is None:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(result), allow_nan=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            # A full disk must not erase the detection result or block streaming.
            result["warnings"].append(f"Estimate log unavailable: {exc}")

    @staticmethod
    def _annotate(frame, result):
        annotated = frame.copy()
        color = (30, 215, 255)
        for person in result["people"]:
            x1, y1, x2, y2 = [int(round(v)) for v in person["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            depth = person.get("depth_m")
            label = f"Person {person['index']} {person['confidence']:.2f}"
            if _finite(depth) is not None:
                label += f" | depth ~{depth:.1f} m"
            ground_range = _finite((person.get("ground_plane") or {}).get("ground_range_m"))
            if ground_range is not None and (person.get("ground_plane") or {}).get("status") == "estimated":
                label += f" | gnd ~{ground_range:.1f} m"
            cv2.putText(annotated, label, (max(0, x1), max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 930), 29), (15, 15, 15), -1)
        cv2.putText(annotated, f"INFERENCE FRAME {result['frame_id']} | approximate position | no persistent IDs",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return annotated
