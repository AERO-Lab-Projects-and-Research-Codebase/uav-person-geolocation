"""
Skydio X10 — Live Telemetry Dashboard
======================================
Flask web app that:
  - Captures RTSP video from the X10 controller
  - Receives CoT telemetry over TCP
  - Serves a live dashboard at http://localhost:5000

Requirements:
    python -m pip install -r requirements.txt

Usage:
    1. Controller: Sharing > RTSP > Internal > stream "skydio" > start
    2. Controller: ATAK > Publish Telemetry ON
    3. Controller: ATAK > Network Connections > add laptop IP, port 4242, TCP
    4. python main.py
    5. Open browser: http://localhost:5000
"""

import socket
import threading
import time
import os
import re
import json
import math
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# ============================================================================
# CONFIG
# ============================================================================

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://10.101.154.86:8554/skydio")
COT_PORT = 4242
COT_PROTOCOL = "TCP"
FLASK_PORT = 5000
LOG_CSV = True
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = str(BASE_DIR / "telemetry_logs")
HUMAN_ANALYSIS_ENABLED = os.environ.get("HUMAN_ANALYSIS_ENABLED", "true").lower() not in {"0", "false", "no", "off"}


def human_config():
    """Resolve files independently of the directory used to start Python."""
    def local_path(name, default):
        path = Path(os.environ.get(name, default)).expanduser()
        return str(path if path.is_absolute() else BASE_DIR / path)
    return {
        "detector_weights": local_path("HUMAN_DETECTOR_WEIGHTS", "yolo26x.pt"),
        "depth_weights": local_path("HUMAN_DEPTH_WEIGHTS", "yolo26x-depth.pt"),
        "device": os.environ.get("HUMAN_DEVICE", "auto"),
        "detector_imgsz": int(os.environ.get("HUMAN_DETECTOR_IMGSZ", "640")),
        "depth_imgsz": int(os.environ.get("HUMAN_DEPTH_IMGSZ", "768")),
        "confidence": float(os.environ.get("HUMAN_CONFIDENCE", "0.4")),
        "max_fps": float(os.environ.get("HUMAN_MAX_FPS", "3")),
        "depth_mode": os.environ.get("HUMAN_DEPTH_MODE", "axial"),
        "depth_scale": float(os.environ.get("HUMAN_DEPTH_SCALE", "1")),
        "max_depth_m": float(os.environ.get("HUMAN_MAX_DEPTH_M", "100")),
        "frame_max_age_s": float(os.environ.get("HUMAN_FRAME_MAX_AGE_S", "3")),
        "telemetry_max_age_s": float(os.environ.get("HUMAN_TELEMETRY_MAX_AGE_S", "3")),
        # Surveyed ground ellipsoid height for the ray/ground-plane cross-check.
        "ground_hae_m": float(ground_hae) if (ground_hae := os.environ.get("GROUND_HAE_M", "").strip()) else None,
        "log_dir": local_path("HUMAN_LOG_DIR", "output/person_estimates"),
    }

# ============================================================================
# CoT PARSER
# ============================================================================

def safe_float(val):
    if val is None:
        return None
    try:
        number = float(val)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def parse_cot_xml(xml_str):
    telem = OrderedDict()
    try:
        if "<?xml" in xml_str:
            parts = xml_str.split("<?xml")
            for part in reversed(parts):
                if "<event" in part:
                    xml_str = "<?xml" + part
                    break
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return parse_cot_fallback(xml_str)

    # Strip element namespaces only; event attributes remain intact.
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "event":
        return telem
    telem["uid"] = root.get("uid", "")
    telem["event_type"] = root.get("type", "")
    telem["timestamp"] = root.get("time", "")

    point = root.find("point")
    if point is not None:
        telem["latitude"] = safe_float(point.get("lat"))
        telem["longitude"] = safe_float(point.get("lon"))
        telem["altitude_hae"] = safe_float(point.get("hae"))
        telem["ce"] = safe_float(point.get("ce"))
        telem["le"] = safe_float(point.get("le"))

    detail = root.find("detail")
    if detail is not None:
        track = detail.find("track")
        if track is not None:
            telem["heading"] = safe_float(track.get("course"))
            telem["speed"] = safe_float(track.get("speed"))

        sensor = detail.find("sensor")
        if sensor is not None:
            telem["sensor_vfov"] = safe_float(sensor.get("vfov"))
            telem["sensor_hfov"] = safe_float(sensor.get("fov"))
            telem["sensor_elevation"] = safe_float(sensor.get("elevation"))
            telem["sensor_roll"] = safe_float(sensor.get("roll"))
            telem["sensor_azimuth"] = safe_float(sensor.get("azimuth"))
            telem["sensor_range"] = safe_float(sensor.get("range"))
            telem["sensor_north"] = safe_float(sensor.get("north"))

        contact = detail.find("contact")
        if contact is not None:
            telem["callsign"] = contact.get("callsign", "")

        status = detail.find("status")
        if status is not None:
            telem["battery"] = safe_float(status.get("battery"))

    return telem


def parse_cot_fallback(xml_str):
    telem = OrderedDict()
    patterns = {
        "latitude": r'lat="([^"]+)"',
        "longitude": r'lon="([^"]+)"',
        "altitude_hae": r'hae="([^"]+)"',
        "heading": r'course="([^"]+)"',
        "speed": r'speed="([^"]+)"',
        "sensor_vfov": r'vfov="([^"]+)"',
        "sensor_hfov": r'(?<!\w)fov="([^"]+)"',
        "sensor_elevation": r'elevation="([^"]+)"',
        "sensor_roll": r'(?<!\w)roll="([^"]+)"',
        "sensor_azimuth": r'azimuth="([^"]+)"',
        "sensor_range": r'(?<!\w)range="([^"]+)"',
        "sensor_north": r'north="([^"]+)"',
        "uid": r'uid="([^"]+)"',
        "event_type": r'<(?:\w+:)?event\b[^>]*\btype="([^"]+)"',
        "timestamp": r'time="([^"]+)"',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, xml_str)
        if m:
            if key in ("uid", "timestamp", "event_type"):
                telem[key] = m.group(1)
            else:
                telem[key] = safe_float(m.group(1))
    return telem


# ============================================================================
# CoT RECEIVER
# ============================================================================

class CoTReceiver:
    def __init__(self, port=4242, protocol="TCP", aircraft_uid=None, telemetry_max_age_s=3):
        self.port = port
        self.protocol = protocol.upper()
        self.latest_telemetry = OrderedDict()
        self.telemetry_history = []
        self.aircraft_uid = aircraft_uid if aircraft_uid is not None else os.environ.get("GEO_AIRCRAFT_UID", "")
        self.telemetry_max_age_s = telemetry_max_age_s
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._msg_count = 0

    def start(self):
        self._running = True
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def get_telemetry(self):
        with self._lock:
            return dict(self.latest_telemetry)

    def record_telemetry(self, telem, received_at=None):
        """Keep complete individual events; never merge aircraft and operator data."""
        if not telem:
            return
        snapshot = dict(telem)
        snapshot["_received_at"] = time.time() if received_at is None else received_at
        with self._lock:
            self.latest_telemetry = OrderedDict(snapshot)
            self.telemetry_history.append(snapshot)
            self.telemetry_history = self.telemetry_history[-1000:]
            self._msg_count += 1

    def get_geolocation_telemetry(self, captured_at):
        """Latest aircraft event received before decode, within the freshness window.

        Receive time is a local approximation, not camera/CoT clock synchronization.
        An explicit UID is authoritative; otherwise require a CoT air-platform type.
        """
        captured_at = safe_float(captured_at)
        if captured_at is None:
            return {}
        with self._lock:
            candidates = []
            for event in self.telemetry_history:
                age = captured_at - event["_received_at"]
                if not 0 <= age <= self.telemetry_max_age_s:
                    continue
                if self.aircraft_uid:
                    selected = event.get("uid") == self.aircraft_uid
                else:
                    selected = bool(event.get("uid")) and bool(re.match(r"^a-[^-]+-A(?:-|$)", event.get("event_type", "")))
                if selected:
                    candidates.append(event)
            return dict(max(candidates, key=lambda item: item["_received_at"])) if candidates else {}

    def get_status(self):
        return {
            "connected": self._connected,
            "msg_count": self._msg_count,
            "protocol": self.protocol,
            "port": self.port,
        }

    def _receive_loop(self):
        if self.protocol == "TCP":
            self._tcp_loop()
        else:
            self._udp_loop()

    def _tcp_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(2)
        server.settimeout(2)
        print(f"[CoT] TCP server on port {self.port}")

        while self._running:
            try:
                conn, addr = server.accept()
                conn.setblocking(True)
                conn.settimeout(5)
                self._connected = True
                print(f"[CoT] Connected: {addr}")
                t = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
        server.close()

    def _handle_conn(self, conn):
        buffer = ""
        while self._running:
            try:
                data = conn.recv(8192)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="replace")
                while (end_event := re.search(r"</(?:[\w.-]+:)?event\s*>", buffer)):
                    idx = end_event.end()
                    msg = buffer[:idx]
                    buffer = buffer[idx:]
                    self.record_telemetry(parse_cot_xml(msg))
                if len(buffer) > 1024 * 1024:
                    buffer = ""  # Bound malformed/incomplete messages.
            except socket.timeout:
                continue
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                break
        self._connected = False
        conn.close()

    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(2)
        self._connected = True
        print(f"[CoT] UDP listener on port {self.port}")
        while self._running:
            try:
                data, addr = sock.recvfrom(8192)
                telem = parse_cot_xml(data.decode("utf-8", errors="replace"))
                self.record_telemetry(telem)
            except socket.timeout:
                continue
        sock.close()


# ============================================================================
# VIDEO CAPTURE
# ============================================================================

class VideoCapture:
    def __init__(self, url):
        self.url = url
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._fps = 0
        self._resolution = (0, 0)
        self._frame_count = 0
        self._captured_at = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_snapshot(self):
        with self._lock:
            return {"frame": self._frame.copy() if self._frame is not None else None,
                    "frame_id": self._frame_count, "captured_at": self._captured_at}

    def get_status(self):
        return {
            "connected": self._cap is not None and self._cap.isOpened() if self._cap else False,
            "fps": round(self._fps, 1),
            "resolution": f"{self._resolution[0]}x{self._resolution[1]}",
            "frames": self._frame_count,
        }

    def _capture_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        print(f"[Video] Connecting to {self.url}")
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            print("[Video] ERROR: Cannot open RTSP stream")
            return

        self._resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        print(f"[Video] Stream opened: {self._resolution[0]}x{self._resolution[1]}")

        fps_timer = time.time()
        fps_count = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1
                self._captured_at = time.time()
            fps_count += 1

            now = time.time()
            if now - fps_timer >= 1.0:
                self._fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

        self._cap.release()


# ============================================================================
# CSV LOGGER
# ============================================================================

class CSVLogger:
    FIELDS = ["log_time", "uid", "event_type", "timestamp", "_received_at", "callsign",
              "latitude", "longitude", "altitude_hae", "ce", "le", "heading", "speed",
              "sensor_vfov", "sensor_hfov", "sensor_elevation", "sensor_roll",
              "sensor_azimuth", "sensor_range", "sensor_north", "battery"]

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"telemetry_{ts}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._lock = threading.Lock()

    def log(self, telem):
        if not telem:
            return
        with self._lock:
            self._writer.writerow({**telem, "log_time": datetime.now().isoformat()})
            self._file.flush()

    def close(self):
        self._file.close()


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
video_capture = None
cot_receiver = None
csv_logger = None
human_pipeline = None
human_start_error = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skydio X10 — Ground Station</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2236;
    --border: #1e293b;
    --border-active: #334155;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-dim: #64748b;
    --accent: #22d3ee;
    --accent-glow: rgba(34, 211, 238, 0.15);
    --green: #34d399;
    --green-glow: rgba(52, 211, 153, 0.2);
    --amber: #fbbf24;
    --red: #f87171;
    --orange: #fb923c;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Outfit', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0f1420 0%, var(--bg-primary) 100%);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 18px;
    color: var(--accent);
    letter-spacing: -0.5px;
  }

  .logo span {
    color: var(--text-dim);
    font-weight: 400;
  }

  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .status-pill.live {
    background: var(--green-glow);
    color: var(--green);
    border: 1px solid rgba(52, 211, 153, 0.3);
  }

  .status-pill.offline {
    background: rgba(248, 113, 113, 0.1);
    color: var(--red);
    border: 1px solid rgba(248, 113, 113, 0.2);
  }

  .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s ease-in-out infinite;
  }

  .pulse.off { background: var(--red); animation: none; }

  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
  }

  .header-stats {
    display: flex;
    gap: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-dim);
  }

  .header-stats strong {
    color: var(--text-secondary);
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 0;
    height: calc(100vh - 53px);
  }

  /* Video Panel */
  .video-panel {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border-right: 1px solid var(--border);
  }

  .video-panel img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .video-overlay {
    position: absolute;
    top: 16px;
    left: 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .video-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(4px);
    color: var(--text-secondary);
    width: fit-content;
  }

  .video-tag.rec {
    color: var(--red);
    border: 1px solid rgba(248,113,113,0.3);
  }

  .crosshair {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40px;
    height: 40px;
    pointer-events: none;
  }

  .crosshair::before, .crosshair::after {
    content: '';
    position: absolute;
    background: rgba(34, 211, 238, 0.4);
  }

  .crosshair::before {
    width: 1px;
    height: 100%;
    left: 50%;
  }

  .crosshair::after {
    width: 100%;
    height: 1px;
    top: 50%;
  }

  /* Telemetry Sidebar */
  .sidebar {
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    background: var(--bg-primary);
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s;
  }

  .card:hover {
    border-color: var(--border-active);
  }

  .card-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .card-title .icon {
    font-size: 14px;
  }

  .data-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .data-grid.three-col {
    grid-template-columns: 1fr 1fr 1fr;
  }

  .data-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .data-item.full-width {
    grid-column: 1 / -1;
  }

  .data-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
  }

  .data-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.2;
  }

  .data-value.large {
    font-size: 22px;
    color: var(--accent);
  }

  .data-value.small {
    font-size: 13px;
  }

  .data-unit {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-dim);
    margin-left: 2px;
  }

  /* Compass */
  .compass-container {
    display: flex;
    justify-content: center;
    padding: 8px 0;
  }

  .compass {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 2px solid var(--border-active);
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle, rgba(34,211,238,0.03) 0%, transparent 70%);
  }

  .compass-labels {
    position: absolute;
    width: 100%;
    height: 100%;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-dim);
    font-weight: 600;
  }

  .compass-labels span {
    position: absolute;
  }

  .compass-labels .n { top: 4px; left: 50%; transform: translateX(-50%); color: var(--accent); }
  .compass-labels .s { bottom: 4px; left: 50%; transform: translateX(-50%); }
  .compass-labels .e { right: 6px; top: 50%; transform: translateY(-50%); }
  .compass-labels .w { left: 6px; top: 50%; transform: translateY(-50%); }

  .compass-needle {
    width: 2px;
    height: 44px;
    background: linear-gradient(to top, var(--red) 50%, var(--accent) 50%);
    transform-origin: bottom center;
    position: absolute;
    bottom: 50%;
    border-radius: 1px;
    transition: transform 0.3s ease;
  }

  .compass-center {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-primary);
    position: absolute;
    z-index: 1;
  }

  .compass-heading {
    position: absolute;
    bottom: -24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
  }

  /* Connection status bar */
  .conn-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
  }

  .conn-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .conn-dot.on { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .conn-dot.off { background: var(--red); }

  .conn-label {
    font-size: 12px;
    color: var(--text-secondary);
    flex: 1;
  }

  .conn-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
  }

  /* No signal overlay */
  .no-signal {
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    text-align: center;
  }

  .no-signal .big {
    font-size: 48px;
    margin-bottom: 12px;
    opacity: 0.3;
  }

  /* Scrollbar */
  .sidebar::-webkit-scrollbar { width: 4px; }
  .sidebar::-webkit-scrollbar-track { background: transparent; }
  .sidebar::-webkit-scrollbar-thumb { background: var(--border-active); border-radius: 2px; }
  .analysis-note { font-size: 12px; color: var(--text-secondary); line-height: 1.5; margin: 6px 0; }
  .analysis-warning { color: var(--amber); }
  .person-entry { border-top: 1px solid var(--border); padding: 10px 0; font-size: 12px; line-height: 1.7; overflow-wrap: anywhere; }
  #rawToggle { accent-color: var(--accent); }
  @media (max-width: 800px) { .main { grid-template-columns: 1fr; height: auto; } .video-panel { height: 55vh; } }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">SKYDIO X10 <span>GROUND STATION</span></div>
    <div class="status-pill live" id="statusPill">
      <div class="pulse" id="statusPulse"></div>
      <span id="statusText">CONNECTING</span>
    </div>
  </div>
  <div class="header-stats">
    <span>FPS <strong id="fpsVal">--</strong></span>
    <span>RES <strong id="resVal">--</strong></span>
    <span>CoT <strong id="msgCount">0</strong></span>
  </div>
</div>

<div class="main">
  <!-- Video Feed -->
  <div class="video-panel">
    <img id="videoFeed" src="/human_video_feed" alt="Live Feed with optional person estimates" />
    <div class="video-overlay">
      <div class="video-tag">● STREAM</div>
      <div class="video-tag" id="videoRes">--</div>
      <div class="video-tag" id="analysisFeedState">Waiting for analysis; raw fallback</div>
      <label class="video-tag"><input id="rawToggle" type="checkbox"> Show raw video</label>
    </div>
    <div class="crosshair"></div>
  </div>

  <!-- Telemetry Sidebar -->
  <div class="sidebar">

    <div class="card">
      <div class="card-title">People — Estimated positions</div>
      <p class="analysis-note analysis-warning">Monocular depth estimate, not a calibrated camera measurement. Position estimates depend on camera geometry and timely aircraft telemetry.</p>
      <p class="analysis-note" id="analysisState">Waiting for analysis</p>
      <p class="analysis-note" id="analysisMetrics">Device -- · Inference -- fps · Result age --</p>
      <p class="analysis-note" id="analysisSource">Aircraft telemetry UID --</p>
      <p class="analysis-note analysis-warning" id="analysisError"></p>
      <p class="analysis-note analysis-warning" id="analysisWarnings"></p>
      <div id="peopleList">No processed frame yet.</div>
    </div>

    <!-- Position -->
    <div class="card">
      <div class="card-title"><span class="icon">◎</span> Position</div>
      <div class="data-grid">
        <div class="data-item">
          <span class="data-label">Latitude</span>
          <span class="data-value" id="lat">--</span>
        </div>
        <div class="data-item">
          <span class="data-label">Longitude</span>
          <span class="data-value" id="lon">--</span>
        </div>
        <div class="data-item">
          <span class="data-label">Altitude (HAE)</span>
          <span class="data-value" id="alt">--<span class="data-unit">m</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">Speed</span>
          <span class="data-value" id="speed">--<span class="data-unit">m/s</span></span>
        </div>
      </div>
    </div>

    <!-- Heading / Compass -->
    <div class="card">
      <div class="card-title"><span class="icon">⊕</span> Heading</div>
      <div class="compass-container">
        <div class="compass">
          <div class="compass-labels">
            <span class="n">N</span>
            <span class="s">S</span>
            <span class="e">E</span>
            <span class="w">W</span>
          </div>
          <div class="compass-needle" id="needle"></div>
          <div class="compass-center"></div>
          <div class="compass-heading" id="headingVal">--°</div>
        </div>
      </div>
    </div>

    <!-- Sensor -->
    <div class="card">
      <div class="card-title"><span class="icon">◉</span> Sensor</div>
      <div class="data-grid three-col">
        <div class="data-item">
          <span class="data-label">V-FOV</span>
          <span class="data-value small" id="vfov">--<span class="data-unit">°</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">H-FOV</span>
          <span class="data-value small" id="hfov">--<span class="data-unit">°</span></span>
        </div>
        <div class="data-item">
          <span class="data-label" title="Reported sensor metadata; not measured distance to a detected person">Reported range</span>
          <span class="data-value small" id="srange">--<span class="data-unit">m</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">Azimuth</span>
          <span class="data-value small" id="azimuth">--<span class="data-unit">°</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">Elevation</span>
          <span class="data-value small" id="elevation">--<span class="data-unit">°</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">Roll</span>
          <span class="data-value small" id="sroll">--<span class="data-unit">°</span></span>
        </div>
      </div>
    </div>

    <!-- Accuracy -->
    <div class="card">
      <div class="card-title"><span class="icon">⊙</span> Accuracy</div>
      <div class="data-grid">
        <div class="data-item">
          <span class="data-label">Circular Error</span>
          <span class="data-value small" id="ce">--<span class="data-unit">m</span></span>
        </div>
        <div class="data-item">
          <span class="data-label">Linear Error</span>
          <span class="data-value small" id="le">--<span class="data-unit">m</span></span>
        </div>
      </div>
    </div>

    <!-- Identity -->
    <div class="card">
      <div class="card-title"><span class="icon">▣</span> Latest CoT event</div>
      <div class="data-grid">
        <div class="data-item">
          <span class="data-label">UID</span>
          <span class="data-value small" id="uid">--</span>
        </div>
        <div class="data-item">
          <span class="data-label">Callsign</span>
          <span class="data-value small" id="callsign">--</span>
        </div>
        <div class="data-item full-width">
          <span class="data-label">Last Update</span>
          <span class="data-value small" id="lastUpdate">--</span>
        </div>
      </div>
    </div>

    <!-- Connection -->
    <div class="card">
      <div class="card-title"><span class="icon">⬡</span> Connections</div>
      <div class="conn-row">
        <div class="conn-dot" id="videoDot"></div>
        <span class="conn-label">RTSP Video</span>
        <span class="conn-value" id="videoStatus">--</span>
      </div>
      <div class="conn-row">
        <div class="conn-dot" id="cotDot"></div>
        <span class="conn-label">CoT Telemetry</span>
        <span class="conn-value" id="cotStatus">--</span>
      </div>
    </div>

  </div>
</div>

<script>
function fmt(val, decimals=4) {
  if (val === null || val === undefined || val === "None") return "--";
  let n = parseFloat(val);
  if (!Number.isFinite(n)) return "--";
  return n.toFixed(decimals);
}

document.getElementById("rawToggle").addEventListener("change", event => {
  document.getElementById("videoFeed").src = event.target.checked ? "/video_feed" : "/human_video_feed";
  updatePeople();
});

function addPersonLine(card, text) {
  const line = document.createElement("div");
  line.textContent = text;
  card.appendChild(line);
}

function updatePeople() {
  fetch("/api/people").then(r => r.json()).then(data => {
    document.getElementById("analysisState").textContent = "Analysis: " + (data.status || "unknown") + (data.stale ? " — stale" : "");
    document.getElementById("analysisMetrics").textContent = "Device " + (data.device || "--") + " · Inference " + fmt(data.fps, 1) + " fps · Result age " + fmt(data.result_age_s, 1) + " s";
    document.getElementById("analysisSource").textContent = "Aircraft telemetry UID " + (data.telemetry_uid || "--") + " · At-frame age " + fmt(data.telemetry_age_s, 1) + " s · Ground " + (data.ground_source || "--") + (data.ground_hae_m == null ? "" : " " + fmt(data.ground_hae_m, 1) + " m HAE");
    document.getElementById("analysisError").textContent = data.error || "";
    document.getElementById("analysisWarnings").textContent = (data.warnings || []).join("; ");
    const raw = document.getElementById("rawToggle").checked;
    document.getElementById("analysisFeedState").textContent = raw ? "Raw video selected" : (data.raw_fallback ? "Raw fallback — analysis unavailable or stale" : "Annotated inference frame");
    const list = document.getElementById("peopleList");
    list.replaceChildren();
    const people = data.people || [];
    if (!people.length) {
      list.textContent = data.status === "running" || data.status === "ready" ? "No people in the latest processed frame." : "No person estimates available.";
    }
    people.forEach((person, index) => {
      const card = document.createElement("div");
      card.className = "person-entry";
      const position = person.position || person;
      addPersonLine(card, "Person " + (index + 1) + " · Confidence " + fmt(person.confidence * 100, 1) + "%");
      addPersonLine(card, "Estimated depth " + fmt(position.depth_m, 2) + " m · Slant range " + fmt(position.slant_range_m, 2) + " m");
      addPersonLine(card, "Latitude " + fmt(position.latitude, 7) + " · Longitude " + fmt(position.longitude, 7));
      addPersonLine(card, "North " + fmt(position.north_m, 2) + " m · East " + fmt(position.east_m, 2) + " m");
      const ground = person.ground_plane || {};
      if (ground.status === "estimated") {
        addPersonLine(card, "Ground-plane (pose only): range " + fmt(ground.ground_range_m, 2) + " m · Lat " + fmt(ground.latitude, 7) + " · Lon " + fmt(ground.longitude, 7) + " · North " + fmt(ground.north_m, 2) + " m · East " + fmt(ground.east_m, 2) + " m · Depression " + fmt(ground.depression_deg, 1) + "°");
        addPersonLine(card, "Depth vs ground-plane disagreement " + fmt(person.estimate_disagreement_m, 2) + " m");
      } else {
        addPersonLine(card, "Ground-plane unavailable: " + (ground.reason || "no estimate"));
      }
      const fusion = person.fusion || {};
      if (fusion.status === "estimated") {
        addPersonLine(card, "Fusion (" + fusion.source + ", ray " + fmt(fusion.depression_deg, 1) + "° down): range " + fmt(fusion.range_m, 2) + " m · Lat " + fmt(fusion.latitude, 7) + " · Lon " + fmt(fusion.longitude, 7));
      }
      if (position.origin_mode === "assumed_zero_origin") {
        addPersonLine(card, "ASSUMED ZERO ORIGIN (0°, 0°) — local test coordinates, not a real world location");
        card.classList.add("analysis-warning");
      } else {
        addPersonLine(card, "Origin: " + (position.origin_mode || "unavailable"));
      }
      if (position.reason) addPersonLine(card, "Position unavailable: " + position.reason);
      list.appendChild(card);
    });
  }).catch(() => {
    document.getElementById("analysisState").textContent = "Analysis status unavailable";
  });
}

function updateDashboard() {
  fetch("/api/telemetry")
    .then(r => r.json())
    .then(data => {
      const t = data.telemetry;
      const vs = data.video_status;
      const cs = data.cot_status;

      // Position
      document.getElementById("lat").textContent = fmt(t.latitude, 6);
      document.getElementById("lon").textContent = fmt(t.longitude, 6);
      document.getElementById("alt").textContent = fmt(t.altitude_hae, 1) + ' m';
      document.getElementById("speed").textContent = fmt(t.speed, 1) + ' m/s';

      // Heading
      let hdg = parseFloat(t.heading);
      if (!isNaN(hdg)) {
        document.getElementById("needle").style.transform = "rotate(" + hdg + "deg)";
        document.getElementById("headingVal").textContent = hdg.toFixed(0) + "°";
      }

      // Sensor
      document.getElementById("vfov").textContent = fmt(t.sensor_vfov, 1) + ' °';
      document.getElementById("hfov").textContent = fmt(t.sensor_hfov, 1) + ' °';
      document.getElementById("srange").textContent = fmt(t.sensor_range, 0) + ' m';
      document.getElementById("azimuth").textContent = fmt(t.sensor_azimuth, 1) + ' °';
      document.getElementById("elevation").textContent = fmt(t.sensor_elevation, 4) + ' °';
      document.getElementById("sroll").textContent = fmt(t.sensor_roll, 1) + ' °';

      // Accuracy
      document.getElementById("ce").textContent = fmt(t.ce, 1) + ' m';
      document.getElementById("le").textContent = fmt(t.le, 1) + ' m';

      // Identity
      document.getElementById("uid").textContent = t.uid || "--";
      document.getElementById("callsign").textContent = t.callsign || "--";
      document.getElementById("lastUpdate").textContent = t.timestamp || "--";

      // Video status
      document.getElementById("fpsVal").textContent = vs.fps;
      document.getElementById("resVal").textContent = vs.resolution;
      document.getElementById("videoRes").textContent = vs.resolution;
      document.getElementById("videoDot").className = "conn-dot " + (vs.connected ? "on" : "off");
      document.getElementById("videoStatus").textContent = vs.connected ? vs.resolution + " @ " + vs.fps + " fps" : "Disconnected";

      // CoT status
      document.getElementById("msgCount").textContent = cs.msg_count;
      document.getElementById("cotDot").className = "conn-dot " + (cs.connected ? "on" : "off");
      document.getElementById("cotStatus").textContent = cs.connected ? cs.msg_count + " msgs" : "Waiting...";

      // Header status
      let isLive = vs.connected || cs.connected;
      document.getElementById("statusPill").className = "status-pill " + (isLive ? "live" : "offline");
      document.getElementById("statusPulse").className = "pulse" + (isLive ? "" : " off");
      document.getElementById("statusText").textContent = isLive ? "LIVE" : "OFFLINE";
    })
    .catch(() => {});
}

setInterval(updateDashboard, 500);
setInterval(updatePeople, 1000);
updateDashboard();
updatePeople();
</script>
</body>
</html>
"""


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/telemetry")
def api_telemetry():
    telem = cot_receiver.get_telemetry() if cot_receiver else {}
    vs = video_capture.get_status() if video_capture else {}
    cs = cot_receiver.get_status() if cot_receiver else {}

    if LOG_CSV and csv_logger and telem:
        csv_logger.log(telem)

    return jsonify({
        "telemetry": telem,
        "video_status": vs,
        "cot_status": cs,
    })


def get_people_result():
    """Analysis is optional; inference errors must leave the raw feed usable."""
    if human_pipeline is None:
        return {"status": "error" if human_start_error else ("waiting" if HUMAN_ANALYSIS_ENABLED else "disabled"),
                "error": human_start_error, "people": [], "raw_fallback": True,
                "result_age_s": None, "device": None, "fps": 0,
                "depth_calibration": "Monocular estimate; not a calibrated camera measurement"}
    try:
        result = dict(human_pipeline.get_result())
        result["raw_fallback"] = human_pipeline.get_frame() is None
        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc), "people": [], "raw_fallback": True}


@app.route("/api/people")
def api_people():
    return jsonify(get_people_result())


def generate_mjpeg(human=False):
    """Generator for MJPEG video stream."""
    while True:
        frame = None
        if human and human_pipeline is not None:
            try:
                frame = human_pipeline.get_frame()
            except Exception:
                pass
        if frame is None:
            frame = video_capture.get_frame() if video_capture else None
        if frame is None:
            # Send a black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Connecting to RTSP...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.1)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/human_video_feed")
def human_video_feed():
    return Response(generate_mjpeg(human=True),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global video_capture, cot_receiver, csv_logger, human_pipeline, human_start_error

    print()
    print("=" * 56)
    print("   Skydio X10 — Ground Station Dashboard")
    print("=" * 56)
    print(f"   RTSP:   {RTSP_URL}")
    print(f"   CoT:    {COT_PROTOCOL} port {COT_PORT}")
    print(f"   Web UI: http://localhost:{FLASK_PORT}")
    print("=" * 56)
    print()

    # Start video capture
    video_capture = VideoCapture(RTSP_URL)
    video_capture.start()

    # Start CoT receiver
    cot_receiver = CoTReceiver(port=COT_PORT, protocol=COT_PROTOCOL)
    cot_receiver.start()

    if HUMAN_ANALYSIS_ENABLED:
        try:
            from human_pipeline import HumanPipeline
            config = human_config()
            cot_receiver.telemetry_max_age_s = config["telemetry_max_age_s"]
            human_pipeline = HumanPipeline(video_capture.get_frame_snapshot,
                                           cot_receiver.get_geolocation_telemetry, config=config)
            human_pipeline.start()
        except Exception as exc:
            human_start_error = str(exc)
            if human_pipeline:
                try:
                    human_pipeline.stop()
                except Exception:
                    pass
            human_pipeline = None
            print(f"[Human] Analysis unavailable; raw video continues: {exc}")

    # Start CSV logger
    if LOG_CSV:
        csv_logger = CSVLogger(LOG_DIR)
        print(f"[CSV] Logging to {csv_logger.path}")

    print()
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop.\n")

    try:
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if human_pipeline:
            try:
                human_pipeline.stop()
            except Exception as exc:
                print(f"[Human] Shutdown error: {exc}")
        video_capture.stop()
        cot_receiver.stop()
        if csv_logger:
            csv_logger.close()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
