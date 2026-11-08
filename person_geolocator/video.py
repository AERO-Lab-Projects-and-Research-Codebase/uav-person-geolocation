"""Geolocate people in DJI video frames, taking each frame's pose from the DJI Fly flight log.

Convert the log first:  dji-log --api-key KEY --csv flight.csv DJIFlightRecord_....txt
Then:  python -m person_geolocator.video DJI_0001.MP4 flight.csv --every 1 --out outputs/video_run
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from person_geolocator.models import MODELS

# Mini SE (FC7203) at 24 mm equivalent: its 16:9 frames are crops of the 4:3 sensor, as metadata.py derives it.
DEFAULT_HFOV = 71.6
COLUMNS = {"lat": "OSD.latitude", "lon": "OSD.longitude", "altitude_hae": "OSD.altitude",
           "height_above_ground_m": "OSD.height", "heading_deg": "OSD.yaw",
           "pitch_deg": "GIMBAL.pitch", "roll_deg": "GIMBAL.roll"}
# Heading is the aircraft yaw: the Mini SE camera cannot yaw relative to the body, OSD.yaw matches the photos'
# FlightYawDegree, and the log's GIMBAL.yaw read a constant 5 deg off it on the 2026-10-27 flight.
# ponytail: a drone with a yaw gimbal needs GIMBAL.yaw (absolute) here instead.
METHODS = ("depth_ray", "ground_plane", "terrain", "fusion", "ground_scaled")


def load_log(path):
    """Flight-log CSV from `dji-log --csv` -> {"t": unix seconds, "recording": bool, field: array}."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("CUSTOM.dateTime", "").startswith("1970")]  # before clock sync
    missing = [c for c in ["CUSTOM.dateTime", "CAMERA.isVideo", "CAMERA.isPhoto", *COLUMNS.values()]
               if rows and c not in rows[0]]
    if not rows or missing:
        raise ValueError(f"{path}: not a dji-log CSV (missing {missing or 'rows'})")
    log = {"t": np.array([datetime.fromisoformat(r["CUSTOM.dateTime"]).timestamp() for r in rows]),
           "recording": np.array([r["CAMERA.isVideo"].lower() == "true" for r in rows]),
           "photo": np.array([r["CAMERA.isPhoto"].lower() == "true" for r in rows])}
    for field, column in COLUMNS.items():
        log[field] = np.array([float(r[column]) for r in rows])
    order = np.argsort(log["t"], kind="stable")
    return {k: v[order] for k, v in log.items()}


def recordings(log, key="recording"):
    """(start, end) unix seconds of every stretch where log[key] (recording, or taking a photo) is true."""
    flag = np.concatenate([[False], log[key], [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(flag) == 1), np.flatnonzero(np.diff(flag) == -1) - 1
    return [(log["t"][s], log["t"][e]) for s, e in zip(starts, ends)]


def pick_recording(spans, duration, index=None):
    """The requested recording, else the one whose length is closest to the video's."""
    if not spans:
        raise ValueError("The flight log has no video recording (CAMERA.isVideo never true)")
    if index is not None:
        return spans[index]
    return min(spans, key=lambda s: abs((s[1] - s[0]) - duration))


def pose_at(log, t, hfov_deg):
    """Interpolate the pose at unix time t, skipping rows without a GPS fix; yaw is unwrapped first."""
    fix = (log["lat"] != 0) | (log["lon"] != 0)
    if not fix.any():
        raise ValueError("The flight log has no GPS fix")
    times = log["t"][fix]
    pose = {"hfov_deg": hfov_deg}
    for field in COLUMNS:
        values = log[field][fix]
        if field == "heading_deg":
            values = np.degrees(np.unwrap(np.radians(values)))
        pose[field] = float(np.interp(t, times, values))
    pose["heading_deg"] = (pose["heading_deg"] + 180) % 360 - 180
    return pose


def default_locate():
    from person_geolocator.engine import locate
    from person_geolocator.models import DepthModels
    models = DepthModels()
    return lambda path, model, pose: locate(path, model, pose, models)


def people_rows(t, pose, result):
    """One people.csv row per detected person, with the drone pose used."""
    base = {"utc": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds"),
            "drone_lat": pose["lat"], "drone_lon": pose["lon"], "height_m": round(pose["height_above_ground_m"], 2),
            "heading_deg": round(pose["heading_deg"], 1), "pitch_deg": round(pose["pitch_deg"], 1)}
    rows = []
    for p in result["people"]:
        row = dict(base, person=p["index"], confidence=round(p["confidence"], 3), depth_m=p["depth_m"])
        for name, m in p["methods"].items():
            if m["status"] == "estimated":
                row[f"{name}_lat"], row[f"{name}_lon"] = m["lat"], m["lon"]
        rows.append(row)
    return rows


ROW_FIELDS = ["utc", "drone_lat", "drone_lon", "height_m", "heading_deg", "pitch_deg", "person", "confidence",
              "depth_m"] + [f"{m}_{k}" for m in METHODS for k in ("lat", "lon")]


def main(argv=None, locate_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("log", type=Path, help="CSV written by `dji-log --csv`")
    ap.add_argument("--model", choices=list(MODELS), default="unidepth")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between processed frames")
    ap.add_argument("--hfov", type=float, default=DEFAULT_HFOV, help="horizontal field of view of the video, degrees")
    ap.add_argument("--recording", type=int, help="which recording in the log (0-based); default: closest length")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds added to the log time of each frame")
    ap.add_argument("--out", type=Path, default=Path("person_geolocator/outputs/video"))
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(str(args.video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps <= 0:
        raise SystemExit(f"Cannot read video {args.video}")
    duration = count / fps
    log = load_log(args.log)
    spans = recordings(log)
    start, end = pick_recording(spans, duration, args.recording)
    print(f"{len(spans)} recording(s) in log; using {datetime.fromtimestamp(start):%H:%M:%S} "
          f"({end - start:.1f} s of log vs {duration:.1f} s of video)")
    if abs((end - start) - duration) > 3:
        print("note: log recording length differs from the video by >3 s; check --recording / --offset")

    locate_fn = locate_fn or default_locate()
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out / "people.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["video_s", "frame"] + ROW_FIELDS, restval="")
        writer.writeheader()
        index, next_s = 0, 0.0
        while cap.grab():
            video_s = index / fps
            if video_s + 0.5 / fps >= next_s:
                next_s += args.every
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = start + video_s + args.offset
                pose = pose_at(log, t, args.hfov)
                path = frames_dir / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result = locate_fn(path, args.model, pose)
                jpeg = result.pop("annotated_jpeg", None)
                if jpeg and result["people"]:
                    path.with_name(path.stem + "_annotated.jpg").write_bytes(jpeg)
                print(f"{video_s:7.1f} s  {len(result['people'])} person(s)")
                for row in people_rows(t, pose, result):
                    writer.writerow(dict(row, video_s=round(video_s, 3), frame=index))
            index += 1
    print(f"wrote {args.out / 'people.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
