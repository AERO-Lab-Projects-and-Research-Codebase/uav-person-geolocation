"""Add terrain-map ground estimates to a saved detection run, without touching it.

Each saved result JSON already holds the capture pose, lens and boxes, so no
model is re-run. Writes outputs/terrain_<run>[_nadir]/ for build_review.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from terrain import TerrainModel, apply_terrain_geometry

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
CODE_FILES = ("terrain.py", "metadata_geometry.py", "run_terrain_ground.py")
CSV_FIELDS = ["source", "person_index", "status", "reason", "latitude", "longitude", "north_m", "east_m",
              "ground_range_m", "ground_hae_m", "height_above_ground_m", "depression_deg",
              "incidence_deg", "grazing", "vertical_offset_m"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nadir_offset(capture, terrain):
    """Launch-plane HAE minus terrain HAE directly under the camera, or None."""
    t = capture.get("telemetry") or {}
    launch = capture.get("assumed_launch_ground_hae_m")
    if launch is None or t.get("latitude") is None or t.get("longitude") is None:
        return None
    ground = terrain.height_hae(t["latitude"], t["longitude"])
    return None if ground is None else float(launch-ground)


TERRAIN_MANIFEST_KEYS = ("detector_run_signature", "terrain", "vertical_mode")


def check_output_folder(run_dir, output):
    """Refuse an output folder that could overwrite the detection run or another kind of run."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output == run_dir or run_dir in output.parents or output in run_dir.parents:
        raise ValueError("Output folder must be separate from the detection run folder.")
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if not isinstance(existing, dict) or not all(key in existing for key in TERRAIN_MANIFEST_KEYS):
            raise ValueError(f"Output folder already holds a manifest that is not a terrain run: {output}")


def run(run_dir, terrain_path, output, *, vertical_mode="absolute", max_range_m=1000.):
    run_dir, terrain_path, output = Path(run_dir), Path(terrain_path), Path(output)
    if vertical_mode not in ("absolute", "nadir_anchor"):
        raise ValueError("unsupported_vertical_mode")
    check_output_folder(run_dir, output)
    detection = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    if detection.get("status") != "complete" or not detection.get("run_signature"):
        raise ValueError("Detection run is not complete or not signed.")
    terrain = TerrainModel.load(terrain_path)
    terrain_sha = sha256_file(terrain_path)
    signature = hashlib.sha256(json.dumps({
        "detector_run_signature": detection["run_signature"], "terrain_sha256": terrain_sha,
        "code": {name: sha256_file(ROOT/name) for name in CODE_FILES},
        "vertical_mode": vertical_mode, "max_range_m": max_range_m}, sort_keys=True).encode()).hexdigest()
    (output/"results").mkdir(parents=True, exist_ok=True)
    records, rows, failures, offsets, reasons = [], [], [], [], Counter()
    for entry in detection["records"]:
        try:
            saved = json.loads((run_dir/entry["result_json"]).read_text(encoding="utf-8"))
            capture, source = saved["capture_metadata"], saved["source"]
            offset = nadir_offset(capture, terrain)
            if offset is not None:
                offsets.append(offset)
            vertical = offset if vertical_mode == "nadir_anchor" else 0.
            people = apply_terrain_geometry(saved, capture, source["processed_shape"], terrain,
                                            max_range_m=max_range_m, vertical_offset_m=vertical)
            name = Path(entry["result_json"]).name
            (output/"results"/name).write_text(json.dumps({
                "schema_version": SCHEMA_VERSION, "run_signature": signature,
                "detection_run_signature": saved["run_signature"], "vertical_mode": vertical_mode,
                "nadir_offset_m": offset,
                "source": {k: source[k] for k in ("relative_path", "sha256", "processed_shape")},
                "people": people}, indent=2, allow_nan=False), encoding="utf-8")
            records.append({"source": entry["source"], "result_json": f"results/{name}"})
            for person in people:
                reasons[person["reason"] or "estimated"] += 1
                rows.append({"source": entry["source"], "person_index": person["index"],
                             **{k: person.get(k) for k in CSV_FIELDS[2:]}})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"source": entry.get("source"), "error": f"{type(exc).__name__}: {exc}"})
    with (output/"terrain_ground.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "failed" if failures else "complete", "failures": failures,
        "run_signature": signature, "detector_run_signature": detection["run_signature"],
        "detection_run": run_dir.name, "vertical_mode": vertical_mode, "max_range_m": max_range_m,
        "terrain": {"path": str(terrain_path), "sha256": terrain_sha,
                    "geoid_height_m": terrain.geoid_height_m, "metadata": terrain.metadata},
        "datum_check": {"photos": len(offsets),
                        "median_m": statistics.median(offsets) if offsets else None,
                        "min_m": min(offsets) if offsets else None, "max_m": max(offsets) if offsets else None,
                        "meaning": "Launch-plane HAE minus terrain HAE under the camera. Near 0 if the drone "
                                   "hovered over takeoff-level ground; a large, uniform value suggests a "
                                   "vertical datum or GPS-altitude bias."},
        "summary": {"estimated": reasons.pop("estimated", 0), "unavailable_reasons": dict(reasons)},
        "records": records, "finished_at": datetime.now(timezone.utc).isoformat()}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="photos_det2048_native")
    parser.add_argument("--terrain", type=Path, default=ROOT/"terrain"/"site_dem.npz")
    parser.add_argument("--vertical-mode", choices=("absolute", "nadir_anchor"), default="absolute")
    parser.add_argument("--max-range-m", type=float, default=1000.)
    parser.add_argument("--output", help="Folder name under outputs/ (default terrain_<run>[_nadir])")
    args = parser.parse_args(argv)
    name = args.output or f"terrain_{args.run}" + ("_nadir" if args.vertical_mode == "nadir_anchor" else "")
    if Path(name).is_absolute() or Path(name).name != name or "/" in name or "\\" in name or not name.startswith("terrain_"):
        parser.error("--output must be a plain folder name starting with 'terrain_' (it is created under outputs/)")
    manifest = run(ROOT/"outputs"/args.run, args.terrain, ROOT/"outputs"/name,
                   vertical_mode=args.vertical_mode, max_range_m=args.max_range_m)
    print(json.dumps({k: manifest[k] for k in ("status", "summary", "datum_check")}, indent=2))
    print(ROOT/"outputs"/name)


if __name__ == "__main__":
    main()
