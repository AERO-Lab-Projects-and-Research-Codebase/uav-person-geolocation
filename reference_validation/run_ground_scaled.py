"""Add the ground-scaled depth estimate to a saved depth run, without re-running any model.

Each saved result JSON holds the capture pose, lens, boxes and torso depths, and the run folder holds the depth
maps, so the estimate is re-derived with metadata_geometry.apply_capture_geometry(depth_map=...). Writes
<run>/ground_scaled.csv, one row per detection.

Usage (from the project root):
    python reference_validation/run_ground_scaled.py --run depth_det2048_native/unidepth_v2
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from metadata_geometry import apply_capture_geometry

ROOT = Path(__file__).resolve().parent
FIELDS = ["source", "person_index", "status", "reason", "ground_scale", "north_m", "east_m", "range_m", "slant_range_m"]


def run(run_dir, max_depth_m=300.):
    rows = []
    for path in sorted((Path(run_dir)/"results").glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not saved.get("depth_map"):
            raise ValueError(f"{path.name}: this run saved no depth map")
        depth = np.load(Path(run_dir)/saved["depth_map"])["depth"]
        placed = apply_capture_geometry(saved, saved["capture_metadata"], saved["source"]["processed_shape"],
                                        max_depth_m=max_depth_m, depth_map=depth)
        for old, new in zip(saved["people"], placed["people"]):
            # the re-derived depth ray must be the saved one, or the scaled estimate would describe a different run
            if (old.get("north_m") is None) != (new["north_m"] is None) or (
                    old.get("north_m") is not None and abs(old["north_m"]-new["north_m"]) > 1e-6):
                raise ValueError(f"{path.name} person {new.get('index')}: depth ray differs from the saved run")
            s = new["ground_scaled"]
            rows.append({"source": saved["source"]["relative_path"], "person_index": new["index"],
                         "status": s["status"], "reason": s["reason"], "ground_scale": placed["ground_scale"],
                         "north_m": s["north_m"], "east_m": s["east_m"], "slant_range_m": s["slant_range_m"],
                         "range_m": None if s["north_m"] is None else math.hypot(s["north_m"], s["east_m"])})
    with open(Path(run_dir)/"ground_scaled.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="folder under outputs/ that holds results/ and depth/")
    ap.add_argument("--max-depth-m", type=float, default=300., help="the limit the saved run was made with")
    args = ap.parse_args()
    rows = run(ROOT/"outputs"/args.run, args.max_depth_m)
    scales = sorted({r["source"]: r["ground_scale"] for r in rows if r["ground_scale"]}.values())
    print(f"{args.run}: {sum(r['status'] == 'estimated' for r in rows)} of {len(rows)} detections placed; "
          f"ground scale median {scales[len(scales)//2]:.2f} ({scales[0]:.2f} to {scales[-1]:.2f}) over {len(scales)} photos")


if __name__ == "__main__":
    main()
