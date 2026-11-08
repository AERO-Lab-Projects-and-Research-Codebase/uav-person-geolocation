"""Score the saved person_geolocator runs of one Mini SE flight against its recorded person position.

Horizontal error of every method (depth ray, flat ground plane, fusion, terrain when present) in each
outputs/<run>* folder listed for the sheet in minise_flights.json, against the sheet's first truth entry.
The saved runs predate the fusion gate, so fusion is derived from the saved depth and plane picks.

Usage (from reference_validation):  python score_minise_flight.py --sheet 8
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skydio live data"))
from human_geolocation import fuse_estimates  # noqa: E402
from compare_to_ground_truth import local_enu, mean, med  # noqa: E402


def fusion(person, pose):
    """fuse_estimates on the saved method picks, with offsets rebuilt from the pose and the plane ray's depression from height / range."""
    def estimate(m, range_key):
        if m["status"] != "estimated":
            return {"status": "unavailable", "reason": m.get("reason")}
        north, east = local_enu(pose["lat"], pose["lon"], m["lat"], m["lon"])
        return {"status": "estimated", "latitude": m["lat"], "longitude": m["lon"], "north_m": north, "east_m": east, range_key: m["range_m"]}
    ground = estimate(person["methods"]["ground_plane"], "ground_range_m")
    if ground["status"] == "estimated":
        ground["depression_deg"] = math.degrees(math.asin(min(1., pose["height_above_ground_m"] / ground["ground_range_m"])))
    fused = fuse_estimates(estimate(person["methods"]["depth_ray"], "slant_range_m"), ground)
    return {"status": fused["status"], "lat": fused["latitude"], "lon": fused["longitude"], "range_m": fused["range_m"]}


def score(root, flight):
    truth = flight["truth"][0]
    errors = defaultdict(list)
    for run in sorted(p for p in (root / "outputs").glob(flight["run"] + "*") if p.is_dir()):
        for path in sorted(run.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if "people" not in record:  # this scorer's own summary lives in the same folder
                continue
            for person in record["people"]:
                methods = dict(person["methods"], fusion=fusion(person, record["pose"]))
                for method, estimate in methods.items():
                    if estimate["status"] != "estimated":
                        continue
                    north, east = local_enu(truth["latitude"], truth["longitude"], estimate["lat"], estimate["lon"])
                    errors[(run.name, method)].append(math.hypot(north, east))
    return {f"{run} / {method}": {"n": len(v), "mean_position_error_m": mean(v), "median_position_error_m": med(v)}
            for (run, method), v in errors.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=int, default=8)
    args = ap.parse_args()
    flights = json.loads((HERE / "minise_flights.json").read_text(encoding="utf-8"))["flights"]
    flight = next(f for f in flights if f["sheet_id"] == args.sheet)
    result = score(HERE, flight)
    (HERE / "outputs" / flight["run"] / "gt_comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
