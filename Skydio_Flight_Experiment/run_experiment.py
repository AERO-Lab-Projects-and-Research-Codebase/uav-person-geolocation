"""Three people photographed by the Skydio from the launch point at 10-50 m; every depth model and placement
method estimates each person's GPS, scored against the drone GPS of the nadir shot taken over that person's head.

Truth: Person_N*.JPG are nadir shots with the drone over person N. The drone hovered 0.2-0.6 m off the head, so
truth N is the ground-plane position of the box under the image centre in those shots (mean over shots); it
reproduces the tape spacing (5.18 m / 5.94 m) where the raw drone GPS does not. truth.csv keeps both.
Detections in a flight photo are matched to the three people by the pose-only ground-plane position (model
independent): the triple whose mutual spacing best matches the truth spacing, ties broken by proximity, so a
heading error (which rotates every estimate about the drone) cannot hand a person's slot to a bystander.

Usage (from the project root, in the depth-comparison venv):
    python Skydio_Flight_Experiment/run_experiment.py [--models unidepth depth_pro ...]
Writes outputs/<photo>_<model>.json and _annotated.jpg (skipped when present), people.csv, truth.csv, summary.csv/.md.
"""
import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "reference_validation")]
from person_geolocator.engine import locate  # noqa: E402
from person_geolocator.metadata import read_pose  # noqa: E402
from person_geolocator.models import MODELS, DepthModels  # noqa: E402
from compare_to_ground_truth import local_enu  # noqa: E402

PHOTOS = HERE / "Skydio"
OUT = HERE / "outputs"
PEOPLE = (1, 2, 3)


def dist(lat0, lon0, lat, lon):
    return math.hypot(*local_enu(lat0, lon0, lat, lon))


def truth(results):
    """Truth lat/lon per person: the nadir shot's ground-plane position of the box under the image centre,
    falling back to the drone GPS when no box was found there."""
    rows = []
    for n in PEOPLE:
        shots = [p for p in sorted(PHOTOS.glob(f"Person_{n}*.JPG"))]
        poses = [read_pose(p)["fields"] for p in shots]
        drone = statistics.fmean(p["lat"] for p in poses), statistics.fmean(p["lon"] for p in poses)
        under = []
        for shot in shots:
            r = results[shot.stem]
            w, h = r["image_size"]
            # the person under the drone is the box nearest the image centre, if the detector saw the head at all
            near = [p for p in r["people"] if p["methods"]["ground_plane"]["status"] == "estimated"
                    and math.hypot((p["bbox"][0] + p["bbox"][2]) / 2 - w / 2, (p["bbox"][1] + p["bbox"][3]) / 2 - h / 2) < h / 4]
            if near:
                g = min(near, key=lambda p: math.hypot((p["bbox"][0] + p["bbox"][2]) / 2 - w / 2, (p["bbox"][1] + p["bbox"][3]) / 2 - h / 2))["methods"]["ground_plane"]
                under.append((g["lat"], g["lon"]))
        lat, lon = map(statistics.fmean, zip(*under)) if under else drone
        rows.append({"person": n, "lat": lat, "lon": lon, "source": "nadir ground plane" if under else "drone gps", "shots": len(shots),
                     "drone_lat": drone[0], "drone_lon": drone[1],
                     "height_m": round(statistics.fmean(p["height_above_ground_m"] for p in poses), 2),
                     "hover_offset_m": round(dist(*drone, lat, lon), 2)})
    return rows


def assign(people, truths):
    """{truth person: detection}: the boxes whose ground-plane mutual spacing best matches the truth spacing
    (rotation and translation invariant), plus 0.2 x the summed distance to the truth points to orient the triple."""
    placed = [p for p in people if p["methods"]["ground_plane"]["status"] == "estimated"]
    k = min(len(placed), len(truths))
    at = lambda p: (p["methods"]["ground_plane"]["lat"], p["methods"]["ground_plane"]["lon"])

    def cost(combo):
        shape = sum(abs(dist(*at(combo[i]), *at(combo[j])) - dist(truths[i]["lat"], truths[i]["lon"], truths[j]["lat"], truths[j]["lon"]))
                    for i, j in itertools.combinations(range(k), 2))
        return shape + .2 * sum(dist(t["lat"], t["lon"], *at(p)) for t, p in zip(truths, combo))
    best = min(itertools.permutations(placed, k), key=cost)
    return {t["person"]: p for t, p in zip(truths, best)}


def bearing(lat0, lon0, lat, lon):
    return math.degrees(math.atan2(*reversed(local_enu(lat0, lon0, lat, lon)))) % 360


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence floor; nadir heads score ~0.2")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    models = DepthModels()
    models.detect = lambda frame: DepthModels.detect(models, frame, conf=args.conf)
    photos = sorted(PHOTOS.glob("*.JPG"))
    results = {}
    for model in args.models:
        for path in photos:
            out = OUT / f"{path.stem}_{model}.json"
            if out.exists():
                results[path.stem, model] = json.loads(out.read_text())
                continue
            pose = read_pose(path)["fields"]
            result = locate(path, model, pose, models)
            (OUT / f"{path.stem}_{model}_annotated.jpg").write_bytes(result.pop("annotated_jpeg"))
            result.update(photo=path.name, pose=pose)
            out.write_text(json.dumps(result, indent=1))
            results[path.stem, model] = result
            print(f"{path.name} {model}: {len(result['people'])} box(es), {result['seconds']:.0f} s", flush=True)

    first = args.models[0]
    truths = truth({k[0]: v for k, v in results.items() if k[1] == first})
    with (OUT / "truth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(truths[0])); w.writeheader(); w.writerows(truths)
    for a, b in ((0, 1), (1, 2)):
        print(f"truth person {a + 1} to {b + 1}: {dist(truths[a]['lat'], truths[a]['lon'], truths[b]['lat'], truths[b]['lon']):.2f} m")

    rows, errors, photos_seen = [], defaultdict(list), {}
    for (stem, model), r in results.items():
        if stem.startswith("Person_"):
            continue
        pose = r["pose"]
        picked = assign(r["people"], truths)
        plane = {n: p["methods"]["ground_plane"] for n, p in picked.items()}
        # pose diagnostics (model independent): rotation of the estimated P1->P3 line against the truth line,
        # a heading error shows here; a drone GPS error does not
        rotation = (bearing(plane[1]["lat"], plane[1]["lon"], plane[3]["lat"], plane[3]["lon"])
                    - bearing(truths[0]["lat"], truths[0]["lon"], truths[2]["lat"], truths[2]["lon"]) + 180) % 360 - 180 if {1, 3} <= plane.keys() else None
        for n, p in picked.items():
            t = truths[n - 1]
            truth_h, truth_b = dist(pose["lat"], pose["lon"], t["lat"], t["lon"]), bearing(pose["lat"], pose["lon"], t["lat"], t["lon"])
            row = {"image": r["photo"], "altitude_label_m": round(pose["height_above_ground_m"] / 10) * 10,
                   "height_m": round(pose["height_above_ground_m"], 2), "pitch_deg": round(pose["pitch_deg"], 2), "heading_deg": round(pose["heading_deg"], 2),
                   "model": model, "person": n, "det_index": p["index"], "confidence": round(p["confidence"], 3),
                   "bbox": " ".join(str(round(v)) for v in p["bbox"]), "depth_m": p["depth_m"],
                   "truth_horizontal_m": round(truth_h, 2), "truth_bearing_deg": round(truth_b, 1),
                   "plane_bearing_offset_deg": round((bearing(pose["lat"], pose["lon"], plane[n]["lat"], plane[n]["lon"]) - truth_b + 180) % 360 - 180, 1),
                   "plane_line_rotation_deg": None if rotation is None else round(rotation, 1)}
            for method, m in p["methods"].items():
                if m["status"] != "estimated":
                    row[f"{method}_error_m"] = f"n/a ({m['reason']})"
                    continue
                row[f"{method}_lat"], row[f"{method}_lon"], row[f"{method}_range_m"] = m["lat"], m["lon"], round(m["range_m"], 2)
                row[f"{method}_horizontal_m"] = round(dist(pose["lat"], pose["lon"], m["lat"], m["lon"]), 2)
                row[f"{method}_range_error_m"] = round(row[f"{method}_horizontal_m"] - truth_h, 2)
                row[f"{method}_error_m"] = round(dist(t["lat"], t["lon"], m["lat"], m["lon"]), 2)
                errors[model, method, row["altitude_label_m"]].append((row[f"{method}_error_m"], row[f"{method}_range_error_m"]))
            rows.append(row)
        photos_seen.setdefault(r["photo"], rows[-3:])
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "people.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)

    # summary: per model x method x altitude, mean position error and mean |horizontal range error| (heading independent)
    altitudes = sorted({a for _, _, a in errors})
    fmean = lambda v: round(statistics.fmean(v), 2) if v else ""
    summary = []
    for model in args.models:
        for method in dict.fromkeys(m for mo, m, _ in errors if mo == model):
            cells = {a: errors.get((model, method, a), []) for a in altitudes}
            total = [e for v in cells.values() for e in v]
            summary.append({"model": MODELS[model], "method": method,
                            **{k: v for a, c in cells.items() for k, v in ((f"{a}m_pos_m", fmean([e for e, _ in c])), (f"{a}m_abs_range_m", fmean([abs(d) for _, d in c])))},
                            "all_pos_mean_m": fmean([e for e, _ in total]), "all_pos_median_m": round(statistics.median(e for e, _ in total), 2),
                            "all_abs_range_mean_m": fmean([abs(d) for _, d in total]), "all_range_bias_m": fmean([d for _, d in total]), "n": len(total)})
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    pose_rows = [{"photo": k, "height_m": v[0]["height_m"], "pitch_deg": v[0]["pitch_deg"], "heading_deg": v[0]["heading_deg"],
                  "plane_line_rotation_deg": v[0]["plane_line_rotation_deg"], "mean_plane_bearing_offset_deg": fmean([r["plane_bearing_offset_deg"] for r in v]),
                  "mean_plane_range_error_m": fmean([r["ground_plane_range_error_m"] for r in v if "ground_plane_range_error_m" in r])}
                 for k, v in sorted(photos_seen.items())]

    def table(dicts):
        return ["| " + " | ".join(dicts[0]) + " |", "|" + "---|" * len(dicts[0])] + ["| " + " | ".join(str(v) for v in d.values()) + " |" for d in dicts]
    text = "\n".join(["## Position error (m, vs nadir truth) and |horizontal range error| (m) per altitude", *table(summary), "",
                      "## Truth", *table(truths), "", "## Pose diagnostics from the ground plane (model independent)", *table(pose_rows)]) + "\n"
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
