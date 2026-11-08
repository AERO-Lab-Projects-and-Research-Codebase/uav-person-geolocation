"""Score two candidate improvements to Method E offline, from the saved runs, without touching the live rule.

  ivw   inverse-variance mean of the depth ray and the plane: depth sigma = s * R (relative scale spread),
        plane sigma = R * sigma_pitch / (sin d cos d) for a ray d below the horizon, so no angle thresholds.
        On gross disagreement (ratio beyond 2x) the depth sigma is widened to the observed ratio gap.
  cal   depth scale calibrated per photo by the plane: k = median(depth range / plane range) over the photo's
        rays at least 15 deg down, applied to every ray in the photo; photos with no steep ray use the run's
        leave-one-photo-out median.  Reported alone and fused with ivw.

Field set: 149 matched detections per model run.  Flights 2-4: position error from the recorded person position.
Lab: one range per photo, image-read plane, no steep rays so no calibration.

Usage (from reference_validation):  python score_fusion_variants.py
"""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skydio live data"))
from human_geolocation import fuse_estimates  # noqa: E402
from compare_to_ground_truth import local_enu, as_float  # noqa: E402

S_DEPTH, SIGMA_PITCH_DEG, CAL_MIN_DEPRESSION_DEG = 0.3, 1.0, 15.0
# Pitch uncertainty of the lab's image-read pose for the live rule's gate ("gated" column): the plane's lab error
# implies pitch errors of 0.05-0.44 deg, the largest on the 7.5 m photo, so 0.5 deg is the conservative bound.
# The Skydio and Mini SE metadata poses have no measured uncertainty, so they keep the 10 deg floor.
IMAGE_POSE_PITCH_SIGMA_DEG = 0.5
FIELD_RUNS = {"YOLO26-depth": "photos_det2048_native", "Depth Pro": "depth_det2048_native/depth_pro",
              "UniDepth V2": "depth_det2048_native/unidepth_v2", "LingBot": "depth_det2048_native/lingbot_depth",
              "Fine-tuned": "finetuned_det2048_native/finetuned"}
FLIGHT_RUNS = {"UniDepth V2": "", "LingBot": "_lingbot", "Depth Pro": "_depth_pro",
               "YOLO26-depth": "_yolo_original", "Fine-tuned": "_yolo_finetuned"}
LAB_COLS = {"YOLO26-depth": "yolo_original_m", "Fine-tuned": "yolo_finetuned_m", "UniDepth V2": "unidepth_m",
            "LingBot": "lingbot_m", "Depth Pro": "depth_pro_m"}


def ivw(depth, plane, depression_deg, s=S_DEPTH, sigma_pitch_deg=SIGMA_PITCH_DEG):
    """Weighted mean of two (north, east) estimates; either may be None.  Returns (north, east) or None."""
    if plane is None or depression_deg is None or depression_deg <= 0:
        return depth
    if depth is None:  # a lone plane keeps the live rule's 15 deg floor
        return plane if depression_deg >= 15 else None
    rd, rp = math.hypot(*depth), math.hypot(*plane)
    ratio = rd / rp
    sd = max(s, abs(ratio - 1)) if (ratio > 2 or ratio < 0.5) else s
    d = math.radians(depression_deg)
    sp = math.radians(sigma_pitch_deg) / (math.sin(d) * math.cos(d))
    wd, wp = 1 / sd ** 2, 1 / sp ** 2
    return tuple((wd * a + wp * b) / (wd + wp) for a, b in zip(depth, plane))


def current(depth, plane, depression_deg, pitch_sigma_deg=None):
    """The live rule on the same inputs, so the baseline is re-derived exactly as the paper's table was."""
    d = {"status": "estimated", "north_m": depth[0], "east_m": depth[1]} if depth else {"status": "unavailable"}
    g = ({"status": "estimated", "north_m": plane[0], "east_m": plane[1], "depression_deg": depression_deg,
          "pitch_sigma_deg": pitch_sigma_deg}
         if plane and depression_deg is not None else {"status": "unavailable"})
    f = fuse_estimates(d, g)
    return (f["north_m"], f["east_m"]) if f["status"] == "estimated" else None


def calibrate(dets):
    """Scale factors per photo from the plane on steep rays; dets: list of dicts with photo, depth, plane, dep."""
    ratios = defaultdict(list)
    for t in dets:
        if t["depth"] and t["plane"] and t["dep"] is not None and t["dep"] >= CAL_MIN_DEPRESSION_DEG:
            ratios[t["photo"]].append(math.hypot(*t["depth"]) / math.hypot(*t["plane"]))
    k = {}
    for t in dets:
        if t["photo"] in k:
            continue
        own = ratios.get(t["photo"])
        others = [r for p, rs in ratios.items() if p != t["photo"] for r in rs]
        k[t["photo"]] = (statistics.median(own) if own else statistics.median(others) if others else None,
                         "photo" if own else "run")
    return k


def variants(dets):
    """Error of every variant on a list of detections with truth; returns {variant: [errors]} plus counts."""
    k = calibrate(dets)
    out, cal_source = defaultdict(list), defaultdict(int)
    for t in dets:
        depth, plane, dep = t["depth"], t["plane"], t["dep"]
        kf, src = k.get(t["photo"], (None, None))
        cal = tuple(v / kf for v in depth) if depth and kf else None
        if cal:
            cal_source[src] += 1
        for name, est in (("depth", depth), ("plane", plane), ("current", current(depth, plane, dep)),
                          ("gated", current(depth, plane, dep, t.get("sigma"))),
                          ("ivw", ivw(depth, plane, dep)), ("cal", cal), ("cal+ivw", ivw(cal, plane, dep))):
            if est:
                out[name].append(t["err"](est))
    return out, dict(cal_source)


def summarize(errs, cal_source):
    table = {name: {"n": len(v), "mean": round(statistics.fmean(v), 1), "median": round(statistics.median(v), 1)}
             for name, v in errs.items()}
    table["_calibration_source"] = cal_source
    return table


def field(run):
    """149 matched detections: error = |horizontal range - true range| (the paper's metric)."""
    folder = HERE / "outputs" / run
    gt = {(r["source"], r["person_index"]): r for r in csv.DictReader(open(folder / "gt_comparison.csv", encoding="utf-8-sig"))
          if r.get("matched_person")}
    dets = []
    for d in csv.DictReader(open(folder / "detections.csv", encoding="utf-8-sig")):
        key = (d["source"], d["person_index"])
        if key not in gt:
            continue
        rec = json.loads((folder / d["result_json"]).read_text(encoding="utf-8")) if d.get("result_json") else None
        person = next((p for p in rec["people"] if str(p["index"]) == d["person_index"]), {}) if rec else {}
        dep = (person.get("ground_plane") or {}).get("depression_deg")
        depth = (as_float(d["north_m"]), as_float(d["east_m"])) if d["status"] == "estimated" else None
        plane = (as_float(d["ground_north_m"]), as_float(d["ground_east_m"])) if d["ground_status"] == "estimated" else None
        truth = float(gt[key]["true_range_m"])
        dets.append({"photo": d["source"], "depth": depth, "plane": plane, "dep": dep,
                     "err": lambda ne, t=truth: abs(math.hypot(*ne) - t)})
    assert len(dets) == 149, len(dets)
    return dets


def flight(run, truth):
    """Position error (m) from the recorded person position, as score_minise_flight does."""
    dets = []
    for path in sorted((HERE / "outputs" / run).glob("DJI_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        pose = rec["pose"]
        for p in rec["people"]:
            def ne(m):
                return local_enu(pose["lat"], pose["lon"], m["lat"], m["lon"]) if m["status"] == "estimated" else None
            g = p["methods"]["ground_plane"]
            dep = math.degrees(math.asin(min(1.0, pose["height_above_ground_m"] / g["range_m"]))) if g["status"] == "estimated" else None
            tn, te = local_enu(pose["lat"], pose["lon"], truth["latitude"], truth["longitude"])
            dets.append({"photo": path.stem, "depth": ne(p["methods"]["depth_ray"]), "plane": ne(g), "dep": dep,
                         "err": lambda est, tn=tn, te=te: math.hypot(est[0] - tn, est[1] - te)})
    return dets


def lab(col):
    """One range per photo along a fixed bearing; the image-read plane and its depression angle."""
    dets = []
    for r in csv.DictReader(open(HERE.parent / "geolocation_experiment_image/depth_results/results.csv", encoding="utf-8-sig")):
        dets.append({"photo": r["image"], "depth": (float(r[col]), 0.0), "plane": (float(r["ground_plane_image_m"]), 0.0),
                     "dep": float(r["image_depression_deg"]), "sigma": IMAGE_POSE_PITCH_SIGMA_DEG,
                     "err": lambda ne, t=float(r["actual_m"]): abs(ne[0] - t)})
    return dets


def main():
    flights = {f["sheet_id"]: f for f in json.loads((HERE / "minise_flights.json").read_text(encoding="utf-8"))["flights"]}
    report = {"field": {m: summarize(*variants(field(run))) for m, run in FIELD_RUNS.items()},
              "lab": {m: summarize(*variants(lab(col))) for m, col in LAB_COLS.items()}}
    for sheet in (6, 7, 8):
        f = flights[sheet]
        report[f"flight{sheet - 4}"] = {m: summarize(*variants(flight(f["run"] + suffix, f["truth"][0])))
                                        for m, suffix in FLIGHT_RUNS.items()}
    # The re-derived live rule must reproduce the paper's fusion table (UniDepth V2 row), or the inputs are wrong.
    assert report["field"]["UniDepth V2"]["current"] == {"n": 149, "mean": 25.0, "median": 20.7}, report["field"]["UniDepth V2"]["current"]
    (HERE / "outputs" / "fusion_variants.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    cols = ("depth", "plane", "current", "gated", "ivw", "cal", "cal+ivw")
    for block, models in report.items():
        print(f"\n== {block}  (mean / median error m, n)")
        print(f"{'model':14}" + "".join(f"{c:>22}" for c in cols))
        for m, t in models.items():
            print(f"{m:14}" + "".join(f"{t[c]['mean']:>8.1f}/{t[c]['median']:<6.1f}({t[c]['n']:>3})" if c in t else f"{'-':>22}" for c in cols)
                  + f"  cal:{t['_calibration_source']}")


if __name__ == "__main__":
    main()
