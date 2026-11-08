"""Compare model person positions with the handwritten reference coordinates.

For every photo detection, the drone-to-person range and bearing implied by the
model (depth-based and ground-plane estimates) are compared with the range and
bearing from the same origin to the surveyed person coordinate.

Detections are matched to people by bearing only, one person per detection,
using the photo's sheet.  No range or position information is used for
matching, so range error is not biased by the assignment.  The reference
coordinates are whole arcseconds, so each true range is also reported as an
interval covering the +/-0.5 arcsecond rounding cell.

--match order pairs by front-to-back order instead: the steepest box-bottom ray
goes with the nearest recorded person; people whose recorded ranges are within
TIE_M of each other cannot be ordered and are told apart by bearing.  It only
pairs a photo that shows exactly
as many people as its sheet records, on a sheet with every coordinate readable,
and writes gt_comparison_order.* so the bearing files are kept.  Bearing cannot
separate people standing in a line away from the camera; order can.

Usage (from the project root):
    python reference_validation/compare_to_ground_truth.py [--run photos] [--gate-deg 30] [--match order]
"""
import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = 6378137.0
E2 = 6.69437999014e-3
HALF_CELL_DEG = 0.5 / 3600.0
AMBIGUITY_MARGIN_DEG = 10.0
TIE_M = 10.0  # recorded ranges closer than this cannot be ordered: the positions are whole arcseconds
# detections.csv columns (north, east, status) per scored method.
METHODS = {"depth": ("north_m", "east_m", "status"),
           "ground": ("ground_north_m", "ground_east_m", "ground_status"),
           "fusion": ("fusion_north_m", "fusion_east_m", "fusion_status")}


def local_enu(lat0, lon0, lat, lon):
    """North/east metres of (lat, lon) from (lat0, lon0)."""
    s = math.sin(math.radians(lat0))
    w = 1.0 - E2 * s * s
    north_per_deg = math.radians(1) * A * (1 - E2) / w ** 1.5
    east_per_deg = math.radians(1) * A * math.cos(math.radians(lat0)) / math.sqrt(w)
    return (lat - lat0) * north_per_deg, (lon - lon0) * east_per_deg


def bearing_deg(north, east):
    return math.degrees(math.atan2(east, north)) % 360.0


def ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def as_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_truth(unreadable=None):
    """People per sheet; sheets with a row that has no readable coordinate are added to `unreadable`."""
    by_sheet = defaultdict(list)
    with open(HERE / "ground_truth.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lat = as_float(row["normalized_latitude"])
            lon = as_float(row["normalized_longitude"])
            if lat is None or lon is None:
                if unreadable is not None:
                    unreadable.add(row["sheet_id"])
                continue
            by_sheet[row["sheet_id"]].append(
                {"name": row["person_name"], "lat": lat, "lon": lon, "status": row["status"]}
            )
    return by_sheet


def range_interval(north, east, half_n, half_e):
    """Min and max distance from the origin to the rounding rectangle."""
    cx = min(max(0.0, north - half_n), north + half_n)
    cy = min(max(0.0, east - half_e), east + half_e)
    lo = math.hypot(cx, cy)
    hi = max(math.hypot(north + sn * half_n, east + se * half_e) for sn in (-1, 1) for se in (-1, 1))
    return lo, hi


def assign(det_bearings, person_bearings, gate):
    """Minimum-cost one-to-one bearing assignment; returns {det_idx: person_idx}.

    A detection left unassigned costs `gate`, so a detection is only matched
    when its bearing error is smaller than the gate.
    """
    n, m = len(det_bearings), len(person_bearings)
    best = (float("inf"), {})

    def dfs(i, used, cost, chosen):
        nonlocal best
        if cost >= best[0]:
            return
        if i == n:
            best = (cost, dict(chosen))
            return
        for j in range(m):
            if j in used or det_bearings[i] is None:
                continue
            d = ang_diff(det_bearings[i], person_bearings[j])
            if d < gate:
                chosen[i] = j
                dfs(i + 1, used | {j}, cost + d, chosen)
                del chosen[i]
        dfs(i + 1, used, cost + gate, chosen)

    dfs(0, frozenset(), 0.0, {})
    return best[1]


def assign_by_order(depressions, ranges, det_bearings=None, person_bearings=None):
    """Steepest box-bottom ray with the nearest person; {} unless the counts match and every ray is known.

    People whose recorded range is within TIE_M of the next one form a group that order cannot separate;
    when bearings are given, such a group goes to its detections by smallest total bearing difference.
    """
    if len(depressions) != len(ranges) or any(d is None for d in depressions):
        return {}
    dets = sorted(range(len(depressions)), key=lambda i: -depressions[i])
    people = sorted(range(len(ranges)), key=lambda j: ranges[j])
    match, start = {}, 0
    for end in range(1, len(people) + 1):
        if end < len(people) and ranges[people[end]] - ranges[people[end - 1]] < TIE_M:
            continue
        group_d, group_p = dets[start:end], people[start:end]
        if len(group_p) > 1 and det_bearings and all(det_bearings[i] is not None for i in group_d):
            group_p = min(itertools.permutations(group_p),
                          key=lambda perm: sum(ang_diff(det_bearings[i], person_bearings[j]) for i, j in zip(group_d, perm)))
        match.update(zip(group_d, group_p))
        start = end
    return match


def depressions(out_dir, group):
    """Depression angle of each detection's box-bottom ray, from the photo's result JSON."""
    with open(out_dir / group[0]["result_json"], encoding="utf-8") as fh:
        by_index = {str(p["index"]): (p.get("ground_plane") or {}).get("depression_deg") for p in json.load(fh)["people"]}
    return [by_index.get(d["person_index"]) for d in group]


def med(values):
    return round(statistics.median(values), 2) if values else None


def mean(values):
    return round(statistics.fmean(values), 2) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="photos", help="folder under outputs/")
    ap.add_argument("--gate-deg", type=float, default=30.0, help="max bearing error to accept a match")
    ap.add_argument("--inventory-run", help="folder under outputs/ holding photo_inventory.csv (default: --run)")
    ap.add_argument("--match", choices=("bearing", "order"), default="bearing", help="how detections are paired with people")
    args = ap.parse_args()

    out_dir = HERE / "outputs" / args.run
    by_order = args.match == "order"
    stem = "gt_comparison_order" if by_order else "gt_comparison"
    unreadable = set()
    truth = load_truth(unreadable)
    with open(HERE / "outputs" / (args.inventory_run or args.run) / "photo_inventory.csv", newline="", encoding="utf-8-sig") as fh:
        inv = {r["source"]: r for r in csv.DictReader(fh)}
    with open(out_dir / "detections.csv", newline="", encoding="utf-8-sig") as fh:
        dets = list(csv.DictReader(fh))

    by_photo = defaultdict(list)
    for d in dets:
        by_photo[d["source"]].append(d)

    rows = []
    for source, group in by_photo.items():
        meta = inv[source]
        sheet = meta["sheet_id"]
        lat0, lon0 = float(meta["camera_latitude"]), float(meta["camera_longitude"])
        people = truth.get(sheet, [])
        half_n, half_e = local_enu(lat0, lon0, lat0 + HALF_CELL_DEG, lon0 + HALF_CELL_DEG)
        for p in people:
            p["north"], p["east"] = local_enu(lat0, lon0, p["lat"], p["lon"])
            p["bearing"] = bearing_deg(p["north"], p["east"])
            p["range"] = math.hypot(p["north"], p["east"])
            p["r_lo"], p["r_hi"] = range_interval(p["north"], p["east"], half_n, half_e)

        est = []
        for d in group:
            e = {}
            for method, (kn, ke, ks) in METHODS.items():
                n, ea = as_float(d.get(kn)), as_float(d.get(ke))
                if d.get(ks) == "estimated" and n is not None and ea is not None:
                    e[method] = {"north": n, "east": ea, "bearing": bearing_deg(n, ea), "range": math.hypot(n, ea)}
            est.append(e)

        # Match on bearing only; use the depth ray, falling back to the ground ray.
        det_bearings = [(e.get("depth") or e.get("ground") or {}).get("bearing") for e in est]
        if by_order:
            match = {} if sheet in unreadable else assign_by_order(depressions(out_dir, group), [p["range"] for p in people],
                                                                           det_bearings, [p["bearing"] for p in people])
        else:
            match = assign(det_bearings, [p["bearing"] for p in people], args.gate_deg)

        for i, d in enumerate(group):
            j = match.get(i)
            row = {
                "source": source,
                "photo_kind": d["photo_kind"],
                "camera_source": d["camera_source"],
                "sheet_id": sheet,
                "relative_altitude_m": meta["relative_altitude_m"],
                "person_index": d["person_index"],
                "confidence": d["confidence"],
                "matched_person": people[j]["name"] if j is not None else "",
                "match_note": "",
                "bearing_used_deg": round(det_bearings[i], 1) if det_bearings[i] is not None else "",
            }
            if j is None:
                row["match_note"] = ("photo_not_paired_by_order" if by_order else "unmatched_no_person_within_gate") if people else "no_ground_truth_for_sheet"
            else:
                p = people[j]
                others = [ang_diff(det_bearings[i], q["bearing"]) for k, q in enumerate(people) if k != j]
                if not by_order and others and min(others) - ang_diff(det_bearings[i], p["bearing"]) < AMBIGUITY_MARGIN_DEG:
                    row["match_note"] = "ambiguous_bearing"
                row.update(
                    true_range_m=round(p["range"], 2),
                    true_range_min_m=round(p["r_lo"], 2),
                    true_range_max_m=round(p["r_hi"], 2),
                    true_bearing_deg=round(p["bearing"], 1),
                    truth_status=p["status"],
                )
                for method in METHODS:
                    m = est[i].get(method)
                    if not m:
                        continue
                    pos_err = math.hypot(m["north"] - p["north"], m["east"] - p["east"])
                    row.update(
                        {
                            f"{method}_range_m": round(m["range"], 2),
                            f"{method}_range_error_m": round(m["range"] - p["range"], 2),
                            f"{method}_range_in_truth_interval": p["r_lo"] <= m["range"] <= p["r_hi"],
                            f"{method}_bearing_deg": round(m["bearing"], 1),
                            f"{method}_bearing_error_deg": round(ang_diff(m["bearing"], p["bearing"]), 1),
                            f"{method}_position_error_m": round(pos_err, 2),
                        }
                    )
            rows.append(row)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def summarize(subset):
        matched = [r for r in subset if r["matched_person"]]
        clean = [r for r in matched if r["match_note"] != "ambiguous_bearing"]
        out = {
            "detections": len(subset),
            "matched": len(matched),
            "matched_unambiguous": len(clean),
            "unmatched": len(subset) - len(matched),
        }
        for method in METHODS:
            vals = [r for r in matched if f"{method}_range_error_m" in r]
            err = [r[f"{method}_range_error_m"] for r in vals]
            out[method] = {
                "n": len(vals),
                "median_true_range_m": med([r["true_range_m"] for r in vals]),
                "median_est_range_m": med([r[f"{method}_range_m"] for r in vals]),
                "median_signed_range_error_m": med(err),
                "median_abs_range_error_m": med([abs(x) for x in err]),
                "mean_abs_range_error_m": mean([abs(x) for x in err]),
                "median_bearing_error_deg": med([r[f"{method}_bearing_error_deg"] for r in vals]),
                "median_position_error_m": med([r[f"{method}_position_error_m"] for r in vals]),
                "mean_position_error_m": mean([r[f"{method}_position_error_m"] for r in vals]),
                "share_range_inside_truth_interval": round(
                    sum(r[f"{method}_range_in_truth_interval"] for r in vals) / len(vals), 3
                )
                if vals
                else None,
            }
        return out

    summary = {"match": args.match, "gate_deg": args.gate_deg, "overall": summarize(rows), "by_photo_kind": {}, "by_altitude_sheet": {}}
    for key, field in (("by_photo_kind", "photo_kind"), ("by_altitude_sheet", "sheet_id")):
        groups = defaultdict(list)
        for r in rows:
            label = f"{r['camera_source']} / {r[field]}" if field == "photo_kind" else f"sheet {r[field]}"
            groups[label].append(r)
        for label, subset in sorted(groups.items()):
            summary[key][label] = summarize(subset)
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    def line(label, s):
        parts = [f"| {label} | {s['detections']} | {s['matched']} |"]
        for method in METHODS:
            m = s[method]
            parts.append(
                f" {m['median_true_range_m']} | {m['median_est_range_m']} | {m['median_signed_range_error_m']} |"
                f" {m['median_abs_range_error_m']} | {m['median_bearing_error_deg']} | {m['median_position_error_m']} |"
                f" {m['share_range_inside_truth_interval']} |"
            )
        return "".join(parts)

    md = ["# Ground-truth comparison", "",
          ("Front-to-back order pairing." if by_order else f"Bearing-only matching, gate {args.gate_deg} deg.")
          + " Ranges are horizontal metres from the camera origin.", "",
          "| Group | Dets | Matched |" + "".join(
              f" True rng ({m} set) | {m.capitalize()} est rng | {m.capitalize()} signed err | {m.capitalize()} abs err |"
              f" {m.capitalize()} bearing err deg | {m.capitalize()} pos err | {m.capitalize()} in GT interval |"
              for m in METHODS),
          "|" + "---|" * (3 + 7 * len(METHODS))]
    md.append(line("ALL", summary["overall"]))
    for label, s in summary["by_photo_kind"].items():
        md.append(line(label, s))
    for label, s in summary["by_altitude_sheet"].items():
        md.append(line(label, s))
    (out_dir / f"{stem}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
