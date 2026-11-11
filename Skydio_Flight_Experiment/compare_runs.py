"""Side-by-side UniDepth V2 errors of two run_experiment.py output folders (6 Nov tape test vs 9 Nov 15 m test).

Usage: python Skydio_Flight_Experiment/compare_runs.py outputs outputs_11_9
Prints, per altitude, the mean |horizontal range error|, the mean slant-range error as a percentage of the truth slant
range (the depth scale error) and the mean position error, for the raw depth ray and the ground-scaled depth, plus
the ground plane as the pose-only reference.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("depth_ray", "ground_scaled", "ground_plane")


def load(folder):
    cells = defaultdict(list)
    for r in csv.DictReader((HERE / folder / "people.csv").open()):
        if r["model"] != "unidepth":
            continue
        slant = math.hypot(float(r["height_m"]), float(r["truth_horizontal_m"]))
        for m in METHODS:
            if r.get(f"{m}_range_error_m", "").replace("-", "").replace(".", "").isdigit():
                e = float(r[f"{m}_range_error_m"])
                cells[m, int(r["altitude_label_m"])].append((e, float(r[f"{m}_range_m"]) / slant - 1, float(r[f"{m}_error_m"])))
    return cells


def main():
    runs = {f: load(f) for f in sys.argv[1:]}
    alts = sorted({a for c in runs.values() for _, a in c})
    for m in METHODS:
        print(f"\n## UniDepth V2 {m}: mean |horizontal range err| m / mean slant range err % / mean position err m")
        print("| altitude | " + " | ".join(runs) + " |")
        print("|---|" + "---|" * len(runs))
        for a in alts + ["all"]:
            row = []
            for c in runs.values():
                v = [x for (mm, aa), xs in c.items() if mm == m and (a == "all" or aa == a) for x in xs]
                row.append(f"{statistics.fmean(abs(e) for e, _, _ in v):.2f} / {100 * statistics.fmean(p for _, p, _ in v):+.0f}% / {statistics.fmean(d for _, _, d in v):.2f} (n={len(v)})" if v else "-")
            print(f"| {a} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
