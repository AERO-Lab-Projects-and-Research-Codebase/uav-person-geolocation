"""Held-out test of a ground-point correction on run_experiment.py output folders: in each flight photo, one or two
of the three people are treated as known ground points, the photo's estimates are corrected from them, and only the
person(s) left out are scored against the nadir truth.

One known point: every estimate is shifted by that point's error. Two known points: the estimates are turned and
shifted so the two known points land on their truth as closely as a rigid move allows (heading and position fix).
A rigid move in the ground plane removes a heading error and a drone position error, and to first order a small
camera tilt, so it does not need to know which of them caused the error.

Usage: python Skydio_Flight_Experiment/ground_point_correction.py outputs outputs_11_9
Prints the mean position error (m) per altitude for the UniDepth V2 run. With two known points the nearer
is 5-9 m from the scored person and with one it is 5-16 m away; how far from a known point the correction holds
is not tested here.
"""
import cmath
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from compare_to_ground_truth import local_enu  # noqa: E402

METHODS = ("ground_plane", "ground_scaled", "scaled_fusion", "fusion", "depth_ray")
CASES = ("no known point", "1 known point (shift)", "2 known points (turn + shift)")


def rigid_from_two(est, tru):
    """The rigid move (as a function of a complex north + i east point) that best maps two estimates onto two truths."""
    turn = (tru[1] - tru[0]) / (est[1] - est[0])
    turn /= abs(turn)
    return lambda z: (z - (est[0] + est[1]) / 2) * turn + (tru[0] + tru[1]) / 2


def held_out_errors(est, tru):
    """{case: [errors]} for one photo; est and tru are {person: complex point}."""
    out = defaultdict(list)
    for p in est:
        known = [k for k in est if k != p]
        out[CASES[0]].append(abs(est[p] - tru[p]))
        for k in known:
            out[CASES[1]].append(abs(est[p] + tru[k] - est[k] - tru[p]))
        if len(known) == 2:
            move = rigid_from_two([est[k] for k in known], [tru[k] for k in known])
            out[CASES[2]].append(abs(move(est[p]) - tru[p]))
    return out


def load(folder):
    """{(method, case, altitude): [errors]} from a run folder's people.csv and truth.csv."""
    truth = {r["person"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader((HERE / folder / "truth.csv").open())}
    lat0, lon0 = next(iter(truth.values()))
    point = lambda lat, lon: complex(*local_enu(lat0, lon0, float(lat), float(lon)))
    tru = {p: point(*ll) for p, ll in truth.items()}
    photos = defaultdict(list)
    for r in csv.DictReader((HERE / folder / "people.csv").open()):
        if r["model"] == "unidepth":
            photos[r["image"]].append(r)
    cells = defaultdict(list)
    for rows in photos.values():
        for m in METHODS:
            est = {r["person"]: point(r[f"{m}_lat"], r[f"{m}_lon"]) for r in rows if r.get(f"{m}_lat")}
            for case, errors in held_out_errors(est, tru).items():
                cells[m, case, int(rows[0]["altitude_label_m"])] += errors
    return cells


def main():
    # self-check: estimates that are the truth turned 10 deg about a far point and shifted are fully recovered
    tru = {"1": 3 + 4j, "2": 8 - 1j, "3": 12 + 6j}
    est = {p: (z + 20) * cmath.rect(1, 0.17) + (1.5 - 2j) for p, z in tru.items()}
    assert max(held_out_errors(est, tru)[CASES[2]]) < 1e-9

    for folder in sys.argv[1:]:
        cells = load(folder)
        alts = sorted({a for _, _, a in cells})
        for m in METHODS:
            print(f"\n## {folder}, UniDepth V2 {m}: mean position error of the held-out person (m)")
            print("| known points | " + " | ".join(f"{a} m" for a in alts) + " | all |")
            print("|---|" + "---|" * (len(alts) + 1))
            for case in CASES:
                per = [cells[m, case, a] for a in alts]
                print(f"| {case} | " + " | ".join(f"{statistics.fmean(v):.2f}" if v else "-" for v in per)
                      + f" | {statistics.fmean(e for v in per for e in v):.2f} |")


if __name__ == "__main__":
    main()
