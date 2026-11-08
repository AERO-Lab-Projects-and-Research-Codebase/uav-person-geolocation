"""Top-down view of one three-person flight photo, in the style of the paper's field-photo figure: UAV at the origin,
the frame's footprint on the launch plane, range rings, and for each person the nadir-shot reference, the UniDepth V2 depth ray, the same
ray after the photo's ground scale (Method F), each labelled with its horizontal range and
its distance from the reference.

Usage: python Skydio_Flight_Experiment/build_topdown.py [30_m_Alt_Flight.JPG]
Writes outputs/topdown_<stem>.png and Paper/figures/three_person_topdown.png.
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from compare_to_ground_truth import local_enu  # noqa: E402

OUT = HERE / "outputs"
PHOTO = sys.argv[1] if len(sys.argv) > 1 else "30_m_Alt_Flight.JPG"
TOL_M = 0.6  # nadir hover offset from the head, truth.csv

pose = json.loads((OUT / (Path(PHOTO).stem + "_unidepth.json")).read_text())["pose"]
enu = lambda lat, lon: local_enu(pose["lat"], pose["lon"], lat, lon)  # (north, east) from the UAV  # noqa: E731
real = {r["person"]: enu(float(r["lat"]), float(r["lon"])) for r in csv.DictReader((OUT / "truth.csv").open())}
rows = {r["person"]: r for r in csv.DictReader((OUT / "people.csv").open()) if r["image"] == PHOTO and r["model"] == "unidepth"}
assert set(rows) == set(real) == {"1", "2", "3"}, set(rows)
est = {m: {p: enu(float(r[f"{m}_lat"]), float(r[f"{m}_lon"])) for p, r in rows.items()} for m in ("depth_ray", "ground_scaled")}


def dist(p, i):
    return math.hypot(p[0] - real[i][0], p[1] - real[i][1])


fig, ax = plt.subplots(figsize=(3.45, 3.8), layout="constrained")
ax.set_aspect("equal")
for r, ang in ((5, -80), (10, -80), (15, None)):  # ring labels on the empty south side, inside the axes
    ax.add_patch(Circle((0, 0), r, fill=False, ls=":", lw=0.8, color="0.55"))
    ang is None or ax.text(r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), f"{r} m", fontsize=5, color="0.45",
            ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.6))


def footprint(pose, aspect=6144 / 8192):
    """(east, north) of the four frame corners on the launch plane: pinhole from hfov, pitch and heading.
    ponytail: ignores roll (0.01 deg here) and lens distortion; redo through calibrated_ray if a rolled photo is drawn."""
    ha = math.radians(pose["hfov_deg"] / 2)
    va = math.atan(aspect * math.tan(ha))
    p, hd = math.radians(-pose["pitch_deg"]), math.radians(pose["heading_deg"])
    pts = []
    for a, b in ((-ha, -va), (ha, -va), (ha, va), (-ha, va)):  # top-left, top-right, bottom-right, bottom-left
        rt, fwd, dn = math.tan(a), math.cos(p) - math.tan(b) * math.sin(p), math.sin(p) + math.tan(b) * math.cos(p)
        assert dn > 0, "frame corner above the horizon"
        t = pose["height_above_ground_m"] / dn
        pts.append((t * (fwd * math.sin(hd) + rt * math.cos(hd)), t * (fwd * math.cos(hd) - rt * math.sin(hd))))
    return pts


ax.add_patch(Polygon(footprint(pose), closed=True, fc="0.5", alpha=0.07, lw=0))
ax.add_patch(Polygon(footprint(pose), closed=True, fill=False, ec="0.45", ls="--", lw=0.9))
ax.plot(0, 0, "k+", ms=7, mew=1.2)
ax.text(0.4, 0.4, "UAV", fontsize=6)
series = [("UniDepth V2", est["depth_ray"], "#7570b3", "D"), ("Ground-scaled UniDepth V2", est["ground_scaled"], "#d95f02", "^")]
# label centres sit about 2.5 m beside a marker (W/E) or 0.6 m above/below it (N/S) at this scale
centre = lambda x, y, dx, dy: (x + dx * 4.5, y + dy * 1.6)  # noqa: E731
# Every marker and label anchor in (east, north); each estimate label takes the side farthest from all of them.
anchors = [(e, n) for n, e in real.values()] + [(e, n) for _, s, _, _ in series for n, e in s.values()] + [(0.0, 0.0)]
SIDES = {"W": (-0.7, 0, "right", "center"), "E": (0.7, 0, "left", "center"), "N": (0, 0.7, "center", "bottom"), "S": (0, -0.7, "center", "top")}
for i, (n, e) in real.items():
    ax.add_patch(Circle((e, n), TOL_M, color="0.1", alpha=0.12, lw=0))
    ax.plot(e, n, "*", color="0.1", ms=8)
    ax.text(e - 0.6, n, f"P{i} real", fontsize=5.5, color="0.1", va="center", ha="right")
    anchors.append((e - 2.5, n))
    for label, s, color, marker in series:
        sn, se = s[i]
        ax.plot([se, e], [sn, n], color=color, lw=0.6, alpha=0.7)
        ax.plot(se, sn, marker=marker, color=color, ms=4.5, ls="none", mfc="none" if marker == "o" else color, mew=1.0 if marker == "o" else 0)  # open plane circle: the scaled ray often sits under it
        dx, dy, ha, va = max(SIDES.values(), key=lambda d: min(math.hypot(centre(se, sn, *d[:2])[0] - ax_, centre(se, sn, *d[:2])[1] - ay) for ax_, ay in anchors if (ax_, ay) != (se, sn)))
        ax.text(se + dx, sn + dy, f"{math.hypot(sn, se):.1f} m, $\\Delta${dist(s[i], i):.1f}", fontsize=5, color=color, ha=ha, va=va)
        anchors.append(centre(se, sn, dx, dy))
for label, s, color, marker in series:
    mean = sum(dist(s[i], i) for i in s) / len(s)
    ax.plot([], [], marker=marker, color=color, ls="none", mfc="none" if marker == "o" else color, label=f"{label}, mean error {mean:.1f} m")
ax.plot([], [], "*", color="0.1", ls="none", label=f"Reference (nadir shot), $\\pm${TOL_M} m")
ax.legend(fontsize=5.5, frameon=False, loc="lower left")
ax.set_xlim(-16.5, 3.5)
ax.set_ylim(-11, 11)
ax.set_xlabel("East offset from UAV, m", fontsize=6.5)
ax.set_ylabel("North offset, m", fontsize=6.5)
ax.tick_params(labelsize=6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(OUT / f"topdown_{Path(PHOTO).stem}.png", dpi=300)
fig.savefig(HERE.parent / "Paper/figures/three_person_topdown.png", dpi=300)
print(PHOTO, {m: {i: round(dist(s[i], i), 2) for i in s} for m, s in est.items()}, "heading", pose["heading_deg"], "pitch", pose["pitch_deg"], "hfov", round(pose["hfov_deg"], 1), "height", pose["height_above_ground_m"])
