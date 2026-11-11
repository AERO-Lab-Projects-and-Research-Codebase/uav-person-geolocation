"""Regenerates figures/uavb_error.png: each method's mean distance to the person on the 1 November second-UAV flight.

Reads the summary that reference_validation/score_minise_flight.py --sheet 8 writes (12 photos, person at the takeoff point).
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUN = "minise_20261101_flight4"
s = json.load(open(HERE.parent / "reference_validation/outputs" / RUN / "gt_comparison_summary.json", encoding="utf-8"))
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from compare_to_ground_truth import local_enu  # noqa: E402

HOME = (33.936054, -84.526896)  # takeoff fix in the flight log, the reference


def scaled_fusion():
    """F1: the fusion rule fed with the ground-scaled ray; every ray is steep, so it averages the E1 and ground positions."""
    rows = list(csv.DictReader(open(HERE.parent / "reference_validation/outputs" / f"{RUN}_ground_scaled" / "people.csv", encoding="utf-8")))
    assert len(rows) == 12
    d = [math.hypot(*local_enu(*HOME, (float(r["ground_scaled_lat"]) + float(r["ground_plane_lat"])) / 2,
                              (float(r["ground_scaled_lon"]) + float(r["ground_plane_lon"])) / 2)) for r in rows]
    return sum(d) / len(d)



rows = [("F1 Scaled fusion (E1 + ground)", "F1", "#e6ab02"),
        ("D1 Fusion (A1 + ground)", f"{RUN} / fusion", "#d95f02"),
        ("A2 LingBot", f"{RUN}_lingbot / depth_ray", "#1b9e77"),
        ("A1 UniDepth V2", f"{RUN} / depth_ray", "#1b9e77"),
        ("A3 Depth Pro", f"{RUN}_depth_pro / depth_ray", "0.6"),
        ("A4 YOLO26-depth", f"{RUN}_yolo_original / depth_ray", "0.6"),
        ("C Fine-tuned YOLO26x", f"{RUN}_yolo_finetuned / depth_ray", "#7570b3")][::-1]
names = [r[0] for r in rows]
vals = [scaled_fusion() if r[1] == "F1" else s[r[1]]["mean_position_error_m"] for r in rows]
assert all(r[1] == "F1" or s[r[1]]["n"] == 12 for r in rows)

fig, ax = plt.subplots(figsize=(3.9, 2.5), layout="constrained")
ax.axvspan(0, 2, color="0.92", lw=0, zorder=0)
ax.barh(names, vals, color=[r[2] for r in rows], height=0.7)
for i, v in enumerate(vals):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7.5)
ax.text(2.3, len(rows) - 0.55, "shaded: takeoff fix, 1-2 m", fontsize=6.5, color="0.4", va="center")
ax.set_xlabel("Mean distance to the person, m", fontsize=8.5)
ax.set_xlim(0, 30)
ax.tick_params(labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(HERE / "figures/uavb_error.png", dpi=300)
