"""Regenerates figures/uavb_error.png: each method's mean distance to the person on the 1 November second-UAV flight.

Reads the summary that reference_validation/score_minise_flight.py --sheet 8 writes (12 photos, person at the takeoff point).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUN = "minise_20261101_flight4"
s = json.load(open(HERE.parent / "reference_validation/outputs" / RUN / "gt_comparison_summary.json", encoding="utf-8"))

rows = [("Fusion (UniDepth V2 + ground)", f"{RUN} / fusion", "#d95f02"),
        ("LingBot (UniDepth)", f"{RUN}_lingbot / depth_ray", "#1b9e77"),
        ("UniDepth V2", f"{RUN} / depth_ray", "#1b9e77"),
        ("Depth Pro", f"{RUN}_depth_pro / depth_ray", "0.6"),
        ("YOLO26-depth", f"{RUN}_yolo_original / depth_ray", "0.6"),
        ("Fine-tuned YOLO26x", f"{RUN}_yolo_finetuned / depth_ray", "#7570b3")][::-1]
names = [r[0] for r in rows]
vals = [s[r[1]]["mean_position_error_m"] for r in rows]
assert all(s[r[1]]["n"] == 12 for r in rows)

fig, ax = plt.subplots(figsize=(3.45, 2.2), layout="constrained")
ax.axvspan(0, 2, color="0.92", lw=0, zorder=0)
ax.barh(names, vals, color=[r[2] for r in rows], height=0.7)
for i, v in enumerate(vals):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=6)
ax.text(2.3, len(rows) - 0.55, "shaded: takeoff fix uncertainty, 1-2 m", fontsize=5, color="0.4", va="center")
ax.set_xlabel("Mean distance to the person, m (12 photos)", fontsize=6.5)
ax.set_xlim(0, 30)
ax.tick_params(labelsize=6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(HERE / "figures/uavb_error.png", dpi=300)
