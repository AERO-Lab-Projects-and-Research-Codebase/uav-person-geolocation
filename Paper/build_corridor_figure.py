"""Regenerates figures/corridor_error.png: signed range error vs measured distance, one panel within +-6.5 m.

Reads geolocation_experiment_image/depth_results/results.csv. Top panel: every depth model and the
fusion on the full scale; the aerial fine-tune is off scale (+22 to +78 m) and is noted instead of
plotted. Bottom panel: the same within +-6.5 m, which holds every model to 30 m, so the
metre-level differences are readable; YOLO26-depth leaves the frame beyond 30 m.
Fusion is the live rule applied to UniDepth V2 and the plane, with the pitch uncertainty the offline
scorer gives that pose. Labels sharing an x are pushed apart vertically.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from score_fusion_variants import IMAGE_POSE_PITCH_SIGMA_DEG, current  # noqa: E402

rows = list(csv.DictReader((HERE.parent / "geolocation_experiment_image/depth_results/results.csv").open()))
dist = [float(r["actual_m"]) for r in rows]
for r in rows:
    r["fusion_m"] = current((float(r["unidepth_m"]), 0.0), (float(r["ground_plane_image_m"]), 0.0),
                            float(r["image_depression_deg"]), IMAGE_POSE_PITCH_SIGMA_DEG)[0]
    r["f1_m"] = current((float(r["ground_scaled_m"]), 0.0), (float(r["ground_plane_image_m"]), 0.0),
                        float(r["image_depression_deg"]), IMAGE_POSE_PITCH_SIGMA_DEG)[0]  # F1: the rule fed with the scaled ray

MODELS = [  # column, label, colour, marker
    ("unidepth", "A1 UniDepth V2", "#1b9e77", "o"),
    ("lingbot", "A2 LingBot", "#7570b3", "s"),
    ("yolo_original", "A4 YOLO26-depth", "#8c8c8c", "D"),
    ("fusion", "D1 Fusion (A1 + ground)", "#d95f02", "^"),
    ("ground_scaled", "E1 Ground-scaled A1", "#e6ab02", "x"),
    ("f1", "F1 Scaled fusion (E1 + ground)", "#a6761d", "*"),
]
ZOOM = 6.5


def errors(col):
    return [float(r[col + "_m"]) - d for r, d in zip(rows, dist)]


def annotate(ax, labels, gap):
    """Print each error beside its point; labels sharing an x stack apart by at least ``gap`` (data units)."""
    for d, group in labels.items():
        group.sort(key=lambda g: g[0])
        points = [g[0] for g in group]
        placed = []
        for i, (err, color) in enumerate(group):
            others = [p for j, p in enumerate(points) if j != i]
            y = err
            if i == 0:  # lowest series: label sits below its point, clear of the markers above
                y = err - gap * 0.6
                while any(abs(y - p) < gap - 1e-6 for p in others):
                    y = min(p for p in others if abs(y - p) < gap - 1e-6) - gap
            else:  # the rest stack upward, clear of markers and of labels already placed
                y = max(y, placed[-1] + gap)
                while any(abs(y - p) < gap - 1e-6 for p in others):
                    y = max(p for p in others if abs(y - p) < gap - 1e-6) + gap
            placed.append(y)
            ax.annotate(f"{err:+.1f}", (d, y), xytext=(3.5, 0), textcoords="offset points",
                        fontsize=6.5, color=color, ha="left", va="center")


fig, ax = plt.subplots(figsize=(3.45, 3.3), layout="constrained")
ax.axhline(0, color="black", lw=0.8)
ax.axvspan(30, 56, color="0.93", lw=0, zorder=0)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# One panel within +-6.5 m, which holds every model to 30 m; the fine-tune and YOLO26-depth beyond 30 m are noted.
labels = {d: [] for d in dist}
for col, label, color, marker in MODELS:
    err = errors(col)
    ax.plot(dist, err, color=color, marker=marker, ms=3, lw=1, label=label, clip_on=True)
    for d, e in zip(dist, err):
        if abs(e) < ZOOM:
            labels[d].append([e, color])
annotate(ax, labels, 0.5)
ax.text(6, 3.4, "C Fine-tuned: +22 to +78 m (off scale)", fontsize=6.5, color="0.4", va="top")
ax.text(55.5, 3.4, "beyond 30 m", fontsize=6.5, color="0.4", ha="right", va="top")
ax.text(55.5, -ZOOM - 0.25, "A4 beyond 30 m: $-$19 to $-$36 m", fontsize=6.5, color="0.4", ha="right", va="bottom")
ax.set_ylim(-ZOOM - 0.3, 3.5)
ax.set_xlim(5, 56)
ax.set_xticks(dist)
ax.set_xlabel("Distance, m", fontsize=8.5)
ax.set_ylabel("Estimate $-$ true, m", fontsize=8.5)
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(HERE / "figures/corridor_error.png", dpi=300)
