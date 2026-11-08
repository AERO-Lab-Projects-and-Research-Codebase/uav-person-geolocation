"""Draws figures/pipeline.png, the offline reference pipeline block diagram (Fig. overview).

Replaces the earlier Illustrae drawing, which showed a thermal branch, a flat-plane method box and the removed
field-set validation. Pure matplotlib; run from anywhere.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
STAGE_FC, STAGE_EC, METHOD_FC, METHOD_EC, INK, MUTED = "#edf2fa", "#8fa6cc", "#f7f3ea", "#c6b793", "#1b1b1b", "#4a4a4a"

STAGES = [("UAV photo", "JPEG with XMP metadata:\nGPS, pose and intrinsics"),
          ("YOLO26x detector", "person boxes at a 2048 px\nlong side, confidence $\\geq$ 0.4"),
          ("Camera geometry", "undo lens distortion, rotate the\npixel ray to north-east-down"),
          ("Range along the ray", "three methods, a fusion rule\nand a ground-scaling step"),
          ("Position per method", "range gives a north/east offset,\nthen latitude and longitude"),
          ("Validation", "distance to each person's\nreference position, CSV export")]
METHODS = [("A  Torso depth", "UniDepth V2, Depth Pro,\nLingBot, YOLO26-depth"),
           ("B  Terrain map", "box-bottom ray meets the\nUSGS 3DEP grid + GEOID18"),
           ("C  Fine-tuned depth", "YOLO26x depth fine-tuned\non AerialMetric"),
           ("D  Fusion", "depth ray and flat-ground intersection,\ngated by depression angle and agreement"),
           ("E  Ground-scaled depth", "depth map rescaled so that its\nground lies at the camera height")]


def box(ax, x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=6.4, weight="bold", color=INK)
    ax.text(x + w / 2, y + 1.8, body, ha="center", va="bottom", fontsize=4.8, color=MUTED, linespacing=1.25)


fig, ax = plt.subplots(figsize=(7.0, 2.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
W, GAP, Y, H = 15.4, 1.5, 22, 12.5
for i, (title, body) in enumerate(STAGES):
    x = 0.3 + i * (W + GAP)
    box(ax, x, Y, W, H, title, body, STAGE_FC, STAGE_EC)
    i and ax.annotate("", (x, Y + H / 2), (x - GAP, Y + H / 2), arrowprops=dict(arrowstyle="-|>", color=STAGE_EC, lw=0.8, shrinkA=0, shrinkB=0))
# the range stage opens into the five method boxes
xr = 0.3 + 3 * (W + GAP) + W / 2
ax.plot([xr, xr], [Y, 17.5], color=STAGE_EC, lw=0.8)
MW, MG, MY, MH = 18.9, 1.2, 1.0, 13.5
xs = [0.3 + i * (MW + MG) for i in range(5)]
ax.plot([xs[0] + MW / 2, xs[-1] + MW / 2], [17.5, 17.5], color=STAGE_EC, lw=0.8)
for x, (title, body) in zip(xs, METHODS):
    ax.annotate("", (x + MW / 2, MY + MH), (x + MW / 2, 17.5), arrowprops=dict(arrowstyle="-|>", color=STAGE_EC, lw=0.8, shrinkA=0, shrinkB=0))
    box(ax, x, MY, MW, MH, title, body, METHOD_FC, METHOD_EC)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(HERE / "figures/pipeline.png", dpi=300)
print("wrote figures/pipeline.png")
