"""Regenerates figures/detections.png: real detections drawn on the original photos.

(a) is the first 10 m photo of the three-person flight test, with the three boxes that the experiment paired with
the people (Skydio_Flight_Experiment/outputs/people.csv, run_experiment.py); (b) is the 37.5 m lab photo with
every detection from geolocation_experiment_image/depth_results/detections.json. No box is invented.
Run from anywhere; needs the local photos under Skydio_Flight_Experiment/Skydio and the lab images.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Image.MAX_IMAGE_PIXELS = None
YELLOW, GREEN, GREY = "#ffd400", "#39ff14", "#cfcfcf"
PHOTO = "10_m_Alt_Flight.JPG"


def boxes_three(image):
    rows = [r for r in csv.DictReader((ROOT / "Skydio_Flight_Experiment/outputs/people.csv").open()) if r["image"] == image and r["model"] == "unidepth"]
    assert len(rows) == 3, image
    return [(int(r["person"]), *(float(v) for v in r["bbox"].split())) for r in rows]


def draw(ax, img, crop, boxes, title, colors=None):
    x0, y0, x1, y1 = crop
    ax.imshow(img.crop(crop), extent=(x0, x1, y1, y0))
    for i, (idx, bx0, by0, bx1, by1) in enumerate(boxes):
        c = colors[i] if colors else YELLOW
        ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=c, lw=0.9))
        ax.text(bx0, by0 - (y1 - y0) * 0.012, str(idx), color=c, fontsize=6, ha="left", va="bottom", weight="bold",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    ax.set_title(title, fontsize=6.5, pad=2)


three = Image.open(ROOT / "Skydio_Flight_Experiment/Skydio" / PHOTO).convert("RGB")
corr = Image.open(ROOT / "geolocation_experiment_image/Images/Human_37.5m.JPG").convert("RGB")
det = json.loads((ROOT / "geolocation_experiment_image/depth_results/detections.json").read_text())["Human_37.5m.JPG"]
target = json.loads((ROOT / "geolocation_experiment_image/chosen.json").read_text())["Human_37.5m.JPG"]
corr_boxes = [(d["index"], *d["bbox"]) for d in det]
corr_colors = [GREEN if d["index"] == target else GREY for d in det]

crop_a = (300, 2700, 6500, 4900)  # all three people, 2.8:1
crop_b = (1810, 1033, 2370, 1367)  # 1.68:1
fig, axes = plt.subplots(1, 2, figsize=(5.2, 1.45), layout="constrained",
                         width_ratios=[(crop_a[2] - crop_a[0]) / (crop_a[3] - crop_a[1]), (crop_b[2] - crop_b[0]) / (crop_b[3] - crop_b[1])])
draw(axes[0], three, crop_a, boxes_three(PHOTO), "(a) Three-person flight, 10 m, cropped")
draw(axes[1], corr, crop_b, corr_boxes, "(b) Lab, 37.5 m, cropped", corr_colors)
fig.savefig(HERE / "figures/detections.png", dpi=300)
