"""Figures from outputs/people.csv: gps_map.png (every model's estimate vs truth), depth_vs_altitude.png
(axial depth and horizontal distance per model), photos_montage.png and photos_detail.png (matched boxes).
Usage: python Skydio_Flight_Experiment/build_charts.py
"""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference_validation"))
from compare_to_ground_truth import local_enu  # noqa: E402

OUT = HERE / "outputs"
MODELS = {"unidepth": "A1 UniDepth V2", "lingbot": "A2 LingBot", "depth_pro": "A3 Depth Pro", "yolo_original": "A4 YOLO26-depth", "yolo_finetuned": "C Fine-tuned YOLO26x"}
COLOR = dict(zip(list(MODELS) + ["ground_plane"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # dataviz categorical slots 1-6
ALT_RAMP = {10: "#9ec5f4", 20: "#6da7ec", 30: "#3987e5", 40: "#256abf", 50: "#104281"}  # sequential blue
MARKER = {"1": "o", "2": "s", "3": "^"}
TEXT, MUTED = "#0b0b0b", "#52514e"

rows = list(csv.DictReader((OUT / "people.csv").open()))
truth = {r["person"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader((OUT / "truth.csv").open())}
origin = truth["2"]
enu = lambda lat, lon: local_enu(*origin, lat, lon)[::-1]  # (east, north) from person 2


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=15)
    ax.grid(True, color="#e6e5e1", linewidth=.6)
    ax.set_axisbelow(True)


def gps_map():
    panels = [(m, f"{n} depth ray") for m, n in MODELS.items()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), sharex=True, sharey=True)
    axes.flat[-1].axis("off")  # five models, six slots
    fig.patch.set_facecolor("#fcfcfb")
    drones = {}
    for r in rows:
        pose = json.loads((OUT / (Path(r["image"]).stem + "_unidepth.json")).read_text())["pose"]
        drones[r["image"]] = enu(pose["lat"], pose["lon"])
    for ax, (key, title) in zip(axes.flat, panels):
        style(ax); ax.set_facecolor("#fcfcfb"); ax.set_aspect("equal")
        outside = 0
        for r in rows:
            if (r["model"] != "unidepth" if key == "ground_plane" else r["model"] != key):
                continue
            method = "ground_plane" if key == "ground_plane" else "depth_ray"
            if not r.get(f"{method}_lat"):
                continue
            e, n = enu(float(r[f"{method}_lat"]), float(r[f"{method}_lon"]))
            if abs(e) > 14 or abs(n) > 14:
                outside += 1
                continue
            ax.scatter(e, n, s=34, marker=MARKER[r["person"]], color=ALT_RAMP[int(r["altitude_label_m"])], edgecolor="#fcfcfb", linewidth=.8, zorder=3)
        for e, n in drones.values():
            ax.scatter(e, n, s=28, marker="x", color=MUTED, linewidth=1, zorder=2)
        for p, (lat, lon) in truth.items():
            e, n = enu(lat, lon)
            ax.scatter(e, n, s=150, marker="*", color=TEXT, zorder=4)
            ax.annotate(f"P{p}", (e, n), xytext=(6, 4), textcoords="offset points", fontsize=16, color=TEXT, weight="bold")
        ax.set_title(title + (f"  ({outside} points off-chart)" if outside else ""), fontsize=18, color=TEXT, loc="left")
        ax.set_xlim(-14, 14); ax.set_ylim(-14, 14)
    for ax in axes[1][:2]:
        ax.set_xlabel("east of person 2 reference (m)", fontsize=17, color=MUTED)
    axes[0][2].tick_params(labelbottom=True)
    axes[0][2].set_xlabel("east of person 2 reference (m)", fontsize=17, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("north (m)", fontsize=17, color=MUTED)
    handles = [plt.Line2D([], [], marker="*", color=TEXT, linestyle="", markersize=11, label="reference (nadir shot)"),
               plt.Line2D([], [], marker="x", color=MUTED, linestyle="", label="aircraft position")]
    handles += [plt.Line2D([], [], marker=MARKER[p], color=MUTED, linestyle="", label=f"person {p}") for p in "123"]
    handles += [plt.Line2D([], [], marker="o", color=c, linestyle="", label=f"{a} m altitude") for a, c in ALT_RAMP.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=15, frameon=False, bbox_to_anchor=(.5, -.005))
    fig.suptitle("Estimated person positions by model, all 12 flight photos", fontsize=20, color=TEXT, x=.01, ha="left")
    fig.tight_layout(rect=(0, .08, 1, .96), h_pad=4)
    fig.savefig(OUT / "gps_map.png", dpi=160)


def depth_vs_altitude():
    # mean over the three people and the photos at each altitude
    depth, horiz, truth_slant, truth_h, plane = (defaultdict(list) for _ in range(5))
    for r in rows:
        a = int(r["altitude_label_m"])
        h, th = float(r["height_m"]), float(r["truth_horizontal_m"])
        if r["depth_m"]:
            depth[r["model"], a].append(float(r["depth_m"]))
        if r.get("depth_ray_horizontal_m"):
            horiz[r["model"], a].append(float(r["depth_ray_horizontal_m"]))
        if r["model"] == "unidepth":
            truth_slant[a].append(math.hypot(h, th)); truth_h[a].append(th); plane[a].append(float(r["ground_plane_horizontal_m"]))
    alts = sorted(truth_h)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, data, ref, ylabel, title in ((ax1, depth, truth_slant, "axial depth at the torso (m)", "Depth read by each model vs camera-to-person slant range"),
                                         (ax2, horiz, truth_h, "horizontal distance drone to person (m)", "Horizontal distance from the depth ray vs truth")):
        style(ax); ax.set_facecolor("#fcfcfb")
        ax.plot(alts, [statistics.fmean(ref[a]) for a in alts], color=TEXT, linestyle="--", linewidth=1.6, label="truth", zorder=4)
        ax.annotate("truth", (alts[-1], statistics.fmean(ref[alts[-1]])), xytext=(5, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
        series = list(MODELS.items()) + ([("ground_plane", "Ground plane")] if ax is ax2 else [])
        for m, name in series:
            y = [statistics.fmean(plane[a]) for a in alts] if m == "ground_plane" else [statistics.fmean(data[m, a]) if data[m, a] else math.nan for a in alts]
            ax.plot(alts, y, color=COLOR[m], linewidth=2, marker="o", markersize=5, label=name)
            if ax is ax1:  # the right panel's lines end within a metre of each other; its legend carries identity
                ax.annotate(name, (alts[-1], y[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
        ax.set_xlabel("flight altitude above launch (m)", fontsize=9, color=MUTED); ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left"); ax.set_xticks(alts); ax.set_xlim(8, 62)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "depth_vs_altitude.png", dpi=160)


def montage():
    font = ImageFont.truetype("arial.ttf", 17) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    by_photo = defaultdict(dict)
    for r in rows:
        by_photo[r["image"]].setdefault(r["person"], {})[r["model"]] = r
    photos = ["10_m_Alt_Flight.JPG", "20_m_Alt_Flight.JPG", "30_m_Alt_Flight.JPG", "40_m_Alt_Flight.JPG", "50_m_Alt_Flight.JPG", "Person_1.JPG", "Person_2.JPG", "Person_3.JPG"]

    def render(im, boxes, title, crop):
        """Resize the crop to 1024x768, then draw boxes and labels at display size."""
        x0, y0, x1, y1 = crop
        sx, sy = 1024 / (x1 - x0), 768 / (y1 - y0)
        out = im.crop(crop).resize((1024, 768))
        draw = ImageDraw.Draw(out)
        for (bx1, by1, bx2, by2), label in boxes:
            bx1, bx2, by1, by2 = (bx1 - x0) * sx, (bx2 - x0) * sx, (by1 - y0) * sy, (by2 - y0) * sy
            draw.rectangle((bx1, by1, bx2, by2), outline="#eda100", width=3)
            draw.text((bx1, max(2, by1 - 22)), label, fill="#eda100", font=font, stroke_width=2, stroke_fill="black")
        draw.text((12, 30), title, fill="white", font=font, stroke_width=2, stroke_fill="black")
        return out

    full, detail = [], []
    for name in photos:
        im = Image.open(HERE / "Skydio" / name).convert("RGB")
        w, h = im.size
        boxes = []
        for p, models in by_photo.get(name, {}).items():
            r = models["unidepth"]
            slant = math.hypot(float(r["height_m"]), float(r["truth_horizontal_m"]))
            boxes.append((tuple(map(int, r["bbox"].split())), f"P{p}  UniDepth {float(r['depth_m']):.1f} m  (truth slant {slant:.1f} m)"))
        if name.startswith("Person_"):  # nadir shot: the head box under the drone is the truth
            j = json.loads((OUT / (Path(name).stem + "_unidepth.json")).read_text())
            under = min(j["people"], key=lambda q: math.hypot((q["bbox"][0] + q["bbox"][2]) / 2 - w / 2, (q["bbox"][1] + q["bbox"][3]) / 2 - h / 2))
            boxes.append((tuple(map(int, under["bbox"])), f"truth P{name[7]} (drone {j['pose']['height_above_ground_m']:.1f} m above)"))
        title = name.replace(".JPG", "").replace("_", " ")
        full.append(render(im, boxes, title, (0, 0, w, h)))
        xs, ys = [v for b, _ in boxes for v in b[0::2]], [v for b, _ in boxes for v in b[1::2]]
        cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
        half = max(1100, (max(xs) - min(xs)) // 2 + 500)
        x0, y0 = min(max(0, cx - half), w - 2 * half), min(max(0, cy - half * 3 // 4), h - half * 3 // 2)
        detail.append(render(im, boxes, title + " (detail)", (x0, y0, x0 + 2 * half, y0 + half * 3 // 2)))
    for frames, out in ((full, "photos_montage.png"), (detail, "photos_detail.png")):
        sheet = Image.new("RGB", (2048, 768 * ((len(frames) + 1) // 2)), "white")
        for i, f in enumerate(frames):
            sheet.paste(f, ((i % 2) * 1024, (i // 2) * 768))
        sheet.save(OUT / out)


if __name__ == "__main__":
    gps_map(); depth_vs_altitude()
    if (HERE / "Skydio").is_dir():  # the photographs are not distributed; the montages need them
        montage()
    print("wrote", [p.name for p in OUT.glob("*.png")])
