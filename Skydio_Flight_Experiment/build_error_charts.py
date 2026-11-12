"""Two charts of the three-person flight test in the layout of the paper's lab and second-UAV figures.

range_error_vs_altitude.png: signed horizontal range error (estimate minus nadir reference), mean over the three
people and the photos at each altitude, in two panels: every method on one scale, then a zoom that holds the
metre-level methods. mean_position_error.png: each method's mean distance to the person over the 36 detections.
Reads outputs/people.csv and outputs/summary.csv (run_experiment.py); also copies both charts to Paper/figures/.
Usage: python Skydio_Flight_Experiment/build_error_charts.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

OUT = HERE / "outputs"
FIG = HERE.parent / "Paper/figures"
rows = list(csv.DictReader((OUT / "people.csv").open()))
summary = {(r["model"], r["method"]): r for r in csv.DictReader((OUT / "summary.csv").open())}

# (model in people.csv, method column prefix, label, colour, marker)
SERIES = [("unidepth", "depth_ray", "A1 UniDepth V2", "#1b9e77", "o"),
          ("lingbot", "depth_ray", "A2 LingBot", "#7570b3", "s"),
          ("yolo_original", "depth_ray", "A4 YOLO26-depth", "#8c8c8c", "D"),
          ("depth_pro", "depth_ray", "A3 Depth Pro", "#bdbdbd", "v"),
          ("yolo_finetuned", "depth_ray", "C Fine-tuned YOLO26x", "#e7298a", "P"),
          ("unidepth", "fusion", "D1 Fusion (A1 + ground)", "#d95f02", "^"),
          ("unidepth", "ground_scaled", "E1 Ground-scaled A1", "#e6ab02", "x"),
          ("unidepth", "scaled_fusion", "F1 Scaled fusion (E1 + ground)", "#a6761d", "*")]  # F1 is the pipeline's scaled_fusion
ZOOM_LO, ZOOM_HI = -2.75, 2.45


def signed(model, method):
    """Mean signed horizontal range error at each altitude label."""
    by_alt = defaultdict(list)
    for r in rows:
        if r["model"] == model and r.get(f"{method}_range_error_m"):
            by_alt[int(r["altitude_label_m"])].append(float(r[f"{method}_range_error_m"]))
    return {a: st.fmean(v) for a, v in sorted(by_alt.items())}


def annotate(ax, labels, gap):
    """Print each error beside its point; labels sharing an x stack apart by at least ``gap`` (data units)."""
    for d, group in labels.items():
        group.sort(key=lambda g: g[0])
        points = [g[0] for g in group]
        placed = []
        for i, (err, color) in enumerate(group):
            others = [p for j, p in enumerate(points) if j != i]
            y = err
            if i == 0:
                y = err - gap * 0.6
                while any(abs(y - p) < gap - 1e-6 for p in others):
                    y = min(p for p in others if abs(y - p) < gap - 1e-6) - gap
            else:
                y = max(y, placed[-1] + gap)
                while any(abs(y - p) < gap - 1e-6 for p in others):
                    y = max(p for p in others if abs(y - p) < gap - 1e-6) + gap
            placed.append(y)
            ax.annotate(f"{err:+.1f}", (d, y), xytext=(3.5, 0), textcoords="offset points", fontsize=6.5, color=color, ha="left", va="center")


def range_error_chart():
    """One panel, the zoomed range; methods that leave it are listed in a note instead of a second panel."""
    alts = [10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(3.45, 3.3), layout="constrained")
    ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    labels = {a: [] for a in alts}
    for model, method, label, color, marker in SERIES:
        err = signed(model, method)
        assert list(err) == alts, (model, method, list(err))
        y = [err[a] for a in alts]
        ax.plot(alts, y, color=color, marker=marker, ms=3, lw=1, label=label)
        for a in alts:
            if ZOOM_LO < err[a] < ZOOM_HI:
                labels[a].append([err[a], color])
    annotate(ax, labels, 0.3)
    ax.text(8.8, ZOOM_LO + 0.08, "off scale: A3, A4 $-$5 to $-$12 m" + chr(10) + "C +1.7 to +34.5 m", fontsize=6, color="0.4", va="bottom", clip_on=True)
    ax.set_ylim(ZOOM_LO, ZOOM_HI)
    ax.set_xlim(8, 56)
    ax.set_xticks(alts)
    ax.set_xlabel("Flight altitude above launch, m", fontsize=8.5)
    ax.set_ylabel("Estimate $-$ reference, m", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for p in (OUT / "range_error_vs_altitude.png", FIG / "three_person_range_error.png"):
        fig.savefig(p, dpi=300)


def position_error_chart():
    bars = [("F1 Scaled fusion (E1 + ground)", ("UniDepth V2", "scaled_fusion"), "#a6761d"),
            ("D1 Fusion (A1 + ground)", ("UniDepth V2", "fusion"), "#d95f02"),
            ("E1 Ground-scaled A1", ("UniDepth V2", "ground_scaled"), "#e6ab02"),
            ("A2 LingBot", ("LingBot (UniDepth-seeded)", "depth_ray"), "#1b9e77"),
            ("A1 UniDepth V2", ("UniDepth V2", "depth_ray"), "#1b9e77"),
            ("A3 Depth Pro", ("Depth Pro", "depth_ray"), "0.6"),
            ("A4 YOLO26-depth", ("YOLO original", "depth_ray"), "0.6"),
            ("C Fine-tuned YOLO26x", ("YOLO fine-tuned (AerialMetric)", "depth_ray"), "#7570b3")][::-1]
    names = [b[0] for b in bars]
    vals = [float(summary[b[1]]["all_pos_mean_m"]) for b in bars]
    assert all(summary[b[1]]["n"] == "36" for b in bars)
    fig, ax = plt.subplots(figsize=(3.9, 2.7), layout="constrained")
    ax.axvspan(0, 0.6, color="0.92", lw=0, zorder=0)
    ax.barh(names, vals, color=[b[2] for b in bars], height=0.7)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=7.5)
    ax.text(0.8, len(bars) - 0.55, "shaded: reference uncertainty, 0.6 m", fontsize=6.5, color="0.4", va="center")
    ax.set_xlabel("Mean distance to the person, m", fontsize=8.5)
    ax.set_xlim(0, 15)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for p in (OUT / "mean_position_error.png", FIG / "three_person_error.png"):
        fig.savefig(p, dpi=300)
    return dict(zip(names, vals))


if __name__ == "__main__":
    range_error_chart()
    print(position_error_chart())
    print({f"{m}/{meth}": {a: round(v, 2) for a, v in signed(m, meth).items()} for m, meth, *_ in SERIES})
