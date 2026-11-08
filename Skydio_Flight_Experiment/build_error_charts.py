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
SERIES = [("unidepth", "depth_ray", "UniDepth V2", "#1b9e77", "o"),
          ("lingbot", "depth_ray", "LingBot", "#7570b3", "s"),
          ("yolo_original", "depth_ray", "YOLO26-depth", "#8c8c8c", "D"),
          ("depth_pro", "depth_ray", "Depth Pro", "#bdbdbd", "v"),
          ("yolo_finetuned", "depth_ray", "Fine-tuned YOLO26x", "#e7298a", "P"),
          ("unidepth", "fusion", "Fusion (UniDepth V2 + ground)", "#d95f02", "^"),
          ("unidepth", "ground_scaled", "Ground-scaled UniDepth V2", "#e6ab02", "x")]
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
            ax.annotate(f"{err:+.1f}", (d, y), xytext=(3.5, 0), textcoords="offset points", fontsize=5, color=color, ha="left", va="center")


def range_error_chart():
    alts = [10, 20, 30, 40, 50]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.45, 4.8), sharex=True, layout="constrained", height_ratios=[1, 1.1])
    for ax in (top, bot):
        ax.axhline(0, color="black", lw=0.8)
        ax.tick_params(labelsize=6.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    top_labels, bot_labels = {a: [] for a in alts}, {a: [] for a in alts}
    for model, method, label, color, marker in SERIES:
        err = signed(model, method)
        assert list(err) == alts, (model, method, list(err))
        y = [err[a] for a in alts]
        top.plot(alts, y, color=color, marker=marker, ms=3, lw=1, label=label)
        bot.plot(alts, y, color=color, marker=marker, ms=3, lw=1, label=label)
        for a in alts:
            if abs(err[a]) >= 3:  # the off-zoom methods are read in the top panel, the rest in the zoom
                top_labels[a].append([err[a], color])
            elif ZOOM_LO < err[a] < ZOOM_HI:
                bot_labels[a].append([err[a], color])
    annotate(top, top_labels, 2.4)
    annotate(bot, bot_labels, 0.22)
    top.set_ylim(-14, 37)
    top.set_ylabel("Estimate $-$ reference, m", fontsize=6.5)
    bot.set_ylim(ZOOM_LO, ZOOM_HI)
    bot.set_xlim(8, 56)
    bot.set_xticks(alts)
    bot.set_xlabel("Flight altitude above launch, m", fontsize=6.5)
    bot.set_ylabel("Estimate $-$ reference, m (zoom)", fontsize=6.5)
    top.legend(fontsize=5, frameon=False, loc="upper right", ncol=2)  # the top panel's upper right is empty
    for p in (OUT / "range_error_vs_altitude.png", FIG / "three_person_range_error.png"):
        fig.savefig(p, dpi=300)


def position_error_chart():
    bars = [("Fusion (UniDepth V2 + ground)", ("UniDepth V2", "fusion"), "#d95f02"),
            ("Ground-scaled UniDepth V2", ("UniDepth V2", "ground_scaled"), "#e6ab02"),
            ("LingBot (UniDepth)", ("LingBot (UniDepth-seeded)", "depth_ray"), "#1b9e77"),
            ("UniDepth V2", ("UniDepth V2", "depth_ray"), "#1b9e77"),
            ("Depth Pro", ("Depth Pro", "depth_ray"), "0.6"),
            ("YOLO26-depth", ("YOLO original", "depth_ray"), "0.6"),
            ("Fine-tuned YOLO26x", ("YOLO fine-tuned (AerialMetric)", "depth_ray"), "#7570b3")][::-1]
    names = [b[0] for b in bars]
    vals = [float(summary[b[1]]["all_pos_mean_m"]) for b in bars]
    assert all(summary[b[1]]["n"] == "36" for b in bars)
    fig, ax = plt.subplots(figsize=(3.45, 2.4), layout="constrained")
    ax.axvspan(0, 0.6, color="0.92", lw=0, zorder=0)
    ax.barh(names, vals, color=[b[2] for b in bars], height=0.7)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6)
    ax.text(0.8, len(bars) - 0.55, "shaded: nadir reference uncertainty, 0.6 m", fontsize=5, color="0.4", va="center")
    ax.set_xlabel("Mean distance to the person, m (36 detections)", fontsize=6.5)
    ax.set_xlim(0, 15)
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for p in (OUT / "mean_position_error.png", FIG / "three_person_error.png"):
        fig.savefig(p, dpi=300)
    return dict(zip(names, vals))


if __name__ == "__main__":
    range_error_chart()
    print(position_error_chart())
    print({f"{m}/{meth}": {a: round(v, 2) for a, v in signed(m, meth).items()} for m, meth, *_ in SERIES})
