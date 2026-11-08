"""One error chart per model (signed error vs actual distance, own y-scale) from depth_results/results.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "depth_results"
MODELS = [("yolo_original", "YOLO original", "#8c8c8c"), ("yolo_finetuned", "YOLO fine-tuned (AerialMetric)", "#d95f02"),
          ("unidepth", "UniDepth V2", "#1b9e77"), ("lingbot", "LingBot (UniDepth-seeded)", "#7570b3"),
          ("depth_pro", "Depth Pro", "#e6ab02")]
rows = list(csv.DictReader((OUT / "results.csv").open()))
labels = [f'{float(r["actual_m"]):g} m' for r in rows]
for key, name, color in MODELS:
    err = [float(r[key + "_m"]) - float(r["actual_m"]) for r in rows]  # + = reads too far, - = reads too near
    mae = sum(map(abs, err)) / len(err)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, err, color=color)
    ax.bar_label(bars, fmt="%+.1f", fontsize=9, padding=3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("Actual distance (from file name)"); ax.set_ylabel("Error = estimate - actual (m)")
    ax.set_title(f"{name}: mean absolute error {mae:.1f} m")
    ax.margins(y=.15); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    fig.text(.01, .01, "Above 0: reads too far. Below 0: reads too near.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(OUT / f"error_{key}.png", dpi=200); plt.close(fig)
print("done")
