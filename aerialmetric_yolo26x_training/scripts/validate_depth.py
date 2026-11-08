#!/usr/bin/env python3
"""Score depth in output meters, plus optional median-aligned shape diagnostics.

The raw metric pass never rescales predictions using test ground truth and does
not run calibration. --median-aligned adds the built-in alignment diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_depth import (
    BUNDLE_ROOT,
    bundle_path,
    check_split,
    environment_report,
    nonnegative_int,
    positive_int,
    read_data_yaml,
    runtime_data_yaml,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--model", required=True, help="pretrained depth checkpoint or trained best.pt")
    command.add_argument("--data", default=str(BUNDLE_ROOT / "dataset" / "data.yaml"))
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="0")
    command.add_argument("--imgsz", type=positive_int, default=640)
    command.add_argument("--batch", type=positive_int, default=2)
    command.add_argument("--workers", type=nonnegative_int, default=0 if os.name == "nt" else 2)
    command.add_argument("--max-depth", type=float, help="evaluation cap in meters; defaults to data YAML")
    command.add_argument("--name", default="evaluation")
    command.add_argument("--project", default=str(BUNDLE_ROOT / "runs" / "evaluation"))
    command.add_argument("--median-aligned", action="store_true", help="also report GT-aligned shape diagnostics")
    command.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    command.add_argument("--check-only", action="store_true", help="check inputs and device without inference")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.imgsz % 32:
        raise ValueError("--imgsz must be divisible by 32.")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a single directory name.")
    data_path = bundle_path(args.data)
    data = read_data_yaml(data_path, args.max_depth)
    split_check = check_split(data, args.split, require_masked_range=False)
    model_path = bundle_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {model_path}")
    env = environment_report(args.device)
    from ultralytics import YOLO
    from ultralytics.models.yolo.depth import DepthValidator

    class RawMetricDepthValidator(DepthValidator):
        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            self.metrics.align = "none"

    model = YOLO(str(model_path), task="depth")
    if model.task != "depth":
        raise ValueError(f"Checkpoint task is '{model.task}', expected depth.")
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "checkpoint": str(model_path),
        "data_source": str(data_path),
        "data_root": data["path"],
        "split": args.split,
        "split_check": split_check,
        "imgsz": args.imgsz,
        "max_depth_m": data["max_depth"],
        "raw_protocol": "No GT scale alignment. Uses checkpoint calibration as saved; does not fit on test.",
        "metric_aggregation": "Mean of per-image metrics. GT <=0.001m or >=max_depth excluded; predictions clamped to bounds.",
    }
    print(json.dumps(report, indent=2))
    if args.check_only:
        print("Validation readiness checks passed. No inference or calibration was run.")
        return 0
    runtime_yaml = runtime_data_yaml(data_path, data, "eval")
    options = {
        "data": str(runtime_yaml),
        "split": args.split,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "plots": args.plots,
        "project": str(bundle_path(args.project)),
        "exist_ok": False,
        "augment": False,
    }
    raw = model.val(validator=RawMetricDepthValidator, name=f"{args.name}_raw", **options)
    report["raw_metric"] = {key: float(value) for key, value in raw.results_dict.items()}
    report["images_scored"] = int(raw._count)
    if report["images_scored"] == 0:
        raise RuntimeError("No images had sufficient valid GT pixels inside the evaluation range.")
    output = Path(raw.save_dir)
    if args.median_aligned:
        aligned = model.val(validator=DepthValidator, name=f"{args.name}_median_aligned", **options)
        report["median_aligned_diagnostic"] = {key: float(value) for key, value in aligned.results_dict.items()}
        report["aligned_protocol"] = "Each prediction rescaled with median(GT)/median(prediction); this is not metric ranging accuracy."
    destination = output / "depth_evaluation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evaluation saved to {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
