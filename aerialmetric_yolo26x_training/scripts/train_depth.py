#!/usr/bin/env python3
"""Fine-tune the copied YOLO26x depth checkpoint on prepared AerialMetric data.

No training starts with --check-only. All defaults are relative to this bundle,
and a temporary absolute-path data YAML is regenerated after copying machines.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from python_compat import compatibility_errors

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_VERSION = "8.4.146"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def bundle_path(value: str | Path) -> Path:
    """Resolve a supplied path, preferring an existing caller-relative path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (path if path.exists() else BUNDLE_ROOT / path).resolve()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return result


def read_data_yaml(data_path: Path, max_depth: float | None = None) -> dict[str, Any]:
    import yaml

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared data YAML is missing: {data_path}. Run scripts/prepare_dataset.py first."
        )
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    root_value = Path(str(data.get("path", "."))).expanduser()
    root = root_value if root_value.is_absolute() else data_path.parent / root_value
    data["path"] = str(root.resolve())
    data.setdefault("nc", 1)
    data.setdefault("names", {0: "depth"})
    data.setdefault("channels", 3)
    data.setdefault("depth_scale", 1000)
    scale = float(data["depth_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be finite and positive (PNG units per meter).")
    if int(data["channels"]) != 3:
        raise ValueError("This RGB AerialMetric recipe requires channels: 3.")
    if max_depth is not None:
        data["max_depth"] = max_depth
    cap = float(data.get("max_depth", 500.0))
    if not math.isfinite(cap) or not 0.001 < cap:
        raise ValueError(
            "max_depth must be finite and greater than 0.001 meters. "
            "Changing this cap affects validation and calibration, not the training loss mask."
        )
    data["max_depth"] = cap
    # An accidental inherited download command must not execute on the training computer.
    data.pop("download", None)
    return data


def split_image_paths(data: dict[str, Any], split: str) -> list[Path]:
    """Read an Ultralytics image directory/list and fail early on empty splits."""
    if not data.get(split):
        raise ValueError(f"Dataset YAML has no '{split}' split.")
    root = Path(data["path"])
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    paths: list[Path] = []
    for entry in entries:
        path = Path(str(entry))
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            paths.extend(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image = Path(line)
                if not image.is_absolute():
                    # Match Ultralytics' ./ list convention, while accepting normal relative paths.
                    image = path.parent / image
                paths.append(image.resolve())
        else:
            raise FileNotFoundError(f"Missing {split} image directory/list: {path}")
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"The '{split}' split contains no images.")
    return paths


def depth_path(image: Path) -> Path:
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "depth"
            break
    else:
        raise ValueError(f"Image path must contain an 'images' directory: {image}")
    png_path = Path(*parts).with_suffix(".png")
    return png_path if png_path.is_file() else png_path.with_suffix(".npy")


def check_split(data: dict[str, Any], split: str, *, require_masked_range: bool = True) -> dict[str, Any]:
    """Check every pair exists and inspect deterministic samples of depth contents."""
    import numpy as np
    from PIL import Image

    images = split_image_paths(data, split)
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Missing {split} image: {image}")
        paired = depth_path(image)
        if not paired.is_file():
            raise FileNotFoundError(f"Missing paired depth map: {paired}")
    sample_indices = sorted({0, len(images) // 2, len(images) - 1})
    sample_info = []
    for index in sample_indices:
        image = images[index]
        paired = depth_path(image)
        if paired.suffix == ".npy":
            depth = np.load(paired, allow_pickle=False, mmap_mode="r")
            if not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"NPY depth must be floating point meters: {paired}")
        else:
            with Image.open(paired) as source:
                raw_depth = np.asarray(source)
                if raw_depth.dtype != np.uint16 and source.mode not in {"I", "I;16"}:
                    raise ValueError(f"Depth PNG must have 16-bit integer values, not an 8-bit visualization: {paired}")
                depth = raw_depth.astype(np.float32) / float(data["depth_scale"])
        with Image.open(image) as source:
            expected_shape = (source.height, source.width)
        if depth.ndim != 2 or depth.shape != expected_shape:
            raise ValueError(f"Image/depth dimensions differ for {image}: {expected_shape} vs {depth.shape}")
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f"Depth has NaN/Inf/negative values; use zero for invalid pixels: {paired}")
        valid = depth[depth > 0.001]
        if valid.size < 10:
            raise ValueError(f"Fewer than 10 valid depth pixels: {paired}")
        if require_masked_range and float(valid.max()) >= float(data["max_depth"]):
            raise ValueError(
                f"Depth sample exceeds the prepared training range: {paired}. "
                "Mask values >= max_depth during preparation; the training loss does not apply this cap."
            )
        sample_info.append({"image": str(image), "valid_pixels": int(valid.size), "max_m": float(valid.max())})
    return {"images": len(images), "samples": sample_info}


def runtime_data_yaml(data_path: Path, data: dict[str, Any], tag: str) -> Path:
    import yaml

    directory = BUNDLE_ROOT / "runtime"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    destination = directory / f"{tag}_{data_path.stem}_{timestamp}_{os.getpid()}.yaml"
    destination.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return destination


def environment_report(device: str) -> dict[str, Any]:
    python_errors = compatibility_errors()
    if python_errors:
        raise RuntimeError(" ".join(python_errors))
    import torch
    installed = importlib.metadata.version("ultralytics")
    if installed != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected ultralytics=={ULTRALYTICS_VERSION}; found {installed}. Install requirements.txt.")
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics": installed,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cpu":
        return result
    try:
        index = int(device.removeprefix("cuda:"))
    except ValueError as exc:
        raise ValueError("Use one NVIDIA GPU index (e.g. --device 0), or --device cpu for inspection.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the NVIDIA-enabled PyTorch build described in README.md.")
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"GPU index {index} does not exist.")
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    result["gpu"] = {
        "index": index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_gib": round(total / 1024**3, 2),
        "free_gib": round(free / 1024**3, 2),
    }
    return result


def reset_fresh_calibration(trainer: Any) -> None:
    """Clear pretraining-domain inference scale on both model and EMA, once."""
    import torch

    if trainer.resume:
        return
    record: dict[str, Any] = {}
    candidates = {"model": trainer.model, "ema": trainer.ema.ema if trainer.ema is not None else None}
    with torch.no_grad():
        for label, model in candidates.items():
            if model is None:
                continue
            for module in model.modules():
                if hasattr(module, "cal_a") and hasattr(module, "cal_b"):
                    record[label] = {"before_a": module.cal_a.tolist(), "before_b": module.cal_b.tolist()}
                    module.cal_a.fill_(1.0)
                    module.cal_b.zero_()
    if set(record) != {"model", "ema"}:
        raise RuntimeError("Expected depth calibration buffers on both training model and EMA.")
    destination = Path(trainer.save_dir) / "fresh_calibration_reset.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data", default=str(BUNDLE_ROOT / "dataset" / "data.yaml"))
    command.add_argument("--model", default=str(BUNDLE_ROOT / "weights" / "yolo26x-depth.pt"))
    command.add_argument("--device", default="0", help="single NVIDIA GPU index; cpu allowed for checks")
    command.add_argument("--epochs", type=positive_int, default=30)
    command.add_argument("--imgsz", type=positive_int, default=640)
    command.add_argument("--batch", type=positive_int, default=2, help="reduce to 1 on GPU out-of-memory")
    command.add_argument("--workers", type=nonnegative_int, default=0 if os.name == "nt" else 2)
    command.add_argument("--lr0", type=float, default=1e-4)
    command.add_argument("--patience", type=nonnegative_int, default=10)
    command.add_argument("--seed", type=nonnegative_int, default=42)
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--max-depth", type=float, help="validation/calibration cap; does not mask training labels")
    command.add_argument("--fraction", type=float, default=1.0, help="fraction of TRAIN images; validation stays full")
    command.add_argument("--project", default=str(BUNDLE_ROOT / "runs" / "depth"))
    command.add_argument("--name", default="aerialmetric_yolo26x")
    command.add_argument("--resume", metavar="LAST_PT", help="resume an interrupted last.pt; original epoch target is retained")
    command.add_argument("--check-only", action="store_true", help="inspect model, data and device without training")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.imgsz % 32:
        raise ValueError("--imgsz must be divisible by 32.")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive.")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a single directory name.")
    data_path = bundle_path(args.data)
    data = read_data_yaml(data_path, args.max_depth)
    split_checks = {split: check_split(data, split) for split in ("train", "val")}
    model_path = bundle_path(args.resume or args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {model_path}. The pretrained checkpoint belongs in weights/.")
    env = environment_report(args.device)
    from ultralytics import YOLO
    from ultralytics.models.yolo.depth import DepthTrainer

    class PortableDepthTrainer(DepthTrainer):
        def check_resume(self, overrides: dict[str, Any]) -> None:
            super().check_resume(overrides)
            if self.resume:
                # Always use regenerated paths, even if the original copied-from path still exists.
                self.args.data = overrides["data"]
                self.args.amp = overrides["amp"]

    model = YOLO(str(model_path), task="depth")
    if model.task != "depth":
        raise ValueError(f"Checkpoint task is '{model.task}'; use YOLO26x-depth, not the detection checkpoint.")
    if args.resume and (
        not model.ckpt or model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None
    ):
        raise ValueError(
            "--resume requires an interrupted last.pt containing optimizer state. "
            "A pretrained or completed checkpoint starts a new fine-tuning run with --model instead."
        )
    project = bundle_path(args.project)
    target = project / args.name
    if not args.resume and target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Run already exists: {target}. Supply a new --name or use --resume path/to/last.pt.")
    report = {
        "environment": env,
        "checkpoint": str(model_path),
        "data_source": str(data_path),
        "data_root": data["path"],
        "max_depth_m": data["max_depth"],
        "split_checks": split_checks,
        "settings": vars(args),
        "validation_protocol": "Training metrics use per-image median alignment; use validate_depth.py for raw metric errors.",
        "calibration": "Fresh runs reset model/EMA scale, then Ultralytics fits final checkpoint scale on val only.",
        "depth_head": "Raw training output exp(clamp(logit, -4, 5)) has a 148.4m ceiling. Scale-invariant loss learns shape; final global val calibration can expand output meters beyond that ceiling.",
    }
    print(json.dumps(report, indent=2))
    if args.check_only:
        print("Readiness checks passed. No optimizer steps, training, or validation were run.")
        return 0
    runtime_yaml = runtime_data_yaml(data_path, data, "train")
    kwargs: dict[str, Any] = {
        "data": str(runtime_yaml),
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "cache": False,
        "plots": args.plots,
        "patience": args.patience,
        "amp": args.amp,
    }
    if args.resume:
        # Ultralytics restores optimizer/scheduler/epoch and retains original train_args.
        # It permits the memory/device overrides above; do not pretend CLI lr/epochs reset it.
        kwargs["resume"] = str(model_path)
        kwargs["save_dir"] = str(model_path.parent.parent if model_path.parent.name == "weights" else target)
        print("Resuming saved optimizer/scheduler and original epoch target; --epochs/--lr0 do not override resume.")
    else:
        kwargs.update(
            epochs=args.epochs,
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=0.1,
            warmup_bias_lr=args.lr0,
            warmup_epochs=1.0,
            weight_decay=0.0005,
            dlam=1.0,
            amp=args.amp,
            seed=args.seed,
            deterministic=True,
            fraction=args.fraction,
            project=str(project),
            name=args.name,
            exist_ok=False,
            save=True,
            save_period=5,
            compile=False,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.1,
        )
        model.add_callback("on_train_start", reset_fresh_calibration)
    # Save the preflight alongside the actual run, including an auto-incremented save_dir if any.
    def record_preflight(trainer: Any) -> None:
        (Path(trainer.save_dir) / "bundle_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    model.add_callback("on_train_start", record_preflight)
    model.train(trainer=PortableDepthTrainer, **kwargs)
    output = Path(model.trainer.save_dir)
    print(f"Training finished. Best checkpoint: {output / 'weights' / 'best.pt'}")
    print("Evaluate pretrained and trained checkpoints with validate_depth.py on the same held-out split.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
