"""Diagnose Python, PyTorch and NVIDIA CUDA without starting training.

Only the Python standard library is required to report missing dependencies.
--runtime-only skips Ultralytics and checkpoint checks during initial setup.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig

from python_compat import compatibility_errors

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSIONS = {"torch": "2.14.0", "torchvision": "0.29.0", "ultralytics": "8.4.146"}
CUDA_INDEX = "https://download.pytorch.org/whl/cu130"


def nvidia_report() -> dict:
    executable = shutil.which("nvidia-smi")
    if executable is None and os.name == "nt":
        for candidate in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ):
            if candidate.is_file():
                executable = str(candidate)
                break
    if executable is None:
        return {"available": False, "detail": "nvidia-smi was not found. Install/check the NVIDIA driver."}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "detail": str(error)}
    if result.returncode:
        return {"available": False, "detail": result.stderr.strip() or result.stdout.strip()}
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) == 4:
            gpus.append(dict(zip(("index", "name", "driver_version", "memory_mib"), (item.strip() for item in row))))
    return {"available": bool(gpus), "gpus": gpus, "detail": result.stdout.strip()}


def inspect_environment(device: str = "0", runtime_only: bool = False) -> dict:
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "runtime" if runtime_only else "runtime_and_checkpoint",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_bits": struct.calcsize("P") * 8,
        "python_implementation": platform.python_implementation(),
        "python_free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "platform": platform.platform(),
        "selected_device": device,
        "nvidia": nvidia_report(),
        "packages": {},
        "errors": [],
        "warnings": [],
        "training_started": False,
        "training_memory_verified": False,
    }
    errors, warnings = report["errors"], report["warnings"]
    errors.extend(compatibility_errors())
    expected = {key: value for key, value in EXPECTED_VERSIONS.items() if not runtime_only or key != "ultralytics"}
    for package, version in expected.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        report["packages"][package] = installed
        if installed is None:
            errors.append(f"{package} is not installed in this Python environment. Run the bundle setup script.")
        elif installed.split("+")[0] != version:
            errors.append(f"Expected {package}=={version}; found {installed}. Rerun the bundle setup script.")
    if device != "cpu":
        if not report["nvidia"]["available"]:
            warnings.append(report["nvidia"]["detail"])
        else:
            try:
                selected_index = device.removeprefix("cuda:")
                gpu = next((gpu for gpu in report["nvidia"]["gpus"] if gpu["index"] == selected_index), report["nvidia"]["gpus"][0])
                driver = gpu["driver_version"]
                if int(driver.split(".")[0]) < 580:
                    errors.append(f"NVIDIA driver {driver} is below the CUDA 13.x minimum (580). Update at https://www.nvidia.com/Download/index.aspx and reboot.")
            except (KeyError, ValueError):
                warnings.append("Could not parse NVIDIA driver version; the CUDA calculation below remains the runtime check.")
    if report["packages"].get("torch") is None:
        return finish_report(report)
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as error:
        errors.append(f"PyTorch could not import: {error}. Rerun setup; on Windows also check the Microsoft Visual C++ x64 runtime.")
        return finish_report(report)
    report["torch"] = str(torch.__version__)
    report["torch_cuda_build"] = torch.version.cuda
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = torch.cuda.device_count()
    except (OSError, RuntimeError) as error:
        errors.append(f"CUDA discovery failed: {error}. Check the NVIDIA driver and restart the PC after updating it.")
        return finish_report(report)
    if device != "cpu":
        if torch.version.cuda is None:
            errors.append(f"This is CPU-only PyTorch. Rerun setup to install the NVIDIA build from {CUDA_INDEX}.")
        elif torch.version.cuda != "13.0":
            errors.append(f"Expected the CUDA 13.0 PyTorch build; found {torch.version.cuda}. Rerun setup to replace this build.")
        if not report["cuda_available"]:
            errors.append("PyTorch cannot access CUDA. Check the NVIDIA driver/GPU visibility, then rerun setup. Installing nvcc alone does not fix this.")
            return finish_report(report)
        try:
            index = int(device.removeprefix("cuda:"))
            if not 0 <= index < report["cuda_device_count"]:
                raise ValueError(f"GPU index {index} does not exist")
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["gpu"] = {
                "index": index, "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_gib": round(total / 1024**3, 2), "free_gib": round(free / 1024**3, 2),
            }
            if total < 19 * 1024**3:
                warnings.append("This GPU has less memory than the intended 20 GB PC. Training memory fit still requires a separate smoke run; try batch=1 if needed.")
            gpu_device = f"cuda:{index}"
            with torch.no_grad():
                tensor = torch.ones((64, 64), device=gpu_device)
                product = tensor @ tensor
                torch.cuda.synchronize(index)
                if not torch.all(product == 64).item():
                    raise RuntimeError("CUDA matrix multiplication returned an incorrect result")
            report["cuda_calculation_passed"] = True
        except (ValueError, OSError, RuntimeError) as error:
            errors.append(f"CUDA calculation failed: {error}. Check the selected GPU and driver compatibility with the installed PyTorch wheel.")
    try:
        torchvision = importlib.import_module("torchvision")
        report["torchvision"] = str(torchvision.__version__)
        if device != "cpu" and report.get("cuda_calculation_passed"):
            with torch.no_grad():
                boxes = torch.tensor([[0., 0., 1., 1.]], device=gpu_device)
                scores = torch.tensor([1.], device=gpu_device)
                kept = torchvision.ops.nms(boxes, scores, 0.5)
                if kept.tolist() != [0]:
                    raise RuntimeError("Unexpected torchvision CUDA NMS result")
            report["torchvision_cuda_passed"] = True
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        errors.append(f"torchvision import/CUDA check failed: {error}. Rerun setup to install matching CUDA torch and torchvision wheels.")
    if not runtime_only and not errors:
        try:
            if "YOLO_CONFIG_DIR" not in os.environ:
                config_directory = BUNDLE_ROOT / "runtime" / "ultralytics"
                config_directory.mkdir(parents=True, exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
            ultralytics = importlib.import_module("ultralytics")
            checkpoint = BUNDLE_ROOT / "weights" / "yolo26x-depth.pt"
            with checkpoint.open("rb") as handle:
                report["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            provenance = BUNDLE_ROOT / "weights" / "provenance.json"
            if provenance.exists() and report["checkpoint_sha256"] != json.loads(provenance.read_text(encoding="utf-8"))["sha256"]:
                raise RuntimeError("Copied checkpoint checksum mismatch")
            model = ultralytics.YOLO(str(checkpoint), task="depth")
            report["checkpoint_task"] = model.task
            if model.task != "depth":
                raise RuntimeError(f"Expected depth checkpoint; found task {model.task}")
            report["parameters"] = sum(parameter.numel() for parameter in model.model.parameters())
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            errors.append(f"Checkpoint/software check failed: {error}")
    return finish_report(report)


def finish_report(report: dict) -> dict:
    report["passed"] = not report["errors"]
    report["gpu_runtime_ready"] = report["passed"] and report["selected_device"] != "cpu" and report.get("cuda_calculation_passed", False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="one NVIDIA GPU index, or cpu for software inspection")
    parser.add_argument("--runtime-only", action="store_true", help="check torch/torchvision/CUDA before installing application packages")
    parser.add_argument("--json-report", type=Path, help="save the diagnostic report to this path")
    args = parser.parse_args()
    report = inspect_environment(args.device, args.runtime_only)
    print(json.dumps(report, indent=2))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        print("Check failed. See the errors above. Run setup_windows.ps1 or bash setup_linux.sh to install this bundle's packages.", file=sys.stderr)
        return 2
    if args.device == "cpu":
        print("CPU software check passed; GPU readiness was not verified.")
    else:
        print("PyTorch/CUDA check passed." if args.runtime_only else "Environment/model check passed.")
    print("No training or model forward inference was performed. The target GPU still needs a separate training memory smoke run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
