"""Adapter for Apple's official Depth Pro metric monocular depth model.

The adapter accepts OpenCV BGR frames and returns an HxW float32 axial-depth
map in metres. It uses the official repository implementation and requires
the checkpoint to be present locally; it never downloads weights.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time


SOURCE_URL = "https://github.com/apple/ml-depth-pro"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "external" / "DepthPro"
DEFAULT_CHECKPOINT = ROOT / "models" / "depth_pro.pt"


class DepthProBackend:
    def __init__(self, device="cuda:0", checkpoint_path=None, precision="float16", **kwargs):
        self.device = str(device)
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT).resolve()
        self.source_dir = Path(kwargs.pop("source_dir", DEFAULT_SOURCE)).resolve()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DepthProBackend options: {unexpected}")
        self.requested_precision = self._precision_name(precision)
        self.precision = self.requested_precision
        self.model = None
        self.transform = None
        self._torch = None
        self._source_revision = None
        self._load_seconds = None

    @staticmethod
    def _precision_name(value):
        names = {"float16": "float16", "half": "float16", "float32": "float32"}
        if value not in names:
            raise ValueError("precision must be 'float16' or 'float32'")
        return names[value]

    def load(self):
        """Load the local official checkpoint once and return this backend."""
        if self.model is not None:
            return self
        started = time.perf_counter()
        if not (self.source_dir / "src" / "depth_pro").is_dir():
            raise FileNotFoundError(
                f"Official Depth Pro source missing at {self.source_dir}. "
                f"Clone {SOURCE_URL} there, or pass source_dir."
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found at {self.checkpoint_path}. "
                "Place the official depth_pro.pt checkpoint there; this adapter does not download weights."
            )
        source_path = str(self.source_dir / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        try:
            import torch
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        except ImportError as exc:
            raise RuntimeError(
                "Depth Pro inference dependencies are unavailable in this Python environment. "
                f"Use the isolated depth-comparison environment. Original import error: {exc}"
            ) from exc
        torch_device = torch.device(self.device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable in this Python environment")
        if torch_device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(f"Requested {self.device}, but MPS is unavailable in this Python environment")
        effective = "float32" if torch_device.type == "cpu" else self.requested_precision
        torch_precision = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[effective]
        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(self.checkpoint_path))
        model, transform = depth_pro.create_model_and_transforms(
            config=config, device=torch_device, precision=torch_precision
        )
        model = model.to(torch_device).eval()
        self._torch = torch
        self.model = model
        self.transform = transform
        self.precision = effective
        try:
            self._source_revision = subprocess.check_output(
                ["git", "-C", str(self.source_dir), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            self._source_revision = None
        self._synchronize()
        self._load_seconds = time.perf_counter() - started
        return self

    def _synchronize(self):
        if self._torch is not None and self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    @staticmethod
    def _depth_map(value, shape):
        import numpy as np
        array = value.detach().float().cpu().numpy()
        if array.shape == (1, 1, *shape):
            array = array[0, 0]
        elif array.shape == (1, *shape):
            array = array[0]
        # Official infer squeezes singleton image axes; restore only those axes.
        if array.shape == tuple(n for n in shape if n != 1):
            array = array.reshape(shape)
        if array.shape != shape:
            raise ValueError(f"Depth Pro depth shape {array.shape} does not match input {shape}")
        return np.ascontiguousarray(array.reshape(shape), dtype=np.float32)

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        try:
            number = float(value.detach().float().cpu().item())
        except AttributeError:
            number = float(value)
        return number if number == number and abs(number) != float("inf") else None

    def predict(self, frame_bgr, intrinsics=None):
        """Predict metric camera-axis Z in metres for a uint8 BGR image.

        ``intrinsics`` is an optional 3x3 pixel K at this frame's resolution.
        When present, its horizontal focal length is supplied to Depth Pro as
        ``f_px``; otherwise the model's FOV head estimates focal length.
        """
        import numpy as np
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be a numpy uint8 HxWx3 array in 0..255")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 1:
            raise ValueError("frame_bgr must have nonempty HxWx3 shape")
        matrix = None
        if intrinsics is not None:
            matrix = np.asarray(intrinsics, dtype=np.float32)
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                    or not np.allclose(matrix[2], [0, 0, 1])):
                raise ValueError("intrinsics must be a finite pixel K (3x3), positive fx/fy, last row [0,0,1]")
        self.load()
        torch = self._torch
        self._synchronize()
        started = time.perf_counter()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image_tensor = self.transform(rgb)
        f_px = None if matrix is None else torch.tensor(
            float(matrix[0, 0]), device=torch.device(self.device), dtype=torch.float32
        )
        with torch.inference_mode():
            outputs = self.model.infer(image_tensor, f_px=f_px)
        shape = frame_bgr.shape[:2]
        depth = self._depth_map(outputs["depth"], shape)
        estimated_focal_px = self._scalar(outputs.get("focallength_px"))
        self._synchronize()
        runtime_seconds = time.perf_counter() - started
        metadata = {
            "backend": "depth_pro",
            "model": "Apple Depth Pro",
            "source_url": SOURCE_URL,
            "source_dir": str(self.source_dir),
            "source_revision": self._source_revision,
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "requested_precision": self.requested_precision,
            "precision": self.precision,
            "depth_convention": "axial_z",
            "depth_units": "meters",
            "input_shape_hw": list(shape),
            "input_rgb_range": "uint8 0..255",
            "intrinsics_source": "provided" if matrix is not None else "model_estimated_focal_length",
            "input_fx_px": None if matrix is None else float(matrix[0, 0]),
            "estimated_focal_length_px": estimated_focal_px if matrix is None else None,
            "effective_focal_length_px": estimated_focal_px,
            "load_seconds": self._load_seconds,
            "runtime_seconds": runtime_seconds,
            "runtime_ms": runtime_seconds * 1000.0,
            "runtime_scope": "preprocess_transfer_inference_and_output_copy_excluding_load",
        }
        return {"depth": depth, "metadata": metadata}
