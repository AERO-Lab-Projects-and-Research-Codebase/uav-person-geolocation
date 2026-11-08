"""Isolated adapter for the official UniDepth V2 inference implementation.

``infer`` accepts RGB in 0..255 and returns camera-axis Z in ``depth``.
Its separate ``radius`` output is Euclidean range, and must not replace Z.
No model import or weight download occurs until ``load`` or ``predict``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


SOURCE_URL = "https://github.com/lpiccinelli-eth/UniDepth"
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "external" / "UniDepth"


class UniDepthV2Backend:
    def __init__(self, device="cuda:0", model_id="lpiccinelli/unidepth-v2-vitl14",
                 resolution_level=5, **kwargs):
        if isinstance(resolution_level, bool) or not isinstance(resolution_level, int) or not 0 <= resolution_level < 10:
            raise ValueError("resolution_level must be an integer from 0 to 9")
        self.device = str(device)
        self.model_id = str(model_id)
        self.resolution_level = resolution_level
        self.source_dir = Path(kwargs.pop("source_dir", DEFAULT_SOURCE)).resolve()
        self.hub_kwargs = kwargs
        self.model = None
        self._torch = None
        self._source_revision = None
        self._load_seconds = None

    def load(self):
        """Load pretrained weights once; return self for explicit initialization."""
        if self.model is not None:
            return self
        started = time.perf_counter()
        if not (self.source_dir / "unidepth" / "models").is_dir():
            raise FileNotFoundError(
                f"Official UniDepth source missing at {self.source_dir}. "
                f"Clone {SOURCE_URL}.git there, or pass source_dir."
            )
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        try:
            import torch
            from unidepth.models import UniDepthV2
        except ImportError as exc:
            raise RuntimeError(
                "UniDepth V2 inference dependencies are unavailable. Use the isolated "
                "depth environment with torch, torchvision, einops, huggingface-hub, "
                "timm, scipy, matplotlib, opencv-python, pillow and wandb. "
                "UniDepth V2 does not require the optional xformers/KNN training ops. "
                f"Original import error: {exc}"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable in this Python environment")
        model = UniDepthV2.from_pretrained(self.model_id, **self.hub_kwargs)
        model = model.to(self.device).eval()
        model.resolution_level = self.resolution_level
        self._torch = torch
        self.model = model
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
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    @staticmethod
    def _output_map(value, shape, name):
        import numpy as np
        result = value.detach().float().cpu().numpy()
        # Strip only batch/channel dimensions, preserving one-pixel image axes.
        if result.shape == (1, 1, *shape):
            result = result[0, 0]
        elif result.shape == (1, *shape):
            result = result[0]
        if result.shape != shape:
            raise ValueError(f"UniDepth {name} shape {result.shape} does not match input {shape}")
        return np.ascontiguousarray(result, dtype=np.float32)

    def predict(self, frame_bgr, intrinsics=None):
        """Return float32 HxW axial-Z meters and JSON-safe provenance metadata.

        ``intrinsics`` is a pinhole 3x3 pixel K at the *original* image size.
        UniDepth handles resizing and padding internally. When K is absent,
        UniDepth estimates its own camera intrinsics.
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
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1].transpose(2, 0, 1))
        image_tensor = torch.from_numpy(rgb).to(self.device)
        camera = None if matrix is None else torch.from_numpy(matrix.copy()).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model.infer(image_tensor, camera=camera, normalize=True)
        shape = frame_bgr.shape[:2]
        depth = self._output_map(outputs["depth"], shape, "depth")
        confidence = None
        if outputs.get("confidence") is not None:
            confidence = self._output_map(outputs["confidence"], shape, "confidence")
        self._synchronize()
        runtime_seconds = time.perf_counter() - started
        predicted_intrinsics = None
        if outputs.get("intrinsics") is not None:
            predicted_intrinsics = outputs["intrinsics"].detach().float().cpu().numpy().reshape(3, 3)
            predicted_intrinsics = predicted_intrinsics.tolist() if np.isfinite(predicted_intrinsics).all() else None
        result = {
            "depth": depth,
            "metadata": {
                "backend": "unidepth_v2",
                "model_id": self.model_id,
                "device": self.device,
                "depth_convention": "axial_z",
                "depth_units": "meters",
                "resolution_level": self.resolution_level,
                "input_shape_hw": list(shape),
                "input_rgb_range": "0..255",
                "intrinsics_source": "provided" if matrix is not None else "model_estimated",
                "input_intrinsics": None if matrix is None else matrix.tolist(),
                "predicted_intrinsics": predicted_intrinsics,
                "source_url": SOURCE_URL,
                "source_dir": str(self.source_dir),
                "source_revision": self._source_revision,
                "weights_revision_requested": self.hub_kwargs.get("revision"),
                "load_seconds": self._load_seconds,
                "runtime_seconds": runtime_seconds,
                "runtime_ms": runtime_seconds * 1000.0,
                "runtime_scope": "transfer_and_inference_and_output_copy_excluding_load",
            },
        }
        if confidence is not None:
            result["confidence"] = confidence
        return result
