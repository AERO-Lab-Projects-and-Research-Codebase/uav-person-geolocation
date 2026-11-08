"""LingBot-Depth v0.5 refinement of a UniDepth estimate from an RGB webcam.

LingBot requires a registered metric depth input. An ordinary webcam has no
sensor depth, so this adapter refines the supplied UniDepth estimate; its
result is not an independent depth measurement. Runtime never downloads weights.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/Robbyant/lingbot-depth"
SOURCE_REVISION = "f3a237e434ae987bc38281476d6cfb5df3e4d739"
WEIGHTS_REPOSITORY = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
WEIGHTS_REVISION = "79204ed6b837f4fdd192cf563e59481fecfa0295"


def _enable_masked_depth_sdpa(model):
    """Preserve upstream depth-token masking without optional xformers nesting.

    Upstream creates one token tensor per input sample, then concatenates them
    using block-diagonal attention. In evaluation, running each tensor through
    the existing tensor Block.forward is equivalent and supports native SDPA.
    The webcam adapter always supplies a single sample. No weights are changed.
    """
    from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock

    def forward(block, tokens):
        if block.training:
            raise RuntimeError("LingBot native masked-depth adapter supports evaluation only")
        if isinstance(tokens, list):
            return [Block.forward(block, tensor) for tensor in tokens]
        return Block.forward(block, tokens)

    for module in model.modules():
        if isinstance(module, NestedTensorBlock):
            module.forward = MethodType(forward, module)


class LingBotDepthBackend:
    def __init__(self, device="cuda:0", model_id=None, source_dir=None,
                 offload=True, resolution_level=0):
        if (isinstance(resolution_level, bool) or not isinstance(resolution_level, int)
                or not 0 <= resolution_level <= 9):
            raise ValueError("resolution_level must be an integer from 0 to 9")
        self.device = str(device)
        self.model_id = Path(model_id or ROOT / "models/lingbot_depth/model.pt").resolve()
        self.source_dir = Path(source_dir or ROOT / "external/LingBotDepth").resolve()
        self.offload = bool(offload)
        self.resolution_level = resolution_level
        self.model = None
        self._torch = None
        self._load_seconds = None
        self._source_revision = None

    def load(self):
        """Load the explicitly local checkpoint on CPU and use native PyTorch SDPA."""
        if self.model is not None:
            return self
        if not self.model_id.is_file():
            raise FileNotFoundError(
                f"LingBot local weights missing at {self.model_id}. "
                "Run webcam_depth/setup_webcam_depth.ps1 first."
            )
        if not (self.source_dir / "mdm/model/v2.py").is_file():
            raise FileNotFoundError(
                f"LingBot source missing at {self.source_dir}. "
                "Run webcam_depth/setup_webcam_depth.ps1 first."
            )
        started = time.perf_counter()
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        # Upstream's pinned xformers wheels do not match this CUDA runtime.
        # Disable optional binary imports, then enable the official SDPA method.
        prior_xformers_setting = os.environ.get("XFORMERS_DISABLED")
        os.environ["XFORMERS_DISABLED"] = "1"
        try:
            import torch
            from mdm.model.v2 import MDMModel
        except ImportError as exc:
            raise RuntimeError(
                "LingBot inference imports failed. Use the depth-comparison Python "
                f"environment and run setup_webcam_depth.ps1. Original error: {exc}"
            ) from exc
        finally:
            if prior_xformers_setting is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = prior_xformers_setting
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable")
        # The official loader loads local files on CPU, using weights_only=True.
        model = MDMModel.from_pretrained(str(self.model_id)).cpu().eval()
        model.enable_pytorch_native_sdpa()
        _enable_masked_depth_sdpa(model)
        self.model = model
        self._torch = torch
        try:
            self._source_revision = subprocess.check_output(
                ["git", "-C", str(self.source_dir), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        self._load_seconds = time.perf_counter() - started
        return self

    def _synchronize(self):
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize(self.device)

    @staticmethod
    def _normalized_intrinsics(intrinsics, shape):
        import numpy as np
        if intrinsics is None:
            return None
        matrix = np.asarray(intrinsics, dtype=np.float32)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])):
            raise ValueError("intrinsics must be a finite pixel K with positive fx/fy")
        height, width = shape
        matrix = matrix.copy()
        matrix[0] /= width
        matrix[1] /= height
        return matrix

    def predict(self, frame_bgr, depth_seed, intrinsics=None):
        """Refine registered UniDepth axial-Z metres, returning full-size float32 depth.

        Pixels LingBot rejects remain NaN so person statistics can abstain.
        ``intrinsics`` must be a 3x3 pixel K for the original frame dimensions.
        """
        import numpy as np
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError("frame_bgr must be a nonempty numpy uint8 HxWx3 array")
        shape = frame_bgr.shape[:2]
        seed = np.asarray(depth_seed, dtype=np.float32)
        if seed.shape != shape:
            raise ValueError(f"depth_seed shape {seed.shape} must match frame {shape}")
        valid_seed = np.isfinite(seed) & (seed > 0)
        if not valid_seed.any():
            raise ValueError("depth_seed must contain at least one finite positive metre value")
        seed = np.where(valid_seed, seed, 0).astype(np.float32)
        normalized_k = self._normalized_intrinsics(intrinsics, shape)
        self.load()
        torch = self._torch
        self._synchronize()
        started = time.perf_counter()
        image_tensor = seed_tensor = camera_tensor = outputs = None
        try:
            self.model.to(self.device)
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1].transpose(2, 0, 1))
            image_tensor = torch.from_numpy(rgb).to(self.device, dtype=torch.float32)[None] / 255.0
            seed_tensor = torch.from_numpy(seed).to(self.device)[None]
            if normalized_k is not None:
                camera_tensor = torch.from_numpy(normalized_k).to(self.device)[None]
            with torch.inference_mode():
                outputs = self.model.infer(
                    image_tensor, depth_in=seed_tensor, intrinsics=camera_tensor,
                    resolution_level=self.resolution_level, apply_mask=True,
                    use_fp16=self.device.startswith("cuda"),
                )
            depth = outputs["depth"].detach().float().cpu().numpy()
            if depth.shape == (1, *shape):
                depth = depth[0]
            if depth.shape != shape:
                raise ValueError(f"LingBot output shape {depth.shape} must match frame {shape}")
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
            self._synchronize()
        finally:
            # Release all output/tensor GPU references before freeing cached memory.
            outputs = image_tensor = seed_tensor = camera_tensor = None
            if self.offload:
                self.model.cpu()
                if self.device.startswith("cuda"):
                    self._synchronize()
                    torch.cuda.empty_cache()
        runtime_seconds = time.perf_counter() - started
        return {
            "depth": depth,
            "metadata": {
                "backend": "lingbot_depth",
                "label": "LingBot (UniDepth refinement)",
                "model_id": str(self.model_id),
                "weights_repository": WEIGHTS_REPOSITORY,
                "weights_revision": WEIGHTS_REVISION,
                "source_url": SOURCE_URL,
                "source_dir": str(self.source_dir),
                "source_revision": self._source_revision,
                "device": self.device,
                "offload": self.offload,
                "resolution_level": self.resolution_level,
                "attention_implementation": "native_sdpa_with_per_sample_masked_tokens",
                "depth_units": "meters",
                "depth_convention": "axial_z",
                "depth_seed_source": "unidepth_v2_estimate",
                "independent_depth_estimate": False,
                "input_shape_hw": list(shape),
                "input_intrinsics": None if intrinsics is None else np.asarray(intrinsics).tolist(),
                "load_seconds": self._load_seconds,
                "runtime_seconds": runtime_seconds,
                "runtime_ms": runtime_seconds * 1000.0,
                "runtime_scope": "transfer_inference_output_copy_and_offload_excluding_load",
                "valid_depth_fraction": float(np.isfinite(depth).mean()),
            },
        }
