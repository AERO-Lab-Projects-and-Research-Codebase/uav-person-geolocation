"""Stubbed Depth Pro adapter contract tests; no model or checkpoint required."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

try:
    from .depth_backends.depth_pro import DepthProBackend
except ImportError:
    from depth_backends.depth_pro import DepthProBackend


class FakeModel:
    def __init__(self, focal=123.0):
        self.focal = focal

    def infer(self, image, f_px=None):
        self.image = image
        self.f_px = f_px
        self.inference_mode = torch.is_inference_mode_enabled()
        h, w = image.shape[-2:]
        return {
            "depth": torch.full((h, w), 4.25),
            "focallength_px": torch.tensor(self.focal),
        }


class DepthProAdapterTests(unittest.TestCase):
    def loaded_backend(self, *, focal=123.0):
        backend = DepthProBackend(device="cpu", checkpoint_path="unused.pt")
        backend.model = FakeModel(focal=focal)
        backend.transform = lambda rgb: torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        backend._torch = torch
        backend.precision = "float32"
        backend._source_revision = "test-revision"
        backend._load_seconds = 0.1
        return backend

    def test_load_uses_local_checkpoint_config_and_eval_mode(self):
        @dataclass
        class Config:
            checkpoint_uri: str

        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = Mock(return_value=(model, Mock()))
        package = types.ModuleType("depth_pro")
        package.create_model_and_transforms = factory
        config_module = types.ModuleType("depth_pro.depth_pro")
        config_module.DEFAULT_MONODEPTH_CONFIG_DICT = Config(checkpoint_uri="upstream-default.pt")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "DepthPro"
            (source / "src" / "depth_pro").mkdir(parents=True)
            checkpoint = Path(temporary) / "depth_pro.pt"
            checkpoint.write_bytes(b"stub checkpoint")
            backend = DepthProBackend(device="cpu", checkpoint_path=checkpoint, source_dir=source)
            with patch.dict(sys.modules, {"depth_pro": package, "depth_pro.depth_pro": config_module}):
                self.assertIs(backend.load(), backend)
                backend.load()
            factory.assert_called_once()
            config = factory.call_args.kwargs["config"]
            self.assertEqual(config.checkpoint_uri, str(checkpoint.resolve()))
            self.assertEqual(factory.call_args.kwargs["device"], torch.device("cpu"))
            self.assertEqual(factory.call_args.kwargs["precision"], torch.float32)
            model.eval.assert_called_once_with()

    def test_bgr_conversion_provided_focal_and_axial_depth_shape(self):
        backend = self.loaded_backend()
        bgr = np.empty((3, 4, 3), dtype=np.uint8)
        bgr[:] = [10, 80, 255]
        intrinsics = np.array([[100, 0, 2], [0, 101, 1], [0, 0, 1]], dtype=np.float64)
        result = backend.predict(bgr, intrinsics)
        self.assertEqual(backend.model.image[:, 0, 0].tolist(), [255, 80, 10])
        self.assertEqual(backend.model.f_px.item(), 100.0)
        self.assertTrue(backend.model.inference_mode)
        self.assertEqual(result["depth"].shape, (3, 4))
        self.assertEqual(result["depth"].dtype, np.float32)
        np.testing.assert_array_equal(result["depth"], 4.25)
        self.assertEqual(result["metadata"]["intrinsics_source"], "provided")
        self.assertEqual(result["metadata"]["input_fx_px"], 100.0)
        self.assertEqual(result["metadata"]["depth_convention"], "axial_z")
        json.dumps(result["metadata"], allow_nan=False)

    def test_missing_intrinsics_uses_model_focal_estimate_and_preserves_singleton_axis(self):
        backend = self.loaded_backend(focal=222.5)
        result = backend.predict(np.zeros((1, 4, 3), dtype=np.uint8))
        self.assertIsNone(backend.model.f_px)
        self.assertEqual(result["depth"].shape, (1, 4))
        self.assertEqual(result["metadata"]["intrinsics_source"], "model_estimated_focal_length")
        self.assertEqual(result["metadata"]["estimated_focal_length_px"], 222.5)

    def test_rejects_invalid_input_before_loading(self):
        backend = DepthProBackend(device="cpu", checkpoint_path="unused.pt")
        with patch.object(backend, "load") as load:
            for frame in [np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2), dtype=np.uint8)]:
                with self.assertRaises(ValueError):
                    backend.predict(frame)
            for matrix in [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)]:
                with self.assertRaises(ValueError):
                    backend.predict(np.zeros((2, 2, 3), dtype=np.uint8), matrix)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
