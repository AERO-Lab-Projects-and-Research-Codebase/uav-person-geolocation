"""Adapter contract checks with a fake model; no network or weights required."""

import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

try:
    from .depth_backends.unidepth_v2 import UniDepthV2Backend
except ImportError:
    from depth_backends.unidepth_v2 import UniDepthV2Backend


class FakeModel:
    def infer(self, rgb, camera=None, normalize=True):
        self.rgb = rgb
        self.camera = camera
        self.normalize = normalize
        self.inference_mode = torch.is_inference_mode_enabled()
        h, w = rgb.shape[-2:]
        return {
            "depth": torch.full((1, 1, h, w), 3.0),
            "radius": torch.full((1, 1, h, w), 5.0),
            "confidence": torch.full((1, 1, h, w), 0.8),
            "intrinsics": torch.eye(3).unsqueeze(0),
        }


class UniDepthAdapterTests(unittest.TestCase):
    def backend(self):
        backend = UniDepthV2Backend(device="cpu")
        backend.model = FakeModel()
        backend._torch = torch
        return backend

    def test_load_sets_evaluation_device_resolution_and_loads_once(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = Mock()
        factory.from_pretrained.return_value = model
        module = types.ModuleType("unidepth.models")
        module.UniDepthV2 = factory
        backend = UniDepthV2Backend(device="cpu", resolution_level=2, local_files_only=True)
        with patch.dict(sys.modules, {"unidepth.models": module}):
            self.assertIs(backend.load(), backend)
            backend.load()
        factory.from_pretrained.assert_called_once_with("lpiccinelli/unidepth-v2-vitl14", local_files_only=True)
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertEqual(model.resolution_level, 2)

    def test_rgb_range_camera_and_axial_depth_contract(self):
        backend = self.backend()
        bgr = np.empty((3, 4, 3), dtype=np.uint8)
        bgr[:] = [10, 80, 255]
        k = np.array([[100, 0, 2], [0, 101, 1], [0, 0, 1]], dtype=np.float64)
        result = backend.predict(bgr, k)
        self.assertEqual(backend.model.rgb[:, 0, 0].tolist(), [255, 80, 10])
        self.assertEqual(backend.model.rgb.dtype, torch.uint8)
        self.assertTrue(backend.model.normalize)
        self.assertTrue(backend.model.inference_mode)
        np.testing.assert_array_equal(backend.model.camera.numpy()[0], k)
        self.assertEqual(result["depth"].shape, (3, 4))
        self.assertEqual(result["depth"].dtype, np.float32)
        np.testing.assert_array_equal(result["depth"], 3.0)  # Not the radius=5 output.
        self.assertEqual(result["metadata"]["intrinsics_source"], "provided")
        self.assertEqual(result["confidence"].shape, (3, 4))
        json.dumps(result["metadata"], allow_nan=False)

    def test_estimated_camera_and_singleton_image_axes(self):
        backend = self.backend()
        result = backend.predict(np.zeros((1, 4, 3), np.uint8))
        self.assertIsNone(backend.model.camera)
        self.assertEqual(result["depth"].shape, (1, 4))
        self.assertEqual(result["metadata"]["intrinsics_source"], "model_estimated")

    def test_rejects_invalid_input_before_loading(self):
        backend = UniDepthV2Backend()
        with patch.object(backend, "load") as load:
            for frame in [np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.uint8)]:
                with self.assertRaises(ValueError):
                    backend.predict(frame)
            for k in [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)]:
                with self.assertRaises(ValueError):
                    backend.predict(np.zeros((2, 2, 3), np.uint8), k)
            load.assert_not_called()

    def test_rejects_wrong_model_output_size(self):
        backend = self.backend()
        with patch.object(backend.model, "infer", return_value={"depth": torch.ones(1, 1, 2, 2)}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                backend.predict(np.zeros((3, 4, 3), np.uint8))

    def test_resolution_bounds(self):
        for level in [-1, 10, 1.5, True]:
            with self.assertRaises(ValueError):
                UniDepthV2Backend(resolution_level=level)


if __name__ == "__main__":
    unittest.main()
