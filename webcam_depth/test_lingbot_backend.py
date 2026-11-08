"""CPU checks for LingBot's eval-only masked-token compatibility adapter."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "LingBotDepth"))
SPEC = importlib.util.spec_from_file_location("lingbot_backend_under_test", Path(__file__).with_name("lingbot_backend.py"))
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
with patch.dict(os.environ, {"XFORMERS_DISABLED": "1"}):
    from mdm.model.dinov2_rgbd.layers.attention import MemEffAttention
    from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock


class MaskedTokenCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(371)
        self.block = NestedTensorBlock(dim=12, num_heads=3, qkv_bias=True,
                                       init_values=.2, drop_path=.2,
                                       attn_class=MemEffAttention).eval()
        self.tokens = [torch.randn(2, 3, 12), torch.randn(1, 5, 12)]

    def test_variable_token_lists_match_original_tensor_path_without_weight_changes(self):
        before = {name: value.clone() for name, value in self.block.state_dict().items()}
        with torch.inference_mode():
            expected = [Block.forward(self.block, value) for value in self.tokens]
            backend._enable_masked_depth_sdpa(self.block)
            actual = self.block(self.tokens)
            single = self.block(self.tokens[0])
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(single, expected[0], rtol=1e-6, atol=1e-6)
        for name, value in self.block.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_list_path_matches_dense_block_diagonal_attention_reference(self):
        """Reference reproduces upstream nesting's mask using dense PyTorch SDPA."""
        merged = torch.cat([value.reshape(1, -1, 12) for value in self.tokens], dim=1)
        allowed = torch.zeros((11, 11), dtype=torch.bool)
        for left, right in ((0, 3), (3, 6), (6, 11)):
            allowed[left:right, left:right] = True
        with torch.inference_mode():
            normed = self.block.norm1(merged)
            qkv = self.block.attn.qkv(normed).reshape(1, 11, 3, 3, 4).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            attention = functional.scaled_dot_product_attention(query, key, value, allowed)
            attention = attention.permute(0, 2, 1, 3).reshape(1, 11, 12)
            attention = self.block.attn.proj_drop(self.block.attn.proj(attention))
            reference = merged + self.block.ls1(attention)
            reference = reference + self.block.ls2(self.block.mlp(self.block.norm2(reference)))
            expected = [reference[:, :6].reshape(2, 3, 12), reference[:, 6:].reshape(1, 5, 12)]
            backend._enable_masked_depth_sdpa(self.block)
            actual = self.block(self.tokens)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)

    def test_other_examples_cannot_change_first_example(self):
        backend._enable_masked_depth_sdpa(self.block)
        changed = [self.tokens[0].clone(), torch.randn(1, 7, 12) * 30]
        changed[0][1] = torch.randn(3, 12) * 100
        with torch.inference_mode():
            original = self.block(self.tokens)
            perturbed = self.block(changed)
        torch.testing.assert_close(original[0][0], perturbed[0][0], rtol=0, atol=0)

    def test_training_mode_is_explicitly_rejected(self):
        backend._enable_masked_depth_sdpa(self.block)
        self.block.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation only"):
            self.block(self.tokens)


class InputAndOutputTests(unittest.TestCase):
    def test_intrinsics_are_normalized_by_original_image_axes_without_mutation(self):
        matrix = np.array([[300., 5., 200.], [0., 250., 100.], [0., 0., 1.]], dtype=np.float32)
        original = matrix.copy()
        normalized = backend.LingBotDepthBackend._normalized_intrinsics(matrix, (200, 400))
        np.testing.assert_allclose(normalized, [[.75, .0125, .5], [0., 1.25, .5], [0., 0., 1.]])
        np.testing.assert_array_equal(matrix, original)

    def test_invalid_intrinsics_are_rejected(self):
        for matrix in ([[1., 0.], [0., 1.]],
                       [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                       [[1., 0., 0.], [0., float("nan"), 0.], [0., 0., 1.]],
                       [[1., 0., 0.], [0., 1., 0.], [0., 1., 1.]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                backend.LingBotDepthBackend._normalized_intrinsics(matrix, (4, 6))

    def test_invalid_rgb_seed_shape_or_empty_seed_fails_before_loading(self):
        instance = backend.LingBotDepthBackend(device="cpu")
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        cases = [(frame.astype(np.float32), np.ones((4, 6))),
                 (frame[:, :, 0], np.ones((4, 6))),
                 (np.zeros((0, 6, 3), dtype=np.uint8), np.ones((0, 6))),
                 (frame, np.ones((6, 4))),
                 (frame, np.full((4, 6), np.nan)),
                 (frame, np.zeros((4, 6)))]
        with patch.object(instance, "load") as load:
            for image, seed in cases:
                with self.subTest(shape=image.shape, seed_shape=seed.shape), self.assertRaises(ValueError):
                    instance.predict(image, seed)
            load.assert_not_called()

    def test_local_loader_fails_clearly_for_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            instance = backend.LingBotDepthBackend(device="cpu", model_id=root / "missing.pt", source_dir=root)
            with self.assertRaisesRegex(FileNotFoundError, "local weights missing"):
                instance.load()
            instance.model_id.write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "source missing"):
                instance.load()

    def stubbed_instance(self, output):
        class Model:
            def __init__(self):
                self.offload_count = 0
                self.inputs = None

            def to(self, device):
                return self

            def cpu(self):
                self.offload_count += 1
                return self

            def infer(self, image, **kwargs):
                self.inputs = image, kwargs
                return {"depth": torch.from_numpy(output.copy())}

        instance = backend.LingBotDepthBackend(device="cpu", offload=True)
        instance.model = Model()
        instance._torch = torch
        return instance

    def test_rgb_units_seed_cleanup_and_output_abstention(self):
        output = np.full((1, 4, 6), 2., dtype=np.float32)
        output[0, 0, :3] = [np.nan, -1., 0.]
        instance = self.stubbed_instance(output)
        frame = np.full((4, 6, 3), [0, 127, 255], dtype=np.uint8)
        seed = np.full((4, 6), 3., dtype=np.float32)
        seed[0, :4] = [np.nan, np.inf, -2., 0.]
        result = instance.predict(frame, seed, intrinsics=[[6., 0., 3.], [0., 4., 2.], [0., 0., 1.]])
        rgb, options = instance.model.inputs
        np.testing.assert_allclose(rgb[0, :, 0, 0], [1., 127 / 255., 0.])
        np.testing.assert_array_equal(options["depth_in"][0, 0], [0., 0., 0., 0., 3., 3.])
        np.testing.assert_allclose(options["intrinsics"][0], [[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
        self.assertTrue(options["apply_mask"])
        self.assertFalse(options["use_fp16"])
        self.assertTrue(np.isnan(result["depth"][0, :3]).all())
        self.assertEqual(result["depth"].dtype, np.float32)
        self.assertFalse(result["metadata"]["independent_depth_estimate"])
        self.assertEqual(instance.model.offload_count, 1)

    def test_misaligned_output_raises_and_still_offloads(self):
        instance = self.stubbed_instance(np.ones((1, 6, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "output shape"):
            instance.predict(np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6)))
        self.assertEqual(instance.model.offload_count, 1)

    def test_resolution_level_range_type_is_enforced(self):
        for value in (True, .5, -1, 10):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.LingBotDepthBackend(resolution_level=value)


if __name__ == "__main__":
    unittest.main()
