"""Fresh-machine diagnostics, tested without importing ML packages or using a GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("bundle_environment_checker", SOURCE)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

NVIDIA = {
    "available": True,
    "gpus": [{"index": "0", "name": "Test NVIDIA GPU", "driver_version": "580.65", "memory_mib": "20480"}],
    "detail": "0, Test NVIDIA GPU, 580.65, 20480",
}


def fake_torch(cuda_build=None, available=False):
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = int(available)
    return SimpleNamespace(
        __version__="2.14.0+cu130" if cuda_build else "2.14.0+cpu",
        version=SimpleNamespace(cuda=cuda_build),
        cuda=cuda,
        ones=Mock(side_effect=AssertionError("This diagnostic test must not allocate GPU tensors")),
    )


class EnvironmentDiagnosticsTests(unittest.TestCase):
    def inspect(self, *, torch=None, versions=None, nvidia=None, importer=None, device="0", python_version=(3, 12, 8), python_bits=64, free_threaded=False):
        versions = {"torch": "2.14.0+cu130", "torchvision": "0.29.0+cu130"} if versions is None else versions

        def package_version(name):
            if versions.get(name) is None:
                raise checker.importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        if importer is None:
            def load_package(name):
                if name == "torch":
                    return torch
                if name == "torchvision":
                    return SimpleNamespace(__version__="0.29.0+cu130")
                raise AssertionError(f"Unexpected ML import in runtime diagnostic: {name}")
            importer = Mock(side_effect=load_package)

        with (
            patch.object(checker.sys, "version_info", python_version),
            patch.object(checker.struct, "calcsize", return_value=python_bits // 8),
            patch.object(checker.sysconfig, "get_config_var", return_value=int(free_threaded)),
            patch.object(checker, "nvidia_report", return_value=NVIDIA if nvidia is None else nvidia) as nvidia_mock,
            patch.object(checker.importlib.metadata, "version", side_effect=package_version),
            patch.object(checker.importlib, "import_module", importer),
        ):
            result = checker.inspect_environment(device=device, runtime_only=True)
        nvidia_mock.assert_called_once_with()
        return result, importer

    def assert_failed(self, report):
        self.assertFalse(report["passed"])
        self.assertFalse(report["gpu_runtime_ready"])
        self.assertFalse(report["training_memory_verified"])
        self.assertFalse(report["training_started"])

    def test_missing_torch_preserves_nvidia_diagnostics_without_ml_import(self):
        importer = Mock(side_effect=AssertionError("No ML package should be imported when torch is missing"))
        report, _ = self.inspect(
            versions={"torch": None, "torchvision": "0.29.0+cu130"}, importer=importer,
        )
        self.assert_failed(report)
        self.assertEqual(report["nvidia"], NVIDIA)
        self.assertIsNone(report["packages"]["torch"])
        self.assertIn("torch is not installed", "\n".join(report["errors"]))
        importer.assert_not_called()

    def test_cpu_only_torch_on_gpu_request_has_installation_guidance(self):
        torch = fake_torch()
        report, importer = self.inspect(torch=torch)
        self.assert_failed(report)
        errors = "\n".join(report["errors"])
        self.assertIn("CPU-only PyTorch", errors)
        self.assertIn(checker.CUDA_INDEX, errors)
        self.assertIn("cannot access CUDA", errors)
        self.assertEqual([call.args[0] for call in importer.call_args_list], ["torch"])
        torch.ones.assert_not_called()

    def test_wrong_cuda_build_and_unavailable_gpu_are_reported_together(self):
        torch = fake_torch(cuda_build="12.8")
        report, _ = self.inspect(torch=torch)
        self.assert_failed(report)
        errors = "\n".join(report["errors"])
        self.assertIn("Expected the CUDA 13.0", errors)
        self.assertIn("12.8", errors)
        self.assertIn("cannot access CUDA", errors)
        self.assertIn("nvcc alone does not fix", errors)
        torch.ones.assert_not_called()

    def test_cuda_build_with_no_visible_gpu_reports_driver_visibility(self):
        torch = fake_torch(cuda_build="13.0")
        report, _ = self.inspect(torch=torch)
        self.assert_failed(report)
        self.assertIn("NVIDIA driver/GPU visibility", "\n".join(report["errors"]))
        self.assertEqual(report["torch_cuda_build"], "13.0")
        torch.ones.assert_not_called()

    def test_torch_dll_import_failure_is_actionable_and_keeps_hardware_report(self):
        importer = Mock(side_effect=OSError("[WinError 126] DLL load failed"))
        report, _ = self.inspect(importer=importer)
        self.assert_failed(report)
        errors = "\n".join(report["errors"])
        self.assertIn("PyTorch could not import", errors)
        self.assertIn("WinError 126", errors)
        self.assertIn("Visual C++ x64 runtime", errors)
        self.assertIn("Rerun setup", errors)
        self.assertEqual(report["nvidia"], NVIDIA)
        importer.assert_called_once_with("torch")

    def test_old_nvidia_driver_is_reported_even_when_torch_is_missing(self):
        nvidia = {
            "available": True,
            "gpus": [{"index": "0", "name": "Test GPU", "driver_version": "579.9", "memory_mib": "20480"}],
            "detail": "Test GPU with old driver",
        }
        report, importer = self.inspect(versions={"torch": None, "torchvision": None}, nvidia=nvidia)
        self.assert_failed(report)
        errors = "\n".join(report["errors"])
        self.assertIn("579.9", errors)
        self.assertIn("minimum (580)", errors)
        self.assertIn("https://www.nvidia.com/Download/index.aspx", errors)
        self.assertIn("reboot", errors)
        importer.assert_not_called()

    def test_cuda_discovery_failure_is_caught_before_tensor_allocation(self):
        torch = fake_torch(cuda_build="13.0")
        torch.cuda.is_available.side_effect = RuntimeError("CUDA driver initialization failed")
        report, _ = self.inspect(torch=torch)
        self.assert_failed(report)
        errors = "\n".join(report["errors"])
        self.assertIn("CUDA discovery failed", errors)
        self.assertIn("restart the PC", errors)
        torch.ones.assert_not_called()

    def test_cpu_software_inspection_cannot_claim_gpu_training_ready(self):
        for gpu_visible in (False, True):
            with self.subTest(gpu_visible=gpu_visible):
                torch = fake_torch(cuda_build="13.0" if gpu_visible else None, available=gpu_visible)
                report, importer = self.inspect(torch=torch, device="cpu")
                self.assertTrue(report["passed"], report["errors"])
                self.assertFalse(report["gpu_runtime_ready"])
                self.assertFalse(report["training_memory_verified"])
                self.assertFalse(report["training_started"])
                self.assertNotIn("cuda_calculation_passed", report)
                self.assertEqual([call.args[0] for call in importer.call_args_list], ["torch", "torchvision"])
                torch.ones.assert_not_called()

    def test_standard_python_314_is_accepted(self):
        report, _ = self.inspect(torch=fake_torch(), device="cpu", python_version=(3, 14, 6))
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["gpu_runtime_ready"])

    def test_python_3141_is_rejected_with_torchvision_patch_guidance(self):
        report, _ = self.inspect(torch=fake_torch(), device="cpu", python_version=(3, 14, 1))
        self.assert_failed(report)
        self.assertIn("torchvision 0.29.0 explicitly excludes Python 3.14.1", "\n".join(report["errors"]))

    def test_free_threaded_python_is_rejected(self):
        report, _ = self.inspect(torch=fake_torch(), device="cpu", python_version=(3, 14, 6), free_threaded=True)
        self.assert_failed(report)
        self.assertIn("free-threaded Python build", "\n".join(report["errors"]))

    def test_unsupported_version_or_32bit_python_is_rejected(self):
        for version, bits in (((3, 11, 9), 64), ((3, 14, 6), 32)):
            with self.subTest(version=version, bits=bits):
                report, _ = self.inspect(torch=fake_torch(), device="cpu", python_version=version, python_bits=bits)
                self.assert_failed(report)


if __name__ == "__main__":
    unittest.main()
