import subprocess
import unittest
from unittest.mock import Mock, patch

import config
from src.modules.hardware_detector import HardwareDetector


class HardwareDetectorTests(unittest.TestCase):
    @patch("src.modules.hardware_detector.HardwareDetector.is_plugged_in", return_value=True)
    @patch("src.modules.hardware_detector.HardwareDetector.detect_gpu", return_value=(True, "NVIDIA RTX 4070 Laptop GPU"))
    def test_select_model_prefers_large_model_when_gpu_and_plugged_in(self, _detect_gpu, _plugged_in):
        detector = HardwareDetector()
        self.assertEqual(detector.select_model(), config.LLM_MODEL_LARGE)
        self.assertEqual(detector.detect_hardware()["gpu_name"], "NVIDIA RTX 4070 Laptop GPU")

    @patch("src.modules.hardware_detector.HardwareDetector.is_plugged_in", return_value=False)
    @patch("src.modules.hardware_detector.HardwareDetector.detect_gpu", return_value=(True, "NVIDIA RTX 4070 Laptop GPU"))
    def test_select_model_prefers_medium_model_when_gpu_on_battery(self, _detect_gpu, _plugged_in):
        detector = HardwareDetector()
        self.assertEqual(detector.select_model(), config.LLM_MODEL_MEDIUM)

    @patch("src.modules.hardware_detector.HardwareDetector.detect_gpu", return_value=(False, "N/A"))
    def test_select_model_prefers_small_model_when_cpu_only(self, _detect_gpu):
        detector = HardwareDetector()
        self.assertEqual(detector.select_model(), config.LLM_MODEL_SMALL)

    @patch("src.modules.hardware_detector.torch")
    def test_detect_gpu_uses_torch_when_cuda_is_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4070 Laptop GPU"

        detector = HardwareDetector()

        self.assertTrue(detector.gpu_available)
        self.assertEqual(detector.gpu_name, "NVIDIA RTX 4070 Laptop GPU")

    @patch("src.modules.hardware_detector.subprocess.run")
    @patch("src.modules.hardware_detector.torch")
    def test_detect_gpu_falls_back_to_nvidia_smi(self, mock_torch, mock_run):
        mock_torch.cuda.is_available.return_value = False
        mock_run.side_effect = [
            Mock(returncode=0, stdout="NVIDIA RTX 4070 Laptop GPU\n"),
        ]

        detector = HardwareDetector()

        self.assertTrue(detector.gpu_available)
        self.assertEqual(detector.gpu_name, "NVIDIA RTX 4070 Laptop GPU")
        mock_run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("src.modules.hardware_detector.subprocess.run")
    @patch("src.modules.hardware_detector.torch")
    def test_detect_gpu_reports_cpu_only_when_all_checks_fail(self, mock_torch, mock_run):
        mock_torch.cuda.is_available.return_value = False
        mock_run.side_effect = [
            Mock(returncode=1, stdout=""),
            Mock(returncode=1, stdout=""),
        ]

        detector = HardwareDetector()

        self.assertFalse(detector.gpu_available)
        self.assertEqual(detector.gpu_name, "N/A")
        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
