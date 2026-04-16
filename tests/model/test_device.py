import torch
from unittest.mock import patch

from src.model.device import get_device


class TestGetDevice:
    def test_returns_torch_device(self):
        """get_device() must return a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_cuda_when_available(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cuda")

    def test_returns_mps_when_no_cuda(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.device = torch.device
            # hasattr on mock returns True by default
            device = get_device()
        assert device == torch.device("mps")

    def test_returns_cpu_as_fallback(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cpu")

    def test_returns_cpu_when_no_mps_attr(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Remove mps from backends
            del mock_torch.backends.mps
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cpu")
