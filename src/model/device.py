"""Auto-detect the best available compute device."""

import torch


def get_device() -> torch.device:
    """Auto-detect best available device.

    Priority: CUDA (Colab/NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
