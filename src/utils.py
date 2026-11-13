"""Shared helpers: seeding, device selection, GPU-synchronized timing, and result saving."""

import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the requested device, or the best available one (cuda, then mps, then cpu)."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded torch.Generator living on the given device."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def synchronize(device: torch.device) -> None:
    """Block until all queued kernels on the device are finished (needed for fair timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Context manager measuring wall-clock time in seconds, synchronizing the device on entry and exit."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start


def save_json(data: Any, path: str) -> None:
    """Write data to a JSON file, creating parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Read a JSON file."""
    with Path(path).open() as f:
        return json.load(f)
    