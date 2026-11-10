"""Select project-local GPU/CPU PyTorch without modifying either installation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for name in ['.torch_deps', '.torch_gpu_deps']:
    if (ROOT / name / 'torch/__init__.py').exists():
        sys.path.insert(0, str(ROOT / name))
import torch

__all__ = ['torch']
