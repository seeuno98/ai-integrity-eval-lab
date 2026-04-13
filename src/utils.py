"""Shared utilities: logging, seeding, directory setup, and metric helpers."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) if they do not exist.

    Args:
        *paths: One or more directory paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def format_metrics(metrics: dict[str, Any], prefix: str = "") -> str:
    """Format a metrics dict as a readable string for logging.

    Args:
        metrics: Dictionary of metric name -> value.
        prefix: Optional prefix string prepended to each line.

    Returns:
        Multi-line formatted string.
    """
    lines = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"{prefix}{k}: {v:.4f}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)
