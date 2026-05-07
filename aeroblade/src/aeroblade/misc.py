import json
from pathlib import Path

import torch


def safe_mkdir(directory: Path) -> None:
    """Create directory if needed, but never prompt (idempotent & restart-friendly)."""
    directory.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Return 'cuda' if available, 'cpu' otherwise"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_config(config: dict, directory: Path) -> None:
    """Write config text file to specified directory."""
    with open(directory / "config.json", "w") as f:
        json.dump({key: str(value) for key, value in config.items()}, f)
