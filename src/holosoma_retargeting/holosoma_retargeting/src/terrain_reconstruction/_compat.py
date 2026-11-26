"""Local replacements for the two terra-internal helpers the vendored core uses.

``atomic_write`` mirrors ``terra._files.atomic_write`` (Apache-2.0, amathislab/terra):
the writer receives a sibling temporary file and the destination is replaced only after
the writer returns successfully.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    """Replace one file only after ``writer`` completes successfully."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = ["atomic_write"]
