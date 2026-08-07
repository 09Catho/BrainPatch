"""Loading behavioural contrast fixtures.

The sets shipped in ``examples/contrast/`` are small, synthetic, hand-written
development fixtures. They are the input to candidate-feature search, and they
are *not* benchmarks -- see the module docstring of
:mod:`brainpatch.schemas.contrast`.
"""

from __future__ import annotations

import os
from pathlib import Path

from brainpatch.schemas.contrast import ContrastSet

#: Fixtures shipped with the repository.
CONTRAST_SET_NAMES: tuple[str, ...] = (
    "sycophancy",
    "verification",
    "verbosity",
    "contradiction",
)


def default_contrast_dir() -> Path:
    """Directory holding the shipped contrast fixtures.

    Resolved relative to the installed package so it works from a checkout and
    from inside a Modal container where the repo lives at ``/root``.
    """
    here = Path(__file__).resolve()
    # brainpatch/datasets/contrast_sets.py -> repo root
    repo_root = here.parent.parent.parent
    return repo_root / "examples" / "contrast"


def list_contrast_sets(directory: str | os.PathLike[str] | None = None) -> list[str]:
    """Names of every contrast set found in ``directory``."""
    d = Path(directory) if directory is not None else default_contrast_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_contrast_set(
    name: str, directory: str | os.PathLike[str] | None = None
) -> ContrastSet:
    """Load a contrast set by name, validating its contents.

    Raises
    ------
    FileNotFoundError
        If no such set exists, listing what is available.
    """
    d = Path(directory) if directory is not None else default_contrast_dir()
    path = d / f"{name}.json"
    if not path.is_file():
        available = list_contrast_sets(d)
        raise FileNotFoundError(
            f"contrast set {name!r} not found at {path}. Available: {available or 'none'}"
        )
    contrast_set = ContrastSet.from_json(path.read_text(encoding="utf-8"))
    contrast_set.validate()
    return contrast_set
