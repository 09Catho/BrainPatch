"""Dataset helpers.

Contrast-set loading is pure Python and lives here. Text-corpus ingestion for
activation extraction needs ``datasets`` and lives in
:mod:`brainpatch.ml.corpus`, which is only imported inside Modal.
"""

from brainpatch.datasets.contrast_sets import (
    CONTRAST_SET_NAMES,
    default_contrast_dir,
    list_contrast_sets,
    load_contrast_set,
)

__all__ = [
    "CONTRAST_SET_NAMES",
    "default_contrast_dir",
    "list_contrast_sets",
    "load_contrast_set",
]
