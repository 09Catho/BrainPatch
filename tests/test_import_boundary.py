"""The local-machine import boundary.

This is the test that protects the project's core operating constraint: the
local machine is not a compute machine. If ``import brainpatch`` ever starts
pulling in torch, the whole "lightweight control plane" property is gone and
nobody would notice until a laptop tried to install 2 GB of CUDA wheels.
"""

from __future__ import annotations

import sys

import pytest


def test_import_brainpatch_without_ml_stack():
    """The top-level package imports with torch and friends blocked."""
    import brainpatch

    assert brainpatch.__version__


@pytest.mark.parametrize(
    "module",
    [
        "brainpatch.config",
        "brainpatch.paths",
        "brainpatch.cli",
        "brainpatch.schemas.patch",
        "brainpatch.schemas.manifest",
        "brainpatch.schemas.sae",
        "brainpatch.schemas.feature",
        "brainpatch.schemas.contrast",
        "brainpatch.steering.schedule",
        "brainpatch.steering.plan",
        "brainpatch.patches.io",
        "brainpatch.evaluation.metrics",
        "brainpatch.datasets.contrast_sets",
    ],
)
def test_lightweight_modules_import(module):
    """Every non-``ml`` module must import without the ML stack."""
    __import__(module)


def test_ml_stack_is_not_loaded_by_importing_brainpatch():
    """Nothing under ``brainpatch.ml`` is imported as a side effect."""
    import brainpatch  # noqa: F401

    for name in ("torch", "transformers", "datasets"):
        assert name not in sys.modules, f"{name} was imported by importing brainpatch"


def test_ml_subpackage_is_not_eagerly_imported():
    """``brainpatch.ml.*`` submodules stay unloaded until explicitly requested."""
    import brainpatch  # noqa: F401

    eager = [m for m in sys.modules if m.startswith("brainpatch.ml.")]
    assert eager == [], f"heavy submodules were imported eagerly: {eager}"


def test_modal_app_modules_do_not_import_torch_at_module_scope():
    """Modal orchestration modules are imported *locally* by the modal CLI.

    A top-level ``import torch`` in any of them would break `modal run` on a
    machine without the ML stack, which is exactly the machine this project
    targets. Checked by source inspection rather than by importing, because
    importing them requires the `modal` package.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []

    for path in sorted((repo_root / "modal_app").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only, not function bodies
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in {"torch", "transformers", "datasets", "accelerate", "gradio"}:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")

    assert not offenders, "module-scope heavy imports in modal_app: " + "; ".join(offenders)
