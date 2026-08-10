"""The runtime/research import boundary.

This is the test that protects the product's core promise: ``pip install
brainpatch`` must work, and be *useful*, on a machine with no ML stack.

Two separate boundaries are enforced here:

1. **Core vs everything heavy.** The patch format, registry, validation, CLI and
   backend discovery must import with torch, transformers, numpy, safetensors,
   vLLM and gradio all blocked. If that ever breaks, a 5 MB install silently
   becomes a 2 GB one.

2. **Runtime vs research.** ``brainpatch.runtime`` and ``brainpatch.backends``
   must never import ``brainpatch.research``. Research code is how patches are
   *made*; a user applying one needs none of it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Modules the *core* package must never need.
HEAVY = ("torch", "transformers", "datasets", "accelerate", "vllm", "mlx_lm", "gradio", "fastapi")

#: Every module a user touches before choosing a backend.
CORE_MODULES = [
    "brainpatch",
    "brainpatch.cli",
    "brainpatch.paths",
    "brainpatch.patch",
    "brainpatch.patch.format",
    "brainpatch.patch.tensors",
    "brainpatch.patch.loader",
    "brainpatch.patch.registry",
    "brainpatch.patch.validation",
    "brainpatch.runtime",
    "brainpatch.runtime.base",
    "brainpatch.runtime.auto",
    "brainpatch.runtime.capabilities",
    "brainpatch.runtime.model",
    "brainpatch.runtime.scheduling",
    "brainpatch.backends",
    "brainpatch.schemas.patch",
    "brainpatch.steering.schedule",
    "brainpatch.evaluation.metrics",
]


def test_import_brainpatch_without_ml_stack():
    import brainpatch

    assert brainpatch.__version__


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_modules_import_without_ml_stack(module):
    """Blocked by the session-wide fixture in conftest."""
    __import__(module)


def test_no_heavy_module_is_loaded_by_importing_brainpatch():
    import brainpatch  # noqa: F401

    for name in HEAVY:
        assert name not in sys.modules, f"{name} was imported by importing brainpatch"


def test_backends_package_does_not_eagerly_import_engines():
    """Importing the package must not pull in torch via a backend module."""
    import brainpatch.backends  # noqa: F401

    loaded = [m for m in sys.modules if m.startswith("brainpatch.backends.")]
    assert loaded == [], f"backend modules imported eagerly: {loaded}"


def test_research_is_not_imported_by_the_runtime():
    import brainpatch  # noqa: F401
    import brainpatch.runtime  # noqa: F401
    import brainpatch.patch  # noqa: F401

    leaked = [m for m in sys.modules if m.startswith("brainpatch.research")]
    assert leaked == [], f"research modules imported by the runtime: {leaked}"


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level import names in a file, ignoring function-body imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_runtime_modules_have_no_module_scope_heavy_imports():
    """Runtime code may import engines only inside function bodies."""
    offenders: list[str] = []
    for directory in ("runtime", "patch"):
        for path in sorted((REPO_ROOT / "brainpatch" / directory).glob("*.py")):
            if path.name == "compiler.py":
                continue  # research extra; documented as needing torch
            for name in _module_scope_imports(path):
                root = name.split(".")[0]
                if root in HEAVY:
                    offenders.append(f"{directory}/{path.name} imports {name}")
    assert not offenders, "module-scope heavy imports: " + "; ".join(offenders)


def test_runtime_does_not_reference_research_at_module_scope():
    offenders: list[str] = []
    for directory in ("runtime", "backends", "patch"):
        for path in sorted((REPO_ROOT / "brainpatch" / directory).glob("*.py")):
            for name in _module_scope_imports(path):
                if name.startswith("brainpatch.research"):
                    offenders.append(f"{directory}/{path.name} imports {name}")
    assert not offenders, "runtime imports research: " + "; ".join(offenders)


def test_backend_modules_import_engines_only_inside_functions():
    """Each backend's is_available()/capabilities() must work uninstalled."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "brainpatch" / "backends").glob("*.py")):
        for name in _module_scope_imports(path):
            root = name.split(".")[0]
            if root in HEAVY:
                offenders.append(f"{path.name} imports {name} at module scope")
    assert not offenders, "; ".join(offenders)


def test_modal_app_modules_do_not_import_torch_at_module_scope():
    """The modal CLI imports these locally, on a machine with no ML stack."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "modal_app").glob("*.py")):
        for name in _module_scope_imports(path):
            root = name.split(".")[0]
            if root in HEAVY:
                offenders.append(f"{path.name} imports {name}")
    assert not offenders, "module-scope heavy imports in modal_app: " + "; ".join(offenders)


def test_core_dependencies_stay_minimal():
    """The base wheel must not acquire a heavy dependency by accident."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps}
    forbidden = {"torch", "transformers", "numpy", "safetensors", "datasets", "vllm", "gradio"}
    assert not (names & forbidden), f"core dependencies must stay light, found: {names & forbidden}"
    assert names <= {"typer", "rich"}, f"unexpected core dependencies: {names}"
