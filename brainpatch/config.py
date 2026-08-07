"""YAML experiment configuration with dotted CLI overrides.

Every BrainPatch experiment is fully described by a YAML file plus a list of
``key.path=value`` overrides. The resolved dictionary is written into the
experiment directory on the Volume, so a run can always be reconstructed from
its own artifacts.

The loader is intentionally boring: no interpolation, no plugins, no imports.
It has to run on a machine with nothing but ``pyyaml`` installed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - exercised implicitly; guard keeps import failures readable
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "brainpatch.config requires PyYAML. Install the tiny control-plane extra: "
        "pip install 'brainpatch[modal]'"
    ) from exc


class ConfigError(ValueError):
    """Raised when a configuration file or override is unusable."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file into a plain dict.

    Raises
    ------
    ConfigError
        If the file is missing, unparseable, or does not contain a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested mappings merge key-by-key; every other type is replaced wholesale.

    >>> deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    {'a': {'x': 1, 'y': 3}}
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _coerce_scalar(raw: str, value: Any) -> Any:
    """Recover numeric types that YAML 1.1 leaves as strings.

    ``yaml.safe_load`` follows the YAML 1.1 float grammar, which requires a
    decimal point and a signed exponent: ``1.0e-4`` parses as a float but
    ``1e-4`` parses as the *string* ``"1e-4"``. That is a nasty failure mode for
    a config override -- ``--set sae.lr=1e-4`` would silently hand a string to
    the optimizer instead of a learning rate. Python's own float grammar is more
    permissive, so retry with it.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse a ``a.b.c=value`` override into a key path and a typed value.

    Values are parsed as YAML scalars with a numeric-coercion fallback, so
    ``k=32`` gives an int, ``lr=1e-4`` a float, ``flag=true`` a bool, and
    ``name=smoke_v0`` a string.

    >>> parse_override("sae.k=32")
    (['sae', 'k'], 32)
    >>> parse_override("run.force=true")
    (['run', 'force'], True)
    >>> parse_override("sae.lr=1e-4")
    (['sae', 'lr'], 0.0001)
    """
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like 'key.path=value'")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value in {text!r}: {exc}") from exc
    return key.split("."), _coerce_scalar(raw, value)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides on top of ``config``."""
    result = json.loads(json.dumps(config)) if config else {}
    for override in overrides:
        path, value = parse_override(override)
        cursor = result
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = value
    return result


def load_config(
    path: str | os.PathLike[str] | None,
    overrides: Iterable[str] = (),
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve defaults -> YAML file -> CLI overrides into one dict."""
    config: dict[str, Any] = dict(defaults or {})
    if path is not None:
        config = deep_merge(config, load_yaml(path))
    return apply_overrides(config, overrides)


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Fetch ``a.b.c`` from a nested config, with a clear error if absent."""
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ConfigError(f"missing required config key {dotted_key!r}")
        cursor = cursor[part]
    return cursor


def get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from a nested config, returning ``default`` if absent."""
    try:
        return require(config, dotted_key)
    except ConfigError:
        return default


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@dataclass
class RunProvenance:
    """Everything needed to say *what exactly* produced an artifact."""

    git_commit: str | None = None
    git_dirty: bool | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    gpu: str | None = None
    hostname: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "package_versions": dict(self.package_versions),
            "gpu": self.gpu,
            "hostname": self.hostname,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **self.extra,
        }


def git_commit(repo: str | os.PathLike[str] | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` outside a repository.

    Never raises: provenance capture must not be able to fail a run.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_is_dirty(repo: str | os.PathLike[str] | None = None) -> bool | None:
    """True if the working tree has uncommitted changes; ``None`` if unknown."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())
