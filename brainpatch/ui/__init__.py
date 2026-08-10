"""Local web UI. Requires ``pip install 'brainpatch[ui]'``."""

__all__ = ["launch"]


def __getattr__(name: str):
    if name == "launch":
        from brainpatch.ui.app import launch

        return launch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
