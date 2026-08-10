"""OpenAI-compatible HTTP serving. Requires ``pip install 'brainpatch[server]'``."""

__all__ = ["build_app"]


def __getattr__(name: str):
    if name == "build_app":
        from brainpatch.server.app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
