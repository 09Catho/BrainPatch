"""OpenAI-compatible HTTP server.

Existing OpenAI clients work unchanged: point ``base_url`` at this server and
every request is served by the patched model. That is the whole point -- a
BrainPatch should slot into the stack an application already has.

Patch state is configured at **startup**, not per request. With a shared model
object, a per-request strength change would be visible to every other in-flight
request, so accepting one would be a correctness bug affecting other users'
output. The ``brainpatch`` extra field is therefore accepted only when it
matches the server's configuration, and rejected with a clear 400 otherwise,
rather than silently ignored.

Security posture: no patch loading over HTTP, no filesystem paths from request
bodies, no code execution, strengths clamped to each patch's declared envelope.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from brainpatch.runtime.base import GenerationConfig

#: Cap on tokens a single request may ask for, whatever it sends.
MAX_REQUEST_TOKENS = 4096


def build_app(model: Any, *, served_model_name: str | None = None) -> Any:
    """Build the FastAPI application around an already-loaded model."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "the server needs FastAPI -- pip install 'brainpatch[server]'"
        ) from exc

    descriptor = model.backend.describe_model()
    model_name = served_model_name or descriptor.model_id
    capabilities = model.capabilities()

    # Freeze patch state for the server's lifetime where the backend supports it.
    begin = getattr(model.backend, "begin_serving", None)
    if callable(begin):
        begin()

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[ChatMessage]
        max_tokens: int | None = Field(default=None, ge=1, le=MAX_REQUEST_TOKENS)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_p: float = Field(default=1.0, gt=0.0, le=1.0)
        stream: bool = False
        stop: list[str] | None = None
        brainpatch: dict[str, float] | None = None

    class CompletionRequest(BaseModel):
        model: str | None = None
        prompt: str
        max_tokens: int | None = Field(default=None, ge=1, le=MAX_REQUEST_TOKENS)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_p: float = Field(default=1.0, gt=0.0, le=1.0)
        stream: bool = False
        brainpatch: dict[str, float] | None = None

    api = FastAPI(title="BrainPatch", version="1.0")

    def _check_patch_override(requested: dict[str, float] | None) -> None:
        """Accept a per-request patch spec only if it matches the server's."""
        if not requested:
            return
        if not capabilities.per_request_strength:
            current = {
                name: model.backend.patches[name].strength for name in model.list_patches()
            }
            mismatched = {
                name: value
                for name, value in requested.items()
                if abs(current.get(name, 0.0) - float(value)) > 1e-9
            }
            if mismatched:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            f"the '{model.backend.name}' backend does not support "
                            "per-request patch strength; requests share one model, so "
                            "honouring this would change other users' output."
                        ),
                        "server_configuration": current,
                        "requested": requested,
                        "hint": "restart the server with the strengths you want",
                    },
                )

    def _config(max_tokens: int | None, temperature: float, top_p: float, stop: list[str] | None) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=min(max_tokens or 256, MAX_REQUEST_TOKENS),
            temperature=temperature,
            top_p=top_p,
            stop=list(stop or []),
        )

    def _render(messages: list[ChatMessage]) -> tuple[str, str | None]:
        system = next((m.content for m in messages if m.role == "system"), None)
        user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if user is None:
            raise HTTPException(status_code=400, detail="no user message in request")
        return user, system

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": model.backend.name,
            "model": model_name,
            "patches": {
                name: {
                    "strength": model.backend.patches[name].strength,
                    "enabled": model.backend.patches[name].enabled,
                    "evidence_level": model.backend.patches[name].manifest.evidence_level,
                }
                for name in model.list_patches()
            },
        }

    @api.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "brainpatch",
                    "brainpatch": {"patches": model.list_patches()},
                }
            ],
        }

    @api.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> Any:
        _check_patch_override(request.brainpatch)
        prompt, system = _render(request.messages)
        cfg = _config(request.max_tokens, request.temperature, request.top_p, request.stop)

        if request.stream:
            return StreamingResponse(
                _stream_chat(prompt, cfg, system, model_name),
                media_type="text/event-stream",
            )

        text = model.generate(prompt, cfg, system=system)
        return _chat_response(text, model_name)

    @api.post("/v1/completions")
    def completions(request: CompletionRequest) -> Any:
        _check_patch_override(request.brainpatch)
        cfg = _config(request.max_tokens, request.temperature, request.top_p, None)
        text = model.generate(request.prompt, cfg, use_chat_template=False)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop", "logprobs": None}],
        }

    def _stream_chat(prompt: str, cfg: GenerationConfig, system: str | None, name: str) -> Any:
        import json as _json

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        try:
            for chunk in model.stream(prompt, cfg, system=system):
                payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": name,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {_json.dumps(payload)}\n\n"
        except NotImplementedError:
            text = model.generate(prompt, cfg, system=system)
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {_json.dumps(payload)}\n\n"
        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {_json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return api


def _chat_response(text: str, model_name: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
