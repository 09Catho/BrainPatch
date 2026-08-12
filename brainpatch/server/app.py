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

.. note::
   This module deliberately has **no** ``from __future__ import annotations``
   and defines its request models at module scope. FastAPI resolves parameter
   annotations at runtime against the module's globals; with postponed
   annotations and locally-defined models it cannot find them, silently demotes
   the request body to a query parameter, and every POST fails with
   ``422 Field required``. Keep both properties as they are.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from brainpatch.runtime.base import GenerationConfig

#: Cap on tokens a single request may ask for, whatever it sends.
MAX_REQUEST_TOKENS = 4096


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=MAX_REQUEST_TOKENS)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None
    #: Namespaced extension; see the module docstring on why it is validated
    #: rather than honoured per request.
    brainpatch: Optional[Dict[str, float]] = None


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=MAX_REQUEST_TOKENS)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    brainpatch: Optional[Dict[str, float]] = None


def build_app(model: Any, served_model_name: Optional[str] = None) -> Any:
    """Build the FastAPI application around an already-loaded model."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
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

    api = FastAPI(title="BrainPatch", version="1.0")

    def check_patch_override(requested: Optional[Dict[str, float]]) -> None:
        """Accept a per-request patch spec only if it matches the server's."""
        if not requested:
            return
        if capabilities.per_request_strength:
            return
        current = {name: model.backend.patches[name].strength for name in model.list_patches()}
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

    def make_config(
        max_tokens: Optional[int], temperature: float, top_p: float, stop: Optional[List[str]]
    ) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=min(max_tokens or 256, MAX_REQUEST_TOKENS),
            temperature=temperature,
            top_p=top_p,
            stop=list(stop or []),
        )

    def render(messages: List[ChatMessage]) -> tuple:
        system = next((m.content for m in messages if m.role == "system"), None)
        user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if user is None:
            raise HTTPException(status_code=400, detail="no user message in request")
        return user, system

    @api.get("/health")
    def health() -> Dict[str, Any]:
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
    def list_models() -> Dict[str, Any]:
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
        check_patch_override(request.brainpatch)
        prompt, system = render(request.messages)
        cfg = make_config(request.max_tokens, request.temperature, request.top_p, request.stop)

        if request.stream:
            return StreamingResponse(
                stream_chat(prompt, cfg, system, model_name), media_type="text/event-stream"
            )
        text = model.generate(prompt, cfg, system=system)
        return chat_response(
            text,
            model_name,
            prompt_tokens=count_tokens(model.backend, prompt),
            completion_tokens=count_tokens(model.backend, text),
        )

    @api.post("/v1/completions")
    def completions(request: CompletionRequest) -> Dict[str, Any]:
        check_patch_override(request.brainpatch)
        cfg = make_config(request.max_tokens, request.temperature, request.top_p, None)
        text = model.generate(request.prompt, cfg, use_chat_template=False)
        return {
            "id": "cmpl-" + uuid.uuid4().hex[:24],
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop", "logprobs": None}],
        }

    def stream_chat(prompt: str, cfg: GenerationConfig, system: Optional[str], name: str):
        import json as _json

        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        def chunk(delta: Dict[str, Any], finish: Optional[str]) -> str:
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return "data: " + _json.dumps(payload) + "\n\n"

        try:
            for piece in model.stream(prompt, cfg, system=system):
                yield chunk({"content": piece}, None)
        except NotImplementedError:
            # Backends without streaming still serve a valid SSE response.
            yield chunk({"content": model.generate(prompt, cfg, system=system)}, None)
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return api


def count_tokens(backend: Any, text: str) -> int:
    """Token count for ``text``, or 0 if the backend cannot tokenise.

    Reported zeros were a real problem, not a cosmetic one: any client that
    meters usage, or any benchmark that normalises by generated tokens, silently
    reads "nothing happened". Falling back to 0 is still possible for backends
    with no tokenizer, but a backend that has one now reports the truth.
    """
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        # vLLM keeps its tokenizer behind the engine handle rather than exposing
        # an attribute, so a plain getattr finds nothing and every count would
        # come back 0 on the backend most likely to be metered.
        engine = getattr(backend, "llm", None)
        getter = getattr(engine, "get_tokenizer", None)
        if callable(getter):
            try:
                tokenizer = getter()
            except Exception:
                tokenizer = None
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return 0


def chat_response(
    text: str, model_name: str, *, prompt_tokens: int = 0, completion_tokens: int = 0
) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
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
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
