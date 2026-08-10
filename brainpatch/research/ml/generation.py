"""Deterministic generation and log-probability measurement.

Every comparison in the causal-validation harness -- baseline, intervention,
control -- must differ *only* in the intervention. That means identical
sampling settings, identical seeds, and identical prompts.
:class:`GenerationConfig` makes those settings one object that gets passed to
every condition, so there is no way to accidentally give the baseline a
different temperature.

Greedy decoding is the default. Sampling adds variance that would have to be
averaged out with many more (paid) generations before an effect could be
distinguished from noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class GenerationConfig:
    """Sampling settings, shared identically across all compared conditions."""

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int = 0

    def to_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                kwargs["top_k"] = self.top_k
        if self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_chat_prompt(tokenizer: Any, user_message: str, system: str | None = None) -> str:
    """Render a user message through the model's chat template.

    Falls back to the raw message for base models with no template, rather than
    silently inventing one -- a wrong template changes the distribution enough
    to invalidate a comparison.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return user_message
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Log-probability the model assigns to ``continuation`` after ``prompt``.

    Used to measure an intervention's effect without generating: the difference
    in logprob between a positive and negative response under baseline versus
    steered conditions is a lower-variance signal than comparing free
    generations.

    Returns total and per-token log-probability, plus the token count.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(
        prompt + continuation, return_tensors="pt", add_special_tokens=True
    ).input_ids
    prompt_len = prompt_ids.shape[1]
    n_continuation = full_ids.shape[1] - prompt_len
    if n_continuation <= 0:
        return {"total_logprob": 0.0, "mean_logprob": 0.0, "num_tokens": 0}

    full_ids = full_ids.to(device)
    logits = model(input_ids=full_ids).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full_ids[:, 1:]

    # Score only the continuation positions.
    start = prompt_len - 1
    selected = log_probs[0, start:, :].gather(-1, targets[0, start:].unsqueeze(-1)).squeeze(-1)
    total = float(selected.sum().item())
    return {
        "total_logprob": total,
        "mean_logprob": total / selected.numel(),
        "num_tokens": int(selected.numel()),
    }
