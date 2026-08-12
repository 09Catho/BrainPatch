"""Direction discovery and behavioural scoring for `anti_sycophancy_v1`.

This module answers one question and is shaped entirely by it: *does a direction
in activation space move a specific behavioural preference on unseen prompts,
more than matched controls do?*

Everything here is built around three cost and correctness constraints.

**One model load.** GPU time is the budget. Activations for every candidate
layer are captured in a single forward pass per sequence, and every scoring pass
is batched, so the whole layer x position x method grid costs a few minutes
rather than an hour.

**Paired, per-token margins.** Each item contributes
``margin = logP(desired)/n_desired - logP(undesired)/n_undesired`` under one
model state, and the reported quantity is the per-item *change* in that margin.
Pairing cancels item difficulty; per-token normalisation blunts the dataset's
known length skew.

**Discovery never touches validation or test.** Functions here take explicit
example lists. Nothing reaches for a global split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

#: Where along the sequence a direction is read off. Which of these works is an
#: empirical question, so it is scanned rather than assumed.
EXTRACTION_POSITIONS: tuple[str, ...] = ("last_prompt", "cont_mean", "cont_last")

#: Where a direction is injected during scoring. ``continuation`` is the
#: analogue of steering every generated token; ``prompt`` steers only the
#: context; ``all`` does both.
INJECTION_SITES: tuple[str, ...] = ("continuation", "prompt", "all")


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------


@dataclass
class EncodedPair:
    """One item's two continuations, tokenised and ready to batch."""

    topic: str
    category: str
    polarity: str
    split: str
    desired_ids: list[int]
    undesired_ids: list[int]
    prompt_len: int

    @property
    def n_desired(self) -> int:
        return len(self.desired_ids) - self.prompt_len

    @property
    def n_undesired(self) -> int:
        return len(self.undesired_ids) - self.prompt_len

    @property
    def length_gap(self) -> int:
        """Desired minus undesired continuation length, in tokens.

        Kept per item so the confound diagnostic can be computed directly
        rather than estimated from character counts.
        """
        return self.n_desired - self.n_undesired


def encode_pairs(
    tokenizer: Any,
    examples: Sequence[Any],
    *,
    use_chat_template: bool = True,
    max_length: int = 512,
) -> list[EncodedPair]:
    """Tokenise contrast examples once, up front.

    Both continuations share a prompt, and the prompt is tokenised *once* so
    ``prompt_len`` is guaranteed identical for the pair. Tokenising
    ``prompt + continuation`` separately for each side and subtracting lengths
    would let a boundary merge shift one side by a token, which silently
    misaligns the scored region.
    """
    encoded: list[EncodedPair] = []
    for example in examples:
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = example.prompt

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        desired = prompt_ids + tokenizer(
            example.positive_response, add_special_tokens=False
        ).input_ids
        undesired = prompt_ids + tokenizer(
            example.negative_response, add_special_tokens=False
        ).input_ids

        metadata = getattr(example, "metadata", {}) or {}
        encoded.append(
            EncodedPair(
                topic=str(metadata.get("topic", "")),
                category=str(getattr(example, "category", "")),
                polarity=str(metadata.get("polarity", "false_claim")),
                split=str(metadata.get("split", "train")),
                desired_ids=desired[:max_length],
                undesired_ids=undesired[:max_length],
                prompt_len=len(prompt_ids),
            )
        )
    return encoded


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


class DirectionInjector:
    """Adds ``strength * unit(direction)`` to one layer's residual stream.

    The additive mask is set per batch, which is what allows the same hook to
    express "steer only the continuation", "steer only the prompt" and "steer
    everything" without reinstalling anything.
    """

    def __init__(self, direction: torch.Tensor, strength: float) -> None:
        norm = torch.linalg.vector_norm(direction)
        if float(norm) <= 0:
            raise ValueError("direction has zero norm")
        self.vector = (direction / norm).detach()
        self.strength = float(strength)
        self.mask: torch.Tensor | None = None
        self._handle: Any = None
        self.calls = 0

    def _hook(self, module: Any, args: Any, output: Any) -> Any:
        if self.mask is None or self.strength == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        delta = self.mask.unsqueeze(-1).to(hidden.dtype) * self.vector.to(
            hidden.device, hidden.dtype
        )
        hidden = hidden + self.strength * delta
        self.calls += 1
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def attach(self, layer_module: Any) -> "DirectionInjector":
        self._handle = layer_module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class ItemScore:
    """One item's per-token margin under one model state."""

    topic: str
    category: str
    polarity: str
    desired_logprob: float
    undesired_logprob: float
    n_desired: int
    n_undesired: int
    length_gap: int

    @property
    def margin(self) -> float:
        return (
            self.desired_logprob / max(1, self.n_desired)
            - self.undesired_logprob / max(1, self.n_undesired)
        )


def _sequence_logprobs(
    model: Any,
    batch_ids: list[list[int]],
    prompt_lens: list[int],
    pad_id: int,
    device: Any,
    injector: DirectionInjector | None,
    inject_site: str,
) -> list[float]:
    """Summed continuation log-probability for each sequence in a batch.

    Right-padded. Padding never enters a score because only positions in
    ``[prompt_len-1, seq_len-1)`` are gathered, and the additive mask is zero on
    pad positions so a padded run cannot perturb a real one.
    """
    width = max(len(ids) for ids in batch_ids)
    input_ids = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch_ids), width), dtype=torch.long)
    steer_mask = torch.zeros((len(batch_ids), width), dtype=torch.float32)

    for row, (ids, prompt_len) in enumerate(zip(batch_ids, prompt_lens)):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[row, : len(ids)] = 1
        if inject_site == "continuation":
            steer_mask[row, prompt_len : len(ids)] = 1.0
        elif inject_site == "prompt":
            steer_mask[row, :prompt_len] = 1.0
        else:
            steer_mask[row, : len(ids)] = 1.0

    input_ids = input_ids.to(device)
    attention = attention.to(device)
    if injector is not None:
        injector.mask = steer_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits

    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    out: list[float] = []
    for row, (ids, prompt_len) in enumerate(zip(batch_ids, prompt_lens)):
        out.append(float(gathered[row, prompt_len - 1 : len(ids) - 1].sum().item()))
    return out


def score_pairs(
    model: Any,
    pairs: Sequence[EncodedPair],
    *,
    pad_id: int,
    device: Any,
    injector: DirectionInjector | None = None,
    inject_site: str = "continuation",
    batch_size: int = 8,
) -> list[ItemScore]:
    """Per-token margins for every pair under the current model state.

    Both sides of every pair go through the same batching path, so any batching
    artefact affects desired and undesired identically and cancels in the
    margin.
    """
    flat_ids: list[list[int]] = []
    flat_prompt: list[int] = []
    for pair in pairs:
        flat_ids.append(pair.desired_ids)
        flat_prompt.append(pair.prompt_len)
        flat_ids.append(pair.undesired_ids)
        flat_prompt.append(pair.prompt_len)

    # Length-sorted batching keeps padding low; the original order is restored
    # afterwards so results line up with `pairs`.
    order = sorted(range(len(flat_ids)), key=lambda i: len(flat_ids[i]))
    totals: list[float] = [0.0] * len(flat_ids)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        values = _sequence_logprobs(
            model,
            [flat_ids[i] for i in chunk],
            [flat_prompt[i] for i in chunk],
            pad_id,
            device,
            injector,
            inject_site,
        )
        for index, value in zip(chunk, values):
            totals[index] = value

    scores: list[ItemScore] = []
    for position, pair in enumerate(pairs):
        scores.append(
            ItemScore(
                topic=pair.topic,
                category=pair.category,
                polarity=pair.polarity,
                desired_logprob=totals[2 * position],
                undesired_logprob=totals[2 * position + 1],
                n_desired=pair.n_desired,
                n_undesired=pair.n_undesired,
                length_gap=pair.length_gap,
            )
        )
    return scores


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


@dataclass
class DeltaSummary:
    """Paired change in margin for one polarity."""

    polarity: str
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    proportion_improved: float
    cohens_d: float
    by_category: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polarity": self.polarity,
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "proportion_improved": self.proportion_improved,
            "cohens_d": self.cohens_d,
            "by_category": self.by_category,
        }


def _bootstrap_ci(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(
        0, len(values), (resamples, len(values)), generator=generator
    )
    means = tensor[indices].mean(dim=1)
    low = float(torch.quantile(means, 0.025).item())
    high = float(torch.quantile(means, 0.975).item())
    return (low, high)


def summarize_deltas(
    baseline: Sequence[ItemScore],
    patched: Sequence[ItemScore],
    *,
    polarity: str,
    seed: int = 0,
) -> DeltaSummary:
    """Paired delta statistics restricted to one polarity.

    Pairs are matched by position, and a mismatch raises rather than silently
    comparing different items -- a misalignment here would produce a plausible
    number that means nothing.
    """
    if len(baseline) != len(patched):
        raise ValueError("baseline and patched score lists differ in length")

    deltas: list[float] = []
    categories: dict[str, list[float]] = {}
    for before, after in zip(baseline, patched):
        if before.topic != after.topic:
            raise ValueError(f"score misalignment: {before.topic!r} vs {after.topic!r}")
        if before.polarity != polarity:
            continue
        delta = after.margin - before.margin
        deltas.append(delta)
        categories.setdefault(before.category, []).append(delta)

    if not deltas:
        return DeltaSummary(polarity, 0, 0.0, 0.0, float("nan"), float("nan"), 0.0, 0.0, {})

    tensor = torch.tensor(deltas, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if len(deltas) > 1 else 0.0
    low, high = _bootstrap_ci(deltas, seed=seed)
    return DeltaSummary(
        polarity=polarity,
        n=len(deltas),
        mean=mean,
        median=float(tensor.median().item()),
        ci_low=low,
        ci_high=high,
        proportion_improved=float((tensor > 0).double().mean().item()),
        cohens_d=(mean / std) if std > 0 else 0.0,
        by_category={k: float(sum(v) / len(v)) for k, v in sorted(categories.items())},
    )


def length_gap_correlation(
    baseline: Sequence[ItemScore], patched: Sequence[ItemScore]
) -> float:
    """Pearson r between per-item delta and per-item token length gap.

    The dataset's desired responses are systematically longer. If steering
    strength tracks that gap, the "behavioural" effect is a length preference
    wearing a costume, and the pre-registered criteria reject it.
    """
    deltas: list[float] = []
    gaps: list[float] = []
    for before, after in zip(baseline, patched):
        deltas.append(after.margin - before.margin)
        gaps.append(float(before.length_gap))
    if len(deltas) < 3:
        return 0.0
    x = torch.tensor(deltas, dtype=torch.float64)
    y = torch.tensor(gaps, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denominator == 0:
        return 0.0
    return float((x @ y).item() / denominator)


# ---------------------------------------------------------------------------
# direction discovery
# ---------------------------------------------------------------------------


def capture_layer_activations(
    model: Any,
    layer_modules: dict[int, Any],
    pairs: Sequence[EncodedPair],
    *,
    pad_id: int,
    device: Any,
    batch_size: int = 8,
) -> dict[int, dict[str, torch.Tensor]]:
    """Residual activations at every candidate layer, at every extraction point.

    One forward pass per sequence serves all layers and all three extraction
    positions. Scanning layers with a separate pass each would multiply the
    dominant cost of the experiment by seven for no information gain.

    Returns ``{layer: {f"{position}_{side}": tensor of shape (n_items, hidden)}}``.
    """
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer: int) -> Callable[..., None]:
        def hook(module: Any, args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach().float()

        return hook

    for layer, module in layer_modules.items():
        handles.append(module.register_forward_hook(make_hook(layer)))

    accumulator: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {f"{p}_{s}": [] for p in EXTRACTION_POSITIONS for s in ("desired", "undesired")}
        for layer in layer_modules
    }

    try:
        for side, attribute in (("desired", "desired_ids"), ("undesired", "undesired_ids")):
            for start in range(0, len(pairs), batch_size):
                chunk = list(pairs[start : start + batch_size])
                batch_ids = [getattr(p, attribute) for p in chunk]
                width = max(len(ids) for ids in batch_ids)
                input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
                attention = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, ids in enumerate(batch_ids):
                    input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention[row, : len(ids)] = 1

                with torch.inference_mode():
                    model(
                        input_ids=input_ids.to(device),
                        attention_mask=attention.to(device),
                        use_cache=False,
                    )

                for layer in layer_modules:
                    hidden = captured[layer]
                    for row, (pair, ids) in enumerate(zip(chunk, batch_ids)):
                        start_index = pair.prompt_len
                        end_index = len(ids)
                        accumulator[layer][f"last_prompt_{side}"].append(
                            hidden[row, start_index - 1].cpu()
                        )
                        if end_index > start_index:
                            accumulator[layer][f"cont_mean_{side}"].append(
                                hidden[row, start_index:end_index].mean(dim=0).cpu()
                            )
                            accumulator[layer][f"cont_last_{side}"].append(
                                hidden[row, end_index - 1].cpu()
                            )
                        else:  # degenerate pair; keep alignment
                            accumulator[layer][f"cont_mean_{side}"].append(
                                hidden[row, start_index - 1].cpu()
                            )
                            accumulator[layer][f"cont_last_{side}"].append(
                                hidden[row, start_index - 1].cpu()
                            )
    finally:
        for handle in handles:
            handle.remove()

    return {
        layer: {key: torch.stack(values) for key, values in buckets.items()}
        for layer, buckets in accumulator.items()
    }


def fit_caa(desired: torch.Tensor, undesired: torch.Tensor) -> torch.Tensor:
    """Difference of means. The baseline every other method has to beat."""
    return desired.mean(dim=0) - undesired.mean(dim=0)


def fit_pca(desired: torch.Tensor, undesired: torch.Tensor) -> torch.Tensor:
    """Top principal component of the paired difference distribution.

    Sign is fixed to agree with the mean difference, since a principal component
    is only defined up to sign and an arbitrary flip would invert the steering.
    """
    difference = desired - undesired
    centered = difference - difference.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=min(8, centered.shape[0] - 1, centered.shape[1]))
    component = v[:, 0]
    if float(component @ difference.mean(dim=0)) < 0:
        component = -component
    return component


def fit_probe(
    desired: torch.Tensor,
    undesired: torch.Tensor,
    *,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Logistic probe separating the two classes; returns (direction, accuracy).

    Accuracy is returned so it can be *reported*, not so it can be used as a
    success criterion. A probe can separate two classes almost perfectly and
    still give a direction that steers nothing -- readable and causal are
    different properties, and conflating them is a common way to overclaim.

    Features are standardised before fitting and the learned weights are mapped
    back to raw activation space, so the direction is comparable with CAA's.
    """
    torch.manual_seed(seed)
    features = torch.cat([desired, undesired], dim=0).double()
    labels = torch.cat(
        [torch.ones(len(desired)), torch.zeros(len(undesired))], dim=0
    ).double()

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = (features - mean) / std

    weights = torch.zeros(standardized.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = standardized @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = ((standardized @ weights + bias) > 0).double()
        accuracy = float((predictions == labels).double().mean().item())
        direction = (weights / std.squeeze(0)).float()

    return direction, accuracy


def random_directions(
    hidden: int, count: int, *, seed: int = 0, device: Any = "cpu"
) -> list[torch.Tensor]:
    """Scale-matched random unit directions.

    These are the control that the first BrainPatch experiment failed. Any
    claimed effect has to beat the *maximum* over this set, not the mean.
    """
    generator = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for _ in range(count):
        vector = torch.randn(hidden, generator=generator)
        out.append((vector / torch.linalg.vector_norm(vector)).to(device))
    return out


def shuffled_label_direction(
    desired: torch.Tensor,
    undesired: torch.Tensor,
    *,
    method: str = "caa",
    seed: int = 0,
) -> torch.Tensor:
    """Refit a direction after permuting which side is which.

    If the pipeline finds a "working" direction from label noise, the method is
    fitting the dataset's incidental structure and no result from it means
    anything. This is the control that catches that.
    """
    generator = torch.Generator().manual_seed(seed)
    stacked = torch.cat([desired, undesired], dim=0)
    permutation = torch.randperm(len(stacked), generator=generator)
    half = len(desired)
    left = stacked[permutation[:half]]
    right = stacked[permutation[half:]]
    if method == "pca":
        size = min(len(left), len(right))
        return fit_pca(left[:size], right[:size])
    return fit_caa(left, right)


class GenerationInjector:
    """Injects during autoregressive generation, where the mask trick fails.

    With a KV cache the first forward pass covers the whole prompt and every
    later pass carries a single token, so "which positions am I steering" has to
    be answered from the pass index rather than from a precomputed mask. Getting
    this wrong is silent: a patch meant to steer only the prompt would go on
    steering every generated token and the free-generation numbers would not
    correspond to the scored configuration at all.
    """

    def __init__(self, direction: torch.Tensor, strength: float, site: str) -> None:
        norm = torch.linalg.vector_norm(direction)
        if float(norm) <= 0:
            raise ValueError("direction has zero norm")
        if site not in INJECTION_SITES:
            raise ValueError(f"unknown injection site {site!r}")
        self.vector = (direction / norm).detach()
        self.strength = float(strength)
        self.site = site
        self.prompt_pass_done = False
        self.applied = 0
        self._handle: Any = None

    def reset(self) -> None:
        self.prompt_pass_done = False
        self.applied = 0

    def _hook(self, module: Any, args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        is_prompt_pass = not self.prompt_pass_done
        self.prompt_pass_done = True

        active = (
            self.site == "all"
            or (self.site == "prompt" and is_prompt_pass)
            or (self.site == "continuation" and not is_prompt_pass)
        )
        if not active or self.strength == 0.0:
            return output

        hidden = hidden + self.strength * self.vector.to(hidden.device, hidden.dtype)
        self.applied += 1
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def attach(self, layer_module: Any) -> "GenerationInjector":
        self._handle = layer_module.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def residual_norm_percentiles(
    activations: torch.Tensor, percentiles: Iterable[float] = (50, 90, 95, 99)
) -> dict[str, float]:
    """Percentiles of the residual-stream norm, for strength calibration.

    Strength is expressed as a fraction of the activation magnitude the model
    naturally carries at that layer, so an intervention stays on-manifold
    instead of being an arbitrary large number that merely breaks the model.
    """
    norms = torch.linalg.vector_norm(activations.float(), dim=-1)
    return {
        f"p{int(p)}": float(torch.quantile(norms, p / 100.0).item()) for p in percentiles
    }
