"""SAE training loop with checkpointing and resume.

A checkpoint captures everything needed to continue a run exactly: model
weights, optimizer moments, step counter, liveness buffers, the measured input
scale, and RNG state. Resume is the default; overwriting requires ``force``,
because silently discarding a paid-for training run is the wrong default when
compute is the scarce resource.

Metrics are appended to a JSONL on the Volume as training proceeds, so a run
that dies still leaves behind everything it measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from brainpatch.research.ml.activation_store import ActivationSubset
from brainpatch.research.ml.sae import TopKSAE, reconstruction_metrics
from brainpatch.paths import VolumePaths
from brainpatch.schemas.sae import SAEConfig


@dataclass
class TrainingState:
    """Mutable progress of a training run."""

    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Measured outcome of an SAE training run."""

    config: SAEConfig
    steps: int
    epochs: int
    seconds: float
    steps_per_second: float
    peak_vram_mb: float
    final_train: dict[str, float]
    final_val: dict[str, float]
    num_dead_features: int
    num_alive_features: int
    checkpoint_path: str
    resumed_from_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "seconds": round(self.seconds, 3),
            "steps_per_second": round(self.steps_per_second, 3),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "final_train": self.final_train,
            "final_val": self.final_val,
            "num_dead_features": self.num_dead_features,
            "num_alive_features": self.num_alive_features,
            "checkpoint": self.checkpoint_path,
            "resumed_from_step": self.resumed_from_step,
            "d_sae": self.config.d_sae,
            "k": self.config.k,
            "input_scale": self.config.input_scale,
        }


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training.

    Note: cuDNN kernel selection and atomics still make GPU training only
    *approximately* reproducible. This is documented rather than papered over --
    see the reproducibility section of the README.
    """
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> None:
    """Write a resumable checkpoint atomically.

    Written to a temporary file and renamed, so a crash mid-write cannot leave
    a truncated checkpoint where a valid one used to be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": sae.config.to_dict(),
        "state_dict": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "tokens_seen": state.tokens_seen,
        "best_val_loss": state.best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, device: str = "cpu") -> dict[str, Any] | None:
    """Load a checkpoint, or ``None`` if there isn't one."""
    if not path.is_file():
        return None
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def evaluate(sae: TopKSAE, data: torch.Tensor, batch_size: int = 1024) -> dict[str, float]:
    """Average reconstruction metrics over a held-out tensor."""
    sae.eval()
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size]
        if batch.shape[0] == 0:
            continue
        out = sae(batch)
        metrics = reconstruction_metrics(batch, out)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch.shape[0]
        count += batch.shape[0]
    sae.train()
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def train_sae(
    config: SAEConfig,
    subset: ActivationSubset,
    paths: VolumePaths,
    experiment: str,
    *,
    device: str = "cuda",
    force: bool = False,
    commit: Callable[[], None] | None = None,
    log_every: int = 25,
    checkpoint_every: int = 200,
    provenance: dict[str, Any] | None = None,
) -> TrainingResult:
    """Train a Top-K SAE on an activation corpus held in memory.

    Parameters
    ----------
    force:
        Start from scratch, discarding any existing checkpoint. Off by default:
        an accidental restart should resume, not waste the previous run.
    commit:
        Volume flush callback, invoked after each checkpoint.
    """
    config.validate()
    set_seed(config.seed)

    checkpoint_path = Path(paths.sae_checkpoint(experiment))
    metrics_path = Path(paths.sae_metrics(experiment))
    config_path = Path(paths.sae_config(experiment))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize inputs so that E[||x||] == sqrt(d_in). The measured scale is
    # stored in the config: any intervention must multiply by it to get back to
    # the raw residual-stream scale.
    if config.input_scale is None:
        config.input_scale = subset.input_scale()
    scale = float(config.input_scale)
    print(f"[sae] input_scale = {scale:.6g} (normalizes E[||x||] to sqrt(d_in))")

    train_raw, val_raw = subset.split(config.val_fraction, seed=config.seed)
    train_data = (train_raw * scale).to(device=device, dtype=torch.float32)
    val_data = (val_raw * scale).to(device=device, dtype=torch.float32)
    print(f"[sae] train rows: {train_data.shape[0]:,}   val rows: {val_data.shape[0]:,}")

    sae = TopKSAE(config).to(device)
    sae.set_decoder_bias_to_mean(train_data[: min(8192, train_data.shape[0])])
    sae.normalize_decoder()

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
    )
    state = TrainingState()

    existing = None if force else load_checkpoint(checkpoint_path, device=device)
    if existing is not None:
        _assert_checkpoint_compatible(existing["config"], config)
        sae.load_state_dict(existing["state_dict"])
        optimizer.load_state_dict(existing["optimizer"])
        state.step = int(existing.get("step", 0))
        state.epoch = int(existing.get("epoch", 0))
        state.tokens_seen = int(existing.get("tokens_seen", 0))
        state.best_val_loss = float(existing.get("best_val_loss", float("inf")))
        print(f"[sae] resuming from step {state.step} (epoch {state.epoch})")
    elif force and metrics_path.exists():
        metrics_path.unlink()

    resumed_from = state.step
    n_train = train_data.shape[0]
    steps_per_epoch = max(1, n_train // config.batch_size)
    total_steps = config.max_steps or (steps_per_epoch * config.epochs)

    if state.step >= total_steps:
        print(f"[sae] already trained for {state.step} >= {total_steps} steps; nothing to do")
        final_val = evaluate(sae, val_data)
        return _build_result(
            config, sae, state, 0.0, final_val, final_val, str(checkpoint_path), resumed_from
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cpu").manual_seed(config.seed + state.epoch)
    metrics_file = metrics_path.open("a", encoding="utf-8")
    last_train: dict[str, float] = {}

    print(f"[sae] training {total_steps} steps ({steps_per_epoch} per epoch)")
    start_time = time.perf_counter()
    sae.train()

    try:
        while state.step < total_steps:
            perm = torch.randperm(n_train, generator=generator)
            for i in range(steps_per_epoch):
                if state.step >= total_steps:
                    break
                idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
                batch = train_data[idx.to(train_data.device)]

                lr = _lr_at(config, state.step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                out = sae(batch)
                mse = torch.nn.functional.mse_loss(out.reconstruction, batch)
                aux = sae.auxk_loss(batch, out)
                loss = mse + config.auxk_alpha * aux

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                sae.project_decoder_grad()
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), config.grad_clip)
                optimizer.step()
                sae.normalize_decoder()
                # Values are required: a Top-K index whose value is zero is not
                # a firing, and counting it would hide a dead feature.
                sae.update_liveness(out.topk_indices, out.topk_values, batch.shape[0])

                state.step += 1
                state.tokens_seen += batch.shape[0]

                if state.step % log_every == 0 or state.step == total_steps:
                    last_train = reconstruction_metrics(batch, out)
                    record = {
                        "step": state.step,
                        "epoch": state.epoch,
                        "lr": lr,
                        "loss": float(loss.item()),
                        "mse": float(mse.item()),
                        "auxk": float(aux.item()),
                        "grad_norm": float(grad_norm),
                        "dead_features": sae.num_dead(),
                        "decoder_norm_mean": float(sae.decoder_norms().mean().item()),
                        "decoder_norm_std": float(sae.decoder_norms().std().item()),
                        **last_train,
                    }
                    state.history.append(record)
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    print(
                        f"[sae] step {state.step}/{total_steps} "
                        f"loss={loss.item():.5f} ev={last_train['explained_variance']:.4f} "
                        f"l0={last_train['l0']:.1f} dead={record['dead_features']}"
                    )

                if state.step % checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, sae, optimizer, state)
                    if commit is not None:
                        commit()

            state.epoch += 1
            generator = torch.Generator(device="cpu").manual_seed(config.seed + state.epoch)
    finally:
        metrics_file.close()

    elapsed = time.perf_counter() - start_time

    final_val = evaluate(sae, val_data)
    state.best_val_loss = min(state.best_val_loss, final_val.get("mse", float("inf")))
    save_checkpoint(checkpoint_path, sae, optimizer, state)
    config_path.write_text(config.to_json(), encoding="utf-8")

    summary = {
        "config": config.to_dict(),
        "steps": state.step,
        "epochs": state.epoch,
        "seconds": round(elapsed, 3),
        "final_val": final_val,
        "final_train": last_train,
        "provenance": dict(provenance or {}),
    }
    Path(paths.sae(experiment) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if commit is not None:
        commit()

    return _build_result(
        config, sae, state, elapsed, last_train, final_val, str(checkpoint_path), resumed_from
    )


def _build_result(
    config: SAEConfig,
    sae: TopKSAE,
    state: TrainingState,
    elapsed: float,
    final_train: dict[str, float],
    final_val: dict[str, float],
    checkpoint_path: str,
    resumed_from: int,
) -> TrainingResult:
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    dead = sae.num_dead()
    steps_done = state.step - resumed_from
    return TrainingResult(
        config=config,
        steps=state.step,
        epochs=state.epoch,
        seconds=elapsed,
        steps_per_second=steps_done / elapsed if elapsed > 0 else 0.0,
        peak_vram_mb=peak,
        final_train=final_train,
        final_val=final_val,
        num_dead_features=dead,
        num_alive_features=config.d_sae - dead,
        checkpoint_path=checkpoint_path,
        resumed_from_step=resumed_from,
    )


def _lr_at(config: SAEConfig, step: int, total_steps: int) -> float:
    """Linear warmup then linear decay to 10% of peak."""
    if step < config.lr_warmup_steps:
        return config.lr * (step + 1) / max(1, config.lr_warmup_steps)
    progress = (step - config.lr_warmup_steps) / max(1, total_steps - config.lr_warmup_steps)
    return config.lr * (1.0 - 0.9 * min(1.0, progress))


def _assert_checkpoint_compatible(saved: dict[str, Any], current: SAEConfig) -> None:
    """Refuse to resume into an architecture that no longer matches."""
    mismatches = [
        f"{key}: checkpoint={saved.get(key)} config={getattr(current, key)}"
        for key in ("d_in", "d_sae", "k")
        if saved.get(key) != getattr(current, key)
    ]
    if mismatches:
        raise ValueError(
            "cannot resume SAE training: architecture changed.\n  "
            + "\n  ".join(mismatches)
            + "\nUse a new experiment name, or pass force=True to retrain from scratch."
        )
