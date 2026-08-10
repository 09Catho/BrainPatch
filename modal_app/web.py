"""Modal-hosted Gradio demo.

Deliberately **not** deployed during v0. ``serve_demo`` is an on-demand
``modal serve`` / ``modal run`` target rather than a ``modal deploy`` target,
because a permanently-warm L4 inference service would consume the entire
project budget in a few days for no research value.

The container holds the model and SAE across requests via ``@modal.enter``, so
a session pays the ~30 s cold start once rather than per request; the short
``scaledown_window`` then releases the GPU as soon as the session goes idle.

Layout:

* left   -- prompt and generation settings
* centre -- baseline vs patched output side by side
* right  -- installed patches, toggles, strength sliders, schedule
* tabs   -- Compare / Feature Explorer / Patch Inspector
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# It would turn `experiment: str = modal.parameter(...)` into the *string*
# annotation "str", which Modal's class-parameter validator cannot resolve
# (it fails with `'str' object has no attribute '__name__'`). Python 3.11
# evaluates the remaining PEP 604 unions in this file natively, so the future
# import buys nothing here anyway.
import modal

from modal_app.image import WEB_IMAGE
from modal_app.resources import VOL_MOUNT, VOLUMES, app

#: Idle GPU is pure cost, so release it quickly.
_SCALEDOWN_SECONDS = 120


@app.cls(
    image=WEB_IMAGE,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 30,
    scaledown_window=_SCALEDOWN_SECONDS,
    max_containers=1,
)
class DemoServer:
    """Holds the model and SAE warm for the lifetime of one container."""

    experiment: str = modal.parameter(default="smoke_v0")

    @modal.enter()
    def load(self) -> None:
        from brainpatch.research.ml.runtime import BrainPatchedModel
        from brainpatch.paths import VolumePaths

        import torch

        paths = VolumePaths(VOL_MOUNT)
        checkpoint = str(paths.sae_checkpoint(self.experiment))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = saved["config"]

        self.model = BrainPatchedModel.from_pretrained(
            config["model"], revision=config.get("model_revision") or None, device="cuda"
        )
        self.model.load_sae(checkpoint, reference=self.experiment)
        self.paths = paths
        print(f"[demo] loaded {config['model']} + SAE {self.experiment}")

    @modal.method()
    def compare(
        self,
        prompt: str,
        feature_id: int,
        strength: float,
        max_new_tokens: int = 128,
        schedule: dict[str, float] | None = None,
    ) -> dict:
        """Generate baseline and patched completions for the same prompt."""
        from brainpatch.evaluation.metrics import compare_generations
        from brainpatch.research.ml.generation import GenerationConfig
        from brainpatch.steering.schedule import StrengthSchedule

        cfg = GenerationConfig(max_new_tokens=max_new_tokens)

        self.model.plan.patches = {}
        baseline = self.model.generate(prompt, config=cfg)

        layer = int(self.model.sae.config.layer)
        self.model.add_feature(
            layer=layer, feature_id=feature_id, strength=strength, name="demo"
        )
        if schedule:
            self.model.set_patch_schedule(
                "demo", StrengthSchedule({int(k): float(v) for k, v in schedule.items()})
            )
        patched = self.model.generate(prompt, config=cfg)
        stats = self.model.last_steering_stats
        self.model.plan.patches = {}

        return {
            "baseline": baseline,
            "patched": patched,
            "steering_stats": stats,
            "comparison": compare_generations(baseline, patched),
        }

    @modal.method()
    def feature_info(self, feature_id: int) -> dict:
        """Statistics and top-activating contexts for one feature."""
        import json
        from pathlib import Path

        path = Path(self.paths.features_jsonl(self.experiment))
        if not path.is_file():
            return {"error": f"no feature database for {self.experiment}"}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["feature_id"] == feature_id:
                    row["disclaimer"] = (
                        "Top-activating contexts are correlational evidence. "
                        "They do not establish what this feature does."
                    )
                    return row
        return {"error": f"feature {feature_id} not found"}


@app.function(image=WEB_IMAGE, volumes=VOLUMES, timeout=60 * 60, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve_demo():
    """Gradio UI. Run with `modal serve modal_app/web.py` -- do not deploy in v0."""
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    server = DemoServer(experiment="smoke_v0")

    def do_compare(prompt, feature_id, strength, max_new_tokens, use_schedule, on_at):
        schedule = {"0": 0.0, str(int(on_at)): 1.0} if use_schedule else None
        out = server.compare.remote(
            prompt, int(feature_id), float(strength), int(max_new_tokens), schedule
        )
        stats = out["steering_stats"]
        info = (
            f"delta norm (mean): {stats.get('mean_delta_norm', 0):.4f}  |  "
            f"applied on {stats.get('applied_passes', 0)}/{stats.get('forward_passes', 0)} passes  |  "
            f"3-gram overlap with baseline: {out['comparison']['jaccard_3']:.3f}"
        )
        return out["baseline"], out["patched"], info

    def do_feature(feature_id):
        info = server.feature_info.remote(int(feature_id))
        if "error" in info:
            return info["error"]
        stats = info["stats"]
        lines = [
            f"# Feature {info['feature_id']}",
            "",
            f"- firing rate: {stats['firing_rate']:.5f}  ({stats['fire_count']} of {stats['total_tokens']} tokens)",
            f"- mean activation (when firing): {stats['mean_activation']:.4f}",
            f"- max activation: {stats['max_activation']:.4f}",
            f"- decoder norm: {stats['decoder_norm']:.4f}",
            f"- evidence level: **{info['evidence_level']}**",
            "",
            "## Top activating contexts",
            "",
        ]
        for c in info["top_contexts"]:
            lines.append(
                f"- `{c['activation']:.3f}`  ...{c['context_before']}**[{c['token_text']}]**{c['context_after']}..."
            )
        lines += ["", f"> {info['disclaimer']}"]
        return "\n".join(lines)

    with gr.Blocks(title="BrainPatch") as blocks:
        gr.Markdown(
            "# BrainPatch\n"
            "Activation-space interventions on a frozen Qwen2.5-1.5B-Instruct.\n\n"
            "> Feature labels are hypotheses. Steering a direction does not mean "
            "the direction *is* the concept it appears to track."
        )
        with gr.Tab("Compare"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", lines=4, value="Explain why the sky is blue.")
                    max_new = gr.Slider(16, 256, value=128, step=16, label="max new tokens")
                with gr.Column(scale=1):
                    feature = gr.Number(label="feature id", value=0, precision=0)
                    strength = gr.Slider(-10, 10, value=4.0, step=0.5, label="strength")
                    use_schedule = gr.Checkbox(label="dynamic schedule", value=False)
                    on_at = gr.Slider(0, 128, value=24, step=4, label="turn on at token")
                    go = gr.Button("Generate", variant="primary")
            with gr.Row():
                baseline_out = gr.Textbox(label="Baseline", lines=12)
                patched_out = gr.Textbox(label="Patched", lines=12)
            stats_out = gr.Markdown()
            go.click(
                do_compare,
                [prompt, feature, strength, max_new, use_schedule, on_at],
                [baseline_out, patched_out, stats_out],
            )

        with gr.Tab("Feature Explorer"):
            fid = gr.Number(label="feature id", value=0, precision=0)
            look = gr.Button("Inspect")
            detail = gr.Markdown()
            look.click(do_feature, [fid], [detail])

        with gr.Tab("Patch Inspector"):
            gr.Markdown(
                "Load a BrainPatch JSON file to view its declared base model, "
                "SAE reference, layer, feature edits and evidence level.\n\n"
                "A patch that declares `evidence_level: none` has no measured "
                "behavioural effect behind its name."
            )

    return mount_gradio_app(app=FastAPI(), blocks=blocks, path="/")
