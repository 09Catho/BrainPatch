"""Local Gradio UI.

Runs entirely on the user's machine: their backend, their model, their installed
patches. No hosted service, no Modal, no telemetry.

The UI deliberately shows each patch's ``evidence_level`` next to its slider. If
a patch has no validated behavioural effect, the interface says so rather than
implying that moving the slider does something known.
"""

from __future__ import annotations

from typing import Any

BASELINE_LABEL = "BASELINE (patch off)"
PATCHED_LABEL = "PATCHED"


def launch(
    *,
    model: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Build and serve the UI."""
    import gradio as gr

    from brainpatch.patch.registry import default_registry
    from brainpatch.runtime.auto import available_backends
    from brainpatch.runtime.base import GenerationConfig
    from brainpatch.runtime.model import BrainPatchedModel

    registry = default_registry()
    state: dict[str, Any] = {"model": None, "model_id": None}

    def installed_names() -> list[str]:
        return [p.name for p in registry.list_patches()]

    def ensure_model(model_id: str, backend_name: str) -> BrainPatchedModel:
        if state["model"] is None or state["model_id"] != (model_id, backend_name):
            if state["model"] is not None:
                state["model"].unload()
            state["model"] = BrainPatchedModel.from_pretrained(
                model_id, backend=backend_name, device=device
            )
            state["model_id"] = (model_id, backend_name)
        return state["model"]

    def run_compare(
        model_id: str,
        backend_name: str,
        patch_name: str,
        prompt: str,
        strength: float,
        max_tokens: int,
        temperature: float,
        use_schedule: bool,
        turn_on_at: int,
    ) -> tuple[str, str, str]:
        if not model_id:
            return "", "", "Enter a model id first."
        if not prompt.strip():
            return "", "", "Enter a prompt."
        try:
            patched = ensure_model(model_id, backend_name)
        except Exception as exc:  # noqa: BLE001
            return "", "", f"Could not load model: {exc}"

        for name in list(patched.list_patches()):
            patched.remove_patch(name)

        info_bits = []
        if patch_name and patch_name != "(none)":
            try:
                handle = patched.install(patch_name, strength=strength)
                if use_schedule:
                    handle.schedule = {0: 0.0, int(turn_on_at): 1.0}
                info_bits.append(
                    f"patch **{handle.name}** · strength {handle.strength} · "
                    f"evidence `{handle.evidence_level}`"
                )
                if handle.evidence_level in {"none", "correlational"}:
                    info_bits.append(
                        "⚠️ This patch has **no validated behavioural effect**. "
                        "Any difference below is an activation perturbation, not a "
                        "demonstrated capability."
                    )
            except Exception as exc:  # noqa: BLE001
                return "", "", f"Could not install patch: {exc}"

        cfg = GenerationConfig(max_new_tokens=int(max_tokens), temperature=float(temperature))
        try:
            result = patched.compare(prompt, cfg)
        except Exception as exc:  # noqa: BLE001
            return "", "", f"Generation failed: {exc}"

        if result["baseline"] == result["patched"]:
            info_bits.append("Outputs are identical at this strength on this prompt.")
        return result["baseline"], result["patched"], "\n\n".join(info_bits)

    def patch_details(name: str) -> str:
        if not name or name == "(none)":
            return "Select a patch."
        try:
            loaded = registry.get(name).load()
        except Exception as exc:  # noqa: BLE001
            return f"Could not read patch: {exc}"
        manifest = loaded.manifest
        lines = [
            f"# {manifest.name}",
            "",
            manifest.description or "_no description_",
            "",
            f"- base model: `{manifest.base_model.model_id}`",
            f"- revision: `{manifest.base_model.revision or 'unpinned'}`",
            f"- layers: {manifest.layers}",
            f"- size: {loaded.archive_bytes / 1024:.1f} KB",
            f"- evidence level: **{manifest.evidence_level}**",
            "",
            "## Backend compatibility",
        ]
        if manifest.compatibility:
            for backend_name, entry in sorted(manifest.compatibility.items()):
                lines.append(f"- `{backend_name}`: **{entry.get('status', 'unsupported')}**")
        else:
            lines.append("- none recorded")
        if manifest.evaluation:
            lines += ["", "## Recorded evaluation", "```json", _pretty(manifest.evaluation), "```"]
        else:
            lines += ["", "_No evaluation recorded: this patch has no measured effect._"]
        return "\n".join(lines)

    def backend_status() -> str:
        rows = ["| backend | status | detail |", "|---|---|---|"]
        for status in available_backends():
            mark = "available" if status.available else "unavailable"
            rows.append(f"| `{status.name}` | {mark} | {status.detail} |")
        return "\n".join(rows)

    with gr.Blocks(title="BrainPatch") as blocks:
        gr.Markdown(
            "# BrainPatch\n"
            "Tiny activation patches for frozen language models. "
            "Everything here runs locally on your machine.\n\n"
            "> Feature directions are not concepts. A patch changes activations; "
            "whether that produces a *specific* behaviour is an empirical question "
            "answered by each patch's evidence level."
        )

        with gr.Tab("Compare"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_box = gr.Textbox(label="Model", value=model or "", placeholder="Qwen/Qwen2.5-1.5B-Instruct")
                    prompt_box = gr.Textbox(label="Prompt", lines=5, value="Evaluate my idea.")
                    with gr.Row():
                        max_tokens = gr.Slider(16, 512, value=128, step=16, label="max new tokens")
                        temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="temperature")
                with gr.Column(scale=1):
                    backend_box = gr.Dropdown(
                        ["auto", "transformers", "llamacpp", "vllm", "mlx"],
                        value=backend,
                        label="backend",
                    )
                    patch_box = gr.Dropdown(
                        ["(none)"] + installed_names(), value="(none)", label="patch"
                    )
                    strength_box = gr.Slider(-8, 8, value=1.0, step=0.1, label="strength")
                    schedule_box = gr.Checkbox(label="dynamic schedule", value=False)
                    turn_on_box = gr.Slider(0, 128, value=24, step=4, label="turn on at token")
                    go = gr.Button("Generate", variant="primary")
            with gr.Row():
                baseline_out = gr.Textbox(label=BASELINE_LABEL, lines=14)
                patched_out = gr.Textbox(label=PATCHED_LABEL, lines=14)
            info_out = gr.Markdown()
            go.click(
                run_compare,
                [model_box, backend_box, patch_box, prompt_box, strength_box,
                 max_tokens, temperature, schedule_box, turn_on_box],
                [baseline_out, patched_out, info_out],
            )

        with gr.Tab("Patch Inspector"):
            inspect_box = gr.Dropdown(
                ["(none)"] + installed_names(), value="(none)", label="installed patch"
            )
            details = gr.Markdown()
            inspect_box.change(patch_details, [inspect_box], [details])

        with gr.Tab("Backends"):
            gr.Markdown(backend_status())

    blocks.launch(server_name=host, server_port=port, share=share)


def _pretty(data: Any) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)
