"""The ``brainpatch`` command-line interface.

Design constraint that shapes everything here: **no local command may execute a
language model.** The local machine is a source-editing and Modal control-plane
machine. Commands split into two kinds:

*Local, lightweight* -- ``list``, ``inspect``, ``validate``, ``compare``,
``contrast``, ``paths``. These only read JSON and print. They need no ML stack.

*Remote* -- everything under ``brainpatch modal``. These construct and run a
``modal run`` command. They never import torch and never download weights.

``brainpatch run`` deliberately does **not** run a model locally. It refuses and
points at ``brainpatch modal run``, so the mistake is loud rather than silent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from brainpatch import __version__
from brainpatch.datasets import list_contrast_sets, load_contrast_set
from brainpatch.paths import MODAL_ENVIRONMENT, VOLUME_NAME, VolumePaths
from brainpatch.patches.io import load_patch, load_patch_dir
from brainpatch.schemas.patch import PatchValidationError

console = Console()

app = typer.Typer(
    name="brainpatch",
    help="Tiny, inspectable activation-space behavioural patches for frozen LLMs.",
    no_args_is_help=True,
    add_completion=False,
)
modal_app = typer.Typer(help="Submit work to Modal. Nothing heavy runs locally.", no_args_is_help=True)
app.add_typer(modal_app, name="modal")

DEFAULT_PATCH_DIR = "patches"
MODAL_ENTRYPOINT = "modal_app/app.py"


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"brainpatch {__version__}")
        raise typer.Exit()
    # invoke_without_command=True is needed for --version to be reachable, so
    # restore the "no arguments shows help" behaviour by hand.
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------
# local commands
# ---------------------------------------------------------------------------


@app.command("list")
def list_patches(
    directory: str = typer.Option(DEFAULT_PATCH_DIR, "--dir", "-d", help="Patch directory."),
) -> None:
    """List available patches and their evidence level."""
    specs, failures = load_patch_dir(directory, strict=False)
    if not specs and not failures:
        console.print(f"[yellow]No patches found in {directory}[/yellow]")
        raise typer.Exit()

    table = Table(title=f"BrainPatches in {directory}")
    table.add_column("name")
    table.add_column("base model")
    table.add_column("layer", justify="right")
    table.add_column("features", justify="right")
    table.add_column("evidence")
    for spec in specs:
        evidence = spec.evidence_level
        colour = "green" if spec.is_validated else "yellow"
        table.add_row(
            spec.name,
            spec.base_model,
            str(spec.sae.layer),
            str(len(spec.features)),
            f"[{colour}]{evidence}[/{colour}]",
        )
    console.print(table)

    for path, message in failures:
        console.print(f"[red]invalid[/red] {path}: {message}")


@app.command()
def inspect(patch: str = typer.Argument(..., help="Path to a .json patch file.")) -> None:
    """Show a patch's full contents and what it claims."""
    try:
        spec = load_patch(patch)
    except (PatchValidationError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{spec.name}[/bold]  ({spec.format_version})")
    console.print(spec.description or "[dim]no description[/dim]")
    console.print()

    table = Table(show_header=False, box=None)
    table.add_row("base model", spec.base_model)
    table.add_row("model revision", spec.model_revision or "[dim]unpinned[/dim]")
    table.add_row("SAE", f"{spec.sae.reference} (d_sae={spec.sae.d_sae}, d_in={spec.sae.d_in})")
    table.add_row("layer / hook", f"{spec.sae.layer} / {spec.sae.hook}")
    table.add_row("input scale", str(spec.sae.input_scale))
    table.add_row("license", spec.license)
    table.add_row("evidence level", spec.evidence_level)
    console.print(table)

    console.print("\n[bold]feature edits[/bold]")
    for edit in spec.features:
        console.print(f"  #{edit.feature_id}  strength={edit.strength:+.3f}  mode={edit.mode}")

    if spec.schedule:
        console.print("\n[bold]schedule[/bold] (generated-token index -> multiplier)")
        for step, value in sorted(spec.schedule.items(), key=lambda kv: int(kv[0])):
            console.print(f"  {step}: {value}")

    if spec.evaluation:
        console.print("\n[bold]recorded evaluation[/bold]")
        console.print_json(json.dumps(spec.evaluation))
    else:
        console.print("\n[yellow]No evaluation recorded: this patch has no measured effect.[/yellow]")

    if not spec.is_validated:
        console.print(
            f"\n[yellow]This patch's evidence level is '{spec.evidence_level}'. "
            "Its name is a label, not a validated behavioural claim.[/yellow]"
        )


@app.command()
def validate(
    patch: str = typer.Argument(..., help="Patch file, or a directory of them."),
) -> None:
    """Validate patch files against the schema."""
    target = Path(patch)
    paths = sorted(target.glob("*.json")) if target.is_dir() else [target]
    if not paths:
        console.print(f"[yellow]nothing to validate at {patch}[/yellow]")
        raise typer.Exit()

    failed = 0
    for path in paths:
        try:
            spec = load_patch(path)
            console.print(f"[green]ok[/green]      {path.name}: {spec.summary()}")
        except (PatchValidationError, FileNotFoundError) as exc:
            failed += 1
            console.print(f"[red]invalid[/red] {path.name}: {exc}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def compare(
    left: str = typer.Argument(..., help="First patch file."),
    right: str = typer.Argument(..., help="Second patch file."),
) -> None:
    """Compare two patches field by field."""
    try:
        a, b = load_patch(left), load_patch(right)
    except (PatchValidationError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"{a.name}  vs  {b.name}")
    table.add_column("field")
    table.add_column(a.name)
    table.add_column(b.name)

    rows = [
        ("base model", a.base_model, b.base_model),
        ("revision", a.model_revision or "-", b.model_revision or "-"),
        ("SAE", a.sae.reference, b.sae.reference),
        ("layer", str(a.sae.layer), str(b.sae.layer)),
        ("d_sae", str(a.sae.d_sae), str(b.sae.d_sae)),
        ("evidence", a.evidence_level, b.evidence_level),
        ("num features", str(len(a.features)), str(len(b.features))),
    ]
    for field, left_value, right_value in rows:
        style = "" if left_value == right_value else "yellow"
        table.add_row(field, f"[{style}]{left_value}[/{style}]" if style else left_value,
                      f"[{style}]{right_value}[/{style}]" if style else right_value)
    console.print(table)

    a_features = {e.feature_id: e.strength for e in a.features}
    b_features = {e.feature_id: e.strength for e in b.features}
    shared = sorted(set(a_features) & set(b_features))
    console.print(f"\nshared features: {shared or 'none'}")
    console.print(f"only in {a.name}: {sorted(set(a_features) - set(b_features)) or 'none'}")
    console.print(f"only in {b.name}: {sorted(set(b_features) - set(a_features)) or 'none'}")

    if a.base_model != b.base_model or a.sae.reference != b.sae.reference:
        console.print(
            "\n[yellow]These patches target different models or SAEs. "
            "Their feature IDs are not comparable.[/yellow]"
        )


@app.command()
def contrast(
    name: Optional[str] = typer.Argument(None, help="Contrast set name; omit to list."),
) -> None:
    """Inspect the synthetic behavioural contrast fixtures."""
    if name is None:
        available = list_contrast_sets()
        console.print("available contrast sets: " + (", ".join(available) or "none"))
        raise typer.Exit()

    try:
        contrast_set = load_contrast_set(name)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{contrast_set.name}[/bold] -- {len(contrast_set)} examples")
    console.print(contrast_set.description)
    console.print(f"\ncategories: {', '.join(contrast_set.categories())}")
    if contrast_set.synthetic:
        console.print(
            "\n[yellow]This is a synthetic development fixture, not a benchmark.[/yellow]"
        )
    example = contrast_set.examples[0]
    console.print(f"\n[bold]example[/bold]\nprompt: {example.prompt}")
    console.print(f"[green]positive:[/green] {example.positive_response[:200]}")
    console.print(f"[red]negative:[/red] {example.negative_response[:200]}")


@app.command()
def paths(experiment: str = typer.Option("smoke_v0", help="Experiment name.")) -> None:
    """Show where an experiment's artifacts live on the Modal Volume."""
    p = VolumePaths()
    console.print(f"[bold]Modal environment[/bold] {MODAL_ENVIRONMENT}")
    console.print(f"[bold]Volume[/bold] {VOLUME_NAME} mounted at {p.root}\n")
    table = Table(show_header=False, box=None)
    table.add_row("activations", str(p.activations(experiment)))
    table.add_row("manifest", str(p.activation_manifest(experiment)))
    table.add_row("examples", str(p.activation_examples(experiment)))
    table.add_row("SAE checkpoint", str(p.sae_checkpoint(experiment)))
    table.add_row("SAE metrics", str(p.sae_metrics(experiment)))
    table.add_row("feature db", str(p.features_jsonl(experiment)))
    table.add_row("experiment", str(p.experiment(experiment)))
    console.print(table)
    console.print(
        "\n[dim]These paths exist inside Modal containers. Nothing is stored locally.[/dim]"
    )


@app.command("run")
def run_local() -> None:
    """Refused: model execution never happens on the local machine."""
    console.print(
        "[red]`brainpatch run` does not execute models locally.[/red]\n\n"
        "BrainPatch treats this machine as a source-editing and Modal control-plane\n"
        "machine. Running a patched model requires a GPU, the ML stack, and the\n"
        "base-model weights -- none of which belong here.\n\n"
        "Use the remote entry points instead:\n\n"
        "  brainpatch modal run intervention_smoke --experiment smoke_v0\n"
        "  brainpatch modal pipeline\n"
        "  brainpatch modal demo\n"
    )
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# modal subcommands
# ---------------------------------------------------------------------------


def _modal_available() -> str:
    binary = shutil.which("modal")
    if binary is None:
        console.print(
            "[red]The `modal` CLI is not on PATH.[/red]\n"
            "Install the control-plane extra: pip install 'brainpatch[modal]'"
        )
        raise typer.Exit(code=127)
    return binary


def _invoke(args: list[str], *, dry_run: bool) -> None:
    printable = " ".join(args)
    console.print(f"[dim]$ {printable}[/dim]")
    if dry_run:
        return
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@modal_app.command("run")
def modal_run(
    function: str = typer.Argument(..., help="Modal function, e.g. gpu_info or train_sae."),
    extra: list[str] = typer.Argument(None, help="Extra --flags passed to the function."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running it."),
) -> None:
    """Run one BrainPatch Modal Function remotely."""
    binary = _modal_available()
    args = [binary, "run", f"{MODAL_ENTRYPOINT}::{function}", *(extra or [])]
    _invoke(args, dry_run=dry_run)


@modal_app.command("pipeline")
def modal_pipeline(
    experiment: str = typer.Option("smoke_v0"),
    layer: int = typer.Option(18),
    target_tokens: int = typer.Option(20_000),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the full smoke pipeline on Modal."""
    binary = _modal_available()
    args = [
        binary, "run", f"{MODAL_ENTRYPOINT}::smoke_pipeline",
        "--experiment", experiment,
        "--layer", str(layer),
        "--target-tokens", str(target_tokens),
    ]
    _invoke(args, dry_run=dry_run)


@modal_app.command("demo")
def modal_demo(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Serve the Gradio demo from Modal (ephemeral; not deployed)."""
    binary = _modal_available()
    _invoke([binary, "serve", "modal_app/web.py"], dry_run=dry_run)


@modal_app.command("status")
def modal_status(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Show Modal auth, environment, volume and secret state."""
    binary = _modal_available()
    for args in (
        [binary, "token", "info"],
        [binary, "environment", "list"],
        [binary, "volume", "list"],
        [binary, "secret", "list"],
    ):
        _invoke(args, dry_run=dry_run)


@modal_app.command("volume")
def modal_volume(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Report what is stored on the BrainPatch Volume."""
    binary = _modal_available()
    _invoke([binary, "run", f"{MODAL_ENTRYPOINT}::volume_report"], dry_run=dry_run)


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
