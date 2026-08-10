"""The ``brainpatch`` command-line interface.

Product-first. The commands a *user* needs -- install, list, inspect, run, chat,
compare, serve, doctor -- work with only the core package plus whichever
inference backend they chose. Nothing here requires Modal, a hosted service, or
network access once a patch and model are local.

Backends are imported lazily, so ``brainpatch list`` and ``brainpatch inspect``
run instantly on a machine with no ML stack at all. ``brainpatch doctor`` is
built to work in exactly that situation, since its whole job is reporting what
is and is not installed.

Research commands live under ``brainpatch research`` and are documented as
requiring the ``research`` extra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from brainpatch import __version__

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="brainpatch",
    help="Tiny, installable activation patches for frozen language models.",
    no_args_is_help=True,
    add_completion=False,
)
research_app = typer.Typer(
    help="Patch-authoring tools. Requires: pip install 'brainpatch[research]'",
    no_args_is_help=True,
)
app.add_typer(research_app, name="research")


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=code)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"brainpatch {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------
# patch management
# ---------------------------------------------------------------------------


@app.command()
def install(
    ref: str = typer.Argument(..., help="Path to a .brainpatch file, or a 'owner/repo' HF reference."),
    force: bool = typer.Option(False, "--force", help="Replace an already-installed patch."),
    offline: bool = typer.Option(False, "--offline", help="Refuse any network access."),
) -> None:
    """Install a patch into the local registry (~/.brainpatch).

    Downloads only the patch artifact -- never the base model.
    """
    from brainpatch.patch.registry import RegistryError, default_registry

    registry = default_registry()
    try:
        installed = registry.install(ref, overwrite=force, offline=offline)
    except RegistryError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not install {ref!r}: {exc}")

    loaded = installed.load()
    manifest = loaded.manifest
    console.print(f"[green]Installed:[/green] [bold]{installed.name}[/bold]")
    console.print(f"  model:    {manifest.base_model.model_id}")
    console.print(f"  size:     {installed.size_bytes / 1024:.1f} KB")
    console.print(f"  evidence: {_evidence_markup(manifest.evidence_level)}")
    if manifest.evidence_level in {"none", "correlational"}:
        console.print(
            "  [yellow]This patch has no validated behavioural effect. "
            "Its name is a label, not a claim.[/yellow]"
        )


@app.command()
def uninstall(name: str = typer.Argument(..., help="Installed patch name.")) -> None:
    """Remove a patch from the local registry."""
    from brainpatch.patch.registry import RegistryError, default_registry

    try:
        default_registry().uninstall(name)
    except RegistryError as exc:
        _fail(str(exc))
    console.print(f"[green]Uninstalled[/green] {name}")


@app.command("list")
def list_command(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List installed patches."""
    from brainpatch.patch.registry import default_registry

    installed = default_registry().list_patches()
    if json_output:
        rows = []
        for item in installed:
            loaded = item.load()
            rows.append({**loaded.describe(), "installed_bytes": item.size_bytes})
        console.print_json(json.dumps(rows))
        return

    if not installed:
        console.print("No patches installed.")
        console.print("\n  brainpatch install <file.brainpatch>")
        console.print("  brainpatch install owner/repo")
        return

    table = Table(title="Installed BrainPatches")
    table.add_column("name")
    table.add_column("base model")
    table.add_column("layers", justify="right")
    table.add_column("size", justify="right")
    table.add_column("evidence")
    for item in installed:
        manifest = item.load().manifest
        table.add_row(
            item.name,
            manifest.base_model.model_id,
            ",".join(str(layer) for layer in manifest.layers),
            f"{item.size_bytes / 1024:.1f} KB",
            _evidence_markup(manifest.evidence_level),
        )
    console.print(table)


@app.command()
def inspect(
    patch: str = typer.Argument(..., help="Installed name or path to a .brainpatch file."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show everything a patch declares, including what it does not claim."""
    loaded = _load_patch_arg(patch)
    manifest = loaded.manifest

    if json_output:
        console.print_json(json.dumps(manifest.to_dict()))
        return

    console.print(
        Panel(
            manifest.description or "[dim]no description[/dim]",
            title=f"[bold]{manifest.name}[/bold]  (format {manifest.format_version})",
        )
    )

    table = Table(show_header=False, box=None)
    spec = manifest.base_model
    table.add_row("base model", spec.model_id)
    table.add_row("revision", spec.revision or "[dim]unpinned[/dim]")
    table.add_row("architecture", spec.architecture or "[dim]unknown[/dim]")
    table.add_row("geometry", f"hidden {spec.hidden_size}, {spec.num_layers} layers")
    table.add_row("evidence", _evidence_markup(manifest.evidence_level))
    table.add_row("strength", f"default {manifest.default_strength}, max ±{manifest.max_abs_strength}")
    table.add_row("license", manifest.license)
    table.add_row("size", f"{loaded.archive_bytes / 1024:.1f} KB")
    console.print(table)

    console.print("\n[bold]interventions[/bold]")
    for item in manifest.interventions:
        tensor = loaded.vectors.get(item.vector)
        dims = tensor.shape[0] if tensor else "?"
        console.print(
            f"  L{item.layer:<3} {item.hook:<14} {item.vector:<12} "
            f"coefficient={item.coefficient:+.4f}  ({dims}-d {tensor.dtype if tensor else ''})"
        )

    if manifest.compatibility:
        console.print("\n[bold]backend compatibility[/bold]")
        for backend, entry in sorted(manifest.compatibility.items()):
            status = str(entry.get("status", "unsupported"))
            colour = {"verified": "green", "experimental": "yellow"}.get(status, "red")
            extra = {k: v for k, v in entry.items() if k != "status"}
            suffix = f"  {extra}" if extra else ""
            console.print(f"  {backend:<14} [{colour}]{status}[/{colour}]{suffix}")
    else:
        console.print("\n[yellow]No backend compatibility recorded.[/yellow]")

    if manifest.evaluation:
        console.print("\n[bold]recorded evaluation[/bold]")
        console.print_json(json.dumps(manifest.evaluation))
    else:
        console.print("\n[yellow]No evaluation recorded: this patch has no measured effect.[/yellow]")

    if manifest.provenance:
        console.print("\n[bold]provenance[/bold] [dim](research metadata; unused at runtime)[/dim]")
        console.print_json(json.dumps(manifest.provenance))


@app.command()
def validate(
    patch: str = typer.Argument(..., help="Installed name or path to a .brainpatch file."),
    model: Optional[str] = typer.Option(None, "--model", help="Check against a loaded model too."),
    backend: str = typer.Option("auto", "--backend"),
    mode: str = typer.Option("strict", "--mode", help="strict | architecture | unsafe"),
) -> None:
    """Validate a patch archive or a directory of patches."""
    target = Path(patch).expanduser()
    if target.is_dir():
        # A directory may hold runtime artifacts, research patches, or both.
        from brainpatch.patch.loader import PatchLoadError
        from brainpatch.patch.loader import load_patch as load_runtime
        from brainpatch.schemas.patch import PatchValidationError
        from brainpatch.schemas.patch_io import load_patch as load_research

        files = sorted(list(target.glob("*.brainpatch")) + list(target.glob("*.json")))
        if not files:
            console.print(f"[yellow]no patches found in {target}[/yellow]")
            raise typer.Exit()
        failed = 0
        for path in files:
            try:
                if path.suffix == ".brainpatch":
                    console.print(f"[green]ok[/green]      {path.name}: {load_runtime(path).manifest.summary()}")
                else:
                    console.print(f"[green]ok[/green]      {path.name}: {load_research(path).summary()}")
            except (PatchLoadError, PatchValidationError, OSError) as exc:
                failed += 1
                console.print(f"[red]invalid[/red] {path.name}: {exc}")
        if failed:
            raise typer.Exit(code=1)
        return

    loaded = _load_patch_arg(patch)
    console.print(f"[green]ok[/green]  archive, checksums and manifest are valid")
    console.print(f"     {loaded.manifest.summary()}")

    if model is None:
        return

    from brainpatch.runtime.model import BrainPatchedModel

    try:
        patched = BrainPatchedModel.from_pretrained(model, backend=backend)
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not load {model!r}: {exc}")
    report = patched.backend.validate_patch(loaded, mode=mode)  # type: ignore[arg-type]
    for warning in report.warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")
    if report.ok:
        console.print(f"[green]ok[/green]  compatible with {model} in '{mode}' mode")
    else:
        for error in report.errors:
            console.print(f"[red]incompatible[/red] {error}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to complete."),
    model: str = typer.Option(..., "--model", "-m", help="Base model id or GGUF path."),
    patch: list[str] = typer.Option([], "--patch", "-p", help="Patch(es) to apply. Repeatable."),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    strength: Optional[float] = typer.Option(None, "--strength", "-s"),
    max_new_tokens: int = typer.Option(128, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    mode: str = typer.Option("strict", "--compatibility", help="strict | architecture | unsafe"),
) -> None:
    """Generate a completion with patches applied."""
    model_obj, _ = _load_model_with_patches(
        model, patch, backend, device, strength, mode
    )
    from brainpatch.runtime.base import GenerationConfig

    cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    console.print(model_obj.generate(prompt, cfg))


@app.command()
def compare(
    model: str = typer.Option(..., "--model", "-m"),
    patch: list[str] = typer.Option(..., "--patch", "-p", help="Patch(es) to apply."),
    prompt: str = typer.Option(..., "--prompt"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    strength: Optional[float] = typer.Option(None, "--strength", "-s"),
    max_new_tokens: int = typer.Option(128, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    mode: str = typer.Option("strict", "--compatibility"),
) -> None:
    """Generate the same prompt with the patch off and on, side by side."""
    model_obj, _ = _load_model_with_patches(model, patch, backend, device, strength, mode)
    from brainpatch.runtime.base import GenerationConfig

    cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    result = model_obj.compare(prompt, cfg)

    console.print(Panel(result["baseline"], title="[bold]BASELINE[/bold]", border_style="dim"))
    console.print(Panel(result["patched"], title="[bold]PATCHED[/bold]", border_style="cyan"))
    if result["baseline"] == result["patched"]:
        console.print(
            "[yellow]Outputs are identical.[/yellow] The patch had no effect at this "
            "strength on this prompt -- try raising --strength, but check the patch's "
            "measured dose-response first."
        )


@app.command()
def chat(
    model: str = typer.Option(..., "--model", "-m"),
    patch: list[str] = typer.Option([], "--patch", "-p"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    strength: Optional[float] = typer.Option(None, "--strength", "-s"),
    max_new_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    mode: str = typer.Option("strict", "--compatibility"),
) -> None:
    """Interactive chat. Type /help for in-session commands."""
    model_obj, handles = _load_model_with_patches(model, patch, backend, device, strength, mode)
    from brainpatch.runtime.base import GenerationConfig

    cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    console.print(
        Panel(
            "/patches  list patches\n"
            "/strength <name> <value>\n"
            "/on <name>   /off <name>\n"
            "/quit",
            title="brainpatch chat",
        )
    )
    while True:
        try:
            line = console.input("[bold cyan]you[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if _handle_chat_command(line, model_obj):
                break
            continue
        console.print("[bold green]model[/bold green] >", end=" ")
        console.print(model_obj.generate(line, cfg))


def _handle_chat_command(line: str, model_obj: Any) -> bool:
    """Handle a ``/`` command. Returns True to exit the loop."""
    parts = line.split()
    command = parts[0].lower()
    if command in {"/quit", "/exit", "/q"}:
        return True
    if command == "/help":
        console.print("/patches  /strength <name> <v>  /on <name>  /off <name>  /quit")
    elif command == "/patches":
        for name in model_obj.list_patches():
            console.print(f"  {model_obj.patch(name)!r}")
    elif command == "/strength" and len(parts) == 3:
        try:
            actual = model_obj.set_patch_strength(parts[1], float(parts[2]))
            console.print(f"  {parts[1]} strength -> {actual}")
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
    elif command in {"/on", "/off"} and len(parts) == 2:
        try:
            model_obj.enable_patch(parts[1]) if command == "/on" else model_obj.disable_patch(parts[1])
            console.print(f"  {parts[1]} {'enabled' if command == '/on' else 'disabled'}")
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
    else:
        console.print("[yellow]unknown command; /help for the list[/yellow]")
    return False


@app.command()
def serve(
    model: str = typer.Option(..., "--model", "-m"),
    patch: list[str] = typer.Option([], "--patch", "-p"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    strength: Optional[float] = typer.Option(None, "--strength", "-s"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    mode: str = typer.Option("strict", "--compatibility"),
) -> None:
    """Serve an OpenAI-compatible HTTP API with patches applied."""
    try:
        import uvicorn
    except ModuleNotFoundError:
        _fail("serving needs FastAPI and uvicorn -- pip install 'brainpatch[server]'")

    from brainpatch.server.app import build_app

    model_obj, _ = _load_model_with_patches(model, patch, backend, device, strength, mode)
    console.print(f"[green]serving[/green] http://{host}:{port}/v1  (backend: {backend})")
    console.print(f"  patches: {', '.join(model_obj.list_patches()) or 'none'}")
    uvicorn.run(build_app(model_obj), host=host, port=port, log_level="info")


@app.command()
def ui(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share", help="Expose a public Gradio link."),
) -> None:
    """Launch the local web UI. Runs entirely on your machine."""
    try:
        from brainpatch.ui.app import launch
    except ModuleNotFoundError:
        _fail("the UI needs Gradio -- pip install 'brainpatch[ui]'")
    launch(model=model, backend=backend, device=device, host=host, port=port, share=share)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    """Report which inference engines are installed and usable."""
    from brainpatch.runtime.auto import environment_report

    report = environment_report()
    if json_output:
        console.print_json(json.dumps(report))
        return

    console.print(f"[bold]BrainPatch {report['brainpatch_version']}[/bold]")
    console.print(f"  Python   {report['python']}")
    console.print(f"  Platform {report['platform']}")
    console.print(f"  Registry {report['registry_home']}")
    console.print(f"  Patches  {len(report['installed_patches'])} installed")
    console.print()

    table = Table(title="Backends")
    table.add_column("backend")
    table.add_column("status")
    table.add_column("detail")
    for entry in report["backends"]:
        ok = entry["available"]
        table.add_row(
            entry["backend"],
            "[green]available[/green]" if ok else "[red]unavailable[/red]",
            entry["detail"],
        )
    console.print(table)

    if not any(e["available"] for e in report["backends"]):
        console.print(
            "\n[yellow]No backend available.[/yellow] Install one:\n"
            "  pip install 'brainpatch[transformers]'"
        )


@app.command()
def backends(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show the capability matrix for every backend."""
    from brainpatch.runtime.auto import available_backends
    from brainpatch.runtime.capabilities import CAPABILITY_FLAGS

    statuses = available_backends()
    if json_output:
        console.print_json(json.dumps([s.to_dict() for s in statuses]))
        return

    table = Table(title="Backend capabilities")
    table.add_column("capability")
    for status in statuses:
        table.add_column(status.name, justify="center")

    for flag in CAPABILITY_FLAGS:
        row = [flag]
        for status in statuses:
            caps = status.capabilities
            if caps is None:
                row.append("?")
            else:
                row.append("[green]yes[/green]" if caps.supports(flag) else "[dim]no[/dim]")
        table.add_row(*row)
    console.print(table)

    console.print("\n[dim]Notes on unsupported capabilities:[/dim]")
    for status in statuses:
        if status.capabilities and status.capabilities.notes:
            console.print(f"\n[bold]{status.name}[/bold]")
            for flag, note in status.capabilities.notes.items():
                console.print(f"  {flag}: {note}")


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", "-m"),
    patch: list[str] = typer.Option([], "--patch", "-p"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    device: str = typer.Option("auto", "--device"),
    max_new_tokens: int = typer.Option(128, "--max-tokens"),
    runs: int = typer.Option(3, "--runs"),
    prompt: str = typer.Option("Explain how a bicycle works.", "--prompt"),
) -> None:
    """Measure patched vs unpatched throughput and patch load time."""
    import time

    from brainpatch.runtime.base import GenerationConfig

    model_obj, _ = _load_model_with_patches(model, patch, backend, device, None, "strict")
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)

    def timed(enabled: bool) -> tuple[float, int]:
        for name in model_obj.list_patches():
            model_obj.backend.set_enabled(name, enabled)
        durations, tokens = [], 0
        for _ in range(runs):
            start = time.perf_counter()
            text = model_obj.generate(prompt, cfg)
            durations.append(time.perf_counter() - start)
            tokens = max(tokens, len(text.split()))
        return sum(durations) / len(durations), tokens

    base_time, _ = timed(False)
    patch_time, _ = timed(True)

    table = Table(title=f"Throughput ({runs} runs, {max_new_tokens} max tokens)")
    table.add_column("condition")
    table.add_column("mean seconds", justify="right")
    table.add_column("tokens/sec", justify="right")
    table.add_row("baseline", f"{base_time:.3f}", f"{max_new_tokens / base_time:.1f}")
    table.add_row("patched", f"{patch_time:.3f}", f"{max_new_tokens / patch_time:.1f}")
    console.print(table)
    overhead = (patch_time - base_time) / base_time * 100 if base_time else 0.0
    console.print(f"\noverhead: {overhead:+.1f}%")
    console.print(
        "[dim]Wall-clock over few runs; treat small differences as noise.[/dim]"
    )


# ---------------------------------------------------------------------------
# compilation
# ---------------------------------------------------------------------------


@app.command()
def compile(
    source: str = typer.Argument(..., help="Research patch .json, or a .brainpatch to re-export."),
    output: str = typer.Option(..., "--output", "-o"),
    sae: Optional[str] = typer.Option(None, "--sae", help="SAE checkpoint the feature IDs index into."),
    backend: Optional[str] = typer.Option(None, "--backend", help="Export for a backend, e.g. llama.cpp"),
    strength: float = typer.Option(1.0, "--strength", help="Scale baked into a backend export."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Compile a research patch into a self-contained runtime artifact.

    Without --backend this produces a portable .brainpatch. With --backend it
    exports a backend-specific representation (currently llama.cpp control
    vectors) from an already-compiled .brainpatch.
    """
    src = Path(source)

    if backend:
        normalized = backend.lower().replace(".", "").replace("_", "")
        if normalized not in {"llamacpp", "llama"}:
            _fail(f"no exporter for backend {backend!r}; supported: llama.cpp")
        try:
            from brainpatch.patch.compiler import export_llamacpp_control_vector
        except ModuleNotFoundError as exc:
            _fail(f"export needs extra dependencies: {exc}")
        try:
            written = export_llamacpp_control_vector(src, output, strength=strength)
        except Exception as exc:  # noqa: BLE001
            _fail(str(exc))
        console.print(f"[green]exported[/green] {written} ({written.stat().st_size / 1024:.1f} KB)")
        return

    if sae is None:
        _fail("compiling a research patch needs --sae pointing at its SAE checkpoint")

    try:
        from brainpatch.patch.compiler import compile_from_sae
        from brainpatch.schemas.patch_io import load_patch as load_research_patch
    except ModuleNotFoundError as exc:
        _fail(f"compiling needs the research extra: {exc}")

    try:
        spec = load_research_patch(src)
        written = compile_from_sae(spec, sae, output, overwrite=force)
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))

    from brainpatch.patch.loader import load_patch, patch_size_report

    report = patch_size_report(load_patch(written))
    console.print(f"[green]compiled[/green] {written}")
    console.print(f"  {report['archive_kb']} KB, {report['num_vectors']} vector(s), "
                  f"{report['hidden_size']}-d {report['dtype']}")


# ---------------------------------------------------------------------------
# research subcommands
# ---------------------------------------------------------------------------


@research_app.command("modal")
def research_modal(
    function: str = typer.Argument(..., help="Modal function, e.g. gpu_info."),
    extra: Optional[list[str]] = typer.Argument(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run a BrainPatch research job on Modal (this repo's research backend)."""
    import shutil
    import subprocess

    binary = shutil.which("modal")
    if binary is None:
        _fail("the modal CLI is not installed -- pip install 'brainpatch[modal]'")
    args = [binary, "run", f"modal_app/app.py::{function}", *(extra or [])]
    console.print(f"[dim]$ {' '.join(args)}[/dim]")
    if dry_run:
        return
    result = subprocess.run(args, check=False)
    raise typer.Exit(code=result.returncode)


@research_app.command("contrast")
def research_contrast(
    name: Optional[str] = typer.Argument(None, help="Contrast set name; omit to list."),
) -> None:
    """Inspect the synthetic behavioural contrast fixtures."""
    from brainpatch.datasets import list_contrast_sets, load_contrast_set

    if name is None:
        console.print("available: " + (", ".join(list_contrast_sets()) or "none"))
        return
    try:
        contrast_set = load_contrast_set(name)
    except FileNotFoundError as exc:
        _fail(str(exc))
    console.print(f"[bold]{contrast_set.name}[/bold] -- {len(contrast_set)} examples")
    console.print(contrast_set.description)
    if contrast_set.synthetic:
        console.print("\n[yellow]Synthetic development fixture, not a benchmark.[/yellow]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _evidence_markup(level: str) -> str:
    colour = {
        "replicated": "green",
        "controlled_interventional": "green",
        "interventional": "yellow",
        "predictive": "yellow",
        "correlational": "yellow",
        "none": "red",
    }.get(level, "white")
    return f"[{colour}]{level}[/{colour}]"


def _load_patch_arg(ref: str) -> Any:
    from brainpatch.patch.loader import PatchLoadError, load_patch
    from brainpatch.patch.registry import RegistryError, default_registry

    try:
        path = default_registry().resolve(ref)
    except RegistryError as exc:
        _fail(str(exc))
    try:
        return load_patch(path)
    except PatchLoadError as exc:
        _fail(str(exc))


def _load_model_with_patches(
    model: str,
    patches: list[str],
    backend: str,
    device: str,
    strength: float | None,
    mode: str,
) -> tuple[Any, list[Any]]:
    from brainpatch.runtime.auto import BackendNotAvailable
    from brainpatch.runtime.model import BrainPatchedModel

    try:
        model_obj = BrainPatchedModel.from_pretrained(
            model, backend=backend, device=device, compatibility_mode=mode  # type: ignore[arg-type]
        )
    except BackendNotAvailable as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not load model {model!r}: {exc}")

    handles = []
    for ref in patches:
        try:
            handles.append(model_obj.install(ref, strength=strength))
        except Exception as exc:  # noqa: BLE001
            _fail(f"could not install patch {ref!r}: {exc}")
    return model_obj, handles


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
