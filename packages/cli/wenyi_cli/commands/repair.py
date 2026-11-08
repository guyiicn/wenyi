"""Controlled repair command for fixing inline marker issues in existing state."""

import json
import shutil
from pathlib import Path

import typer

from wenyi_core.markup.inline_markers import InlineNode, decode_inline_markers, has_inline_markers
from wenyi_core.storage.file import FileStorage


def register_repair_commands(app: typer.Typer) -> None:
    """Register repair commands."""
    repair_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="Repair inline emphasis and apply confirmed text corrections.",
    )
    app.add_typer(repair_app, name="repair", rich_help_panel="Repair")

    @repair_app.command("fix-inline-markers")
    def fix_inline_markers_cmd(
        state_dir: str = typer.Argument(..., help="Path to state/Book/targets/zh"),
        chapter: int = typer.Option(..., "--chapter", help="Chapter index"),
        segment: int = typer.Option(..., "--segment", help="Segment index"),
        expected_text: str = typer.Option(..., "--expected-text", help="Current target (must match)"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without modifying"),
    ):
        """Fix segment with literal [[I_START]]/[[EM_START]] markers."""
        _fix_inline_markers(state_dir, chapter, segment, expected_text, dry_run)


def _fix_inline_markers(state_dir: str, chapter: int, segment: int, expected_text: str, dry_run: bool):
    """Implementation of fix-inline-markers."""
    state_path = Path(state_dir)
    if not state_path.is_absolute():
        state_path = Path.cwd() / state_path
    
    store = FileStorage(state_path)
    chapter_data = store.load_chapter(chapter)
    if chapter_data is None:
        typer.echo(f"❌ Chapter {chapter} not found", err=True)
        raise typer.Abort()
    
    segments = chapter_data.segments if hasattr(chapter_data, 'segments') else chapter_data.get("segments", [])
    seg = next((s for s in segments if (s.index if hasattr(s, 'index') else s.get("index")) == segment), None)
    
    if seg is None:
        typer.echo(f"❌ Segment {segment} not found", err=True)
        raise typer.Abort()
    
    current_target = seg.target if hasattr(seg, 'target') else seg.get("target", "")
    
    if current_target != expected_text:
        typer.echo("❌ Current target does not match expected:", err=True)
        typer.echo(f"   Expected: {expected_text}", err=True)
        typer.echo(f"   Current:  {current_target}", err=True)
        raise typer.Abort()
    
    if not has_inline_markers(current_target):
        typer.echo(f"✓ No literal markers - nothing to fix")
        return
    
    typer.echo(f"Found literal markers in segment {chapter}:{segment}")
    typer.echo(f"  Current: {current_target}")
    
    meta = seg.meta if hasattr(seg, 'meta') else seg.get("meta", {})
    epub_inline = meta.get("epub_inline") if isinstance(meta, dict) else getattr(meta, 'epub_inline', None)
    
    if not isinstance(epub_inline, dict):
        typer.echo("❌ No source epub_inline metadata", err=True)
        raise typer.Abort()
    
    source_nodes_raw = epub_inline.get("nodes", []) if isinstance(epub_inline, dict) else getattr(epub_inline, 'nodes', [])
    if not source_nodes_raw:
        typer.echo("❌ No source inline nodes", err=True)
        raise typer.Abort()
    
    source_nodes = []
    for n in source_nodes_raw:
        tag = n.get("tag") if isinstance(n, dict) else getattr(n, 'tag', None)
        if tag:
            source_nodes.append(InlineNode(tag=tag, start=0, end=1, marker_name=tag.upper()))
    
    try:
        decoded_text, positions, errors = decode_inline_markers(current_target, source_nodes)
    except Exception as e:
        typer.echo(f"❌ Decode failed: {e}", err=True)
        raise typer.Abort()
    
    if errors:
        typer.echo("❌ Decode errors:", err=True)
        for err in errors:
            typer.echo(f"   {err}", err=True)
        raise typer.Abort()
    
    typer.echo(f"  Decoded: {decoded_text}")
    typer.echo(f"  Ranges:  {len(positions)}")
    
    if dry_run:
        typer.echo("\n[DRY RUN] Would apply:")
        typer.echo(f"  target: {current_target} → {decoded_text}")
        if positions:
            typer.echo(f"  Add target_inline with {len(positions)} ranges")
        return
    
    # Backup
    chapter_file = state_path / "chapters" / f"ch{chapter}.json"
    if chapter_file.exists():
        backup_file = chapter_file.with_suffix(f".bak-seg{segment}.json")
        shutil.copy2(chapter_file, backup_file)
        typer.echo(f"✓ Backup: {backup_file}")
    
    # Apply
    if hasattr(seg, 'target'):
        seg.target = decoded_text
    else:
        seg["target"] = decoded_text
    
    if positions:
        ranges = [{"start": s, "end": e, "tag": t, "css_style": ""} for s, e, t in positions]
        target_inline_meta = {"version": 1, "ranges": ranges}
        if hasattr(seg, 'meta'):
            if isinstance(seg.meta, dict):
                seg.meta["target_inline"] = target_inline_meta
            else:
                seg.meta.target_inline = target_inline_meta
        else:
            seg["meta"]["target_inline"] = target_inline_meta
    
    store.save_chapter(chapter_data)
    typer.echo(f"✓ Repaired segment {chapter}:{segment}")
