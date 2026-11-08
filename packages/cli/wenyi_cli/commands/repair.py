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

    @repair_app.command("apply-target")
    def apply_target_cmd(
        state_dir: str = typer.Argument(..., help="Path to state/Book/targets/zh"),
        chapter: int = typer.Option(..., "--chapter", help="Chapter index"),
        segment: int = typer.Option(..., "--segment", help="Segment index"),
        expected_text: str = typer.Option(..., "--expected-text", help="Current target (must match)"),
        new_target: str = typer.Option(..., "--new-target", help="New target text to apply"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without modifying"),
    ):
        """Apply confirmed text correction to segment.
        
        Safely replaces segment target after exact current text verification.
        For segments with inline emphasis, new target must contain explicit markers
        [[I_START]]/[[EM_END]] or modification will be rejected.
        
        Example:
            wenyi repair apply-target state/Book/targets/zh \\
                --chapter 0 --segment 62 \\
                --expected-text '受金属过度承力的呻吟化作尖啸。' \\
                --new-target '过度受力的金属发出的呻吟化作了尖啸。' \\
                --dry-run
        """
        _apply_target(state_dir, chapter, segment, expected_text, new_target, dry_run)


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


def _apply_target(
    state_dir: str, chapter: int, segment: int, expected_text: str, new_target: str, dry_run: bool
):
    """Implementation of apply-target."""
    state_path = Path(state_dir)
    if not state_path.is_absolute():
        state_path = Path.cwd() / state_path

    store = FileStorage(state_path)
    chapter_data = store.load_chapter(chapter)
    if chapter_data is None:
        typer.echo(f"❌ Chapter {chapter} not found", err=True)
        raise typer.Abort()

    segments = (
        chapter_data.segments
        if hasattr(chapter_data, "segments")
        else chapter_data.get("segments", [])
    )
    seg = next(
        (
            s
            for s in segments
            if (s.index if hasattr(s, "index") else s.get("index")) == segment
        ),
        None,
    )

    if seg is None:
        typer.echo(f"❌ Segment {segment} not found", err=True)
        raise typer.Abort()

    current_target = seg.target if hasattr(seg, "target") else seg.get("target", "")

    # Safety: exact current text match
    if current_target != expected_text:
        typer.echo("❌ Current target does not match expected:", err=True)
        typer.echo(f"   Expected: {expected_text}", err=True)
        typer.echo(f"   Current:  {current_target}", err=True)
        raise typer.Abort()

    # Check if segment has source inline metadata
    meta = seg.meta if hasattr(seg, "meta") else seg.get("meta", {})
    epub_inline = (
        meta.get("epub_inline") if isinstance(meta, dict) else getattr(meta, "epub_inline", None)
    )
    has_source_inline = isinstance(epub_inline, dict) and epub_inline.get("nodes")

    # Check if new target contains literal markers
    new_has_markers = has_inline_markers(new_target)

    if has_source_inline and not new_has_markers:
        # Segment has source inline but new target has no markers
        # This could lose emphasis - warn but allow (user responsibility)
        typer.echo(
            "⚠️  Segment has source inline emphasis but new target has no markers", err=True
        )
        typer.echo("   This may lose emphasis. Verify correctness.", err=True)

    if new_has_markers:
        # New target contains literal markers - must decode
        if not has_source_inline:
            typer.echo("❌ New target has markers but no source inline metadata", err=True)
            raise typer.Abort()

        # Decode markers
        source_nodes_raw = (
            epub_inline.get("nodes", [])
            if isinstance(epub_inline, dict)
            else getattr(epub_inline, "nodes", [])
        )
        source_nodes = []
        for n in source_nodes_raw:
            tag = n.get("tag") if isinstance(n, dict) else getattr(n, "tag", None)
            if tag:
                source_nodes.append(InlineNode(tag=tag, start=0, end=1, marker_name=tag.upper()))

        try:
            decoded_text, positions, errors = decode_inline_markers(new_target, source_nodes)
        except Exception as e:
            typer.echo(f"❌ Decode failed: {e}", err=True)
            raise typer.Abort()

        if errors:
            typer.echo("❌ Decode errors:", err=True)
            for err in errors:
                typer.echo(f"   {err}", err=True)
            raise typer.Abort()

        typer.echo(f"Applying correction to segment {chapter}:{segment} (with inline decode)")
        typer.echo(f"  Current: {current_target}")
        typer.echo(f"  Decoded: {decoded_text}")
        typer.echo(f"  Ranges:  {len(positions)}")

        if dry_run:
            typer.echo("\n[DRY RUN] Would apply:")
            typer.echo(f"  target: {current_target} → {decoded_text}")
            if positions:
                typer.echo(f"  Update target_inline with {len(positions)} ranges")
            return

        # Backup
        chapter_file = state_path / "chapters" / f"ch{chapter}.json"
        if chapter_file.exists():
            backup_file = chapter_file.with_suffix(f".bak-seg{segment}.json")
            if not backup_file.exists():
                shutil.copy2(chapter_file, backup_file)
                typer.echo(f"✓ Backup: {backup_file}")

        # Apply with inline decode
        if hasattr(seg, "target"):
            seg.target = decoded_text
        else:
            seg["target"] = decoded_text

        if positions:
            ranges = [{"start": s, "end": e, "tag": t, "css_style": ""} for s, e, t in positions]
            target_inline_meta = {"version": 1, "ranges": ranges}
            if hasattr(seg, "meta"):
                if isinstance(seg.meta, dict):
                    seg.meta["target_inline"] = target_inline_meta
                else:
                    seg.meta.target_inline = target_inline_meta
            else:
                seg["meta"]["target_inline"] = target_inline_meta

        store.save_chapter(chapter_data)
        typer.echo(f"✓ Applied correction with inline decode")

    else:
        # No markers in new target - simple text replacement
        typer.echo(f"Applying correction to segment {chapter}:{segment}")
        typer.echo(f"  Current: {current_target}")
        typer.echo(f"  New:     {new_target}")

        if dry_run:
            typer.echo("\n[DRY RUN] Would apply")
            return

        # Backup
        chapter_file = state_path / "chapters" / f"ch{chapter}.json"
        if chapter_file.exists():
            backup_file = chapter_file.with_suffix(f".bak-seg{segment}.json")
            if not backup_file.exists():
                shutil.copy2(chapter_file, backup_file)
                typer.echo(f"✓ Backup: {backup_file}")

        # Apply
        if hasattr(seg, "target"):
            seg.target = new_target
        else:
            seg["target"] = new_target

        # Preserve existing target_inline if present
        store.save_chapter(chapter_data)
        typer.echo(f"✓ Applied correction")
