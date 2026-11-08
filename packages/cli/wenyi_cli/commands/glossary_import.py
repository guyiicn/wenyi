"""Glossary import from external TSV files with validation and conflict detection."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.models.glossary import GlossaryTerm, normalize_term_type
from wenyi_core.storage.protocol import Storage
from wenyi_core.storage.run_store import source_sha256


class ImportError(Exception):
    """Errors during glossary import validation or execution."""

    pass


def _initialize_minimal_state(store: Storage, input_path: str, config) -> None:
    """Initialize minimal manifest for fresh projects to enable glossary import.

    Creates a minimal valid manifest with source identity but no chapters or analysis.
    This allows importing glossary before any AI preparation or model calls.
    Subsequent prepare/translate will complete initialization normally.
    """
    if not os.path.exists(input_path):
        raise ImportError(
            f"Source file not found: {input_path}. "
            f"Ensure the file exists before importing glossary."
        )

    # Compute source hash for identity tracking
    digest = source_sha256(input_path)

    # Create minimal manifest with source identity
    # This is sufficient for glossary import; prepare/translate will complete initialization
    minimal_manifest = {
        "fmt": os.path.splitext(input_path)[1][1:].lower() or "txt",
        "title": os.path.splitext(os.path.basename(input_path))[0],
        "source_lang": config.source_lang,
        "target_lang": config.target_lang,
        "source_sha256": digest,
        "chapters": [],  # Will be populated by prepare/translate
        "glossary_import_only": True,  # Marker for minimal initialization
    }

    # Begin initialization with source identity
    store.begin_initialization(digest)

    # Save minimal manifest
    store.save_manifest(minimal_manifest)

    # Initialization complete - glossary can now be imported


def _parse_aliases(text: str) -> list[str]:
    """Parse pipe-separated aliases from TSV; return empty list for empty text."""
    if not text or not text.strip():
        return []
    return [alias.strip() for alias in text.split("|") if alias.strip()]


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 of the import file for idempotency tracking."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_tsv_records(
    path: Path, *, source_col: str, target_col: str, console: Console
) -> list[dict[str, Any]]:
    """Load and validate TSV structure; return raw records."""
    if not path.exists():
        raise ImportError(f"Import file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ImportError("TSV file is empty or has no header row")

        if source_col not in reader.fieldnames:
            raise ImportError(
                f"Required source column '{source_col}' not found in TSV. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        if target_col not in reader.fieldnames:
            raise ImportError(
                f"Required target column '{target_col}' not found in TSV. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        records = []
        for row_num, row in enumerate(reader, start=2):
            if not row.get(source_col, "").strip():
                console.print(f"[yellow]Warning: Row {row_num} has empty source, skipping[/]")
                continue
            if not row.get(target_col, "").strip():
                console.print(f"[yellow]Warning: Row {row_num} has empty target, skipping[/]")
                continue
            records.append(row)

        if not records:
            raise ImportError("No valid records found in TSV after filtering empty source/target")

        return records


def _parse_term_from_record(
    row: dict[str, Any],
    *,
    source_col: str,
    target_col: str,
    type_col: str | None,
    aliases_col: str | None,
    note_col: str | None,
    status_col: str | None,
) -> GlossaryTerm:
    """Parse one TSV row into a GlossaryTerm with validated types."""
    source = row[source_col].strip()
    target = row[target_col].strip()

    term_type = "term"
    if type_col and row.get(type_col):
        raw_type = row[type_col].strip()
        # Map known external types to internal types
        type_mapping = {
            "person": "person",
            "organization": "term",
            "rank": "term",
            "concept": "term",
            "weapon": "term",
            "ship": "term",
            "droid": "term",
            "franchise": "term",
            "order-or-member": "term",
            "person-title": "person",
            "place": "term",
        }
        term_type = type_mapping.get(raw_type.lower(), raw_type)
        term_type = normalize_term_type(term_type)

    aliases = []
    if aliases_col and row.get(aliases_col):
        aliases = _parse_aliases(row[aliases_col])

    note = ""
    if note_col and row.get(note_col):
        note = row[note_col].strip()

    status = "ok"
    if status_col and row.get(status_col):
        raw_status = row[status_col].strip().lower()
        # Map external status to internal status
        if raw_status in ("verified", "confirmed"):
            status = "ok"
        elif raw_status in ("provisional", "pending", "draft"):
            status = "provisional"
        else:
            status = raw_status  # Preserve unknown status for validation

    return GlossaryTerm(
        source=source,
        target=target,
        type=term_type,
        aliases=aliases,
        note=note,
        status=status,
    )


def _validate_collision_strategy(
    terms: list[GlossaryTerm], *, console: Console
) -> dict[str, list[GlossaryTerm]]:
    """Group terms by source and detect collisions; raise on ambiguous same-name entities."""
    groups: dict[str, list[GlossaryTerm]] = {}
    for term in terms:
        groups.setdefault(term.source, []).append(term)

    collisions = {source: group for source, group in groups.items() if len(group) > 1}

    if collisions:
        console.print(
            f"[red]Error: {len(collisions)} source term(s) have multiple conflicting entries:[/]"
        )
        table = Table("Source", "Target variants", "Count")
        for source, group in sorted(collisions.items()):
            targets = ", ".join(f'"{t.target}"' for t in group)
            table.add_row(source, targets, str(len(group)))
        console.print(table)
        raise ImportError(
            "Cannot import TSV with duplicate source entries. "
            "Ensure entity_id separation is resolved to unique source terms, "
            "or provide a project-specific unambiguous subset."
        )

    return groups


def _execute_import(
    store: Storage,
    terms: list[GlossaryTerm],
    *,
    dry_run: bool,
    console: Console,
) -> dict[str, int]:
    """Execute glossary import with conflict detection; return outcome counts."""
    if dry_run:
        console.print("[yellow]Dry-run mode: no changes will be written[/]")

    glossary = GlossaryStore(store.glossary_path)
    try:
        results = {"inserted": 0, "unchanged": 0, "conflict": 0, "skipped": 0}
        conflicts_detail: list[dict[str, str]] = []

        for term in terms:
            if dry_run:
                # Check existing without writing
                existing = glossary.get_term(term.source)
                if existing is None:
                    results["inserted"] += 1
                elif existing.target == term.target:
                    results["unchanged"] += 1
                else:
                    results["conflict"] += 1
                    conflicts_detail.append(
                        {
                            "source": term.source,
                            "existing_target": existing.target,
                            "proposed_target": term.target,
                        }
                    )
            else:
                outcome = glossary.upsert_term(term, chapter=None)
                results[outcome] += 1
                if outcome == "conflict":
                    existing = glossary.get_term(term.source)
                    if existing:
                        conflicts_detail.append(
                            {
                                "source": term.source,
                                "existing_target": existing.target,
                                "proposed_target": term.target,
                            }
                        )

        # Report conflicts
        if conflicts_detail:
            console.print(
                f"\n[yellow]Warning: {len(conflicts_detail)} term(s) conflicted with existing translations:[/]"
            )
            table = Table("Source", "Existing target", "Proposed target")
            for conflict in conflicts_detail[:10]:
                table.add_row(
                    conflict["source"], conflict["existing_target"], conflict["proposed_target"]
                )
            if len(conflicts_detail) > 10:
                console.print(f"... and {len(conflicts_detail) - 10} more conflicts")
            console.print(table)
            console.print(
                "[yellow]Conflicts are recorded for manual review. "
                "Use `wenyi glossary conflicts` to list them and "
                "`wenyi glossary resolve` to resolve.[/]"
            )

        return results
    finally:
        glossary.close()


def _track_import_metadata(
    store: Storage, file_path: Path, file_hash: str, results: dict[str, int], *, dry_run: bool
) -> None:
    """Record import provenance in artifact storage for idempotency."""
    if dry_run:
        return

    key = f"glossary/imports/{file_hash}.json"
    metadata = {
        "schema_version": 1,
        "file_path": str(file_path),
        "file_sha256": file_hash,
        "imported_at": None,  # Will be set by artifact storage
        "results": results,
    }
    store.write_artifact(key, metadata)


def import_glossary(
    store: Storage | None,
    file_path: Path,
    *,
    input_path: str | None = None,
    config: Any = None,
    source_col: str = "english",
    target_col: str = "preferred_zh_cn",
    type_col: str | None = "type",
    aliases_col: str | None = "aliases_and_regional",
    note_col: str | None = "usage_conflicts",
    status_col: str | None = "status",
    dry_run: bool = False,
    console: Console,
) -> dict[str, int]:
    """Import glossary terms from TSV; validate collisions and return outcome counts.

    For fresh projects without manifest, creates minimal state to enable import before
    AI preparation. This allows importing frozen terminology before any model calls.
    Required parameters input_path and config are only needed for fresh project initialization.

    Dry-run mode performs pure validation without creating any directories or files.
    """
    file_hash = _compute_file_hash(file_path)

    # Check if this is a fresh project
    is_fresh = store is None or not store.exists()

    if is_fresh and dry_run:
        # Pure validation mode: no store creation, no filesystem writes
        console.print(
            "[yellow]Dry-run on fresh project (no state yet). "
            "Validating TSV without creating any files...[/]"
        )
        store = None  # Ensure no store operations
    elif is_fresh and not dry_run:
        # Actual import on fresh project: initialize minimal state
        console.print("Initializing fresh project for glossary import...")
        if store is None:
            raise ImportError("Cannot initialize without store object")
        if input_path is None or config is None:
            raise ImportError(
                "No translation state found. "
                "To import on a fresh project, provide input_path and config for initialization."
            )
        _initialize_minimal_state(store, input_path, config)
    elif not is_fresh and dry_run:
        # Existing project dry-run: validate without writes
        pass  # Store exists, will be used read-only

    # Load and parse TSV (always needed for validation)
    records = _load_tsv_records(
        file_path, source_col=source_col, target_col=target_col, console=console
    )
    console.print(f"Loaded {len(records)} valid records from TSV")

    terms = [
        _parse_term_from_record(
            row,
            source_col=source_col,
            target_col=target_col,
            type_col=type_col,
            aliases_col=aliases_col,
            note_col=note_col,
            status_col=status_col,
        )
        for row in records
    ]

    # Validate collisions
    _validate_collision_strategy(terms, console=console)

    # For fresh project dry-run, stop here (no store operations)
    if is_fresh and dry_run:
        # Pure memory preview: count what would be inserted
        results = {"inserted": len(terms), "unchanged": 0, "conflict": 0, "skipped": 0}
        return results

    # From here on, we need a valid store
    if store is None:
        raise ImportError("Store is required for non-dry-run operations")

    # Check idempotency
    import_key = f"glossary/imports/{file_hash}.json"
    previous_import = store.read_artifact(import_key)
    if isinstance(previous_import, dict) and not dry_run:
        console.print(
            f"[yellow]This file (SHA256: {file_hash[:16]}...) was previously imported.[/]"
        )
        console.print("Re-importing to handle any new conflicts or changes...")

    # Execute import
    results = _execute_import(store, terms, dry_run=dry_run, console=console)

    # Track metadata
    _track_import_metadata(store, file_path, file_hash, results, dry_run=dry_run)

    return results


def register_import_command(app: typer.Typer, context) -> None:
    """Register the import subcommand under glossary."""

    @app.command("import")
    def glossary_import(
        input: str = typer.Argument(..., help="Source file with translation state"),
        file: Path = typer.Option(
            ..., "--file", "-f", help="TSV file to import (absolute or relative path)"
        ),
        source_col: str = typer.Option(
            "english", "--source-col", help="TSV column name for source terms"
        ),
        target_col: str = typer.Option(
            "preferred_zh_cn", "--target-col", help="TSV column name for target translations"
        ),
        type_col: str = typer.Option("type", "--type-col", help="TSV column for term type"),
        aliases_col: str = typer.Option(
            "aliases_and_regional", "--aliases-col", help="TSV column for pipe-separated aliases"
        ),
        note_col: str = typer.Option(
            "usage_conflicts", "--note-col", help="TSV column for usage notes"
        ),
        status_col: str = typer.Option("status", "--status-col", help="TSV column for status"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Preview import without writing changes"
        ),
    ) -> None:
        """Import glossary from external TSV before translation.

        The TSV must have a header row with at minimum source and target columns.
        Optional columns include type, aliases (pipe-separated), note, and status.

        Status mapping: 'verified'/'confirmed' → 'ok', 'provisional'/'pending'/'draft' → 'provisional'.
        Type mapping: known external types (person, organization, rank, etc.) → internal types.

        Duplicate source entries are rejected; resolve entity_id collisions externally.
        Conflicts with existing translations are recorded but not silently overwritten.
        Use --dry-run to preview without making changes.
        """
        console = context().console
        from .validation import runstore_for_cli

        try:
            config = context().load_config()

            # For dry-run on potentially fresh project, don't create store yet
            # Only create store for actual import or existing project dry-run
            store = None
            if not dry_run:
                # Actual import: need store
                store = runstore_for_cli(config, input, console=console, allow_missing=True)
            else:
                # Dry-run: check if state exists without creating it
                try:
                    store = runstore_for_cli(config, input, console=console, allow_missing=False)
                except typer.Exit:
                    # Fresh project: pure validation, no store needed
                    store = None

            results = import_glossary(
                store,
                file,
                input_path=input,
                config=config,
                source_col=source_col,
                target_col=target_col,
                type_col=type_col if type_col else None,
                aliases_col=aliases_col if aliases_col else None,
                note_col=note_col if note_col else None,
                status_col=status_col if status_col else None,
                dry_run=dry_run,
                console=console,
            )

            # Summary
            if dry_run:
                console.print("\n[bold]Dry-run summary:[/]")
            else:
                console.print("\n[bold green]Import complete:[/]")

            table = Table("Outcome", "Count")
            for outcome, count in results.items():
                if count > 0:
                    table.add_row(outcome.capitalize(), str(count))
            console.print(table)

            if not dry_run and results.get("inserted", 0) > 0:
                console.print(
                    f"\n[green]Imported {results['inserted']} new term(s). "
                    f"Run `wenyi glossary list {input}` to verify.[/]"
                )

        except ImportError as error:
            console.print(f"[red]Import error: {error}[/]")
            raise typer.Exit(1) from None
        except (OSError, ValueError, RuntimeError) as error:
            console.print(f"[red]Error: {error}[/]")
            raise typer.Exit(1) from None
