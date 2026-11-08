"""Validate inline preservation before export and fail-closed on issues."""

from __future__ import annotations

from ..markup.contracts import INLINE_META_KEY
from ..models.document import Chapter


class InlinePreservationError(Exception):
    """Raised when inline markup cannot be reliably preserved in export."""

    pass


def validate_inline_preservation_for_export(
    chapters: list[Chapter],
    *,
    allow_proportional_fallback: bool = False,
) -> None:
    """Validate that inline markup can be reliably preserved.

    Args:
        chapters: Chapters to validate
        allow_proportional_fallback: If False, reject segments with inline markup but no target_inline

    Raises:
        InlinePreservationError: If inline preservation is unreliable
    """
    issues: list[str] = []

    for chapter in chapters:
        for segment in chapter.segments:
            # Check if segment has source inline markup
            raw_inline = segment.meta.get(INLINE_META_KEY)
            if not isinstance(raw_inline, dict):
                continue

            nodes = raw_inline.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                continue

            # Count semantic emphasis nodes (skip images, links, etc.)
            semantic_count = sum(
                1
                for node in nodes
                if isinstance(node, dict)
                and node.get("tag") in ("i", "em", "strong", "b")
            )

            if semantic_count == 0:
                continue

            # Check if target exists
            if segment.target is None or not segment.target.strip():
                issues.append(
                    f"Chapter {chapter.index}, Segment {segment.index}: "
                    f"Untranslated segment with {semantic_count} inline emphasis nodes. "
                    f"Complete translation before export."
                )
                continue

            # Check for verified target positions
            raw_target_inline = segment.meta.get("target_inline")

            if not isinstance(raw_target_inline, dict):
                if not allow_proportional_fallback:
                    issues.append(
                        f"Chapter {chapter.index}, Segment {segment.index}: "
                        f"Has {semantic_count} source inline nodes but no verified target positions. "
                        f"This segment was translated without inline marker preservation. "
                        f"To restore emphasis accurately, retranslate with current version "
                        f"or use --allow-proportional-fallback to accept unreliable placement."
                    )
                continue

            # Validate target_inline structure
            ranges = raw_target_inline.get("ranges")
            if not isinstance(ranges, list):
                if not allow_proportional_fallback:
                    issues.append(
                        f"Chapter {chapter.index}, Segment {segment.index}: "
                        f"Invalid target_inline structure (no ranges). "
                        f"Retranslate this segment."
                    )
                continue

            # Check for validation warnings
            warnings = raw_target_inline.get("validation_warnings")
            if isinstance(warnings, list) and warnings:
                critical_warnings = [
                    w
                    for w in warnings
                    if any(
                        keyword in str(w).lower()
                        for keyword in ["unclosed", "mismatched", "unmatched"]
                    )
                ]
                if critical_warnings and not allow_proportional_fallback:
                    issues.append(
                        f"Chapter {chapter.index}, Segment {segment.index}: "
                        f"Critical inline preservation warnings: {'; '.join(str(w) for w in critical_warnings)}. "
                        f"Retranslate this segment."
                    )

    if issues:
        message = (
            "Cannot export: inline markup preservation is unreliable.\n\n"
            + "\n".join(f"  - {issue}" for issue in issues)
            + "\n\n"
            + "To fix:\n"
            + "  1. Retranslate affected segments with current wenyi version\n"
            + "  2. Or use --allow-proportional-fallback to accept character-offset estimation (unreliable)\n"
        )
        raise InlinePreservationError(message)
