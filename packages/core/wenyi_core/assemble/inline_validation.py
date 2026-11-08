"""Validate inline preservation before export and fail-closed on issues."""

from __future__ import annotations

import os
import zipfile
from typing import Any

from bs4 import BeautifulSoup, Tag
from charset_normalizer import from_bytes

from ..markup.contracts import INLINE_META_KEY
from ..models.document import Chapter, Segment


class InlinePreservationError(Exception):
    """Raised when inline markup cannot be reliably preserved in export."""

    pass


def _detect_emphasis_in_epub_source(
    source_path: str,
    manifest: dict[str, Any],
    chapter: Chapter,
    segment: Segment,
) -> bool:
    """Detect if segment's source HTML in EPUB contains semantic emphasis markup.
    
    Returns True if source contains em/strong/i/b tags or spans with italic/bold CSS,
    False otherwise or on any parsing error (fail-open for detection, fail-closed for validation).
    """
    if not isinstance(source_path, str) or not source_path.lower().endswith(".epub"):
        return False
    if not os.path.exists(source_path):
        return False
    
    resource_href = segment.resource_href
    if not resource_href:
        return False
    
    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            if resource_href not in zf.namelist():
                return False
            
            html_data = zf.read(resource_href)
            # Detect encoding
            detected = from_bytes(html_data).best()
            if detected is None:
                html = html_data.decode("utf-8", errors="replace")
            else:
                html = str(detected)
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Find blocks containing segment's source text (heuristic: exact text match)
            # This is imperfect but catches most cases
            from ..markup.segments import _BLOCK_CANDIDATE_TAGS
            
            blocks = soup.find_all(_BLOCK_CANDIDATE_TAGS)
            for block in blocks:
                block_text = block.get_text(strip=False)
                # Normalize whitespace for comparison
                block_normalized = " ".join(block_text.split())
                segment_normalized = " ".join(segment.source.split())
                
                if segment_normalized and segment_normalized in block_normalized:
                    # Check for semantic emphasis tags
                    emphasis_tags = block.find_all(["em", "strong", "i", "b"])
                    if emphasis_tags:
                        return True
                    
                    # Check for spans with italic/bold CSS (simplified check)
                    spans = block.find_all("span")
                    for span in spans:
                        style = span.get("style", "")
                        class_attr = span.get("class", [])
                        if isinstance(class_attr, list):
                            class_str = " ".join(class_attr)
                        else:
                            class_str = str(class_attr)
                        
                        # Simple heuristic: style contains "italic" or "bold", or class contains these keywords
                        if any(keyword in style.lower() for keyword in ["italic", "oblique", "font-weight:bold", "font-weight: bold"]):
                            return True
                        if any(keyword in class_str.lower() for keyword in ["italic", "bold", "emphasis"]):
                            return True
            
            return False
    except Exception:
        # Fail-open for detection errors: cannot conclusively detect emphasis
        return False


def validate_inline_preservation_for_export(
    chapters: list[Chapter],
    *,
    allow_proportional_fallback: bool = False,
    source_path: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Validate that inline markup can be reliably preserved.

    Args:
        chapters: Chapters to validate
        allow_proportional_fallback: If False, reject segments with inline markup but no target_inline
        source_path: Path to source EPUB for re-scanning old state without INLINE_META_KEY
        manifest: Manifest dict for EPUB metadata

    Raises:
        InlinePreservationError: If inline preservation is unreliable
    """
    issues: list[str] = []

    for chapter in chapters:
        for segment in chapter.segments:
            # Check if segment has source inline markup metadata (new state)
            raw_inline = segment.meta.get(INLINE_META_KEY)
            
            # Old state detection: no INLINE_META_KEY but might have emphasis in source EPUB
            if not isinstance(raw_inline, dict):
                # For old state, re-scan source EPUB to detect emphasis
                if source_path and manifest:
                    has_source_emphasis = _detect_emphasis_in_epub_source(
                        source_path, manifest, chapter, segment
                    )
                    if has_source_emphasis:
                        # Old state with emphasis detected: check target_inline
                        raw_target_inline = segment.meta.get("target_inline")
                        if not isinstance(raw_target_inline, dict) or not raw_target_inline.get("ranges"):
                            issues.append(
                                f"Chapter {chapter.index}, Segment {segment.index}: "
                                f"Source HTML contains emphasis markup (detected from EPUB), "
                                f"but this segment was translated with an old version that did not preserve "
                                f"inline positions. All emphasis will be lost in export. "
                                f"Retranslate this segment with current wenyi version."
                            )
                continue

            nodes = raw_inline.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                continue

            # Count semantic emphasis nodes (skip images, links, etc.)
            semantic_count = sum(
                1
                for node in nodes
                if isinstance(node, dict) and node.get("tag") in ("i", "em", "strong", "b", "span")
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

            # Check for explicit preservation failure flag
            if raw_target_inline.get("preservation_failed") or raw_target_inline.get(
                "decode_failed"
            ):
                issues.append(
                    f"Chapter {chapter.index}, Segment {segment.index}: "
                    f"Inline marker preservation failed during translation/polish. "
                    f"Error: {raw_target_inline.get('error', 'markers lost or corrupted')}. "
                    f"Retranslate this segment."
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

            # Check if ranges are empty but source had nodes (preservation failure)
            if len(ranges) == 0:
                source_node_count = raw_target_inline.get("source_node_count", 0)
                if source_node_count > 0 and not allow_proportional_fallback:
                    issues.append(
                        f"Chapter {chapter.index}, Segment {segment.index}: "
                        f"All {source_node_count} inline emphasis markers were lost. "
                        f"Retranslate or re-polish this segment."
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
                        for keyword in ["unclosed", "mismatched", "unmatched", "lost"]
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
