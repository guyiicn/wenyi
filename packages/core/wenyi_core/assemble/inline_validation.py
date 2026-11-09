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


class SourceEmphasisResult:
    """Result of source emphasis detection."""
    def __init__(self, has_emphasis: bool, verified: bool, reason: str = ""):
        self.has_emphasis = has_emphasis
        self.verified = verified  # False if detection failed/ambiguous
        self.reason = reason


def _detect_emphasis_in_epub_source(
    source_path: str,
    manifest: dict[str, Any],
    chapter: Chapter,
    segment: Segment,
) -> SourceEmphasisResult:
    """Detect if segment's source HTML in EPUB contains semantic emphasis markup.
    
    Returns SourceEmphasisResult with:
    - has_emphasis: True if emphasis detected, False if verified no emphasis
    - verified: False if detection failed/ambiguous (fail-closed: must reject export)
    - reason: Explanation when verification failed
    """
    # Non-EPUB formats don't have HTML inline markup
    if not isinstance(source_path, str):
        return SourceEmphasisResult(False, True, "non-string path")
    
    source_lower = source_path.lower()
    if not source_lower.endswith(".epub"):
        # TXT, DOCX etc. - no HTML inline emphasis to preserve
        return SourceEmphasisResult(False, True, "non-EPUB format")
    
    # EPUB must exist for verification
    if not os.path.exists(source_path):
        return SourceEmphasisResult(False, False, f"source EPUB not found: {source_path}")
    
    resource_href = segment.resource_href
    if not resource_href:
        # Old state might lack resource_href
        return SourceEmphasisResult(False, False, "segment missing resource_href")
    
    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            if resource_href not in zf.namelist():
                return SourceEmphasisResult(
                    False, False, f"resource {resource_href} not in EPUB"
                )
            
            html_data = zf.read(resource_href)
            # Detect encoding
            detected = from_bytes(html_data).best()
            if detected is None:
                html = html_data.decode("utf-8", errors="replace")
            else:
                html = str(detected)
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Find blocks containing segment's source text
            # Use normalized text matching but require unambiguous match
            from ..markup.segments import _BLOCK_CANDIDATE_TAGS
            
            blocks = soup.find_all(_BLOCK_CANDIDATE_TAGS)
            segment_normalized = " ".join(segment.source.split())
            
            if not segment_normalized:
                # Empty segment, no emphasis to preserve
                return SourceEmphasisResult(False, True, "empty segment")
            
            matching_blocks = []
            for block in blocks:
                block_text = block.get_text(strip=False)
                block_normalized = " ".join(block_text.split())
                
                # Exact match or containment
                if segment_normalized == block_normalized or segment_normalized in block_normalized:
                    matching_blocks.append(block)
            
            if len(matching_blocks) == 0:
                # Cannot find segment in source HTML - ambiguous
                return SourceEmphasisResult(
                    False, False, 
                    f"segment text not found in resource {resource_href}"
                )
            
            if len(matching_blocks) > 1:
                # Multiple matching blocks - ambiguous (e.g., repeated text)
                # Check if they all have same emphasis status
                emphasis_statuses = []
                for block in matching_blocks:
                    has_emph = len(block.find_all(["em", "strong", "i", "b"])) > 0
                    if not has_emph:
                        # Check CSS spans
                        spans = block.find_all("span")
                        for span in spans:
                            style = span.get("style", "")
                            if any(kw in style.lower() for kw in ["italic", "oblique", "font-weight:bold", "font-weight: bold"]):
                                has_emph = True
                                break
                    emphasis_statuses.append(has_emph)
                
                if len(set(emphasis_statuses)) > 1:
                    # Ambiguous: same text with different emphasis
                    return SourceEmphasisResult(
                        False, False,
                        f"ambiguous match: {len(matching_blocks)} blocks with same text but different emphasis"
                    )
                # All matches have same emphasis status - use it
                has_emphasis = emphasis_statuses[0]
                return SourceEmphasisResult(has_emphasis, True, "multiple blocks with consistent emphasis")
            
            # Single unambiguous match
            block = matching_blocks[0]
            
            # Check for semantic emphasis tags
            emphasis_tags = block.find_all(["em", "strong", "i", "b"])
            if emphasis_tags:
                return SourceEmphasisResult(True, True, "has em/strong/i/b tags")
            
            # Check for spans with italic/bold CSS
            spans = block.find_all("span")
            for span in spans:
                style = span.get("style", "")
                # Check CSS properties
                if any(keyword in style.lower() for keyword in ["italic", "oblique", "font-weight:bold", "font-weight: bold"]):
                    return SourceEmphasisResult(True, True, "has CSS emphasis span")
            
            # Verified no emphasis
            return SourceEmphasisResult(False, True, "no emphasis tags found")
            
    except zipfile.BadZipFile:
        return SourceEmphasisResult(False, False, "corrupted EPUB file")
    except Exception as e:
        return SourceEmphasisResult(False, False, f"parsing error: {type(e).__name__}")


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
        manifest: Manifest dict for inline_detection_version and EPUB metadata

    Raises:
        InlinePreservationError: If inline preservation is unreliable
    """
    issues: list[str] = []
    
    # Check if this is new reliable state (version 1+)
    # New state: segments without epub_inline are verified plain text (no emphasis)
    # Old state: segments without epub_inline need re-scan to verify emphasis
    inline_version = manifest.get("inline_detection_version", 0) if manifest else 0
    is_new_state = inline_version >= 1

    for chapter in chapters:
        for segment in chapter.segments:
            # Check if segment has source inline markup metadata
            raw_inline = segment.meta.get(INLINE_META_KEY)
            
            # Segment without inline metadata
            if not isinstance(raw_inline, dict):
                # New state (version 1+): missing inline_meta means verified plain text
                if is_new_state:
                    continue
                
                # Old state (version 0): re-scan source EPUB to detect emphasis
                if source_path and manifest:
                    result = _detect_emphasis_in_epub_source(
                        source_path, manifest, chapter, segment
                    )
                    
                    # Fail-closed: if verification failed, reject export
                    if not result.verified:
                        issues.append(
                            f"Chapter {chapter.index}, Segment {segment.index}: "
                            f"Cannot verify whether source has emphasis markup. "
                            f"Reason: {result.reason}. "
                            f"This old-state segment lacks inline metadata. "
                            f"Retranslate with current wenyi version to enable reliable export."
                        )
                        continue
                    
                    # Verified result
                    if result.has_emphasis:
                        # Old state with emphasis detected: check target_inline
                        raw_target_inline = segment.meta.get("target_inline")
                        if not isinstance(raw_target_inline, dict) or not raw_target_inline.get("ranges"):
                            issues.append(
                                f"Chapter {chapter.index}, Segment {segment.index}: "
                                f"Source HTML contains emphasis markup ({result.reason}), "
                                f"but this segment was translated with an old version that did not preserve "
                                f"inline positions. All emphasis will be lost in export. "
                                f"Retranslate this segment with current wenyi version."
                            )
                    # else: verified no emphasis, safe to export (backward compatible)
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
