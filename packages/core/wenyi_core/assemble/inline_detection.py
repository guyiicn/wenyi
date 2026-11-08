"""Detect segments with inline markup that cannot be preserved in export."""

from __future__ import annotations

from ..markup.contracts import INLINE_META_KEY
from ..models.document import Chapter


def detect_unpreservable_inline(
    chapters: list[Chapter],
) -> dict[int, list[dict[str, object]]]:
    """Detect segments with source inline markup but no target preservation info.
    
    Returns mapping of chapter_index -> list of segment issues.
    Each issue contains: segment_index, source_text (truncated), inline_node_count, reason.
    """
    issues: dict[int, list[dict[str, object]]] = {}
    
    for chapter in chapters:
        chapter_issues: list[dict[str, object]] = []
        
        for segment in chapter.segments:
            # Check if segment has inline metadata
            raw_inline = segment.meta.get(INLINE_META_KEY)
            if not isinstance(raw_inline, dict):
                continue
            
            raw_nodes = raw_inline.get("nodes")
            if not isinstance(raw_nodes, list) or not raw_nodes:
                continue
            
            # Has inline markup in source
            node_count = len(raw_nodes)
            
            # Check if target exists
            if segment.target is None or not segment.target.strip():
                chapter_issues.append({
                    "segment_index": segment.index,
                    "source_preview": segment.source[:80] + "..." if len(segment.source) > 80 else segment.source,
                    "inline_node_count": node_count,
                    "inline_tags": [node.get("tag") for node in raw_nodes if isinstance(node, dict)],
                    "reason": "untranslated",
                })
                continue
            
            # Target exists but inline nodes use proportional mapping
            # This is unreliable for semantic emphasis preservation
            chapter_issues.append({
                "segment_index": segment.index,
                "source_preview": segment.source[:80] + "..." if len(segment.source) > 80 else segment.source,
                "inline_node_count": node_count,
                "inline_tags": [node.get("tag") for node in raw_nodes if isinstance(node, dict)],
                "reason": "proportional_mapping_unreliable",
                "note": "Current implementation uses character-offset proportions which may misplace emphasis",
            })
        
        if chapter_issues:
            issues[chapter.index] = chapter_issues
    
    return issues


def format_inline_report(issues: dict[int, list[dict[str, object]]]) -> str:
    """Format inline preservation issues as human-readable report."""
    if not issues:
        return "No inline markup preservation issues detected."
    
    lines: list[str] = []
    lines.append("INLINE MARKUP PRESERVATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    total_segments = sum(len(chapter_issues) for chapter_issues in issues.values())
    lines.append(f"Found {total_segments} segments with preservation concerns")
    lines.append("")
    
    for chapter_index in sorted(issues.keys()):
        chapter_issues = issues[chapter_index]
        lines.append(f"Chapter {chapter_index}: {len(chapter_issues)} segments")
        lines.append("-" * 60)
        
        for issue in chapter_issues:
            seg_idx = issue["segment_index"]
            preview = issue["source_preview"]
            node_count = issue["inline_node_count"]
            tags = issue.get("inline_tags", [])
            reason = issue["reason"]
            
            lines.append(f"  Segment {seg_idx}: {node_count} inline nodes")
            lines.append(f"    Tags: {', '.join(f'<{tag}>' for tag in tags if tag)}")
            lines.append(f"    Source: {preview}")
            lines.append(f"    Issue: {reason}")
            
            if "note" in issue:
                lines.append(f"    Note: {issue['note']}")
            
            lines.append("")
    
    lines.append("=" * 60)
    lines.append("")
    lines.append("RECOMMENDATIONS:")
    lines.append("- For 'untranslated': Complete translation before export")
    lines.append("- For 'proportional_mapping_unreliable': Current export uses")
    lines.append("  character-offset proportions which may misplace <i>, <em>,")
    lines.append("  <strong> tags in the translated text. For accurate semantic")
    lines.append("  emphasis preservation, retranslate with inline-aware workflow.")
    lines.append("")
    
    return "\n".join(lines)
