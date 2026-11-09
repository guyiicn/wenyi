"""Inline markup preservation during translation.

Integrate protected markers into the translation workflow:
1. Encode inline tags before translation
2. Add preservation instructions to prompts
3. Decode and validate markers after translation
4. Store verified target positions
"""

from __future__ import annotations

from .contracts import INLINE_META_KEY
from .inline_markers import (
    InlineNode,
    decode_inline_markers,
    encode_inline_markers,
)


def prepare_sources_with_inline(
    sources: list[str],
    segment_metas: list[dict[str, object]],
) -> tuple[list[str], list[list[InlineNode] | None]]:
    """Encode inline markers in sources before translation.

    Args:
        sources: Source texts
        segment_metas: Segment metadata containing inline node info

    Returns:
        (encoded_sources, inline_nodes_per_segment)
        where inline_nodes_per_segment[i] is None if segment has no inline markup
    """
    encoded_sources: list[str] = []
    inline_nodes_per_segment: list[list[InlineNode] | None] = []

    for source, meta in zip(sources, segment_metas):
        raw_inline = meta.get(INLINE_META_KEY)
        if not isinstance(raw_inline, dict):
            encoded_sources.append(source)
            inline_nodes_per_segment.append(None)
            continue

        nodes = raw_inline.get("nodes")
        source_length = raw_inline.get("source_length")

        if not isinstance(nodes, list) or not nodes or not isinstance(source_length, int):
            encoded_sources.append(source)
            inline_nodes_per_segment.append(None)
            continue

        try:
            encoded, inline_nodes = encode_inline_markers(source, nodes, source_length)
            if inline_nodes:
                encoded_sources.append(encoded)
                inline_nodes_per_segment.append(inline_nodes)
            else:
                # No preservable inline nodes
                encoded_sources.append(source)
                inline_nodes_per_segment.append(None)
        except ValueError:
            # Encoding failed (invalid metadata), proceed without markers
            encoded_sources.append(source)
            inline_nodes_per_segment.append(None)

    return encoded_sources, inline_nodes_per_segment


def decode_targets_with_inline(
    targets: list[str],
    original_nodes_per_segment: list[list[InlineNode] | None],
) -> tuple[list[str], list[dict[str, object]], list[str]]:
    """Decode inline markers from translated texts and validate.

    Args:
        targets: Translated texts with [[TAG_START]]/[[TAG_END]] markers
        original_nodes_per_segment: Original inline nodes per segment (from prepare)

    Returns:
        (clean_targets, target_inline_metas, warnings)
        where target_inline_metas contains verified position data for each segment
        and warnings lists all validation issues
    """
    clean_targets: list[str] = []
    target_inline_metas: list[dict[str, object]] = []
    all_warnings: list[str] = []

    for idx, (target, original_nodes) in enumerate(zip(targets, original_nodes_per_segment)):
        if original_nodes is None:
            # No inline markup expected
            clean_targets.append(target)
            target_inline_metas.append({})
            continue

        try:
            clean_text, ranges, warnings = decode_inline_markers(target, original_nodes)

            if warnings:
                for warning in warnings:
                    all_warnings.append(f"Segment {idx}: {warning}")

            # Store verified target positions
            # Match ranges with original nodes to preserve css_style
            range_list = []
            for range_idx, (start, end, tag) in enumerate(ranges):
                range_dict = {"start": start, "end": end, "tag": tag}
                # Find matching original node to get css_style
                if range_idx < len(original_nodes):
                    original_node = original_nodes[range_idx]
                    if original_node.css_style:
                        range_dict["css_style"] = original_node.css_style
                range_list.append(range_dict)

            target_inline_meta = {
                "version": 1,
                "target_length": len(clean_text),
                "ranges": range_list,
                "source_node_count": len(original_nodes),
                "target_node_count": len(ranges),
                "validation_warnings": warnings if warnings else [],
            }

            # Critical: empty ranges with source nodes = preservation failure
            if len(original_nodes) > 0 and len(ranges) == 0:
                target_inline_meta["preservation_failed"] = True
                all_warnings.append(
                    f"Segment {idx}: All {len(original_nodes)} inline markers lost during translation/polish"
                )

            clean_targets.append(clean_text)
            target_inline_metas.append(target_inline_meta)

        except Exception as e:
            # Decode failed critically
            all_warnings.append(f"Segment {idx}: Critical decode error: {e}")
            clean_targets.append(target)
            target_inline_metas.append(
                {
                    "version": 1,
                    "decode_failed": True,
                    "preservation_failed": True,
                    "error": str(e),
                    "source_node_count": len(original_nodes) if original_nodes else 0,
                }
            )

    return clean_targets, target_inline_metas, all_warnings


def inline_preservation_prompt_suffix() -> str:
    """Return prompt instructions for preserving inline markers during translation."""
    return """

CRITICAL: The source text may contain protected inline markers like [[I_START]], [[EM_START]], [[STRONG_START]], [[B_START]] and their corresponding [[*_END]] markers. These markers preserve semantic emphasis (italic, bold, emphasis) and MUST be maintained in your translation.

Requirements:
1. Keep ALL markers in your translation: [[TAG_START]] and [[TAG_END]]
2. Place markers around the translated equivalent of the emphasized text
3. Maintain proper nesting: if source has [[EM_START]] [[STRONG_START]] text [[STRONG_END]] [[EM_END]], your translation must have the same structure
4. Do NOT translate the markers themselves
5. Do NOT add or remove markers

Example:
Source: "The [[I_START]]Force[[I_END]] is strong."
Correct: "[[I_START]]原力[[I_END]]很强大。"
Wrong: "原力很强大。" (markers lost)
Wrong: "The Force is strong." (not translated)"""


def validate_inline_preservation_result(
    warnings: list[str],
    *,
    allow_count_mismatch: bool = False,
) -> tuple[bool, str]:
    """Check if inline preservation was successful.

    Args:
        warnings: Validation warnings from decode
        allow_count_mismatch: If True, tolerate node count differences

    Returns:
        (is_valid, reason) where is_valid indicates if result is acceptable
    """
    if not warnings:
        return True, "no_warnings"

    critical_warnings = [
        w for w in warnings if "Unclosed" in w or "Unmatched" in w or "Mismatched" in w
    ]

    count_warnings = [w for w in warnings if "count mismatch" in w.lower()]

    if critical_warnings:
        return False, f"critical_markers_invalid: {'; '.join(critical_warnings)}"

    if count_warnings and not allow_count_mismatch:
        return False, f"node_count_mismatch: {'; '.join(count_warnings)}"

    # Only minor warnings
    return True, "minor_warnings_only"
