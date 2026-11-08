"""Protected inline markup markers for translation preservation.

Encode inline tags (<i>, <em>, <strong>) as protected markers before translation,
then decode them back after translation. This ensures semantic emphasis survives
the translation process without relying on proportional character offset estimation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Marker format: [[TAG_NAME_START]] and [[TAG_NAME_END]]
# Use double brackets to reduce collision with natural text
_MARKER_PATTERN = re.compile(r"\[\[([A-Z_]+)_(START|END)\]\]")

# Supported inline tags
_TAG_MAP = {
    "i": "I",
    "em": "EM",
    "strong": "STRONG",
    "b": "B",
    "span": "SPAN",
}

_REVERSE_TAG_MAP = {v: k for k, v in _TAG_MAP.items()}


@dataclass
class InlineNode:
    """Inline markup node with position info."""

    tag: str  # Original HTML tag name (i, em, strong, b)
    start: int  # Character offset in source text
    end: int  # Character offset in source text
    marker_name: str  # Internal marker name (I, EM, STRONG, B)


def _validate_nesting(nodes: list[InlineNode]) -> bool:
    """Check if inline nodes are properly nested (not overlapping)."""
    stack: list[InlineNode] = []

    # Sort by start position
    events: list[tuple[int, bool, InlineNode]] = []
    for node in nodes:
        events.append((node.start, True, node))  # Start event
        events.append((node.end, False, node))  # End event

    events.sort(key=lambda x: (x[0], not x[1]))  # Start before end at same position

    for pos, is_start, node in events:
        if is_start:
            stack.append(node)
        else:
            if not stack or stack[-1] != node:
                # Improperly nested
                return False
            stack.pop()

    return len(stack) == 0


def encode_inline_markers(
    text: str,
    nodes: list[dict[str, object]],
    source_length: int,
) -> tuple[str, list[InlineNode]]:
    """Encode inline markup as protected markers.

    Args:
        text: Source text without markup
        nodes: Inline node metadata from INLINE_META_KEY
        source_length: Expected source length for validation

    Returns:
        (encoded_text, inline_nodes) where encoded_text contains [[TAG_START]]/[[TAG_END]] markers

    Raises:
        ValueError: If nodes are improperly nested or invalid
    """
    if len(text) != source_length:
        raise ValueError(f"Source text length mismatch: expected {source_length}, got {len(text)}")

    # Parse inline nodes
    inline_nodes: list[InlineNode] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        tag = node.get("tag")
        offset = node.get("offset")
        placement = node.get("placement")
        length = node.get("length")

        if not isinstance(tag, str) or tag not in _TAG_MAP:
            # Skip unsupported tags (e.g., img, a, span without text)
            continue

        if not isinstance(offset, int) or offset < 0 or offset > source_length:
            raise ValueError(f"Invalid offset {offset} for tag <{tag}>")

        if placement not in ("before", "after", "inline"):
            continue

        marker_name = _TAG_MAP[tag]

        # Create paired start/end nodes
        # Use recorded length if available, otherwise use heuristic
        if isinstance(length, int) and length > 0:
            end_offset = min(offset + length, source_length)
        else:
            # Heuristic fallback: assume inline emphasis spans to next punctuation or end
            end_offset = offset
            while end_offset < source_length and text[end_offset].isalnum():
                end_offset += 1

        if end_offset > offset:
            inline_nodes.append(
                InlineNode(
                    tag=tag,
                    start=offset,
                    end=end_offset,
                    marker_name=marker_name,
                )
            )

    # Validate nesting
    if not _validate_nesting(inline_nodes):
        raise ValueError("Inline nodes are not properly nested")

    # Encode markers into text
    # Sort by position (reverse) to insert from end to start
    sorted_nodes = sorted(inline_nodes, key=lambda n: n.start, reverse=True)

    encoded = text
    for node in sorted_nodes:
        # Insert end marker first
        encoded = encoded[: node.end] + f"[[{node.marker_name}_END]]" + encoded[node.end :]
        # Insert start marker
        encoded = encoded[: node.start] + f"[[{node.marker_name}_START]]" + encoded[node.start :]

    return encoded, inline_nodes


def decode_inline_markers(
    translated_text: str,
    original_nodes: list[InlineNode],
) -> tuple[str, list[tuple[int, int, str]], list[str]]:
    """Decode protected markers from translated text.

    Args:
        translated_text: Translated text with [[TAG_START]]/[[TAG_END]] markers
        original_nodes: Original inline nodes for validation

    Returns:
        (clean_text, [(start, end, tag)], warnings)
        where clean_text has markers removed and positions are in the clean text

    Raises:
        ValueError: If markers are critically malformed
    """
    warnings: list[str] = []

    # Find all markers in translated text
    markers: list[tuple[int, str, str]] = []  # (position, marker_name, direction)
    for match in _MARKER_PATTERN.finditer(translated_text):
        marker_name = match.group(1)
        direction = match.group(2)  # START or END
        if marker_name not in _REVERSE_TAG_MAP:
            warnings.append(f"Unknown marker {marker_name} in translation")
            continue
        markers.append((match.start(), marker_name, direction))

    # Build clean text by removing markers
    clean_text = _MARKER_PATTERN.sub("", translated_text)

    # Match START/END pairs
    stack: list[tuple[int, str]] = []  # (position_in_clean_text, marker_name)
    matched_ranges: list[tuple[int, int, str]] = []

    offset = 0  # Track position adjustment due to marker removal
    for marker_pos, marker_name, direction in sorted(markers):
        clean_pos = marker_pos - offset
        marker_len = len(f"[[{marker_name}_{direction}]]")
        offset += marker_len

        tag = _REVERSE_TAG_MAP[marker_name]

        if direction == "START":
            stack.append((clean_pos, marker_name))
        elif direction == "END":
            if not stack:
                warnings.append(f"Unmatched [[{marker_name}_END]] at position {clean_pos}")
                continue

            start_pos, start_marker = stack.pop()
            if start_marker != marker_name:
                warnings.append(
                    f"Mismatched markers: [[{start_marker}_START]] closed by [[{marker_name}_END]]"
                )
                continue

            matched_ranges.append((start_pos, clean_pos, tag))

    # Check for unclosed markers
    for pos, marker_name in stack:
        warnings.append(f"Unclosed [[{marker_name}_START]] at position {pos}")

    # Validate we got expected number of ranges
    if len(matched_ranges) != len(original_nodes):
        warnings.append(
            f"Inline node count mismatch: expected {len(original_nodes)}, got {len(matched_ranges)}"
        )

    return clean_text, matched_ranges, warnings


def can_preserve_inline(segment_meta: dict[str, object]) -> tuple[bool, str]:
    """Check if segment has preservable inline markup.

    Returns:
        (can_preserve, reason) where reason explains why preservation may fail
    """
    raw_inline = segment_meta.get("epub_inline")
    if not isinstance(raw_inline, dict):
        return True, "no_inline_markup"

    nodes = raw_inline.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return True, "no_inline_nodes"

    # Check for supported tags
    supported_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        tag = node.get("tag")
        if isinstance(tag, str) and tag in _TAG_MAP:
            supported_count += 1

    if supported_count == 0:
        return True, "no_supported_tags"

    # Has preservable inline markup
    return True, f"has_{supported_count}_inline_nodes"
