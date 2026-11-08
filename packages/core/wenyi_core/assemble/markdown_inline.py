"""Insert markdown emphasis markers based on verified inline positions."""

from __future__ import annotations


def insert_markdown_emphasis(
    text: str,
    target_inline: dict[str, object],
) -> str:
    """Insert markdown * and ** markers based on target_inline ranges.

    Args:
        text: Plain translated text
        target_inline: Verified inline metadata with ranges

    Returns:
        Text with markdown emphasis markers inserted
    """
    if not isinstance(target_inline, dict):
        return text

    ranges = target_inline.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return text

    # Sort ranges by start position (reverse for insertion)
    sorted_ranges = sorted(
        [
            r
            for r in ranges
            if isinstance(r, dict)
            and isinstance(r.get("tag"), str)
            and isinstance(r.get("start"), int)
            and isinstance(r.get("end"), int)
        ],
        key=lambda r: r["start"],
        reverse=True,
    )

    result = text
    for range_info in sorted_ranges:
        tag = range_info["tag"]
        start = range_info["start"]
        end = range_info["end"]
        css_style = range_info.get("css_style")

        # Validate bounds
        if start < 0 or end > len(result) or start >= end:
            continue

        # Select markdown syntax
        if tag in ("i", "em"):
            marker = "*"
        elif tag in ("strong", "b"):
            marker = "**"
        elif tag == "span" and css_style == "italic":
            # CSS italic span
            marker = "*"
        elif tag == "span" and css_style == "bold":
            # CSS bold span
            marker = "**"
        else:
            # Unsupported tag or plain span without CSS styling
            continue

        # Insert markers
        result = result[:end] + marker + result[end:]
        result = result[:start] + marker + result[start:]

    return result
