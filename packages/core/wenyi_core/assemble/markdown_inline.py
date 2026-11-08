"""Insert markdown emphasis markers based on verified inline positions."""

from __future__ import annotations


def insert_markdown_emphasis(
    text: str,
    target_inline: dict[str, object],
) -> str:
    """Insert markdown * and ** markers based on target_inline ranges.
    
    Uses event-based algorithm to handle nested ranges correctly.

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

    # Filter and prepare valid ranges
    valid_ranges = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        tag = r.get("tag")
        start = r.get("start")
        end = r.get("end")
        css_style = r.get("css_style")
        
        if not (isinstance(tag, str) and isinstance(start, int) and isinstance(end, int)):
            continue
        if start < 0 or end > len(text) or start >= end:
            continue
        
        # Select markdown syntax
        if tag in ("i", "em"):
            marker = "*"
        elif tag in ("strong", "b"):
            marker = "**"
        elif tag == "span" and css_style == "italic":
            marker = "*"
        elif tag == "span" and css_style == "bold":
            marker = "**"
        else:
            # Unsupported tag or plain span
            continue
        
        valid_ranges.append((start, end, marker))
    
    if not valid_ranges:
        return text
    
    # Build events: (position, is_end, marker)
    # is_end=False for START, is_end=True for END
    events = []
    for start, end, marker in valid_ranges:
        events.append((start, False, marker))
        events.append((end, True, marker))
    
    # Sort events: by position, then END before START at same position
    # This ensures proper nesting: when tags share boundary, outer closes last
    events.sort(key=lambda e: (e[0], e[1]))
    
    # Insert markers from back to front to preserve positions
    result = text
    for pos, is_end, marker in reversed(events):
        if 0 <= pos <= len(result):
            result = result[:pos] + marker + result[pos:]
    
    return result
