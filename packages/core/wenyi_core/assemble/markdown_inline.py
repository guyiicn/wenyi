"""Insert markdown emphasis markers based on verified inline positions."""

from __future__ import annotations

import re


def _is_unsafe_markdown_delimiter(
    text: str, start: int, end: int, marker: str
) -> bool:
    """Check if markdown delimiters for range would be invalid per CommonMark.
    
    CommonMark emphasis requires flanking rules. Patterns that often fail:
    1. Emphasized text ends with Chinese punctuation, followed by CJK:
       *text，*她 -> renders as literal * (punct inside, CJK after)
    2. Emphasized text starts/ends surrounded by CJK without spaces:
       汉字*text*汉字 -> may fail
    
    Args:
        text: Full text
        start: Start position of emphasis range
        end: End position of emphasis range
        marker: '*' or '**'
    
    Returns True if range is unsafe and should use HTML fallback.
    """
    if start < 0 or end > len(text) or start >= end:
        return True
    
    emphasized_text = text[start:end]
    
    # Check if emphasized text ends with Chinese punctuation followed by CJK
    if emphasized_text and emphasized_text[-1] in "，。！？；：""''（）【】《》、":
        # Only unsafe if followed by CJK character (not end of text or space)
        if end < len(text):
            after = text[end]
            # Unsafe only if CJK follows directly
            if "\u4e00" <= after <= "\u9fff":
                return True
        # If at end of text or followed by space/punct, it's safe - continue checking other conditions
    
    # Check if emphasized text starts with Chinese punctuation
    if emphasized_text and emphasized_text[0] in "，。！？；：""''（）【】《》、":
        # And preceded by CJK character
        if start > 0:
            before = text[start - 1]
            if "\u4e00" <= before <= "\u9fff":
                return True
    
    return False


def insert_markdown_emphasis(
    text: str,
    target_inline: dict[str, object],
) -> str:
    """Insert markdown * and ** markers based on target_inline ranges.
    
    Uses event-based algorithm to handle nested ranges correctly.
    Falls back to HTML <em>/<strong> for unsafe CommonMark delimiters.

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
    
    # Build insertion events for ALL ranges (HTML and markdown)
    # Each event: (position, is_end, insertion_text, original_range_index)
    # Use original text positions throughout - adjust offset as we insert
    
    insertion_events = []
    for idx, (start, end, marker) in enumerate(valid_ranges):
        # Check if this range needs HTML fallback
        use_html = _is_unsafe_markdown_delimiter(text, start, end, marker)
        
        if use_html:
            # HTML tags
            if marker == "*":
                open_tag, close_tag = "<em>", "</em>"
            elif marker == "**":
                open_tag, close_tag = "<strong>", "</strong>"
            else:
                open_tag, close_tag = "<span>", "</span>"
            
            # Add events: (position, is_end, text_to_insert, range_idx)
            insertion_events.append((start, False, open_tag, idx))
            insertion_events.append((end, True, close_tag, idx))
        else:
            # Markdown delimiters
            insertion_events.append((start, False, marker, idx))
            insertion_events.append((end, True, marker, idx))
    
    # Sort events: by position (ascending), then END before START at same position
    # This ensures proper nesting when ranges share boundaries
    insertion_events.sort(key=lambda e: (e[0], e[1]))
    
    # Insert from back to front to preserve positions
    # As we insert, track cumulative offset adjustment
    result = text
    for pos, is_end, insert_text, _ in reversed(insertion_events):
        if 0 <= pos <= len(result):
            result = result[:pos] + insert_text + result[pos:]
    
    return result
