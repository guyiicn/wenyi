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
    
    # Check if emphasized text ends with Chinese punctuation
    if emphasized_text and emphasized_text[-1] in "，。！？；：""''（）【】《》、":
        # And followed by CJK character
        if end < len(text):
            after = text[end]
            if "\u4e00" <= after <= "\u9fff":
                return True
    
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
    
    # Separate ranges into markdown-safe and HTML-fallback
    markdown_ranges = []
    html_ranges = []
    
    for start, end, marker in valid_ranges:
        # Check if delimiters would be unsafe per CommonMark
        if _is_unsafe_markdown_delimiter(text, start, end, marker):
            # Use HTML tag
            if marker == "*":
                tag = "em"
            elif marker == "**":
                tag = "strong"
            else:
                tag = "span"  # Fallback
            html_ranges.append((start, end, tag))
        else:
            # Safe for markdown
            markdown_ranges.append((start, end, marker))
    
    # Process HTML ranges first (they don't interact with markdown syntax)
    result = text
    for start, end, tag in reversed(html_ranges):
        # Insert from back to preserve positions
        result = result[:end] + f"</{tag}>" + result[end:]
        result = result[:start] + f"<{tag}>" + result[start:]
    
    # Process markdown ranges if any
    if markdown_ranges:
        # Build events for markdown ranges
        events = []
        for start, end, marker in markdown_ranges:
            events.append((start, False, marker))
            events.append((end, True, marker))
        
        # Sort events: by position, then END before START at same position
        events.sort(key=lambda e: (e[0], e[1]))
        
        # Insert markers from back to front
        for pos, is_end, marker in reversed(events):
            if 0 <= pos <= len(result):
                result = result[:pos] + marker + result[pos:]
    
    return result
