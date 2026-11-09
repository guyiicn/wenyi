"""Test MD multiple emphasis with HTML fallback - must preserve all positions."""

from wenyi_core.assemble.markdown_inline import insert_markdown_emphasis


def test_two_ranges_one_html_one_markdown():
    """Two emphasis ranges: first unsafe (HTML), second safe (markdown).
    
    Regression: segment 9 had two emphasis ranges.
    First ends with punct+CJK follows (unsafe → HTML).
    Second at text end (safe → markdown).
    Must use original text positions for all insertions.
    """
    text = "不过真正的问题，她想，把手指梳到耳后，一边想，是年头太多了。"
    target_inline = {
        "version": 1,
        "ranges": [
            {"start": 0, "end": 8, "tag": "i", "css_style": ""},  # 不过真正的问题，(unsafe: punct + CJK "她" follows)
            {"start": 23, "end": 30, "tag": "i", "css_style": ""},  # 是年头太多了。(safe: at text end)
        ],
    }
    
    result = insert_markdown_emphasis(text, target_inline)
    
    # First range: HTML fallback (unsafe delimiter)
    assert "<em>不过真正的问题，</em>" in result
    
    # Second range: markdown emphasis at correct position
    assert "*是年头太多了。*" in result
    
    # Negative check: should NOT emphasize wrong text due to offset error
    assert "*别到耳后，一边*" not in result


def test_three_ranges_mixed():
    """Three ranges with mixed HTML and markdown."""
    text = "第一处强调，后续。第二处强调。第三处强调。"
    target_inline = {
        "version": 1,
        "ranges": [
            {"start": 0, "end": 6, "tag": "i", "css_style": ""},  # 第一处强调，(ends with punct + CJK "后" follows = unsafe)
            {"start": 9, "end": 14, "tag": "i", "css_style": ""},  # 第二处强调。(ends with punct + CJK "第" follows = unsafe)
            {"start": 15, "end": 20, "tag": "i", "css_style": ""},  # 第三处强调。(ends with punct at text end = safe)
        ],
    }
    
    result = insert_markdown_emphasis(text, target_inline)
    
    # All three should be emphasized
    assert "第一处强调" in result
    assert "第二处强调" in result
    assert "第三处强调" in result
    
    # Count emphasis markers/tags (should be 3 ranges = 6 delimiters total)
    em_count = result.count("<em>") + result.count("</em>")
    star_pairs = result.count("*") // 2
    total_ranges = em_count // 2 + star_pairs
    
    assert total_ranges == 3, f"Expected 3 emphasis ranges, got {total_ranges}"


def test_nested_with_html_fallback():
    """Nested ranges where outer needs HTML fallback."""
    text = "外层强调，内层强调"
    target_inline = {
        "version": 1,
        "ranges": [
            {"start": 0, "end": 11, "tag": "em", "css_style": ""},  # Outer: 外层强调，内层强调 (unsafe: ends with char after punct)
            {"start": 5, "end": 9, "tag": "strong", "css_style": ""},  # Inner: 内层强调
        ],
    }
    
    result = insert_markdown_emphasis(text, target_inline)
    
    # Both should be rendered
    assert "外层强调" in result
    assert "内层强调" in result
