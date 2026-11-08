"""Test CSS-styled span ingest."""

from wenyi_core.markup.segments import annotate_epub_resource


def test_css_italic_span_preserved():
    """CSS italic span should be preserved as inline node."""
    html = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Test</title>
    <style>span.italic { font-style: italic; }</style>
</head>
<body>
    <p>Text with <span class="italic">CSS italic text</span> here.</p>
</body>
</html>"""
    
    title, segments, template, image_blocks = annotate_epub_resource(
        html, 0, "test.xhtml"
    )
    
    assert len(segments) == 1
    seg = segments[0]
    assert "CSS italic text" in seg.source
    
    inline = seg.meta.get("epub_inline")
    assert inline is not None, "epub_inline metadata must exist"
    
    nodes = inline.get("nodes", [])
    assert len(nodes) == 1, f"Expected 1 inline node (CSS span), got {len(nodes)}"
    assert nodes[0]["tag"] == "span"


def test_css_bold_span_preserved():
    """CSS bold span should be preserved as inline node."""
    html = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Test</title>
    <style>span.bold { font-weight: bold; }</style>
</head>
<body>
    <p>Text with <span class="bold">CSS bold text</span> here.</p>
</body>
</html>"""
    
    title, segments, template, image_blocks = annotate_epub_resource(
        html, 0, "test.xhtml"
    )
    
    seg = segments[0]
    inline = seg.meta.get("epub_inline")
    nodes = inline.get("nodes", [])
    assert len(nodes) == 1, f"Expected 1 inline node (CSS span), got {len(nodes)}"


def test_plain_span_not_emphasis():
    """Plain span without CSS styling should not be treated as emphasis."""
    html = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
    <p>Text with <span class="note">plain span</span> here.</p>
</body>
</html>"""
    
    title, segments, template, image_blocks = annotate_epub_resource(
        html, 0, "test.xhtml"
    )
    
    seg = segments[0]
    inline = seg.meta.get("epub_inline")
    nodes = inline.get("nodes", [])
    # Plain span with class should be preserved as structural span
    assert len(nodes) == 1, "Plain span with class should be preserved"


def test_span_with_id_preserved():
    """Span with id should be preserved as structural element."""
    html = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
    <p>Text with <span id="ref-1">span with id</span> here.</p>
</body>
</html>"""
    
    title, segments, template, image_blocks = annotate_epub_resource(
        html, 0, "test.xhtml"
    )
    
    seg = segments[0]
    inline = seg.meta.get("epub_inline")
    nodes = inline.get("nodes", [])
    assert len(nodes) == 1, "Span with id should be preserved"


if __name__ == "__main__":
    test_css_italic_span_preserved()
    test_css_bold_span_preserved()
    test_plain_span_not_emphasis()
    test_span_with_id_preserved()
    print("✓ All CSS span ingest tests passed")
