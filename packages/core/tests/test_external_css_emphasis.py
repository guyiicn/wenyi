"""Regression test: external CSS files must be loaded for inline emphasis detection.

Real EPUB span.char-first-i with class='italic' was not detected because
extract_css_rules_from_epub() only read inline <style> tags, not <link rel="stylesheet">.
"""

import zipfile
from pathlib import Path

from wenyi_core.ingest.epub_reader import read_epub


def test_external_css_italic_detection(tmp_path: Path):
    """Real EPUB with <link rel="stylesheet" href="../styles/main.css"> must detect .italic spans."""

    # Create minimal EPUB with external CSS (mimics real Light of Jedi structure)
    epub_path = tmp_path / "test.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        # mimetype
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        # container.xml
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        # External CSS file with .italic rule
        zf.writestr(
            "styles/main.css",
            """
.italic {
    font-style: italic;
}
.char-first-i {
    display: inline;
}
""",
        )

        # XHTML with external CSS link and span.char-first-i italic
        zf.writestr(
            "text/chapter1.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter One</title>
    <link rel="stylesheet" type="text/css" href="../styles/main.css"/>
</head>
<body>
    <p>A<span class="char-first-i italic">ll is well.</span></p>
    <p>Second paragraph with <em>normal emphasis</em>.</p>
</body>
</html>""",
        )

        # content.opf
        zf.writestr(
            "content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter1" href="text/chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="css" href="styles/main.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
        )

    # Read EPUB - should detect span.italic via external CSS
    doc = read_epub(epub_path, "test.epub", target_lang="en")

    # Verify segments have inline metadata
    assert len(doc.chapters) == 1
    chapter = doc.chapters[0]

    # Segment 0: "A<span>ll is well.</span>" should have inline
    assert len(chapter.segments) >= 2
    seg0 = chapter.segments[0]
    assert seg0.source == "All is well."
    assert "epub_inline" in seg0.meta
    inline_data = seg0.meta["epub_inline"]

    # epub_inline is dict with "ranges" and "plain_text"
    if isinstance(inline_data, dict) and "ranges" in inline_data:
        ranges = inline_data["ranges"]
        assert len(ranges) >= 1, "Should have at least 1 inline range"

        # Check that css_style contains italic for external CSS detection
        span_range = next((r for r in ranges if r.get("tag") == "span"), None)
        assert span_range is not None, "span range should be detected"
        assert "italic" in span_range.get("css_style", "").lower(), (
            f"External CSS .italic rule not loaded: css_style={span_range.get('css_style')}"
        )
