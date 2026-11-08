"""Strict CSS span EPUB restoration test - verifies spans are non-empty and contain translation."""

import json
import re
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator


def _marker_preserving_translation_handler(messages, tier, json_mode):
    """Handler that preserves [[MARKERS]] and translates text between them."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    
    # Language detection
    if "language detector" in system:
        return json.dumps({"language": "ja"}, ensure_ascii=False)
    
    # Pre-translation analyst
    if "pre-translation analyst" in system:
        return json.dumps({"genre": "测试", "tone": "正式", "style_guide": "标准"}, ensure_ascii=False)
    
    # Chapter title
    if "chapter title translator" in system:
        return json.dumps({"titles": ["第一章"]}, ensure_ascii=False)
    
    # Translation
    if "literary translator" in system or "academic translator" in system:
        # Find numbered paragraphs
        paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
        if paragraphs:
            translations = []
            for idx, text in paragraphs:
                # Split by markers
                marker_pattern = r"(\[\[[A-Z_0-9]+_(?:START|END)\]\])"
                parts = re.split(marker_pattern, text)
                
                result = []
                for part in parts:
                    if re.match(marker_pattern, part):
                        # Keep marker as-is
                        result.append(part)
                    elif part.strip():
                        # Simple translation
                        result.append(f"译{idx}")
                    else:
                        result.append(part)
                
                translations.append("".join(result))
            
            return json.dumps({"translations": translations}, ensure_ascii=False)
    
    # Terminology
    if "terminology" in system and "extractor" in system:
        return json.dumps({"terms": []}, ensure_ascii=False)
    
    # Fallback
    return json.dumps({"translations": ["译文"]}, ensure_ascii=False)


def test_css_span_epub_non_empty_with_text():
    """Verify CSS emphasis spans in EPUB output are non-empty and contain translation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        state_dir = str(tmp / "state")

        # Create EPUB with CSS italic and bold spans
        with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">'
                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test</dc:title><dc:language>en</dc:language></metadata>'
                '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                '<item id="css1" href="style.css" media-type="text/css"/></manifest>'
                '<spine><itemref idref="c1"/></spine></package>',
            )
            zf.writestr(
                "OEBPS/c1.xhtml",
                '''<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Test</title>
    <link href="style.css" rel="stylesheet" type="text/css"/>
</head>
<body>
    <h1>Chapter Title</h1>
    <p>First paragraph with <span class="italic">CSS italic text</span> here.</p>
    <p>Second paragraph with <span class="bold">CSS bold text</span> and more.</p>
    <p>Third has <span class="char-first-i italic">dropcap styled text</span> content.</p>
</body>
</html>''',
            )
            zf.writestr(
                "OEBPS/style.css",
                """.italic { font-style: italic; }
.bold { font-weight: bold; }
span.char-first-i { font-style: italic; }""",
            )

        # Configure and run pipeline with ja->zh (routing_handler default)
        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},  # routing_handler expects ja
                "llm": {"preset": "fake"},
                "pipeline": {
                    "polish": False,
                    "review": False,
                    "review_autofix": False,
                },
                "paths": {"state_dir": state_dir},
            }
        )

        client = FakeClient(handler=_marker_preserving_translation_handler)
        orch = Orchestrator(config, client=client)

        # Run translation
        store = orch.run(epub_path)

        # Export EPUB
        epub_out = str(tmp / "output.epub")
        orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        # Verify EPUB output
        with zipfile.ZipFile(epub_out, "r") as zf:
            chapter_file = next((n for n in zf.namelist() if n.endswith(".xhtml") and "c" in n.lower()), None)
            assert chapter_file, "No chapter XHTML found"
            
            chapter_html = zf.read(chapter_file).decode("utf-8")
            soup = BeautifulSoup(chapter_html, "lxml-xml")

        # Find all spans with italic or bold class
        css_spans = soup.find_all("span", class_=lambda c: c and ("italic" in c or "bold" in c))
        assert len(css_spans) >= 3, f"Expected at least 3 CSS spans, found {len(css_spans)}"

        # STRICT VERIFICATION: Each span must be non-empty
        for span in css_spans:
            span_text = span.get_text(strip=True)
            
            # Must not be empty - THIS IS THE CRITICAL FIX
            assert span_text, f"CSS span is EMPTY (text moved outside): {str(span)}"
            
            # Text must be INSIDE the span, not after it
            direct_text = "".join(span.strings)
            assert direct_text.strip(), \
                f"CSS span has no direct text content: {str(span)}"

        # Additional verification: Check no English source text leaked
        body = soup.find("body")
        body_text = body.get_text()
        assert "CSS italic text" not in body_text, "Source English text leaked to output"
        assert "CSS bold text" not in body_text, "Source English text leaked to output"
        assert "dropcap styled text" not in body_text, "Source English text leaked to output"


def test_css_span_specific_classes_preserved():
    """Verify CSS spans preserve their original classes (char-first-i, italic, etc)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test.epub")
        state_dir = str(tmp / "state")

        # Create EPUB with specific span classes like real books
        with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            zf.writestr(
                "OEBPS/content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">'
                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test</dc:title><dc:language>en</dc:language></metadata>'
                '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
                '<spine><itemref idref="c1"/></spine></package>',
            )
            zf.writestr(
                "OEBPS/c1.xhtml",
                '''<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title>
<style>
    .char-first-i { font-style: italic; }
    .char-cda-i { font-style: italic; }
</style>
</head>
<body>
    <p>Opening <span class="char-first-i italic">paragraph text</span> continues.</p>
    <p>Another <span class="char-cda-i">styled span</span> here.</p>
</body>
</html>''',
            )

        config = Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": False, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )

        client = FakeClient(handler=_marker_preserving_translation_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)

        epub_out = str(tmp / "output.epub")
        orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")

        with zipfile.ZipFile(epub_out, "r") as zf:
            chapter_file = next(n for n in zf.namelist() if n.endswith(".xhtml") and "c" in n.lower())
            chapter_html = zf.read(chapter_file).decode("utf-8")
            soup = BeautifulSoup(chapter_html, "html.parser")

        # Check char-first-i span
        char_first_span = soup.find("span", class_=lambda c: c and "char-first-i" in c)
        assert char_first_span, "char-first-i span not found"
        assert char_first_span.get_text(strip=True), "char-first-i span is empty"
        assert ("译" in char_first_span.get_text() or "润" in char_first_span.get_text()), \
            f"char-first-i span has no translation: {char_first_span}"

        # Check char-cda-i span
        char_cda_span = soup.find("span", class_=lambda c: c and "char-cda-i" in c)
        assert char_cda_span, "char-cda-i span not found"
        assert char_cda_span.get_text(strip=True), "char-cda-i span is empty"
        assert ("译" in char_cda_span.get_text() or "润" in char_cda_span.get_text()), \
            f"char-cda-i span has no translation: {char_cda_span}"
