"""Simplified CSS span E2E test focusing on CSS italic/bold only."""

import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from wenyi_core.config import Config
from wenyi_core.ingest.epub_reader import read_epub
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.orchestrator import Orchestrator


def _create_css_epub(epub_path: str) -> str:
    """Create EPUB with CSS spans. Returns CSS hash."""
    
    css_content = """span.italic { font-style: italic; }
span.bold { font-weight: bold; }"""
    
    chapter_html = """<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Test Chapter</title>
    <style>
        span.italic { font-style: italic; }
        span.bold { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Chapter One</h1>
    <p>First paragraph with <span class="italic">CSS italic text</span> here.</p>
    <p>Second paragraph with <span class="bold">CSS bold text</span> there.</p>
    <p>Third paragraph with plain text only.</p>
</body>
</html>"""
    
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test CSS Book</dc:title>
    <dc:language>en</dc:language>
    <dc:identifier id="bookid">test-css-123</dc:identifier>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="css1" href="styles.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
  </spine>
</package>""",
        )
        zf.writestr("OEBPS/chapter1.xhtml", chapter_html)
        zf.writestr("OEBPS/styles.css", css_content)
    
    return hashlib.sha256(css_content.encode()).hexdigest()


def simple_css_handler(messages, tier, json_mode):
    """Simple handler for CSS spans without complex nesting."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    
    trans = {
        "First paragraph with ": "首段有",
        "CSS italic text": "CSS斜体文本",
        " here.": "在这里。",
        "Second paragraph with ": "次段有",
        "CSS bold text": "CSS粗体文本",
        " there.": "在那里。",
        "Third paragraph with plain text only.": "第三段只有普通文本。",
    }
    
    if "translator" in system.lower() or "prose editor" in system.lower():
        if "Polish the translations" in user or "polish" in system.lower():
            # No polish for simplicity
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "assistant":
                    try:
                        prev = json.loads(msg["content"])
                        translations = prev.get("translations", [])
                        if translations:
                            return json.dumps({"polished": translations}, ensure_ascii=False)
                    except Exception:
                        pass
        
        # Translation
        paragraphs = re.findall(r"^\[(\d+)\]\s*(.+?)$", user, re.MULTILINE)
        
        if paragraphs:
            trans_list = []
            for idx, text in paragraphs:
                marker_pattern = r"(\[\[[A-Z_]+_(?:START|END)\]\])"
                parts = re.split(marker_pattern, text)
                
                result = []
                for part in parts:
                    if re.match(marker_pattern, part):
                        result.append(part)
                    elif part.strip():
                        translated = part
                        for eng, chi in trans.items():
                            translated = translated.replace(eng, chi)
                        result.append(translated)
                    else:
                        result.append(part)
                
                trans_list.append("".join(result))
            
            return json.dumps({"translations": trans_list}, ensure_ascii=False)
    
    return json.dumps({"translations": ["中文翻译"]}, ensure_ascii=False)


def test_css_span_simple_e2e():
    """Simple E2E test for CSS italic and bold spans."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        epub_path = str(tmp / "test_css.epub")
        css_hash = _create_css_epub(epub_path)
        
        # Verify ingest
        doc = read_epub(epub_path, source_lang="en", target_lang="zh")
        ch = doc.chapters[0]
        
        para_segs = [seg for seg in ch.segments if seg.kind == "text"]
        assert len(para_segs) == 3
        
        # First segment has CSS italic span
        seg1 = para_segs[0]
        assert "CSS italic text" in seg1.source
        inline1 = seg1.meta.get("epub_inline")
        assert inline1 is not None
        nodes1 = inline1.get("nodes", [])
        assert len(nodes1) == 1
        assert nodes1[0]["tag"] == "span"
        assert nodes1[0].get("length") == len("CSS italic text")
        
        # Second segment has CSS bold span
        seg2 = para_segs[1]
        assert "CSS bold text" in seg2.source
        inline2 = seg2.meta.get("epub_inline")
        nodes2 = inline2.get("nodes", [])
        assert len(nodes2) == 1
        assert nodes2[0]["tag"] == "span"
        
        # Run pipeline
        state_dir = str(tmp / "state")
        config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"polish": True, "review": False, "review_autofix": False},
                "paths": {"state_dir": state_dir},
            }
        )
        
        client = FakeClient(handler=simple_css_handler)
        orch = Orchestrator(config, client=client)
        store = orch.run(epub_path)
        
        # Reload and verify
        reloaded_ch = store.load_chapter(0)
        reloaded_paras = [seg for seg in reloaded_ch.segments if seg.kind == "text"]
        
        # First paragraph: CSS italic
        p1 = reloaded_paras[0]
        assert "CSS斜体文本" in p1.target
        assert "[[" not in p1.target
        
        target_inline_1 = p1.meta.get("target_inline")
        assert target_inline_1 is not None
        ranges_1 = target_inline_1.get("ranges", [])
        assert len(ranges_1) == 1
        assert ranges_1[0]["tag"] == "span"
        assert ranges_1[0].get("css_style") == "italic"
        phrase1 = p1.target[ranges_1[0]["start"]:ranges_1[0]["end"]]
        assert "CSS斜体文本" == phrase1 or "CSS斜体" in phrase1
        
        # Second paragraph: CSS bold
        p2 = reloaded_paras[1]
        assert "CSS粗体文本" in p2.target
        
        target_inline_2 = p2.meta.get("target_inline")
        assert target_inline_2 is not None
        ranges_2 = target_inline_2.get("ranges", [])
        assert len(ranges_2) == 1
        assert ranges_2[0]["tag"] == "span"
        
        # Export MD
        md_path = str(tmp / "output.md")
        orch.run_assemble(epub_path, out_path=md_path, out_format="markdown")
        
        assert Path(md_path).exists()
        md_content = Path(md_path).read_text()
        
        # CSS italic -> *text* or _text_
        assert ("*CSS斜体文本*" in md_content or "_CSS斜体文本_" in md_content)
        # CSS bold -> **text**
        assert "**CSS粗体文本**" in md_content
        # No English
        assert "CSS italic text" not in md_content
        assert "CSS bold text" not in md_content
        
        # Export EPUB
        epub_out = str(tmp / "output.epub")
        orch.run_assemble(epub_path, out_path=epub_out, out_format="epub")
        
        assert Path(epub_out).exists()
        
        with zipfile.ZipFile(epub_out, "r") as zf:
            chapter_file = None
            for name in zf.namelist():
                if name.endswith(".xhtml") and "chapter" in name.lower():
                    chapter_file = name
                    break
            
            assert chapter_file is not None
            chapter_html = zf.read(chapter_file).decode("utf-8")
            soup = BeautifulSoup(chapter_html, "html.parser")
            
            paragraphs = soup.find_all("p")
            assert len(paragraphs) >= 3
            
            # First paragraph: CSS italic span
            p1_html = paragraphs[0]
            assert "CSS斜体文本" in p1_html.get_text()
            
            # Second paragraph: CSS bold span
            p2_html = paragraphs[1]
            assert "CSS粗体文本" in p2_html.get_text()
            
            # No English
            all_text = soup.get_text()
            assert "CSS italic text" not in all_text
            assert "CSS bold text" not in all_text


if __name__ == "__main__":
    test_css_span_simple_e2e()
    print("✓ Simple CSS span E2E test passed")
